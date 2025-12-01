from typing import NamedTuple, Tuple, Any

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle
import flax.linen as nn
from functools import partial

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from relax.algorithm.base import Algorithm
from relax.network.mf2_sac_ent2 import MF2SACENT2Net, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class Diffv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float


class MF2SACENT2(Algorithm):

    def __init__(
        self,
        agent: MF2SACENT2Net,
        params: Diffv2Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
        use_ema: bool = True,
        sample_k: int = 500,
        alpha_value: float = 0.01,
        fixed_alpha: bool = True,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0
        self.fixed_alpha = fixed_alpha
        self.alpha_value = alpha_value

        self.state = Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                # policy=self.optim.init(params.policy),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema
        self.K=sample_k

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, acts_key, flow_noise_key, r_key, mask_key, t_key = jax.random.split(
                key, 6)

            reward *= self.reward_scale

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            # next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha, q1_params, q2_params),
            #                                     next_obs)
            # Get next action and its entropy from the policy
            # next_action, next_entropy = self.agent.get_action_ent(next_eval_key,
            #                                                       (policy_params, log_alpha, q1_params, q2_params),
            #                                                       next_obs)
            next_action, next_entropy=self.agent.get_action_entropy_singlestep(next_eval_key,
                                                                  (policy_params, log_alpha, q1_params, q2_params),
                                                                  next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            #TODO: positive or negative
            if self.fixed_alpha:
                q_target = jnp.minimum(q1_target, q2_target)  - jnp.float32(self.alpha_value) * next_entropy
            else:
                q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_entropy
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            flow_noise_key,noise_rng=jax.random.split(flow_noise_key,2)

            r0 = jax.random.uniform(r_key, shape=(action.shape[0],), minval=1e-3, maxval=0.9946)
            mask = jax.random.bernoulli(mask_key, p=0.0, shape=(action.shape[0],))
            t0 = jax.random.uniform(t_key, shape=(action.shape[0],), minval=1e-3, maxval=0.9946)
            is_t_gt_r = t0 > r0
            t_swap = jnp.where(is_t_gt_r, t0, r0)
            r_swap = jnp.where(is_t_gt_r, r0, t0)
            r = jnp.where(mask, r0, r_swap)
            t = jnp.where(mask, r0, t_swap)

            t = jnp.expand_dims(t, axis=1)
            r = jnp.expand_dims(r, axis=1)

            noise_sample = jax.random.normal(flow_noise_key, action.shape)

            def q_sample(t: jax.Array, x_start: jax.Array, noise: jax.Array):
                return t * x_start + (1 - t) * noise

            noisy_actions = q_sample(t, action, noise_sample)
            noisy_actions_repeat = jnp.repeat(jnp.expand_dims(noisy_actions, axis=1), axis=1, repeats=self.K)
            std = jnp.expand_dims((1-t) / t, axis=-1)
            lower_bound = 1 / (1-t)[:, :, None] * noisy_actions_repeat - (1 / std)
            upper_bound = 1 / (1-t)[:, :, None] * noisy_actions_repeat + (1 / std)
            tnormal_noise = jax.random.truncated_normal(
                key, lower=lower_bound, upper=upper_bound, shape=(action.shape[0], self.K, action.shape[1]))
            flow_noise_key,noise_rng=jax.random.split(flow_noise_key,2)
            normal_noise = jax.random.normal(flow_noise_key, shape=((action.shape[0], self.K, action.shape[1])))
            normal_noise_clip = jnp.clip(normal_noise, min=lower_bound, max=upper_bound)
            noise = jnp.where(jnp.isnan(tnormal_noise), normal_noise_clip, tnormal_noise)
            # noise=tnormal_noise
            clean_samples = 1 / t[:, :, None] * noisy_actions_repeat - std * noise

            observations_repeat = jnp.repeat(jnp.expand_dims(obs, axis=1), axis=1, repeats=self.K)

            devices = jax.devices()
            compute_Q_DDP = partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))(get_min_q)
            critic = compute_Q_DDP( observations_repeat, clean_samples)  # batch_size, K   sample B-K-A

            if self.fixed_alpha:
                critic=critic/jnp.float32(agent.alpha_value)
            else:
                safe_alpha = jnp.maximum(jnp.exp(log_alpha), 0.001)
                critic=critic/safe_alpha

            q_mean, q_std = critic.mean(), critic.std()
            Z=jax.nn.logsumexp(critic, axis=1, keepdims=True)  # B-1
            q_weights = jnp.exp(critic - Z) # B-K

            """
            if self.fixed_alpha:
                weight = nn.softmax((1 / jnp.float32(self.alpha_value)) * critic, axis=1)
            else:
                safe_alpha = jnp.maximum(jnp.exp(log_alpha), 0.001)
                weight = nn.softmax((1 / jnp.exp(safe_alpha)) * critic, axis=1)

            u_estimation = jnp.sum(weight[:,:,None] * (clean_samples-noise), axis=1)
            """
            obs_expanded = jnp.repeat(obs, self.K, axis=0) # B*K -S



            def policy_loss_fn(policy_params) -> tuple[Any, tuple[Array, Any, Array]]:
                def denoiser(x, r, t):
                    return self.agent.policy(policy_params,obs_expanded, x, r, t) # B*K -A

                loss,dudt,u_out,dudt_out,dudt_max = self.agent.flow.reverse_weighted_p_loss(q_weights, denoiser, r, t,clean_samples,noise,
                                                               noisy_actions)
                u_pred=jnp.mean(u_out)
                dudt_pred=jnp.mean(dudt_out)
                # loss*

                acts = self.agent.get_vanilla_action(acts_key, (policy_params, log_alpha, q1_params, q2_params), obs)
                q1_target = self.agent.q(target_q1_params, obs, acts)
                q2_target = self.agent.q(target_q2_params, obs, acts)
                q_target = jnp.minimum(q1_target, q2_target)
                # loss += jnp.mean(-q_target)

                return loss, (jnp.sum(q_weights),
                              Z,
                              jnp.mean(jnp.sum(q_weights)),
                              jnp.std(jnp.sum(q_weights)),
                              dudt,u_pred,dudt_pred,dudt_max)

            (total_loss, (q_weights, Z, q_mean, q_std,dudt,u_pred,dudt_pred,dudt_max)), policy_grads = jax.value_and_grad(policy_loss_fn,
                                                                                                  has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = jnp.mean(next_entropy)
                log_alpha_loss = jnp.mean(log_alpha * (approx_entropy-self.agent.target_entropy))
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params),
                                                           opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads,
                                                                 policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = Diffv2TrainState(
                params=Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params,
                                    target_policy_params, log_alpha),
                opt_state=Diffv2OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state,
                                          log_alpha=log_alpha_opt_state),
                step=step + 1,
                entropy=jnp.float32(jnp.mean(next_entropy)),
                running_mean=new_running_mean,
                running_std=new_running_std
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                "q2_loss": q2_loss,
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "scale_q_mean": jnp.mean(Z),
                "scale_q_std": jnp.std(Z),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                # "entropy_approx": 0.5 * self.agent.act_dim * jnp.log(
                #     2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
                "entropy_approx": jnp.mean(next_entropy),
                "u_pred": u_pred,
                "dudt": dudt_pred,
                "dudt_max": dudt_max
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action,
                                        stateless_get_vanilla_action=self.agent.get_vanilla_action,
                                        stateless_get_vanilla_action_step=self.agent.get_vanilla_action_step)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
