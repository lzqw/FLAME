from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import pickle
from functools import partial

from relax.algorithm.base import Algorithm
from relax.network.rf2_sac_ent_v import RF2SACENTNet_V, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class Diffv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    encoder: optax.OptState


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float


@jax.jit
def augment_batch(obs: jnp.ndarray,
                  next_obs: jnp.ndarray,
                  obs_key: jax.Array,
                  next_obs_key: jax.Array,
                  padding: int = 4
                  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def random_crop(key, img, padding):
        crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
        crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
        padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                             mode='edge')
        return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    obs_keys = jax.random.split(obs_key, obs.shape[0])
    obs = jnp.reshape(obs, (obs.shape[0], -1, 84, 84))
    obs = obs.transpose((0, 2, 3, 1))
    obs = jax.vmap(random_crop, (0, 0, None))(obs_keys, obs, padding)
    obs = obs.transpose((0, 3, 1, 2))

    next_obs_keys = jax.random.split(next_obs_key, next_obs.shape[0])
    next_obs = jnp.reshape(next_obs, (next_obs.shape[0], -1, 84, 84))
    next_obs = next_obs.transpose((0, 2, 3, 1))
    next_obs = jax.vmap(random_crop, (0, 0, None))(next_obs_keys, next_obs, padding)
    next_obs = next_obs.transpose((0, 3, 1, 2))

    return jnp.squeeze(jnp.reshape(obs, (obs.shape[0], -1))), jnp.squeeze(
        jnp.reshape(next_obs, (next_obs.shape[0], -1)))


class RF2SACENT_V(Algorithm):
    def __init__(
        self,
        agent: RF2SACENTNet_V,
        params: Diffv2Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 1.0,
        use_ema: bool = True,
        temperature: float = 1.0,
        total_step: int = 100000,
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
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.encoder_optim = optax.adam(lr)
        self.entropy = 0.0

        self.state = Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                # policy=self.optim.init(params.policy),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
                encoder=self.encoder_optim.init(params.encoder),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema
        self.temperature = temperature
        self.fixed_alpha = fixed_alpha
        self.alpha_value = alpha_value
        self.K=sample_k
        self.total_step = total_step

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Tuple
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, discount = data
            if len(action.shape) == 2 and action.shape[0] == 1:
                action = np.squeeze(action, axis=0)
            reward = np.squeeze(reward)
            discount = np.squeeze(discount)
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha, encoder_params = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, encoder_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, new_eval_key,obs_aug_key, next_obs_aug_key, log_alpha_key,flow_time_key, flow_noise_key = jax.random.split(key, 7)

            # data augmentation
            obs, next_obs = augment_batch(obs, next_obs, obs_aug_key, next_obs_aug_key)

            reward *= self.reward_scale
            # reward = jnp.log(1.0 + reward)
            next_obs = jax.lax.stop_gradient(self.agent.encoder(encoder_params, next_obs))

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            next_action = self.agent.get_action(next_eval_key,
                                                (policy_params, log_alpha, q1_params, q2_params, encoder_params),
                                                next_obs)
            # next_action = self.agent.get_action(next_eval_key, (policy_params, jnp.log(current_noise_scale), q1_params, q2_params, encoder_params), next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = reward + discount * q_target

            def q_loss_fn(q1_params: hk.Params, q2_params: hk.Params, encoder_params: hk.Params) -> jax.Array:
                obs_latent = self.agent.encoder(encoder_params, obs)
                q1 = self.agent.q(q1_params, obs_latent, action)
                q1_loss = jnp.mean((q1 - q_backup) ** 2)
                q2 = self.agent.q(q2_params, obs_latent, action)
                q2_loss = jnp.mean((q2 - q_backup) ** 2)
                q_loss = q1_loss + q2_loss
                return q_loss, (q1_loss, q2_loss, q1, q2)

            (q_loss, (q1_loss, q2_loss, q1, q2)), (q1_grads, q2_grads, encoder_grads) = jax.value_and_grad(q_loss_fn,
                                                                                                           argnums=(0,
                                                                                                                    1,
                                                                                                                    2),
                                                                                                           has_aux=True)(
                q1_params, q2_params, encoder_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            encoder_update, encoder_opt_state = self.optim.update(encoder_grads, encoder_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)
            encoder_params = optax.apply_updates(encoder_params, encoder_update)

            flow_time_key, time_rng = jax.random.split(flow_time_key, 2)
            flow_noise_key, noise_rng = jax.random.split(flow_noise_key, 2)
            t = jax.random.uniform(flow_time_key, shape=(next_obs.shape[0],), minval=1e-3, maxval=0.9946)
            t = jnp.expand_dims(t, axis=1)
            noise_sample = jax.random.normal(flow_noise_key, action.shape)

            def q_sample(t: int, x_start: jax.Array, noise: jax.Array):
                return t * x_start + (1 - t) * noise

            noisy_actions = q_sample(t, action, noise_sample)
            noisy_actions=noisy_actions.clip(-1,1)

            #TODO: is a1=(1/t)at+(1-t)/t*et or a1=(1/t)at-(1-t)/t*et
            noisy_actions_repeat = jnp.repeat(jnp.expand_dims(noisy_actions, axis=1), axis=1, repeats=self.K)
            std = jnp.expand_dims((1-t) / t, axis=-1)
            lower_bound = 1 / (1-t)[:, :, None] * noisy_actions_repeat - (1 / std)
            upper_bound = 1 / (1-t)[:, :, None] * noisy_actions_repeat + (1 / std)
            #action: batch_size, action_dim
            tnormal_noise = jax.random.truncated_normal(
                key, lower=lower_bound, upper=upper_bound, shape=(action.shape[0], self.K, action.shape[1]))
            flow_noise_key,noise_rng=jax.random.split(flow_noise_key,2)
            normal_noise = jax.random.normal(flow_noise_key, shape=((action.shape[0], self.K, action.shape[1])))
            normal_noise_clip = jnp.clip(normal_noise, min=lower_bound, max=upper_bound)
            noise = jnp.where(jnp.isnan(tnormal_noise), normal_noise_clip, tnormal_noise)
            clean_samples = 1 / t[:, :, None] * noisy_actions_repeat - std * noise

            obs_encoded = self.agent.encoder(encoder_params, obs)
            observations_repeat = jnp.repeat(jnp.expand_dims(obs_encoded, axis=1), axis=1, repeats=self.K)
            devices = jax.devices()
            compute_Q_DDP = partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))(get_min_q)
            critic = compute_Q_DDP( observations_repeat, clean_samples)  # batch_size, K

            critic = critic / jnp.float32(agent.alpha_value)

            q_mean, q_std = critic.mean(), critic.std()
            Z = jax.nn.logsumexp(critic, axis=1, keepdims=True)  # [batch_size, 1]
            q_weights = jnp.exp(critic - Z) # [batch_size, mc_num]

            clean_samples_reshape=clean_samples.reshape(-1, clean_samples.shape[-1])
            obs_reshape = observations_repeat.reshape(-1, observations_repeat.shape[-1])
            noise_reshape=noise.reshape(-1, noise.shape[-1])
            u=clean_samples_reshape-noise_reshape
            t_reshape = jnp.repeat(t.squeeze(), repeats=self.K)
            noisy_actions_reshape=noisy_actions_repeat.reshape(-1, noisy_actions_repeat.shape[-1])
            weight_reshape=q_weights.reshape(-1,1)


            def policy_loss_fn(policy_params) -> jax.Array:

                def denoiser(t, x):
                    return self.agent.policy(policy_params, obs_reshape, x, t)

                loss = self.agent.flow.reverse_weighted_p_loss2(denoiser, t_reshape, noisy_actions_reshape,
                                                                jax.lax.stop_gradient(weight_reshape),
                                                            jax.lax.stop_gradient(u))
                return loss, (weight_reshape, critic, q_mean, q_std)


            (total_loss, (weight, u_estimation, critic_mean, critic_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                log_alpha_loss = -1 * log_alpha * (
                        -1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)
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
                                    target_policy_params, log_alpha, encoder_params),
                opt_state=Diffv2OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state,
                                          log_alpha=log_alpha_opt_state, encoder=encoder_opt_state),
                step=step + 1,
                entropy=jnp.float32(0.0),
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
                "weights_std": jnp.std(weight),
                "weights_mean": jnp.mean(weight),
                "weights_min": jnp.min(weight),
                "weights_max": jnp.max(weight),
                "u_estimation_mean": jnp.mean(u_estimation),
                "u_estimation_std": jnp.std(u_estimation),
                "critic_mean": critic_mean,
                "critic_std": critic_std,
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action,
                                        stateless_get_action_full=self.agent.get_action_full,
                                        stateless_get_vanilla_action=self.agent.get_vanilla_action,
                                        stateless_get_vanilla_action_step=self.agent.get_vanilla_action_step)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2,
                self.state.params.encoder)

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2,
                self.state.params.encoder)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)

    def warmup(self, data: tuple) -> None:
        key = jax.random.key(0)
        obs, _, _, _, _ = data
        obs = obs[0]
        policy_params = self.get_policy_params()
        self._update(key, self.state, data)
        self._get_action(key, policy_params, obs)
        self._get_deterministic_action(policy_params, obs)
