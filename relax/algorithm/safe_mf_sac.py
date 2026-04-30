from typing import NamedTuple, Tuple
import pickle

import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_mf_sac import SafeMFSACNet, SafeMFSACParams
from relax.safety.double_integrator_filter import project_action_jax
from relax.utils.typing import Metric
from scripts.safe_experience import FilteredExperience


class SafeMFSACOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    qh: optax.OptState
    vh: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class SafeMFSACTrainState(NamedTuple):
    params: SafeMFSACParams
    opt_state: SafeMFSACOptStates
    step: int


class SafeMFSAC(Algorithm):
    def __init__(
        self,
        agent: SafeMFSACNet,
        params: SafeMFSACParams,
        *,
        gamma: float = 0.99,
        gamma_h: float = 0.99,
        beta_h: float = 1.0,
        tau_h: float = 0.90,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        sample_k: int = 128,
        use_filter_in_target: bool = True,
        use_feasibility_guidance: bool = True,
    ):
        self.agent = agent
        self.gamma = gamma
        self.gamma_h = gamma_h
        self.beta_h = beta_h
        self.tau_h = tau_h
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.K = sample_k
        self.use_filter_in_target = use_filter_in_target
        self.use_feasibility_guidance = use_feasibility_guidance

        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)

        self.state = SafeMFSACTrainState(
            params=params,
            opt_state=SafeMFSACOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                qh=self.optim.init(params.qh),
                vh=self.optim.init(params.vh),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: SafeMFSACTrainState, data: FilteredExperience
        ) -> Tuple[SafeMFSACTrainState, Metric]:
            obs = data.obs
            raw_action = data.raw_action
            exec_action = data.action
            reward = data.reward
            next_obs = data.next_obs
            done = data.done.astype(jnp.float32)
            safe_violation = data.safe_violation.astype(jnp.float32)

            (
                q1_params,
                q2_params,
                target_q1_params,
                target_q2_params,
                qh_params,
                vh_params,
                target_vh_params,
                policy_params,
                target_policy_params,
                log_alpha,
            ) = state.params
            q1_opt_state, q2_opt_state, qh_opt_state, vh_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step

            keys = jax.random.split(key, 8)
            next_eval_key, flow_noise_key, r_key, mask_key, t_key, clean_key, _, _ = keys

            reward_scaled = reward * self.reward_scale

            def get_min_q(s, a):
                return jnp.minimum(self.agent.q(q1_params, s, a), self.agent.q(q2_params, s, a))

            raw_next_action = self.agent.get_action(
                next_eval_key,
                (policy_params, log_alpha, q1_params, q2_params),
                next_obs,
            )
            if self.use_filter_in_target:
                exec_next_action, _, _, _ = project_action_jax(next_obs, raw_next_action)
            else:
                exec_next_action = raw_next_action

            q_target = jnp.minimum(
                self.agent.q(target_q1_params, next_obs, exec_next_action),
                self.agent.q(target_q2_params, next_obs, exec_next_action),
            )
            q_backup = reward_scaled + (1.0 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params):
                pred_q = self.agent.q(q_params, obs, exec_action)
                return jnp.mean((pred_q - q_backup) ** 2), pred_q

            (q1_loss, q1_pred), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2_pred), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)

            qh_target = safe_violation + (1.0 - safe_violation) * self.gamma_h * (1.0 - done) * self.agent.vh(
                target_vh_params, next_obs
            )

            def qh_loss_fn(qh_params_inner):
                qh_pred = self.agent.qh(qh_params_inner, obs, raw_action)
                loss = jnp.mean((qh_pred - jax.lax.stop_gradient(qh_target)) ** 2)
                return loss, qh_pred

            (qh_loss, qh_pred), qh_grads = jax.value_and_grad(qh_loss_fn, has_aux=True)(qh_params)

            def reversed_expectile_loss(diff):
                weight = jnp.abs(self.tau_h - (diff > 0).astype(jnp.float32))
                return weight * diff ** 2

            def vh_loss_fn(vh_params_inner):
                vh_pred = self.agent.vh(vh_params_inner, obs)
                qh_stop = jax.lax.stop_gradient(self.agent.qh(qh_params, obs, raw_action))
                return jnp.mean(reversed_expectile_loss(qh_stop - vh_pred)), vh_pred

            (vh_loss, vh_pred), vh_grads = jax.value_and_grad(vh_loss_fn, has_aux=True)(vh_params)

            # MeanFlow policy update on raw actions
            r0 = jax.random.uniform(r_key, shape=(raw_action.shape[0],), minval=1e-3, maxval=0.9946)
            mask = jax.random.bernoulli(mask_key, p=0.0, shape=(raw_action.shape[0],))
            t0 = jax.random.uniform(t_key, shape=(raw_action.shape[0],), minval=1e-3, maxval=0.9946)
            is_t_gt_r = t0 > r0
            t_swap = jnp.where(is_t_gt_r, t0, r0)
            r_swap = jnp.where(is_t_gt_r, r0, t0)
            r = jnp.where(mask, r0, r_swap)
            t = jnp.where(mask, r0, t_swap)
            t = jnp.expand_dims(t, axis=1)
            r = jnp.expand_dims(r, axis=1)

            noise_sample = jax.random.normal(flow_noise_key, raw_action.shape)

            def q_sample(t_inner: jax.Array, x_start: jax.Array, noise: jax.Array):
                return t_inner * x_start + (1 - t_inner) * noise

            noisy_actions = q_sample(t, raw_action, noise_sample)
            noisy_actions_repeat = jnp.repeat(jnp.expand_dims(noisy_actions, axis=1), axis=1, repeats=self.K)
            std = jnp.expand_dims((1 - t) / t, axis=-1)
            lower_bound = 1 / (1 - t)[:, :, None] * noisy_actions_repeat - (1 / std)
            upper_bound = 1 / (1 - t)[:, :, None] * noisy_actions_repeat + (1 / std)
            tnormal_noise = jax.random.truncated_normal(
                clean_key, lower=lower_bound, upper=upper_bound, shape=(raw_action.shape[0], self.K, raw_action.shape[1])
            )
            normal_noise = jax.random.normal(flow_noise_key, shape=(raw_action.shape[0], self.K, raw_action.shape[1]))
            normal_noise_clip = jnp.clip(normal_noise, min=lower_bound, max=upper_bound)
            noise = jnp.where(jnp.isnan(tnormal_noise), normal_noise_clip, tnormal_noise)
            clean_samples = 1 / t[:, :, None] * noisy_actions_repeat - std * noise

            obs_repeat = jnp.repeat(jnp.expand_dims(obs, axis=1), repeats=self.K, axis=1)
            exec_clean, _, _, _ = project_action_jax(obs_repeat, clean_samples)
            critic = get_min_q(obs_repeat, exec_clean)

            qh_values = self.agent.qh(qh_params, obs_repeat, clean_samples)
            qh_values = jnp.clip(qh_values, 0.0, 1.0)

            if self.use_feasibility_guidance:
                score = critic - self.beta_h * qh_values
            else:
                score = critic
            weight = nn.softmax((1.0 / jnp.exp(log_alpha)) * score, axis=1)

            obs_expanded = jnp.repeat(obs, self.K, axis=0)

            def policy_loss_fn(policy_params_inner):
                def denoiser(x, r_in, t_in):
                    return self.agent.policy(policy_params_inner, obs_expanded, x, r_in, t_in)

                loss, dudt, u_out, dudt_out, dudt_max = self.agent.flow.reverse_weighted_p_loss(
                    weight, denoiser, r, t, clean_samples, noise, noisy_actions
                )
                return loss, (dudt, u_out, dudt_out, dudt_max)

            (policy_loss, (_, u_out, dudt_out, dudt_max)), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(policy_params)

            def log_alpha_loss_fn(log_alpha_inner):
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha_inner)) ** 2
                )
                return -1 * log_alpha_inner * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)

            def param_update(optim, params_inner, grads, opt_state_inner):
                update, new_opt_state = optim.update(grads, opt_state_inner)
                new_params = optax.apply_updates(params_inner, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params_inner, grads, opt_state_inner):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda p, o: param_update(optim, p, grads, o),
                    lambda p, o: (p, o),
                    params_inner,
                    opt_state_inner,
                )

            def delay_alpha_param_update(optim, params_inner, opt_state_inner):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda p, o: param_update(optim, p, jax.grad(log_alpha_loss_fn)(p), o),
                    lambda p, o: (p, o),
                    params_inner,
                    opt_state_inner,
                )

            def delay_target_update(params_inner, target_params_inner, tau_inner):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda tp: optax.incremental_update(params_inner, tp, tau_inner),
                    lambda tp: tp,
                    target_params_inner,
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            qh_params, qh_opt_state = param_update(self.optim, qh_params, qh_grads, qh_opt_state)
            vh_params, vh_opt_state = param_update(self.optim, vh_params, vh_grads, vh_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_vh_params = delay_target_update(vh_params, target_vh_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_state = SafeMFSACTrainState(
                params=SafeMFSACParams(
                    q1=q1_params,
                    q2=q2_params,
                    target_q1=target_q1_params,
                    target_q2=target_q2_params,
                    qh=qh_params,
                    vh=vh_params,
                    target_vh=target_vh_params,
                    policy=policy_params,
                    target_policy=target_policy_params,
                    log_alpha=log_alpha,
                ),
                opt_state=SafeMFSACOptStates(
                    q1=q1_opt_state,
                    q2=q2_opt_state,
                    qh=qh_opt_state,
                    vh=vh_opt_state,
                    policy=policy_opt_state,
                    log_alpha=log_alpha_opt_state,
                ),
                step=step + 1,
            )

            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "policy_loss": policy_loss,
                "qh_loss": qh_loss,
                "vh_loss": vh_loss,
                "qh_mean": jnp.mean(qh_pred),
                "qh_max": jnp.max(qh_pred),
                "vh_mean": jnp.mean(vh_pred),
                "safe_violation_batch": jnp.mean(safe_violation),
                "filter_active_batch": jnp.mean(data.filter_active),
                "projection_gap_batch": jnp.mean(jnp.abs(data.projection_gap)),
                "fg_score_mean": jnp.mean(score),
                "fg_score_std": jnp.std(score),
                "alpha": jnp.exp(log_alpha),
                "u_pred": jnp.mean(u_out),
                "dudt": jnp.mean(dudt_out),
                "dudt_max": dudt_max,
                "q1_mean": jnp.mean(q1_pred),
                "q2_mean": jnp.mean(q2_pred),
            }
            return new_state, info

        self._implement_common_behavior(
            stateless_update,
            self.agent.get_action,
            self.agent.get_deterministic_action,
            stateless_get_vanilla_action=self.agent.get_vanilla_action,
        )
        self._get_qh = jax.jit(lambda params, obs, act: self.agent.qh(params, obs, act))
        self._get_vh = jax.jit(lambda params, obs: self.agent.vh(params, obs))

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)

    def get_policy_params_to_save(self):
        return (
            self.state.params.target_policy,
            self.state.params.log_alpha,
            self.state.params.q1,
            self.state.params.q2,
        )

    def get_qh_params(self):
        return self.state.params.qh

    def get_vh_params(self):
        return self.state.params.vh

    def get_qh(self, obs, raw_action):
        return np.asarray(self._get_qh(self.get_qh_params(), obs, raw_action))

    def get_vh(self, obs):
        return np.asarray(self._get_vh(self.get_vh_params(), obs))

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
