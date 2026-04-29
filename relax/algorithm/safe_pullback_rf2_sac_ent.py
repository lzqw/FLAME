from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_pullback_rf2_sac_ent import SafePullbackRF2SACENTNet, SafePullbackRF2Params
from relax.safety.obstacle_navigation_filter import ObstacleNavConfig, make_action_grid, project_action_jax_batched


class SafePullbackRF2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    qp: optax.OptState
    vp: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class SafePullbackRF2TrainState(NamedTuple):
    params: SafePullbackRF2Params
    opt_state: SafePullbackRF2OptStates
    step: int
    entropy: float


class SafePullbackRF2SACENT(Algorithm):
    def __init__(self, agent: SafePullbackRF2SACENTNet, params: SafePullbackRF2Params, gamma=0.99, gamma_p=0.99,
                 lr=3e-4, alpha_lr=1e-2, tau=0.005, reward_scale=1.0, sample_k=64, lambda_p=1.0,
                 use_projection_critic=True, fixed_alpha=False, alpha_value=0.01,
                 lambda_p_warmup_steps=100000, lambda_d=0.5):
        self.agent = agent
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.tau = tau
        self.reward_scale = reward_scale
        self.K = sample_k
        self.lambda_p = lambda_p
        self.use_projection_critic = use_projection_critic
        self.lambda_p_warmup_steps = lambda_p_warmup_steps
        self.lambda_d = lambda_d
        self.fixed_alpha = fixed_alpha
        self.alpha_value = alpha_value
        self.optim = optax.adam(lr)
        self.policy_optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)
        self.cfg = ObstacleNavConfig()
        self.action_grid = jnp.asarray(make_action_grid(61))

        self.state = SafePullbackRF2TrainState(
            params=params,
            opt_state=SafePullbackRF2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                qp=self.optim.init(params.qp),
                vp=self.optim.init(params.vp),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=0,
            entropy=0.0,
        )

        @jax.jit
        def _update(key, state, data):
            obs, exec_action, raw_action = data.obs, data.action, data.raw_action
            reward, next_obs, done = data.reward, data.next_obs, data.done
            projection_cost = data.projection_cost
            p = state.params
            o = state.opt_state

            k1, k2, k3 = jax.random.split(key, 3)
            raw_next_action, entropy = self.agent.get_action_ent(k1, (p.policy, p.log_alpha, p.q1, p.q2), next_obs)
            exec_next_action, _, _ = project_action_jax_batched(next_obs, raw_next_action, self.action_grid, self.cfg)

            q1_t = self.agent.q(p.target_q1, next_obs, exec_next_action)
            q2_t = self.agent.q(p.target_q2, next_obs, exec_next_action)
            alpha = jnp.float32(self.alpha_value) if self.fixed_alpha else jnp.exp(p.log_alpha)
            q_backup = reward * self.reward_scale + (1.0 - done) * self.gamma * (jnp.minimum(q1_t, q2_t) - alpha * entropy)

            def qloss(qp, target):
                pred = self.agent.q(qp, obs, exec_action)
                return jnp.mean((pred - jax.lax.stop_gradient(target)) ** 2), pred

            (q1_loss, q1_pred), q1_grads = jax.value_and_grad(qloss, has_aux=True)(p.q1, q_backup)
            (q2_loss, q2_pred), q2_grads = jax.value_and_grad(qloss, has_aux=True)(p.q2, q_backup)

            vp_next = self.agent.get_vp(p.target_vp, next_obs)
            yp = projection_cost + self.gamma_p * (1.0 - done) * vp_next

            def qploss(qp):
                pred = self.agent.get_qp(qp, obs, raw_action)
                return jnp.mean((pred - jax.lax.stop_gradient(yp)) ** 2), pred

            if self.use_projection_critic:
                (qp_loss, qp_pred), qp_grads = jax.value_and_grad(qploss, has_aux=True)(p.qp)
            else:
                qp_loss, qp_pred = jnp.float32(0.0), jnp.zeros_like(reward)
                qp_grads = jax.tree_util.tree_map(jnp.zeros_like, p.qp)

            def vploss(vp):
                pred = self.agent.get_vp(vp, obs)
                policy_actions = jax.random.uniform(k2, (obs.shape[0], 8, raw_action.shape[-1]), minval=-1.0, maxval=1.0)
                policy_obs = jnp.repeat(obs[:, None, :], 8, axis=1)
                target = jax.lax.stop_gradient(jnp.mean(self.agent.get_qp(p.qp, policy_obs, policy_actions), axis=1))
                return jnp.mean((pred - target) ** 2), pred

            if self.use_projection_critic:
                (vp_loss, vp_pred), vp_grads = jax.value_and_grad(vploss, has_aux=True)(p.vp)
            else:
                vp_loss, vp_pred = jnp.float32(0.0), jnp.zeros_like(reward)
                vp_grads = jax.tree_util.tree_map(jnp.zeros_like, p.vp)

            t = jax.random.uniform(k2, (obs.shape[0], 1), minval=1e-3, maxval=0.994)
            noise = jax.random.normal(k3, raw_action.shape)
            noisy = jnp.clip(t * raw_action + (1 - t) * noise, -1.0, 1.0)
            noisy_rep = jnp.repeat(noisy[:, None, :], self.K, axis=1)
            obs_rep = jnp.repeat(obs[:, None, :], self.K, axis=1)
            std = jnp.expand_dims((1 - t) / jnp.maximum(t, 1e-6), axis=-1)
            lower_bound = noisy_rep / jnp.maximum(1 - t[:, :, None], 1e-6) - (1.0 / jnp.maximum(std, 1e-6))
            upper_bound = noisy_rep / jnp.maximum(1 - t[:, :, None], 1e-6) + (1.0 / jnp.maximum(std, 1e-6))
            trunc_noise = jax.random.truncated_normal(k1, lower=lower_bound, upper=upper_bound, shape=(raw_action.shape[0], self.K, raw_action.shape[1]))
            normal_noise = jax.random.normal(k2, shape=(raw_action.shape[0], self.K, raw_action.shape[1]))
            noise_k = jnp.where(jnp.isnan(trunc_noise), jnp.clip(normal_noise, lower_bound, upper_bound), trunc_noise)
            clean = jnp.clip(noisy_rep / jnp.maximum(t[:, :, None], 1e-6) - std * noise_k, -1.0, 1.0)
            exec_clean, _, _ = project_action_jax_batched(obs_rep, clean, self.action_grid, self.cfg)

            q_reward = jnp.minimum(self.agent.q(p.q1, obs_rep, exec_clean), self.agent.q(p.q2, obs_rep, exec_clean))
            q_proj = self.agent.get_qp(p.qp, obs_rep, clean) if self.use_projection_critic else jnp.zeros_like(q_reward)
            d_proj = jnp.linalg.norm(clean - exec_clean, axis=-1)
            lambda_p_current = self.lambda_p * jnp.minimum(1.0, state.step / jnp.maximum(self.lambda_p_warmup_steps, 1))
            score = jax.lax.stop_gradient(q_reward - lambda_p_current * q_proj - self.lambda_d * d_proj)
            critic = score / jnp.maximum(alpha, 1e-3)
            w = jnp.exp(critic - jax.nn.logsumexp(critic, axis=1, keepdims=True))

            obs_r = obs_rep.reshape(-1, obs.shape[-1])
            clean_r = clean.reshape(-1, raw_action.shape[-1])
            noisy_r = noisy_rep.reshape(-1, raw_action.shape[-1])
            t_r = jnp.repeat(t.squeeze(-1), self.K)
            w_r = w.reshape(-1, 1)
            u_r = clean_r - noisy_r

            def ploss(pp):
                denoiser = lambda tt, xx: self.agent.policy(pp, obs_r, xx, tt)
                loss = self.agent.flow.reverse_weighted_p_loss2(denoiser, t_r, noisy_r, jax.lax.stop_gradient(w_r), jax.lax.stop_gradient(u_r))
                return loss

            policy_loss, policy_grads = jax.value_and_grad(ploss)(p.policy)

            if self.use_projection_critic:
                cf_policy = clean[:, :8, :]
                cf_uniform = jax.random.uniform(k3, (obs.shape[0], 8, raw_action.shape[1]), minval=-1.0, maxval=1.0)
                cf_actions = jnp.concatenate([cf_policy, cf_uniform], axis=1)
                cf_obs = jnp.repeat(obs[:, None, :], cf_actions.shape[1], axis=1)
                cf_exec, _, _ = project_action_jax_batched(cf_obs, cf_actions, self.action_grid, self.cfg)
                d_cf = jnp.linalg.norm(cf_actions - cf_exec, axis=-1)
                q_cf = self.agent.get_qp(p.qp, cf_obs, cf_actions)
                l_cf = jnp.mean((q_cf - jax.lax.stop_gradient(d_cf)) ** 2)
                lb = jnp.mean(jax.nn.relu(projection_cost - self.agent.get_qp(p.qp, obs, raw_action)) ** 2)
                qp_loss = qp_loss + 0.5 * l_cf + 0.5 * lb

            def aloss(log_alpha):
                return jnp.mean(log_alpha * (jnp.mean(entropy) - self.agent.target_entropy))

            alpha_grads = jax.grad(aloss)(p.log_alpha)

            def apply(optim, params, grads, st):
                upd, ns = optim.update(grads, st)
                return optax.apply_updates(params, upd), ns

            nq1, oq1 = apply(self.optim, p.q1, q1_grads, o.q1)
            nq2, oq2 = apply(self.optim, p.q2, q2_grads, o.q2)
            nqp, oqp = apply(self.optim, p.qp, qp_grads, o.qp)
            nvp, ovp = apply(self.optim, p.vp, vp_grads, o.vp)
            npol, opol = apply(self.policy_optim, p.policy, policy_grads, o.policy)
            nloga, ologa = apply(self.alpha_optim, p.log_alpha, alpha_grads, o.log_alpha)

            t_q1 = optax.incremental_update(nq1, p.target_q1, self.tau)
            t_q2 = optax.incremental_update(nq2, p.target_q2, self.tau)
            t_vp = optax.incremental_update(nvp, p.target_vp, self.tau)
            t_pol = optax.incremental_update(npol, p.target_policy, self.tau)

            ns = SafePullbackRF2TrainState(
                params=SafePullbackRF2Params(nq1, nq2, t_q1, t_q2, nqp, nvp, t_vp, npol, t_pol, nloga),
                opt_state=SafePullbackRF2OptStates(oq1, oq2, oqp, ovp, opol, ologa),
                step=state.step + 1,
                entropy=jnp.mean(entropy),
            )
            info = dict(q1_loss=q1_loss, q2_loss=q2_loss, qp_loss=qp_loss, vp_loss=vp_loss,
                        policy_loss=policy_loss, alpha=jnp.exp(nloga),
                        q_reward_mean=jnp.mean(q_reward), q_projection_mean=jnp.mean(q_proj),
                        projection_cost_batch=jnp.mean(projection_cost),
                        safe_pullback_score_mean=jnp.mean(score))
            return ns, info

        self._update = _update

    def update(self, key, data):
        self.state, info = self._update(key, self.state, data)
        return {k: float(v) for k, v in info.items()}

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.agent.get_action(key, (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2), obs), dtype=np.float32)
