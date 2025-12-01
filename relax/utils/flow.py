from typing import Protocol, Tuple
from dataclasses import dataclass
from jax.lax import scan
import numpy as np
import jax, jax.numpy as jnp
import optax

class FlowModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...

class MeanFlowModel(Protocol):
    def __call__(self, x: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class OTFlow:
    num_timesteps: int

    def p_sample(self, key: jax.Array, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        #x = jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = t * dt
            drift = model(tau, x)
            x_next = x + drift * dt
            return x_next, None
        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def p_sample_traj(self, key: jax.Array, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = t * dt
            drift = model(tau, x)
            x_next = x + drift * dt
            return x_next, x_next
        t_seq = jnp.arange(self.num_timesteps)
        _, x = jax.lax.scan(body_fn, x, t_seq)
        return x

    def p_sample_fast(self, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = jnp.zeros(shape)
        drift = model(0, x)
        return drift

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        return t * x_start + (1 - t) * noise

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: FlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        # noise = 0.3 * jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(t, x_t)
        loss = weights * optax.squared_error(v_pred, (x_start - noise))
        return loss.mean()

    def reverse_weighted_p_loss(self,  model: FlowModel, t: jax.Array,
                        x_t: jax.Array, u_estimation:jax.Array):
        t_squeezed = jnp.squeeze(t)
        v_pred = model(t_squeezed, x_t)
        loss = optax.squared_error(v_pred, u_estimation)
        return loss.mean()

    def reverse_weighted_p_loss2(self,  model: FlowModel, t: jax.Array,
                        x_t: jax.Array,weight:jax.Array, u:jax.Array):
        # t_squeezed = jnp.squeeze(t)
        v_pred = model(t, x_t)
        loss = weight * optax.squared_error(v_pred, u)
        return loss.mean()

    def weighted_p_loss_coupled(self, noise: jax.Array, weights: jax.Array, model: FlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(t, x_t)
        loss = weights * optax.squared_error(v_pred, (x_start - noise))
        return loss.mean()

    def recon_sample(self, t: jax.Array, x_t: jax.Array, noise: jax.Array):
        return (1 / t[:, jnp.newaxis]) * x_t - (1-t[:, jnp.newaxis])/t[:, jnp.newaxis] * noise

#t_final_unique,at,ut
    def recon_weighted_p_loss(self,  model, t_final_unique:jax.Array,at:jax.Array,ut:jax.Array):
        at=jnp.mean(at,axis=1)
        v_pred = model(t_final_unique, at)
        loss = optax.squared_error(v_pred, ut)
        return loss.mean()


@dataclass(frozen=True)
class MeanFlow:
    num_timesteps: int

    def p_sample(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps

        def body_fn(x, t):
            tau = (self.num_timesteps - t) * dt
            # drift = model(x, tau, tau)
            drift = model(x, tau - dt, tau)
            x_next = x - drift
            # x_next = x - drift * dt
            return x_next, None

        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def p_sample_traj(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps

        def body_fn(x, t):
            tau = t * dt
            drift = model(x, tau, tau + dt)
            x_next = x - drift
            return x_next, x_next

        t_seq = jnp.arange(self.num_timesteps)
        _, x = jax.lax.scan(body_fn, x, t_seq)
        return x

    def p_sample_fast(self, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = jnp.zeros(shape)
        drift = model(x, 0, 1)
        return -drift

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        return (1 - t) * x_start + t * noise

    def p_sample_ent(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> Tuple[jax.Array, jax.Array]:
        """
        采样动作并同时利用 Hutchinson 迹估计器计算其对数概率 (Entropy)。
        """
        key_z, key_probe = jax.random.split(key)

        # 2. 初始状态采样 x ~ N(0, 0.5^2)
        x = 0.5 * jax.random.normal(key_z, shape)

        # 3. 计算初始分布的 Log Probability
        # [CRITICAL FIX]: 使用 axis=-1 确保对每个样本独立求和，保留 Batch 维度
        log_p = jnp.sum(jax.scipy.stats.norm.logpdf(x, loc=0.0, scale=1.0), axis=-1)

        # 4. Hutchinson 探测向量
        epsilon = jax.random.normal(key_probe, shape)

        dt = 1.0 / self.num_timesteps
        t_seq = jnp.arange(self.num_timesteps)

        def body_fn(carry, t_idx):
            x, current_log_p = carry
            tau = (self.num_timesteps - t_idx) * dt
            prev_tau = tau - dt

            def drift_fn(x_in):
                return model(x_in, prev_tau, tau)

            drift, jvp_val = jax.jvp(drift_fn, (x,), (epsilon,))

            # [CRITICAL FIX]: 只在特征维度求和 (axis=-1)，保留 (Batch,) 维度
            div_drift = jnp.sum(epsilon * jvp_val, axis=-1)
            div_drift = jnp.clip(div_drift, -10.0, 10.0)
            x_next = x - drift
            next_log_p = current_log_p + div_drift

            return (x_next, next_log_p), None

        (final_x, final_log_p), _ = jax.lax.scan(body_fn, (x, log_p), t_seq)

        return final_x, final_log_p

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert r.ndim == 1 and t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v = noise - x_start
        zero_r = jnp.zeros_like(r, dtype=jnp.float32)
        one_t = jnp.ones_like(t, dtype=jnp.float32)
        u_pred, dudt = jax.jvp(model, (x_t, r, t), (v, zero_r, one_t))
        u_tgt = jax.lax.stop_gradient(v - (t - r)[:, None] * dudt)
        loss = weights * optax.squared_error(u_pred, u_tgt)
        return loss.mean(),jax.lax.stop_gradient(dudt)

    def reverse_weighted_p_loss(self, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
                                x_start: jax.Array, noise: jax.Array, x_t: jax.Array):
        u =  noise - x_start  # Shape: (B, K, A)
        B, K, D = x_start.shape
        # Flatten inputs for the model
        x_t_flat = jnp.repeat(jnp.expand_dims(x_t, axis=1), repeats=K, axis=1).reshape(B * K, D) # B*K-D
        # <<< FIX: Reshape r and t to be 1D vectors (B*K,) instead of 2D (B*K, 1)
        r_flat = jnp.repeat(jnp.expand_dims(r, axis=1), repeats=K, axis=1).reshape(B * K)
        t_flat = jnp.repeat(jnp.expand_dims(t, axis=1), repeats=K, axis=1).reshape(B * K)

        # Reshape tangents to match the primals
        u_flat = u.reshape(B * K, D)
        # Their shape will now correctly be (B * K,)
        zero_tangent_flat = jnp.zeros_like(r_flat)
        one_tangent_flat = jnp.ones_like(t_flat)
        # one_tangent_flat = jnp.zeros_like(t_flat)
        # Call jvp with the flattened tensors
        u_pred_flat, dudt_flat = jax.jvp(
            model,
            (x_t_flat, r_flat, t_flat),
            (u_flat, zero_tangent_flat, one_tangent_flat)
        )
        dudt_max_value=jnp.max(jnp.abs(dudt_flat))
        # dudt_flat = jnp.clip(dudt_flat, min=-1, max=1)
        # u_pred_flat = jnp.clip(u_pred_flat, min=-1, max=1)
        #clip is efficient


        u_pred_b_k_d = u_pred_flat.reshape(B, K, D)
        dudt = dudt_flat.reshape(B, K, D)

        # --- 核心修改：接受你的建议进行优化 ---
        # 由于 u_pred 在 K 维度上是冗余的，我们只取第一个切片，将其降维
        # Shape: (B, K, D) -> (B, D)
        # u_pred = u_pred_b_k_d[:, 0, :]
        u_pred = u_pred_b_k_d

        # 目标 u_tgt 的计算仍然依赖于 K 个不同的噪声，所以这部分不变
        u_tgt = jax.lax.stop_gradient(u - (t - r)[:, None] * dudt)
        # u_tgt_estimation = jax.lax.stop_gradient(jnp.sum(weights[:, :, None] * u_tgt, axis=1))
        u_tgt_estimation = jax.lax.stop_gradient(u_tgt)
        weighted_error = weights[:, :, None] * optax.squared_error(u_pred, u_tgt_estimation)

        loss = jnp.mean(weighted_error)

        return loss, jax.lax.stop_gradient(dudt), jax.lax.stop_gradient(u_pred_b_k_d),jax.lax.stop_gradient(dudt),jax.lax.stop_gradient(dudt_max_value)

    # def reverse_weighted_p_loss(self, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
    #                             x_start: jax.Array, noise: jax.Array, x_t: jax.Array):
    #     """
    #     计算加权的 flow-matching 损失 (内存优化版)。
    #
    #     使用 jax.lax.scan 逐个处理 K 个样本，避免 OOM。
    #     """
    #
    #     u = x_start - noise  # Shape: (B, K, D)
    #     B, K, D = x_start.shape
    #
    #     u_transposed = jnp.transpose(u, (1, 0, 2))  # Shape: (K, B, D)
    #
    #     zero_tangent_r = jnp.zeros_like(r.squeeze(axis=-1))  # Shape: (B,)
    #     one_tangent_t = jnp.ones_like(t.squeeze(axis=-1))  # Shape: (B,)
    #     r_squeezed = r.squeeze(axis=-1)  # Shape: (B,)
    #     t_squeezed = t.squeeze(axis=-1)  # Shape: (B,)
    #
    #     def scan_body(carry, u_k):
    #
    #         u_pred_k, dudt_k = jax.jvp(
    #             model,
    #             # Primals: x_t, r, t 的批次大小都是 B
    #             (x_t, r_squeezed, t_squeezed),
    #             # Tangents: u_k 的批次大小是 B
    #             (u_k, zero_tangent_r, one_tangent_t)
    #         )
    #         # 返回 carry 和本次迭代的结果
    #         return carry, (u_pred_k, dudt_k)
    #
    #
    #     _, (u_pred_stacked, dudt_stacked) = scan(scan_body, None, u_transposed)
    #
    #
    #     u_pred = jnp.transpose(u_pred_stacked, (1, 0, 2))  # Shape: (B, K, D)
    #     dudt = jnp.transpose(dudt_stacked, (1, 0, 2))  # Shape: (B, K, D)
    #
    #     u_tgt = u - (t - r)[:, None] * dudt
    #     u_tgt = jax.lax.stop_gradient(u_tgt)
    #
    #     u_tgt_estimation = jnp.sum(weights[:, :, None] * u_tgt, axis=1)
    #
    #     squared_error = optax.squared_error(u_pred, jax.lax.stop_gradient(u_tgt_estimation))
    #
    #     loss = jnp.mean(weights[:, :, None] * squared_error)
    #     return loss
