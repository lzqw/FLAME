import os
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal
import optax
import math

# ==========================================
# [防爆显存设置]
# ==========================================
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


# ==========================================
# 1. 定义真实分布 (Ground Truth)
# ==========================================
class ToyGMM:
    def __init__(self):
        self.means = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
        self.std = 0.1
        self.cov = np.eye(2) * (self.std ** 2)
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])

    def sample(self, n):
        indices = np.random.choice(4, size=n, p=self.weights)
        samples = np.array([np.random.multivariate_normal(self.means[i], self.cov) for i in indices])
        return jnp.array(samples)

    def log_prob(self, x, y=None):
        if y is None:
            pos = x
        else:
            pos = np.dstack((x, y))
        p = np.zeros(pos.shape[:-1])
        for i, mean in enumerate(self.means):
            p += self.weights[i] * multivariate_normal.pdf(pos, mean=mean, cov=self.cov)
        return np.log(p + 1e-10)


# ==========================================
# 2. 网络定义
# ==========================================
class TimestepEmbedding(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if self.dim % 2 == 1:
            emb = jnp.pad(emb, [[0, 0], [0, 1]])
        return emb


def net_fn(x, t):
    if t.ndim == 0: t = t * jnp.ones((x.shape[0], 1))
    if t.ndim == 1: t = t.reshape(-1, 1)

    t_emb = TimestepEmbedding(32)(t)
    x_embed = hk.Linear(64)(x)
    t_embed_split = hk.Linear(64)(t_emb)

    h = jax.nn.silu(x_embed + t_embed_split)

    mlp = hk.Sequential([
        hk.Linear(64), jax.nn.silu,
        hk.Linear(64), jax.nn.silu,
        hk.Linear(64), jax.nn.silu,
        hk.Linear(2)
    ])
    return mlp(h)


def mean_flow_wrapper(params, network_apply, x, r, t):
    """
    MeanFlow 适配器 (Fix: Handle Unbatched Inputs)
    """
    # [Fix Start] --------------------------------------------
    # 检测输入是否为单样本 (D,)，如果是，扩展为 (1, D)
    is_unbatched = (x.ndim == 1)
    if is_unbatched:
        x_in = x[None, :]  # Shape: (1, D)
    else:
        x_in = x
    # [Fix End] ----------------------------------------------

    # 网络前向传播
    v = network_apply(params, x_in, t)

    # [Fix Start] --------------------------------------------
    # 如果输入是单样本，将输出 squeeze 回 (D,)
    if is_unbatched:
        v = v[0]
    # [Fix End] ----------------------------------------------

    # 计算位移 drift
    dt = t - r

    # 广播 dt
    if dt.ndim == 0:
        dt = jnp.full_like(x, dt)
    elif dt.ndim == 1:
        if not is_unbatched:
            dt = dt[:, None]

    return v * dt


# ==========================================
# 3. MeanFlow 核心逻辑
# ==========================================
class MeanFlowEstimator:
    def __init__(self, network_apply, params, num_timesteps=20):
        self.network_apply = network_apply
        self.params = params
        self.num_timesteps = num_timesteps

    def model_fn(self, x, r, t):
        return mean_flow_wrapper(self.params, self.network_apply, x, r, t)

    def p_sample_ent(self, key: jax.Array, shape: tuple):
        key_z, key_probe = jax.random.split(key)
        x = jax.random.normal(key_z, shape)
        log_p = jnp.sum(jax.scipy.stats.norm.logpdf(x), axis=-1)

        epsilon = jax.random.rademacher(key_probe, shape, dtype=jnp.float32)
        dt = 1.0 / self.num_timesteps

        def body_fn(carry, step_idx):
            x, current_log_p = carry
            t = step_idx * dt
            t_next = (step_idx + 1) * dt

            def step_fn(x_in):
                return self.model_fn(x_in, t, t_next)

            drift, jvp_val = jax.jvp(step_fn, (x,), (epsilon,))
            trace = jnp.sum(epsilon * jvp_val, axis=-1)
            trace = jnp.clip(trace, -10.0, 10.0)

            x_next = x + drift
            next_log_p = current_log_p - trace

            return (x_next, next_log_p), None

        t_seq = jnp.arange(self.num_timesteps)
        (final_x, final_log_p), _ = jax.lax.scan(body_fn, (x, log_p), t_seq)
        return final_x, final_log_p


# ==========================================
# 4. 主程序
# ==========================================
def main():
    total_steps = 10000
    batch_size = 512
    meanflow_steps = 20

    rng = jax.random.PRNGKey(42)
    gmm = ToyGMM()

    dummy_x = jnp.zeros((1, 2))
    dummy_t = jnp.zeros((1, 1))
    network = hk.without_apply_rng(hk.transform(net_fn))
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, dummy_x, dummy_t)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x_1, key):
        x_0 = jax.random.normal(key, x_1.shape)
        t = jax.random.uniform(key, (x_1.shape[0], 1))
        x_t = t * x_1 + (1 - t) * x_0
        target_v = x_1 - x_0
        pred_v = network.apply(params, x_t, t.flatten())
        return jnp.mean((pred_v - target_v) ** 2)

    @jax.jit
    def update(params, opt_state, batch, key):
        grads = jax.grad(loss_fn)(params, batch, key)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    print(f"Start training for {total_steps} steps...")
    for i in range(total_steps):
        rng, key_step = jax.random.split(rng)
        batch = gmm.sample(batch_size)
        params, opt_state = update(params, opt_state, batch, key_step)
        if i % 2000 == 0:
            print(f"Step {i}")

    print("Training finished. Evaluating Entropy...")

    estimator = MeanFlowEstimator(network.apply, params, num_timesteps=meanflow_steps)

    eval_batch_size = 2000
    rng, eval_key = jax.random.split(rng)
    keys = jax.random.split(eval_key, eval_batch_size)

    def single_sample(k):
        return estimator.p_sample_ent(k, (2,))

    generated_samples, estimated_logp = jax.vmap(single_sample)(keys)
    true_logp = gmm.log_prob(np.array(generated_samples))

    mae = jnp.mean(jnp.abs(estimated_logp - true_logp))
    corr = np.corrcoef(estimated_logp, true_logp)[0, 1]
    print(f"MAE: {mae:.4f}, Correlation: {corr:.4f}")

    # Plot
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)

    ax0 = fig.add_subplot(gs[0])
    x = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x, x)
    Z_true = gmm.log_prob(X, Y)
    ax0.contourf(X, Y, Z_true, levels=50, cmap='viridis')
    ax0.set_title("Ground Truth")

    ax1 = fig.add_subplot(gs[1])
    sc = ax1.scatter(generated_samples[:, 0], generated_samples[:, 1], c=estimated_logp, s=2, cmap='viridis')
    ax1.set_title("Generated Samples (Color=Est LogP)")
    plt.colorbar(sc, ax=ax1)

    ax2 = fig.add_subplot(gs[2])
    min_val = min(true_logp.min(), estimated_logp.min())
    max_val = max(true_logp.max(), estimated_logp.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax2.scatter(true_logp, estimated_logp, s=5, alpha=0.5)
    ax2.set_xlabel("True LogP")
    ax2.set_ylabel("Est LogP")
    ax2.set_title(f"Accuracy (Corr={corr:.3f})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
