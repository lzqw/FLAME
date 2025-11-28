import os

# ==========================================
# [关键修改 1] 设置环境变量防止 OOM
# 必须在 import jax 之前设置
# ==========================================
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# 如果仍然报错，取消下面这行的注释，限制只使用 50% 显存
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

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

    def log_prob(self, x, y):
        pos = np.dstack((x, y))
        p = np.zeros(x.shape)
        for i, mean in enumerate(self.means):
            p += self.weights[i] * multivariate_normal.pdf(pos, mean=mean, cov=self.cov)
        return np.log(p + 1e-10)


# ==========================================
# 2. 网络定义 (优化版)
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
    # [关键修改 3] 网络宽度从 256 降为 128，节省显存，同时保持表达能力
    hidden_dim = 128

    if t.ndim == 0: t = t * jnp.ones((x.shape[0], 1))
    if t.ndim == 1: t = t.reshape(-1, 1)

    t_emb = TimestepEmbedding(64)(t)

    x_embed = hk.Linear(hidden_dim)(x)
    t_embed = hk.Linear(hidden_dim)(t_emb)

    h = jax.nn.silu(x_embed + t_embed)

    mlp = hk.Sequential([
        hk.Linear(hidden_dim), jax.nn.silu,
        hk.Linear(hidden_dim), jax.nn.silu,
        hk.Linear(hidden_dim), jax.nn.silu,
        hk.Linear(2)
    ])

    return mlp(h)


# ==========================================
# 3. 核心逻辑：Estimator
# ==========================================
class Estimator:
    def __init__(self, policy_apply):
        self.policy = policy_apply

    def compute_log_likelihood(self, key: jax.Array, params: hk.Params, act: jax.Array,
                               num_steps: int) -> jax.Array:
        def model_fn(t, x):
            return self.policy(params, x, t)

        def log_p0(z):
            return jax.scipy.stats.norm.logpdf(z).sum(axis=-1)

        z_key, _ = jax.random.split(key)
        z = jax.random.normal(z_key, act.shape)

        def ode_dynamics(state, t):
            f_t, _ = state
            u_t_fn = lambda x: model_fn(t, x)
            _, vjp_fn = jax.vjp(u_t_fn, f_t)
            vjp_z = vjp_fn(z)[0]
            trace_term = jnp.sum(vjp_z * z, axis=-1)
            df_dt = u_t_fn(f_t)
            dg_dt = -trace_term
            return df_dt, dg_dt

        dt = -1.0 / num_steps

        def solver_step(state, t):
            f_t, g_t = state
            df_dt, dg_dt = ode_dynamics(state, t)
            f_next = f_t + df_dt * dt
            g_next = g_t + dg_dt * dt
            return (f_next, g_next), None

        timesteps = jnp.linspace(1.0, 1.0 / num_steps, num_steps)
        initial_state = (act, jnp.zeros(act.shape[:-1]))
        final_state, _ = jax.lax.scan(solver_step, initial_state, timesteps)
        f_0, g_0 = final_state

        log_p1 = log_p0(f_0) - g_0
        return log_p1


# ==========================================
# 4. 主程序
# ==========================================
def main():
    rng = jax.random.PRNGKey(42)
    gmm = ToyGMM()

    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.zeros((1, 2))
    dummy_t = jnp.zeros((1, 1))
    network = hk.without_apply_rng(hk.transform(net_fn))
    params = network.init(init_rng, dummy_x, dummy_t)

    # 保持 20000 步以获得好效果
    total_steps = 20000
    lr_schedule = optax.cosine_decay_schedule(
        init_value=1e-3,
        decay_steps=total_steps,
        alpha=1e-2
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
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

    print(f"Training toy model (Flow Matching) for {total_steps} steps...")

    # [关键修改 2] Batch Size 降为 256
    batch_size = 256

    for i in range(total_steps):
        rng, step_key = jax.random.split(rng)
        batch = gmm.sample(batch_size)
        params, opt_state = update(params, opt_state, batch, step_key)

        if (i + 1) % 2000 == 0:
            print(f"Step {i + 1}/{total_steps}")

    print("Generating Figure 11...")
    estimator = Estimator(network.apply)

    # 网格
    grid_size = 50
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points_jax = jnp.array(points)

    def estimate_grid_impl(key, T, N):
        keys = jax.random.split(key, N)
        logp_samples = jax.vmap(
            lambda k: estimator.compute_log_likelihood(k, params, points_jax, T)
        )(keys)
        logp_avg = jnp.mean(logp_samples, axis=0)
        return logp_avg.reshape(grid_size, grid_size)

    # 显式 JIT
    estimate_grid = jax.jit(estimate_grid_impl, static_argnums=(1, 2))

    fig = plt.figure(figsize=(14, 6), dpi=100)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 0.2, 1, 1])

    # (a) True Log Probability
    ax_true = fig.add_subplot(gs[:, 0])
    Z_true = gmm.log_prob(X, Y)

    vmin, vmax = Z_true.min(), Z_true.max()

    im_true = ax_true.contourf(X, Y, Z_true, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    ax_true.set_aspect('equal')
    ax_true.set_title('(a) True log probability', y=-0.15)
    ax_true.set_xlabel('$x_0$')
    ax_true.set_ylabel('$x_1$')
    plt.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)

    # (b) Estimated Log Probability
    settings = [
        (20, 50, 0, 2),
        (20, 10, 0, 3),
        (50, 50, 1, 2),
        (50, 10, 1, 3)
    ]

    for T_val, N_val, row, col in settings:
        rng, key = jax.random.split(rng)
        print(f"Computing for T={T_val}, N={N_val}...")
        Z_est = estimate_grid(key, T_val, N_val)

        ax = fig.add_subplot(gs[row, col])
        im = ax.contourf(X, Y, Z_est, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')

        if row == 0: ax.set_title(f'N={N_val}')
        if col == 2:
            ax.set_ylabel('$x_1$')
            ax.text(-1.5, 0, f'T={T_val}', va='center', ha='right', fontsize=12, fontweight='bold')
        else:
            ax.set_yticks([])
        if row == 1:
            ax.set_xlabel('$x_0$')
        else:
            ax.set_xticks([])

    fig.text(0.65, 0.05, '(b) Estimated log probability', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
