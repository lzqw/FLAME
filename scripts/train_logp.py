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
    """复现图中描述的四个高斯分量"""

    def __init__(self):
        self.means = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
        self.std = 0.1
        self.cov = np.eye(2) * (self.std ** 2)
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])

    def sample(self, n):
        """用于训练的数据采样"""
        indices = np.random.choice(4, size=n, p=self.weights)
        samples = np.array([np.random.multivariate_normal(self.means[i], self.cov) for i in indices])
        return jnp.array(samples)

    def log_prob(self, x, y):
        """用于绘制 Ground Truth (a)"""
        pos = np.dstack((x, y))
        p = np.zeros(x.shape)
        for i, mean in enumerate(self.means):
            p += self.weights[i] * multivariate_normal.pdf(pos, mean=mean, cov=self.cov)
        return np.log(p + 1e-10)


# ==========================================
# 2. 改进后的网络定义 (Time Embedding + Wide MLP)
# ==========================================
class TimestepEmbedding(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, t):
        # t: [batch, 1]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if self.dim % 2 == 1:
            emb = jnp.pad(emb, [[0, 0], [0, 1]])
        return emb


def net_fn(x, t):
    """
    改进点 1: 引入正弦时间编码 + 宽 MLP (256)
    """
    # 确保 t 的形状正确
    if t.ndim == 0: t = t * jnp.ones((x.shape[0], 1))
    if t.ndim == 1: t = t.reshape(-1, 1)

    # 1. 时间编码 (64维)
    t_embed_dim = 64
    t_emb = TimestepEmbedding(t_embed_dim)(t)

    # 2. 特征融合
    x_embed = hk.Linear(256)(x)
    t_embed = hk.Linear(256)(t_emb)

    # 使用 Swish (SiLU) 激活函数，通常在 Flow Matching 中表现更好
    h = jax.nn.silu(x_embed + t_embed)

    # 3. 主干网络 (宽 MLP)
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.silu,
        hk.Linear(256), jax.nn.silu,
        hk.Linear(256), jax.nn.silu,
        hk.Linear(2)  # 输出维度
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
        """
        基于 rf2_sac_ent_net.py 逻辑的对数似然估算
        """

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
    # 初始化
    rng = jax.random.PRNGKey(42)
    gmm = ToyGMM()

    # 初始化网络
    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.zeros((1, 2))
    dummy_t = jnp.zeros((1, 1))
    network = hk.without_apply_rng(hk.transform(net_fn))
    params = network.init(init_rng, dummy_x, dummy_t)

    # 改进点 2: 学习率衰减 (Cosine Decay Schedule)
    # 增加总步数到 20000 以获得精细收敛
    total_steps = 20000
    lr_schedule = optax.cosine_decay_schedule(
        init_value=1e-3,
        decay_steps=total_steps,
        alpha=1e-2  # 结束时 lr = 1e-5
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    # Loss 函数: Flow Matching
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

    # ---------------------------
    # 训练循环
    # ---------------------------
    print(f"Training toy model (Flow Matching) for {total_steps} steps...")

    # 使用 jax.lax.scan 或者简单的 python loop (这里用 python loop 方便打印进度)
    for i in range(total_steps):
        rng, step_key = jax.random.split(rng)
        batch = gmm.sample(512)  # batch size 512
        params, opt_state = update(params, opt_state, batch, step_key)

        if (i + 1) % 2000 == 0:
            print(f"Step {i + 1}/{total_steps}")

    # ---------------------------
    # 绘图逻辑
    # ---------------------------
    print("Generating Figure 11...")

    estimator = Estimator(network.apply)

    # 网格数据
    grid_size = 50
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points_jax = jnp.array(points)

    # 改进点 4: 修复 JIT 报错，显式包裹
    def estimate_grid_impl(key, T, N):
        """内部实现函数"""
        keys = jax.random.split(key, N)
        # vmap over keys (Sample Number N)
        logp_samples = jax.vmap(
            lambda k: estimator.compute_log_likelihood(k, params, points_jax, T)
        )(keys)
        # Average over N samples
        logp_avg = jnp.mean(logp_samples, axis=0)
        return logp_avg.reshape(grid_size, grid_size)

    # 显式 JIT 编译，指定 T(arg 1) 和 N(arg 2) 为静态参数
    estimate_grid = jax.jit(estimate_grid_impl, static_argnums=(1, 2))

    # 准备绘图
    fig = plt.figure(figsize=(14, 6), dpi=100)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 0.2, 1, 1])

    # (a) True Log Probability
    ax_true = fig.add_subplot(gs[:, 0])
    Z_true = gmm.log_prob(X, Y)

    # 改进点 3: 统一色阶 (Vmin/Vmax)
    # 获取真实分布的范围，强制应用于所有子图
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
        # 使用统一色阶 vmin/vmax
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
