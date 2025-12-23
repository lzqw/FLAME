import os
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import optax
import math

# ==========================================
# [环境设置]
# ==========================================
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


# ==========================================
# 1. 真实分布 (Toy GMM)
# ==========================================
class ToyGMM:
    def __init__(self):
        # 参数设置：让Mode分得稍微开一点，以增加流场的非线性
        self.means = np.array([[-1.5, -1.5], [-1.5, 1.5], [1.5, 1.5], [1.5, -1.5]])
        self.std = 0.4
        self.cov = np.eye(2) * (self.std ** 2)
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])

    def sample(self, n):
        indices = np.random.choice(4, size=n, p=self.weights)
        samples = np.array([np.random.multivariate_normal(self.means[i], self.cov) for i in indices])
        return jnp.array(samples / 2.0)  # 缩放至 [-1, 1] 附近

    def log_prob(self, x, y):
        pos = np.dstack((x * 2.0, y * 2.0))
        p = np.zeros(x.shape)
        for i, mean in enumerate(self.means):
            p += self.weights[i] * multivariate_normal.pdf(pos, mean=mean, cov=self.cov)
        return np.log(p + 1e-10) + np.log(4.0)  # Jacobian修正


# ==========================================
# 2. 网络定义 (RF 和 MF 共用同一种结构)
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
        if self.dim % 2 == 1: emb = jnp.pad(emb, [[0, 0], [0, 1]])
        return emb


def net_fn(x, t):
    # 输入处理
    if t.ndim == 0: t = t * jnp.ones((x.shape[0], 1))
    if t.ndim == 1: t = t.reshape(-1, 1)

    t_emb = TimestepEmbedding(32)(t)
    x_embed = hk.Linear(64)(x)
    t_embed_split = hk.Linear(64)(t_emb)
    h = jax.nn.silu(x_embed + t_embed_split)

    # 稍微加深网络以拟合更复杂的流场
    mlp = hk.Sequential([
        hk.Linear(128), jax.nn.silu,
        hk.Linear(128), jax.nn.silu,
        hk.Linear(128), jax.nn.silu,
        hk.Linear(2)
    ])
    return mlp(h)


# ==========================================
# 3. 不同的 Loss 实现 (RF vs MF)
# ==========================================

def get_loss_fns(network):
    # --- A. Rectified Flow Loss (Standard) ---
    def loss_rf(params, x_1, key):
        """
        L_RF = || u(x_t, t) - (x_1 - x_0) ||^2
        """
        batch_size = x_1.shape[0]
        x_0 = jax.random.normal(key, x_1.shape)
        t = jax.random.uniform(key, (batch_size, 1))

        # Interpolation (t=0: noise, t=1: data)
        x_t = t * x_1 + (1 - t) * x_0
        target_v = x_1 - x_0  # Straight path

        pred_v = network.apply(params, x_t, t.flatten())
        return jnp.mean((pred_v - target_v) ** 2)

    # --- B. MeanFlow Loss (From meanflow.py) ---
    def loss_mf(params, x_1, key):
        """
        参考 meanflow.py:
        u_tgt = v - (t - r) * dudt
        L_MF = || u - sg(u_tgt) ||^2
        """
        batch_size = x_1.shape[0]
        key, k1, k2 = jax.random.split(key, 3)
        x_0 = jax.random.normal(k1, x_1.shape)

        # 采样 t 和 r (t > r)
        t_raw = jax.random.uniform(k2, (batch_size, 2))
        t = jnp.max(t_raw, axis=1, keepdims=True)
        r = jnp.min(t_raw, axis=1, keepdims=True)

        # 构造输入 x_t
        x_t = t * x_1 + (1 - t) * x_0

        # 基础向量 v (noise -> data)
        # 注意：meanflow.py 中 v = e - x (noise - data)，因为它的 t=0 是 data
        # 这里我们保持 train_logp.py 的习惯 t=1 是 data。
        # 所以 v = x_1 - x_0 (data - noise)，流向一致即可
        v = x_1 - x_0

        # 定义辅助函数用于求 JVP
        def forward_fn(x, time):
            return network.apply(params, x, time.flatten())

        # 计算 dudt (Total Derivative w.r.t time)
        # primal: u(x_t, t)
        # tangents: (dx/dt, dt/dt) = (v, 1)
        # jvp 返回: (u_val, du_dt_total)
        # 注意 t 的 tangent 应该是 全1向量
        ones = jnp.ones_like(t)
        u_val, dudt = jax.jvp(forward_fn, (x_t, t), (v, ones))

        # 构造 Target (修正项)
        # u_tgt = v - (t - r) * dudt
        # sg: stop_gradient
        dt = t - r
        u_tgt = v - dt * dudt
        u_tgt = jax.lax.stop_gradient(u_tgt)

        # Loss
        return jnp.mean((u_val - u_tgt) ** 2)

    return loss_rf, loss_mf


# ==========================================
# 4. 熵估计器 (Hutchinson Trace)
# ==========================================
class Estimator:
    def __init__(self, policy_apply):
        self.policy = policy_apply

    def compute_log_likelihood(self, key, params, act, num_steps):
        def model_fn(t, x):
            return self.policy(params, x, t)

        def log_p0(z):
            return jax.scipy.stats.norm.logpdf(z).sum(axis=-1)

        z_key, _ = jax.random.split(key)
        epsilon = jax.random.normal(z_key, act.shape)  # Noise for trace estimation

        def ode_dynamics(state, t):
            f_t, _ = state
            u_t_fn = lambda x: model_fn(t, x)
            # JVP for trace: epsilon^T * J
            u_val, vjp_fn = jax.vjp(u_t_fn, f_t)
            vjp_val = vjp_fn(epsilon)[0]
            trace_approx = jnp.sum(vjp_val * epsilon, axis=-1)
            return u_val, -trace_approx  # d(logp)/dt = -Tr(J)

        # 积分: t=1 (Data) -> t=0 (Noise)
        # dt 为负
        dt = -1.0 / num_steps
        timesteps = jnp.linspace(1.0, 1.0 / num_steps, num_steps)

        def solver_step(state, t):
            df, dg = ode_dynamics(state, t)
            # Euler step
            return (state[0] + df * dt, state[1] + dg * dt), None

        init_state = (act, jnp.zeros(act.shape[:-1]))
        final_state, _ = jax.lax.scan(solver_step, init_state, timesteps)

        # log p(x) = log p(z) + delta_logp
        return log_p0(final_state[0]) + final_state[1]


# ==========================================
# 5. 主程序
# ==========================================
def main():
    rng = jax.random.PRNGKey(2025)
    gmm = ToyGMM()

    # Init Network
    network = hk.without_apply_rng(hk.transform(net_fn))
    params_init = network.init(rng, jnp.zeros((1, 2)), jnp.zeros((1, 1)))

    loss_rf, loss_mf = get_loss_fns(network)

    # 训练函数
    def train_model(loss_fn, name):
        print(f"Training {name} Model...")
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params_init)
        params = params_init

        @jax.jit
        def step(p, opt, batch, k):
            grads = jax.grad(loss_fn)(p, batch, k)
            updates, new_opt = optimizer.update(grads, opt)
            return optax.apply_updates(p, updates), new_opt

        # 训练循环 (15000步足够Toy Task收敛)
        key = jax.random.PRNGKey(0)
        for i in range(15001):
            key, k_batch, k_step = jax.random.split(key, 3)
            batch = gmm.sample(512)
            params, opt_state = step(params, opt_state, batch, k_step)
            if i % 5000 == 0:
                print(f"  Step {i}")
        return params

    # 1. 训练 RF 模型 (用于 FLAME-R)
    params_rf = train_model(loss_rf, "RF (Rectified Flow)")

    # 2. 训练 MF 模型 (用于 FLAME-M)
    params_mf = train_model(loss_mf, "MF (MeanFlow)")

    # --- 绘图 ---
    print("Generating Plots...")
    estimator = Estimator(network.apply)

    # 准备网格
    grid_size = 100
    L = 2.5
    x = np.linspace(-L, L, grid_size)
    X, Y = np.meshgrid(x, x)
    pts = jnp.array(np.stack([X.ravel(), Y.ravel()], 1))

    # 预编译推理函数
    @jax.jit
    def calc_logp(key, p, n_steps, n_trace):
        # 对每个点做 n_trace 次 Hutchinson 采样取平均，让图更平滑
        keys = jax.random.split(key, n_trace)
        # vmap over keys
        vals = jax.vmap(lambda k: estimator.compute_log_likelihood(k, p, pts, n_steps))(keys)
        return jnp.mean(vals, axis=0).reshape(grid_size, grid_size)

    # 实验配置
    # (Title, Params, Num Steps)
    # n_trace 固定为 20
    experiments = [
        ("Ground Truth", None, None),
        ("FLAME-R (N=20)", params_rf, 20),
        ("FLAME-M (Naive N=1)", params_mf, 1),  # 关键：单步估计，Bias显著
        ("FLAME-M (Ours N=10)", params_mf, 10)  # 关键：多步估计，修复Bias
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    vmin, vmax = -8, -1.0  # 固定色阶以便对比

    Z_true = gmm.log_prob(X, Y)

    for i, (name, p, n_steps) in enumerate(experiments):
        ax = axes[i]
        if i == 0:
            Z = Z_true
        else:
            rng, k = jax.random.split(rng)
            Z = calc_logp(k, p, n_steps, 20)

        im = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Log Probability')

    plt.suptitle("Impact of Integration Steps on Entropy Estimation", fontsize=16, y=1.05)
    plt.savefig('toy_entropy_comparison_final.pdf', bbox_inches='tight')
    plt.show()
    print("Done. Saved to toy_entropy_comparison_final.pdf")


if __name__ == "__main__":
    main()
