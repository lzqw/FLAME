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
        # 稍微调整均值距离，让Mode更明显
        self.means = np.array([[-1.5, -1.5], [-1.5, 1.5], [1.5, 1.5], [1.5, -1.5]])
        self.std = 0.4  # 稍微大一点的std让分布连接处更平滑
        self.cov = np.eye(2) * (self.std ** 2)
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])

    def sample(self, n):
        indices = np.random.choice(4, size=n, p=self.weights)
        samples = np.array([np.random.multivariate_normal(self.means[i], self.cov) for i in indices])
        # 将数据缩放到 [-2, 2] 之间以便网络学习
        return jnp.array(samples / 2.0)

    def log_prob(self, x, y):
        # 对应sample的缩放，计算log_prob时坐标也要还原
        pos = np.dstack((x * 2.0, y * 2.0))
        p = np.zeros(x.shape)
        for i, mean in enumerate(self.means):
            p += self.weights[i] * multivariate_normal.pdf(pos, mean=mean, cov=self.cov)
        # 加上缩放产生的行列式变化 log(2*2) = log(4)
        return np.log(p + 1e-10) + np.log(4.0)


# ==========================================
# 2. 网络定义 (保持不变)
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
        hk.Linear(128), jax.nn.silu,
        hk.Linear(128), jax.nn.silu,
        hk.Linear(128), jax.nn.silu,
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

        # Hutchinson's Estimator noise
        z_key, _ = jax.random.split(key)
        epsilon = jax.random.normal(z_key, act.shape)

        def ode_dynamics(state, t):
            f_t, _ = state
            u_t_fn = lambda x: model_fn(t, x)
            # 计算 VJP: epsilon^T * J
            u_val, vjp_fn = jax.vjp(u_t_fn, f_t)
            vjp_val = vjp_fn(epsilon)[0]
            # Trace approx = epsilon^T * J * epsilon
            trace_term = jnp.sum(vjp_val * epsilon, axis=-1)

            df_dt = u_val
            dg_dt = -trace_term  # d(log_p)/dt
            return df_dt, dg_dt

        # 从 t=1 (Data) 积分到 t=0 (Noise)
        dt = -1.0 / num_steps

        def solver_step(state, t):
            f_t, g_t = state
            df_dt, dg_dt = ode_dynamics(state, t)
            f_next = f_t + df_dt * dt
            g_next = g_t + dg_dt * dt
            return (f_next, g_next), None

        timesteps = jnp.linspace(1.0, 1.0 / num_steps, num_steps)  # t from 1.0 to >0
        initial_state = (act, jnp.zeros(act.shape[:-1]))

        final_state, _ = jax.lax.scan(solver_step, initial_state, timesteps)
        f_0, delta_logp = final_state

        # log p(x) = log p(z) + delta_logp
        log_p1 = log_p0(f_0) + delta_logp  # 注意这里符号，积分累积的是变化量
        return log_p1


# ==========================================
# 4. 主程序
# ==========================================
def main():
    WEIGHTS_FILE = "toy_gmm_params_v2.pkl"
    RUN_MODE = "auto"  # "train", "load", "auto"

    rng = jax.random.PRNGKey(2025)
    gmm = ToyGMM()

    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.zeros((1, 2))
    dummy_t = jnp.zeros((1, 1))
    network = hk.without_apply_rng(hk.transform(net_fn))
    params = network.init(init_rng, dummy_x, dummy_t)

    # --- [Training / Loading] ---
    do_train = False
    if RUN_MODE == "train":
        do_train = True
    elif RUN_MODE == "load":
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, 'rb') as f:
                params = pickle.load(f)
            print("Loaded weights.")
        else:
            raise FileNotFoundError
    else:  # auto
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, 'rb') as f:
                params = pickle.load(f)
            print("Auto-loaded weights.")
        else:
            do_train = True

    if do_train:
        # 训练更久一点以保证流场变直
        total_steps = 30000
        lr_schedule = optax.cosine_decay_schedule(1e-3, total_steps, alpha=1e-2)
        optimizer = optax.adam(learning_rate=lr_schedule)
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(params, x_1, key):
            x_0 = jax.random.normal(key, x_1.shape)
            t = jax.random.uniform(key, (x_1.shape[0], 1))
            # Conditional Flow Matching (Optimal Transport Path)
            x_t = t * x_1 + (1 - t) * x_0
            target_v = x_1 - x_0  # Vector field should act as x_1 - x_0
            pred_v = network.apply(params, x_t, t.flatten())
            return jnp.mean((pred_v - target_v) ** 2)

        @jax.jit
        def update(params, opt_state, batch, key):
            grads = jax.grad(loss_fn)(params, batch, key)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        print("Training...")
        for i in range(total_steps):
            rng, step_key = jax.random.split(rng)
            batch = gmm.sample(1024)
            params, opt_state = update(params, opt_state, batch, step_key)
            if i % 5000 == 0: print(f"Step {i}")

        with open(WEIGHTS_FILE, 'wb') as f:
            pickle.dump(params, f)
        print("Training done.")

    # --- [Plotting for Paper] ---
    print("Generating Experiment Plots...")
    estimator = Estimator(network.apply)

    # 网格设置
    grid_size = 100
    limit = 2.5  # 扩大一点范围看背景
    x = np.linspace(-limit, limit, grid_size)
    y = np.linspace(-limit, limit, grid_size)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points_jax = jnp.array(points)

    # 编译估计函数 (vmap over Hutchinson samples)
    def estimate_grid_impl(key, num_steps, num_trace_samples):
        keys = jax.random.split(key, num_trace_samples)
        # 对同一个点，进行多次 trace 估计取平均，减少噪声，让图好看
        logp_samples = jax.vmap(
            lambda k: estimator.compute_log_likelihood(k, params, points_jax, num_steps)
        )(keys)
        logp_avg = jnp.mean(logp_samples, axis=0)
        return logp_avg.reshape(grid_size, grid_size)

    # 只需要编译 num_steps 即可，trace samples 固定
    estimate_grid = jax.jit(estimate_grid_impl, static_argnums=(1, 2))

    # 设置论文需要的三个实验配置
    # (Label, Num Steps, Hutchinson Samples)
    experiments = [
        ("Ground Truth", None, None),
        ("FLAME-R (N=20)", 20, 20),  # 标准 ODE
        ("FLAME-M (N=1)", 1, 20),  # 单步 (Bias显著)
        ("FLAME-M (N=10)", 10, 20)  # 多步修正 (Bias消失)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # 计算 Ground Truth 范围，用于统一 Colorbar
    Z_true = gmm.log_prob(X, Y)
    vmin, vmax = -8, -1.0  # 手动截断一下让图更好看

    for i, (name, n_steps, n_samples) in enumerate(experiments):
        ax = axes[i]

        if i == 0:
            Z = Z_true
        else:
            print(f"Running {name}...")
            rng, key = jax.random.split(rng)
            Z = estimate_grid(key, n_steps, n_samples)

        # 绘图
        im = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    # 统一 Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Log Probability')

    plt.suptitle("Impact of Integration Steps on Entropy Estimation", fontsize=16, y=1.05)
    plt.savefig('toy_entropy_comparison_v2.pdf', bbox_inches='tight', dpi=300)
    print("Done! Saved to toy_entropy_comparison_v2.pdf")
    plt.show()


if __name__ == "__main__":
    main()
