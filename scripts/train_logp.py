import os
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
import optax
import math
from scipy.stats import multivariate_normal


# ==========================================
# [配置] ICML 绘图风格设置
# ==========================================
def configure_icml_style():
    """
    配置 Matplotlib 以符合 ICML 论文风格:
    1. 字体: Times New Roman
    2. 字号: 适配两栏排版 (Main text ~10pt)
    3. 样式: 去掉顶部和右侧边框
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",  # 数学公式字体风格接近 Times
        "font.size": 10,  # 全局基础字号
        "axes.labelsize": 12,  # 坐标轴标签字号
        "axes.titlesize": 12,  # 标题字号
        "xtick.labelsize": 10,  # X轴刻度字号
        "ytick.labelsize": 10,  # Y轴刻度字号
        "legend.fontsize": 10,  # 图例字号
        "figure.titlesize": 14,  # 整个画布标题
        # ICML 单栏宽度约为 3.25 英寸，跨栏约为 6.75 英寸
        # 这里设置为适合单栏插入的比例
        "figure.figsize": (6, 4.5),
        "axes.linewidth": 1.0,  # 边框粗细
        "lines.linewidth": 1.5,  # 线条粗细
        "grid.alpha": 0.3,  # 网格透明度
    })


# 应用配置
configure_icml_style()


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
# 2. 网络定义 (Lite版, 64宽)
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
# 4. 新增功能: Entropy Accuracy Plotter
# ==========================================
def plot_entropy_accuracy(estimator, params, gmm, rng, grid_size=50):
    """
    计算并绘制 Estimated Log-Prob 与 Ground Truth 之间的 MSE 随 ODE Steps 变化的曲线。
    (已适配 ICML 字体风格)
    """
    print("\n--- Running plot_entropy_accuracy ---")

    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    points_np = np.stack([X.flatten(), Y.flatten()], axis=1)
    points = jnp.array(points_np)

    log_prob_true = jnp.array(gmm.log_prob(X, Y).flatten())

    steps_list = [1, 2, 4, 8, 16, 32, 50]
    mses = []

    N_samples = 10

    for steps in steps_list:
        print(f"Evaluating MSE for ODE steps = {steps}...")
        rng, key = jax.random.split(rng)

        @jax.jit
        def eval_step(k):
            keys = jax.random.split(k, N_samples)
            logp = jax.vmap(
                lambda k_: estimator.compute_log_likelihood(k_, params, points, steps)
            )(keys)
            return jnp.mean(logp, axis=0)

        log_prob_est = eval_step(key)
        mse = jnp.mean((log_prob_est - log_prob_true) ** 2)
        mses.append(mse)

    # --- 绘图 (ICML Style) ---
    plt.figure()  # 使用全局设置的 figsize

    # 绘制曲线，使用较深的蓝色，并在数据点上加标记
    plt.plot(steps_list, mses, 'o-', color='#1f77b4', linewidth=1.5, markersize=5, label='Log-Prob MSE')

    # 移除之前的 hardcoded fontsize，使用全局rcParams
    plt.xlabel('ODE Solver Steps (NFE)')
    plt.ylabel('MSE (Est. vs True Log-Prob)')

    # 学术图表通常推荐使用半透明网格
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=False)  # ICML风格通常不喜欢图例有边框

    # 移除上方和右侧的边框 (Despine)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filename = "plot_entropy_accuracy.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to {filename}")


# ==========================================
# 5. 主程序
# ==========================================
def main():
    # --- [配置区域] ---
    WEIGHTS_FILE = "toy_gmm_params.pkl"
    # 模式: "auto", "train", "load"
    RUN_MODE = "auto"

    # --- [初始化] ---
    rng = jax.random.PRNGKey(42)
    gmm = ToyGMM()

    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.zeros((1, 2))
    dummy_t = jnp.zeros((1, 1))
    network = hk.without_apply_rng(hk.transform(net_fn))
    params = network.init(init_rng, dummy_x, dummy_t)

    # --- [加载/训练逻辑] ---
    do_train = False
    if RUN_MODE == "train":
        print(f"[Mode: TRAIN] Force retraining...")
        do_train = True
    elif RUN_MODE == "load":
        print(f"[Mode: LOAD] Loading weights from {WEIGHTS_FILE}...")
        if not os.path.exists(WEIGHTS_FILE):
            raise FileNotFoundError(f"Weight file '{WEIGHTS_FILE}' not found!")
        with open(WEIGHTS_FILE, 'rb') as f:
            params = pickle.load(f)
        print("Weights loaded.")
    else:  # auto
        if os.path.exists(WEIGHTS_FILE):
            print(f"[Mode: AUTO] Found {WEIGHTS_FILE}, loading...")
            try:
                with open(WEIGHTS_FILE, 'rb') as f:
                    params = pickle.load(f)
                print("Weights loaded successfully.")
            except Exception as e:
                print(f"Error loading weights ({e}), switching to training.")
                do_train = True
        else:
            print(f"[Mode: AUTO] No weights found, starting training...")
            do_train = True

    if do_train:
        total_steps = 20000
        lr_schedule = optax.cosine_decay_schedule(1e-3, total_steps, alpha=1e-2)
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

        print(f"Start training for {total_steps} steps...")
        for i in range(total_steps):
            rng, step_key = jax.random.split(rng)
            batch = gmm.sample(512)
            params, opt_state = update(params, opt_state, batch, step_key)
            if (i + 1) % 2000 == 0:
                print(f"Step {i + 1}/{total_steps}")

        print(f"Saving weights to {WEIGHTS_FILE}...")
        with open(WEIGHTS_FILE, 'wb') as f:
            pickle.dump(params, f)
        print("Training done & saved.")

    # --- [绘图逻辑] ---
    print("Generating Separate PDF Figures...")
    estimator = Estimator(network.apply)

    grid_size = 100
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

    estimate_grid = jax.jit(estimate_grid_impl, static_argnums=(1, 2))

    Z_true = gmm.log_prob(X, Y)
    vmin, vmax = Z_true.min(), Z_true.max()

    def save_single_plot(Z, filename, colorbar_mode=None):
        # 调整 figsize 为正方形，但保持字体比例
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.axis('off')
        if colorbar_mode == 'left':
            cbar = fig.colorbar(im, ax=ax, location='left', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)  # 单独设置 colorbar 字体

        print(f"Saving {filename}...")
        plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()

    save_single_plot(Z_true, "plot_ground_truth.pdf", colorbar_mode='left')

    rng, key = jax.random.split(rng)
    print("Computing N=50, T=10...")
    Z_est_10 = estimate_grid(key, 10, 50)
    save_single_plot(Z_est_10, "plot_N50_T10.pdf", colorbar_mode=None)

    rng, key = jax.random.split(rng)
    print("Computing N=50, T=20...")
    Z_est_20 = estimate_grid(key, 15, 50)
    save_single_plot(Z_est_20, "plot_N50_T15.pdf", colorbar_mode=None)

    rng, key = jax.random.split(rng)
    print("Computing N=50, T=20...")
    Z_est_20 = estimate_grid(key, 20, 50)
    save_single_plot(Z_est_20, "plot_N50_T20.pdf", colorbar_mode=None)

    # 运行准确率曲线绘图
    rng, key = jax.random.split(rng)
    plot_entropy_accuracy(estimator, params, gmm, key)

    print("All tasks finished.")


if __name__ == "__main__":
    main()
