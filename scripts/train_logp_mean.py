import jax
import jax.numpy as jnp
import optax
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import NamedTuple, Tuple
import os
import pickle
import math

# Importing from your provided package structure
from relax.network.blocks import DACERPolicyNet2
from relax.utils.flow import MeanFlow


# ==========================================
# [配置] ICML 绘图风格设置
# ==========================================
def configure_icml_style():
    """
    配置 Matplotlib 以符合 ICML 论文风格
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.figsize": (6, 4.5),
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    })


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
        return jnp.array(samples, dtype=jnp.float32)

    def log_prob(self, x, y):
        pos = np.dstack((x, y))
        p = np.zeros(x.shape)
        for i, mean in enumerate(self.means):
            p += self.weights[i] * multivariate_normal.pdf(pos, mean=mean, cov=self.cov)
        return np.log(p + 1e-10)


# ==========================================
# 2. Train State Definition
# ==========================================
class MF2GMMTrainState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    step: int


# ==========================================
# 3. The Fitting Class (MF2 Logic)
# ==========================================
class MF2GMMFitter:
    def __init__(
        self,
        act_dim: int = 2,
        hidden_sizes: list = [128, 128, 128],
        lr: float = 1e-4,
        num_timesteps: int = 1,  # Training steps (usually 1 for continuous time flow matching logic, or discrete)
    ):
        self.act_dim = act_dim
        self.num_timesteps = num_timesteps
        self.flow = MeanFlow(num_timesteps=num_timesteps)

        # Defines the Flow Network (Policy)
        def network_fn(obs, act, r, t):
            # obs is dummy context
            return DACERPolicyNet2(
                hidden_sizes=hidden_sizes,
                activation=jax.nn.gelu,
                time_dim=16
            )(obs, act, r, t)

        self.network = hk.without_apply_rng(hk.transform(network_fn))
        self.optimizer = optax.adam(lr)

    def init_state(self, rng_key):
        dummy_obs = jnp.zeros((1, 1))
        dummy_x = jnp.zeros((1, self.act_dim))
        dummy_r = jnp.zeros((1,))
        dummy_t = jnp.zeros((1,))

        params = self.network.init(rng_key, dummy_obs, dummy_x, dummy_r, dummy_t)
        opt_state = self.optimizer.init(params)
        return MF2GMMTrainState(params, opt_state, step=0)

    @property
    def update_step(self):
        @jax.jit
        def _update(state: MF2GMMTrainState, batch_x: jnp.ndarray, key: jax.random.PRNGKey):
            params = state.params
            opt_state = state.opt_state
            batch_size = batch_x.shape[0]
            obs = jnp.zeros((batch_size, 1))

            key, flow_key, r_key, mask_key, t_key = jax.random.split(key, 5)

            # MF2 Sampling Logic
            r0 = jax.random.uniform(r_key, shape=(batch_size,), minval=0.0, maxval=1.0)
            mask = jax.random.bernoulli(mask_key, p=0.0, shape=(batch_size,))
            t0 = jax.random.uniform(t_key, shape=(batch_size,), minval=0.0, maxval=1.0)
            is_t_gt_r = t0 > r0
            t_swap = jnp.where(is_t_gt_r, t0, r0)
            r_swap = jnp.where(is_t_gt_r, r0, t0)
            r_final = jnp.where(mask, r0, r_swap)
            t_final = jnp.where(mask, r0, t_swap)

            def denoiser(x, r, t):
                return self.network.apply(params, obs, x, r, t)

            q_weights = jnp.ones((batch_size, 1))

            def loss_fn(p):
                def denoiser_p(x, r, t):
                    return self.network.apply(p, obs, x, r, t)

                loss, _ = self.flow.weighted_p_loss(
                    flow_key, q_weights, denoiser_p, r_final, t_final, x_start=batch_x
                )
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            new_state = MF2GMMTrainState(new_params, new_opt_state, state.step + 1)
            return new_state, loss

        return _update

    def save(self, state: MF2GMMTrainState, path: str):
        params_cpu = jax.device_get(state.params)
        with open(path, "wb") as f:
            pickle.dump(params_cpu, f)
        print(f"Weights saved to {path}")

    def load(self, path: str, rng_key):
        print(f"Loading weights from {path}...")
        with open(path, "rb") as f:
            params = pickle.load(f)
        params = jax.device_put(params)
        dummy_state = self.init_state(rng_key)
        return MF2GMMTrainState(params, dummy_state.opt_state, dummy_state.step)

    # --- 新增: 用于计算固定 Grid Points 的 Log Probability (Inverse Flow) ---
    def compute_log_likelihood(self, key: jax.Array, params: hk.Params, x: jax.Array, num_steps: int) -> jax.Array:
        """
        Calculates log p(x) by integrating from Data (t=0) to Noise (t=1).
        Using Hutchinson Trace Estimator.
        """
        batch_size = x.shape[0]
        obs = jnp.zeros((batch_size, 1))  # Dummy context

        # Base distribution: N(0, I)
        def log_p_prior(z):
            return jax.scipy.stats.norm.logpdf(z).sum(axis=-1)

        z_key, _ = jax.random.split(key)
        epsilon = jax.random.normal(z_key, x.shape)  # For Hutchinson Trace

        dt = 1.0 / num_steps

        # Integrate t from 0.0 to 1.0
        def scan_body(carry, step_idx):
            x_curr, trace_accum = carry

            # Current time segment [t_start, t_end]
            t_start = step_idx * dt
            t_end = (step_idx + 1) * dt

            def step_flow_map(x_in):
                # Predict displacement u(x, r, t).
                # According to MF2 logic in relax/algorithm/mf2.py (stateless_update),
                # the network learns x_{t} - x_{r}.
                # For inverse flow (0 -> 1), we want to predict direction towards 1.
                # We pass r=t_start, t=t_end.
                return self.network.apply(params, obs, x_in, t_start, t_end)

            drift, tangent = jax.jvp(step_flow_map, (x_curr,), (epsilon,))

            # Trace = epsilon^T * J * epsilon
            step_trace = jnp.sum(epsilon * tangent, axis=-1)

            # Update state: x_{t+1} = x_t + drift
            x_next = x_curr + drift

            # Accumulate trace (Log det Jacobian)
            trace_next = trace_accum + step_trace

            return (x_next, trace_next), None

        (x_final, total_trace), _ = jax.lax.scan(scan_body, (x, jnp.zeros((batch_size,))), jnp.arange(num_steps))

        # log p(x) = log p_prior(f(x)) + log |det J|
        # The trace accumulates log |det J| approximation.
        # Note: In standard Flow Matching v = dx/dt.
        # If we define forward as 1->0, then inverse is 0->1.
        # Here we moved 0->1.
        # log p(data) = log p(noise) + integral(trace).
        log_prob = log_p_prior(x_final) + total_trace

        return log_prob


# ==========================================
# 4. 绘图辅助函数
# ==========================================
def save_single_plot(X, Y, Z, filename, colorbar_mode=None, vmin=None, vmax=None):
    """
    保存单张等高线图，符合 ICML 风格
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.axis('off')  # 去掉坐标轴

    if colorbar_mode == 'left':
        # 在左侧放置 colorbar (通过 fig.add_axes 或 location='left' 取决于版本，这里使用标准右侧但调整参数或独立绘制)
        # 为简单起见，这里放在右侧，但参数调整使其紧凑
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

    print(f"Saving {filename}...")
    plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_entropy_accuracy(fitter, params, gmm, rng, grid_size=50):
    """
    计算 MSE vs ODE Steps 并绘图
    """
    print("\n--- Running plot_entropy_accuracy ---")

    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    points_np = np.stack([X.flatten(), Y.flatten()], axis=1)
    points = jnp.array(points_np)

    # 1. 计算 Ground Truth
    log_prob_true = jnp.array(gmm.log_prob(X, Y).flatten())

    steps_list = [1, 2, 4, 8, 16, 32, 50]
    mses = []

    # 随机多次采样取平均以减少 Trace Estimator 的方差
    N_samples = 5

    for steps in steps_list:
        print(f"Evaluating MSE for ODE steps = {steps}...")
        rng, key = jax.random.split(rng)

        @jax.jit
        def eval_step(k):
            keys = jax.random.split(k, N_samples)
            # 使用 vmap 对 N_samples 个随机 key 进行并行计算
            # 每次 compute_log_likelihood 内部会生成一个新的 epsilon
            logp = jax.vmap(
                lambda k_: fitter.compute_log_likelihood(k_, params, points, steps)
            )(keys)
            return jnp.mean(logp, axis=0)

        log_prob_est = eval_step(key)
        mse = jnp.mean((log_prob_est - log_prob_true) ** 2)
        mses.append(mse)

    # --- 绘图 (ICML Style) ---
    plt.figure()

    plt.plot(steps_list, mses, 'o-', color='#1f77b4', linewidth=1.5, markersize=5, label='Log-Prob MSE')
    plt.xlabel('ODE Solver Steps (NFE)')
    plt.ylabel('MSE (Est. vs True Log-Prob)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=False)

    # 去掉上方和右侧边框
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
    # 配置
    SEED = 42
    BATCH_SIZE = 256
    ITERATIONS = 5000
    CHECKPOINT_PATH = "mf2_gmm_weights.pkl"

    rng = jax.random.PRNGKey(SEED)
    gmm = ToyGMM()
    fitter = MF2GMMFitter(act_dim=2, hidden_sizes=[128, 128, 128], lr=1e-4)

    rng, init_key = jax.random.split(rng)

    # --- 检查权重与训练 ---
    if os.path.exists(CHECKPOINT_PATH):
        state = fitter.load(CHECKPOINT_PATH, init_key)
        print("Skipping training phase.")
    else:
        print("No checkpoint found. Starting training...")
        state = fitter.init_state(init_key)
        print(f"Starting training for {ITERATIONS} iterations...")
        for i in range(ITERATIONS):
            batch_x = gmm.sample(BATCH_SIZE)
            rng, step_key = jax.random.split(rng)
            state, loss = fitter.update_step(state, batch_x, step_key)
            if i % 1000 == 0:
                print(f"Iter {i}, Loss: {loss:.4f}")
        print("Training Complete.")
        fitter.save(state, CHECKPOINT_PATH)

    params = state.params

    # --- 绘图逻辑 ---
    print("\nGenerating Separate PDF Figures...")

    # 准备网格
    grid_size = 100
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points_jax = jnp.array(points)

    # 1. 绘制 Ground Truth
    Z_true = gmm.log_prob(X, Y)
    vmin, vmax = Z_true.min(), Z_true.max()
    save_single_plot(X, Y, Z_true, "plot_ground_truth.pdf", colorbar_mode='left', vmin=vmin, vmax=vmax)

    # 2. 绘制不同 Step 的估计值
    def estimate_grid_impl(key, T, N_avg=1):
        keys = jax.random.split(key, N_avg)
        logp_samples = jax.vmap(
            lambda k: fitter.compute_log_likelihood(k, params, points_jax, T)
        )(keys)
        logp_avg = jnp.mean(logp_samples, axis=0)
        return logp_avg.reshape(grid_size, grid_size)

    estimate_grid = jax.jit(estimate_grid_impl, static_argnums=(1, 2))

    rng, key = jax.random.split(rng)
    print("Computing T=10...")
    Z_est_10 = estimate_grid(key, 1, N_avg=50)
    save_single_plot(X, Y, Z_est_10, "plot_N50_T10.pdf", vmin=vmin, vmax=vmax)

    rng, key = jax.random.split(rng)
    print("Computing T=15...")
    Z_est_15 = estimate_grid(key, 15, N_avg=50)
    save_single_plot(X, Y, Z_est_15, "plot_N50_T15.pdf", vmin=vmin, vmax=vmax)

    rng, key = jax.random.split(rng)
    print("Computing T=20...")
    Z_est_20 = estimate_grid(key, 20, N_avg=50)
    save_single_plot(X, Y, Z_est_20, "plot_N50_T20.pdf", vmin=vmin, vmax=vmax)

    # 3. 绘制 Accuracy 曲线
    # 注意：这里使用较小的 range (-1, 1) 来计算 MSE，集中在数据分布核心区域
    rng, key = jax.random.split(rng)
    plot_entropy_accuracy(fitter, params, gmm, key, grid_size=50)

    print("All tasks finished.")


if __name__ == "__main__":
    main()
