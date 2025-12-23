import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path
from functools import partial

# 导入项目中定义的组件
from relax.env import create_env
from relax.utils.random_utils import seeding
from relax.network.rf2_sac_ent import create_rf2_sac_ent_net, Diffv2Params


def load_policy_parameters(file_path):
    """读取 policy-*.pkl 文件并转换为 Diffv2Params 格式"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {file_path}")
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    return Diffv2Params(
        policy=loaded_data[0],
        log_alpha=loaded_data[1],
        q1=loaded_data[2],
        q2=loaded_data[3],
        target_poicy=loaded_data[0],
        target_q1=loaded_data[2],
        target_q2=loaded_data[3]
    )


def get_args_from_yaml(config_path):
    """读取 config.yaml 并转换为 Namespace 对象"""

    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', 'y', '1') if isinstance(v, str) else v

    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="rf2_sac_ent")
    parser.add_argument("--env", type=str, default="Ant-v5")
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=1)
    parser.add_argument("--target_entropy_scale", type=float, default=1.0)
    parser.add_argument("--fix_alpha", type=str2bool, default=False)
    parser.add_argument("--init_alpha", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)

    args = parser.parse_args([])
    config_file = Path(config_path) / "config.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            if hasattr(args, key): setattr(args, key, value)
    return args


if __name__ == "__main__":
    # 路径配置
    CONFIG_DIR = "/home/lzqw/PycharmProject/DP_RL/DP_result/Ant-v5/rf2_sac_ent_2025-11-21_15-40-12_s100_test_use_atp1"
    WEIGHT_FILE = CONFIG_DIR + "/policy-1125000-1125000.pkl"

    args = get_args_from_yaml(CONFIG_DIR)

    # 1. 初始化
    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    init_key = jax.random.key(int(master_rng.integers(0, 2 ** 32 - 1)))
    env, obs_dim, act_dim = create_env(args.env, 0, 0)

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num


    def mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))


    agent, _ = create_rf2_sac_ent_net(
        init_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
        num_timesteps=args.diffusion_steps, num_particles=args.num_particles,
        noise_scale=args.noise_scale, target_entropy_scale=args.target_entropy_scale,
        alpha_value=args.alpha, fixed_alpha=args.fix_alpha, init_alpha=args.init_alpha
    )
    params = load_policy_parameters(WEIGHT_FILE)

    # --- 轨迹生成逻辑 ---
    # ... (前面的初始化代码保持不变)

    # --- 轨迹生成逻辑 ---

    # 【手动设置区】：在这里定义你想要的初始噪声坐标
    # 每一行代表一个轨迹的起始点，每一列对应 action 的一个维度
    manual_points = [
        [0.2, -1.23],  # 轨迹 1 的 (dim0, dim1)
        [0.1, -0.6],  # 轨迹 2
        [0.6, -0.7],  # 轨迹 3
        [0.5, -0.4],  # 轨迹 4
        [0.4, -0.2],  # 轨迹 5
        # [-0.3,-0.1],  # 轨迹 6
    ]

    # 将 2 维坐标填充到环境所需的 act_dim (Ant-v5 为 8)
    a0_base = jnp.zeros((len(manual_points), act_dim))
    # 将手动设置的 [x, y] 填入 act 的前两个维度 (或者你想要的维度)
    for i, p in enumerate(manual_points):
        a0_base = a0_base.at[i, 0].set(p[0])
        a0_base = a0_base.at[i, 1].set(p[1])

    num_samples = a0_base.shape[0]
    num_steps = 20
    dt = 1.0 / num_steps

    # 固定状态
    obs_single, _ = env.reset()
    obs_batch = jnp.repeat(jnp.expand_dims(obs_single, 0), num_samples, axis=0)

    a_t = a0_base
    trajectories = [a_t]

    print(f"正在模拟 ODE 积分。初始动作点已手动设置...")
    for i in range(num_steps):
        t_current = i * dt
        v = agent.policy(params.policy, obs_batch, a_t, t_current)
        a_t = a_t + v * dt
        trajectories.append(a_t)

    trajectories = jnp.stack(trajectories)

    # --- 绘图逻辑 (Times New Roman & Bold) ---
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    plt.figure(figsize=(9, 9))
    total_points = trajectories.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, total_points))

    for i in range(num_samples):
        # 绘制你设置的维度（例如 0 和 1）
        tx = trajectories[:, i, 0]
        ty = trajectories[:, i, 1]

        for j in range(total_points - 1):
            plt.plot(tx[j:j + 2], ty[j:j + 2], color=colors[j], alpha=0.5, linewidth=1.2, zorder=1)
            plt.scatter(tx[j], ty[j], color=colors[j], s=10, zorder=2)

        plt.scatter(tx[0], ty[0], color='blue', marker='^', s=60, edgecolors='white', zorder=5)
        plt.scatter(tx[-1], ty[-1], color='red', marker='*', s=120, edgecolors='black', zorder=6)

    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.xticks(np.linspace(-1, 1, 5), weight='bold')
    plt.yticks(np.linspace(-1, 1, 5), weight='bold')
    plt.xlabel("Action Dimension 0", fontsize=14, fontweight='bold')
    plt.ylabel("Action Dimension 1", fontsize=14, fontweight='bold')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()

    save_path = Path(CONFIG_DIR) / "rf_manual_points.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
