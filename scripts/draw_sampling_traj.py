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
    CONFIG_DIR = "/home/lzqw/PycharmProject/DP_RL/DP_result/HalfCheetah-v5/rf2_sac_ent_2025-11-20_08-20-44_s100_test_use_atp1"
    WEIGHT_FILE = CONFIG_DIR + "/policy-1500000-1500000.pkl"

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
    num_samples = 10  # 绘制 20 条彼此接近的轨迹
    num_steps = 20  # ODE 积分步数
    dt = 1.0 / num_steps

    # 固定状态
    obs_single, _ = env.reset()
    obs_batch = jnp.repeat(jnp.expand_dims(obs_single, 0), num_samples, axis=0)

    # 【关键修改】：生成彼此接近的初始 a0
    rng = jax.random.PRNGKey(42)
    key_center, key_noise = jax.random.split(rng)
    # key_center*=0

    # 1. 先随机选一个中心点
    center_a0 = jax.random.normal(key_center, (1, act_dim))
    # 2. 在中心点周围添加微小扰动 (scale 设为 0.1)
    a_t = center_a0 + jax.random.normal(key_noise, (num_samples, act_dim)) * 0.2
    # a_t = jnp.clip(a_t, -1, 1)
    # a_t = jax.random.normal(key_noise, (num_samples, act_dim)) *0.5
    trajectories = [a_t]

    print(f"正在模拟 ODE 积分。初始动作点彼此接近，观察演化过程...")
    for i in range(num_steps):
        t_current = i * dt
        v = agent.policy(params.policy, obs_batch, a_t, t_current)
        a_t = a_t + v * dt
        trajectories.append(a_t)

    trajectories = jnp.stack(trajectories)  # [steps+1, samples, act_dim]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    # 设置全局加粗
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    plt.figure(figsize=(9, 9))
    total_points = trajectories.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, total_points))

    for i in range(num_samples):
        # 注意：你代码中选择了索引 3 和 4
        tx = trajectories[:, i, 0]
        ty = trajectories[:, i, 1]

        # 绘制渐变点和连线
        for j in range(total_points - 1):
            plt.plot(tx[j:j + 2], ty[j:j + 2], color=colors[j], alpha=0.5, linewidth=1.2, zorder=1)
            plt.scatter(tx[j], ty[j], color=colors[j], s=10, zorder=2)

        # 初始位置 a0: 蓝色三角形
        plt.scatter(tx[0], ty[0], color='blue', marker='^', s=60, edgecolors='white', zorder=5)

        # 终点 a1: 红色星号
        plt.scatter(tx[-1], ty[-1], color='red', marker='*', s=120, edgecolors='black', zorder=6)

    # 刻度设置与加粗
    plt.xticks(np.linspace(-1, 1, 5), weight='bold')
    plt.yticks(np.linspace(-1, 1, 5), weight='bold')

    # 设置标题和标签，显式指定字体加粗
    # plt.title(f"RF Trajectory Evolution (Env: {args.env})\nSame Obs, Local Cluster $a_0$",
    #           fontsize=16, fontweight='bold')
    plt.xlabel("Action Dimension 3", fontsize=14, fontweight='bold')
    plt.ylabel("Action Dimension 4", fontsize=14, fontweight='bold')

    # 保持坐标轴比例一致
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.4)

    # 渲染 pdf 前确保 layout
    plt.tight_layout()

    save_path = Path(CONFIG_DIR) / "rf_local_cluster_fixed_axis.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"坐标轴固定后的局部簇采样轨迹图（Times New Roman 加粗版）已保存至: {save_path}")
    plt.show()
