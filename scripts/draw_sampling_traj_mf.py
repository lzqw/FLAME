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
from relax.network.mf2_sac_ent2 import create_mf2_sac_ent2_net, Diffv2Params


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
    parser.add_argument("--alg", type=str, default="mf2_sac_ent2")
    parser.add_argument("--env", type=str, default="Ant-v5")
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=1)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.001)
    parser.add_argument("--target_entropy_scale", type=float, default=1.0)
    parser.add_argument("--fix_alpha", type=str2bool, default=True)
    parser.add_argument("--init_alpha", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.01)

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
    CONFIG_DIR = "/home/lzqw/PycharmProject/DP_RL/DP_result/Ant-v5/mf2_sac_ent2_2025-11-15_01-07-30_s100_test_use_atp1"
    WEIGHT_FILE = CONFIG_DIR + "/policy-2000000-1000000.pkl"

    args = get_args_from_yaml(CONFIG_DIR)

    # 初始化环境与网络
    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    init_key = jax.random.key(int(master_rng.integers(0, 2 ** 32 - 1)))
    env, obs_dim, act_dim = create_env(args.env, 0, 0)

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    # 创建 MF2 网络组件 (注意 MF2 采样过程)
    agent, _ = create_mf2_sac_ent2_net(
        init_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, jax.nn.relu,
        num_timesteps=args.diffusion_steps,
        num_particles=args.num_particles,
        noise_scale=args.noise_scale,
        target_entropy_scale=args.target_entropy_scale,
        alpha_value=args.alpha,
        fixed_alpha=args.fix_alpha,
        init_alpha=args.init_alpha
    )

    # 加载参数
    params = load_policy_parameters(WEIGHT_FILE)

    # --- 轨迹生成逻辑 ---
    num_samples = 10
    num_steps = 1
    dt = 1.0 / num_steps

    # 1. 修正观测值准备：确保 obs_batch 是 (num_samples, 27)
    obs, _ = env.reset()
    obs_batch = jnp.repeat(jnp.expand_dims(obs, 0), num_samples, axis=0)

    # 2. 初始噪声 a0
    rng = jax.random.PRNGKey(42)
    a_0 = jax.random.normal(rng, (num_samples, act_dim))

    trajectories = [a_0]

    print(f"正在生成 MF2 采样轨迹 (基于 u(x_0, 0, t))...")


    @jax.jit
    def get_mf_position(p, o, x0, t_val):
        batch_size = x0.shape[0]
        # 核心修正：时间参数必须是一维向量，与 obs 的 batch 维度对齐
        r_batch = jnp.zeros((batch_size,))
        t_batch = jnp.full((batch_size,), t_val)

        # 根据 MF2SACENT2Net 定义，policy 接受 (params, obs, x, r, t)
        displacement = agent.policy(p, o, x0, r_batch, t_batch)
        return x0 + displacement


    for i in range(1, num_steps + 1):
        curr_t = i * dt
        a_t = get_mf_position(params.policy, obs_batch, a_0, curr_t)
        trajectories.append(a_t)

    trajectories = jnp.stack(trajectories)

    # --- 绘图逻辑 (满足：无图例、起点三角形、彩色渐变点+连线) ---
    plt.figure(figsize=(8, 8))
    total_points = trajectories.shape[0]
    colors = plt.cm.plasma(np.linspace(0, 1, total_points))

    for i in range(num_samples):
        tx = trajectories[:, i, 0]
        ty = trajectories[:, i, 1]

        # 1. 连线与点（渐变）
        for j in range(total_points - 1):
            plt.plot(tx[j:j + 2], ty[j:j + 2], color=colors[j], alpha=0.6, linewidth=1.5, zorder=1)
            plt.scatter(tx[j], ty[j], color=colors[j], s=12, zorder=2)

        # 2. 初始位置 a0: 蓝色三角形
        plt.scatter(tx[0], ty[0], color='blue', marker='^', s=80, edgecolors='white', zorder=5)

        # 3. 终点 a1: 红色星号 (突出显示)
        plt.scatter(tx[-1], ty[-1], color='red', marker='*', s=120, edgecolors='black', zorder=6)

    plt.title(f"MF2: {args.env}", fontsize=15)
    plt.xlabel("Action Axis 0", fontsize=12)
    plt.ylabel("Action Axis 1", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.3)

    save_path = Path(CONFIG_DIR) / "mf2_sampling_traj_gradient_fixed.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"修正后的 MF2 渐变轨迹图已保存至: {save_path}")
    plt.show()
