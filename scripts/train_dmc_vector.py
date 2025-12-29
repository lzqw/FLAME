import argparse
import os.path
import time
from functools import partial
import yaml
import numpy as np

import jax
import gymnasium as gym

from relax.algorithm.sac import SAC
from relax.network.sac import create_sac_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.buffer import TreeBuffer
from scripts.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.env import create_env  # 直接使用库里的 create_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sac")
    # 修改默认值为注册过的向量环境 ID
    parser.add_argument("--env", type=str, default="dm_control_vector_walker_walk-v0")
    parser.add_argument("--suffix", type=str, default="vector_test")
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(1e4))
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6))
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        from jax import config

        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    # 生成所需的各种随机种子
    env_seed, env_action_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2 ** 32 - 1, 5)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)

    print(f"Loading environment: {args.env}")

    # 使用 create_env 创建，它会自动调用 register_dm_control_envs
    # 并且返回的 env 会包含正确的 spec.id
    env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)

    print(f"Observation Dim: {obs_dim}, Action Dim: {act_dim}")

    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.replay_buffer_size, seed=buffer_seed)
    gelu = partial(jax.nn.gelu, approximate=False)

    if args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)
    else:
        raise ValueError(f"Algorithm {args.alg} not supported for vector DMC yet.")

    exp_dir = PROJECT_ROOT / "logs" / args.env / (
        f"{args.alg}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_s{args.seed}_{args.suffix}")

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 10),
        warmup_with="random",
        log_path=exp_dir,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))

    os.makedirs(exp_dir, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)

    print("Starting training...")
    trainer.run(train_key)
