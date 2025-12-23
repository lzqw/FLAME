import os
# os.environ["OMP_NUM_THREADS"] = "8"

import sys
from pathlib import Path
import argparse
import pickle
import csv
import yaml
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util
from matplotlib import pyplot as plt

from relax.env import create_env
from relax.network.rf import create_rf_net
from relax.network.mf import create_mf_net
from relax.network.sdac import create_sdac_net

def evaluate(env, policy_fn, policy_params, num_episodes, iter_key):
    ep_len_list = []
    ep_ret_list = []
    policy_params = tree_util.tree_map(lambda x: jnp.array(x), policy_params)
    for episode in range(num_episodes):
        obs, _ = env.reset()
        ep_len = 0
        ep_ret = 0.0
        ep_time = 0.0
        while True:
            iter_key, iter_key2 = random.split(iter_key)
            start_time = time.time()
            # act = policy_fn(iter_key2, policy_params, obs)[-1].clip(-1, 1)
            obs = jnp.array(obs)
            act = policy_fn(iter_key2, policy_params, obs).block_until_ready()
            end_time = time.time()
            # print(end_time - start_time)
            # print("------------------")
            ep_time += (end_time - start_time)

            act = act[-1].clip(-1, 1)
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_len += 1
            ep_ret += reward
            if terminated or truncated:
                break
            if ep_len % 100 == 0:
                vis_sample_path(obs, policy_fn, policy_params, iter_key2, episode, ep_len)
        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
        print("ep_time:", ep_time / ep_len)
    return ep_len_list, ep_ret_list

def vis_sample_path(obs, policy_fn, policy_params, iterkey, episode, ep_len, num_samples=5):
    save_dir = Path("figures/rf_sample_paths_5000")
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for _ in range(num_samples):
        iterkey, iterkey1 = random.split(iterkey)
        path = policy_fn(iterkey1, policy_params, obs)
        paths.append(path)
    paths = jnp.stack(paths, axis=0)
    spans = jnp.ptp(paths, axis=1)
    avg_spans = jnp.mean(spans, axis=0)
    top2 = jnp.argsort(avg_spans)[-2:][::-1]
    idx1, idx2 = int(top2[0]), int(top2[1])
    plt.figure(figsize=(6,6))
    for path in paths:
        x_vals = path[:, idx1]
        y_vals = path[:, idx2]
        line = plt.plot(x_vals, y_vals, '-o', alpha=0.35)
        color = line[0].get_color()
        plt.plot(x_vals[-1], y_vals[-1], '^', color=color, alpha=1.0, markersize=12)
        plt.xlabel(f"Action axis {idx1}", fontsize=18)
        plt.ylabel(f"Action axis {idx2}", fontsize=18)
        plt.title("Ant-v5", fontsize=20)
        plt.grid(True)
        plt.tight_layout()
    plt.savefig(save_dir / f"{episode}-{ep_len}.png")

class Logger(object):
	def __init__(self, log_dir, sample_steps):
		self.path = os.path.join(log_dir, f'log_sample_steps_{sample_steps}.csv')
		with open(self.path, mode='w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['step', 'avg_ret', 'std_ret', 'time'])

	def log(self, step, avg_ret, std_ret, time_step):
		with open(self.path, mode='a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([step, avg_ret, std_ret, time_step])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_root", type=Path, default="/home/lzqw/PycharmProject/DP_RL/DP_result/Hopper-v5/sdac_2025-12-22_23-12-02_s2_test_use_atp1" )
    parser.add_argument("--env", type=str, default="Hopper-v5")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    type="sdac"
    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)
    master_rng = np.random.default_rng(0)
    env_seed, env_action_seed, policy_seed, net_seed, iter_seed = map(int, master_rng.integers(0, 2**32 - 1, 5))
    net_key = random.PRNGKey(net_seed)
    iter_key = random.PRNGKey(iter_seed)
    env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)

    def mish(x: jax.Array):
        return x * jnp.tanh(jax.nn.softplus(x))

    policy_root = Path(args.policy_root)
    with open(policy_root / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        hidden_sizes = [cfg["hidden_dim"]] * cfg["hidden_num"]
        diffusion_hidden_sizes = [cfg["diffusion_hidden_dim"]] * cfg["hidden_num"]
        if type == "mf":
            mf_net, _ = create_mf_net(net_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, cfg["diffusion_steps"],
                               num_timesteps_test=args.sample_steps,
                               num_particles=cfg["num_particles"],
                               noise_scale=cfg["noise_scale"],
                               target_entropy_scale=cfg["target_entropy_scale"])
        elif type == "sdac":
            sdac_net, _ = create_sdac_net(net_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, cfg["diffusion_steps"],
                               num_particles=cfg["num_particles"],
                               noise_scale=cfg["noise_scale"],
                               target_entropy_scale=cfg["target_entropy_scale"])
        else:
            rf_net, _ = create_rf_net(net_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, cfg["diffusion_steps"],
                               num_timesteps_test=args.sample_steps,
                               num_particles=cfg["num_particles"],
                               noise_scale=cfg["noise_scale"],
                               target_entropy_scale=cfg["target_entropy_scale"])

    @jax.jit
    def policy_fn(key, policy_params, obs):
        if type == "mf":
            return mf_net.get_vanilla_action_traj(key, policy_params, obs)
        elif type == "sdac":
            return sdac_net.get_action_traj(key, policy_params, obs)
        else:
            return rf_net.get_vanilla_action_step(key, policy_params, obs)

    logger = Logger(policy_root, args.sample_steps)

    for step in range(8334, 2339444, 8334):
        policy_path = policy_root / f"policy-{3*step}-{step}.pkl"

        with open(policy_path, "rb") as f:
            policy_params = pickle.load(f)
        time_start = time.time()
        iter_key, iter_key1 = random.split(iter_key)
        ep_len_list, ep_ret_list = evaluate(env, policy_fn, policy_params, args.num_episodes, iter_key1)
        time_step = time.time() - time_start
        ep_len = np.array(ep_len_list)
        ep_ret = np.array(ep_ret_list)
        logger.log(2*step, ep_ret.mean(), ep_ret.std(), time_step)
