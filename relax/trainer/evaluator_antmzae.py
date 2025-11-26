import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import argparse
import pickle
import csv
import numpy as np
import jax
import matplotlib.pyplot as plt  # Added for plotting
from tensorboardX import SummaryWriter

# Import your custom package to trigger environment registration
import relax.env.antmaze
from relax.env import create_env
from relax.utils.persistence import PersistFunction


def evaluate(env, policy_fn, policy_params, num_episodes):
    """
    Evaluates the policy and collects trajectory data.
    Returns:
        ep_len_list: List of episode lengths.
        ep_ret_list: List of episode returns.
        all_trajectories: List of numpy arrays containing (x, y) positions.
    """
    ep_len_list = []
    ep_ret_list = []

    # List to store the trajectory of each episode: [traj1_array, traj2_array, ...]
    all_trajectories = []

    for _ in range(num_episodes):
        obs, info = env.reset()

        ep_len = 0
        ep_ret = 0.0

        # List to store (x, y) points for the current episode
        current_traj = []

        # Record initial position
        # Try getting it from info['achieved_goal'], otherwise fallback to physical data
        if 'achieved_goal' in info:
            current_traj.append(info['achieved_goal'][:2])
        else:
            try:
                # Fallback: Read directly from Mujoco data if info is empty
                current_traj.append(env.unwrapped.data.qpos[:2].copy())
            except:
                pass

        while True:
            act = policy_fn(policy_params, obs)
            obs, reward, terminated, truncated, info = env.step(act)

            # Record current position (x, y)
            if 'achieved_goal' in info:
                pos = info['achieved_goal'][:2]
            else:
                pos = env.unwrapped.data.qpos[:2].copy()
            current_traj.append(pos)

            ep_len += 1
            ep_ret += reward
            if terminated or truncated:
                break

        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
        all_trajectories.append(np.array(current_traj))

    return ep_len_list, ep_ret_list, all_trajectories


def save_heatmap(trajectories, step, save_dir):
    """
    Generates and saves a 2D heatmap of the agent's trajectories.
    """
    if len(trajectories) == 0:
        return

    # Concatenate all points from all trajectories into a single array
    all_points = np.concatenate(trajectories, axis=0)
    x = all_points[:, 0]
    y = all_points[:, 1]

    # Create figure
    plt.figure(figsize=(8, 8))

    # Plot 2D histogram (Heatmap)
    # bins: resolution of the grid
    # cmap: color map (e.g., 'inferno', 'viridis', 'hot')
    # cmin: minimum count to display (1 makes zero-visit areas transparent/white)
    plt.hist2d(x, y, bins=60, cmap='inferno', cmin=1)

    plt.colorbar(label='Visits')
    plt.title(f"Trajectory Heatmap - Step {step}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis('equal')  # Ensure aspect ratio is correct for the maze
    plt.grid(True, alpha=0.3, linestyle='--')

    # Save the figure
    save_path = save_dir / f"heatmap_step_{step}.png"
    plt.savefig(save_path, dpi=100)
    plt.close()  # Close figure to free memory


class Logger(object):
    def __init__(self, log_dir):
        self.path = os.path.join(log_dir, 'log.csv')
        # Check if file exists to avoid overwriting header during resume
        if not os.path.exists(self.path):
            with open(self.path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'avg_ret', 'std_ret'])

    def log(self, step, avg_ret, std_ret):
        with open(self.path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, avg_ret, std_ret])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_root", type=Path)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    # Create a directory for plots
    plot_dir = args.policy_root / "plots"
    plot_dir.mkdir(exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2 ** 32 - 1, 3))

    # Create the environment
    env, _, _ = create_env(args.env, env_seed, env_action_seed)

    policy = PersistFunction.load(args.policy_root / "deterministic.pkl")


    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs).clip(-1, 1)


    # logger = SummaryWriter(args.policy_root)
    logger = Logger(args.policy_root)

    # Main evaluation loop reading from stdin
    while payload := sys.stdin.readline():
        try:
            step_str, policy_path = payload.strip().split(",", maxsplit=1)
            step = int(step_str)

            with open(policy_path, "rb") as f:
                policy_params = pickle.load(f)

            # Evaluate and get trajectories
            ep_len_list, ep_ret_list, trajectories = evaluate(env, policy_fn, policy_params, args.num_episodes)

            ep_len = np.array(ep_len_list)
            ep_ret = np.array(ep_ret_list)

            # Log results
            logger.log(step, ep_ret.mean(), ep_ret.std())

            # Generate and save heatmap
            print(f"Step {step}: Generating heatmap with {len(trajectories)} trajectories...")
            save_heatmap(trajectories, step, plot_dir)

        except ValueError:
            print(f"Skipping invalid payload: {payload}")
            continue
