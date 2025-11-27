import os

# Force CPU usage to avoid JAX memory conflicts (sufficient for simple evaluation)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import argparse
import pickle
import csv
import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Import custom environment to trigger registration ---
import relax.env.antmaze
from relax.env import create_env
from relax.utils.persistence import PersistFunction


def evaluate(env, policy_fn, policy_params, num_episodes):
    """
    Run evaluation and collect trajectory coordinates (x, y).
    """
    ep_len_list = []
    ep_ret_list = []
    all_trajectories = []  # Store trajectories for all episodes

    print(f"Starting evaluation of {num_episodes} episodes...")

    for i in range(num_episodes):
        obs, info = env.reset()
        ep_len = 0
        ep_ret = 0.0
        current_traj = []

        # --- 1. Record initial position ---
        # Try getting it from info['achieved_goal'], otherwise fallback to physical data
        if 'achieved_goal' in info:
            current_traj.append(info['achieved_goal'][:2])
        else:
            try:
                current_traj.append(env.unwrapped.data.qpos[:2].copy())
            except:
                pass

        # --- 2. Run episode ---
        while True:
            act = policy_fn(policy_params, obs)
            obs, reward, terminated, truncated, info = env.step(act)

            # Record current position
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
        print(f"Episode {i + 1}/{num_episodes}: Return={ep_ret:.2f}, Length={ep_len}")

    return ep_len_list, ep_ret_list, all_trajectories


def save_heatmap_with_walls(env, trajectories, save_path):
    """
    Plot trajectory heatmap with wall overlay.
    """
    if len(trajectories) == 0:
        return

    all_points = np.concatenate(trajectories, axis=0)
    x = all_points[:, 0]
    y = all_points[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Draw Heatmap (Bottom layer)
    # bins=150 ensures high resolution; zorder=0 puts it at the bottom
    plt.hist2d(x, y, bins=150, cmap='inferno', cmin=1, zorder=0)
    plt.colorbar(label='Visits')

    # 2. Draw Wall Mask (Top layer)
    try:
        maze_map = env.unwrapped.maze_map
        scale = getattr(env.unwrapped, 'maze_size_scaling', 4.0)
        height = len(maze_map)
        width = len(maze_map[0])

        for i in range(height):
            for j in range(width):
                if maze_map[i][j] == 1:
                    # AntMaze coordinate mapping: i -> x, j -> y
                    x_corner = i * scale - scale / 2
                    y_corner = j * scale - scale / 2

                    # Draw a gray rectangle to represent the wall
                    rect = patches.Rectangle(
                        (x_corner, y_corner), scale, scale,
                        linewidth=0, edgecolor='none', facecolor='#808080',
                        zorder=10  # Draw ON TOP of the heatmap
                    )
                    ax.add_patch(rect)

        # Set axis limits to fit the maze
        ax.set_xlim(-scale, height * scale)
        ax.set_ylim(-scale, width * scale)

    except AttributeError:
        print("[Warning] Could not retrieve maze_map from env. Plotting heatmap only.")

    plt.title("Evaluation Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True, alpha=0.3, linestyle='--', zorder=5)

    print(f"Saving heatmap to: {save_path}")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()  # Show in PyCharm SciView
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Required Arguments ---
    parser.add_argument("--env", type=str, default="AntMaze_CenterBlock-v0", help="Environment ID")
    # policy_root must contain 'deterministic.pkl' (Network structure definition)
    parser.add_argument("--policy_root", type=Path,  help="Folder containing deterministic.pkl",
                        default="/home/lzqw/PycharmProject/DP_RL/DP_result/AntMaze_CenterBlock-v0/rf2_sac_ent_2025-11-27_00-24-56_s100_test_use_atp1",
                               )
    # checkpoint_path is the specific weight file (e.g., params_10000.pkl)
    parser.add_argument("--checkpoint_path", type=Path, help="Path to the specific .pkl parameter file",
                        default="/home/lzqw/PycharmProject/DP_RL/DP_result/AntMaze_CenterBlock-v0/rf2_sac_ent_2025-11-27_00-24-56_s100_test_use_atp1/policy-112500-37500.pkl",)
    # --- Optional Arguments ---
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of trajectories to evaluate")
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()

    # 1. Set seed and create environment
    print(f"Creating environment: {args.env}")
    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2 ** 32 - 1, 3))
    env, _, _ = create_env(args.env, env_seed, env_action_seed)

    # 2. Load policy structure (deterministic.pkl)
    print("Loading policy structure...")
    policy_structure_path = args.policy_root / "deterministic.pkl"
    if not policy_structure_path.exists():
        raise FileNotFoundError(f"Policy structure file not found: {policy_structure_path}")

    policy = PersistFunction.load(policy_structure_path)


    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs).clip(-1, 1)


    # 3. Load specific parameter weights
    print(f"Loading model weights: {args.checkpoint_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    with open(args.checkpoint_path, "rb") as f:
        policy_params = pickle.load(f)

    # 4. Execute evaluation
    ep_len_list, ep_ret_list, trajectories = evaluate(env, policy_fn, policy_params, args.num_episodes)

    # 5. Output results
    ep_len = np.array(ep_len_list)
    ep_ret = np.array(ep_ret_list)
    print("-" * 30)
    print(f"Average Return: {ep_ret.mean():.4f} ± {ep_ret.std():.4f}")
    print(f"Average Length: {ep_len.mean():.2f}")
    print("-" * 30)

    # 6. Plot and save heatmap
    # Save the plot in the same directory as the checkpoint
    plot_save_path = args.checkpoint_path.parent / f"eval_heatmap_{args.checkpoint_path.stem}.png"
    # save_heatmap_with_walls(env, trajectories, plot_save_path)
