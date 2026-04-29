import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from scripts.train_safe_obstacle_navigation import make_algo, goal_controller


def classify_route(pos_traj):
    near_idx = np.where(np.abs(pos_traj[:, 0]) < 0.5)[0]
    y_mean = np.mean(pos_traj[near_idx, 1]) if len(near_idx) > 0 else np.mean(pos_traj[:, 1])
    return "upper" if y_mean > 0 else "lower"


def load_agent(checkpoint_path, algo):
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    args = argparse.Namespace(seed=ckpt['seed'], diffusion_steps=10, num_ent_timesteps=10, alpha_value=0.01,
                              fixed_alpha=False, init_alpha=0.01, gamma=0.99, gamma_p=0.99, lr=3e-4,
                              alpha_lr=1e-2, sample_k=64, lambda_p=1.0, use_projection_critic=True)
    agent = make_algo(args)
    agent.state = ckpt['agent_state']
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--algo', required=True)
    p.add_argument('--eval_episodes', type=int, default=200)
    p.add_argument('--save_dir', required=True)
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = SafeObstacleNavigation2DEnv(use_filter=args.algo != 'rf2_no_filter', seed=0)
    agent = None if args.algo == 'goal_filter' else load_agent(args.checkpoint, args.algo)
    key = jax.random.PRNGKey(123)

    T, N = env.episode_len, args.eval_episodes
    positions = np.zeros((N, T + 1, 2), np.float32)
    obs_all = np.zeros((N, T + 1, 8), np.float32)
    raw_actions = np.zeros((N, T, 2), np.float32)
    exec_actions = np.zeros((N, T, 2), np.float32)
    rewards = np.zeros((N, T), np.float32)
    state_violation = np.zeros((N, T), bool)
    tightened_violation = np.zeros((N, T), bool)
    safe_violation = np.zeros((N, T), bool)
    filter_active = np.zeros((N, T), bool)
    projection_residual = np.zeros((N, T), np.float32)
    projection_cost = np.zeros((N, T), np.float32)
    distance_to_goal = np.zeros((N, T), np.float32)
    distance_to_obstacle = np.zeros((N, T), np.float32)
    is_success = np.zeros((N,), bool)
    time_to_goal = np.full((N,), T, np.int32)
    episode_return = np.zeros((N,), np.float32)

    for i in range(N):
        obs, _ = env.reset(seed=i)
        positions[i, 0] = env.state
        obs_all[i, 0] = obs
        for t in range(T):
            if args.algo == 'goal_filter':
                raw = goal_controller(env.state, env.goal)
            else:
                key, ak = jax.random.split(key)
                raw = np.asarray(agent.get_action(ak, obs[None, :])[0])
            nobs, r, term, trunc, info = env.step(raw)
            raw_actions[i, t] = info['raw_action']
            exec_actions[i, t] = info['exec_action']
            rewards[i, t] = r
            state_violation[i, t] = info['state_violation']
            tightened_violation[i, t] = info['tightened_violation']
            safe_violation[i, t] = info['safe_violation']
            filter_active[i, t] = info['filter_active']
            projection_residual[i, t] = info['projection_residual']
            projection_cost[i, t] = info['projection_cost']
            distance_to_goal[i, t] = info['distance_to_goal']
            distance_to_obstacle[i, t] = info['distance_to_obstacle']
            episode_return[i] += r
            positions[i, t + 1] = env.state
            obs_all[i, t + 1] = nobs
            obs = nobs
            if term and not is_success[i]:
                is_success[i] = True
                time_to_goal[i] = t + 1
            if term or trunc:
                break

    np.savez(save_dir / 'rollouts.npz', positions=positions, obs=obs_all, raw_actions=raw_actions, exec_actions=exec_actions,
             rewards=rewards, state_violation=state_violation, tightened_violation=tightened_violation,
             safe_violation=safe_violation, filter_active=filter_active, projection_residual=projection_residual,
             projection_cost=projection_cost, distance_to_goal=distance_to_goal, distance_to_obstacle=distance_to_obstacle,
             is_success=is_success, time_to_goal=time_to_goal, episode_return=episode_return)

    routes = [classify_route(positions[i, :time_to_goal[i] + 1]) for i in range(N) if is_success[i]]
    upper = routes.count('upper')
    lower = routes.count('lower')
    ns = max(len(routes), 1)
    p_up, p_low = upper / ns, lower / ns
    route_entropy = -(p_up * np.log(p_up + 1e-8) + p_low * np.log(p_low + 1e-8))

    summary = {
        'success_rate': float(np.mean(is_success)),
        'collision_rate': float(np.mean(np.any(distance_to_obstacle < 0.0, axis=1))),
        'state_violation_rate': float(np.mean(state_violation)),
        'episode_return_mean': float(np.mean(episode_return)),
        'episode_return_std': float(np.std(episode_return)),
        'time_to_goal_mean': float(np.mean(time_to_goal)),
        'filter_activation_rate': float(np.mean(filter_active)),
        'avg_projection_residual': float(np.mean(projection_residual)),
        'feasible_raw_action_ratio': float(np.mean(1 - safe_violation.astype(np.float32))),
        'route_upper_ratio': float(p_up),
        'route_lower_ratio': float(p_low),
        'route_entropy': float(route_entropy),
    }
    (save_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    with open(save_dir / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)


if __name__ == '__main__':
    main()
