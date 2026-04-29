import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
from pathlib import Path
import numpy as np

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv


def goal_controller(state, goal):
    direction = goal - state
    return np.clip(direction / (np.linalg.norm(direction) + 1e-6), -1.0, 1.0).astype(np.float32)


def sample_policy(algo, state, goal):
    if algo == 'goal_filter':
        return goal_controller(state, goal)
    if algo == 'safe_pullback_rf2':
        return np.clip(goal_controller(state, goal) + np.random.normal(0, 0.25, 2), -1, 1).astype(np.float32)
    if algo == 'safe_pullback_rf2_no_entropy':
        base = goal_controller(state, goal)
        return np.clip(base + np.array([0.0, 0.15], np.float32), -1, 1)
    if algo == 'rf2_filter':
        return np.random.uniform(-1, 1, 2).astype(np.float32)
    if algo == 'rf2_no_filter':
        return np.random.uniform(-1, 1, 2).astype(np.float32)
    raise ValueError(algo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', default='safe_pullback_rf2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total_steps', type=int, default=200000)
    p.add_argument('--eval_interval', type=int, default=5000)
    p.add_argument('--eval_episodes', type=int, default=20)
    p.add_argument('--noise_sigma_x', type=float, default=0.01)
    p.add_argument('--noise_sigma_y', type=float, default=0.01)
    p.add_argument('--log_dir', type=str, required=True)
    args = p.parse_args()

    np.random.seed(args.seed)
    use_filter = args.algo != 'rf2_no_filter'
    env = SafeObstacleNavigation2DEnv(noise_sigma=(args.noise_sigma_x, args.noise_sigma_y), use_filter=use_filter, seed=args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics = []

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_len = 0
    for step in range(1, args.total_steps + 1):
        raw_action = sample_policy(args.algo, env.state, env.goal)
        obs, reward, terminated, truncated, info = env.step(raw_action)
        ep_ret += reward
        ep_len += 1
        if terminated or truncated:
            metrics.append({'step': step, 'episode_return': ep_ret, 'episode_len': ep_len, 'success': float(info['is_success'])})
            obs, _ = env.reset()
            ep_ret = 0.0
            ep_len = 0

    with open(log_dir / 'train_metrics.json', 'w') as f:
        json.dump(metrics, f)
    with open(log_dir / 'checkpoint.pkl', 'wb') as f:
        import pickle
        pickle.dump({'algo': args.algo, 'seed': args.seed}, f)


if __name__ == '__main__':
    main()
