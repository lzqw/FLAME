import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from relax.safety.obstacle_navigation_filter import ObstacleNavConfig, make_action_grid, project_action_jax_batched
from eval.eval_safe_obstacle_navigation import load_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--algo', required=True)
    p.add_argument('--num_states', type=int, default=32)
    p.add_argument('--samples_per_state', type=int, default=256)
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    env = SafeObstacleNavigation2DEnv(use_filter=True, seed=0)
    agent = load_agent(args.checkpoint, args.algo)
    cfg = ObstacleNavConfig()
    grid = jnp.asarray(make_action_grid(61))
    key = jax.random.PRNGKey(0)

    records = []
    for i in range(args.num_states):
        obs, _ = env.reset(seed=i)
        key, ak = jax.random.split(key)
        raw = np.asarray(agent.get_action(ak, obs[None, :])[0])
        raws = np.clip(raw[None, :] + 0.25 * np.random.randn(args.samples_per_state, 2), -1.0, 1.0)
        obs_rep = np.repeat(obs[None, :], args.samples_per_state, axis=0)
        exec_a, _, _ = project_action_jax_batched(obs_rep, raws, grid, cfg)
        d = np.linalg.norm(np.asarray(raws - exec_a), axis=1)
        records.append({'obs': obs.tolist(), 'raw_samples': raws.tolist(), 'residuals': d.tolist()})

    Path(args.save_path).write_text(json.dumps({'records': records}, indent=2))

if __name__ == '__main__':
    main()
