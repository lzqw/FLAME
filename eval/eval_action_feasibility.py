import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from relax.safety.obstacle_navigation_filter import ObstacleNavConfig, make_action_grid, project_action_jax_batched
from eval.eval_safe_obstacle_navigation import load_agent


def build_obs(pos):
    goal = np.array([-2.5, 0.0], dtype=np.float32)
    vec = goal - pos
    d_goal = np.linalg.norm(vec)
    d_obs = np.linalg.norm(pos)
    return np.array([pos[0], pos[1], vec[0], vec[1], d_goal, d_obs, 0.0, 0.0], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--algo', required=True)
    p.add_argument('--grid_size', type=int, default=61)
    p.add_argument('--samples_per_state', type=int, default=512)
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    rep_states = np.asarray([[1.2, 0.0], [0.8, 0.75], [0.8, -0.75], [2.0, 1.2]], dtype=np.float32)
    cfg = ObstacleNavConfig()
    action_grid = jnp.asarray(make_action_grid(args.grid_size))
    agent = load_agent(args.checkpoint, args.algo)

    key = jax.random.PRNGKey(0)
    records = []
    for pos in rep_states:
        obs = build_obs(pos)
        obs_b = jnp.repeat(jnp.asarray(obs)[None, :], action_grid.shape[0], axis=0)
        grid_exec, safe_violation, _ = project_action_jax_batched(obs_b, action_grid, action_grid, cfg)
        safe_mask = np.asarray(1.0 - safe_violation)
        qp_grid = np.asarray(agent.agent.get_qp(agent.state.params.qp, obs_b, action_grid))

        key, sk = jax.random.split(key)
        sample_obs = jnp.repeat(jnp.asarray(obs)[None, :], args.samples_per_state, axis=0)
        raw_samples = np.asarray(agent.get_action(sk, np.asarray(sample_obs)))
        exec_samples, sample_violation, _ = project_action_jax_batched(sample_obs, raw_samples, action_grid, cfg)

        raw_samples_np = np.asarray(raw_samples)
        exec_samples_np = np.asarray(exec_samples)
        sample_violation_np = np.asarray(sample_violation)
        feasible_ratio = float(np.mean(1.0 - sample_violation_np))  # rho_feas
        apr = float(np.mean(np.linalg.norm(raw_samples_np - exec_samples_np, axis=-1)))  # action projection residual
        action_dispersion = float(np.mean(np.var(raw_samples_np, axis=0)))  # D_a
        safe_diversity = float(np.mean(np.var(exec_samples_np, axis=0)))  # D_safe

        bin_x = np.clip(((exec_samples_np[:, 0] + 1.0) * 0.5 * (args.grid_size - 1)).astype(int), 0, args.grid_size - 1)
        bin_y = np.clip(((exec_samples_np[:, 1] + 1.0) * 0.5 * (args.grid_size - 1)).astype(int), 0, args.grid_size - 1)
        hist = np.zeros((args.grid_size, args.grid_size), dtype=np.float32)
        np.add.at(hist, (bin_y, bin_x), 1.0)
        p_route = hist.ravel() / np.maximum(hist.sum(), 1.0)
        action_route_entropy = float(-(p_route * np.log(p_route + 1e-8)).sum())

        records.append({
            'state': pos.tolist(),
            'grid_actions': np.asarray(action_grid).tolist(),
            'safe_mask': safe_mask.tolist(),
            'qp_heatmap': qp_grid.tolist(),
            'grid_projected_actions': np.asarray(grid_exec).tolist(),
            'raw_samples': raw_samples_np.tolist(),
            'projected_samples': exec_samples_np.tolist(),
            'rho_feas': feasible_ratio,
            'APR': apr,
            'D_a': action_dispersion,
            'D_safe': safe_diversity,
            'H_act_route': action_route_entropy,
        })

    Path(args.save_path).write_text(json.dumps({'records': records}, indent=2))


if __name__ == '__main__':
    main()
