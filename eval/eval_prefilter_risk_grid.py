import argparse
import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_safe_obstacle_navigation import make_algo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--save_dir', required=True)
    args = p.parse_args()

    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)

    fake_args = argparse.Namespace(seed=ckpt['seed'], diffusion_steps=10, num_ent_timesteps=10, alpha_value=0.01,
                                   fixed_alpha=False, init_alpha=0.01, gamma=0.99, gamma_p=0.99, lr=3e-4,
                                   alpha_lr=1e-2, sample_k=64, lambda_p=1.0, use_projection_critic=True)
    agent = make_algo(fake_args)
    agent.state = ckpt['agent_state']

    px_grid = np.linspace(-3.5, 3.5, 300)
    py_grid = np.linspace(-2.0, 2.0, 180)
    PX, PY = np.meshgrid(px_grid, py_grid)
    pos = np.stack([PX.reshape(-1), PY.reshape(-1)], axis=-1)
    goal = np.array([-2.6, 0.0], dtype=np.float32)
    obstacle_center = np.array([0.0, 0.0], dtype=np.float32)
    obstacle_radius = 0.8
    rel_goal = pos - goal[None, :]
    rel_obs = pos - obstacle_center[None, :]
    d_obs = np.linalg.norm(rel_obs, axis=-1, keepdims=True) - obstacle_radius
    d_goal = np.linalg.norm(rel_goal, axis=-1, keepdims=True)
    obs = np.concatenate([pos, rel_goal, rel_obs, d_obs, d_goal], axis=-1).astype(np.float32)

    vp_flat = np.asarray(agent.agent.get_vp(agent.state.params.vp, jnp.asarray(obs)))
    vp_grid = vp_flat.reshape(len(py_grid), len(px_grid))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_dir / 'prefilter_risk_grid.npz', px_grid=px_grid, py_grid=py_grid, Vp_grid=vp_grid)


if __name__ == '__main__':
    main()
