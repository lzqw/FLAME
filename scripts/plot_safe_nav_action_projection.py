#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from relax.safety.obstacle_navigation_filter import ObstacleNavigationFilterConfig


def feasible_raw_action_mask(position: np.ndarray, cfg: ObstacleNavigationFilterConfig, grid_size: int = 61):
    lin = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    ax, ay = np.meshgrid(lin, lin, indexing="xy")
    actions = np.stack([ax.reshape(-1), ay.reshape(-1)], axis=-1)

    next_pos = position[None, :] + cfg.dt * cfg.u_max * actions
    low = np.array([cfg.workspace_x[0] + cfg.eps_box, cfg.workspace_y[0] + cfg.eps_box])
    high = np.array([cfg.workspace_x[1] - cfg.eps_box, cfg.workspace_y[1] - cfg.eps_box])
    in_box = np.all((next_pos >= low) & (next_pos <= high), axis=-1)
    outside_obs = np.linalg.norm(next_pos - np.array(cfg.obstacle_center)[None, :], axis=-1) >= cfg.tightened_obstacle_radius
    return actions, in_box & outside_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-actions", type=Path, required=True, help=".npy with shape [N,2]")
    parser.add_argument("--position", type=float, nargs=2, required=True, help="position where samples were collected")
    parser.add_argument("--output", type=Path, default=Path("raw_action_projection_overlay.png"))
    args = parser.parse_args()

    raw_actions = np.load(args.raw_actions)
    pos = np.array(args.position, dtype=np.float32)
    cfg = ObstacleNavigationFilterConfig()

    grid_actions, feasible = feasible_raw_action_mask(pos, cfg)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(grid_actions[~feasible, 0], grid_actions[~feasible, 1], s=4, alpha=0.15, label="infeasible")
    ax.scatter(grid_actions[feasible, 0], grid_actions[feasible, 1], s=5, alpha=0.3, label="safe action set")
    ax.scatter(raw_actions[:, 0], raw_actions[:, 1], s=10, c="tab:orange", alpha=0.7, label="raw samples")
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_aspect("equal")
    ax.set_title(f"Raw Action Samples and Safe Set @ pos=({pos[0]:.2f}, {pos[1]:.2f})")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
