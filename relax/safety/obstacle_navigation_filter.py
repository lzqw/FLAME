from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ObstacleNavigationFilterConfig:
    dt: float = 0.1
    u_max: float = 1.0
    grid_size: int = 61
    workspace_x: Tuple[float, float] = (-3.5, 3.5)
    workspace_y: Tuple[float, float] = (-2.0, 2.0)
    obstacle_center: Tuple[float, float] = (0.0, 0.0)
    tightened_obstacle_radius: float = 0.88
    eps_box: float = 0.05


@lru_cache(maxsize=16)
def _action_grid_numpy(grid_size: int) -> np.ndarray:
    lin = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    ax, ay = np.meshgrid(lin, lin, indexing="xy")
    return np.stack([ax.reshape(-1), ay.reshape(-1)], axis=-1)


@lru_cache(maxsize=16)
def _action_grid_jax(grid_size: int) -> jnp.ndarray:
    lin = jnp.linspace(-1.0, 1.0, grid_size, dtype=jnp.float32)
    ax, ay = jnp.meshgrid(lin, lin, indexing="xy")
    return jnp.stack([ax.reshape(-1), ay.reshape(-1)], axis=-1)


def _feasible_mask_numpy(position: np.ndarray, actions: np.ndarray, cfg: ObstacleNavigationFilterConfig) -> np.ndarray:
    u = cfg.u_max * actions
    next_pos = position[None, :] + cfg.dt * u

    low = np.array([cfg.workspace_x[0] + cfg.eps_box, cfg.workspace_y[0] + cfg.eps_box], dtype=np.float32)
    high = np.array([cfg.workspace_x[1] - cfg.eps_box, cfg.workspace_y[1] - cfg.eps_box], dtype=np.float32)
    inside_box = np.all((next_pos >= low) & (next_pos <= high), axis=-1)

    center = np.array(cfg.obstacle_center, dtype=np.float32)
    outside_obs = np.linalg.norm(next_pos - center[None, :], axis=-1) >= cfg.tightened_obstacle_radius
    return inside_box & outside_obs


def _feasible_mask_jax(position: jnp.ndarray, actions: jnp.ndarray, cfg: ObstacleNavigationFilterConfig) -> jnp.ndarray:
    u = cfg.u_max * actions
    next_pos = position[None, :] + cfg.dt * u

    low = jnp.array([cfg.workspace_x[0] + cfg.eps_box, cfg.workspace_y[0] + cfg.eps_box], dtype=jnp.float32)
    high = jnp.array([cfg.workspace_x[1] - cfg.eps_box, cfg.workspace_y[1] - cfg.eps_box], dtype=jnp.float32)
    inside_box = jnp.all((next_pos >= low) & (next_pos <= high), axis=-1)

    center = jnp.array(cfg.obstacle_center, dtype=jnp.float32)
    outside_obs = jnp.linalg.norm(next_pos - center[None, :], axis=-1) >= cfg.tightened_obstacle_radius
    return jnp.logical_and(inside_box, outside_obs)


def _radial_backup_numpy(position: np.ndarray, cfg: ObstacleNavigationFilterConfig) -> np.ndarray:
    center = np.array(cfg.obstacle_center, dtype=np.float32)
    d = position - center
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = d / norm
    return np.clip(direction, -1.0, 1.0)


def project_action_numpy(
    position: np.ndarray,
    raw_action: np.ndarray,
    cfg: ObstacleNavigationFilterConfig = ObstacleNavigationFilterConfig(),
) -> Dict[str, np.ndarray | float | bool]:
    position = np.asarray(position, dtype=np.float32).reshape(2)
    raw_action = np.clip(np.asarray(raw_action, dtype=np.float32).reshape(2), -1.0, 1.0)

    grid = _action_grid_numpy(cfg.grid_size)
    feasible_mask = _feasible_mask_numpy(position, grid, cfg)

    raw_feasible = bool(_feasible_mask_numpy(position, raw_action[None, :], cfg)[0])
    safe_set_empty = not bool(np.any(feasible_mask))

    if raw_feasible:
        exec_action = raw_action
        filter_active = False
    elif safe_set_empty:
        exec_action = _radial_backup_numpy(position, cfg)
        filter_active = True
    else:
        feasible_actions = grid[feasible_mask]
        distances = np.linalg.norm(feasible_actions - raw_action[None, :], axis=-1)
        exec_action = feasible_actions[int(np.argmin(distances))]
        filter_active = True

    projection_gap = float(np.linalg.norm(exec_action - raw_action))
    safe_violation = (not raw_feasible)

    return {
        "exec_action": exec_action.astype(np.float32),
        "filter_active": filter_active,
        "projection_gap": projection_gap,
        "safe_violation": safe_violation,
        "safe_set_empty": safe_set_empty,
    }


def project_action_jax(
    position: jnp.ndarray,
    raw_action: jnp.ndarray,
    cfg: ObstacleNavigationFilterConfig = ObstacleNavigationFilterConfig(),
) -> Dict[str, jnp.ndarray]:
    position = jnp.asarray(position, dtype=jnp.float32).reshape(2)
    raw_action = jnp.clip(jnp.asarray(raw_action, dtype=jnp.float32).reshape(2), -1.0, 1.0)

    grid = _action_grid_jax(cfg.grid_size)
    feasible_mask = _feasible_mask_jax(position, grid, cfg)

    raw_feasible = _feasible_mask_jax(position, raw_action[None, :], cfg)[0]
    safe_set_empty = jnp.logical_not(jnp.any(feasible_mask))

    center = jnp.array(cfg.obstacle_center, dtype=jnp.float32)
    d = position - center
    norm = jnp.linalg.norm(d)
    backup = jnp.where(norm < 1e-8, jnp.array([1.0, 0.0], dtype=jnp.float32), d / (norm + 1e-8))
    backup = jnp.clip(backup, -1.0, 1.0)

    distances = jnp.linalg.norm(grid - raw_action[None, :], axis=-1)
    masked_distances = jnp.where(feasible_mask, distances, jnp.inf)
    nearest_action = grid[jnp.argmin(masked_distances)]

    exec_action = jnp.where(raw_feasible, raw_action, jnp.where(safe_set_empty, backup, nearest_action))
    filter_active = jnp.logical_not(raw_feasible)
    projection_gap = jnp.linalg.norm(exec_action - raw_action)

    return {
        "exec_action": exec_action,
        "filter_active": filter_active,
        "projection_gap": projection_gap,
        "safe_violation": jnp.logical_not(raw_feasible),
        "safe_set_empty": safe_set_empty,
    }
