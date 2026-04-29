from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp


@dataclass
class ObstacleNavConfig:
    dt: float = 0.1
    u_max: float = 1.0
    x_min: float = -3.5
    x_max: float = 3.5
    y_min: float = -2.0
    y_max: float = 2.0
    eps_box: float = 0.05
    obstacle_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float32))
    obstacle_radius: float = 0.8
    eps_obs: float = 0.08


def make_action_grid(grid_size=61):
    vals = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    gx, gy = np.meshgrid(vals, vals)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1).astype(np.float32)


def is_state_safe_original_np(pos, cfg: ObstacleNavConfig):
    x, y = float(pos[0]), float(pos[1])
    in_box = cfg.x_min <= x <= cfg.x_max and cfg.y_min <= y <= cfg.y_max
    outside_obs = np.linalg.norm(pos - cfg.obstacle_center) >= cfg.obstacle_radius
    return bool(in_box and outside_obs)


def is_state_safe_tight_np(pos, cfg: ObstacleNavConfig):
    x, y = float(pos[0]), float(pos[1])
    in_box = (cfg.x_min + cfg.eps_box <= x <= cfg.x_max - cfg.eps_box and
              cfg.y_min + cfg.eps_box <= y <= cfg.y_max - cfg.eps_box)
    outside_obs = np.linalg.norm(pos - cfg.obstacle_center) >= (cfg.obstacle_radius + cfg.eps_obs)
    return bool(in_box and outside_obs)


def is_action_feasible_np(pos, action_norm, cfg: ObstacleNavConfig):
    action_norm = np.clip(np.asarray(action_norm, dtype=np.float32), -1.0, 1.0)
    next_pos = pos + cfg.dt * cfg.u_max * action_norm
    return is_state_safe_tight_np(next_pos, cfg)


def project_action_np(pos, raw_action_norm, action_grid_norm, cfg: ObstacleNavConfig):
    raw_action_norm = np.clip(np.asarray(raw_action_norm, dtype=np.float32), -1.0, 1.0)
    if is_action_feasible_np(pos, raw_action_norm, cfg):
        return raw_action_norm, False, 0.0, False, False

    feasible_actions = []
    distances = []
    for a in action_grid_norm:
        if is_action_feasible_np(pos, a, cfg):
            feasible_actions.append(a)
            distances.append(np.sum((a - raw_action_norm) ** 2))

    if len(feasible_actions) == 0:
        direction = pos - cfg.obstacle_center
        norm = np.linalg.norm(direction)
        backup = np.array([1.0, 0.0], dtype=np.float32) if norm < 1e-6 else direction / norm
        backup = np.clip(backup, -1.0, 1.0).astype(np.float32)
        gap = float(np.linalg.norm(raw_action_norm - backup))
        return backup, True, gap, True, True

    idx = int(np.argmin(distances))
    exec_action = np.asarray(feasible_actions[idx], dtype=np.float32)
    gap = float(np.linalg.norm(raw_action_norm - exec_action))
    return exec_action, gap > 1e-6, gap, True, False


def is_state_safe_tight_jax(pos, cfg: ObstacleNavConfig):
    x = pos[..., 0]
    y = pos[..., 1]
    in_box = (
        (x >= cfg.x_min + cfg.eps_box) & (x <= cfg.x_max - cfg.eps_box) &
        (y >= cfg.y_min + cfg.eps_box) & (y <= cfg.y_max - cfg.eps_box)
    )
    obs_center = jnp.asarray(cfg.obstacle_center)
    dist = jnp.linalg.norm(pos - obs_center, axis=-1)
    outside_obs = dist >= (cfg.obstacle_radius + cfg.eps_obs)
    return in_box & outside_obs


def project_action_jax_flat(obs, raw_action, action_grid, cfg: ObstacleNavConfig):
    pos = obs[:, 0:2]
    raw_next = pos + cfg.dt * cfg.u_max * raw_action
    raw_feasible = is_state_safe_tight_jax(raw_next, cfg)
    next_pos_grid = pos[:, None, :] + cfg.dt * cfg.u_max * action_grid[None, :, :]
    feasible_mask = is_state_safe_tight_jax(next_pos_grid, cfg)
    dist = jnp.sum((action_grid[None, :, :] - raw_action[:, None, :]) ** 2, axis=-1)
    dist = jnp.where(feasible_mask, dist, 1e9)
    idx = jnp.argmin(dist, axis=1)
    grid_exec = action_grid[idx]
    safe_set_empty = jnp.logical_not(jnp.any(feasible_mask, axis=1))
    direction = pos - jnp.asarray(cfg.obstacle_center)
    direction = direction / (jnp.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6)
    backup = jnp.clip(direction, -1.0, 1.0)
    exec_action = jnp.where(raw_feasible[:, None], raw_action, grid_exec)
    exec_action = jnp.where(safe_set_empty[:, None], backup, exec_action)
    residual = jnp.linalg.norm(raw_action - exec_action, axis=-1)
    safe_violation = jnp.logical_not(raw_feasible).astype(jnp.float32)
    return exec_action, safe_violation, residual


def project_action_jax_batched(obs, raw_action, action_grid, cfg: ObstacleNavConfig):
    original_shape = raw_action.shape[:-1]
    obs_flat = obs.reshape((-1, obs.shape[-1]))
    raw_flat = raw_action.reshape((-1, raw_action.shape[-1]))
    exec_flat, safe_flat, residual_flat = project_action_jax_flat(obs_flat, raw_flat, action_grid, cfg)
    exec_action = exec_flat.reshape(original_shape + (raw_action.shape[-1],))
    safe_violation = safe_flat.reshape(original_shape)
    residual = residual_flat.reshape(original_shape)
    return exec_action, safe_violation, residual
