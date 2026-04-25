from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DoubleIntegratorSafetyConfig:
    dt: float = 0.1
    u_min: float = -2.0
    u_max: float = 2.0
    p_min: float = -5.0
    p_max: float = 5.0
    v_min: float = -3.0
    v_max: float = 3.0
    eps_p: float = 0.25
    eps_v: float = 0.15


def safe_interval_np(
    p: float,
    v: float,
    cfg: DoubleIntegratorSafetyConfig = DoubleIntegratorSafetyConfig(),
) -> Tuple[float, float, bool]:
    dt = cfg.dt
    uv_low = (cfg.v_min + cfg.eps_v - v) / dt
    uv_high = (cfg.v_max - cfg.eps_v - v) / dt
    up_low = 2.0 * (cfg.p_min + cfg.eps_p - p - dt * v) / (dt ** 2)
    up_high = 2.0 * (cfg.p_max - cfg.eps_p - p - dt * v) / (dt ** 2)

    safe_low = max(cfg.u_min, uv_low, up_low)
    safe_high = min(cfg.u_max, uv_high, up_high)
    safe_set_empty = safe_low > safe_high
    return float(safe_low), float(safe_high), bool(safe_set_empty)


def project_action_np(
    raw_u: float,
    p: float,
    v: float,
    cfg: DoubleIntegratorSafetyConfig = DoubleIntegratorSafetyConfig(),
) -> Tuple[float, float, float, bool]:
    safe_low, safe_high, safe_set_empty = safe_interval_np(p, v, cfg)
    if safe_set_empty:
        center = 0.5 * (safe_low + safe_high)
        exec_u = np.clip(center, cfg.u_min, cfg.u_max)
    else:
        exec_u = np.clip(raw_u, safe_low, safe_high)
    return float(exec_u), float(safe_low), float(safe_high), bool(safe_set_empty)


def safe_interval_jax(
    obs,
    *,
    dt: float = 0.1,
    u_min: float = -2.0,
    u_max: float = 2.0,
    p_min: float = -5.0,
    p_max: float = 5.0,
    v_min: float = -3.0,
    v_max: float = 3.0,
    eps_p: float = 0.25,
    eps_v: float = 0.15,
):
    p = obs[..., 0]
    v = obs[..., 1]

    uv_low = (v_min + eps_v - v) / dt
    uv_high = (v_max - eps_v - v) / dt
    up_low = 2.0 * (p_min + eps_p - p - dt * v) / (dt ** 2)
    up_high = 2.0 * (p_max - eps_p - p - dt * v) / (dt ** 2)

    safe_low = jnp.maximum(jnp.maximum(uv_low, up_low), u_min)
    safe_high = jnp.minimum(jnp.minimum(uv_high, up_high), u_max)
    safe_set_empty = safe_low > safe_high
    return safe_low, safe_high, safe_set_empty


def project_action_jax(
    obs,
    raw_action_norm,
    *,
    u_scale: float = 2.0,
    dt: float = 0.1,
    u_min: float = -2.0,
    u_max: float = 2.0,
    p_min: float = -5.0,
    p_max: float = 5.0,
    v_min: float = -3.0,
    v_max: float = 3.0,
    eps_p: float = 0.25,
    eps_v: float = 0.15,
):
    raw_u = raw_action_norm[..., 0] * u_scale
    safe_low, safe_high, safe_set_empty = safe_interval_jax(
        obs,
        dt=dt,
        u_min=u_min,
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        v_min=v_min,
        v_max=v_max,
        eps_p=eps_p,
        eps_v=eps_v,
    )

    clipped_u = jnp.clip(raw_u, safe_low, safe_high)
    center = 0.5 * (safe_low + safe_high)
    exec_u_empty = jnp.clip(center, u_min, u_max)
    exec_u = jnp.where(safe_set_empty, exec_u_empty, clipped_u)
    exec_action_norm = exec_u[..., None] / u_scale

    safe_violation = jnp.logical_or(raw_u < safe_low, raw_u > safe_high)
    safe_violation = jnp.where(safe_set_empty, jnp.ones_like(safe_violation), safe_violation)
    return exec_action_norm, safe_violation.astype(jnp.float32), safe_low, safe_high
