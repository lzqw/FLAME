#!/usr/bin/env python3
"""Utilities for the safe MeanFlow-SAC obstacle-navigation experiment.

This script keeps the framework split:
raw MeanFlow action -> safety projection -> reward critic on executed action
-> feasibility critic on raw action -> feasibility-guided MeanFlow update.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from relax.safety.obstacle_navigation_filter import project_action_numpy


BASELINES = [
    "gaussian_sac",
    "gaussian_sac_filter",
    "meanflow_no_filter",
    "meanflow_filter",
    "meanflow_filter_qh",
    "meanflow_filter_qh_alpha_fixed_0p01",
    "goal_seeking_filter",
]


@dataclass
class SafeTransition:
    obs: np.ndarray
    raw_action: np.ndarray
    exec_action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    safe_violation: bool
    filter_active: bool
    projection_gap: float
    safe_set_empty: bool


def actor_score(qr_value: float, qh_value: float, beta_h: float) -> float:
    return float(qr_value - beta_h * qh_value)


def goal_seeking_policy(obs: np.ndarray) -> np.ndarray:
    rel_goal = -obs[2:4]
    n = np.linalg.norm(rel_goal)
    if n < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return np.clip(rel_goal / n, -1.0, 1.0).astype(np.float32)


def rollout_with_filter(env: SafeObstacleNavigation2DEnv, policy_fn, max_steps: int = 200):
    obs, _ = env.reset()
    traj = [obs[:2].copy()]
    transitions: List[SafeTransition] = []

    for _ in range(max_steps):
        raw_action = np.asarray(policy_fn(obs), dtype=np.float32)
        proj = project_action_numpy(obs[:2], raw_action)
        exec_action = proj["exec_action"]

        next_obs, reward, terminated, truncated, info = env.step(exec_action)
        done = bool(terminated or truncated)

        transitions.append(
            SafeTransition(
                obs=obs,
                raw_action=raw_action,
                exec_action=exec_action,
                reward=float(reward),
                next_obs=next_obs,
                done=done,
                safe_violation=bool(proj["safe_violation"]),
                filter_active=bool(proj["filter_active"]),
                projection_gap=float(proj["projection_gap"]),
                safe_set_empty=bool(proj["safe_set_empty"]),
            )
        )
        obs = next_obs
        traj.append(obs[:2].copy())
        if done:
            break

    return transitions, np.asarray(traj, dtype=np.float32), info


def evaluate_episode(transitions: List[SafeTransition], final_info: Dict) -> Dict[str, float]:
    positions = np.array([t.obs[:2] for t in transitions], dtype=np.float32)
    exec_actions = np.array([t.exec_action for t in transitions], dtype=np.float32)

    if len(positions) > 1:
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum())
    else:
        path_length = 0.0

    route_upper_ratio = float(np.mean(positions[:, 1] > 0.0)) if len(positions) else 0.0
    route_lower_ratio = float(np.mean(positions[:, 1] < 0.0)) if len(positions) else 0.0

    y_bins = np.digitize(positions[:, 1], bins=np.array([-2.0, -0.2, 0.2, 2.0])) if len(positions) else np.array([])
    if len(y_bins):
        probs = np.bincount(y_bins, minlength=6).astype(np.float32)
        probs = probs / np.sum(probs)
        route_entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0] + 1e-12)).sum())
    else:
        route_entropy = 0.0

    if len(positions):
        bins = [np.linspace(-3.5, 3.5, 30), np.linspace(-2.0, 2.0, 20)]
        hist, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=bins)
        p = hist.flatten()
        p = p / max(np.sum(p), 1.0)
        occupancy_entropy = float(-(p[p > 0] * np.log(p[p > 0] + 1e-12)).sum())
        boundary_mask = (
            (positions[:, 0] <= -3.4)
            | (positions[:, 0] >= 3.4)
            | (positions[:, 1] <= -1.9)
            | (positions[:, 1] >= 1.9)
        )
        boundary_coverage = float(np.mean(boundary_mask))
    else:
        occupancy_entropy = 0.0
        boundary_coverage = 0.0

    metrics = {
        "success_rate": float(bool(final_info.get("success", False))),
        "collision_rate": float(bool(final_info.get("collision", False))),
        "violation_rate": float(bool(final_info.get("state_violation", False))),
        "episode_return": float(np.sum([t.reward for t in transitions])),
        "path_length": path_length,
        "time_to_goal": float(len(transitions) if final_info.get("success", False) else np.nan),
        "filter_activation_rate": float(np.mean([t.filter_active for t in transitions])) if transitions else 0.0,
        "avg_projection_residual": float(np.mean([t.projection_gap for t in transitions])) if transitions else 0.0,
        "feasible_raw_action_ratio": float(np.mean([not t.safe_violation for t in transitions])) if transitions else 0.0,
        "route_upper_ratio": route_upper_ratio,
        "route_lower_ratio": route_lower_ratio,
        "route_entropy": route_entropy,
        "occupancy_entropy": occupancy_entropy,
        "boundary_coverage": boundary_coverage,
    }

    _ = exec_actions  # retained for compatibility with downstream analysis hooks.
    return metrics


if __name__ == "__main__":
    env = SafeObstacleNavigation2DEnv()
    transitions, traj, info = rollout_with_filter(env, goal_seeking_policy, max_steps=200)
    metrics = evaluate_episode(transitions, info)
    print("Baselines:", BASELINES)
    print("Trajectory shape:", traj.shape)
    print("Metrics:", metrics)
