from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


@dataclass
class SafeObstacleNavigation2DConfig:
    dt: float = 0.1
    u_max: float = 1.0
    workspace_x: Tuple[float, float] = (-3.5, 3.5)
    workspace_y: Tuple[float, float] = (-2.0, 2.0)
    obstacle_center: Tuple[float, float] = (0.0, 0.0)
    obstacle_radius: float = 0.8
    tightened_obstacle_radius: float = 0.88
    eps_box: float = 0.05
    goal: Tuple[float, float] = (-2.6, 0.0)
    start_x: float = 2.6
    start_y_low: float = -0.15
    start_y_high: float = 0.15
    goal_tolerance: float = 0.15
    episode_len: int = 200
    noise_sigma: Tuple[float, float] = (0.01, 0.01)


class SafeObstacleNavigation2DEnv(gym.Env):
    """2D point-mass safe navigation with a circular obstacle.

    State: x=[px, py]
    Observation:
        [px, py,
         px-goal_x, py-goal_y,
         px-obs_x, py-obs_y,
         distance_to_obstacle_boundary,
         distance_to_goal]
    Action: normalized 2D action in [-1, 1]^2
    Physical action: u = u_max * action
    Dynamics: x_next = x + dt * u + noise
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: Optional[SafeObstacleNavigation2DConfig] = None):
        super().__init__()
        self.config = config or SafeObstacleNavigation2DConfig()

        self.goal = np.asarray(self.config.goal, dtype=np.float32)
        self.obstacle_center = np.asarray(self.config.obstacle_center, dtype=np.float32)
        self.workspace_low = np.array([self.config.workspace_x[0], self.config.workspace_y[0]], dtype=np.float32)
        self.workspace_high = np.array([self.config.workspace_x[1], self.config.workspace_y[1]], dtype=np.float32)

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_bound = np.array([10.0] * 8, dtype=np.float32)
        self.observation_space = Box(low=-obs_bound, high=obs_bound, dtype=np.float32)

        self.pos = np.zeros(2, dtype=np.float32)
        self.prev_exec_u = np.zeros(2, dtype=np.float32)
        self.t = 0

    def _sample_start(self) -> np.ndarray:
        y = self.np_random.uniform(self.config.start_y_low, self.config.start_y_high)
        return np.array([self.config.start_x, y], dtype=np.float32)

    def _dist_to_obstacle_boundary(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.obstacle_center) - self.config.obstacle_radius)

    def _dist_to_goal(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.goal))

    def _is_workspace_violation(self, pos: np.ndarray) -> bool:
        return bool(np.any(pos < self.workspace_low) or np.any(pos > self.workspace_high))

    def _is_collision(self, pos: np.ndarray) -> bool:
        return bool(np.linalg.norm(pos - self.obstacle_center) <= self.config.obstacle_radius)

    def _build_obs(self, pos: np.ndarray) -> np.ndarray:
        d_goal = self._dist_to_goal(pos)
        d_obs = self._dist_to_obstacle_boundary(pos)
        obs = np.array(
            [
                pos[0],
                pos[1],
                pos[0] - self.goal[0],
                pos[1] - self.goal[1],
                pos[0] - self.obstacle_center[0],
                pos[1] - self.obstacle_center[1],
                d_obs,
                d_goal,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.pos = self._sample_start()
        self.prev_exec_u = np.zeros(2, dtype=np.float32)
        self.t = 0
        return self._build_obs(self.pos), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        exec_u = self.config.u_max * action
        noise = np.asarray(self.config.noise_sigma, dtype=np.float32) * self.np_random.normal(size=2).astype(np.float32)

        next_pos = self.pos + self.config.dt * exec_u + noise

        collision = self._is_collision(next_pos)
        workspace_violation = self._is_workspace_violation(next_pos)
        state_violation = collision or workspace_violation

        dist_goal = self._dist_to_goal(next_pos)
        success = dist_goal <= self.config.goal_tolerance

        reward = -dist_goal**2
        reward -= 0.02 * float(np.dot(exec_u, exec_u))
        du = exec_u - self.prev_exec_u
        reward -= 0.01 * float(np.dot(du, du))
        if success:
            reward += 20.0
        if state_violation:
            reward -= 50.0

        self.pos = next_pos.astype(np.float32)
        self.prev_exec_u = exec_u.astype(np.float32)
        self.t += 1

        terminated = bool(success or state_violation)
        truncated = self.t >= self.config.episode_len

        info = {
            "success": success,
            "collision": collision,
            "workspace_violation": workspace_violation,
            "state_violation": state_violation,
            "distance_to_goal": dist_goal,
            "distance_to_obstacle_boundary": self._dist_to_obstacle_boundary(self.pos),
            "position": self.pos.copy(),
            "executed_action": action.copy(),
        }

        return self._build_obs(self.pos), float(reward), terminated, truncated, info
