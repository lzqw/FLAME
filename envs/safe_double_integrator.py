from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

from relax.safety.double_integrator_filter import (
    DoubleIntegratorSafetyConfig,
    project_action_np,
    safe_interval_np,
)


class SafeDoubleIntegratorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        ref_type: str = "piecewise_sine",
        noise_sigma: Tuple[float, float] = (0.01, 0.05),
        use_filter: bool = True,
        episode_len: int = 200,
        seed: int | None = None,
    ):
        super().__init__()
        self.ref_type = ref_type
        self.noise_sigma = noise_sigma
        self.use_filter = use_filter
        self.episode_len = episode_len

        self.cfg = DoubleIntegratorSafetyConfig()
        self.dt = self.cfg.dt
        self.u_min = self.cfg.u_min
        self.u_max = self.cfg.u_max
        self.p_min = self.cfg.p_min
        self.p_max = self.cfg.p_max
        self.v_min = self.cfg.v_min
        self.v_max = self.cfg.v_max
        self.eps_p = self.cfg.eps_p
        self.eps_v = self.cfg.eps_v
        self.p_eps_low = self.p_min + self.eps_p
        self.p_eps_high = self.p_max - self.eps_p
        self.v_eps_low = self.v_min + self.eps_v
        self.v_eps_high = self.v_max - self.eps_v

        self.w_p = 1.0
        self.w_v = 0.2
        self.w_u = 0.02
        self.w_du = 0.01

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((2,), dtype=np.float32)
        self.prev_exec_u = 0.0
        self.t = 0

        self._segment_len = 50
        self._ref_params: Dict[str, float] = {}

    def _sample_reference_params(self) -> None:
        if self.ref_type == "step":
            self._ref_params = {"p_ref": float(self.rng.choice([-2.0, 0.0, 2.0])), "v_ref": 0.0}
        else:
            self._ref_params = {
                "A": float(self.rng.uniform(1.0, 2.5)),
                "f": float(self.rng.uniform(0.05, 0.20)),
                "phi": float(self.rng.uniform(0.0, 2.0 * np.pi)),
            }

    def _get_ref(self, t: int) -> Tuple[float, float]:
        if self.ref_type == "step":
            return self._ref_params["p_ref"], self._ref_params["v_ref"]

        if self.ref_type == "piecewise_sine" and t > 0 and t % self._segment_len == 0:
            self._sample_reference_params()

        A = self._ref_params["A"]
        f = self._ref_params["f"]
        phi = self._ref_params["phi"]
        arg = 2.0 * np.pi * f * t * self.dt + phi
        p_ref = A * np.sin(arg)
        v_ref = 2.0 * np.pi * f * A * np.cos(arg)
        return float(p_ref), float(v_ref)

    def _obs(self) -> np.ndarray:
        p, v = self.state
        p_ref, v_ref = self._get_ref(self.t)
        return np.array([p, v, p - p_ref, v - v_ref, p_ref, v_ref], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        p0 = self.rng.uniform(-1.0, 1.0)
        v0 = self.rng.uniform(-0.5, 0.5)
        self.state = np.array([p0, v0], dtype=np.float32)
        self.prev_exec_u = 0.0
        self.t = 0
        self._sample_reference_params()
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, raw_action_norm):
        raw_action_norm = np.asarray(raw_action_norm, dtype=np.float32).reshape(-1)
        raw_action_norm = np.clip(raw_action_norm, -1.0, 1.0)
        raw_u = float(raw_action_norm[0] * self.u_max)

        p, v = float(self.state[0]), float(self.state[1])
        safe_low, safe_high, safe_set_empty = safe_interval_np(p, v, self.cfg)
        if self.use_filter:
            exec_u, safe_low, safe_high, safe_set_empty = project_action_np(raw_u, p, v, self.cfg)
        else:
            exec_u = float(np.clip(raw_u, self.u_min, self.u_max))

        exec_action_norm = np.array([exec_u / self.u_max], dtype=np.float32)
        sigma_p, sigma_v = self.noise_sigma
        noise_p = float(self.rng.normal(0.0, sigma_p))
        noise_v = float(self.rng.normal(0.0, sigma_v))

        p_ref, v_ref = self._get_ref(self.t)
        old_prev_exec_u = self.prev_exec_u

        reward = -(
            self.w_p * (p - p_ref) ** 2
            + self.w_v * (v - v_ref) ** 2
            + self.w_u * exec_u ** 2
            + self.w_du * (exec_u - old_prev_exec_u) ** 2
        )

        p_next = p + self.dt * v + 0.5 * (self.dt ** 2) * exec_u + noise_p
        v_next = v + self.dt * exec_u + noise_v

        self.prev_exec_u = exec_u
        self.state = np.array([p_next, v_next], dtype=np.float32)
        self.t += 1

        terminated = False
        truncated = self.t >= self.episode_len

        state_violation = bool(
            p_next < self.p_min or p_next > self.p_max or v_next < self.v_min or v_next > self.v_max
        )
        tightened_violation = bool(
            p_next < self.p_eps_low
            or p_next > self.p_eps_high
            or v_next < self.v_eps_low
            or v_next > self.v_eps_high
        )

        safe_violation = bool(raw_u < safe_low or raw_u > safe_high) if not safe_set_empty else True
        projection_gap = raw_u - exec_u

        info = {
            "raw_action": raw_action_norm.copy(),
            "exec_action": exec_action_norm.copy(),
            "raw_u": raw_u,
            "exec_u": exec_u,
            "safe_low_u": safe_low,
            "safe_high_u": safe_high,
            "safe_low_norm": safe_low / self.u_max,
            "safe_high_norm": safe_high / self.u_max,
            "safe_violation": safe_violation,
            "filter_active": bool(abs(projection_gap) > 1e-6),
            "projection_gap": projection_gap,
            "safe_set_empty": bool(safe_set_empty),
            "state_violation": state_violation,
            "tightened_violation": tightened_violation,
            "p_ref": p_ref,
            "v_ref": v_ref,
            "tracking_error": p - p_ref,
            "control_energy": exec_u ** 2,
            "control_variation": (exec_u - old_prev_exec_u) ** 2,
            "noise_p": noise_p,
            "noise_v": noise_v,
        }

        return self._obs(), float(reward), terminated, truncated, info
