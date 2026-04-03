import numpy as np
from gymnasium import Env
from gymnasium.envs.registration import register, registry
from gymnasium.spaces import Box


class SafeDoubleIntegratorEnv(Env):
    """Stochastic safe double-integrator environment.

    State: [p, v]
      p: position
      v: velocity
    Action: [u]
      u: acceleration command clipped by ``u_max``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.1,
        Sigma_w: np.ndarray | None = None,
        u_max: float = 2.0,
        p_max: float = 2.0,
        epsilon: float = 0.05,
        T: int = 200,
        q_p: float = 1.0,
        q_v: float = 0.1,
        r_u: float = 0.05,
        lambda_viol: float = 50.0,
    ):
        super().__init__()

        self.dt = float(dt)
        self.Sigma_w = np.diag([0.0025, 0.0025]) if Sigma_w is None else np.asarray(Sigma_w, dtype=np.float64)
        self.u_max = float(u_max)
        self.p_max = float(p_max)
        self.epsilon = float(epsilon)
        self.T = int(T)
        self.q_p = float(q_p)
        self.q_v = float(q_v)
        self.r_u = float(r_u)
        self.lambda_viol = float(lambda_viol)

        obs_bound = np.finfo(np.float32).max
        self.observation_space = Box(
            low=-obs_bound,
            high=obs_bound,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-2.0,
            high=2.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.state = np.zeros(2, dtype=np.float64)
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0

        init_state = np.zeros(2, dtype=np.float64)
        if options is not None and "state" in options:
            init_state = np.asarray(options["state"], dtype=np.float64).reshape(2)

        self.state = init_state
        return self.state.astype(np.float32), {}

    def step(self, action):
        u = float(np.asarray(action).reshape(-1)[0])
        u = float(np.clip(u, -self.u_max, self.u_max))

        p, v = self.state
        w = self.np_random.multivariate_normal(mean=np.zeros(2), cov=self.Sigma_w)

        next_p = p + self.dt * v + w[0]
        next_v = v + self.dt * u + w[1]
        self.state = np.array([next_p, next_v], dtype=np.float64)
        self.t += 1

        pos_violation = max(0.0, abs(next_p) - self.p_max)
        stage_cost = (
            self.q_p * next_p**2
            + self.q_v * next_v**2
            + self.r_u * u**2
            + self.lambda_viol * pos_violation
        )
        reward = -float(stage_cost)

        terminated = abs(next_p) > (self.p_max + self.epsilon)
        truncated = self.t >= self.T

        info = {
            "cost": float(stage_cost),
            "position_violation": float(pos_violation),
            "control": float(u),
        }
        return self.state.astype(np.float32), reward, terminated, truncated, info


if "SafeDoubleIntegrator-v0" not in registry:
    register(
        id="SafeDoubleIntegrator-v0",
        entry_point="relax.env.safe_double_integrator:SafeDoubleIntegratorEnv",
        max_episode_steps=200,
    )
