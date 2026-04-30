import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gymnasium import Env
from gymnasium.spaces import Box

from gymnasium.envs.registration import register

register(
    id='MultiGoal-Custom-v0',  # 这是你将在训练脚本中使用的 ID
    entry_point='relax.env.multi_goal.multi_goal_env:MultiGoalEnv', # 格式为 "文件名:类名"
    max_episode_steps=50,      # 设置最大步数，gym 会自动添加 TimeLimit wrapper
)

class MultiGoalEnv(Env):
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, goal_reward=10, actuation_cost_coeff=30,
                 distance_cost_coeff=1, init_sigma=0.1, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            [
                [5, 0],
                [-5, 0],
                [0, 5],
                [0, -5]
            ],
            dtype=np.float32
        )
        self.goal_threshold = 1.
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.observation = None

        self.reward_range = (-float('inf'), float('inf'))

        # Plotting cache
        self.fig = None
        self.ax = None
        self._agent_plot = None

    def reset(self, *, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        p0 = self.np_random.uniform(-1.0, 1.0)
        v0 = self.np_random.uniform(-0.5, 0.5)
        self.observation = np.array([p0, v0], dtype=np.float32)

        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        self.observation = np.clip(self.observation, o_lb, o_ub).astype(np.float32, copy=False)

        # 在一步动态 x_{t+1} = x_t + a_t (+ noise) 下，当前状态对应的“安全动作区间”
        # 这里只基于无噪声项计算，以便用于调试/可视化。
        a_lb, a_ub = self.action_space.low, self.action_space.high
        safe_low = np.maximum(a_lb, o_lb - self.observation).astype(np.float32)
        safe_high = np.minimum(a_ub, o_ub - self.observation).astype(np.float32)
        distance_to_boundary = float(np.min(np.minimum(self.observation - o_lb, o_ub - self.observation)))
        info = {
            "distance_to_boundary": distance_to_boundary,
            "safe_action_interval": (safe_low, safe_high),
        }

        if self.render_mode == "human":
            self.render()

        return self.observation, info

    @property
    def observation_space(self):
        return Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            shape=(2,),
            dtype=np.float32
        )

    @property
    def action_space(self):
        return Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,),
            dtype=np.float32
        )

    def step(self, action):
        action = action.ravel()

        a_lb, a_ub = self.action_space.low, self.action_space.high
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.dynamics.forward(self.observation, action, self.np_random)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        next_obs = np.clip(next_obs, o_lb, o_ub)

        self.observation = np.copy(next_obs)

        reward = self.compute_reward(self.observation, action)

        dist_to_goal = np.amin([
            np.linalg.norm(self.observation - goal_position)
            for goal_position in self.goal_positions
        ])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        if self.render_mode == "human":
            self.render()

        return next_obs.astype(np.float32), reward, done, False, {'pos': next_obs}

    # -----------------------------------------------------------
    # Visualization Methods
    # -----------------------------------------------------------

    def plot_trajectories(self, trajectories, ax=None):
        """
        Plot multiple trajectories on the cost landscape.

        Args:
            trajectories: A list of numpy arrays. Each array should have shape (T, 2)
                          representing the sequence of (x, y) positions for one episode.
            ax: Optional matplotlib axis. If None, a new figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            show_when_done = True
        else:
            show_when_done = False

        # 1. Setup canvas and background
        ax.axis('equal')
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Trajectories of {len(trajectories)} Episodes')

        self._plot_position_cost_static(ax)

        # 2. Plot each trajectory
        # Use a colormap to distinguish different episodes
        colors = cm.rainbow(np.linspace(0, 1, len(trajectories)))

        for i, (traj, color) in enumerate(zip(trajectories, colors)):
            traj = np.array(traj)  # Ensure it's a numpy array

            # Plot the line
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, linewidth=1.5, label=f'Ep {i + 1}')

            # Mark the start point (Green 'x')
            ax.plot(traj[0, 0], traj[0, 1], 'gx', markersize=5)

            # Mark the end point (Blue dot)
            ax.plot(traj[-1, 0], traj[-1, 1], 'b.', markersize=8)

        # Avoid too many legend entries if there are many episodes
        if len(trajectories) <= 5:
            ax.legend(loc='upper right')

        if show_when_done:
            plt.show()

        return ax

    def _init_plot(self):
        """Initialize the interactive animation canvas."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.ax.axis('equal')
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')

            self._plot_position_cost_static(self.ax)
            self._agent_plot, = self.ax.plot([], [], 'bo', markersize=10, label='Agent')

    def render(self):
        if self.render_mode is None: return
        if self.fig is None: self._init_plot()

        if self.observation is not None:
            self._agent_plot.set_data([self.observation[0]], [self.observation[1]])

        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.01)
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def compute_reward(self, observation, action):
        action_cost = np.sum(action ** 2) * self.action_cost_coeff
        cur_position = observation
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward

    def _plot_position_cost_static(self, ax):
        delta = 0.05
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )

        sigma = 1.7
        goal_costs = np.sum([
            40 / (2 * np.pi * (sigma ** 2)) * np.exp(-((X - goal_x) ** 2 + (Y - goal_y) ** 2) / (2 * sigma ** 2))
            for goal_x, goal_y in self.goal_positions
        ], axis=0)

        costs = goal_costs
        levels = np.linspace(np.min(costs), np.max(costs), 20)

        contours = ax.contour(X, Y, costs, levels=levels, alpha=0.5, zorder=1)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
        ax.plot(self.goal_positions[:, 0], self.goal_positions[:, 1], 'ro', label='Goal', zorder=5)

    def horizon(self):
        return None


class PointDynamics(object):
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action, np_random):
        mu_next = state + action
        state_next = mu_next + self.sigma * np_random.normal(size=self.s_dim)
        return state_next


if __name__ == "__main__":
    # Initialize environment without interactive render
    env = MultiGoalEnv(render_mode=None)

    all_trajectories = []
    num_episodes = 5

    print(f"Collecting {num_episodes} episodes...")

    for i in range(num_episodes):
        obs, _ = env.reset()
        current_traj = [obs.copy()]  # Store initial observation
        done = False
        step_count = 0

        # Run one episode
        while not done and step_count < 50:  # Limit steps to avoid infinite loops in random walk
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            current_traj.append(obs.copy())
            step_count += 1

        all_trajectories.append(np.array(current_traj))
        print(f"Episode {i + 1} finished with {len(current_traj)} steps.")

    # Close env resources
    env.close()

    # Plot all collected trajectories
    print("Plotting trajectories...")
    env.plot_trajectories(all_trajectories)
