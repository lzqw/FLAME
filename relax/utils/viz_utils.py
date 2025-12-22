import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os


class MultiGoalVisualizer:
    def __init__(self, env, algorithm, log_dir):
        """
        Visualizer for Multi-Goal environments.
        Args:
            env: The environment instance.
            algorithm: The SAC/RL algorithm instance.
            log_dir: Directory to save results.
        """
        self.env = env
        self.algorithm = algorithm
        self.log_dir = log_dir
        self.save_dir = os.path.join(log_dir, "visualizations")
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self, step, key):
        """Execute and save separate PDF visualizations for Value Function and Trajectories."""

        # --- 1. Plot Value Function & Policy Arrows ---
        fig_vf, ax_vf = plt.subplots(figsize=(7, 6))
        self._plot_value_function(ax_vf, key, show_arrows=True)
        # Remove titles and save as PDF
        save_path_vf = os.path.join(self.save_dir, f"vf_step_{step}.pdf")
        fig_vf.savefig(save_path_vf, bbox_inches='tight')
        plt.close(fig_vf)

        # --- 2. Plot Trajectories ---
        fig_traj, ax_traj = plt.subplots(figsize=(7, 6))
        self._plot_trajectories(ax_traj, key, num_episodes=50)
        # Remove titles and save as PDF
        save_path_traj = os.path.join(self.save_dir, f"traj_step_{step}.pdf")
        fig_traj.savefig(save_path_traj, bbox_inches='tight')
        plt.close(fig_traj)

        print(f"[Visualizer] Exported PDFs: {save_path_vf} and {save_path_traj}")

    def _plot_value_function(self, ax, key, show_arrows=False):
        """Plots the learned Value Function landscape and policy vector field."""
        # Plot static env background (e.g., goals, walls)
        self.env._plot_position_cost_static(ax)

        # Create grid for inference
        delta = 0.1
        x_min, x_max = self.env.xlim
        y_min, y_max = self.env.ylim
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )

        flat_obs = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

        if not hasattr(self.algorithm, 'state'):
            return

        sac_params = self.algorithm.state.params

        # Calculate V(s) = min(Q1(s, pi(s)), Q2(s, pi(s)))
        flat_actions = self.algorithm.get_deterministic_action(flat_obs)
        q1_vals = self.algorithm.agent.q(sac_params.q1, flat_obs, flat_actions)
        q2_vals = self.algorithm.agent.q(sac_params.q2, flat_obs, flat_actions)
        v_vals = jnp.minimum(q1_vals, q2_vals)

        # Plot Value Heatmap
        Z = np.array(v_vals).reshape(X.shape)
        ax.contourf(X, Y, Z, levels=35, cmap='viridis', alpha=0.8, zorder=1)

        if show_arrows:
            # Downsample grid for clearer arrow visualization
            skip = 7
            action_grid = flat_actions.reshape(X.shape[0], X.shape[1], 2)

            ax.quiver(
                X[::skip, ::skip], Y[::skip, ::skip],
                action_grid[::skip, ::skip, 0],
                action_grid[::skip, ::skip, 1],
                color='white',
                scale=22,  # Larger scale results in shorter, cleaner arrows
                width=0.006,  # Thicker arrows for visibility
                headwidth=3.5,  # Arrow head size
                pivot='mid',  # Center arrow on the grid point
                alpha=0.85,
                zorder=10
            )

    def _plot_trajectories(self, ax, key, num_episodes=50):
        """Rollout evaluation episodes and plot trajectories in a uniform color."""
        self.env._plot_position_cost_static(ax)

        # Professional uniform color for all trajectories
        traj_color = 'royalblue'

        for i in range(num_episodes):
            key, sub_key = jax.random.split(key)
            obs, _ = self.env.reset()
            traj = [obs.copy()]
            done = False
            steps = 0

            while not done and steps < 60:
                action = self.algorithm.get_action(sub_key, obs)
                obs, r, done, _, _ = self.env.step(action)
                traj.append(obs.copy())
                steps += 1

            traj = np.array(traj)
            # Plot path with transparency to show density of common routes
            ax.plot(traj[:, 0], traj[:, 1], color=traj_color, linewidth=1.0, alpha=0.35, zorder=5)
            # Mark the final position of each trajectory
            ax.scatter(traj[-1, 0], traj[-1, 1], color='crimson', s=6, alpha=0.6, zorder=6)

        # Set consistent limits based on environment bounds
        ax.set_xlim(self.env.xlim)
        ax.set_ylim(self.env.ylim)
