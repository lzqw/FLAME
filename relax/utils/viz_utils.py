import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os


class MultiGoalVisualizer:
    def __init__(self, env, algorithm, log_dir):
        self.env = env
        self.algorithm = algorithm
        self.log_dir = log_dir
        self.save_dir = os.path.join(log_dir, "visualizations")
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self, step, key):
        """Execute all visualizations and save the image."""
        fig = plt.figure(figsize=(15, 6))

        # --- Left Plot: Value Function + Policy Arrows ---
        ax1 = fig.add_subplot(121)
        # show_arrows=True: Plot both V(s) heatmap and action arrows
        self._plot_value_function(ax1, key, show_arrows=True)
        ax1.set_title(f"Value Function & Policy Field (Step {step})")

        # --- Right Plot: Trajectories Only (No Value Heatmap) ---
        ax2 = fig.add_subplot(122)

        # Plot trajectories (adjusted num_episodes to 20 for better coverage)
        # This function draws the static env background but NOT the learned Value heatmap
        self._plot_trajectories(ax2, key, num_episodes=50)

        ax2.set_title(f"Trajectories Only (Step {step})")

        # Save the figure
        save_path = os.path.join(self.save_dir, f"viz_step_{step}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"[Visualizer] Saved visualization to {save_path}")

    def _plot_value_function(self, ax, key, show_arrows=False):
        """
        Plots the true Value Function V(s) = Q(s, pi(s)).
        Optionally overlays the policy vector field (Arrows).
        """
        # 1. Plot background (Ground Truth Cost)
        self.env._plot_position_cost_static(ax)

        # 2. Generate mesh grid (State Grid)
        delta = 0.1
        x_min, x_max = self.env.xlim
        y_min, y_max = self.env.ylim
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )

        # Flatten grid for batch prediction: (N, 2)
        flat_obs = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

        # 3. Get parameters
        if hasattr(self.algorithm, 'state'):
            sac_params = self.algorithm.state.params
        else:
            print("[Viz Warning] No state found, skipping plot.")
            return

        # 4. Calculate V(s)
        # 4.1 Get Deterministic Action for every state in the grid
        flat_actions = self.algorithm.get_deterministic_action(flat_obs)

        # 4.2 Calculate Q-values using the Critic
        # Q(s, pi(s)) is the Value V(s)
        q1_vals = self.algorithm.agent.q(sac_params.q1, flat_obs, flat_actions)
        q2_vals = self.algorithm.agent.q(sac_params.q2, flat_obs, flat_actions)

        # Take min(Q1, Q2) as a conservative estimate
        v_vals = jnp.minimum(q1_vals, q2_vals)

        # 5. Plot Contour Map (Value Landscape)
        Z = np.array(v_vals).reshape(X.shape)

        # Use contourf for coloring
        cs = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7, zorder=2)

        # 6. Plot Policy Vector Field (if requested)
        if show_arrows:
            # Downsample to avoid overcrowding arrows (e.g., plot every 4th arrow)
            skip = 4

            # Reshape actions to grid shape (H, W, 2)
            action_grid = flat_actions.reshape(X.shape[0], X.shape[1], 2)

            ax.quiver(
                X[::skip, ::skip], Y[::skip, ::skip],
                action_grid[::skip, ::skip, 0],  # U (x-velocity)
                action_grid[::skip, ::skip, 1],  # V (y-velocity)
                color='white',
                scale=20,
                width=0.003,
                alpha=0.8,
                zorder=10
            )

    def _plot_trajectories(self, ax, key, num_episodes=5):
        """
        Runs evaluation episodes and plots the resulting trajectories.
        """
        # Plot static environment background (Gray contours + Goals)
        self.env._plot_position_cost_static(ax)

        colors = plt.cm.rainbow(np.linspace(0, 1, num_episodes))

        for i in range(num_episodes):
            key, sub_key = jax.random.split(key)
            obs, _ = self.env.reset()
            traj = [obs.copy()]
            done = False
            steps = 0

            # Rollout one episode
            while not done and steps < 50:
                # Get action
                action = self.algorithm.get_action(sub_key, obs)
                obs, r, done, _, _ = self.env.step(action)
                traj.append(obs.copy())
                steps += 1

            traj = np.array(traj)
            # Use lower alpha (transparency) to see overlapping trajectories
            ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=1.5, alpha=0.5)
            ax.plot(traj[-1, 0], traj[-1, 1], 'b.', markersize=5)
