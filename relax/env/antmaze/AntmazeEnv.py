import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium_robotics.envs.maze.ant_maze_v4 import AntMazeEnv

MAP_CENTER_BLOCK = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 'g', 1, 1, 1, 1, 1, 'r', 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
]

MAP_H_SHAPE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'g', 0, 0, 1, 1, 1, 0, 0, 'g', 1],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 'r', 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    [1, 'g', 0, 0, 1, 1, 1, 0, 0, 'g', 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

MAP_COMPLEX = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'g', 0, 0, 0, 1, 1, 1, 0, 0, 0, 'g', 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 'r', 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 'g', 0, 0, 0, 1, 1, 1, 0, 0, 0, 'g', 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


class AntMazeVectorWrapper(gym.ObservationWrapper):
    """
    Flattens the AntMaze Dict observation into a single Box (Vector).
    Structure: [observation, achieved_goal, desired_goal]
    """

    def __init__(self, env):
        super().__init__(env)

        # Get dimensions from the original Dict space
        obs_dim = env.observation_space['observation'].shape[0]
        achieved_dim = env.observation_space['achieved_goal'].shape[0]
        desired_dim = env.observation_space['desired_goal'].shape[0]

        total_dim = obs_dim + achieved_dim + desired_dim

        # Define the new flattened Box space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float64
        )

    def observation(self, obs_dict):
        """Concatenate the dictionary components into a vector."""
        return np.concatenate([
            obs_dict['observation'],
            obs_dict['achieved_goal'],
            obs_dict['desired_goal']
        ])


# --- 3. Factory Function (The Entry Point) ---
def make_antmaze_env(maze_map, reward_type='sparse', continuing_task=True, **kwargs):
    """
    Creates the AntMazeEnv (which is a Dict env) and immediately wraps it
    to return a Vector env.
    """
    # 1. Create the base environment (internally it satisfies GoalEnv's Dict requirement)
    env = AntMazeEnv(
        maze_map=maze_map,
        reward_type=reward_type,
        continuing_task=continuing_task,
        **kwargs
    )

    # 2. Wrap it to change the interface to Vector
    env = AntMazeVectorWrapper(env)

    return env
