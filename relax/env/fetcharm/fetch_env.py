import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv,MujocoPyFetchEnv
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from gymnasium_robotics.envs.fetch.slide import MujocoFetchSlideEnv

class FetchWrapper(gym.ObservationWrapper):
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
def make_fetch_pick_and_place_env(reward_type='sparse', continuing_task=True, **kwargs):
    # 1. Create the base environment (internally it satisfies GoalEnv's Dict requirement)
    env = MujocoFetchPickAndPlaceEnv(
        reward_type=reward_type,
        **kwargs
    )

    # 2. Wrap it to change the interface to Vector
    env = FetchWrapper(env)

    return env


def make_fetch_push_env(reward_type='sparse', continuing_task=True, **kwargs):
    # 1. Create the base environment (internally it satisfies GoalEnv's Dict requirement)
    env = MujocoFetchPushEnv(
        reward_type=reward_type,
        **kwargs
    )

    # 2. Wrap it to change the interface to Vector
    env = FetchWrapper(env)

    return env

def make_fetch_reach_env(reward_type='sparse', continuing_task=True, **kwargs):
    # 1. Create the base environment (internally it satisfies GoalEnv's Dict requirement)
    env = MujocoFetchReachEnv(
        reward_type=reward_type,
        **kwargs
    )
    # 2. Wrap it to change the interface to Vector
    env = FetchWrapper(env)

    return env

def make_fetch_slide_env(reward_type='sparse', continuing_task=True, **kwargs):
    # 1. Create the base environment (internally it satisfies GoalEnv's Dict requirement)
    env = MujocoFetchSlideEnv(
        reward_type=reward_type,
        **kwargs
    )
    # 2. Wrap it to change the interface to Vector
    env = FetchWrapper(env)

    return env
