from gymnasium.envs.registration import register
from relax.env.antmaze.AntmazeEnv import (
    MAP_CENTER_BLOCK,
    MAP_H_SHAPE,
    MAP_COMPLEX
)

# NOTE: The 'entry_point' now points to the function 'make_antmaze_env'.
# This ensures the environment is created AND wrapped automatically.

register(
    id='AntMaze_CenterBlock-v0',
    entry_point='relax.env.antmaze.AntmazeEnv:make_antmaze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': MAP_CENTER_BLOCK,
        'reward_type': 'dense',
        'continuing_task': True,
        'reset_target': True
    }
)

register(
    id='AntMaze_HShape-v0',
    entry_point='relax.env.antmaze.AntmazeEnv:make_antmaze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': MAP_H_SHAPE,
        'reward_type': 'dense',
        'continuing_task': True,
        'reset_target': True
    }
)

register(
    id='AntMaze_Complex-v0',
    entry_point='relax.env.antmaze.AntmazeEnv:make_antmaze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': MAP_COMPLEX,
        'reward_type': 'dense',
        'continuing_task': True,
        'reset_target': True
    }
)

if __name__ == "__main__":
    import gymnasium as gym
    env=gym.make('AntMaze_CenterBlock-v0', render_mode='human')
    for _ in range(1000):
        obs, _ = env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(reward)
            env.render()
            if terminated or truncated:
                obs, _ = env.reset()
