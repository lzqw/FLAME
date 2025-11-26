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
        'reward_type': 'sparse',
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
        'reward_type': 'sparse',
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
        'reward_type': 'sparse',
        'continuing_task': True,
        'reset_target': True
    }
)
