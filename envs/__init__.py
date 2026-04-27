from gymnasium.envs.registration import register

register(
    id="SafeObstacleNavigation2D-v0",
    entry_point="envs.safe_obstacle_navigation_2d:SafeObstacleNavigation2DEnv",
    max_episode_steps=200,
)
