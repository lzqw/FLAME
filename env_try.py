import gymnasium as gym
env_hopper = gym.make("Hopper-v5")

# 获取动作空间
action_space_hopper = env_hopper.action_space
print(action_space_hopper)
