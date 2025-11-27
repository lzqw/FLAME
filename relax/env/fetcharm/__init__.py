from gymnasium.envs.registration import register
import gymnasium as gym

# --- Register Fetch Environments ---

# Register Fetch Pick and Place
register(
    id='Fetch_PickandPlace-v0',
    entry_point='relax.env.fetcharm.fetch_env:make_fetch_pick_and_place_env',
    max_episode_steps=1000,
    kwargs={
        'reward_type': 'dense',
    }
)

# Register Fetch Push
register(
    id='Fetch_Push-v0',
    entry_point='relax.env.fetcharm.fetch_env:make_fetch_push_env',
    max_episode_steps=1000,
    kwargs={
        'reward_type': 'dense',
    }
)

# Register Fetch Reach
register(
    id='Fetch_Reach-v0',
    entry_point='relax.env.fetcharm.fetch_env:make_fetch_reach_env',
    max_episode_steps=1000,
    kwargs={
        'reward_type': 'dense',
    }
)

# Register Fetch Slide
register(
    id='Fetch_Slide-v0',
    entry_point='relax.env.fetcharm.fetch_env:make_fetch_slide_env',
    max_episode_steps=1000,
    kwargs={
        'reward_type': 'dense',

    }
)

# --- Test Procedure (Test Car/Process) ---
if __name__ == "__main__":
    print("=== Start Fetch Environment Test Procedure ===")

    # Define the environment ID to test
    env_id = 'Fetch_PickandPlace-v0'

    try:
        print(f"Loading environment: {env_id} ...")

        # Initialize environment with human rendering
        env = gym.make(env_id, render_mode='human')

        # Reset the environment
        obs, info = env.reset()

        print("Environment loaded successfully!")
        # Verify if the Wrapper is working (Observation space should be Box, not Dict)
        print(f"Observation Space Type: {type(env.observation_space)}")
        print(f"Observation Space Shape: {env.observation_space.shape}")
        print(f"Action Space Shape: {env.action_space.shape}")

        print("\nStarting 50-step simulation loop...")

        for i in range(50):
            # Sample a random action
            action = env.action_space.sample()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Print reward for debugging (optional)
            # print(f"Step {i}: Reward = {reward}")

            # Render the current frame
            env.render()

            # Reset if the episode ends
            if terminated or truncated:
                print(f"Episode finished at step {i}, resetting environment...")
                obs, info = env.reset()

        env.close()
        print("\n=== Test procedure finished successfully ===")

    except ImportError:
        print("\n[Error] Module import failed.")
        print("Please ensure 'fetch_env.py' is in 'relax/env/fetch/' and you are running this from the root directory.")
    except Exception as e:
        print(f"\n[Error] An unknown exception occurred: {e}")
