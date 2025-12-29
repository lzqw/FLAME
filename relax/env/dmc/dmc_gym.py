import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
import dm_env
from dm_env import specs
import numpy as np
import collections
from dm_control.rl.control import FLAT_OBSERVATION_KEY
from dm_control.suite.wrappers import action_scale, pixels


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

        original_spec = self._env.observation_spec()
        # Ensure consistent dtype
        dtype = original_spec[next(iter(original_spec))].dtype

        # Calculate total flattened size
        num_elem = sum([int(np.prod(v.shape)) for v in original_spec.values()])
        self._obs_spec = collections.OrderedDict()
        self._obs_spec[FLAT_OBSERVATION_KEY] = specs.Array([num_elem], dtype, name=FLAT_OBSERVATION_KEY)

    def _transform_time_step(self, time_step):
        observation = time_step.observation
        if isinstance(observation, collections.OrderedDict):
            keys = observation.keys()
        else:
            keys = sorted(observation.keys())

        # Flatten and concatenate
        observation_arrays = [np.array([observation[k]]) if np.isscalar(observation[k]) else observation[k].ravel() for
                              k in keys]

        # Ensure we have data to concatenate
        if observation_arrays:
            flat_obs = np.concatenate(observation_arrays)
        else:
            flat_obs = np.zeros(0, dtype=np.float32)

        observation = type(observation)([(FLAT_OBSERVATION_KEY, flat_obs)])
        return time_step._replace(observation=observation)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_time_step(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        time_step = self._env.reset()
        return self._transform_time_step(time_step)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ChannelsFirstWrapper(dm_env.Environment):
    def __init__(self, env, pixels_key):
        self._env = env
        self._pixels_key = pixels_key
        observation_spec = self._env.observation_spec().copy()
        assert self._pixels_key in observation_spec
        pixel_spec = observation_spec[self._pixels_key]
        h, w, c = pixel_spec.shape
        observation_spec[self._pixels_key] = specs.Array((c, h, w), dtype=pixel_spec.dtype, name=pixel_spec.name)
        self._obs_spec = observation_spec

    def _transform_time_step(self, time_step):
        observation = time_step.observation
        observation[self._pixels_key] = observation[self._pixels_key].transpose(2, 0, 1).copy()
        return time_step._replace(observation=observation)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_time_step(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        time_step = self._env.reset()
        return self._transform_time_step(time_step)

    def __getattr__(self, name):
        return getattr(self._env, name)


class StackWrapper(dm_env.Environment):
    def __init__(self, env, stack):
        self._env = env
        self._stack = stack
        self._queues = collections.OrderedDict()
        self._obs_spec = collections.OrderedDict()
        for k, v in env.observation_spec().items():
            new_shape = np.concatenate([[stack], v.shape], axis=0)
            self._obs_spec[k] = specs.Array(new_shape, dtype=v.dtype, name=v.name)
            self._queues[k] = collections.deque([], maxlen=stack)

    def _transform_time_step(self, time_step):
        new_observation = collections.OrderedDict()
        for k in time_step.observation.keys():
            new_observation[k] = np.stack(self._queues[k], axis=0)
        return time_step._replace(observation=new_observation)

    def reset(self):
        time_step = self._env.reset()
        for k, v in time_step.observation.items():
            for _ in range(self._stack):
                self._queues[k].append(v)
        return self._transform_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        for k, v in time_step.observation.items():
            self._queues[k].append(v)
        return self._transform_time_step(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def _spec_to_space(spec):
    if isinstance(spec, specs.BoundedArray):
        dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
        return spaces.Box(low=spec.minimum.astype(dtype),
                          high=spec.maximum.astype(dtype),
                          shape=spec.shape, dtype=dtype)
    elif isinstance(spec, specs.Array):
        dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
        return spaces.Box(low=-np.inf, high=np.inf, shape=spec.shape, dtype=dtype)
    elif isinstance(spec, (dict, collections.OrderedDict)):
        dict_of_spaces = collections.OrderedDict()
        for k, v in spec.items():
            dict_of_spaces[k] = _spec_to_space(v)
        return spaces.Dict(spaces=dict_of_spaces)
    else:
        raise ValueError("Invalid spec encountered.")


class DMControlEnv(gym.Env):
    def __init__(self, domain_name, task_name,
                 task_kwargs=None,
                 environment_kwargs=None,
                 visualize_reward=False,
                 action_dtype=np.float32,
                 action_repeat=1,
                 action_minimum=None,
                 action_maximum=None,
                 from_pixels=False,
                 height=84,
                 width=84,
                 camera_id=0,
                 channels_first=True,
                 flatten=True,
                 stack=1):

        self._pixels_key = 'pixels'
        self._height = height
        self._width = width
        self._camera_id = camera_id

        if task_kwargs is None:
            task_kwargs = {}
        # DMC uses 'random' in task_kwargs for seeding during creation
        self.task_kwargs = task_kwargs

        self.env = suite.load(domain_name, task_name,
                              task_kwargs=task_kwargs,
                              environment_kwargs=environment_kwargs,
                              visualize_reward=visualize_reward)

        env = ActionDTypeWrapper(self.env, action_dtype)
        if action_repeat > 1:
            env = ActionRepeatWrapper(env, action_repeat)
        if action_minimum is not None and action_maximum is not None:
            env = action_scale.Wrapper(env, minimum=action_minimum, maximum=action_maximum)
        if from_pixels:
            render_kwargs = dict(height=height, width=width, camera_id=camera_id)
            env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs, observation_key=self._pixels_key)
            if channels_first:
                env = ChannelsFirstWrapper(env, pixels_key=self._pixels_key)
        if flatten:
            env = FlattenWrapper(env)
        if stack > 1:
            env = StackWrapper(env, stack)

        self._env = env

        obs_spec = self._env.observation_spec()
        if (isinstance(obs_spec, (dict, collections.OrderedDict))) and len(obs_spec) == 1:
            self._unwrap_obs = True
            obs_spec = obs_spec[next(iter(obs_spec))]
        else:
            self._unwrap_obs = False
        self.observation_space = _spec_to_space(obs_spec)
        self.action_space = _spec_to_space(self._env.action_spec())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def seed(self, seed=None):
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            # Note: DMC environments are seeded at creation via task_kwargs.
            # Dynamic reseeding of the underlying physics engine is not standard in DMC wrappers
            # without reloading the task, but we set the space seeds here.
        return [seed]

    def _extract_obs(self, time_step):
        obs = time_step.observation
        # Convert float64 to float32
        if isinstance(obs, (dict, collections.OrderedDict)):
            for k in obs.keys():
                if obs[k].dtype == np.float64:
                    obs[k] = obs[k].astype(np.float32)
            if self._unwrap_obs:
                return obs[next(iter(obs))]
        elif isinstance(obs, np.ndarray) and obs.dtype == np.float64:
            obs = obs.astype(np.float32)
        return obs

    # --- Gymnasium API Update ---
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        time_step = self._env.reset()
        obs = self._extract_obs(time_step)
        info = {}  # Gymnasium requires info return
        return obs, info

    def step(self, action):
        time_step = self._env.step(action)

        obs = self._extract_obs(time_step)
        reward = time_step.reward or 0.0

        # DMC definitions
        terminated = time_step.last()  # Natural termination (e.g. fallen over)
        truncated = False  # Time limit usually handled by wrapper, default False here

        discount = time_step.discount
        info = {'discount': discount}
        if discount == 0.0:
            info['early_termination'] = 1.0  # Sometimes used to distinguish death from timeout
        else:
            info['early_termination'] = 0.0

        return obs, reward, terminated, truncated, info
