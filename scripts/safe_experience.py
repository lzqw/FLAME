from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax


def probe_batch_size(reward: "jax.Array") -> Optional[int]:
    try:
        if reward.ndim > 0:
            return reward.shape[0]
        return None
    except AttributeError:
        return None


class FilteredExperience(NamedTuple):
    obs: "jax.Array"
    raw_action: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    safe_violation: "jax.Array"
    filter_active: "jax.Array"
    projection_gap: "jax.Array"
    safe_low: "jax.Array"
    safe_high: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return FilteredExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            raw_action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            safe_violation=np.zeros(leading_dims, dtype=np.float32),
            filter_active=np.zeros(leading_dims, dtype=np.float32),
            projection_gap=np.zeros(leading_dims, dtype=np.float32),
            safe_low=np.zeros(leading_dims, dtype=np.float32),
            safe_high=np.zeros(leading_dims, dtype=np.float32),
        )

    @staticmethod
    def create(obs, raw_action, exec_action, reward, terminated, truncated, next_obs, info):
        done = np.logical_or(terminated, truncated)
        return FilteredExperience(
            obs=obs,
            raw_action=raw_action,
            action=exec_action,
            reward=reward,
            done=done,
            next_obs=next_obs,
            safe_violation=np.float32(info["safe_violation"]),
            filter_active=np.float32(info["filter_active"]),
            projection_gap=np.float32(info["projection_gap"]),
            safe_low=np.float32(info["safe_low_norm"]),
            safe_high=np.float32(info["safe_high_norm"]),
        )
