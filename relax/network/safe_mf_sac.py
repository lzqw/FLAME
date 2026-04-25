from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union
import math

import haiku as hk
import jax
import jax.numpy as jnp

from relax.network.blocks import Activation, DACERPolicyNet2, QNet, ValueNet
from relax.utils.flow import MeanFlow
from relax.utils.jax_utils import random_key_from_data


class SafeMFSACParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    qh: hk.Params
    vh: hk.Params
    target_vh: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    log_alpha: jax.Array


@dataclass
class SafeMFSACNet:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    qh: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    vh: Callable[[hk.Params, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    num_timesteps_test: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float

    @property
    def flow(self) -> MeanFlow:
        return MeanFlow(self.num_timesteps)

    @property
    def flow_test(self) -> MeanFlow:
        return MeanFlow(self.num_timesteps_test)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(sample_key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(sample_key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            return act.clip(-1, 1), jnp.minimum(q1, q2)

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act = sample(key)[0]
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act.clip(-1, 1)

    def get_vanilla_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _ = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        return act.clip(-1, 1)

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, _, q1_params, q2_params = policy_params
        no_noise = jnp.array(-jnp.inf, dtype=jnp.float32)
        return self.get_action(key, (policy_params, no_noise, q1_params, q2_params), obs)


def create_safe_mf_sac_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_timesteps_test: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale: float = 0.9,
) -> Tuple[SafeMFSACNet, SafeMFSACParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    qh = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    vh = hk.without_apply_rng(hk.transform(lambda obs: ValueNet(hidden_sizes, activation)(obs)))
    policy = hk.without_apply_rng(
        hk.transform(lambda obs, act, r, t: DACERPolicyNet2(diffusion_hidden_sizes, activation)(obs, act, r, t))
    )

    @jax.jit
    def init(init_key, obs, act):
        q1_key, q2_key, qh_key, vh_key, policy_key = jax.random.split(init_key, 5)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        qh_params = qh.init(qh_key, obs, act)
        vh_params = vh.init(vh_key, obs)
        policy_params = policy.init(policy_key, obs, act, 0.0, 0.0)
        return SafeMFSACParams(
            q1=q1_params,
            q2=q2_params,
            target_q1=q1_params,
            target_q2=q2_params,
            qh=qh_params,
            vh=vh_params,
            target_vh=vh_params,
            policy=policy_params,
            target_policy=policy_params,
            log_alpha=jnp.array(math.log(5.0), dtype=jnp.float32),
        )

    sample_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    sample_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    params = init(key, sample_obs, sample_act)

    net = SafeMFSACNet(
        q=q.apply,
        qh=qh.apply,
        vh=vh.apply,
        policy=policy.apply,
        num_timesteps=num_timesteps,
        num_timesteps_test=num_timesteps_test,
        act_dim=act_dim,
        num_particles=num_particles,
        target_entropy=-act_dim * target_entropy_scale,
        noise_scale=noise_scale,
    )
    return net, params
