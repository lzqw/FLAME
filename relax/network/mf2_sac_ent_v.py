from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DACERPolicyNet2_V, QNet_V, EncoderNet
from relax.utils.flow import MeanFlow
from relax.utils.jax_utils import random_key_from_data


class Diffv2Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_poicy: hk.Params
    log_alpha: jax.Array
    encoder: hk.Params


@dataclass
class MF2SACENTNet_V:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    encoder: Callable[[hk.Params, jax.Array], jax.Array]
    num_timesteps: int
    num_timesteps_test: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    noise_schedule: str
    num_ent_timesteps: int
    alpha_value: float
    fixed_alpha: bool

    @property
    def flow(self) -> MeanFlow:
        return MeanFlow(self.num_timesteps)

    @property
    def flow_test(self) -> MeanFlow:
        return MeanFlow(self.num_timesteps_test)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params, _ = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act = sample(key)[0]
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        if self.fixed_alpha:
            act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(0.1) * self.noise_scale
        else:
            act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_action_full(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params, encoder_params = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q

        obs = self.encoder(encoder_params, obs)

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act = sample(key)[0]
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        if self.fixed_alpha:
            act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(0.1) * self.noise_scale
        else:
            act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_vanilla_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _, encoder_params = policy_params

        # obs = self.encoder(encoder_params, obs)

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act.clip(-1, 1)

        act = sample(key)
        return act

    def get_vanilla_action_step(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _, encoder_params = policy_params
        obs = self.encoder(encoder_params, obs)

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow_test.p_sample_traj(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act

        act = sample(key)
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha, q1_params, q2_params, encoder_params = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha, q1_params, q2_params, encoder_params)
        return self.get_action_full(key, policy_params, obs)


    def get_action_entropy_multistep(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        num_steps=self.num_ent_timesteps
        batch_size = obs.shape[0]
        base_entropy = (self.act_dim / 2.0) * (1.0 + math.log(2 * math.pi))

        policy_params, log_alpha, q1_params, q2_params, _ = policy_params

        # Define single sample processing function (to be vmapped)
        def single_sample_fn(k, o):
            key_z, key_eps = jax.random.split(k)

            # Initial state Z1 (Noise, t=1)
            z_current = jax.random.normal(key_z, shape=(self.act_dim,))

            # Hutchinson probe vector (reusing the same epsilon across the trajectory is common to save computation)
            epsilon = jax.random.normal(key_eps, shape=(self.act_dim,))

            # Define time step interval dt
            dt = 1.0 / num_steps

            # Scan function: integrate from t=1 to t=0
            def scan_body(carry, step_idx):
                z, trace_accum = carry

                # Current time segment: from t_start down to t_end
                # E.g., 2 steps: [1.0 -> 0.5], [0.5 -> 0.0]
                t_start = 1.0 - step_idx * dt
                t_end = 1.0 - (step_idx + 1) * dt

                # Define the flow function for this small step
                def step_flow_map(x_in):
                    # MeanFlow model predicts displacement: u(x, r, t)
                    # Following flow.py convention, input is (x, small_time, large_time)
                    # We want to predict displacement from t_start to t_end, i.e., x_start - x_end
                    return self.policy(policy_params, o, x_in, t_end, t_start)

                # Compute displacement (drift) and JVP
                drift, tangent = jax.jvp(step_flow_map, (z,), (epsilon,))

                # Compute divergence contribution for current step: epsilon^T * J * epsilon
                step_trace = jnp.dot(epsilon, tangent)

                # [Trick] Clip Trace to prevent numerical explosion causing Alpha oscillation
                # This trick is also used in flow.py p_sample_ent
                step_trace = jnp.clip(step_trace, -10.0, 10.0)

                # Update state: x_next = x - drift (evolve towards t=0)
                z_next = z - drift
                trace_next = trace_accum + step_trace

                return (z_next, trace_next), None

            # Execute multi-step loop
            # carry initialization: (z_init, 0.0)
            (z_final, total_trace), _ = jax.lax.scan(scan_body, (z_current, 0.0), jnp.arange(num_steps))

            # Final action
            act = z_final

            # Entropy calculation: H(Data) = H(Noise) - Sum(Trace)
            entropy = base_entropy - total_trace

            # If you previously commented out base_entropy, you can revert to: entropy = -total_trace
            # But to reach Target (e.g., -6), you need base_entropy (approx +8) to offset
            return act.clip(-1, 1), entropy

        keys = jax.random.split(key, batch_size)
        acts, entropies = jax.vmap(single_sample_fn)(keys, obs)

        # Add exploration noise (training phase)
        noise_key = jax.random.split(key)[0]
        if self.fixed_alpha:
            acts = acts + jax.random.normal(noise_key, acts.shape) * jnp.float32(0.1) * self.noise_scale
        else:
            # Note: log_alpha here is automatically adjusted by SAC; with more accurate entropy estimation, it should work properly now
            acts = acts + jax.random.normal(noise_key, acts.shape) * jnp.exp(log_alpha) * self.noise_scale

        return acts, entropies


def create_mf2_sac_ent_net_visual(
    key: jax.Array,
    obs_dim: int,
    latent_obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_timesteps_test: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale=0.9,
    num_ent_timesteps: int = 2,
    alpha_value: float = 0.01,
    fixed_alpha: bool = True,
    init_alpha: float = 0.01,
) -> Tuple[MF2SACENTNet_V, Diffv2Params]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet_V(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(
        hk.transform(lambda obs, act, r, t: DACERPolicyNet2_V(diffusion_hidden_sizes, activation)(obs, act, r, t)))
    encoder = hk.without_apply_rng(hk.transform(lambda obs: EncoderNet()(obs)))

    @jax.jit
    def init(key, obs, latent_obs, act):
        q1_key, q2_key, policy_key, encoder_key = jax.random.split(key, 4)
        q1_params = q.init(q1_key, latent_obs, act)
        q2_params = q.init(q2_key, latent_obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, latent_obs, act, 0, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(init_alpha), dtype=jnp.float32)   # math.log(3) or math.log(5) choose one
        encoder_params = encoder.init(encoder_key, obs)
        return Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params,
                            target_policy_params, log_alpha, encoder_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_latent_obs = jnp.zeros((1, latent_obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_latent_obs, sample_act)

    net = MF2SACENTNet_V(q=q.apply, policy=policy.apply, encoder=encoder.apply, num_timesteps=num_timesteps,
                   num_timesteps_test=num_timesteps_test, act_dim=act_dim,num_ent_timesteps=num_ent_timesteps,
                   target_entropy=-act_dim * target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                   noise_schedule='linear', alpha_value = alpha_value,fixed_alpha=fixed_alpha)
    return net, params
