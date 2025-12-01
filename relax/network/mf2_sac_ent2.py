from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union
import functools

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DACERPolicyNet2, QNet
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


@dataclass
class MF2SACENT2Net:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    num_timesteps_test: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    noise_schedule: str
    alpha_value: float
    fixed_alpha: bool

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

    def get_vanilla_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _ = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act.clip(-1, 1)

        act = sample(key)
        return act

    def get_vanilla_action_step(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _ = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow_test.p_sample_traj(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act

        act = sample(key)
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha, q1_params, q2_params = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha, q1_params, q2_params)
        return self.get_action(key, policy_params, obs)

    def compute_log_likelihood(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array,
                               act: jax.Array) -> jax.Array:
        """
        Computes the log-likelihood of actions using the instantaneous change of variables formula.

        This implementation approximates the instantaneous velocity v(z_t, t) required by the
        ODE solver using the MeanFlow network's average velocity over the *previous*
        small time step: v(z_t, t) ≈ u_theta(z_t, r = t - dt, t) / dt.
        """

        # --- Modification 1: Define num_steps and dt ---
        # num_steps = self.num_timesteps
        num_steps = 20  # As in your example
        dt_val = 1.0 / num_steps
        dt_solver = -dt_val  # The solver goes backwards

        # --- Modification 2: model_fn now uses r = t - dt ---
        def model_fn(t, x):
            """
            Approximates the instantaneous velocity v(t, x) using the average velocity
            from the interval [t - dt, t].
            t is a time scalar.
            x is a batch of states.
            """
            # r is the start time, t is the end time
            r_scalar = jnp.maximum(0.0, t - dt_val)

            # Convert time scalars r and t to batch-shaped tensors
            t_batch = jnp.full((x.shape[0],), t)
            r_batch = jnp.full((x.shape[0],), r_scalar)

            # The policy predicts displacement: x_t - x_r
            displacement = self.policy(policy_params, obs, x, r_batch, t_batch)

            # FIX: Convert displacement to velocity for the ODE solver
            # v = dx / dt
            # Using jnp.maximum to avoid division by zero (though t >= dt_val in loop)
            current_dt = jnp.maximum(t - r_scalar, 1e-5)
            velocity = displacement / current_dt

            return velocity

        # Base distribution p_0 is a standard normal
        def log_p0(z):
            return jax.scipy.stats.norm.logpdf(z).sum(axis=-1)

        # We need a single random vector Z for the trace estimator for the entire trajectory
        z_key, _ = jax.random.split(key)
        z = jax.random.normal(z_key, act.shape)

        # --- Modification 3: ode_dynamics now uses the new model_fn ---
        # The dynamics of the augmented ODE system for [f(t), g(t)]
        def ode_dynamics(state, t):
            # t is a scalar (float)
            f_t, _ = state

            # Function for VJP: v_t(f_t) ≈ u(f_t, t-dt, t) / dt
            v_t_fn = lambda x: model_fn(t, x)

            # Calculate v_t for df_dt
            v_t = v_t_fn(f_t)

            # Calculate VJP to get Z^T * J
            # J is d(velocity)/dx
            _, vjp_fn = jax.vjp(v_t_fn, f_t)
            vjp_z = vjp_fn(z)[0]

            # Hutchinson's trace estimator: trace(J) ≈ Z^T * J * Z
            trace_term = jnp.sum(vjp_z * z, axis=-1)

            df_dt = v_t
            dg_dt = -trace_term
            return df_dt, dg_dt

        # --- Modification 4: solver_step uses the defined dt_solver ---
        def solver_step(state, t):
            f_t, g_t = state

            # t is the current time (scalar) from timesteps
            df_dt, dg_dt = ode_dynamics(state, t)

            f_next = f_t + df_dt * dt_solver  # negative or posiiitive
            g_next = g_t + dg_dt * dt_solver
            return (f_next, g_next), None

        # Time steps for reverse integration from t=1.0 down to t=dt_val
        timesteps = jnp.linspace(1.0, dt_val, num_steps)

        # Initial state at t=1
        initial_state = (act, jnp.zeros(act.shape[:-1]))

        # Run the solver
        final_state, _ = jax.lax.scan(solver_step, initial_state, timesteps)
        f_0, g_0 = final_state

        # Final log probability: log p_1(x) = log p_0(f(0)) - g(0)
        log_p1 = log_p0(f_0) - g_0

        return log_p1

    def get_action_ent(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # Unpack params
        policy_params_only, log_alpha, q1_params, q2_params = policy_params

        # Define the model for sampling, which uses both r and t.
        # For sampling, r is typically fixed to 0.
        def model_fn(x, r, t):
            return self.policy(policy_params_only, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q

        # Split keys for sampling, noise, and entropy calculation
        sample_key, noise_key, entropy_key = jax.random.split(key, 3)

        # Sample action(s) from the policy
        if self.num_particles == 1:
            act, _ = sample(sample_key)
        else:
            keys = jax.random.split(sample_key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)

        # Compute log probability of the original action (before adding exploration noise)
        log_prob = self.compute_log_likelihood(entropy_key, policy_params_only, obs, act)

        # Entropy is the negative log probability
        entropy = -log_prob

        # Add exploration noise to the action that will be executed
        # noisy_act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        if self.fixed_alpha:
            noisy_act = act + jax.random.normal(noise_key, act.shape) * jnp.float32(0.1) * self.noise_scale
        else:
            noisy_act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return noisy_act, entropy



def create_mf2_sac_ent2_net(
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
    target_entropy_scale=0.9,
    alpha_value: float = 0.01,
    fixed_alpha: bool = True,
    init_alpha: float = 0.01,
) -> Tuple[MF2SACENT2Net, Diffv2Params]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(
        hk.transform(lambda obs, act, r, t: DACERPolicyNet2(diffusion_hidden_sizes, activation)(obs, act, r, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(init_alpha), dtype=jnp.float32)  # math.log(3) or math.log(5) choose one
        return Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params,
                            target_policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = MF2SACENT2Net(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, num_timesteps_test=num_timesteps_test,
                 act_dim=act_dim,
                 target_entropy=-act_dim * target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                 noise_schedule='linear', alpha_value = alpha_value,fixed_alpha=fixed_alpha
    )
    return net, params
