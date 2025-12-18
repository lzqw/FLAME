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
    num_ent_timesteps: int
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
        #log_prob = self.compute_log_likelihood3(entropy_key, policy_params_only, obs, act)

        # Entropy is the negative log probability
        #entropy = -log_prob

        # Add exploration noise to the action that will be executed
        # noisy_act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        if self.fixed_alpha:
            noisy_act = act + jax.random.normal(noise_key, act.shape) * jnp.float32(0.1) * self.noise_scale
        else:
            noisy_act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return noisy_act#, entropy

    def compute_log_likelihood2(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array,
                               act: jax.Array) -> jax.Array:
        """
        Computes the log-likelihood log p(a|s) using the ODE probability flow
        and Hutchinson's trace estimator.
        Integration path: t=1 (Data/Action) -> t=0 (Noise).
        """

        # 1. 准备参数和常量
        # 注意：这里 policy_params 已经被外部解包过了，还是完整的 Tuple 取决于调用的位置
        # 根据 get_action_ent 的上下文，这里传入的是 policy_params_only

        # 基础分布 p_0 是标准正态分布: log p_0(x)
        def log_p0(x):
            return jax.scipy.stats.norm.logpdf(x).sum(axis=-1)

        batch_size = act.shape[0]
        act_dim = act.shape[-1]

        # 使用配置中的步数，或者为了精度可以使用更多步数 (e.g. 20-50)
        num_steps = 4
        dt = 1.0 / num_steps

        # 2. 为 Hutchinson Trace Estimator 生成随机噪声 epsilon
        # 形状为 (Batch, Dim)，整个轨迹复用同一个 epsilon (这也是常见做法，节省计算且方差可控)
        epsilon_key, _ = jax.random.split(key)
        epsilon = jax.random.normal(epsilon_key, shape=act.shape)

        # 3. 定义 ODE 的单步积分逻辑 (Solver Step)
        # 我们从 t=1 积分到 t=0，所以是逆向时间

        def scan_body(carry, step_idx):
            x_t, delta_logp_accum = carry

            # 当前时间 t (从 1.0 递减到 0.0)
            t_current = 1.0 - step_idx * dt
            # 下一步时间 (t - dt)
            t_next = 1.0 - (step_idx + 1) * dt

            # [Safety] 避免 t=0 或 t=1 的边界数值问题
            t_current_safe = jnp.clip(t_current, 1e-5, 1.0 - 1e-5)

            # 定义计算速度场(Velocity/Drift)的函数，用于 JVP
            # 注意：Flow Matching 中，policy 输出的是 vector field u_t(x)
            def vector_field_fn(x_in):
                # 这里的传参取决于你的 Policy 定义。
                # 通常 Flow Matching 的 u(x, t) 只依赖当前 t。
                # 如果你的 policy 需要 (x, r, t)，通常 r=0 或 r=t-dt。
                # 鉴于标准 Flow Matching，我们假设它是瞬时速度场。
                # 我们传入 t_current_safe 作为时间条件。
                return self.policy(policy_params, obs, x_in, t_current_safe, t_current_safe)

            # --- 核心计算：值与散度 ---

            # 使用 jax.jvp 同时计算 速度(drift) 和 雅可比向量积(tangent = J * epsilon)
            # drift = v(x, t)
            # tangent = \nabla v \cdot epsilon
            drift, tangent = jax.jvp(vector_field_fn, (x_t,), (epsilon,))

            # Hutchinson Trace: Tr(J) ≈ epsilon^T * (J * epsilon)
            trace = jnp.sum(epsilon * tangent, axis=-1)

            # --- 数值稳定性保护 (关键) ---

            # 1. 处理 NaN/Inf: 如果网络输出炸了，强制归零，防止崩溃
            drift = jnp.nan_to_num(drift, nan=0.0, posinf=1.0, neginf=-1.0)
            trace = jnp.nan_to_num(trace, nan=0.0, posinf=10.0, neginf=-10.0)

            # 2. 裁剪 Trace: 防止 log_prob 极速发散
            trace = jnp.clip(trace, -100.0, 100.0)

            # --- Euler 积分更新 ---
            # ODE: dx/dt = v(x,t)
            # 我们向后积分 (backward integration): dt 是负的 (-1/num_steps)
            # x_{t-dt} = x_t + v(x_t) * (-dt) = x_t - drift * dt
            x_prev = x_t - drift * dt

            # Log Density Change Formula:
            # d(log p)/dt = -Tr(J)
            # log p_{t-dt} - log p_t = -Tr(J) * (-dt) = Tr(J) * dt
            # 所以我们要积累的变化量是 trace * dt
            delta_logp_new = delta_logp_accum + trace * dt

            # [Safety] 限制 x 的范围，防止推理过程中数值爆炸
            x_prev = jnp.clip(x_prev, -10.0, 10.0)

            return (x_prev, delta_logp_new), None

        # 4. 执行 Scan 循环
        # 初始状态: (数据样本 act, 累积的 log_det 变化量 0.0)
        init_state = (act, jnp.zeros((batch_size,)))

        (x_0, total_delta_logp), _ = jax.lax.scan(scan_body, init_state, jnp.arange(num_steps))

        # 5. 计算最终 Log Likelihood
        # 公式: log p_1(x) = log p_0(x_0) + \int_0^1 Tr(J) dt
        # 我们在 scan 中累积的是 \int Tr(J) dt
        # 注意正负号：我们是从 t=1 -> 0 积分。
        # 按照标准推导：
        # log p(x_1) = log p(x_0) - \int_{0}^{1} div(v) dt
        # 这里的 total_delta_logp 计算的是 \int_{1}^{0} -div(v) (-dt) = \int_{0}^{1} -div(v) dt (如果是准确反向)
        # 简单来说：log_p_data = log_p_noise + sum(trace * dt)

        log_prob = log_p0(x_0) + total_delta_logp

        # 最后的数值保护
        log_prob = jnp.nan_to_num(log_prob, nan=-10.0, neginf=-100.0, posinf=100.0)

        return log_prob

    def compute_log_likelihood3(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array,
                                act: jax.Array) -> jax.Array:
        """
        Estimates the log-likelihood of the given action using Inverse Multi-step Flow Matching.
        Integrates from Data (t=0) to Noise (t=1) to compute log p(a).

        Args:
            key: PRNG key for Hutchinson trace estimator.
            policy_params: Policy network parameters.
            obs: Observation.
            act: Action to evaluate.
        """
        # 1. Setup constants
        num_steps = 2
        batch_size = act.shape[0]
        dt = 1.0 / num_steps

        # 2. Hutchinson Probe Vector
        epsilon = jax.random.normal(key, shape=act.shape)

        # 3. Define the inverse step function (Data t=0 -> Noise t=1)
        def scan_body(carry, step_idx):
            z, trace_accum = carry

            # Current time window: moving from t_start to t_end (increasing t)
            t_start = step_idx * dt
            t_end = (step_idx + 1) * dt

            # Define flow map for this step: u(z, t_start, t_end)
            # Note: For reverse flow, we ask the model to predict the displacement
            # that moves us from t_current towards t=1.
            def step_flow_map(x_in):
                return self.policy(policy_params, obs, x_in, t_start, t_end)

            # Compute displacement (u) and Jacobian-Vector Product (J * epsilon)
            drift, tangent = jax.jvp(step_flow_map, (z,), (epsilon,))

            # Estimate Trace: epsilon^T * J * epsilon
            step_trace = jnp.sum(epsilon * tangent, axis=-1)

            # [Trick] Clip trace for numerical stability
            step_trace = jnp.clip(step_trace, -10.0, 10.0)

            # Update state: z_{t+1} = z_t + u * dt
            # In MeanFlow formulation, the network outputs the displacement (x1 - x0) or similar.
            # Here we add the drift to move towards noise.
            z_next = z + drift

            # Accumulate Log Det Jacobian (Trace)
            # log p(z_next) approx log p(z) - Trace, but we want log p(x).
            # Using Change of Variables: log p(x) = log p(z) + log |det dz/dx|
            # Here trace represents log |det|. So we accumulate it.
            trace_next = trace_accum + step_trace

            return (z_next, trace_next), None

        # 4. Run the inverse ODE solver
        # Start with act (Data) at t=0, accumulate trace starting from 0.0
        (z_final, total_trace), _ = jax.lax.scan(scan_body, (act, jnp.zeros((batch_size,))), jnp.arange(num_steps))

        # 5. Calculate Log Probability
        # log p(act) = log p_prior(z_final) + log |det(d z_final / d act)|
        # log |det| is approximated by total_trace

        # log p_prior(z) where z ~ N(0, I)
        log_p_prior = jnp.sum(jax.scipy.stats.norm.logpdf(z_final), axis=-1)

        log_prob = log_p_prior + total_trace

        return log_prob

    def get_action_entropy_multistep(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array,
                                     ) -> Tuple[jax.Array, jax.Array]:
        """
        Entropy estimation and action sampling based on Multi-step Flow Matching.
        Mitigates the truncation error of single-step Jacobian linearization by splitting the [0, 1] interval into num_steps segments.

        Args:
            num_steps: Number of steps. 2 or 4 is recommended. More steps yield more accurate estimation and a lower reachable entropy lower bound.
        """
        num_steps=3
        policy_params_only, log_alpha, q1_params, q2_params = policy_params
        batch_size = obs.shape[0]

        # 1. Base entropy constant H(Prior)
        base_entropy = (self.act_dim / 2.0) * (1.0 + math.log(2 * math.pi))

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
                    return self.policy(policy_params_only, o, x_in, t_end, t_start)

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

def create_mf2_sac_ent2_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_timesteps_test: int = 20,
    num_ent_timesteps: int=2,
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
                 act_dim=act_dim, num_ent_timesteps=num_ent_timesteps,
                 target_entropy=-act_dim * target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                 noise_schedule='linear', alpha_value = alpha_value,fixed_alpha=fixed_alpha
    )
    return net, params
