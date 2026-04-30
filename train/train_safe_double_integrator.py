import argparse
import csv
from pathlib import Path

import jax
import numpy as np

from envs.safe_double_integrator import SafeDoubleIntegratorEnv
from relax.algorithm.mf_sac import MFSAC
from relax.algorithm.safe_mf_sac import SafeMFSAC
from relax.buffer import TreeBuffer
from relax.network.mf_sac import create_mf_sac_net
from relax.network.safe_mf_sac import create_safe_mf_sac_net
from scripts.experience import Experience
from scripts.safe_experience import FilteredExperience


def evaluate_policy(agent, algo_name: str, ref_type: str, eval_episodes: int, seed: int) -> dict:
    eval_env = SafeDoubleIntegratorEnv(ref_type=ref_type, noise_sigma=(0.01, 0.05), use_filter=(algo_name != "mf_no_filter"), seed=seed)
    returns, vrs = [], []
    key = jax.random.PRNGKey(seed + 999)
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_ret = 0.0
        violations = []
        while not done:
            key, subkey = jax.random.split(key)
            act = agent.get_action(subkey, obs[None, :])[0]
            obs, reward, terminated, truncated, info = eval_env.step(act)
            done = terminated or truncated
            ep_ret += reward
            violations.append(float(info["state_violation"]))
        returns.append(ep_ret)
        vrs.append(float(np.mean(violations)))
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "vr_mean": float(np.mean(vrs)),
        "vr_std": float(np.std(vrs)),
    }


def build_agent(args, obs_dim, act_dim, key):
    hidden_sizes = [256, 256, 256]
    diffusion_hidden_sizes = [256, 256, 256]
    if args.algo == "mf_no_filter":
        net, params = create_mf_sac_net(
            key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, num_timesteps=20, num_timesteps_test=20
        )
        algo = MFSAC(net, params, sample_k=args.n_est)
        return algo, Experience.create_example(obs_dim, act_dim, batch_size=1)

    net, params = create_safe_mf_sac_net(
        key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, num_timesteps=20, num_timesteps_test=20
    )
    use_guidance = args.algo == "safe_mf"
    beta_h = args.beta_h if use_guidance else 0.0
    algo = SafeMFSAC(
        net,
        params,
        gamma_h=args.gamma_h,
        tau_h=args.tau_h,
        beta_h=beta_h,
        sample_k=args.n_est,
        use_feasibility_guidance=use_guidance,
    )
    return algo, FilteredExperience.create_example(obs_dim, act_dim, batch_size=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["mf_no_filter", "mf_filter", "safe_mf"], default="safe_mf")
    parser.add_argument("--ref_type", type=str, choices=["step", "sine", "piecewise_sine"], default="piecewise_sine")
    parser.add_argument("--noise_sigma_p", type=float, default=0.01)
    parser.add_argument("--noise_sigma_v", type=float, default=0.05)
    parser.add_argument("--total_steps", type=int, default=int(1e6))
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--update_after", type=int, default=10000)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta_h", type=float, default=1.0)
    parser.add_argument("--gamma_h", type=float, default=0.99)
    parser.add_argument("--tau_h", type=float, default=0.90)
    parser.add_argument("--n_est", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="logs/double_integrator/safe_mf_seed0")
    args = parser.parse_args()

    use_filter = args.algo in ["mf_filter", "safe_mf"]
    env = SafeDoubleIntegratorEnv(
        ref_type=args.ref_type,
        noise_sigma=(args.noise_sigma_p, args.noise_sigma_v),
        use_filter=use_filter,
        seed=args.seed,
    )
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    agent, example = build_agent(args, obs_dim, act_dim, init_key)

    buffer = TreeBuffer.from_example(example, size=1_000_000, seed=args.seed + 1, remove_batch_dim=False)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "eval_metrics.csv"
    ckpt_path = log_dir / "checkpoint.pkl"

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "return_mean", "return_std", "vr_mean", "vr_std"])
        writer.writeheader()

        obs, _ = env.reset(seed=args.seed)
        for step in range(args.total_steps):
            if step < args.start_steps:
                raw_action = env.action_space.sample()
            else:
                key, action_key = jax.random.split(key)
                raw_action = agent.get_action(action_key, obs[None, :])[0]

            next_obs, reward, terminated, truncated, info = env.step(raw_action)
            done = terminated or truncated

            if args.algo == "mf_no_filter":
                transition = Experience.create(
                    obs=obs,
                    action=info["exec_action"],
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    next_obs=next_obs,
                    info=info,
                )
            else:
                transition = FilteredExperience.create(
                    obs=obs,
                    raw_action=info["raw_action"],
                    exec_action=info["exec_action"],
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    next_obs=next_obs,
                    info=info,
                )

            buffer.add(transition)
            obs = next_obs
            if done:
                obs, _ = env.reset()

            if step >= args.update_after and step % args.update_every == 0:
                batch = buffer.sample(args.batch_size, to_jax=True)
                key, update_key = jax.random.split(key)
                agent.update(update_key, batch)

            if step % args.eval_interval == 0:
                eval_metrics = evaluate_policy(agent, args.algo, args.ref_type, args.eval_episodes, args.seed + step)
                writer.writerow({"step": step, **eval_metrics})
                f.flush()
                agent.save(str(ckpt_path))


if __name__ == "__main__":
    main()
