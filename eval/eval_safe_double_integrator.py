import argparse
import csv
import json
from pathlib import Path

import jax
import numpy as np

from envs.safe_double_integrator import SafeDoubleIntegratorEnv
from relax.algorithm.mf_sac import MFSAC
from relax.algorithm.safe_mf_sac import SafeMFSAC
from relax.network.mf_sac import create_mf_sac_net
from relax.network.safe_mf_sac import create_safe_mf_sac_net


def noise_tuple(level: str):
    if level == "nominal":
        return (0.01, 0.05)
    if level == "medium":
        return (0.02, 0.08)
    if level == "high":
        return (0.03, 0.10)
    raise ValueError(level)


def build_agent(algo_name: str, obs_dim: int, act_dim: int, seed: int):
    key = jax.random.PRNGKey(seed)
    hidden_sizes = [256, 256, 256]
    diffusion_hidden_sizes = [256, 256, 256]
    if algo_name == "mf_no_filter":
        net, params = create_mf_sac_net(key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes)
        return MFSAC(net, params)
    net, params = create_safe_mf_sac_net(key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes)
    return SafeMFSAC(net, params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--algo", type=str, choices=["mf_no_filter", "mf_filter", "safe_mf"], required=True)
    parser.add_argument("--ref_type", type=str, choices=["step", "sine", "piecewise_sine"], default="piecewise_sine")
    parser.add_argument("--noise_level", type=str, choices=["nominal", "medium", "high"], default="nominal")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sigma = noise_tuple(args.noise_level)
    env = SafeDoubleIntegratorEnv(
        ref_type=args.ref_type,
        noise_sigma=sigma,
        use_filter=(args.algo != "mf_no_filter"),
        seed=args.seed,
    )
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    agent = build_agent(args.algo, obs_dim, act_dim, args.seed)
    agent.load(args.checkpoint)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    per_episode_metrics = []
    episode_logs = []
    key = jax.random.PRNGKey(args.seed + 1234)

    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        log = {k: [] for k in [
            "obs", "p", "v", "p_ref", "v_ref", "raw_action", "exec_action", "raw_u", "exec_u", "safe_low_u", "safe_high_u",
            "reward", "state_violation", "safe_violation", "filter_active", "projection_gap", "noise_p", "noise_v"
        ]}

        while not done:
            key, akey = jax.random.split(key)
            action = agent.get_action(akey, obs[None, :])[0]
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            log["obs"].append(obs.copy())
            log["p"].append(float(obs[0]))
            log["v"].append(float(obs[1]))
            log["p_ref"].append(float(info["p_ref"]))
            log["v_ref"].append(float(info["v_ref"]))
            log["raw_action"].append(float(info["raw_action"][0]))
            log["exec_action"].append(float(info["exec_action"][0]))
            log["raw_u"].append(float(info["raw_u"]))
            log["exec_u"].append(float(info["exec_u"]))
            log["safe_low_u"].append(float(info["safe_low_u"]))
            log["safe_high_u"].append(float(info["safe_high_u"]))
            log["reward"].append(float(reward))
            log["state_violation"].append(float(info["state_violation"]))
            log["safe_violation"].append(float(info["safe_violation"]))
            log["filter_active"].append(float(info["filter_active"]))
            log["projection_gap"].append(float(info["projection_gap"]))
            log["noise_p"].append(float(info["noise_p"]))
            log["noise_v"].append(float(info["noise_v"]))

            obs = next_obs

        p = np.asarray(log["p"])
        p_ref = np.asarray(log["p_ref"])
        exec_u = np.asarray(log["exec_u"])
        sv = np.asarray(log["state_violation"])
        far = np.asarray(log["filter_active"])
        gap = np.asarray(log["projection_gap"])
        reward_arr = np.asarray(log["reward"])

        metric = {
            "episode": ep,
            "VR": float(np.mean(sv)),
            "ZVS": float(np.mean(sv) == 0.0),
            "e_RMS": float(np.sqrt(np.mean((p - p_ref) ** 2))),
            "e_max": float(np.max(np.abs(p - p_ref))),
            "J_u": float(np.mean(exec_u ** 2)),
            "J_du": float(np.mean(np.diff(exec_u) ** 2)) if len(exec_u) > 1 else 0.0,
            "R_ep": float(np.sum(reward_arr)),
            "FAR": float(np.mean(far)),
            "APR": float(np.mean(np.abs(gap))),
        }
        per_episode_metrics.append(metric)
        episode_logs.append({k: np.asarray(v) for k, v in log.items()})

    fieldnames = list(per_episode_metrics[0].keys())
    with (save_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_episode_metrics)

    summary = {
        f"{name}_mean": float(np.mean([m[name] for m in per_episode_metrics]))
        for name in ["VR", "ZVS", "e_RMS", "e_max", "R_ep", "J_u", "J_du", "FAR", "APR"]
    }
    summary.update({
        f"{name}_std": float(np.std([m[name] for m in per_episode_metrics]))
        for name in ["VR", "ZVS", "e_RMS", "e_max", "R_ep", "J_u", "J_du", "FAR", "APR"]
    })

    with (save_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        save_dir / "episodes.npz",
        episodes=np.asarray(episode_logs, dtype=object),
        metrics=np.asarray(per_episode_metrics, dtype=object),
    )


if __name__ == "__main__":
    main()
