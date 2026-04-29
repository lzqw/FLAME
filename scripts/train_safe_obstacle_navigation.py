import argparse
import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from relax.algorithm.safe_pullback_rf2_sac_ent import SafePullbackRF2SACENT
from relax.network.safe_pullback_rf2_sac_ent import create_safe_pullback_rf2_sac_ent_net
from scripts.safe_pullback_experience import SafePullbackExperience


class Batch(NamedTuple):
    obs: jnp.ndarray
    raw_action: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray
    projection_residual: jnp.ndarray
    projection_cost: jnp.ndarray


def goal_controller(state, goal):
    d = goal - state
    return np.clip(d / (np.linalg.norm(d) + 1e-6), -1.0, 1.0).astype(np.float32)


def make_algo(args, obs_dim=8, act_dim=2):
    key = jax.random.PRNGKey(args.seed)
    net, params = create_safe_pullback_rf2_sac_ent_net(
        key, obs_dim, act_dim, hidden_sizes=[256, 256, 256], diffusion_hidden_sizes=[256, 256, 256],
        num_timesteps=args.diffusion_steps, num_ent_timesteps=args.num_ent_timesteps,
        alpha_value=args.alpha_value, fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha,
    )
    return SafePullbackRF2SACENT(
        net, params, gamma=args.gamma, gamma_p=args.gamma_p, lr=args.lr, alpha_lr=args.alpha_lr,
        sample_k=args.sample_k, lambda_p=args.lambda_p, use_projection_critic=args.use_projection_critic,
        fixed_alpha=args.fixed_alpha, alpha_value=args.alpha_value,
    )


def sample_batch(buf, batch_size):
    idx = np.random.randint(0, len(buf), size=batch_size)
    items = [buf[i] for i in idx]
    return Batch(
        obs=jnp.asarray(np.stack([x.obs for x in items])),
        raw_action=jnp.asarray(np.stack([x.raw_action for x in items])),
        action=jnp.asarray(np.stack([x.action for x in items])),
        reward=jnp.asarray(np.stack([x.reward for x in items])),
        done=jnp.asarray(np.stack([x.done for x in items]).astype(np.float32)),
        next_obs=jnp.asarray(np.stack([x.next_obs for x in items])),
        projection_residual=jnp.asarray(np.stack([x.projection_residual for x in items])),
        projection_cost=jnp.asarray(np.stack([x.projection_cost for x in items])),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', default='safe_pullback_rf2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total_steps', type=int, default=200000)
    p.add_argument('--start_steps', type=int, default=10000)
    p.add_argument('--update_after', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--eval_interval', type=int, default=5000)
    p.add_argument('--noise_sigma_x', type=float, default=0.01)
    p.add_argument('--noise_sigma_y', type=float, default=0.01)
    p.add_argument('--log_dir', required=True)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha_lr', type=float, default=1e-2)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gamma_p', type=float, default=0.99)
    p.add_argument('--sample_k', type=int, default=64)
    p.add_argument('--lambda_p', type=float, default=1.0)
    p.add_argument('--use_projection_critic', action='store_true', default=True)
    p.add_argument('--fixed_alpha', action='store_true', default=False)
    p.add_argument('--alpha_value', type=float, default=0.01)
    p.add_argument('--init_alpha', type=float, default=0.01)
    p.add_argument('--diffusion_steps', type=int, default=10)
    p.add_argument('--num_ent_timesteps', type=int, default=10)
    args = p.parse_args()

    np.random.seed(args.seed)
    use_filter = args.algo != 'rf2_no_filter'
    env = SafeObstacleNavigation2DEnv(noise_sigma=(args.noise_sigma_x, args.noise_sigma_y), use_filter=use_filter, seed=args.seed)
    agent = make_algo(args)
    key = jax.random.PRNGKey(args.seed + 7)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    buffer = []
    train_log = []

    obs, _ = env.reset(seed=args.seed)
    for step in range(1, args.total_steps + 1):
        if step < args.start_steps:
            raw_action = env.action_space.sample()
        elif args.algo == 'goal_filter':
            raw_action = goal_controller(env.state, env.goal)
        else:
            key, ak = jax.random.split(key)
            raw_action = np.asarray(agent.get_action(ak, obs[None, :])[0])

        next_obs, reward, terminated, truncated, info = env.step(raw_action)
        exp = SafePullbackExperience.create(obs, raw_action, info['exec_action'], reward, terminated, truncated, next_obs, info)
        buffer.append(exp)
        if len(buffer) > 1_000_000:
            buffer.pop(0)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if step >= args.update_after and len(buffer) >= args.batch_size and args.algo != 'goal_filter':
            key, uk = jax.random.split(key)
            batch = sample_batch(buffer, args.batch_size)
            out = agent.update(uk, batch)
            out['step'] = step
            train_log.append(out)

    with open(log_dir / 'train_metrics.pkl', 'wb') as f:
        pickle.dump(train_log, f)
    with open(log_dir / 'checkpoint.pkl', 'wb') as f:
        pickle.dump({'algo': args.algo, 'seed': args.seed, 'agent_state': agent.state}, f)


if __name__ == '__main__':
    main()
