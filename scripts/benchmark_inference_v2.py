import time
import argparse
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym

# -----------------------------------------------------------------------------
# 导入所需的算法类
# -----------------------------------------------------------------------------
from relax.algorithm.sac import SAC
from relax.algorithm.dacer import DACER
from relax.algorithm.qsm import QSM
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.algorithm.sdac import SDAC
from relax.algorithm.rf2_sac_ent import RF2SACENT
from relax.algorithm.mf2_sac_ent2 import MF2SACENT2

# -----------------------------------------------------------------------------
# 导入网络创建函数
# -----------------------------------------------------------------------------
from relax.network.sac import create_sac_net
from relax.network.dacer import create_dacer_net
from relax.network.qsm import create_qsm_net
from relax.network.dipo import create_dipo_net
from relax.network.sdac import create_sdac_net
from relax.network.qvpo import create_qvpo_net
from relax.network.rf2_sac_ent import create_rf2_sac_ent_net
from relax.network.mf2_sac_ent2 import create_mf2_sac_ent2_net

# -----------------------------------------------------------------------------
# 导入 Buffer (DIPO 需要)
# -----------------------------------------------------------------------------
from relax.buffer import TreeBuffer
from scripts.experience import ObsActionPair


def get_args():
    parser = argparse.ArgumentParser()
    # 默认环境设为 Ant-v5 (如果无法加载，用户可自行指定 Ant-v4)
    parser.add_argument("--env", type=str, default="Ant-v5", help="Environment name, e.g., Ant-v5 or Ant-v4")
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=0.005)
    parser.add_argument("--delay_alpha_update", type=float, default=20)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=1)
    parser.add_argument("--target_entropy_scale", type=float, default=3.0)
    parser.add_argument("--use_ema_policy", default=True, action="store_true")
    parser.add_argument("--sample_k", type=int, default=300)  # 用于 rf2/mf2

    # 测试参数
    parser.add_argument("--n_warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--n_iter", type=int, default=100, help="Benchmark iterations")
    return parser.parse_args()


# 激活函数定义
def mish(x: jax.Array):
    return x * jnp.tanh(jax.nn.softplus(x))


gelu = partial(jax.nn.gelu, approximate=False)


def init_algorithm(alg_name, args, obs_dim, act_dim, key):
    """
    根据算法名称和特定要求初始化算法
    """
    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    # -------------------------------------------------------------------------
    # 配置 Diffusion Steps
    # 要求:
    #   qsm, dacer, sdac, dipo, qvpo -> 20
    #   rf2_sac_ent, mf2_sac_ent2 -> 1
    # -------------------------------------------------------------------------
    if alg_name in ["qsm", "dacer", "sdac", "dipo", "qvpo"]:
        steps = 20
        steps_test = 20
    elif alg_name in ["rf2_sac_ent", "mf2_sac_ent2"]:
        steps = 1
        steps_test = 1
    else:
        steps = 0  # SAC 不需要
        steps_test = 0

    algorithm = None

    if alg_name == 'qsm':
        # QSM: (原代码 hardcode 为 20，这里传入变量)
        agent, params = create_qsm_net(key, obs_dim, act_dim, hidden_sizes,
                                       num_timesteps=steps,
                                       num_particles=args.num_particles)
        algorithm = QSM(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)

    elif alg_name == 'dacer':
        # DACER:
        agent, params = create_dacer_net(key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                         num_timesteps=steps)
        algorithm = DACER(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)

    elif alg_name == 'sdac':
        # SDAC:
        agent, params = create_sdac_net(key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                        num_timesteps=steps,
                                        num_particles=args.num_particles,
                                        noise_scale=args.noise_scale,
                                        target_entropy_scale=args.target_entropy_scale)
        algorithm = SDAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                         delay_alpha_update=args.delay_alpha_update,
                         lr_schedule_end=args.lr_schedule_end,
                         use_ema=args.use_ema_policy)

    elif alg_name == 'dipo':
        # DIPO: 需要一个 dummy buffer
        dummy_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            size=1000, seed=0, remove_batch_dim=False
        )
        # train_antmaze.py 默认为 100，这里我们设为 20
        agent, params = create_dipo_net(key, obs_dim, act_dim, hidden_sizes, num_timesteps=steps)
        algorithm = DIPO(agent, params, dummy_buffer, lr=args.lr,
                         action_gradient_steps=30, policy_target_delay=2, action_grad_norm=0.16)

    elif alg_name == 'qvpo':
        # QVPO:
        agent, params = create_qvpo_net(key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                        num_timesteps=steps,
                                        num_particles=args.num_particles,
                                        noise_scale=args.noise_scale)
        algorithm = QVPO(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update)

    elif alg_name == 'rf2_sac_ent':
        # RF2_SAC_ENT: steps=1
        agent, params = create_rf2_sac_ent_net(key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                               num_timesteps=steps,
                                               num_timesteps_test=steps_test,
                                               num_particles=args.num_particles,
                                               noise_scale=args.noise_scale,
                                               target_entropy_scale=args.target_entropy_scale,
                                               alpha_value=0.01, fixed_alpha=False, init_alpha=1.0)  # 使用默认参数
        algorithm = RF2SACENT(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                              delay_alpha_update=args.delay_alpha_update,
                              lr_schedule_end=args.lr_schedule_end,
                              use_ema=args.use_ema_policy,
                              sample_k=args.sample_k, alpha_value=0.01, fixed_alpha=False)

    elif alg_name == 'mf2_sac_ent2':
        # MF2_SAC_ENT2: steps=1
        agent, params = create_mf2_sac_ent2_net(key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                                num_timesteps=steps,
                                                num_timesteps_test=steps_test,
                                                num_particles=args.num_particles,
                                                noise_scale=args.noise_scale,
                                                target_entropy_scale=args.target_entropy_scale,
                                                alpha_value=0.01, fixed_alpha=False, init_alpha=1.0)
        algorithm = MF2SACENT2(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                               delay_alpha_update=args.delay_alpha_update,
                               lr_schedule_end=args.lr_schedule_end,
                               use_ema=args.use_ema_policy,
                               sample_k=args.sample_k, alpha_value=0.01, fixed_alpha=False)

    elif alg_name == 'sac':
        # SAC
        agent, params = create_sac_net(key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)

    return algorithm, steps


def main():
    args = get_args()

    # -------------------------------------------------------------------------
    # 环境初始化 (获取 Obs/Act Dim)
    # -------------------------------------------------------------------------
    try:
        env = gym.make(args.env)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        print(f"Loaded Environment: {args.env} | Obs: {obs_dim}, Act: {act_dim}")
        env.close()
    except Exception as e:
        print(f"Warning: Could not load env '{args.env}' (Error: {e}).")
        print("Falling back to default Ant dimensions (Obs: 27, Act: 8) for testing.")
        obs_dim = 27
        act_dim = 8

    # -------------------------------------------------------------------------
    # 待测试算法列表
    # -------------------------------------------------------------------------
    target_algos = [
        "qsm", "dacer", "sdac", "dipo", "qvpo",
        "rf2_sac_ent", "mf2_sac_ent2",
        "sac"
    ]

    print("\n" + "=" * 95)
    print(f"{'Algorithm':<15} | {'Steps':<5} | {'Type':<15} | {'Mean (ms)':<10} | {'Std (ms)':<10} | {'FPS':<10}")
    print("=" * 95)

    rng = jax.random.PRNGKey(0)

    for alg_name in target_algos:
        rng, key = jax.random.split(rng)

        try:
            # 1. 初始化算法
            algo, steps_val = init_algorithm(alg_name, args, obs_dim, act_dim, key)

            # 2. 准备输入数据
            dummy_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
            action_key = jax.random.PRNGKey(1)

            # -----------------------------------------------------------------
            # 测试 1: Stochastic Policy (get_action)
            # -----------------------------------------------------------------

            # JIT 编译 & Warmup
            # 第一次运行会触发 JAX 编译
            _ = algo.get_action(action_key, dummy_obs)
            for _ in range(args.n_warmup):
                _ = algo.get_action(action_key, dummy_obs)

            # 计时循环
            times = []
            for _ in range(args.n_iter):
                start = time.perf_counter()

                # [Fix] get_action 返回的是 numpy array，已经完成了同步，不需要 block_until_ready
                res = algo.get_action(action_key, dummy_obs)

                end = time.perf_counter()
                times.append((end - start) * 1000)  # 转为毫秒

            mean_t = np.mean(times)
            std_t = np.std(times)
            fps = 1000.0 / mean_t

            print(
                f"{alg_name:<15} | {steps_val:<5} | {'Stochastic':<15} | {mean_t:<10.3f} | {std_t:<10.3f} | {fps:<10.1f}")

            # -----------------------------------------------------------------
            # 测试 2: Deterministic Policy (get_deterministic_action)
            # -----------------------------------------------------------------

            # JIT 编译 & Warmup
            _ = algo.get_deterministic_action(dummy_obs)
            for _ in range(args.n_warmup):
                _ = algo.get_deterministic_action(dummy_obs)

            # 计时循环
            times_det = []
            for _ in range(args.n_iter):
                start = time.perf_counter()

                # [Fix] 同样不需要 block_until_ready
                res = algo.get_deterministic_action(dummy_obs)

                end = time.perf_counter()
                times_det.append((end - start) * 1000)

            mean_t_det = np.mean(times_det)
            std_t_det = np.std(times_det)
            fps_det = 1000.0 / mean_t_det

            print(
                f"{'':<15} | {steps_val:<5} | {'Deterministic':<15} | {mean_t_det:<10.3f} | {std_t_det:<10.3f} | {fps_det:<10.1f}")
            print("-" * 95)

        except Exception as e:
            print(f"{alg_name:<15} | ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            print("-" * 95)


if __name__ == "__main__":
    main()
