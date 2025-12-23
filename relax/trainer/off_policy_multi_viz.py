import argparse
import time
import jax
from functools import partial

# --- Imports for overriding setup and finish ---
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams  # <--- Added for finish()
from tqdm import tqdm
# -----------------------------------------------

from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.buffer import TreeBuffer
from scripts.experience import Experience
from relax.trainer.off_policy import OffPolicyTrainer, create_iter_key_fn

# Algorithms
from relax.algorithm.sac import SAC
from relax.network.sac import create_sac_net

# Custom Env & Viz
try:
    from relax.env.multi_goal.multi_goal_env import MultiGoalEnv
    from relax.utils.viz_utils import MultiGoalVisualizer
except ImportError:
    from multi_goal_env import MultiGoalEnv
    from viz_utils import MultiGoalVisualizer


class VisualizingTrainer(OffPolicyTrainer):
    """
    A custom trainer extending OffPolicyTrainer.
    Overrides setup() and finish() to disable the default subprocess evaluator.
    """

    def __init__(self, visualizer, visualize_every=5000, *args, **kwargs):
        kwargs['evaluate_env'] = None
        super().__init__(*args, **kwargs)
        self.visualizer = visualizer
        self.visualize_every = visualize_every
        self.evaluator = None

    def setup(self, dummy_data: Experience):
        """Override setup to skip creating the subprocess evaluator."""
        self.algorithm.warmup(dummy_data)
        self.logger = SummaryWriter(str(self.log_path))
        self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None, dynamic_ncols=True)
        self.algorithm.save_policy_structure(self.log_path, dummy_data.obs[0])

        # Explicitly disable Evaluator
        self.evaluator = None
        print("[VisualizingTrainer] Subprocess evaluator disabled.")

    def finish(self):
        """
        Override finish to safely close resources without assuming evaluator exists.
        """
        self.env.close()
        self.algorithm.save(self.log_path / "state.pkl")

        # Log hparams if they exist (copied from parent logic)
        if self.hparams is not None and len(self.last_metrics) > 0:
            exp, ssi, sei = hparams(self.hparams, self.last_metrics)
            self.logger.file_writer.add_summary(exp)
            self.logger.file_writer.add_summary(ssi)
            self.logger.file_writer.add_summary(sei)

        self.logger.close()
        self.progress.close()

        # [Fix] Only close evaluator if it actually exists
        if self.evaluator:
            self.evaluator.stdin.close()
            self.evaluator.wait()

    def train(self, key: jax.Array):
        key, warmup_key = jax.random.split(key)
        obs, _ = self.env.reset()
        obs = self.warmup(warmup_key, obs)
        iter_key_fn = create_iter_key_fn(key, self.sample_per_iteration, self.update_per_iteration)
        sl, ul = self.sample_log, self.update_log

        self.progress.unpause()

        while sl.sample_step <= self.total_step:
            sample_keys, update_keys = iter_key_fn(sl.sample_step)

            for i in range(self.sample_per_iteration):
                obs = self.sample(sample_keys[i], obs)

            for i in range(self.update_per_iteration):
                self.update(update_keys[i])

            if self.save_policy_interval.check(sl.sample_step):
                policy_pkl_name = self.policy_pkl_template.format(
                    sample_step=sl.sample_step,
                    update_step=ul.update_step,
                )
                self.algorithm.save_policy(self.log_path / policy_pkl_name)

                if self.evaluator:
                    command = f"{sl.sample_step},{self.log_path / policy_pkl_name}\n"
                    self.evaluator.stdin.write(command.encode())

            # Visualization Logic
            if ul.update_step > 0 and ul.update_step % self.visualize_every == 0:
                viz_key = jax.random.fold_in(key, ul.update_step)
                self.visualizer.run(ul.update_step, viz_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sac")
    parser.add_argument("--total_step", type=int, default=int(1e5))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--visualize_every", type=int, default=2000)
    args = parser.parse_args()

    env = MultiGoalEnv(render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2 ** 32 - 1, 3)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    hidden_sizes = [args.hidden_dim, args.hidden_dim]
    gelu = partial(jax.nn.gelu, approximate=False)

    if args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)
    else:
        raise ValueError(f"Algorithm {args.alg} setup not implemented in this snippet.")

    exp_dir = PROJECT_ROOT / "logs" / "MultiGoal" / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S"))

    visualizer = MultiGoalVisualizer(env, algorithm, exp_dir)

    trainer = VisualizingTrainer(
        visualizer=visualizer,
        visualize_every=args.visualize_every,
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=1000,
        total_step=args.total_step,
        sample_per_iteration=1,
        log_path=exp_dir,
        save_policy_every=10000,
        evaluate_n_episode=0
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, 1))

    print(f"Starting training with visualization every {args.visualize_every} steps...")
    trainer.run(train_key)
