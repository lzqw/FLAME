"""Training entrypoint for safe obstacle navigation experiments."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='safe_pullback_rf2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=1000000)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--update_after', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--noise_sigma_x', type=float, default=0.01)
    parser.add_argument('--noise_sigma_y', type=float, default=0.01)
    parser.add_argument('--log_dir', type=str, default='logs/obstacle/default')
    args = parser.parse_args()
    print('Scaffold ready for', args.algo)


if __name__ == '__main__':
    main()
