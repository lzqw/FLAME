import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--algo', required=True)
    parser.add_argument('--eval_episodes', type=int, default=200)
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'summary.json').write_text(json.dumps({'algo': args.algo, 'eval_episodes': args.eval_episodes}, indent=2))


if __name__ == '__main__':
    main()
