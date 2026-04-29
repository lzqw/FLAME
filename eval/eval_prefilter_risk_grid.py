import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()
    px_grid = np.linspace(-3.5, 3.5, 300)
    py_grid = np.linspace(-2.0, 2.0, 180)
    Vp_grid = np.zeros((len(py_grid), len(px_grid)), dtype=np.float32)
    p = Path(args.save_dir)
    p.mkdir(parents=True, exist_ok=True)
    np.savez(p / 'prefilter_risk_grid.npz', px_grid=px_grid, py_grid=py_grid, Vp_grid=Vp_grid)


if __name__ == '__main__':
    main()
