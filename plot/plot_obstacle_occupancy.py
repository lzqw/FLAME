import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def entropy_from_hist(H):
    rho = H / (H.sum() + 1e-8)
    nz = rho > 0
    return float(-np.sum(rho[nz] * np.log(rho[nz] + 1e-8)))


def boundary_coverage(H):
    xedges = np.linspace(-3.5, 3.5, H.shape[0] + 1)
    yedges = np.linspace(-2.0, 2.0, H.shape[1] + 1)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    dist = np.sqrt(Xc ** 2 + Yc ** 2)
    mask = (dist >= 0.88) & (dist <= 1.23)
    visited = H > 0
    return float(np.sum(visited & mask) / (np.sum(mask) + 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', required=True)
    p.add_argument('--base_dir', default='results/obstacle')
    p.add_argument('--out_dir', default='figures')
    args = p.parse_args()

    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs = axs.reshape(-1)
    rows = []

    for ax, m in zip(axs, args.methods):
        pos = np.load(Path(args.base_dir) / m / 'rollouts.npz')['positions'].reshape(-1, 2)
        H, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=[140, 80], range=[[-3.5, 3.5], [-2.0, 2.0]])
        Hn = (H.T / (H.max() + 1e-8))
        ax.imshow(Hn, origin='lower', extent=[-3.5, 3.5, -2.0, 2.0], aspect='auto')
        ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, color='w', lw=1.0))
        ax.add_patch(plt.Circle((0, 0), 0.88, fill=False, color='w', lw=1.0, ls='--'))
        ax.set_title(m)
        rows.append({'method': m, 'occupancy_entropy': entropy_from_hist(H), 'boundary_coverage_ratio': boundary_coverage(H)})

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / 'obstacle_occupancy_heatmaps.png', dpi=220)
    fig.savefig(Path(args.out_dir) / 'obstacle_occupancy_heatmaps.pdf')

    metrics_path = Path(args.base_dir) / 'occupancy_metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'occupancy_entropy', 'boundary_coverage_ratio'])
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
