import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw_scene(ax):
    ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, color='k', lw=1.2))
    ax.add_patch(plt.Circle((0, 0), 0.88, fill=False, color='k', lw=1.0, ls='--'))
    ax.scatter([2.6], [0.0], c='tab:green', marker='o', s=40, label='start')
    ax.scatter([-2.6], [0.0], c='tab:red', marker='*', s=80, label='goal')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal', adjustable='box')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', required=True)
    p.add_argument('--base_dir', default='results/obstacle')
    p.add_argument('--out_dir', default='figures')
    args = p.parse_args()

    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs = axs.reshape(-1)
    for ax, m in zip(axs, args.methods):
        data = np.load(Path(args.base_dir) / m / 'rollouts.npz')
        pos = data['positions']
        for i in range(min(20, pos.shape[0])):
            ax.plot(pos[i, :, 0], pos[i, :, 1], alpha=0.6, lw=1.0)
        draw_scene(ax)
        ax.set_title(m)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / 'obstacle_rollouts.png', dpi=220)
    fig.savefig(Path(args.out_dir) / 'obstacle_rollouts.pdf')


if __name__ == '__main__':
    main()
