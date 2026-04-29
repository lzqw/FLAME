import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_json', required=True)
    p.add_argument('--out_prefix', required=True)
    p.add_argument('--state_idx', type=int, default=0)
    args = p.parse_args()

    data = json.loads(Path(args.input_json).read_text())
    rec = data['records'][args.state_idx]
    grid = np.asarray(rec['grid_actions'])
    safe_mask = np.asarray(rec['safe_mask'])
    qp = np.asarray(rec['qp_heatmap'])
    raw = np.asarray(rec['raw_samples'])
    proj = np.asarray(rec['projected_samples'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sc0 = axes[0].scatter(grid[:, 0], grid[:, 1], c=safe_mask, s=12, cmap='coolwarm', vmin=0, vmax=1)
    axes[0].set_title('Safe action mask')
    plt.colorbar(sc0, ax=axes[0], label='safe(1)/unsafe(0)')

    sc1 = axes[1].scatter(grid[:, 0], grid[:, 1], c=qp, s=12, cmap='viridis')
    axes[1].scatter(raw[:, 0], raw[:, 1], s=4, alpha=0.25, c='tab:blue', label='raw')
    axes[1].scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.25, c='tab:orange', label='projected')
    axes[1].set_title('Qp heatmap + samples')
    axes[1].legend(loc='upper right')
    plt.colorbar(sc1, ax=axes[1], label='Q_S(s,a)')

    for ax in axes:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('action x')
        ax.set_ylabel('action y')

    fig.suptitle(f"state={rec['state']}, feasible_ratio={rec['feasible_raw_action_ratio']:.3f}, entropy={rec['action_route_entropy']:.3f}")
    plt.tight_layout()
    plt.savefig(f'{args.out_prefix}.png', dpi=200)
    plt.savefig(f'{args.out_prefix}.pdf')


if __name__ == '__main__':
    main()
