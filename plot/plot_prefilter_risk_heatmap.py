import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out_dir', default='figures')
    args = p.parse_args()

    data = np.load(args.input)
    px, py, V = data['px_grid'], data['py_grid'], data['Vp_grid']
    X, Y = np.meshgrid(px, py)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(V, origin='lower', extent=[px[0], px[-1], py[0], py[-1]], aspect='auto', cmap='magma')
    ax.contour(X, Y, V, levels=[0.10, 0.30, 0.60], colors='w', linewidths=0.8)
    ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, color='c', lw=1.2))
    ax.add_patch(plt.Circle((0, 0), 0.88, fill=False, color='c', lw=1.0, ls='--'))
    ax.scatter([2.6], [0.0], c='lime', s=50)
    ax.scatter([-2.6], [0.0], c='red', s=70, marker='*')
    fig.colorbar(im, ax=ax)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / 'prefilter_feasibility_risk_heatmap.png', dpi=220)
    fig.savefig(out_dir / 'prefilter_feasibility_risk_heatmap.pdf')


if __name__ == '__main__':
    main()
