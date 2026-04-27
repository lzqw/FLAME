#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True, help=".npy with shape [N,2]")
    parser.add_argument("--output", type=Path, default=Path("occupancy_heatmap.png"))
    parser.add_argument("--bins-x", type=int, default=70)
    parser.add_argument("--bins-y", type=int, default=40)
    args = parser.parse_args()

    positions = np.load(args.positions)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    h = ax.hist2d(
        positions[:, 0],
        positions[:, 1],
        bins=[args.bins_x, args.bins_y],
        range=[[-3.5, 3.5], [-2.0, 2.0]],
        cmap="magma",
    )
    plt.colorbar(h[3], ax=ax, label="count")

    obstacle = plt.Circle((0.0, 0.0), 0.8, color="cyan", fill=False, linewidth=1.5)
    ax.add_patch(obstacle)
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-2.0, 2.0])
    ax.set_aspect("equal")
    ax.set_title("State Occupancy Heatmap")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
