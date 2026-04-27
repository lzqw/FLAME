#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rollouts(path: Path):
    data = json.loads(path.read_text())
    return [np.asarray(traj, dtype=np.float32) for traj in data["trajectories"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True, help="JSON file with key 'trajectories'.")
    parser.add_argument("--output", type=Path, default=Path("scene_rollouts.png"))
    args = parser.parse_args()

    trajectories = load_rollouts(args.rollouts)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-2.0, 2.0])
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title("Safe Obstacle Navigation Rollouts")

    obstacle = plt.Circle((0.0, 0.0), 0.8, color="tab:red", alpha=0.35, label="obstacle")
    tightened = plt.Circle((0.0, 0.0), 0.88, color="tab:red", fill=False, linestyle="--", label="tightened")
    ax.add_patch(obstacle)
    ax.add_patch(tightened)
    ax.plot([2.6], [0.0], "go", label="start nominal")
    ax.plot([-2.6], [0.0], "b*", markersize=12, label="goal")

    for traj in trajectories:
        if len(traj) == 0:
            continue
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.5)

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
