import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_npz", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="figures")
    args = parser.parse_args()

    data = np.load(args.rollout_npz)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(len(data["p"]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, data["p"], label="p")
    axes[0].plot(t, data["p_ref"], "--", label="p_ref")
    axes[0].set_ylabel("position")
    axes[0].legend()

    axes[1].plot(t, data["v"], label="v")
    axes[1].plot(t, data["v_ref"], "--", label="v_ref")
    axes[1].set_ylabel("velocity")
    axes[1].set_xlabel("step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_dir / "closed_loop_tracking.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, data["raw_u"], label="raw_u")
    axes[0].plot(t, data["exec_u"], label="exec_u")
    axes[0].plot(t, data["safe_low_u"], "--", label="safe_low")
    axes[0].plot(t, data["safe_high_u"], "--", label="safe_high")
    axes[0].set_ylabel("u")
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].step(t, data["filter_active"], where="post", label="filter_active")
    axes[1].set_ylabel("active")
    axes[1].set_xlabel("step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_dir / "filter_behavior.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, data["projection_gap"], label="projection_gap")
    ax.axhline(0.0, color="k", linewidth=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("raw_u - exec_u")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "projection_gap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
