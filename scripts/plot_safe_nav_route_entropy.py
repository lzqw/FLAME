#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--step-col", type=str, default="step")
    parser.add_argument("--entropy-col", type=str, default="route_entropy")
    parser.add_argument("--output", type=Path, default=Path("route_entropy_curve.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df[args.step_col], df[args.entropy_col], color="tab:purple", linewidth=2)
    ax.set_xlabel(args.step_col)
    ax.set_ylabel(args.entropy_col)
    ax.set_title("Route Entropy During Training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
