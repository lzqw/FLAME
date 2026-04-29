import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_json', required=True)
    p.add_argument('--out_prefix', required=True)
    args = p.parse_args()

    data = json.loads(Path(args.input_json).read_text())
    rec = data['records'][0]
    raw = np.asarray(rec['raw_samples'])
    residuals = np.asarray(rec['residuals'])

    plt.figure(figsize=(5, 4))
    sc = plt.scatter(raw[:, 0], raw[:, 1], c=residuals, s=8, cmap='viridis')
    plt.colorbar(sc, label='projection residual')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('raw action x')
    plt.ylabel('raw action y')
    plt.title('Action Feasibility Heatmap')
    plt.tight_layout()
    plt.savefig(f'{args.out_prefix}.png', dpi=200)
    plt.savefig(f'{args.out_prefix}.pdf')

if __name__ == '__main__':
    main()
