import argparse, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--methods', nargs='+', required=True); p.add_argument('--base_dir', default='results/obstacle'); p.add_argument('--out_dir', default='figures'); args=p.parse_args()
    fig,axs=plt.subplots(2,2,figsize=(10,6)); axs=axs.reshape(-1)
    for ax,m in zip(axs,args.methods):
        pos=np.load(Path(args.base_dir)/m/'rollouts.npz')['positions'].reshape(-1,2)
        H,xe,ye=np.histogram2d(pos[:,0],pos[:,1],bins=[140,80],range=[[-3.5,3.5],[-2,2]])
        ax.imshow((H.T/(H.max()+1e-8)),origin='lower',extent=[-3.5,3.5,-2,2],aspect='auto'); ax.set_title(m)
    Path(args.out_dir).mkdir(parents=True,exist_ok=True); fig.tight_layout(); fig.savefig(Path(args.out_dir)/'obstacle_occupancy_heatmaps.png',dpi=200); fig.savefig(Path(args.out_dir)/'obstacle_occupancy_heatmaps.pdf')

if __name__=='__main__': main()
