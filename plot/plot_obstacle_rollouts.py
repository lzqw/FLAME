import argparse, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--methods', nargs='+', required=True); p.add_argument('--base_dir', default='results/obstacle'); p.add_argument('--out_dir', default='figures'); args=p.parse_args()
    fig,axs=plt.subplots(2,2,figsize=(10,6)); axs=axs.reshape(-1)
    for ax,m in zip(axs,args.methods):
        d=np.load(Path(args.base_dir)/m/'rollouts.npz'); pos=d['positions']
        for i in range(min(20,pos.shape[0])): ax.plot(pos[i,:,0],pos[i,:,1],alpha=0.5)
        ax.add_patch(plt.Circle((0,0),0.8,color='k',fill=False)); ax.add_patch(plt.Circle((0,0),0.88,color='k',fill=False,ls='--')); ax.set_title(m); ax.set_xlim(-3.5,3.5); ax.set_ylim(-2,2)
    Path(args.out_dir).mkdir(parents=True,exist_ok=True); fig.tight_layout(); fig.savefig(Path(args.out_dir)/'obstacle_rollouts.png',dpi=200); fig.savefig(Path(args.out_dir)/'obstacle_rollouts.pdf')

if __name__=='__main__': main()
