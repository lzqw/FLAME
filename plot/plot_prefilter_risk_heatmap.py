import argparse, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--out_dir', default='figures'); args=p.parse_args()
    d=np.load(args.input); X,Y=np.meshgrid(d['px_grid'],d['py_grid']); V=d['Vp_grid']
    fig,ax=plt.subplots(figsize=(8,4)); im=ax.imshow(V,origin='lower',extent=[d['px_grid'][0],d['px_grid'][-1],d['py_grid'][0],d['py_grid'][-1]],aspect='auto'); ax.contour(X,Y,V,levels=[0.1,0.3,0.6],colors='w',linewidths=0.8)
    fig.colorbar(im,ax=ax); Path(args.out_dir).mkdir(parents=True,exist_ok=True); fig.savefig(Path(args.out_dir)/'prefilter_feasibility_risk_heatmap.png',dpi=200); fig.savefig(Path(args.out_dir)/'prefilter_feasibility_risk_heatmap.pdf')

if __name__=='__main__': main()
