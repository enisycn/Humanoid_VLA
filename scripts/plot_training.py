from pathlib import Path
import argparse,csv
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=list(csv.DictReader(a.csv.open()));x=[int(v['step'])for v in rows]
fig,axes=plt.subplots(1,2,figsize=(9,3.1),layout='constrained')
for key,label in [('BC24','All 24 contexts'),('new6_BC','New 6 contexts')]:axes[0].plot(x,[float(v[key])for v in rows],label=label)
axes[0].set_title('Supervised training loss');axes[0].set_ylabel('Flow matching loss');axes[0].legend(frameon=False)
axes[1].plot(x,[float(v['KD_old18'])for v in rows],color='#986a35');axes[1].set_title('Replay teacher matching');axes[1].set_ylabel('Velocity field MSE (18 contexts)')
for ax in axes:ax.set_xlabel('Update');ax.grid(alpha=.2)
fig.savefig(a.out,dpi=170)
