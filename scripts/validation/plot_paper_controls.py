"""Plot executed RF/resource/RET controls from an already validated trace."""
import argparse
import json
import os
from pathlib import Path


def plot(case, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('/tmp/paper_matplotlib_cache')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    assert json.loads((case/'paper_validation.json').read_text())['passed']
    rows=[json.loads(line) for line in (case/'rt_sys_trace.jsonl').open()]
    slots=[r['slot_id'] for r in rows]
    cell='cell_0'
    budget=[r['cells'][cell]['budget_w'] for r in rows]
    used=[r['cells'][cell]['average_used_w'] for r in rows]
    profile=[sum(r['paper_interfaces']['runtime_config'][cell]['prb_power_w']) if r['paper_interfaces']['runtime_config'][cell]['prb_power_w'] is not None else np.nan for r in rows]
    prbs=[]
    for r in rows:
        occupied=set()
        for u in r['ues']:
            if u['cell_id']==cell and u['resource_allocation']['scheduled']:occupied.update(u['resource_allocation']['prbs'])
        prbs.append(len(occupied))
    tilt=[r['paper_interfaces']['runtime_config'][cell]['ret_tilt_deg'] if r['paper_interfaces']['runtime_config'][cell]['ret_enabled'] else np.nan for r in rows]
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,layout='constrained')
    axes[0].step(slots,budget,where='post',label='Cell RF budget',color='0.5',linestyle='--')
    axes[0].step(slots,profile,where='post',label='Absolute PRB profile sum',color='#c76819')
    axes[0].step(slots,used,where='post',label='Actually emitted RF',color='#1673a5')
    axes[0].set(ylabel='Slot-average power (W)',ylim=(-.03,1.08));axes[0].legend(loc='upper right',fontsize=8,ncol=3)
    axes[1].step(slots,prbs,where='post',color='#1673a5');axes[1].set(ylabel='Occupied physical PRBs',ylim=(-1,25))
    axes[2].step(slots,tilt,where='post',color='#7c4ba5');axes[2].set(ylabel='RET enabled angle (deg)',xlabel='Completed slot',ylim=(-.5,15.5))
    for ax in axes:
        ax.grid(alpha=.2)
        for x in (20,50,80,100):ax.axvline(x,color='0.6',linewidth=.7,linestyle=':')
    axes[0].set_title(case.name+' | cell_0 | actual RT/NR trace',fontsize=11)
    axes[2].text(.01,.92,'Requested stimuli: 20 profile, 50 RET, 80 over-budget rejection, 100 clear profile',transform=axes[2].transAxes,fontsize=8,va='top')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.case,a.output)
