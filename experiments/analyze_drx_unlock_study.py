"""Summarize matched frozen-policy measurements; no fitted effects are invented."""
import argparse,gzip,hashlib,json,math,sys,tarfile
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ARMS=('base','native_drx','unlocked_drx')
METRICS=[('network_energy_j','UE energy (all 6)','J','min'),('target_energy_j','X-01 target UE energy','J','min'),
         ('network_goodput_mbps','Network DL goodput','Mbit/s','max'),('packet_mean_ms','Mean DL packet latency','ms','min'),
         ('packet_p95_ms','DL packet P95 (mean across seeds)','ms','min'),('task_mean_ms','Mean MEC task latency','ms','min'),
         ('task_on_time_pct','MEC on-time completion','%','max')]


def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checkpoint_source_hash(archive):
    h=hashlib.sha256()
    with tarfile.open(archive,'r:gz')as t:
        for directory in('core','phy','ran','ric','state','models','traffic','experiments','configs'):
            for m in sorted((m for m in t.getmembers()if m.isfile()and m.name.startswith(directory+'/')and Path(m.name).suffix in('.py','.json','.yaml','.yml')),key=lambda m:Path(m.name)):
                h.update(m.name.encode());h.update(t.extractfile(m).read())
    return h.hexdigest()


def metrics(folder,comparison,arm):
    import numpy as np
    final=load(folder/'final_summary.json');cfg=load(folder/'config_snapshot.json');status=load(folder/'run_status.json')
    assert status['completed'] and status['replay']['audits_passed']
    duration=final['total_slots']*final['slot_duration_ms']/1000
    task=({u:v['off']for u,v in comparison['comparisons']['native_drx']['per_ue'].items()}if arm=='base'else
          {u:v['frozen']for u,v in comparison['comparisons'][arm]['per_ue'].items()})
    packet=final['packet_metrics'];count=sum(v['completed_packets']for v in packet.values())
    latencies=[]
    with gzip.open(folder/'packet_events.jsonl.gz','rt')as stream:
        for line in stream:
            row=json.loads(line)
            if row['kind']=='delivery'and row.get('completed'):
                value=row['latency_ms'];assert math.isfinite(value);latencies.append(value)
    assert len(latencies)==count
    sums=sum(v['completion_latency_ms_sum']for v in packet.values())
    assert abs(sum(latencies)-sums)<1e-5
    completed=sum(v['completed']for v in task.values());generated=sum(v['generated']for v in task.values())
    latency_sum=sum(v['completed_task_mean_latency_ms']*v['completed']for v in task.values()if v['completed'])
    on_time=sum(v['on_time_completed']for v in task.values())
    energy=final['ue_services']['per_ue']
    result=dict(network_energy_j=sum(v['ue_energy_total_j']for v in energy.values()),target_energy_j=sum(energy[u]['ue_energy_total_j']for u in('ue_0','ue_4')),
        network_goodput_mbps=sum(v['delivered_bytes']for v in packet.values())*8/duration/1e6,
        packet_mean_ms=sums/count if count else None,packet_p95_ms=float(np.percentile(latencies,95))if count else None,
        task_mean_ms=latency_sum/completed if completed else None,task_on_time_pct=100*on_time/generated if generated else None,
        task_completed=completed,task_generated=generated,task_on_time=on_time,task_latency_sum_ms=latency_sum,
        packet_completed=count,packet_latency_sum_ms=sums,ue_delivered_bytes={u:v['delivered_bytes']for u,v in packet.items()},
        urllc_failures=sum(packet[u]['deadline_failed_packets']+packet[u]['overflow_packets']for u in('ue_2','ue_6')),
        source_sha256=status['source_sha256'],duration_s=duration)
    if arm!='base':
        params=cfg['configs']['ric']['xapps'][0]['parameters'];assert params['deploy_drx']==(arm=='unlocked_drx')and params['mode']=='frozen'
        assert final['ue_energy_xapp']['mode']=='frozen'
        result['actions']=comparison['action_audits'][arm]['chosen_actions']
        result['executed_drx_profiles']=comparison['action_audits'][arm]['executed_drx_profiles']
        result['executed_non_native_drx_samples']=comparison['action_audits'][arm]['executed_non_native_drx_samples']
    return result,cfg


def analyze(inputs,output):
    inputs=Path(inputs);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    expected_source=checkpoint_source_hash(inputs/'run-source.tgz');seeds=[]
    for device in('agx1','agx2'):
        root=inputs/device;plan=load(root/'plan.json');comparison=load(root/'comparison.json')
        assert sha(root/'input-model.json')==plan['model_sha256']
        arms={};configs={}
        for arm in ARMS:
            arms[arm],configs[arm]=metrics(root/'cases'/arm,comparison,arm)
            assert arms[arm]['source_sha256']==expected_source
        assert all(configs[a]['manifest']==configs['base']['manifest']for a in ARMS)
        assert all(configs[a]['configs']['ue']==configs['base']['configs']['ue']for a in ARMS)
        assert all(configs[a]['configs']['phy']['rt_sys']['ue_services']==configs['base']['configs']['phy']['rt_sys']['ue_services']for a in ARMS)
        assert len({tuple((u,v['off']['generated'])for u,v in comparison['comparisons'][a]['per_ue'].items())for a in('native_drx','unlocked_drx')})==1
        seeds.append(dict(seed=plan['seed'],device=device,arms=arms,model_sha256=plan['model_sha256']))
    aggregate={a:{key:mean(s['arms'][a][key]for s in seeds)for key,_,_,_ in METRICS}for a in ARMS}
    # Completion latency/rate aggregate events, not averages of per-UE means.
    for a in ARMS:
        task_n=sum(s['arms'][a]['task_completed']for s in seeds);packet_n=sum(s['arms'][a]['packet_completed']for s in seeds)
        aggregate[a]['task_mean_ms']=sum(s['arms'][a]['task_latency_sum_ms']for s in seeds)/task_n if task_n else None
        aggregate[a]['packet_mean_ms']=sum(s['arms'][a]['packet_latency_sum_ms']for s in seeds)/packet_n if packet_n else None
        aggregate[a]['task_on_time_pct']=100*sum(s['arms'][a]['task_on_time']for s in seeds)/sum(s['arms'][a]['task_generated']for s in seeds)
    changes={a:{key:(100*(aggregate['base'][key]-aggregate[a][key])/aggregate['base'][key]if direction=='min'else 100*(aggregate[a][key]-aggregate['base'][key])/aggregate['base'][key])if aggregate['base'][key]else None for key,_,_,direction in METRICS}for a in ARMS if a!='base'}
    report=dict(schema='oran.drx-unlock-summary.v1',duration_s=3,seeds=seeds,mean=aggregate,improvement_pct_vs_base=changes,
        source_sha256=expected_source,model_sha256=seeds[0]['model_sha256'],model_retrained=False,matched_inputs_verified=True,
        scope=dict(energy='Whole run, including startup; all-six population and UE0+UE4 target reported separately',goodput='All delivered payload; all six UEs',packet_latency='Completed DL packets; P95 is mean of per-seed P95',task_latency='Completion-weighted mean of mature matched MEC creation cohorts; start >=100 ms',task_rate='On-time completed / generated, including failed and pending demand',evidence='Two paired evaluation seeds; no population confidence interval or broad generalization claim',attribution='Effect of enabling joint DRX control; do not present it as a scalar inactivity-timer derivative'))
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def plot(report,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(12,6.4));colors=['#d7b793','#7aa5a0'];labels=['X-01 · native DRX','X-01 · DRX control']
    for ax,(key,title,unit,direction)in zip(axes.flat,METRICS[:6]):
        values=[report['improvement_pct_vs_base'][a][key]for a in('native_drx','unlocked_drx')]
        ax.bar([0,1],values,color=colors,edgecolor='#364744',linewidth=.6,width=.56)
        ax.axhline(0,color='#87918f',linestyle='--',linewidth=.8);ax.set_title(title,fontsize=10)
        ax.set_xticks([0,1],['X-01\nDRX locked','X-01\nDRX enabled']);ax.set_ylabel('Reduction (%)'if direction=='min'else'Increase (%)')
        spread=max(2.,max(abs(v)for v in values))
        for i,v in enumerate(values):ax.text(i,v+(spread*.04 if v>=0 else-spread*.04),f'{v:+.2f}%',ha='center',va='bottom'if v>=0 else'top',fontsize=9)
        ax.margins(y=.27);base=report['mean']['base'][key]
        ax.text(.5,-.25,f'Base: {base:.3f} {unit}',ha='center',transform=ax.transAxes,color='#465451',fontsize=8)
    fig.suptitle('X-01 DRX control: energy benefit and network tradeoffs',fontsize=13,y=.99)
    fig.text(.5,.008,'Change relative to matched Base · 2 seeds × 3 s per arm · same frozen RF · positive = favorable direction',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.95]);output=Path(output)
    for suffix in('png','pdf','svg'):fig.savefig(output/('drx-control-comparison.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    r=analyze(a.input,a.output)
    if a.plot:plot(r,a.output)
    print(json.dumps({'means':r['mean'],'improvement_pct_vs_base':r['improvement_pct_vs_base']},indent=2))
