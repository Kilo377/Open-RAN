"""Compare physical channels with common RT samples and slot-level reference."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from phy.rt_sys.scenario import build_case
from phy.rt_sys.channel import RtChannelProvider,geometry_blocks


def compare(scene,speed,slots=2000,update_ms=None):
    root=ROOT/'outputs/rt_sys_targeted_repairs/geometry';root.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/rt_sys/default.json').read_text());config['ue_speed_mps']=speed
    file=root/f'config_{speed}.json';file.write_text(json.dumps(config))
    base,_,_=build_case(scene,3,12,11,slots,config_path=file)
    providers={}
    for name,interval,cap in [('reference',.5,.01),('default',update_ms or base['path_update_ms'],.01),('legacy10',10.,None)]:
        m=json.loads(json.dumps(base));m['path_update_ms']=interval
        if cap is None:m.pop('max_geometry_displacement_m',None)
        else:m['max_geometry_displacement_m']=cap
        providers[name]=RtChannelProvider(m,ROOT/'channel_cache')
    indices=sorted(set([1,2,3,7,9,19,20,21,39,40]+np.linspace(41,slots,40,dtype=int).tolist()))
    values={name:[] for name in ('default','legacy10')};replay=True
    for slot in indices:
        reference=providers['reference'].snapshot(slot).cfr;gx=np.mean(abs(reference)**2,axis=(2,3,4,5))
        strongest=np.argmax(gx,axis=1);significant=gx>=gx.max(axis=1,keepdims=True)*1e-3
        for name in values:
            h=providers[name].snapshot(slot).cfr;gy=np.mean(abs(h)**2,axis=(2,3,4,5))
            both=(gx>1e-25)&(gy>1e-25)
            delta=np.zeros_like(gx,dtype=float);delta[both]=abs(10*np.log10(gx[both]/gy[both]))
            d=delta[np.arange(len(gx)),strongest]
            values[name].append(dict(slot=slot,relative_cfr_l2=float(np.linalg.norm(reference-h)/max(np.linalg.norm(reference),1e-30)),
                max_all_nonzero_link_gain_db=float(delta.max()),max_strongest_link_gain_db=float(d.max()),
                max_significant_candidate_gain_db=float(delta[significant&both].max(initial=0)),
                significant_path_presence_changes=int(np.sum(significant&((gx>1e-25)!=(gy>1e-25))))))
    p=providers['default'];other=RtChannelProvider(p.manifest,ROOT/'channel_cache')
    for slot in indices:replay=replay and np.array_equal(p.snapshot(slot).cfr,other.snapshot(slot).cfr)
    results={}
    for name,rows in values.items():
        results[name]={k:max(r[k] for r in rows) for k in rows[0] if k!='slot'}
        results[name]['effective_max_interval_ms']=providers[name].update_slots*base['slot_duration_ms'];results[name]['samples']=rows
    passed=results['default']['max_strongest_link_gain_db']<=.5 and results['default']['max_significant_candidate_gain_db']<=1 and results['default']['significant_path_presence_changes']==0 and replay
    return dict(scene=scene,speed_mps=speed,passed=bool(passed),device='cuda',tested_slots=indices,comparison=results,cache_replay_exact=bool(replay),
        scope='Slot-reference convergence on sampled times; all-link worst cases retained. No universal map/speed accuracy guarantee.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--slots',type=int,default=2000);p.add_argument('--default-interval',type=float);p.add_argument('--output',type=Path);args=p.parse_args()
    results=[]
    for scene in ('simple_street_canyon','munich'):
        for speed in (1.5,10.,20.):
            r=compare(scene,speed,args.slots,args.default_interval);results.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('comparison','tested_slots')}),flush=True)
            out=args.output or ROOT/'outputs/rt_sys_targeted_repairs/geometry_validation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'passed':all(r['passed'] for r in results),'complete':len(results)==6,'cases':results},indent=2)+'\n')
    if not all(r['passed'] for r in results):raise SystemExit('Geometry update convergence criterion failed')
