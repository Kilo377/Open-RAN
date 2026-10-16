"""Isolate scheduler parameters in paired real RT/NR contention fixtures.

This deliberately co-locates two UEs on an existing legal route, limits SU
selection to one UE and caps each at two PRBs. It is a consumer test, not a
capacity result for the original deployments. All raw evidence stays on Spark.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def read(path):return json.loads(path.read_text())
def rows(path):
    with path.open() as stream:
        for line in stream:yield json.loads(line)


def run_case(a):
    from experiments import run_rt_sys as runner
    original=runner.build_case
    def fixture(*values,**kwargs):
        manifest,configs,intent=original(*values,**kwargs)
        manifest['ues'][1]['trajectory']=deepcopy(manifest['ues'][0]['trajectory'])
        position=manifest['ues'][0]['trajectory']['positions_m'][0]
        configs['sim'].update(scheduler_max_ues_per_slot=1,scheduler_enable_spatial_multiplexing=False)
        template=deepcopy(configs['ue']['ues'][0])
        configs['ue']['ues']=[]
        for i in range(2):
            u=deepcopy(template)
            u.update(ue_id=f'ue_{i}',slice_id=('urllc' if i==0 else 'embb'),priority=1,
                latency_budget_ms=(1 if i==0 and a.kind=='latency_guard' else 100),
                queue_bytes=(40000 if i==1 and a.kind=='latency_guard' else 20000),
                arrival_model='cbr',arrival_rate_bps=100000,max_queue_bytes=200000,logical_channels=[])
            u.update(dict(zip(('x_m','y_m','z_m'),position)))
            configs['ue']['ues'].append(u)
        if a.arm=='rule':
            configs['ric'].update(rapps=[],xapps=[dict(type='rule_based_v1',kind=a.kind,name='rule_'+a.kind)],run_rapps_on_start=False)
            configs['phy']['rt_sys']['paper_interfaces']['rule_observations']=True
        return manifest,configs,intent
    runner.build_case=fixture
    mode,hint=('slice_aware','slice_bias') if a.kind=='slice_sla' else ('latency_aware','latency_bias')
    initial=dict(target_cell_id='cell_1',scheduler_mode=mode,scheduler_policy_hint=hint,ue_prb_limits={'ue_0':2,'ue_1':2})
    key='slice_weight_adjustment' if a.kind=='slice_sla' else 'latency_bias'
    base={'urllc':1.,'embb':1.} if a.kind=='slice_sla' else .1
    treatment={'urllc':1.,'embb':4.} if a.kind=='slice_sla' else 4.
    initial[key]=base
    timeline=a.output.parent/(a.output.name+'_timeline.json')
    entries=[dict(slot=5,config=initial),dict(slot=20,config={'target_cell_id':'cell_1',key:treatment if a.arm=='changed' else base})]
    if a.arm=='rule':entries=[dict(slot=5,config=dict(target_cell_id='cell_1',ue_prb_limits={'ue_0':2,'ue_1':2}))]
    write(timeline,entries)
    sys.argv=[sys.argv[0],'--config',str(ROOT/'configs/rt_sys/efficiency_experiment.json'),
        '--scene',a.scene,'--cells','2','--ues','2','--slots','128','--seed','512',
        '--initial-queue-bytes','20000','--ue-speed-mps','0','--warmup-ms','0',
        '--control-mode','orchestrated' if a.arm=='rule' else 'baseline','--feedback-mode','nr_direct','--nr-execution','batched',
        '--control-timeline',str(timeline),'--output',str(a.output)]
    runner.run_case(runner.parse_args())


def validate_pair(root,scene,kind):
    from scripts.validation.validate_efficiency import audit
    folders=[root/f'{scene}_{kind}_{arm}' for arm in ('reference','changed')]
    for p in folders:
        assert read(p/'final_summary.json')['link_reference_statistics'].get('actual_tb_decodes',0)>0, 'No real transmissions; check coverage fixture'
    audits=[audit(p) for p in folders]
    assert read(folders[0]/'initial_conditions.json')['environment_sha256']==read(folders[1]/'initial_conditions.json')['environment_sha256']
    traces=[list(rows(p/'rt_sys_trace.jsonl')) for p in folders]
    assert len(traces[0])==len(traces[1])==128
    # Before the slot-20 request can take effect, all TB decisions must match.
    fields=('ue_id','scheduled','actual_decode','ack','mcs','delivered_bytes','queue_bytes','resource_allocation')
    for x,y in zip(*traces):
        if x['slot_id']>=21:break
        assert [[u.get(k) for k in fields] for u in x['ues']]==[[u.get(k) for k in fields] for u in y['ues']]
    key='slice_weight_adjustment' if kind=='slice_sla' else 'latency_bias'
    executed=[];metrics=[]
    for folder,trace in zip(folders,traces):
        values=[r['after']['cell_1'][key] for r in rows(folder/'control_receipts.jsonl')
            if r.get('status')=='applied' and r.get('slot_id',0)>=21 and 'cell_1' in r.get('after',{})]
        assert values, 'Missing final consumer receipts'
        executed.append(values[-1]);totals={uid:Counter() for uid in ('ue_0','ue_1')}
        for row in trace:
            if row['slot_id']<21:continue
            for u in row['ues']:
                m=totals[u['ue_id']]
                m['scheduled_slots']+=bool(u['scheduled'])
                m['prb_slots']+=len(u.get('resource_allocation',{}).get('prbs',[]))
                m['delivered_bytes']+=u['delivered_bytes'];m['final_queue_bytes']=u['queue_bytes']
        metrics.append({uid:dict(m) for uid,m in totals.items()})
    assert executed[0]!=executed[1], 'Parameter was cleared or did not reach consumer'
    assert any(metrics[0][u]['prb_slots']!=metrics[1][u]['prb_slots'] for u in metrics[0]), 'No resource response'
    assert any(metrics[0][u]['delivered_bytes']!=metrics[1][u]['delivered_bytes'] for u in metrics[0]), 'No actual NR delivery response'
    assert any(metrics[0][u]['final_queue_bytes']!=metrics[1][u]['final_queue_bytes'] for u in metrics[0]), 'No queue response'
    return dict(scene=scene,kind=kind,passed=True,parameter=key,executed=executed,metrics=metrics,audits=audits,
        interpretation='Parameter-only causal pair; a response is required, an optimization gain is not.')


def main(a):
    from phy.rt_sys.contracts import source_fingerprint
    from scripts.validation.validate_rule_based import inventory
    root=a.output.resolve()
    if not root.is_relative_to(ROOT/'outputs'):raise ValueError('Evidence must stay in Spark outputs')
    root.mkdir(parents=True,exist_ok=False);start=time.time()
    source=source_fingerprint(ROOT);files=inventory(ROOT)
    write(root/'source_sha256.json',files)
    with tarfile.open(root/'executed_sources.tar.gz','w:gz') as archive:
        for name in files:archive.add(ROOT/name,arcname=name)
    status=dict(completed=False,passed=False,source_sha256=source,fixture=__doc__,results=[],rule_consumers=[])
    try:
        for scene in ('simple_street_canyon','munich'):
            for kind in ('slice_sla','latency_guard'):
                for arm in ('reference','changed'):
                    folder=root/f'{scene}_{kind}_{arm}'
                    cmd=[sys.executable,__file__,'--case','--scene',scene,'--kind',kind,'--arm',arm,'--output',str(folder)]
                    with (root/(folder.name+'.log')).open('w') as log:
                        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=validate_pair(root,scene,kind);status['results'].append(result)
                print('PASS',scene,kind,result['metrics'],flush=True)
                write(root/'validation.json',status)
        from scripts.validation.validate_efficiency import audit
        for scene in ('simple_street_canyon','munich'):
            for kind in ('slice_sla','latency_guard'):
                folder=root/f'{scene}_{kind}_rule'
                with (root/(folder.name+'.log')).open('w') as log:
                    subprocess.run([sys.executable,__file__,'--case','--scene',scene,'--kind',kind,'--arm','rule','--output',str(folder)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=audit(folder)
                key='slice_weight_adjustment' if kind=='slice_sla' else 'latency_bias'
                requested=[r['values'][key] for r in rows(folder/'rule_decisions.jsonl')
                    if r.get('reason')=='action_requested' and r.get('cell_id')=='cell_1' and r.get('values',{}).get(key)]
                executed=[r['after']['cell_1'][key] for r in rows(folder/'control_receipts.jsonl')
                    if r.get('status')=='applied' and 'cell_1' in r.get('after',{})]
                assert requested and executed and executed[-1]==requested[-1], 'Rule request never reached final consumer'
                result.update(scene=scene,kind=kind,parameter=key,last_requested=requested[-1],last_executed=executed[-1])
                status['rule_consumers'].append(result)
                print('PASS actual rule consumer',scene,kind,executed[-1],flush=True)
        assert source_fingerprint(ROOT)==source and inventory(ROOT)==files, 'Source changed during acceptance'
        assert all(r['source_sha256']==source for pair in status['results'] for r in pair['audits'])
        assert all(r['source_sha256']==source for r in status['rule_consumers'])
        status.update(completed=True,passed=True,elapsed_s=time.time()-start)
    except BaseException as error:
        status.update(error=f'{type(error).__name__}: {error}',elapsed_s=time.time()-start)
        raise
    finally:
        write(root/'validation.json',status)
        write(root/'SHA256.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.name!='SHA256.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='store_true');p.add_argument('--scene')
    p.add_argument('--kind',choices=('slice_sla','latency_guard'))
    p.add_argument('--arm',choices=('reference','changed','rule'))
    args=p.parse_args()
    (run_case if args.case else main)(args)
