"""Staged, serial, actual-NR efficiency acceptance. Raw data stays on Spark."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def source_fingerprint(root):
    # Delay package import until a worker has selected its frozen source root.
    from phy.rt_sys.contracts import source_fingerprint as fingerprint
    return fingerprint(root)


def read(path):return json.loads(Path(path).read_text())


def audit(folder):
    folder=Path(folder);status=read(folder/'run_status.json');config=read(folder/'config_snapshot.json');final=read(folder/'final_summary.json')
    assert status['completed'] and status['spark_verified']
    assert status['source_sha256']==config['source_sha256']
    stats=final['link_reference_statistics']
    assert all(stats.get(k,0)==0 for k in ('amc_probe_trials','bler_probe_trials','calibration_decode_trials','incompatible_no_tb','unsupported_no_tb'))
    counts=Counter();contracts={};acked=set();last={};energy=Counter();slots=0
    for line in (folder/'rt_sys_trace.jsonl').open():
        row=json.loads(line);slots+=1
        for cid,c in row['cells'].items():
            assert 0<=c['average_used_w']<=c['budget_w']+1e-10
            assert 0<=c['occupied_re']<=c['grid_re']
            energy[cid]+=c['average_used_w']*config['manifest']['slot_duration_ms']/1000
        for u in row['ues']:
            last[u['ue_id']]=u
            assert not u.get('mac_capacity_guard')
            assert u.get('ack_source') not in ('harq_incompatible_no_tb','unsupported_nr_code_no_tb')
            assert u['delivered_bytes']==(u['tb_bytes'] if u['ack'] else 0)
            assert u['packet_initial_bytes']+u['packet_offered_bytes']==u['packet_pending_bytes']+u['packet_overflow_bytes']+u['packet_delivered_bytes']
            if not u['scheduled']:assert not u['actual_decode'] and not u['delivered_bytes']
            if u.get('ack_source')=='no_path':assert not u['actual_decode'] and not u['delivered_bytes']
            if not u['actual_decode']:continue
            counts['actual_decodes']+=1
            tb=u['tb_id'];assert tb
            contract=u['tx_contract'];assert contract and contract['payload_bytes']>0
            assert contract['payload_bytes']==u['tb_bytes']<=contract['nr_tbs_bits']//8
            if u['is_retx']:
                assert contracts[tb]==contract
                counts['retx']+=1
            else:
                assert tb not in contracts;contracts[tb]=contract;counts['first']+=1
                counts['first_errors']+=not u['ack']
            if u['ack']:
                assert tb not in acked;acked.add(tb)
    assert slots==config['manifest']['total_slots']
    assert counts['actual_decodes']==stats['actual_tb_decodes']
    assert len(acked)==sum(v['successful_tbs'] for v in final['tb_outcomes'].values())
    ranges={};deliveries=Counter()
    for line in (folder/'packet_events.jsonl').open():
        e=json.loads(line)
        if e['kind']!='delivery':continue
        assert e['tb_id'] in acked
        assert e['bytes']==e['byte_end']-e['byte_start']
        ranges.setdefault((e['ue_id'],e['packet_id']),[]).append((e['byte_start'],e['byte_end']))
        deliveries[e['ue_id']]+=e['bytes']
    for intervals in ranges.values():
        ordered=sorted(intervals)
        assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:]))
    for uid,m in final['packet_metrics'].items():
        assert m['initial_bytes']+m['offered_bytes']==m['pending_bytes']+m['overflow_bytes']+m['delivered_bytes']
        assert deliveries[uid]==m['delivered_bytes']==last[uid]['packet_delivered_bytes']
    if final.get('paper_interfaces'):
        for cid,joules in final['paper_interfaces']['rf_energy_j'].items():assert abs(joules-energy[cid])<1e-9
    return dict(passed=True,slots=slots,**dict(counts),elapsed_s=status['elapsed_s'],output_bytes=status.get('output_bytes'),source_sha256=status['source_sha256'])


EXACT='ue_id cell_id mcs mcs_table ack tb_id is_retx scheduled actual_decode tb_bytes delivered_bytes queue_bytes allocated_re data_re layers tx_contract resource_allocation packet_pending_bytes packet_delivered_bytes packet_overflow_bytes'.split()


def equivalent(reference,candidate):
    import itertools,math
    a=Path(reference);b=Path(candidate);slots=0
    assert read(a/'initial_conditions.json')['environment_sha256']==read(b/'initial_conditions.json')['environment_sha256']
    with (a/'rt_sys_trace.jsonl').open() as fa,(b/'rt_sys_trace.jsonl').open() as fb:
        for ra,rb in itertools.zip_longest(fa,fb):
            assert ra is not None and rb is not None,'Trace lengths differ'
            x,y=json.loads(ra),json.loads(rb);slots+=1
            assert x['slot_id']==y['slot_id']
            assert x['connection_states']==y['connection_states'] and x['handover_counts']==y['handover_counts']
            assert len(x['ues'])==len(y['ues'])
            for u,v in zip(x['ues'],y['ues']):
                for k in EXACT:assert u.get(k)==v.get(k),(slots,u['ue_id'],k,u.get(k),v.get(k))
                for k in ('sinr_db','reference_rsrp_dbm','reference_sinr_db'):
                    if u.get(k) is None or v.get(k) is None:assert u.get(k)==v.get(k)
                    else:assert math.isclose(u[k],v[k],rel_tol=1e-6,abs_tol=1e-4),(slots,u['ue_id'],k)
            assert x['cells'].keys()==y['cells'].keys()
            for cid in x['cells']:
                for k in ('budget_w','average_used_w'):
                    assert math.isclose(x['cells'][cid][k],y['cells'][cid][k],rel_tol=1e-6,abs_tol=1e-10)
    for name in ('packet_events.jsonl','harq_events.jsonl'):
        with (a/name).open() as fa,(b/name).open() as fb:
            for x,y in itertools.zip_longest(fa,fb):assert x==y,(name,'Event streams differ')
    return dict(passed=True,slots=slots,exact_fields=EXACT,float_rtol=1e-6,sinr_atol_db=1e-4,power_atol_w=1e-10)


def worker(args):
    # This driver also loads frozen source without editing it. The observer
    # measures the complete post-warmup span, including logging/finalization.
    root=args.root.resolve();sys.path.insert(0,str(root))
    from core.simulator import Simulator
    from experiments.run_rt_sys import parse_args,run_case
    original=Simulator.step;steady_start=None;step_s=0.;step_count=0
    warmup=64 if args.slots==512 else 0
    def measured(sim):
        nonlocal steady_start,step_s,step_count
        if steady_start is None and sim.state.sim_time_ms>=warmup:steady_start=time.perf_counter()
        start=time.perf_counter()
        result=original(sim)
        step_s+=time.perf_counter()-start;step_count+=1
        return result
    Simulator.step=measured
    sys.argv=['run_rt_sys.py','--scene',args.scene,'--cells','3','--ues',str(args.ues),
        '--slots',str(args.slots),'--seed','512','--ue-speed-mps','0','--initial-queue-bytes','0',
        '--warmup-ms',str(warmup),'--control-mode','baseline','--config',str(root/'configs/rt_sys'/args.config),
        '--output',str(args.output)]
    if args.execution:sys.argv+=['--nr-execution',args.execution]
    start=time.perf_counter();run_case(parse_args());end=time.perf_counter()
    (args.output/'serial_timing.json').write_text(json.dumps(dict(total_s=end-start,
        steady_including_finalization_s=end-steady_start,step_s=step_s,step_count=step_count,
        warmup_ms=warmup,source_sha256=source_fingerprint(root)),indent=2))


def run_one(out,root,scene,ues,slots,label,repeat=0):
    folder=(root/'outputs/efficiency_reference'/f'{scene}_u{ues}_{slots}_{label}_r{repeat}') if label=='reference' else out/f'{scene}_u{ues}_{slots}_{label}_r{repeat}'
    if folder.exists():
        if (folder/'serial_timing.json').exists():
            assert read(folder/'run_status.json')['source_sha256']==source_fingerprint(root),'Refusing mixed source versions'
            return folder
        raise FileExistsError(f'Incomplete case retained: {folder}')
    folder.mkdir(parents=True)
    config='paper_interfaces.json' if label=='reference' else f'efficiency_{label.split("_")[0]}.json'
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--root',str(root),'--scene',scene,
        '--ues',str(ues),'--slots',str(slots),'--config',config,'--output',str(folder)]
    if label!='reference':cmd+=['--execution',label.split('_')[1]]
    with (folder/'process.log').open('w') as log:subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    return folder


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);sha=source_fingerprint(ROOT)
    report=dict(stage=args.stage,source_sha256=sha,completed=False,cases=[],comparisons=[])
    choices=['audit_scalar','audit_batched','experiment_scalar','experiment_batched']
    for scene in ('simple_street_canyon','munich'):
        for ues in (4,24):
            for slots in ((8,128) if args.stage=='gate' else (512,)):
                for repeat in range(1 if args.stage=='gate' else 3):
                    reference=run_one(out,args.reference_root,scene,ues,slots,'reference',repeat)
                    ar=audit(reference);report['cases'].append(dict(label='reference',scene=scene,ues=ues,slots=slots,repeat=repeat,path=str(reference),audit=ar,timing=read(reference/'serial_timing.json')))
                    for label in choices:
                        candidate=run_one(out,ROOT,scene,ues,slots,label,repeat)
                        audit_result=audit(candidate);comparison=equivalent(reference,candidate)
                        report['cases'].append(dict(label=label,scene=scene,ues=ues,slots=slots,repeat=repeat,path=str(candidate),audit=audit_result,timing=read(candidate/'serial_timing.json')))
                        report['comparisons'].append(dict(scene=scene,ues=ues,label=label,repeat=repeat,**comparison))
                        (out/'progress.json').write_text(json.dumps(report,indent=2))
                        print(scene,ues,slots,label,repeat,'equivalent',flush=True)
    report['completed']=True;report['equivalence_passed']=True
    if args.stage=='performance':
        rows=[]
        for scene in ('simple_street_canyon','munich'):
            for ues in (4,24):
                medians={label:statistics.median(r['timing']['steady_including_finalization_s'] for r in report['cases'] if r['label']==label and r['scene']==scene and r['ues']==ues) for label in ['reference']+choices}
                rows.append(dict(scene=scene,ues=ues,medians_s=medians,
                    speedups={label:medians['reference']/medians[label] for label in choices},
                    experiment_batched_target_met=medians['experiment_batched']<=.8*medians['reference'],
                    audit_batched_no_regression=medians['audit_batched']<=1.05*medians['reference']))
        report['performance']=rows
        report['performance_target_met']=all(r['experiment_batched_target_met'] and r['audit_batched_no_regression'] for r in rows)
    assert source_fingerprint(ROOT)==sha,'Source changed during acceptance'
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','comparisons')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['gate','performance'])
    p.add_argument('--reference-root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--root',type=Path);p.add_argument('--scene');p.add_argument('--ues',type=int);p.add_argument('--slots',type=int);p.add_argument('--config');p.add_argument('--execution')
    a=p.parse_args()
    if a.worker:worker(a)
    else:
        if not a.stage or not a.reference_root:p.error('stage and reference-root required')
        main(a)
