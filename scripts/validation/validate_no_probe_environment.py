"""Staged finite RT + actual NR integration acceptance, with immutable artifacts."""
import argparse,json,subprocess,sys,csv,time,signal
from threading import Event,Lock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint
from scripts.validation.validate_environment_repairs import audit

STOP=Event();CHILDREN=set();LOCK=Lock()

def stop_children(signum,frame):
    STOP.set()
    with LOCK:
        for child in list(CHILDREN):child.terminate()
    raise SystemExit(128+signum)

signal.signal(signal.SIGTERM,stop_children)
signal.signal(signal.SIGINT,stop_children)


def audit_contract(folder):
    result=audit(folder)
    final=json.loads((folder/'final_summary.json').read_text())
    stats=final['link_reference_statistics']
    assert all(stats.get(k,0)==0 for k in ('amc_probe_trials','bler_probe_trials','calibration_decode_trials','incompatible_no_tb','unsupported_no_tb'))
    assert result['counters']['capacity_guards']==result['counters']['unsupported_code']==0
    config=json.loads((folder/'config_snapshot.json').read_text())
    warmup=config['warmup_ms'];dt=config['manifest']['slot_duration_ms']
    post=Counter();per_ue={}
    attempts=first=nacks=retx=0;contracts={};acked=set();first_ids=set()
    for line in (folder/'rt_sys_trace.jsonl').open():
        record=json.loads(line)
        for u in record['ues']:
            assert not u.get('mac_capacity_guard',False)
            if not u['actual_decode']:continue
            attempts+=1
            c=u['tx_contract'];assert c and c['payload_bytes']==u['tb_bytes']
            assert c['nr_tbs_bits']//8==u['physical_tb_capacity_bytes']
            assert c['coded_bits']>0 and c['payload_bytes']*8<=c['nr_tbs_bits']
            assert c['mcs']==u['mcs'] and c['table']==u['mcs_table']
            tb=u['tb_id']
            if u['is_retx']:
                retx+=1;assert contracts[tb]==c
            else:
                first+=1;nacks+=not u['ack'];assert tb not in contracts
                first_ids.add(tb);contracts[tb]=c
                if record['slot_id']*dt>warmup:
                    post['first_samples']+=1;post['first_errors']+=not u['ack']
                    entity=per_ue.setdefault(u['ue_id'],Counter())
                    entity['first_samples']+=1;entity['first_errors']+=not u['ack']
                    entity['mcs_boundary_samples']+=u['mcs'] in (0,16 if u['mcs_table']=='qam64_low_se' else 27)
            if u['ack']:
                assert tb not in acked;acked.add(tb)
    assert attempts==stats['actual_tb_decodes']
    assert len(acked)==sum(m['successful_tbs'] for m in final['tb_outcomes'].values())
    # Delivered byte intervals can complete in any HARQ order, but never overlap.
    delivered_ranges={}
    for line in (folder/'packet_events.jsonl').open():
        event=json.loads(line)
        if event['kind']=='delivery':
            assert event['tb_id'] in acked
            key=(event['ue_id'],event['packet_id'])
            delivered_ranges.setdefault(key,[]).append((event['byte_start'],event['byte_end']))
    for ranges in delivered_ranges.values():
        ordered=sorted(ranges)
        assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:]))
    from phy.rt_sys.reference import probability_interval
    for counts in [post,*per_ue.values()]:
        n=counts['first_samples'];e=counts['first_errors']
        counts['first_bler']=e/n if n else None
        counts['wilson_95']=probability_interval(e,n) if n else None
        counts['mcs_boundary_fraction']=counts['mcs_boundary_samples']/n if n else None
        counts['acceptance_status']='insufficient_samples' if n<2000 else 'dynamic_RT_case_not_fixed_link_target_gate'
    result.update(post_warmup_first_tx=dict(post),post_warmup_per_ue={k:dict(v) for k,v in per_ue.items()},actual_decode_attempts=attempts,first_tx_attempts=first,first_tx_nacks=nacks,
        first_tx_bler=nacks/first if first else None,retx_attempts=retx,
        retx_deferred_resource=sum(m['retx_deferred_resource'] for m in final['tb_outcomes'].values()),
        du_olla=final['du_olla'],tb_outcomes=final['tb_outcomes'],steady_state=final['steady_state'])
    (folder/'contract_validation.json').write_text(json.dumps(result,indent=2))
    return result


def execute(folder,scene,n,load,seed,slots,warmup=0):
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir()
    command=[sys.executable,str(ROOT/'experiments/run_rt_sys.py'),'--scene',scene,'--cells','3',
        '--ues',str(n),'--seed',str(seed),'--slots',str(slots),'--initial-queue-bytes','0',
        '--ue-speed-mps','0','--traffic-load',str(load),'--warmup-ms',str(warmup),
        '--feedback-mode','nr_direct','--control-mode','baseline','--output',str(folder)]
    if STOP.is_set():raise InterruptedError('Validation stopped before child launch')
    with (folder/'process.log').open('w') as log:
        with subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as child:
            with LOCK:
                if STOP.is_set():child.terminate()
                CHILDREN.add(child)
            try:
                rc=child.wait()
                if rc:raise subprocess.CalledProcessError(rc,command)
            finally:
                with LOCK:CHILDREN.discard(child)
    return audit_contract(folder)


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sha=source_fingerprint(ROOT);started=time.perf_counter();reports=[]
    if args.stage=='gate':
        for scene in ('simple_street_canyon','munich'):
            folder=out/f'{scene}_timing8'
            cmd=[sys.executable,str(ROOT/'scripts/validation/benchmark_rt_cost.py'),'--worker','--mode','nr_direct',
                '--scene',scene,'--slots','8','--output',str(folder)]
            with (out/f'{scene}_timing.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            profile=json.loads((folder/'cost_profile.json').read_text());assert profile['total_s']<=14.52
            reports.append(audit_contract(folder));print(f'{scene} timing {profile["total_s"]:.2f}s',flush=True)
        for scene in ('simple_street_canyon','munich'):
            reports.append(execute(out/f'{scene}_128',scene,24,1,512,128))
            print(f'{scene} 128-slot passed',flush=True)
    else:
        specs=[(scene,n,load,seed) for scene in ('simple_street_canyon','munich')
            for n,load in ((4,1.),(24,.1),(24,1.)) for seed in (512,513,514)]
        def case(spec):
            scene,n,load,seed=spec
            folder=out/f'{scene}_u{n}_load{load}_seed{seed}'
            return execute(folder,scene,n,load,seed,2000,200)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(case,spec):spec for spec in specs}
            try:
                for future in as_completed(futures):
                    result=future.result();reports.append(result)
                    (out/'progress.json').write_text(json.dumps(dict(completed=len(reports),total=len(specs),cases=reports,source_sha256=sha),indent=2))
                    print(f'{len(reports)}/{len(specs)} {result["case"]} {result["elapsed_s"]:.1f}s',flush=True)
            except Exception:
                for future in futures:future.cancel()
                raise
    assert source_fingerprint(ROOT)==sha
    report=dict(passed=True,stage=args.stage,source_sha256=sha,elapsed_s=time.perf_counter()-started,cases=reports,
        scope='Finite integration acceptance; overload and weak coverage are reported, not silently waived as QoS success')
    (out/'validation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['gate','matrix'],required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
