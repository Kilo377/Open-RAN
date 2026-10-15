"""No-fault deployment baselines and moving RT evidence; no learned policy."""
import argparse,json,subprocess,sys,time,math,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint
from scripts.validation.validate_efficiency import audit,read


def execute(out,scene,ues,load,seed,slots,warmup,speed,execution):
    name=f'{scene}_u{ues}_load{load}_s{seed}_speed{speed}_{slots}'
    folder=out/name
    if folder.exists():
        if (folder/'acceptance.json').exists():
            old=read(folder/'acceptance.json');assert old['source_sha256']==source_fingerprint(ROOT)
            return old
        raise FileExistsError(f'Partial case preserved: {folder}')
    folder.mkdir()
    cmd=[sys.executable,str(ROOT/'experiments/run_rt_sys.py'),'--config',str(ROOT/'configs/rt_sys/efficiency_experiment.json'),
        '--scene',scene,'--cells','3','--ues',str(ues),'--traffic-load',str(load),'--seed',str(seed),
        '--slots',str(slots),'--warmup-ms',str(warmup),'--ue-speed-mps',str(speed),'--initial-queue-bytes','0',
        '--control-mode','baseline','--feedback-mode','nr_direct','--nr-execution',execution,'--output',str(folder)]
    with (folder/'process.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=audit(folder);config=read(folder/'config_snapshot.json');final=read(folder/'final_summary.json')
    assert not config['configs']['ric']['xapps'] and not config['configs']['ric']['rapps']
    assert not (folder/'control_timeline.json').exists()
    assert config['configs']['phy']['rt_sys']['paper_interfaces']['connection_events']
    assert all(v['initial_bytes']==0 for v in final['packet_metrics'].values())
    noops=0
    for line in (folder/'control_receipts.jsonl').open():
        receipt=json.loads(line)
        assert (receipt.get('kind')=='config' and not receipt.get('fields')
            and receipt.get('before')==receipt.get('after') and receipt.get('status')=='applied'
            and receipt.get('source')=='controller_manager'),('No-fault baseline changed control parameters',receipt)
        noops+=1
    diag=final['runtime_diagnostics'];zero=[u for u,m in diag['per_ue'].items() if not m['steady_state']['delivered_bytes']]
    weak=diag['per_ue'].get('ue_18')
    result.update(case=name,source_sha256=config['source_sha256'],scene=scene,ues=ues,seed=seed,load=load,speed_mps=speed,
        warmup_ms=warmup,simulated_ms=slots*.5,zero_delivery_ues=zero,
        offered_mbps=sum(m['offered_mbps'] for m in final['steady_state']['per_ue'].values()),
        delivered_mbps=sum(m['delivered_mbps'] for m in final['steady_state']['per_ue'].values()),
        overflow_bytes=sum(m['overflow_bytes'] for m in final['steady_state']['per_ue'].values()),
        deadline_misses=sum(m['deadline_misses'] for m in final['steady_state']['per_ue'].values()),
        diagnoses={u:m['diagnoses'] for u,m in diag['per_ue'].items()},baseline_noop_receipts=noops,
        munich_ue18=weak if scene=='munich' else None,
        scope='Finite no-fault environment evidence; weak coverage/overload are retained, not structural failures.')
    (folder/'acceptance.json').write_text(json.dumps(result,indent=2));return result


def main(a):
    a.output.mkdir(parents=True,exist_ok=True);sha=source_fingerprint(ROOT);start=time.perf_counter()
    if a.stage=='baseline':
        specs=[(scene,n,load,seed,4000,400,0.) for scene in ('simple_street_canyon','munich')
            for n,load in ((4,1.),(24,.1),(24,1.)) for seed in (512,513,514)]
    else:
        slots=128 if a.stage=='moving_gate' else 4000
        specs=[(scene,4,1.,seed,slots,0 if slots==128 else 400,speed)
            for scene in ('simple_street_canyon','munich') for speed in (1.5,10.) for seed in (512,513)]
    if a.stage=='moving':
        gate=read(a.gate/'validation.json');assert gate['completed'] and gate['source_sha256']==sha
        estimates=[]
        for case in gate['cases']:
            folder=a.gate/case['case'];perf=read(folder/'performance.json')
            cache=ROOT/'channel_cache'/perf['cache_key']
            bytes_cached=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
            estimates.append(dict(case=case['case'],estimated_case_s=case['elapsed_s']*4000/128,
                estimated_output_bytes=read(folder/'run_status.json')['output_bytes']*4000/128,
                estimated_channel_bytes=bytes_cached*4000/129))
        (a.output/'expansion_estimate.json').write_text(json.dumps(dict(estimates=estimates,workers=a.workers,
            interpretation='Conservative short-gate extrapolation; actual cold RT generation and output may differ.'),indent=2))
        # Explicit per-case RT cache quota is 5 GiB; never evict old evidence to fit a long run.
        if any(e['estimated_channel_bytes']>5*1024**3 for e in estimates):raise SystemExit('Estimated RT cache exceeds per-case quota; preserve gates and inspect before expansion')
    cases=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(execute,a.output,*s,a.execution):s for s in specs}
        for future in as_completed(futures):
            result=future.result();cases.append(result)
            (a.output/'progress.json').write_text(json.dumps(dict(completed=len(cases),total=len(specs),cases=cases,source_sha256=sha),indent=2))
            print(len(cases),len(specs),result['case'],round(result['elapsed_s'],1),flush=True)
    extensions=[]
    if a.stage=='moving':
        for result in list(cases):
            folder=a.output/result['case'];diag=read(folder/'runtime_diagnostics.json');candidates=[]
            for key,c in diag['per_link_segment'].items():
                uid,cell,table,segment=key.split('|');u=diag['per_ue'][uid];n=c.get('first_samples',0)
                if (cell==u['serving_cell_id'] and 1000<=n<2000
                    and c.get('saturated_samples',0)/n<.05
                    and c.get('stale_or_missing_measurements',0)/n<.05
                    and u.get('connected_slots',0)/max(u['evaluated_slots'],1)>.95):
                    candidates.append(dict(link=key,first_samples=n))
            if not candidates:continue
            slots=min(20000,int(math.ceil((800+3200*2000/min(c['first_samples'] for c in candidates)*1.2)/2000))*2000)
            perf=read(folder/'performance.json');cache=ROOT/'channel_cache'/perf['cache_key']
            estimate=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())*slots/4000
            request=dict(case=result['case'],requested_slots=slots,candidates=candidates,
                estimated_channel_bytes=estimate,reason='Additional finite samples on continuously connected, unsaturated final links; not a target certificate.')
            if estimate>5*1024**3 or shutil.disk_usage(a.output).free<estimate*2:
                request['status']='not_extended_storage_budget';extensions.append(request);continue
            request['status']='running';extensions.append(request)
            (a.output/'extensions.json').write_text(json.dumps(extensions,indent=2))
            extended=execute(a.output,result['scene'],4,1.,result['seed'],slots,400,result['speed_mps'],a.execution)
            cases.append(extended);request['status']='completed';request['extended_case']=extended['case']
            (a.output/'extensions.json').write_text(json.dumps(extensions,indent=2))
    assert source_fingerprint(ROOT)==sha
    report=dict(completed=True,structural_passed=True,source_sha256=sha,cases=cases,elapsed_s=time.perf_counter()-start,
        stage=a.stage,extensions=extensions,interpretation='SLA and target BLER are reported independently; not all-UE QoS or asymptotic stability acceptance.')
    (a.output/'validation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['baseline','moving_gate','moving'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3);p.add_argument('--execution',choices=['scalar','batched'],default='batched')
    a=p.parse_args()
    if a.stage=='moving' and not a.gate:p.error('moving requires an accepted gate directory')
    main(a)
