"""Finite integration matrix, distinct from formal generalization evaluation."""
import argparse
from pathlib import Path
import sys
import subprocess
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]


def cases(slots=1000,update_ms=.5):
    return [{'scene':scene,'cells':3,'ues':ues,'seed':seed,'slots':slots,'path_update_ms':update_ms}
            for scene in ('simple_street_canyon','munich') for ues in (12,24) for seed in (1,2,3)]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--slots',type=int,default=1000);p.add_argument('--path-update-ms',type=float,default=.5)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--skip-completed',action='store_true')
    p.add_argument('--model',choices=['full','fast'],default='full')
    p.add_argument('--fast-profile',choices=['efficient','reference','system'])
    p.add_argument('--fast-gain-update-ms',type=float)
    p.add_argument('--workers',type=int,default=1,choices=range(1,5))
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/rt_sys_matrix')
    args=p.parse_args();plan=cases(args.slots,args.path_update_ms)
    if args.fast_profile and args.model!='fast':p.error('--fast-profile requires --model fast')
    if args.model=='fast':args.fast_profile=args.fast_profile or 'system'
    if args.fast_gain_update_ms is not None and args.fast_profile!='system':p.error('--fast-gain-update-ms requires system profile')
    if args.fast_gain_update_ms is not None:
        import math
        slots=args.fast_gain_update_ms/.5
        if not math.isfinite(slots) or slots<1 or not math.isclose(slots,round(slots),rel_tol=1e-12,abs_tol=1e-8):
            p.error('Fast gain interval must align with positive whole 0.5 ms slots')
    if args.dry_run:print(json.dumps(plan,indent=2));sys.exit(0)
    sys.path.insert(0,str(ROOT))
    from phy.rt_sys.contracts import source_fingerprint
    source_hash=source_fingerprint(ROOT)
    root=args.output_root.resolve()/args.model
    if args.fast_profile:root=root/args.fast_profile
    if args.fast_profile=='system':root=root/f'gain_{args.fast_gain_update_ms or 2.:g}ms'
    root.mkdir(parents=True,exist_ok=True)
    if args.device=='cuda':
        sys.path.insert(0,str(ROOT))
        from scripts.spark.preflight import check
        (root/'preflight.json').write_text(json.dumps(check(),indent=2))
    (root/'run_plan.json').write_text(json.dumps(plan,indent=2))
    def run(case):
        out=root/f'{case["scene"]}_u{case["ues"]}_s{case["seed"]}_dt{args.path_update_ms}'
        status=out/'run_status.json'
        if args.skip_completed and status.exists():
            saved=out/'config_snapshot.json'
            if saved.exists():
                m=json.loads(saved.read_text())['manifest']
                saved_status=json.loads(status.read_text())
                saved_device=json.loads(saved.read_text())['configs']['phy']['rt_sys']['device']
                if (saved_status.get('completed') and saved_status.get('stage')=='completed' and
                    saved_status.get('source_sha256')==source_hash and saved_status.get('model')==args.model and
                    saved_status.get('fast_profile')==args.fast_profile and
                    saved_device==args.device and m['total_slots']==args.slots and m['path_update_ms']==args.path_update_ms):
                    return {'case':case,'skipped':True,'returncode':0}
        command=[sys.executable,str(ROOT/'experiments/run_rt_sys.py'),'--output',str(out),'--device',args.device,'--model',args.model]
        if args.fast_profile:command.extend(['--fast-profile',args.fast_profile])
        if args.fast_gain_update_ms is not None:command.extend(['--fast-gain-update-ms',str(args.fast_gain_update_ms)])
        for key,value in case.items(): command.extend(['--'+key.replace('_','-'),str(value)])
        out.mkdir(parents=True,exist_ok=True)
        with (out/'process.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        return {'case':case,'skipped':False,'returncode':result.returncode}

    # Device-wide free memory includes RT/Dr.Jit as well as Torch, but also
    # other processes and shared-memory effects on GB10. It is not a per-job
    # allocation counter, particularly when cases run concurrently.
    stopped=threading.Event()
    def monitor():
        import torch
        with (root/'device_memory_samples.jsonl').open('w') as log:
            while not stopped.is_set():
                row={'elapsed_s':time.monotonic()-started}
                try:
                    free,total=torch.cuda.mem_get_info()
                    row.update(cuda_global_free_bytes=free,cuda_global_total_bytes=total)
                except Exception as exc:row['cuda_memory_error']=str(exc)
                if Path('/proc/meminfo').exists():
                    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                    row['system_available_bytes']=int(mem['MemAvailable'].split()[0])*1024
                log.write(json.dumps(row)+'\n');log.flush()
                stopped.wait(1)
    started=time.monotonic()
    sampler=threading.Thread(target=monitor,daemon=True) if args.device=='cuda' else None
    if sampler:sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(run,plan))
    finally:
        stopped.set()
        if sampler:sampler.join()
    (root/'matrix_status.json').write_text(json.dumps({'workers':args.workers,
        'elapsed_s':time.monotonic()-started,'completed':all(r['returncode']==0 for r in results),
        'cases':results,'memory_scope':'device-wide; includes concurrent and unrelated allocations'},indent=2))
    if any(r['returncode'] for r in results):raise SystemExit('Matrix has failed cases; inspect per-case process.log')
