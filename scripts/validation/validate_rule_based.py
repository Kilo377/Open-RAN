"""Short gates and detached long acceptance; raw artifacts stay on the execution host."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from threading import Event,Lock
STOP=Event();CHILDREN=set();LOCK=Lock()
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint
from scripts.validation.validate_efficiency import audit,equivalent,read
from ric.near_rt.rule_based_v1 import CATALOG,PROFILES
SCENES=('simple_street_canyon','munich')


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def inventory(root):
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ('core','ric','phy','models','ran','state','traffic','configs','scripts','experiments','scenarios','tests')
        for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml')}


def run_one(out,spec,source=ROOT):
    scene,ues,slots,seed,profile=spec;label=f'{scene}_u{ues}_{slots}_s{seed}_{profile}'
    if STOP.is_set():raise RuntimeError('Acceptance stopped after a prior failure')
    folder=out/label
    if (folder/'acceptance.json').exists():
        saved=read(folder/'acceptance.json');assert saved['source_sha256']==source_fingerprint(source);return saved
    if folder.exists():raise FileExistsError(f'Partial output retained: {folder}')
    folder.mkdir(parents=True)
    config=ROOT/'configs/rt_sys/rule_based_v1'/f'{profile}.json'
    # Frozen reference uses the exact original disabled recording/physics config.
    if source!=ROOT:config=source/'configs/rt_sys/efficiency_experiment.json'
    cmd=[sys.executable,str(source/'experiments/run_rt_sys.py'),'--config',str(config),
        '--scene',scene,'--cells','3','--ues',str(ues),'--slots',str(slots),'--seed',str(seed),
        '--initial-queue-bytes','0','--ue-speed-mps','1.5','--warmup-ms',str(200 if slots>=1000 else 0),
        '--control-mode','baseline' if profile=='disabled' else 'orchestrated',
        '--feedback-mode','nr_direct','--nr-execution','batched','--output',str(folder)]
    start=time.perf_counter()
    with (folder/'process.log').open('w') as log:
        with subprocess.Popen(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT) as child:
            with LOCK:CHILDREN.add(child)
            try:
                if STOP.is_set():child.terminate()
                code=child.wait()
                if code:raise RuntimeError(f'Case exited {code}: {label}; see {folder}/process.log')
            finally:
                with LOCK:CHILDREN.discard(child)
    result=audit(folder);summary=read(folder/'final_summary.json');perf=read(folder/'performance.json')
    decisions=0;reasons={};requested=set();rejections={}
    if (folder/'rule_decisions.jsonl').exists():
        for line in (folder/'rule_decisions.jsonl').open():
            row=json.loads(line);decisions+=1;reason=row['reason'];reasons[reason]=reasons.get(reason,0)+1
            if reason=='action_requested':requested.add(row['kind'])
            assert reason not in ('unknown_required_kpi','unsupported_rt_action')
    for line in (folder/'control_receipts.jsonl').open():
        row=json.loads(line)
        if row.get('status')=='rejected':
            reason=row.get('reason','');rejections[reason]=rejections.get(reason,0)+1
            assert not reason.startswith(('unsupported_fields','invalid_ue_prb_limit','invalid_target_cell','power_vector_length'))
    if profile!='disabled':
        assert summary.get('rule_based_v1'), 'Rules silently missing'
        assert decisions>0,'No rule decision receipts'
        assert all(app.get('statistics') for app in summary['rule_based_v1'].values()),'An application never received executable public observations'
        assert all(app.get('kind') in CATALOG or app.get('kind')=='monitor' for app in summary['rule_based_v1'].values())
    result.update(case=label,scene=scene,ues=ues,seed=seed,profile=profile,path=str(folder),
        process_wall_s=time.perf_counter()-start,simulation_s=perf['simulation_s'],
        initialization_s=perf['initialization_s'],stages=perf['stages'],cache=summary['rt_sys'],
        rules=summary.get('rule_based_v1',{}),decisions=decisions,reasons=reasons,
        requested_kinds=sorted(requested),rejected_controls=rejections,
        missing_trigger_interpretation='Hold/sample insufficiency is not an optimization pass; controlled unit/physical tests exercise triggers.')
    write(folder/'acceptance.json',result);print('PASS',label,round(result['process_wall_s'],2),flush=True)
    return result


def specs(stage):
    if stage=='short':
        return [(s,n,128,512,'disabled') for s in SCENES for n in (4,24)]+[(s,4,128,512,k) for s in SCENES for k in CATALOG]+[(s,n,128,512,k) for s in SCENES for n in (4,24) for k in PROFILES]
    return ([(s,4,4000,512,k) for s in SCENES for k in CATALOG]+
        [(s,4,4000,seed,k) for seed in (512,513) for s in SCENES for k in PROFILES]+
        [(s,24,1000,512,k) for s in SCENES for k in PROFILES])


def estimate(cases,workers):
    rows=[]
    for scene,n,slots,seed,profile in specs('long'):
        match=next(r for r in cases if r['scene']==scene and r['ues']==n and r['profile']==profile)
        # Serial short extrapolation; conservative upper includes startup and RT cold-cache uncertainty.
        seconds=match['initialization_s']+match['simulation_s']*slots/128
        rows.append(dict(scene=scene,ues=n,slots=slots,seed=seed,profile=profile,serial_estimate_s=seconds))
    serial=sum(r['serial_estimate_s'] for r in rows)
    rt=sum(r['stages'].get('rt_generation',{}).get('wall_s',0) for r in cases)
    # Do not divide by worker count and claim a measured parallel speedup.
    return dict(cases=len(rows),total_slots=sum(r['slots'] for r in rows),workers=workers,
        estimated_wall_lower_s=serial/max(1,workers*.65),estimated_wall_upper_s=serial*1.4,
        serial_short_extrapolation_s=serial,short_rt_generation_s=rt,
        cold_cache_note='RT generation is included conservatively in short rate; long cache keys/trajectory length differ. No cold-cache speed claim.',
        rows=rows)


def main(a):
    if a.reference_root:a.reference_root=a.reference_root.resolve()
    if a.gate:a.gate=a.gate.resolve()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Evidence must stay in independent Spark project outputs')
    sha=source_fingerprint(ROOT);files=inventory(ROOT);start=time.time()
    status=dict(completed=False,structural_passed=False,stage=a.stage,source_sha256=sha,started_epoch=start,cases=[],workers=a.workers)
    def save():
        status['elapsed_s']=time.time()-start;write(out/'pipeline_status.json',status)
    save();write(out/'source_sha256.json',files)
    if not (out/'executed_sources.tar.gz').exists():
        with tarfile.open(out/'executed_sources.tar.gz','w:gz') as archive:
            for file in files:archive.add(ROOT/file,arcname=file)
    try:
        if a.stage=='long':
            gate=read(a.gate/'validation.json')
            assert gate['completed'] and gate['structural_passed'] and gate['source_sha256']==sha
            write(out/'runtime_estimate.json',estimate(gate['cases'],a.workers))
        elif a.reference_root:
            status['stage']='disabled_reference_regression';save()
            for scene in SCENES:
                for n in (4,24):
                    spec=(scene,n,128,512,'disabled')
                    before=run_one(a.reference_root/'outputs'/('rule_reference_'+out.name),spec,a.reference_root)
                    current=run_one(out,spec)
                    eq=equivalent(before['path'],current['path'])
                    status.setdefault('disabled_comparisons',[]).append(dict(scene=scene,ues=n,**eq));save()
        status['stage']=a.stage;save()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(run_one,out,spec):spec for spec in specs(a.stage)}
            try:
                for f in as_completed(futures):
                    status['cases'].append(f.result());save()
                    assert source_fingerprint(ROOT)==sha,'Source changed during acceptance'
            except BaseException:
                STOP.set()
                with LOCK:
                    for child in list(CHILDREN):child.terminate()
                for future in futures:future.cancel()
                raise
        assert inventory(ROOT)==files,'Source/config/test changed during acceptance'
        status.update(completed=True,structural_passed=True,stage='completed');save()
        write(out/'validation.json',status)
        if a.stage=='short':write(out/'long_estimate.json',estimate(status['cases'],a.long_workers))
        # Writer logs are outside inventory; final integrity verification is automatic, not agent polling.
        manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob('*') if p.is_file() and p.name!='SHA256.json'}
        write(out/'SHA256.json',manifest)
        for name,digest in manifest.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
    except BaseException as error:
        status.update(stage='failed',error=f'{type(error).__name__}: {error}');save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['short','long'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',type=Path);p.add_argument('--reference-root',type=Path)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=1);p.add_argument('--long-workers',type=int,default=2)
    a=p.parse_args()
    if a.stage=='long' and not a.gate:p.error('--gate required for long stage')
    main(a)
