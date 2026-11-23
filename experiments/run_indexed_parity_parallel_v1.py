"""Parallel read-only execution of an unchanged exact-parity comparator.

No simulation or remote operation occurs here.  Workers invoke the pinned
compare_pair directly.  Its complete results are retained, input seals are
freshly verified again, and the original compare_study assembles the same
stable grid using only those pinned results.  A genuine OFF pair is also
computed serially and its entire return value must match its pooled result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import indexed_physical_parity_v4 as core


def now():return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def value_sha(value):return hashlib.sha256(encoded(value)).hexdigest()


def save(path,value):
    path=Path(path)
    if path.exists():raise ValueError('Preserve prior output: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    path.chmod(0o444)
    return core.ref(path)


def code_pins(expected):
    paths=[Path(core.__file__),Path(core.exact.__file__),ROOT/'experiments/execution_parallel_parity.py']
    pins={str(p.resolve()):core.sha(p) for p in paths}
    if pins[str(Path(core.__file__).resolve())]!=expected:
        raise ValueError('The loaded core comparator differs from the explicitly pinned code')
    return pins


def require_pins(pins):
    if any(core.sha(path)!=pin for path,pin in pins.items()):
        raise ValueError('A pinned code or input file changed')


def build_jobs(reference,candidate):
    roots=[Path(reference).resolve(),Path(candidate).resolve()]
    if roots[0]==roots[1]:raise ValueError('Independent recordings are required')
    jobs=[]
    for worker in ('agx1','agx2'):
        plans=[root/'protocol/pilot'/(worker+'.json') for root in roots]
        values=[json.loads(p.read_text()) for p in plans]
        if not all(p.get('worker_id')==worker and all(c.get('worker_id')==worker for c in p['cases'])
                   and [c['arm'] for c in p['cases']]==list(core.ARMS) for p in values):
            raise ValueError('Require the exact same-host OFF+A..F grid')
        for old,new in zip(*(p['cases'] for p in values)):
            if (old['arm']!=new['arm'] or old['seed']!=new['seed']
                    or old['app_ids']!=core.ARMS[old['arm']] or new['app_ids']!=core.ARMS[new['arm']]):
                raise ValueError('Stable arm/seed/controller identity differs')
            cps=[root/'runs/workers'/worker/'checkpoints'/(case['id']+'.json')
                 for root,case in zip(roots,(old,new))]
            args=[str(p.resolve()) for p in (*cps,*plans)]
            jobs.append(dict(index=len(jobs),worker_id=worker,arm=old['arm'],seed=old['seed'],args=args,
                             input_pins={p:core.sha(p) for p in args}))
    return jobs


def compute_job(job,pins):
    """Spawn worker entry; no wrapper timing is inserted into core results."""
    require_pins(pins);require_pins(job['input_pins'])
    started=now();t=time.monotonic()
    result=core.compare_pair(*job['args'])
    require_pins(pins);require_pins(job['input_pins'])
    return dict(job=job,core_result=result,core_result_sha256=value_sha(result),execution=dict(
        pid=os.getpid(),started_utc=started,finished_utc=now(),wall_s=time.monotonic()-t,
        core_code_pins=pins,physical_comparison_function_unchanged=True))


def fresh_verify(jobs,computed):
    """Invalidate any cached result if a plan, seal or raw member changed."""
    for job in jobs:
        require_pins(job['input_pins'])
        result=computed[tuple(job['args'])]['core_result']
        for i,side in enumerate(('reference','candidate')):
            seal,plan,run,files,audit=core.verify(job['args'][i],job['args'][i+2],side)
            # Successful comparisons contain these original references.  A
            # failed admission is never promoted through cache assembly.
            if side in result:
                if (result[side]['checkpoint']['sha256']!=core.sha(job['args'][i])
                        or result[side]['plan']['sha256']!=core.sha(job['args'][i+2])):
                    raise ValueError('A computed pair belongs to another input revision')
        require_pins(job['input_pins'])


def cached_assembly(reference,candidate,output,oracle_proof,source_diff_proof,jobs,computed):
    expected={tuple(job['args']):computed[tuple(job['args'])]['core_result'] for job in jobs}
    seen=[]
    original=core.compare_pair
    def lookup(*args):
        key=tuple(str(Path(p).resolve()) for p in args)
        if key not in expected or key in seen:
            raise ValueError('Unexpected, duplicate or changed aggregation pair')
        seen.append(key)
        return deepcopy(expected[key])
    try:
        core.compare_pair=lookup
        result=core.compare_study(reference,candidate,output,oracle_proof=oracle_proof,source_diff_proof=source_diff_proof)
    finally:
        core.compare_pair=original
    if seen!=[tuple(j['args']) for j in jobs]:
        raise ValueError('Original aggregator did not consume the exact stable grid')
    return result


def wait_v3(source_diff_proof,timeout):
    path=Path(source_diff_proof).parent/'indexed-physical-parity-result-v1.json'
    deadline=time.monotonic()+timeout
    while True:
        try:
            value=json.loads(path.read_text())
            if len(value.get('trial_pairs',[]))==14:
                # This original helper also refuses every v3 difference other
                # than the nine explicitly authorized instrumentation aliases.
                core.schema_correction_evidence(path.parent)
                return core.ref(path)
        except (FileNotFoundError,json.JSONDecodeError):
            pass
        if time.monotonic()>=deadline:raise TimeoutError('Complete retained v3 result is not ready')
        time.sleep(.5)


def run(args):
    artifact=Path(args.artifact_dir).resolve()
    if artifact.exists() or Path(args.output).exists():raise ValueError('Choose new immutable execution/output paths')
    artifact.mkdir(parents=True)
    pins=code_pins(args.core_sha256)
    wrapper_pin=core.ref(__file__)
    jobs=build_jobs(args.reference,args.candidate)
    save(artifact/'execution-start.json',dict(schema='oran.parallel-exact-comparison-start.v1',created_utc=now(),
        wrapper=wrapper_pin,core_code_pins=pins,requested_processes=args.workers,jobs=jobs,
        physical_simulation_launched=False,remote_operations=False))
    computed={};saved=[];serial=None;serial_job=next(j for j in jobs if j['worker_id']=='agx2' and j['arm']=='OFF')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(compute_job,j,pins):j for j in jobs}
        # Real small serial/parallel regression, using one complete physical
        # OFF recording pair rather than an approximate synthetic substitute.
        serial=compute_job(serial_job,pins)
        serial_ref=save(artifact/'serial-agx2-OFF.json',serial)
        for future in as_completed(futures):
            job=futures[future]
            try:value=future.result()
            except BaseException as error:
                save(artifact/('execution-error-%02d.json'%job['index']),dict(job=job,error=type(error).__name__+': '+str(error)))
                raise
            if value.get('job')!=job:
                raise ValueError('A worker returned a result for another input pair')
            key=tuple(job['args']);computed[key]=value
            saved.append(save(artifact/('pair-%02d-%s-%s.json'%(job['index'],job['worker_id'],job['arm'])),value))
            print(json.dumps(dict(completed_pairs=len(computed),worker_id=job['worker_id'],arm=job['arm'],
                passed=value['core_result']['passed'],pid=value['execution']['pid'])),flush=True)
    pooled=computed[tuple(serial_job['args'])]
    regression_equal=encoded(serial['core_result'])==encoded(pooled['core_result'])
    regression=save(artifact/'serial-parallel-regression.json',dict(schema='oran.exact-comparator-serial-parallel-regression.v1',
        passed=regression_equal,case='agx2 OFF complete recorded pair',serial=serial_ref,
        serial_result_sha256=serial['core_result_sha256'],pooled_result_sha256=pooled['core_result_sha256'],
        number_tolerance=0,rounding=False,entire_core_return_value_compared=True))
    if not regression_equal:raise ValueError('Serial and pooled exact comparison results differ')
    previous=wait_v3(args.source_diff_proof,args.v3_wait_seconds)
    require_pins(pins);fresh_verify(jobs,computed)
    if core.sha(__file__)!=wrapper_pin['sha256']:raise ValueError('Execution wrapper changed while running')
    result=cached_assembly(args.reference,args.candidate,args.output,args.oracle_proof,args.source_diff_proof,jobs,computed)
    require_pins(pins)
    # Post-assembly check closes the small interval between the fresh sealed
    # cache validation and publication by the original aggregator.
    fresh_verify(jobs,computed)
    if encoded(json.loads(Path(args.output).read_text()))!=encoded(result):
        raise ValueError('The assembled result changed before execution sealing')
    evidence=dict(schema='oran.parallel-exact-comparison-execution.v1',created_utc=now(),completed=True,
        physical_result_passed=result['passed'],output=core.ref(args.output),wrapper=wrapper_pin,
        core_comparison=core.ref(core.__file__),core_code_pins=pins,requested_processes=args.workers,
        observed_worker_pids=sorted({v['execution']['pid'] for v in computed.values()}),
        aggregation_pid=os.getpid(),stable_pair_order=[dict(worker_id=j['worker_id'],arm=j['arm'],index=j['index'])for j in jobs],
        retained_pair_results=saved,serial_parallel_regression=regression,retained_original_v3_result=previous,
        complete_raw_seals_fresh_verified_before_and_after_assembly=True,core_function_bodies_or_files_modified=False,
        aggregation_pair_binding_temporarily_replaced_by_freshly_validated_result_lookup=True,
        post_data_instrumentation_schema_alias_correction=True,physical_simulations_launched=False,remote_operations=False)
    final=save(artifact/'execution-final.json',evidence)
    print(json.dumps(dict(passed=result['passed'],output=str(args.output),execution=final)),flush=True)
    return 0 if result['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','candidate','output','oracle-proof','source-diff-proof','artifact-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--core-sha256',required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--workers4',dest='workers',action='store_const',const=4)
    p.add_argument('--v3-wait-seconds',type=float,default=600)
    args=p.parse_args()
    if not 1<=args.workers<=4:p.error('Use one to four local analysis processes')
    if args.v3_wait_seconds<0:p.error('Use a nonnegative bounded original-v3 wait')
    try:return run(args)
    except BaseException as error:
        artifact=Path(args.artifact_dir)
        if artifact.is_dir():
            failure=artifact/'execution-failed.json'
            if not failure.exists():save(failure,dict(schema='oran.parallel-exact-comparison-execution-failed.v1',
                created_utc=now(),completed=False,error=type(error).__name__+': '+str(error),
                output_published=Path(args.output).exists(),physical_parity_admission_authorized=False))
        raise


if __name__=='__main__':raise SystemExit(main())
