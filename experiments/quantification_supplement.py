"""Separate preregistered BCE/BCDE queue using the immutable V2 simulator.

The executable wrapper lives outside source-v2. No source, original physical
plan, original queue, model or original prediction seal is modified.
"""
from copy import deepcopy
from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_plan(plan,pin):
    value=deepcopy(plan);recorded=value.pop('plan_sha256',None)
    if recorded!=pin or digest(value)!=pin or plan.get('schema')!='oran.quantification-supplement-plan.v1':raise ValueError('Supplement plan digest differs')
    if (plan.get('source_version')!='v2'or plan['physical_run_count']!=12 or len(plan['origins'])!=6
        or plan['baseline_slot']!=2040 or plan['observe_slots']!=2000 or plan['total_slots']!=4440
        or plan.get('heldout_after_kpis_used')is not False or plan.get('original_plan_or_seals_changed')is not False):
        raise ValueError('Unexpected independent supplement protocol')
    expected=[dict(id=arm,app_ids=[plan['apps'][a]for a in arm])for arm in('BCE','BCDE')]
    if plan['arms']!=expected or len({o['origin_key']for o in plan['origins']})!=6:raise ValueError('Supplement arms/origins differ')
    return plan


def assignment(plan,pin):
    verify_plan(plan,pin);result={'agx1':[],'agx2':[]}
    # Round-robin contexts, preserving exactly the two registered controls.
    for arm in plan['arms']:
        for origin in plan['origins']:
            name=origin['origin_key']+'_'+arm['id']
            result[origin['worker_id']].append(dict(id=hashlib.sha256((pin+'|'+name).encode()).hexdigest()[:32],name=name,
                origin_key=origin['origin_key'],scene=origin['scene'],seed=origin['seed'],arm=arm['id'],app_ids=arm['app_ids'],
                group_id=plan['group_id_prefix']+arm['id'],origin_id=origin['origin_id']))
    if any(len(cases)!=6 for cases in result.values()):raise ValueError('Six supplemental cases per device required')
    return result


def run_worker(path):
    path=Path(path).resolve();out=path.parent;p=read(path);plan=verify_plan(p['supplement'],p['supplement_plan_sha256']);root=Path(p['project_root']).resolve()
    if root!=Path('/dev/shm/oran_quantification_20261122/source-v2')or out!=root/'outputs/prospective_supplement':raise ValueError('Frozen V2/new output directory required')
    if sha(__file__)!=p['runner_sha256']:raise ValueError('Owned supplement wrapper differs')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
    from experiments.quantification_study import command,completed,isolated_environment,validate_owned
    from experiments.context_identification import external_simulations,resources
    from experiments.run_rcp_perturbation import atomic
    original_out=root/'outputs/quantification_study';original=read(original_out/'plan.json')
    if original['physical_plan_sha256']!=plan['parent_physical_plan_sha256']:raise ValueError('Original physical plan differs')
    validate_owned(root,original)
    if original['source_sha256']!=plan['source_sha256']or original['source_archive_sha256']!=plan['source_archive_sha256']:raise ValueError('Supplement physical source differs')
    certificate=read(out/'precontrol-admission.json')
    if certificate['sha256']!=p['precontrol_admission_sha256']or certificate['supplement_plan_sha256']!=p['supplement_plan_sha256']:
        raise ValueError('Independent prospective supplement gate is missing')
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in('jobs','checkpoints','partial'):(out/name).mkdir(parents=True,exist_ok=True)
    env=isolated_environment(root);statuspath=out/'study_status.json';old=read(statuspath)if statuspath.exists()else{}
    status=dict(worker_id=p['worker_id'],worker_pid=os.getpid(),phase='initializing',done=[],failed=old.get('failed',[]),running=[],
        completed=False,supplement_plan_sha256=p['supplement_plan_sha256'],source_sha256=plan['source_sha256'],after_kpis_read_for_development=False,raw_data_retained=True)
    def publish(**kw):status.update(kw,updated_unix_s=time.time(),free=resources(root));atomic(statuspath,status)
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp();reserve=plan['analysis_reserve_s']
    if status['failed']:publish(phase='retained_failure_requires_review',completed=True);return
    while True:
        if(out/'STOP').exists()or time.time()+650>deadline-reserve:publish(phase='budget_complete',completed=True);return
        native=read(original_out/'study_status.json')
        if native.get('completed')and native.get('phase')=='complete'and not native.get('failed')and all(completed(original_out,c)for c in original['cases']):break
        publish(phase='waiting_for_original_matrix',running=[],original_completed=len(native.get('done',[])));time.sleep(3)
    for case in p['cases']:
        if completed(out,case):status['done'].append(case['id']);continue
        while external_simulations():
            if time.time()+650>deadline-reserve:publish(phase='budget_complete',completed=True);return
            publish(phase='waiting_for_external_simulation');time.sleep(3)
        if(out/'STOP').exists()or time.time()+650>deadline-reserve:publish(phase='budget_complete',completed=True);return
        validate_owned(root,original)
        if sha(__file__)!=p['runner_sha256']:raise ValueError('Frozen supplement wrapper changed')
        free=resources(root)
        if free['available_memory_bytes']<8*1024**3 or free['free_work_bytes']<5*1024**3 or free['free_system_bytes']<256*1024**2:
            publish(phase='storage_guard',completed=True);return
        job=out/'jobs'/case['id'];data=job/'data';run=data/'run'
        if job.exists():job.rename(out/'partial'/(case['id']+'_'+str(time.time_ns())))
        data.mkdir(parents=True)
        origin=out/'origins'/(case['origin_key']+'.json')
        expected=next(o for o in plan['origins']if o['origin_key']==case['origin_key'])
        if sha(origin)!=expected['origin_proof']['sha256']:raise ValueError('Original OFF proof pin differs')
        (data/'expected_origin.json').write_bytes(origin.read_bytes())
        atomic(data/'request.json',dict(case=case,supplement_plan_sha256=p['supplement_plan_sha256'],physical_plan_sha256=plan['parent_physical_plan_sha256'],source_archive_sha256=plan['source_archive_sha256'],observe_slots=plan['observe_slots']))
        atomic(data/'timeline.json',[dict(at_ms=plan['baseline_slot']*.5,operation='set',apps=[a.removeprefix('xapp:')for a in case['app_ids']],cell='*')])
        args=command(root,out,case,original['physical']);atomic(data/'command.json',dict(argv=args,cache_environment={k:v for k,v in env.items()if str(root)in v}))
        started=time.time();running={**case,'started_unix_s':started,'slot':0}
        with(data/'simulation.log').open('wb')as stream:
            child=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            running['pid']=child.pid;publish(phase='running',running=[running])
            while child.poll()is None:
                try:running['slot']=read(run/'live.json').get('slot',0)
                except(OSError,ValueError):pass
                publish()
                if(out/'STOP').exists()or time.time()>deadline-180:
                    os.killpg(child.pid,signal.SIGINT)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM)
                time.sleep(2)
        running.update(exit_code=child.returncode,wall_s=time.time()-started)
        try:
            statusrun=read(run/'run_status.json');guard=read(run/'evaluation_origin_verified.json')
            if child.returncode or not statusrun.get('completed')or not statusrun.get('replay',{}).get('audits_passed'):raise ValueError('Incomplete supplemental run; raw data retained')
            if statusrun['source_sha256']!=plan['source_sha256']or guard.get('verified_before_activation')is not True or guard['origin_id']!=case['origin_id']:
                raise ValueError('Supplement physical source/common OFF origin differs')
            seal=dict(schema='oran.prospective-supplement-run.v1',completed=True,case=case,worker_id=p['worker_id'],wall_s=running['wall_s'],
                supplement_plan_sha256=p['supplement_plan_sha256'],physical_plan_sha256=plan['parent_physical_plan_sha256'],source_sha256=plan['source_sha256'],
                source_archive_sha256=plan['source_archive_sha256'],runner_sha256=p['runner_sha256'],origin_id=guard['origin_id'],finished_unix_s=time.time(),
                after_kpis_read_for_development=False,files_sha256={str(f.relative_to(job)):sha(f)for f in job.rglob('*')if f.is_file()})
            atomic(out/'checkpoints'/(case['id']+'.json'),seal);status['done'].append(case['id']);publish(running=[])
        except Exception as exc:
            running['error']=f'{type(exc).__name__}: {exc}';status['failed'].append(running);publish(phase='failed_retained',running=[],completed=True);return
    publish(phase='complete',running=[],completed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker-plan',type=Path,required=True);run_worker(parser.parse_args().worker_plan)
