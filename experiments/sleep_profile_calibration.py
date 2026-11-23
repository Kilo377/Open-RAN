"""Independent fixed-profile calibration, isolated from held-out validation."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.quantification_study import isolated_environment,completed,digest
from experiments.context_identification import SCENARIOS,resources,external_simulations
from experiments.run_rcp_perturbation import atomic,load,sha,source_hash
from experiments.sleep_profile_probe import SCHEMA,validate


def verify_plan(plan,pin):
    value=deepcopy(plan);recorded=value.pop('plan_sha256',None)
    if recorded!=pin or digest(value)!=pin or plan.get('schema')!='oran.parameter-profile-calibration-plan.v1':
        raise ValueError('Calibration preregistration differs')
    if (plan['training_seeds']!=[20261125,20261126,20261127] or set(plan['training_seeds'])&set(plan['validation_seeds_excluded'])
        or plan['switch_slot']!=2040 or plan['total_slots']!=4440 or plan['scoring_start_slot']!=2440 or plan['scoring_end_slot']!=4440
        or plan['reference_profile']!={cid:'active'for cid in('cell_0','cell_1','cell_2')}):
        raise ValueError('Calibration design differs from the independent fixed protocol')
    if set(plan['profiles'])!={'native_1x','steady_2x','burst_2x'}:raise ValueError('Three prespecified contexts required')
    return plan


def assignment(plan,pin):
    verify_plan(plan,pin);result={'agx2':[],'agx1':[]}
    for i,(scene,profiles)in enumerate(plan['profiles'].items()):
        for j,seed in enumerate(plan['training_seeds']):
            worker=('agx2','agx1')[(i+j)%2];pair=f'{scene}_s{seed}'
            for ident,profile in [('REF',plan['reference_profile']),*profiles.items()]:
                name=pair+'_'+ident
                result[worker].append(dict(id=hashlib.sha256((pin+'|'+name).encode()).hexdigest()[:32],name=name,
                    origin_key=pair,scene=scene,seed=seed,arm=ident,profile=profile))
    return result


def command(root,out,case,plan):
    root=Path(root);run=Path(out)/'jobs'/case['id']/'data/run';ctx=SCENARIOS[case['scene']]
    return ['bash',str(root/'scripts/orin/run-six-ue-cuda-nr.sh'),'--model','fast','--fast-profile','system',
        '--slots',str(plan['total_slots']),'--seed',str(case['seed']),'--traffic-load',str(ctx['traffic_load']),
        '--urllc-scenario',ctx['urllc_scenario'],'--ue-services','drx-mec','--xapp-pool','chongyu',
        '--ap-services-profile',str(root/'configs/rt_sys/ap_services_reference.json'),'--record-compression','gzip',
        '--sleep-profile-spec',str(run.parent/'spec.json'),
        '--channel-cache-root',str(root/'channel_cache/sleep_profile_calibration'/str(case['seed'])),
        '--replay-max-frames','100','--live-state',str(run/'live.json'),'--output',str(run)]


def validate_owned(root,p):
    root=Path(root).resolve()
    if root!=Path('/dev/shm/oran_quantification_20261122/source-profile-calib'):
        raise ValueError('Independent calibration RAM source required')
    if load(root/'.smo-owned-worker.json')['sha256']!=p['source_archive_sha256'] or source_hash(root)!=p['source_sha256']:
        raise ValueError('Frozen calibration source differs')
    if any(sha(root/f)!=pin for f,pin in p['frozen_models'].items()):raise ValueError('Reference model pins differ')


def run_worker(path):
    path=Path(path).resolve();out=path.parent;p=load(path);plan=verify_plan(p['physical'],p['physical_plan_sha256']);root=Path(p['project_root'])
    validate_owned(root,p)
    if out!=root/'outputs/quantification_study':raise ValueError('Owned calibration output required')
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in('jobs','checkpoints','pair-proofs','partial'):(out/name).mkdir(parents=True,exist_ok=True)
    env=isolated_environment(root);status_path=out/'study_status.json'
    old=load(status_path)if status_path.exists()else{}
    status=dict(worker_id=p['worker_id'],worker_pid=os.getpid(),phase='initializing',done=[],failed=old.get('failed',[]),running=[],
        completed=False,physical_plan_sha256=p['physical_plan_sha256'],source_sha256=p['source_sha256'],
        heldout_after_kpis_read=False,raw_data_retained=True)
    def publish(**kw):status.update(kw,updated_unix_s=time.time(),free=resources(root));atomic(status_path,status)
    if status['failed']:publish(phase='retained_failure_requires_review',completed=True);return
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    from experiments.profile_calibration_audit import audit_run,audit_pair
    for case in p['cases']:
        if completed(out,case):status['done'].append(case['id']);continue
        while external_simulations():
            if time.time()>deadline-3600:publish(phase='budget_complete',completed=True);return
            publish(phase='waiting_for_external_simulation',running=[]);time.sleep(3)
        if time.time()+420>deadline-3600 or(out/'STOP').exists():publish(phase='budget_complete',completed=True);return
        free=resources(root)
        if free['available_memory_bytes']<8*1024**3 or free['free_work_bytes']<5*1024**3 or free['free_system_bytes']<256*1024**2:
            publish(phase='storage_guard',completed=True);return
        validate_owned(root,p)
        job=out/'jobs'/case['id'];data=job/'data';run=data/'run'
        if job.exists():job.rename(out/'partial'/(case['id']+'_'+str(time.time_ns())))
        data.mkdir(parents=True)
        spec=validate(dict(schema=SCHEMA,profile_id=case['arm'],switch_slot=plan['switch_slot'],profile=case['profile'],calibration_plan_sha256=p['physical_plan_sha256']))
        atomic(data/'spec.json',spec);atomic(data/'request.json',dict(case=case,calibration_plan_sha256=p['physical_plan_sha256'],source_archive_sha256=p['source_archive_sha256']))
        args=command(root,out,case,plan);atomic(data/'command.json',dict(argv=args,cache_environment={k:v for k,v in env.items()if str(root)in v}))
        started=time.time();running={**case,'started_unix_s':started,'slot':0}
        with(data/'simulation.log').open('wb')as stream:
            child=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            running['pid']=child.pid;publish(phase='running',running=[running])
            while child.poll()is None:
                try:running['slot']=load(run/'live.json').get('slot',0)
                except(OSError,ValueError):pass
                publish()
                if(out/'STOP').exists()or time.time()>deadline-180:
                    os.killpg(child.pid,signal.SIGINT)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM)
                time.sleep(2)
        running.update(exit_code=child.returncode,wall_s=time.time()-started)
        try:
            result=load(run/'run_status.json')
            if child.returncode or not result.get('completed')or not result.get('replay',{}).get('audits_passed'):
                raise ValueError('Physical calibration run incomplete; retained logs available')
            if result['source_sha256']!=p['source_sha256']:raise ValueError('Run source differs')
            proof=audit_run(run,plan,case['scene'],case['seed'],case['arm']);atomic(data/'parameter-admission-proof.json',proof)
            if case['arm']!='REF':
                ref=next(c for c in p['cases']if c['origin_key']==case['origin_key']and c['arm']=='REF')
                if not completed(out,ref):raise ValueError('Paired reference must be sealed first')
                pair=audit_pair(out/'jobs'/ref['id']/'data/run',run,plan,case['scene'],case['seed'])
                atomic(out/'pair-proofs'/(case['origin_key']+'.json'),pair)
            seal=dict(completed=True,case=case,worker_id=p['worker_id'],wall_s=running['wall_s'],physical_plan_sha256=p['physical_plan_sha256'],
                source_sha256=p['source_sha256'],finished_unix_s=time.time(),heldout_after_kpis_read=False,
                files_sha256={str(f.relative_to(job)):sha(f)for f in job.rglob('*')if f.is_file()})
            atomic(out/'checkpoints'/(case['id']+'.json'),seal);status['done'].append(case['id']);publish(running=[])
        except Exception as exc:
            running['error']=f'{type(exc).__name__}: {exc}';status['failed'].append(running);publish(phase='failed_retained',running=[],completed=True);return
    publish(phase='complete',running=[],completed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker-plan',type=Path,required=True);run_worker(parser.parse_args().worker_plan)
