"""Frozen, outcome-blind physical arms for a preregistered intent study.

Runs trained xApps through their normal hot/RIC path. No intent-specific action,
fixed RCP, learning, or result-derived tuning is introduced. All writes live in
an owned RAM source copy; raw runs and sealed checksums remain available.
"""
from copy import deepcopy
from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.context_identification import SCENARIOS,external_simulations,resources
from experiments.run_rcp_perturbation import atomic,load,sha,source_hash

SCHEMA='oran.quantification-study-plan.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_plan(plan,pin):
    value=deepcopy(plan);recorded=value.pop('plan_sha256',None)
    if recorded!=pin or digest(value)!=pin or plan.get('schema')!=SCHEMA:
        raise ValueError('Preregistered physical plan digest differs')
    if plan.get('observe_slots')!=2000 or plan.get('baseline_slot')!=2040 or plan.get('total_slots')!=4440:
        raise ValueError('Unexpected preregistered observation protocol')
    if len(plan.get('arms',[]))!=20 or len({a['id']for a in plan['arms']})!=20:
        raise ValueError('Twenty distinct frozen arms required')
    if plan['arms'][0]!=dict(id='OFF',app_ids=[]):raise ValueError('OFF must be the first frozen arm')
    if plan.get('physical_run_count')!=len(plan['arms'])*len(plan['scenarios'])*len(plan['seeds']):
        raise ValueError('Physical run count differs from the frozen grid')
    registered=set(plan['apps'].values())
    for arm in plan['arms']:
        if len(arm['app_ids'])!=len(set(arm['app_ids'])) or not set(arm['app_ids'])<=registered:
            raise ValueError('Arm differs from the registered trained applications')
    if any(s not in SCENARIOS for s in plan['scenarios']):raise ValueError('Unknown traffic context')
    return plan


def assignment(plan,pin,worker_ids=('agx2','agx1')):
    """Block scenes/seeds across devices, then randomize arm execution order."""
    verify_plan(plan,pin)
    if len(worker_ids)!=2:raise ValueError('This frozen grid uses exactly two devices')
    result={w:[]for w in worker_ids};rng=random.Random(plan['randomization_seed'])
    # Every scene is represented on both devices and with both new seeds.
    for scene_index,scene in enumerate(plan['scenarios']):
        for seed_index,seed in enumerate(plan['seeds']):
            worker=worker_ids[(scene_index+seed_index)%2]
            arms=deepcopy(plan['arms'][1:]);rng.shuffle(arms)
            for arm in [plan['arms'][0],*arms]:
                key=f'{scene}_s{seed}';name=key+'_'+arm['id']
                result[worker].append(dict(id=hashlib.sha256((pin+'|'+name).encode()).hexdigest()[:32],
                    name=name,origin_key=key,scene=scene,seed=seed,arm=arm['id'],app_ids=arm['app_ids']))
    # Capture all independent native prefixes early; no AFTER values are read.
    for worker,cases in result.items():result[worker]=sorted(cases,key=lambda c:c['arm']!='OFF')
    return result


def isolated_environment(root):
    root=Path(root).resolve();cache=root/'.cache';tmp=root/'tmp'
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),
        XDG_CACHE_HOME=str(cache),CUDA_CACHE_PATH=str(cache/'cuda-orin'),CUPY_CACHE_DIR=str(cache/'cupy'),
        CUDA_CACHE_MAXSIZE=str(128*1024**2),TORCH_HOME=str(cache/'torch'),TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),
        TORCHINDUCTOR_CACHE_DIR=str(cache/'torchinductor'),TRITON_CACHE_DIR=str(cache/'triton'),
        NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'matplotlib'),DRJIT_CACHE_DIR=str(cache/'drjit'),
        MESA_SHADER_CACHE_DIR=str(cache/'mesa'),HF_HOME=str(cache/'huggingface'),PIP_CACHE_DIR=str(cache/'pip'),UV_CACHE_DIR=str(cache/'uv'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return env


def command(root,out,case,plan):
    root=Path(root);out=Path(out);job=out/'jobs'/case['id'];data=job/'data';run=data/'run'
    ctx=SCENARIOS[case['scene']]
    args=['bash',str(root/'scripts/orin/run-six-ue-cuda-nr.sh'),'--model','fast','--fast-profile','system',
        '--slots',str(plan['total_slots']),'--seed',str(case['seed']),'--traffic-load',str(ctx['traffic_load']),
        '--urllc-scenario',ctx['urllc_scenario'],'--ue-services','drx-mec','--xapp-pool','chongyu',
        '--ap-services-profile',str(root/'configs/rt_sys/ap_services_reference.json'),'--record-compression','gzip',
        '--live-xapps','--xapps-start','','--xapp-toggle-timeline',str(data/'timeline.json'),
        '--xapp-decision-preview','--xapp-preview-before-slot',str(plan['baseline_slot']),
        '--channel-cache-root',str(root/'channel_cache/quantification_study'/str(case['seed'])),
        '--replay-max-frames','100','--live-state',str(run/'live.json'),'--output',str(run)]
    if case['arm']!='OFF':args+=['--evaluation-origin-file',str(data/'expected_origin.json')]
    return args


def completed(out,case):
    seal=Path(out)/'checkpoints'/(case['id']+'.json');folder=Path(out)/'jobs'/case['id']
    if not seal.is_file():return False
    info=load(seal)
    return info.get('completed')is True and all((folder/name).is_file()and sha(folder/name)==pin for name,pin in info['files_sha256'].items())


def make_origin(out,case,plan,archive):
    from smo.evaluation_reference import prepare_origin
    batch=dict(seed=case['seed'],worker_source_archive_sha256=plan['source_archive_sha256'],
        members=[dict(job_id=case['id'],phase='complete',comparison=dict(baseline_origin='fresh_run_all_xapps_off',baseline=dict(slot=plan['physical']['baseline_slot'])))])
    result=prepare_origin(batch,Path(out)/'jobs',observe_slots=plan['physical']['observe_slots'],source_archive=archive)
    if result['status']!='ready':raise ValueError('OFF prefix is not certifiable: '+result.get('reason','unknown'))
    atomic(Path(out)/'origins'/(case['origin_key']+'.json'),result['proof'])
    atomic(Path(out)/'origins'/(case['origin_key']+'-before-preview.json'),dict(frame=result['frame'],preview=result['preview'],
        origin_id=result['proof']['origin_id'],after_kpis_used=False))
    return result['proof']['origin_id']


def validate_owned(root,plan):
    root=Path(root).resolve();scope=Path('/dev/shm/oran_quantification_20261122')
    if not root.is_relative_to(scope)or root==scope:raise ValueError('Use the dedicated owned RAM source copy')
    marker=load(root/'.smo-owned-worker.json')
    if marker.get('sha256')!=plan['source_archive_sha256']:raise ValueError('Owned source archive pin differs')
    if source_hash(root)!=plan['source_sha256']:raise ValueError('Frozen physical source differs')
    if any(sha(root/p)!=pin for p,pin in plan['frozen_models'].items()):raise ValueError('Frozen trained model differs')


def run_worker(plan_path):
    plan_path=Path(plan_path).resolve();out=plan_path.parent;plan=load(plan_path);root=Path(plan['project_root']).resolve()
    verify_plan(plan['physical'],plan['physical_plan_sha256']);validate_owned(root,plan)
    if out!=root/'outputs/quantification_study':raise ValueError('Unexpected owned study output')
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in('jobs','checkpoints','origins','partial','logs'):(out/name).mkdir(parents=True,exist_ok=True)
    env=isolated_environment(root)
    for name in('tmp','.cache'):(root/name).mkdir(exist_ok=True)
    status_path=out/'study_status.json';old=load(status_path)if status_path.exists()else{}
    status=dict(worker_id=plan['worker_id'],worker_pid=os.getpid(),phase='initializing',done=[],failed=old.get('failed',[]),running=[],
        started_unix_s=old.get('started_unix_s',time.time()),completed=False,physical_plan_sha256=plan['physical_plan_sha256'],
        source_sha256=plan['source_sha256'],after_kpis_read_for_development=False,raw_data_retained=True)
    seen=set();failed={c['id']for c in status['failed']};wall_per_slot=[]
    deadline=datetime.fromisoformat(plan['physical']['deadline_utc']).timestamp();reserve=plan.get('analysis_reserve_s',3600)
    def publish(**values):status.update(values,updated_unix_s=time.time(),free=resources(root));atomic(status_path,status)
    if any(c.get('arm')=='OFF'for c in status['failed']):
        publish(phase='failed_OFF_prefix',completed=True,running=[])
        return # A failed native prefix cannot silently be skipped on restart.
    while True:
        entries=[c for c in plan['cases']if c['id']not in seen and c['id']not in failed]
        publish(queued=len(entries))
        estimate=max(180,statistics.median(wall_per_slot[-8:])*plan['physical']['total_slots']*1.2)if wall_per_slot else 650
        if (out/'STOP').exists()or time.time()+reserve+estimate>deadline:
            publish(phase='budget_complete',completed=True,unstarted=len(entries));break
        if not entries:publish(phase='complete',completed=True);break
        if not(out/'ADMISSION_ALLOWED').is_file()or external_simulations():
            publish(phase='waiting_for_admission_or_external_work');time.sleep(3);continue
        allowed=entries if(out/'ALLOW_NONEMPTY').is_file()else[c for c in entries if c['arm']=='OFF']
        if not allowed:publish(phase='waiting_for_nonempty_admission');time.sleep(3);continue
        free=status['free']
        if free['available_memory_bytes']<8*1024**3 or free['free_work_bytes']<5*1024**3 or free['free_system_bytes']<256*1024**2:
            publish(phase='waiting_for_storage');time.sleep(5);continue
        case=allowed[0]
        if completed(out,case):seen.add(case['id']);status['done'].append(case['id']);continue
        validate_owned(root,plan)
        job=out/'jobs'/case['id'];data=job/'data';run=data/'run'
        if job.exists():job.rename(out/'partial'/(case['id']+'_'+str(time.time_ns())))
        data.mkdir(parents=True)
        atomic(data/'request.json',dict(case=case,physical_plan_sha256=plan['physical_plan_sha256'],
            source_archive_sha256=plan['source_archive_sha256'],observe_slots=plan['physical']['observe_slots']))
        atomic(data/'timeline.json',[dict(at_ms=plan['physical']['baseline_slot']*.5,operation='set',apps=[a.removeprefix('xapp:')for a in case['app_ids']],cell='*')])
        if case['arm']!='OFF':
            origin=out/'origins'/(case['origin_key']+'.json')
            if not origin.exists():raise ValueError('OFF prefix must be certified before a nonempty arm')
            atomic(data/'expected_origin.json',load(origin))
        args=command(root,out,case,plan['physical']);atomic(data/'command.json',dict(argv=args,
            cache_environment={k:v for k,v in env.items()if k.endswith('_DIR')or k.endswith('_HOME')or k in('TMPDIR','CUDA_CACHE_PATH','PYTHONDONTWRITEBYTECODE')}))
        started=time.time();state={**case,'started_unix_s':started,'slot':0}
        with (data/'simulation.log').open('wb')as log:
            process=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['pid']=process.pid;publish(phase='running',running=[state])
            while process.poll()is None:
                try:live=load(run/'live.json');state.update(slot=live.get('slot',0),stage=live.get('stage'))
                except (OSError,ValueError):pass
                publish()
                if (out/'STOP').exists()or time.time()>deadline-180:
                    os.killpg(process.pid,signal.SIGINT)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM)
                time.sleep(2)
            code=process.wait()
        state.update(exit_code=code,wall_s=time.time()-started)
        try:
            result=load(run/'run_status.json')
            if code or not result.get('completed')or not result.get('replay',{}).get('audits_passed'):
                raise ValueError('Physical run incomplete; inspect retained simulation.log')
            if result.get('source_sha256')!=plan['source_sha256']:raise ValueError('Run physical source pin differs')
            if case['arm']=='OFF':state['origin_id']=make_origin(out,case,plan,Path(plan.get('source_archive_path',root.parent/'source.tgz')))
            else:
                verified=load(run/'evaluation_origin_verified.json')
                if not verified.get('verified_before_activation')or verified['origin_id']!=load(data/'expected_origin.json')['origin_id']:
                    raise ValueError('Preactivation origin check did not pass')
                state['origin_id']=verified['origin_id']
            seal=dict(completed=True,case=case,worker_id=plan['worker_id'],wall_s=state['wall_s'],
                physical_plan_sha256=plan['physical_plan_sha256'],source_sha256=plan['source_sha256'],
                finished_unix_s=time.time(),origin_id=state['origin_id'],after_kpis_read_for_development=False,
                files_sha256={str(p.relative_to(job)):sha(p)for p in job.rglob('*')if p.is_file()})
            atomic(out/'checkpoints'/(case['id']+'.json'),seal);seen.add(case['id']);status['done'].append(case['id'])
            wall_per_slot.append(state['wall_s']/plan['physical']['total_slots'])
        except Exception as error:
            state['error']=str(error);status['failed'].append(state);failed.add(case['id'])
            if case['arm']=='OFF':publish(phase='failed_OFF_prefix',running=[],completed=True);break
        publish(running=[],seconds_per_slot=statistics.median(wall_per_slot[-8:])if wall_per_slot else None)
    publish(models_unchanged=all(sha(root/p)==pin for p,pin in plan['frozen_models'].items()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-plan',type=Path,required=True);a=p.parse_args();run_worker(a.worker_plan)


if __name__=='__main__':main()
