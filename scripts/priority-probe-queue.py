#!/usr/bin/env python3
"""Owned, budget-bounded priority-component supplements; never changes source."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def remote_run(path):
    path=Path(path).resolve();out=path.parent;root=out.parents[1];sys.path.insert(0,str(root))
    from experiments.context_identification import case_command,external_simulations,resources
    from experiments.rcp_perturbation import vector
    from experiments.run_rcp_perturbation import atomic,load,sha
    lock=(out/'priority-probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=load(path);base=load(out/'plan.json');gate=out/'ADMISSION_ALLOWED';reserve=plan.get('analysis_reserve_s',1500)
    status=dict(phase='waiting_for_case_boundary',cases=[]);atomic(out/'priority-status.json',status)
    try:
        if plan.get('defer_until_primary_drained'):
            status['phase']='waiting_for_primary_completion';atomic(out/'priority-status.json',status)
            while True:
                primary=load(out/'study_status.json')
                if not primary.get('running') and primary.get('queued')==0:break
                if time.time()+reserve+1400>base['deadline_unix_s']:
                    status['phase']='budget_complete_before_supplements';return
                time.sleep(5)
        gate.unlink(missing_ok=True)
        while external_simulations():time.sleep(2)
        if sha(out/'priority-component-stimulus.py')!=plan['driver_sha256']:raise ValueError('Stimulus driver pin changed')
        for case in plan['cases']:
            # Admit complete four-level seed blocks, not isolated favorable
            # levels that cannot support the prespecified range fit.
            identity=(case['context'],case['seed'])
            pending=[c for c in plan['cases']if(c['context'],c['seed'])==identity and not(out/'checkpoints'/(c['name']+'.json')).exists()]
            if pending and time.time()+sum(c.get('estimated_wall_s',450)for c in pending)+reserve>base['deadline_unix_s']:
                status['phase']='budget_complete';break
            if time.time()+case.get('estimated_wall_s',450)+reserve>base['deadline_unix_s']:
                status['phase']='budget_complete';break
            if min(resources(root)['free_work_bytes'],resources(root)['available_memory_bytes'])<8*1024**3:
                status['phase']='storage_complete';break
            folder=out/'cases'/case['name'];seal=out/'checkpoints'/(case['name']+'.json')
            if seal.exists():continue
            if folder.exists():raise ValueError('Existing unsealed output retained; choose a fresh supplement name')
            spec=out/'priority-specs'/(case['arm']+'.json');atomic(spec,case['timeline_spec'])
            safe_case={**case,'arm':'native'}
            command=case_command(root,out,safe_case,spec)
            j=command.index('--perturbation-spec');del command[j:j+2]
            args=['--device','cuda','--rt-device','cpu','--scene','munich','--cells','3','--ues','6',
                '--config',str(root/'configs/rt_sys/agx_munich_6ue_depth2_fast.json'),'--algebra-backend','torch',
                '--nr-execution','batched','--recording-mode','experiment','--diagnostic-window-slots','40',
                *command[2:],'--control-timeline',str(spec),'--direct-conflict-mitigation','priority']
            command=['bash','-c','source "$1"; shift; exec python "$@"','priority-probe',str(root/'scripts/orin/activate-gpu.sh'),
                str(out/'priority-component-stimulus.py'),'--owned-root',str(root),*args]
            env=os.environ.copy();env.update(XDG_CACHE_HOME=str(root/'.cache'),TMPDIR=str(root/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(root/'.cache/torchinductor'))
            item=dict(name=case['name'],phase='running',started_unix_s=time.time(),slot=0);status['cases'].append(item);status['phase']='running'
            with(out/'logs'/(case['name']+'.log')).open('wb')as log:
                proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True);item['pid']=proc.pid
                while proc.poll()is None:
                    try:item['slot']=load(folder/'live.json').get('slot')
                    except (OSError,ValueError):pass
                    atomic(out/'priority-status.json',status);time.sleep(2)
                code=proc.wait()
            item.update(exit_code=code,wall_s=time.time()-item['started_unix_s'],phase='complete'if code==0 else'failed')
            if code:raise RuntimeError('Priority probe failed; raw partial output retained')
            run=load(folder/'run_status.json')
            if not run.get('completed')or not run.get('replay',{}).get('audits_passed')or run['source_sha256']!=base['source_sha256']:raise ValueError('Unverified priority probe')
            shutil.copy2(out/'priority-component-stimulus.py',folder/'stimulus_driver.py');atomic(folder/'stimulus_command.json',dict(argv=command,driver_sha256=plan['driver_sha256']))
            target=vector(base['specs']['native'],perturbed=True);target.update(scheduler_mode='latency_aware',ue_scheduler_weights=case['timeline_spec']['ue_scheduler_weights'])
            atomic(seal,dict(completed=True,case=case,kind='identified_priority_component',stimulus_kind='fixed_scheduler_probe',
                stimulus_driver_sha256=plan['driver_sha256'],executed_target_vector=target,
                worker_id=base['worker_id'],source_sha256=base['source_sha256'],finished_unix_s=time.time(),wall_s=item['wall_s'],
                files_sha256={str(p.relative_to(folder)):sha(p)for p in folder.rglob('*')if p.is_file()}))
        else:status['phase']='complete'
        status['models_unchanged']=all(sha(root/name)==digest for name,digest in base['frozen_models'].items())
    except Exception as e:status.update(phase='failed',error=repr(e))
    finally:atomic(out/'priority-status.json',status);gate.touch()


def initialize(artifact,worker_id,cases,*,defer=False,reserve_s=1500):
    sys.path.insert(0,str(ROOT));from experiments.run_rcp_perturbation import atomic
    spec=importlib.util.spec_from_file_location('coord_priority',ROOT/'scripts/coordinate-context-identification.py');coord=importlib.util.module_from_spec(spec);spec.loader.exec_module(coord)
    w=next(w for w in coord.workers(artifact)if w['id']==worker_id)
    driver=(ROOT/'scripts/priority-component-stimulus.py').read_text();runner=Path(__file__).read_text()
    base=json.loads((artifact/'worker-plans'/(worker_id+'.json')).read_text());plan=dict(source_sha256=base['source_sha256'],driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),cases=cases,defer_until_primary_drained=defer,analysis_reserve_s=reserve_s)
    code=r'''
from pathlib import Path
import sys,json,subprocess
c=json.loads(sys.argv[1]);root=Path(c['root']);out=root/'outputs/context';sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import atomic,load
if (out/'priority-plan.json').exists():
 import time,shutil
 old=load(out/'priority-plan.json');history=out/'priority-plan-history'/str(time.time_ns());history.mkdir(parents=True)
 for name in ('priority-plan.json','priority-component-stimulus.py','priority-probe-queue.py'):
  if(out/name).exists():shutil.copy2(out/name,history/name)
 keep=[e for e in old['cases']if(out/'checkpoints'/(e['name']+'.json')).exists()]
 names={e['name']for e in keep};c['plan']['cases']=keep+[e for e in c['plan']['cases']if e['name']not in names]
(out/'priority-component-stimulus.py').write_text(c['driver']);(out/'priority-probe-queue.py').write_text(c['runner'])
atomic(out/'priority-plan.json',c['plan'])
if not c['plan'].get('defer_until_primary_drained'):(out/'ADMISSION_ALLOWED').unlink(missing_ok=True)
with(out/'priority-driver.log').open('ab')as log:
 child=subprocess.Popen([str(root/'.venv-orin/bin/python'),str(out/'priority-probe-queue.py'),'--remote-plan',str(out/'priority-plan.json')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps(dict(pid=child.pid,cases=len(c['plan']['cases']))))
'''
    result=json.loads(coord.ssh(w,code,dict(root=coord.REMOTE_ROOT,driver=driver,runner=runner,plan=plan)))
    atomic(artifact/'priority-supplements'/(worker_id+'-'+str(time.time_ns())+'.json'),dict(worker_id=worker_id,plan=plan,runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),launch=result))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--remote-plan',type=Path);p.add_argument('--artifact',type=Path);p.add_argument('--worker');p.add_argument('--cases',type=Path);p.add_argument('--defer-until-primary-drained',action='store_true');p.add_argument('--analysis-reserve-s',type=int,default=1500);a=p.parse_args()
    if a.remote_plan:remote_run(a.remote_plan)
    else:print(json.dumps(initialize(a.artifact,a.worker,json.loads(a.cases.read_text()),defer=a.defer_until_primary_drained,reserve_s=a.analysis_reserve_s),indent=2))


if __name__=='__main__':main()
