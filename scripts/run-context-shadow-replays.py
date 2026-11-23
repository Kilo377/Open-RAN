#!/usr/bin/env python3
"""Capture trained cold-activation proposals without deployment or source edits.

Pause only our own admission marker at a case boundary; preserve every running
case. Shadow replicas are retained separately and never counted as new seeds.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.context_identification import make_case
from experiments.run_rcp_perturbation import atomic
spec=importlib.util.spec_from_file_location('coord_shadow',ROOT/'scripts/coordinate-context-identification.py');coord=importlib.util.module_from_spec(spec);spec.loader.exec_module(coord)

REMOTE=r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);sys.path.insert(0,str(root))
from experiments.context_identification import case_command,external_simulations
from experiments.run_rcp_perturbation import atomic,load,sha,source_hash
out=root/'outputs/context';gate=out/'ADMISSION_ALLOWED';gate.unlink(missing_ok=True)
plan=load(out/'plan.json')
if source_hash(root)!=plan['source_sha256']:raise ValueError('Frozen source changed')
atomic(out/'shadow-plan.json',dict(cases=c['cases'],driver_sha256=c['driver_sha256'],source_sha256=plan['source_sha256']))
with (out/'shadow-driver.log').open('ab')as driverlog:
 command=[str(root/'.venv-orin/bin/python'),str(out/'shadow-launcher.py'),str(out/'shadow-plan.json')]
 child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=driverlog,stderr=driverlog,start_new_session=True)
print(json.dumps(dict(pid=child.pid,cases=len(c['cases']))))
'''

DRIVER=r'''
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from experiments.context_identification import case_command,external_simulations
from experiments.run_rcp_perturbation import atomic,load,sha
out=root/'outputs/context';gate=out/'ADMISSION_ALLOWED';plan=load(out/'plan.json');items=load(sys.argv[1])
status=dict(phase='waiting_for_case_boundary',cases=[]);atomic(out/'shadow-status.json',status)
try:
 while external_simulations():time.sleep(2)
 env=os.environ.copy();env.update(XDG_CACHE_HOME=str(root/'.cache'),TMPDIR=str(root/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(root/'.cache/torchinductor'))
 for case in items['cases']:
  folder=out/'cases'/case['name'];seal=out/'checkpoints'/(case['name']+'.json')
  if folder.exists():raise ValueError('Shadow output exists; do not overwrite')
  cmd=case_command(root,out,case,out/'specs'/('native.json'))+['--xapp-decision-preview']
  state={**case,'started_unix_s':time.time(),'phase':'running'};status['cases'].append(state);status['phase']='running'
  with (out/'logs'/(case['name']+'.log')).open('wb')as log:
   child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True);state['pid']=child.pid;atomic(out/'shadow-status.json',status)
   while child.poll()is None:
    try:state['slot']=load(folder/'live.json').get('slot')
    except (OSError,ValueError):pass
    atomic(out/'shadow-status.json',status);time.sleep(2)
   code=child.wait()
  state.update(exit_code=code,wall_s=time.time()-state['started_unix_s'],phase='complete'if code==0 else'failed')
  if code:raise RuntimeError('Shadow capture failed')
  run=load(folder/'run_status.json')
  if not run.get('completed')or not run.get('replay',{}).get('audits_passed')or run['source_sha256']!=plan['source_sha256']:raise ValueError('Unverified shadow run')
  atomic(seal,dict(completed=True,case=case,kind='shadow_only_not_an_independent_replication',worker_id=plan['worker_id'],source_sha256=plan['source_sha256'],finished_unix_s=time.time(),
   wall_s=state['wall_s'],files_sha256={str(p.relative_to(folder)):sha(p)for p in folder.rglob('*')if p.is_file()}))
 status['phase']='complete';status['models_unchanged']=all(sha(root/name)==digest for name,digest in plan['frozen_models'].items())
except Exception as e:status.update(phase='failed',error=repr(e))
finally:
 atomic(out/'shadow-status.json',status);gate.touch()
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();artifact=a.artifact
    schedule={'agx1':[('burst_2x',20261120)],'agx2':[('native_1x',20261120),('steady_2x',20261121)]}
    result=[]
    for w in coord.workers(artifact):
        cases=[make_case(ctx,seed,'native',stage='shadow_replay')for ctx,seed in schedule[w['id']]]
        upload="from pathlib import Path;import sys,json;c=json.loads(sys.argv[1]);p=Path(c['root'])/'outputs/context/shadow-launcher.py';p.write_text(c['driver']);print('saved')"
        coord.ssh(w,upload,dict(root=coord.REMOTE_ROOT,driver=DRIVER))
        reply=json.loads(coord.ssh(w,REMOTE,dict(root=coord.REMOTE_ROOT,cases=cases,driver_sha256=hashlib.sha256(DRIVER.encode()).hexdigest())))
        result.append(dict(worker_id=w['id'],**reply,entries=cases))
    atomic(artifact/'shadow-replay-plan.json',dict(workers=result,created_unix_s=time.time(),driver_sha256=hashlib.sha256(DRIVER.encode()).hexdigest(),not_used_as_independent_training_samples=True))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
