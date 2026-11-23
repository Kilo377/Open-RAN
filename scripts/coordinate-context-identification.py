#!/usr/bin/env python3
"""Initialize owned dual-AGX probes, await existing jobs, and durably mirror data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.context_identification import design,make_case,SCENARIOS
from experiments.run_rcp_perturbation import atomic

REMOTE_ROOT='/dev/shm/oran_gain_context_20261119/source'
DEADLINE=1795131020+12*3600


def ssh(w,code,payload=None):
    cmd=shlex.join(['python3','-c',code]+([json.dumps(payload)]if payload is not None else []))
    return subprocess.check_output(['ssh','-S',w['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=10',w['host'],cmd],text=True,timeout=90)


def workers(artifact):
    result=[]
    for name in ('agx1','agx2'):
        cfg=json.loads((artifact/'deployment'/name/'worker-config.json').read_text());cfg['id']=name;result.append(cfg)
    return result


def initialize(artifact):
    specs,grids,contrasts=design(ROOT)
    atomic(artifact/'design.json',dict(schema='oran.context-identification-design.v1',specs=specs,grids=grids,contrasts=contrasts,
        scenarios=SCENARIOS,deadline_unix_s=DEADLINE,primary_contexts=['native_1x','steady_2x','burst_2x'],
        primary_seeds=[20261122,20261123,20261124],requested_scope='All five RCPs and all current graph KPIs',
        scoring=dict(duration_s=1.8,warmup_end_ms=100,scoring_start_ms=400,scoring_end_ms=1600,followup_ms=200),
        inference='Independent-seed slopes/paired contrasts per scenario; no pooling across traffic regimes or windows'))
    code=r'''
from pathlib import Path
import json,sys,subprocess,platform
c=json.loads(sys.argv[1]);root=Path(c['root']);sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import source_hash,atomic,sha
out=root/'outputs/context';out.mkdir(parents=True,exist_ok=True)
if (out/'plan.json').exists():raise ValueError('Existing plan retained; use monitor/resume')
p=c['plan'];p['project_root']=str(root);p['output']=str(out);p['source_sha256']=source_hash(root)
release=root/'artifacts/xapps/chongyu/20261117';r=json.loads((release/'release.json').read_text())
p['frozen_models']={str((release/e['path']).relative_to(root)):e['sha256']for e in r['models'].values()}
for f,d in p['frozen_models'].items():
 if sha(root/f)!=d:raise ValueError('Model pin differs')
p['source_archive_sha256']=json.loads((root/'.smo-owned-worker.json').read_text())['sha256']
for name,s in p['specs'].items():atomic(out/'specs'/(name+'.json'),s)
atomic(out/'plan.json',p);atomic(out/'environment.json',dict(uname=list(platform.uname()),python=sys.version,source_sha256=p['source_sha256'],source_archive_sha256=p['source_archive_sha256'],immutable_models=p['frozen_models']))
with (out/'worker.log').open('ab')as log:
 child=subprocess.Popen([str(root/'.venv-orin/bin/python'),'-u',str(root/'experiments/context_identification.py'),'--worker-plan',str(out/'plan.json')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps(dict(pid=child.pid,source_sha256=p['source_sha256'],output=str(out),plan=p)))
'''
    for i,w in enumerate(workers(artifact)):
        seed=20261120+i
        cases=[make_case(ctx,seed,arm,stage='pilot')for ctx,arm in [('native_1x','native'),('steady_2x','native'),('burst_2x','native'),('burst_2x','qos_native'),('alternating_2x','native')]]
        plan=dict(schema='oran.context-worker-plan.v1',worker_id=w['id'],deadline_unix_s=DEADLINE,analysis_reserve_s=1500,
            specs=specs,grids=grids,contrasts=contrasts,cases=cases,sealed=False,population=True)
        result=json.loads(ssh(w,code,dict(root=REMOTE_ROOT,plan=plan)))
        atomic(artifact/'worker-plans'/(w['id']+'.json'),result['plan']);atomic(artifact/'deployment'/w['id']/'runner.json',{k:v for k,v in result.items()if k!='plan'})
        print(w['id'],'waiting worker',result['pid'],len(cases),'pilot cases',flush=True)


def mirrored(w,artifact):
    dest=artifact/'workers'/w['id'];dest.mkdir(parents=True,exist_ok=True)
    shell=shlex.join(['ssh','-S',w['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=10'])
    cmd=['rsync','-a','--partial','--exclude','worker.lock','--exclude','*.tmp','-e',shell,w['host']+':'+REMOTE_ROOT+'/outputs/context/',str(dest)+'/']
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    # rsync 24 is a source-side atomic temporary disappearing during a live
    # mirror. Completed case files still require the sealed SHA verification.
    if proc.returncode not in(0,24):return dict(worker=w['id'],error=proc.stderr[-1000:])
    received_path=dest/'received-checkpoints.json';received=json.loads(received_path.read_text())if received_path.exists()else{}
    mismatches=[]
    for p in (dest/'checkpoints').glob('*.json'):
        if p.stem in received:continue
        seal=json.loads(p.read_text());folder=dest/'cases'/p.stem
        bad=[rel for rel,d in seal['files_sha256'].items()if not(folder/rel).is_file()or hashlib.sha256((folder/rel).read_bytes()).hexdigest()!=d]
        if bad:
            retry=['rsync','-ac','--partial','-e',shell,w['host']+':'+REMOTE_ROOT+'/outputs/context/cases/'+p.stem+'/',str(folder)+'/']
            subprocess.run(retry,capture_output=True,timeout=120,check=True)
            bad=[rel for rel,d in seal['files_sha256'].items()if not(folder/rel).is_file()or hashlib.sha256((folder/rel).read_bytes()).hexdigest()!=d]
        if bad:mismatches.append(dict(case=p.stem,files=bad))
        else:received[p.stem]=dict(files=len(seal['files_sha256']),verified_unix_s=time.time(),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    atomic(received_path,received)
    s=json.loads((dest/'study_status.json').read_text())if(dest/'study_status.json').exists()else{}
    return dict(worker=w['id'],phase=s.get('phase'),verified=len(received),done=len(s.get('done',[])),failed=s.get('failed',[]),running=s.get('running',[]),queued=s.get('queued'),free=s.get('free'),mismatches=mismatches,completed=s.get('completed'))


def monitor(artifact):
    ws=workers(artifact);allowed=(artifact/'admission-approved.json').exists();last=None
    while time.time()<DEADLINE+300:
        if not allowed:
            try:
                state=json.load(urllib.request.urlopen('http://127.0.0.1:53505/api/deployment',timeout=15));b=state.get('latest_batch')or{}
                if not state.get('running_jobs') and b.get('phase') in('complete','complete_with_failures','cancelled'):
                    code="from pathlib import Path;Path('/dev/shm/oran_gain_context_20261119/source/outputs/context/ADMISSION_ALLOWED').touch();print('admitted')"
                    for w in ws:ssh(w,code)
                    atomic(artifact/'admission-approved.json',dict(previous_batch_id=b.get('id'),previous_batch_phase=b.get('phase'),approved_unix_s=time.time()));allowed=True
            except Exception as e:atomic(artifact/'admission-wait.json',dict(error=type(e).__name__,updated_unix_s=time.time()))
        with ThreadPoolExecutor(max_workers=2)as pool:
            futures=[pool.submit(mirrored,w,artifact)for w in ws];results=[]
            for f in futures:
                try:results.append(f.result())
                except Exception as e:results.append(dict(error=repr(e)))
        status=dict(admission_allowed=allowed,workers=results,updated_unix_s=time.time(),deadline_unix_s=DEADLINE)
        atomic(artifact/'coordinator-status.json',status)
        stamp=[(w.get('worker'),w.get('phase'),w.get('verified'),len(w.get('failed',[])))for w in results]
        if stamp!=last:print(json.dumps(status),flush=True);last=stamp
        if len(results)==2 and all(w.get('completed')for w in results):break
        time.sleep(30)


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--initialize',action='store_true');p.add_argument('--monitor',action='store_true');a=p.parse_args()
    if a.initialize:initialize(a.artifact)
    if a.monitor:monitor(a.artifact)


if __name__=='__main__':main()
