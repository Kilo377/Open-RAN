#!/usr/bin/env python3
"""Prepare/coordinate two isolated AGX workers; mirror raw data without labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.quantification_study import assignment,verify_plan
from experiments.run_rcp_perturbation import atomic

REMOTE_ROOT='/dev/shm/oran_quantification_20261122/source'
SOURCES=('core','phy','ran','ric','state','models','traffic','configs','experiments','scenarios','smo','scripts/orin','scripts/spark','artifacts/xapps/chongyu/20261117')


def ssh(worker,code,payload=None,*,raw=None,timeout=90):
    command=shlex.join(['python3','-c',code]+([json.dumps(payload)]if payload is not None else []))
    result=subprocess.run(['ssh','-S',worker['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=10',worker['host'],command],
        input=raw,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-2000:])
    return result.stdout.decode()


def workers(artifact):
    return [json.loads(p.read_text())for p in sorted((artifact/'deployment').glob('*/worker-config.json'))]


def round_robin_cases(cases):
    """Keep OFF first and each seeded block order; alternate traffic regimes."""
    offs=[c for c in cases if c['arm']=='OFF'];blocks={}
    for case in cases:
        if case['arm']!='OFF':blocks.setdefault(case['origin_key'],[]).append(case)
    result=list(offs)
    for index in range(max(map(len,blocks.values()),default=0)):
        result.extend(block[index]for block in blocks.values()if index<len(block))
    return result


def interleave(artifact,pin):
    code=r'''
import json,os,signal,subprocess,sys,time
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.quantification_study import isolated_environment,verify_plan,validate_owned,completed
from experiments.run_rcp_perturbation import atomic
out=root/'outputs/quantification_study';p=json.loads((out/'plan.json').read_text());s=json.loads((out/'study_status.json').read_text())
verify_plan(p['physical'],c['pin']);validate_owned(root,p)
if(out/'ALLOW_NONEMPTY').exists():raise ValueError('Queue order must be fixed before any nonempty admission')
if s.get('running')or s.get('phase')!='waiting_for_nonempty_admission':raise ValueError('Wait until all OFF cases finish and the worker is idle')
if not all(completed(out,case)for case in p['cases']if case['arm']=='OFF'):raise ValueError('All OFF prefixes must be sealed')
if s.get('failed'):raise ValueError('Resolve failures before amending the queue')
offs=[case for case in p['cases']if case['arm']=='OFF'];blocks={}
for case in p['cases']:
 if case['arm']!='OFF':blocks.setdefault(case['origin_key'],[]).append(case)
ordered=list(offs)
for index in range(max(map(len,blocks.values()),default=0)):
 ordered.extend(block[index]for block in blocks.values()if index<len(block))
before=[case['id']for case in p['cases']];after=[case['id']for case in ordered]
if sorted(before)!=sorted(after):raise ValueError('Queue amendment changed the registered physical arms')
pid=s.get('worker_pid')
if not isinstance(pid,int):raise ValueError('Owned idle worker PID unavailable')
if b'experiments.quantification_study'not in(Path('/proc')/str(pid)/'cmdline').read_bytes():raise ValueError('PID is not the owned worker')
os.kill(pid,signal.SIGTERM)
for _ in range(100):
 try:
  stat=(Path('/proc')/str(pid)/'stat').read_text()
  if stat.split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(.05)
else:raise ValueError('Idle worker did not exit; no duplicate runner started')
p['cases']=ordered;p['queue_order']='OFF_then_round_robin_origin_blocks_preserving_seeded_arm_order'
atomic(out/'queue-order-addendum.json',dict(physical_plan_sha256=c['pin'],source_sha256=p['source_sha256'],before=before,after=after,
 after_kpis_used=False,nonempty_controls_deployed=False,amended_unix_s=time.time(),reason='Maintain scene/seed coverage if wall budget truncates the queue'))
atomic(out/'plan.json',p);s.update(worker_pid=None,phase='queue_reordered_restarting',completed=False);atomic(out/'study_status.json',s)
with(out/'worker.log').open('ab')as log:
 child=subprocess.Popen([c['python'],'-u','-m','experiments.quantification_study','--worker-plan',str(out/'plan.json')],cwd=root,
  env=isolated_environment(root),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps(dict(pid=child.pid,old_pid=pid,cases=len(after),physical_cases_unchanged=True,source_sha256=p['source_sha256'])))
'''
    result=[]
    for w in workers(artifact):
        reply=json.loads(ssh(w,code,dict(root=w['project_root'],python=w['python'],pin=pin)))
        result.append(dict(worker_id=w['id'],**reply))
    atomic(artifact/'queue-order-amendment.json',dict(physical_plan_sha256=pin,workers=result,after_kpis_used=False,amended_unix_s=time.time()))
    print(json.dumps(result),flush=True)


def prepare(artifact,plan,pin,*,remote_root=REMOTE_ROOT):
    source=artifact/'source.tgz'
    if (artifact/'deployment').exists() and any((artifact/'deployment').glob('*/worker-config.json')):
        raise ValueError('Retain existing prepared source; use launch/monitor, not overwrite')
    with tarfile.open(source,'w:gz')as archive:
        for name in SOURCES:archive.add(ROOT/name,arcname=name,filter=lambda m:None if '__pycache__'in m.name or m.name.endswith('.pyc')else m)
    archive_pin=hashlib.sha256(source.read_bytes()).hexdigest()
    cfg=json.loads(Path('/Users/kilobao/Desktop/ORAN/SMO/evaluation-v2-20261121/workers-v2b/pool-config.json').read_text())
    selected={w['id']:w for w in cfg['workers']};schedule=assignment(plan,pin)
    code=r'''
import hashlib,json,os,sys,tarfile
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);scope=Path('/dev/shm/oran_quantification_20261122')
if not root.is_relative_to(scope)or root==scope:raise ValueError('Owned RAM scope required')
if root.exists():raise ValueError('Do not overwrite an existing source copy')
venv=Path(c['venv'])
if not(venv/'bin/python').is_file():raise ValueError('Read-only environment is unavailable')
root.mkdir(parents=True);archive=root.parent/(root.name+'.tgz');archive.write_bytes(sys.stdin.buffer.read())
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['archive_sha256']:raise ValueError('Source archive checksum differs')
with tarfile.open(archive,'r:gz')as a:
 for member in a.getmembers():
  if Path(member.name).is_absolute()or'..'in Path(member.name).parts:raise ValueError('Invalid archive path')
 a.extractall(root)
(root/'.venv-orin').symlink_to(venv,target_is_directory=True)
(root/'.smo-owned-worker.json').write_text(json.dumps(dict(root=str(root),venv=str(venv),sha256=c['archive_sha256'])))
sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import atomic,sha,source_hash
from experiments.quantification_study import isolated_environment,verify_plan
p=c['plan'];verify_plan(p['physical'],p['physical_plan_sha256']);p['project_root']=str(root);p['source_sha256']=source_hash(root);p['source_archive_path']=str(archive)
release=root/'artifacts/xapps/chongyu/20261117';r=json.loads((release/'release.json').read_text())
p['frozen_models']={str((release/e['path']).relative_to(root)):e['sha256']for e in r['models'].values()}
if any(sha(root/f)!=d for f,d in p['frozen_models'].items()):raise ValueError('Frozen models differ')
out=root/'outputs/quantification_study';out.mkdir(parents=True)
atomic(out/'plan.json',p);(root/'tmp').mkdir(exist_ok=True);(root/'.cache').mkdir(exist_ok=True)
atomic(out/'environment.json',dict(python=sys.version,uname=list(os.uname()),source_sha256=p['source_sha256'],source_archive_sha256=c['archive_sha256'],
 frozen_models=p['frozen_models'],cache_environment={k:v for k,v in isolated_environment(root).items()if str(root)in v},simulation_started=False))
print(json.dumps(dict(source_sha256=p['source_sha256'],cases=len(p['cases']),output=str(out))))
'''
    for worker_id,cases in schedule.items():
        original=selected[worker_id];w={**original,'project_root':remote_root,'python':original['python']}
        planworker=dict(worker_id=worker_id,physical=plan,physical_plan_sha256=pin,cases=cases,source_archive_sha256=archive_pin,analysis_reserve_s=3600)
        reply=json.loads(ssh(w,code,dict(root=remote_root,venv=str(Path(w['python']).parents[1]),archive_sha256=archive_pin,plan=planworker),raw=source.read_bytes(),timeout=180))
        w.update(source_archive_sha256=archive_pin,physical_plan_sha256=pin,local_output=str(artifact/'workers'/worker_id),**reply)
        atomic(artifact/'deployment'/worker_id/'worker-config.json',w)
        print(json.dumps(dict(worker=worker_id,prepared=True,**reply)),flush=True)
    atomic(artifact/'source-publication.json',dict(source_archive_sha256=archive_pin,physical_plan_sha256=pin,simulation_started=False,prepared_unix_s=time.time()))


def admission(artifact,pin,*,full=False):
    precontrol=None
    if full:
        from experiments.precontrol_admission import validate_full_admission
        precontrol=validate_full_admission(artifact,pin)
        # Check both workers before mutating either gate. The remote launcher
        # repeats this check to close a stale-local-mirror admission race.
        for worker in workers(artifact):
            check=r'''
import json,sys
from pathlib import Path
c=json.loads(sys.argv[1]);out=Path(c['root'])/'outputs/quantification_study'
sys.dont_write_bytecode=True;sys.path.insert(0,c['root'])
from experiments.context_identification import external_simulations
if external_simulations():raise ValueError('Another physical simulation is still running')
marker=out/'precontrol-admission.json'
if marker.exists():
 if json.loads(marker.read_text())['sha256']!=c['precontrol']['sha256']:raise ValueError('Existing admission certificate differs')
else:
 s=json.loads((out/'study_status.json').read_text())
 if s.get('running')or s.get('phase')!='waiting_for_nonempty_admission':raise ValueError('Worker must be idle after all OFF prefixes')
 for path in(out/'jobs').glob('*/data/request.json'):
  if json.loads(path.read_text())['case']['arm']!='OFF':raise ValueError('A nonempty request predates prospective admission')
 if(out/'ALLOW_NONEMPTY').exists():raise ValueError('Nonempty gate was opened without a prospective certificate')
print(json.dumps(dict(nonempty_requests_before_first_admission=0,worker_idle=True)))
'''
            ssh(worker,check,dict(root=worker['project_root'],precontrol=precontrol))
    code=r'''
import json,os,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.quantification_study import isolated_environment,verify_plan,validate_owned
out=root/'outputs/quantification_study';p=json.loads((out/'plan.json').read_text());verify_plan(p['physical'],c['pin']);validate_owned(root,p)
if c['full']:
 from experiments.context_identification import external_simulations
 if external_simulations():raise ValueError('Another physical simulation is still running')
 certificate=c.get('precontrol')
 if not certificate or certificate['source_sha256']!=p['source_sha256']or certificate['source_archive_sha256']!=p['source_archive_sha256']:raise ValueError('Prospective certificate source differs')
 marker=out/'precontrol-admission.json'
 if marker.exists():
  if json.loads(marker.read_text())['sha256']!=certificate['sha256']:raise ValueError('Existing prospective certificate differs')
 else:
  s=json.loads((out/'study_status.json').read_text())
  if s.get('running')or s.get('phase')!='waiting_for_nonempty_admission':raise ValueError('Worker must be idle before controls')
  if any(json.loads(path.read_text())['case']['arm']!='OFF'for path in(out/'jobs').glob('*/data/request.json')):raise ValueError('Nonempty request predates admission')
  if(out/'ALLOW_NONEMPTY').exists():raise ValueError('Gate opened before prospective admission')
  marker.write_text(json.dumps(certificate))
(out/'ADMISSION_ALLOWED').touch()
if c['full']:(out/'ALLOW_NONEMPTY').touch()
status=out/'study_status.json';existing=json.loads(status.read_text())if status.exists()else{}
pid=existing.get('worker_pid');alive=False
if isinstance(pid,int):
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:pass
if alive:print(json.dumps(dict(pid=pid,existing=True,full=c['full'])))
else:
 with (out/'worker.log').open('ab')as log:
  process=subprocess.Popen([c['python'],'-u','-m','experiments.quantification_study','--worker-plan',str(out/'plan.json')],cwd=root,
   env=isolated_environment(root),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 print(json.dumps(dict(pid=process.pid,existing=False,full=c['full'])))
'''
    result=[]
    for w in workers(artifact):
        reply=json.loads(ssh(w,code,dict(root=w['project_root'],python=w['python'],pin=pin,full=full,precontrol=precontrol)))
        result.append(dict(worker_id=w['id'],**reply));atomic(artifact/'deployment'/w['id']/'runner.json',result[-1])
    atomic(artifact/('full-admission.json'if full else'off-admission.json'),dict(physical_plan_sha256=pin,workers=result,admitted_unix_s=time.time(),precontrol=precontrol))
    print(json.dumps(result),flush=True)


def mirror(w,artifact):
    dest=artifact/'workers'/w['id'];dest.mkdir(parents=True,exist_ok=True)
    shell=shlex.join(['ssh','-S',w['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=10'])
    remote=w['host']+':'+w['project_root']+'/outputs/'+w.get('output_subdir','quantification_study')+'/'
    reply=subprocess.run(['rsync','-a','--partial','--exclude','worker.lock','--exclude','*.tmp','-e',shell,remote,str(dest)+'/'],capture_output=True,text=True,timeout=180)
    if reply.returncode not in(0,24):return dict(worker_id=w['id'],error=reply.stderr[-1200:])
    received_path=dest/'received-checkpoints.json';received=json.loads(received_path.read_text())if received_path.exists()else{}
    errors=[]
    for path in(dest/'checkpoints').glob('*.json'):
        if path.stem in received:continue
        seal=json.loads(path.read_text());folder=dest/'jobs'/path.stem
        bad=[p for p,pin in seal['files_sha256'].items()if not(folder/p).is_file()or hashlib.sha256((folder/p).read_bytes()).hexdigest()!=pin]
        if bad:
            subprocess.run(['rsync','-ac','--partial','-e',shell,remote+'jobs/'+path.stem+'/',str(folder)+'/'],capture_output=True,timeout=180,check=True)
            bad=[p for p,pin in seal['files_sha256'].items()if not(folder/p).is_file()or hashlib.sha256((folder/p).read_bytes()).hexdigest()!=pin]
        if bad:errors.append(dict(case_id=path.stem,files=bad))
        else:received[path.stem]=dict(case=seal['case'],files=len(seal['files_sha256']),verified_unix_s=time.time(),wall_s=seal['wall_s'])
    atomic(received_path,received)
    s=json.loads((dest/'study_status.json').read_text())if(dest/'study_status.json').exists()else{}
    return dict(worker_id=w['id'],phase=s.get('phase'),worker_pid=s.get('worker_pid'),done=len(s.get('done',[])),verified=len(received),
        failed=s.get('failed',[]),running=s.get('running',[]),queued=s.get('queued'),free=s.get('free'),completed=s.get('completed'),mismatches=errors)


def monitor(artifact,plan):
    from datetime import datetime
    lock=(artifact/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp();last=None
    while time.time()<deadline+300:
        with ThreadPoolExecutor(max_workers=2)as pool:
            futures=[pool.submit(mirror,w,artifact)for w in workers(artifact)];result=[]
            for f in futures:
                try:result.append(f.result())
                except Exception as error:result.append(dict(error=repr(error)))
        status=dict(workers=result,updated_unix_s=time.time(),deadline_unix_s=deadline,labels_unblinded=False)
        atomic(artifact/'coordinator-status.json',status)
        signature=[(r.get('worker_id'),r.get('phase'),r.get('done'),r.get('verified'),len(r.get('failed',[])))for r in result]
        if signature!=last:print(json.dumps(status),flush=True);last=signature
        if len(result)==2 and all(r.get('completed')for r in result):break
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--version',choices=('v1','v2'),default='v1')
    p.add_argument('--prepare',action='store_true');p.add_argument('--launch-off',action='store_true');p.add_argument('--admit-full',action='store_true');p.add_argument('--interleave-queue',action='store_true');p.add_argument('--monitor',action='store_true');a=p.parse_args()
    base=a.artifact.resolve();plan_bytes=(base/'physical-plan.json').read_bytes();plan=json.loads(plan_bytes);verify_plan(plan,a.plan_sha256)
    artifact=base if a.version=='v1'else base/'versions'/a.version
    artifact.mkdir(parents=True,exist_ok=True)
    if artifact!=base:
        saved=artifact/'physical-plan.json'
        if saved.exists()and saved.read_bytes()!=plan_bytes:raise ValueError('Do not replace a versioned preregistration')
        if not saved.exists():saved.write_bytes(plan_bytes)
    remote_root=REMOTE_ROOT if a.version=='v1'else REMOTE_ROOT+'-'+a.version
    if a.prepare:prepare(artifact,plan,a.plan_sha256,remote_root=remote_root)
    if a.interleave_queue:interleave(artifact,a.plan_sha256)
    if a.launch_off or a.admit_full:admission(artifact,a.plan_sha256,full=a.admit_full)
    if a.monitor:monitor(artifact,plan)


if __name__=='__main__':main()
