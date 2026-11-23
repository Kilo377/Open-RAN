#!/usr/bin/env python3
"""Prepare, launch and mirror the independently preregistered sleep probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.sleep_profile_calibration import assignment,verify_plan
from experiments.run_rcp_perturbation import atomic

spec=importlib.util.spec_from_file_location('validation_coordinator',ROOT/'scripts/run-quantification-study.py')
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)
REMOTE='/dev/shm/oran_quantification_20261122/source-profile-calib'


def prepare(artifact,plan,pin):
    if(artifact/'deployment').exists():raise ValueError('Retain prepared source; do not overwrite')
    artifact.mkdir(parents=True,exist_ok=True);archive=artifact/'source.tgz'
    with tarfile.open(archive,'w:gz')as tar:
        for name in validation.SOURCES:tar.add(ROOT/name,arcname=name,filter=lambda m:None if '__pycache__'in m.name or m.name.endswith('.pyc')else m)
    archive_pin=hashlib.sha256(archive.read_bytes()).hexdigest()
    pool=json.loads(Path('/Users/kilobao/Desktop/ORAN/SMO/evaluation-v2-20261121/workers-v2b/pool-config.json').read_text())
    selected={w['id']:w for w in pool['workers']};schedule=assignment(plan,pin)
    code=r'''
import hashlib,json,os,sys,tarfile
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);sys.dont_write_bytecode=True
if root!=Path('/dev/shm/oran_quantification_20261122/source-profile-calib')or root.exists():raise ValueError('New owned calibration source required')
venv=Path(c['venv'])
if not(venv/'bin/python').is_file():raise ValueError('Existing read-only environment required')
root.mkdir(parents=True);archive=root.parent/(root.name+'.tgz');archive.write_bytes(sys.stdin.buffer.read())
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['archive_sha256']:raise ValueError('Archive pin differs')
with tarfile.open(archive,'r:gz')as tar:
 for item in tar.getmembers():
  if Path(item.name).is_absolute()or'..'in Path(item.name).parts:raise ValueError('Invalid archive')
 tar.extractall(root)
(root/'.venv-orin').symlink_to(venv,target_is_directory=True)
(root/'.smo-owned-worker.json').write_text(json.dumps(dict(root=str(root),venv=str(venv),sha256=c['archive_sha256'])))
sys.path.insert(0,str(root))
from experiments.sleep_profile_calibration import verify_plan
from experiments.run_rcp_perturbation import atomic,source_hash,sha
from smo.evaluation_reference import dynamics_fingerprint
p=c['plan'];verify_plan(p['physical'],p['physical_plan_sha256']);p.update(project_root=str(root),source_sha256=source_hash(root),source_archive_path=str(archive),source_dynamics_sha256=dynamics_fingerprint(root))
release=root/'artifacts/xapps/chongyu/20261117';models=json.loads((release/'release.json').read_text())['models']
p['frozen_models']={str((release/e['path']).relative_to(root)):e['sha256']for e in models.values()}
if any(sha(root/f)!=pin for f,pin in p['frozen_models'].items()):raise ValueError('Frozen release pins differ')
out=root/'outputs/quantification_study';out.mkdir(parents=True);atomic(out/'plan.json',p)
(root/'tmp').mkdir();(root/'.cache').mkdir()
atomic(out/'environment.json',dict(python=sys.version,uname=list(os.uname()),source_sha256=p['source_sha256'],source_dynamics_sha256=p['source_dynamics_sha256'],source_archive_sha256=c['archive_sha256'],simulation_started=False))
print(json.dumps(dict(source_sha256=p['source_sha256'],source_dynamics_sha256=p['source_dynamics_sha256'],cases=len(p['cases']))))
'''
    cases_manifest=[]
    for ident,cases in schedule.items():
        original=selected[ident];worker={**original,'project_root':REMOTE,'local_output':str(artifact/'workers'/ident)}
        workerplan=dict(worker_id=ident,physical=plan,physical_plan_sha256=pin,cases=cases,source_archive_sha256=archive_pin)
        reply=json.loads(validation.ssh(worker,code,dict(root=REMOTE,venv=str(Path(worker['python']).parents[1]),archive_sha256=archive_pin,plan=workerplan),raw=archive.read_bytes(),timeout=180))
        worker.update(source_archive_sha256=archive_pin,physical_plan_sha256=pin,**reply);atomic(artifact/'deployment'/ident/'worker-config.json',worker)
        for case in cases:
            cases_manifest.append({**case,'worker_id':ident,'run_path':str(artifact/'workers'/ident/'jobs'/case['id']/'data/run'),
                'remote_run_path':REMOTE+'/outputs/quantification_study/jobs/'+case['id']+'/data/run','source_sha256':reply['source_sha256'],
                'source_archive_sha256':archive_pin,'source_dynamics_sha256':reply['source_dynamics_sha256']})
        print(json.dumps(dict(worker=ident,prepared=True,**reply)),flush=True)
    atomic(artifact/'cases-manifest.json',dict(schema='oran.profile-calibration-cases.v1',calibration_plan_sha256=pin,
        source_archive_sha256=archive_pin,scoring_window=dict(start_slot=plan['scoring_start_slot'],end_slot=plan['scoring_end_slot'],slot_duration_ms=.5),
        cases=cases_manifest,heldout_after_kpis_read=False))
    atomic(artifact/'source-publication.json',dict(source_archive_sha256=archive_pin,calibration_plan_sha256=pin,simulation_started=False,prepared_unix_s=time.time()))


def launch(artifact,pin):
    code=r'''
import json,os,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.sleep_profile_calibration import validate_owned,verify_plan
from experiments.quantification_study import isolated_environment
out=root/'outputs/quantification_study';p=json.loads((out/'plan.json').read_text());verify_plan(p['physical'],c['pin']);validate_owned(root,p)
s=json.loads((out/'study_status.json').read_text())if(out/'study_status.json').exists()else{}
pid=s.get('worker_pid');alive=False
if isinstance(pid,int):
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:pass
if alive:print(json.dumps(dict(pid=pid,existing=True)))
else:
 with(out/'worker.log').open('ab')as stream:
  child=subprocess.Popen([c['python'],'-u','-m','experiments.sleep_profile_calibration','--worker-plan',str(out/'plan.json')],cwd=root,
   env=isolated_environment(root),stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
 print(json.dumps(dict(pid=child.pid,existing=False)))
'''
    replies=[]
    for worker in validation.workers(artifact):
        reply=json.loads(validation.ssh(worker,code,dict(root=REMOTE,python=worker['python'],pin=pin)))
        result=dict(worker_id=worker['id'],**reply);replies.append(result);atomic(artifact/'deployment'/worker['id']/'runner.json',result)
    atomic(artifact/'admission.json',dict(calibration_plan_sha256=pin,admitted_unix_s=time.time(),workers=replies));print(json.dumps(replies),flush=True)


def monitor(artifact):
    while True:
        with ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(lambda worker:validation.mirror(worker,artifact),validation.workers(artifact)))
        atomic(artifact/'coordinator-status.json',dict(updated_unix_s=time.time(),workers=rows));print(json.dumps(rows),flush=True)
        if all(row.get('completed')and row.get('verified')==row.get('done')for row in rows):break
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    for flag in('prepare','launch','monitor'):group.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();plan=verify_plan(json.loads(args.plan.read_text()),args.plan_sha256)
    if args.prepare:prepare(args.artifact,plan,args.plan_sha256)
    elif args.launch:launch(args.artifact,args.plan_sha256)
    else:monitor(args.artifact)
