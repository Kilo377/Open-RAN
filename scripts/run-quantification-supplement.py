#!/usr/bin/env python3
"""Prepare/admit/mirror the independent BCE/BCDE queue, preserving V2."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.quantification_supplement import assignment,verify_plan,read,sha
from experiments.run_rcp_perturbation import atomic

spec=importlib.util.spec_from_file_location('validation_coordinator',ROOT/'scripts/run-quantification-study.py');validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)
RUNNER=Path('/dev/shm/oran_quantification_20261122/supplement/runner.py')


def prepare(artifact,plan,pin):
    artifact=Path(artifact).resolve();v2=artifact.parent
    if artifact.name!='supplement'or v2.name!='v2'or(artifact/'deployment').exists():raise ValueError('New independent V2 supplement output required')
    verify_plan(plan,pin);artifact.mkdir(parents=True,exist_ok=True)
    runner=ROOT/'experiments/quantification_supplement.py';runnerpin=sha(runner);(artifact/'runner.py').write_bytes(runner.read_bytes())
    atomic(artifact/'supplement-plan.json',plan);schedule=assignment(plan,pin);manifest=[]
    code=r'''
import hashlib,json,sys
from pathlib import Path
c=json.loads(sys.argv[1]);c.update(json.loads(sys.stdin.buffer.read()));root=Path(c['worker']['project_root']);runner=Path(c['runner']);sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.quantification_study import validate_owned
from experiments.run_rcp_perturbation import atomic
if root!=Path('/dev/shm/oran_quantification_20261122/source-v2')or runner!=Path('/dev/shm/oran_quantification_20261122/supplement/runner.py'):raise ValueError('Owned frozen V2/supplement roots required')
original=json.loads((root/'outputs/quantification_study/plan.json').read_text());validate_owned(root,original)
if original['source_sha256']!=c['plan']['supplement']['source_sha256']or original['source_archive_sha256']!=c['plan']['supplement']['source_archive_sha256']:raise ValueError('Original source differs')
out=root/'outputs/prospective_supplement'
if out.exists()or runner.parent.exists():raise ValueError('Do not overwrite retained supplement source/output')
runner.parent.mkdir();runner.write_text(c['runner_source'])
if hashlib.sha256(runner.read_bytes()).hexdigest()!=c['runner_sha256']:raise ValueError('Wrapper pin differs')
out.mkdir();(out/'origins').mkdir();atomic(out/'plan.json',c['plan'])
for origin in c['origins']:
 raw=origin['bytes'].encode()
 if hashlib.sha256(raw).hexdigest()!=origin['sha256']:raise ValueError('Original OFF proof differs')
 (out/'origins'/(origin['origin_key']+'.json')).write_bytes(raw)
atomic(out/'environment.json',dict(source_sha256=original['source_sha256'],source_archive_sha256=original['source_archive_sha256'],runner_sha256=c['runner_sha256'],physical_simulation_started=False))
print(json.dumps(dict(prepared=True,cases=len(c['plan']['cases']),source_sha256=original['source_sha256'])))
'''
    for worker in validation.workers(v2):
        cases=schedule[worker['id']];worker={**worker,'output_subdir':'prospective_supplement','local_output':str(artifact/'workers'/worker['id'])}
        workerplan=dict(worker_id=worker['id'],project_root=worker['project_root'],supplement=plan,supplement_plan_sha256=pin,
            runner_sha256=runnerpin,cases=cases,precontrol_admission_sha256=None)
        origins=[dict(origin_key=o['origin_key'],sha256=o['origin_proof']['sha256'],bytes=Path(o['origin_proof']['path']).read_text())for o in plan['origins']if o['worker_id']==worker['id']]
        package=json.dumps(dict(runner_source=runner.read_text(),origins=origins)).encode()
        reply=json.loads(validation.ssh(worker,code,dict(worker=worker,runner=str(RUNNER),runner_sha256=runnerpin,plan=workerplan),raw=package))
        worker.update(supplement_plan_sha256=pin,runner_sha256=runnerpin);atomic(artifact/'deployment'/worker['id']/'worker-config.json',worker)
        for case in cases:manifest.append({**case,'worker_id':worker['id'],'run_path':str(artifact/'workers'/worker['id']/'jobs'/case['id']/'data/run'),
            'remote_run_path':worker['project_root']+'/outputs/prospective_supplement/jobs/'+case['id']+'/data/run',
            'source_sha256':plan['source_sha256'],'source_archive_sha256':plan['source_archive_sha256'],'runner_sha256':runnerpin})
        print(json.dumps(dict(worker=worker['id'],**reply)),flush=True)
    atomic(artifact/'cases-manifest.json',dict(schema='oran.prospective-supplement-cases.v1',supplement_plan_sha256=pin,
        parent_physical_plan_sha256=plan['parent_physical_plan_sha256'],source_version='v2',cases=manifest,after_values_used=False))
    atomic(artifact/'source-publication.json',dict(supplement_plan_sha256=pin,parent_physical_plan_sha256=plan['parent_physical_plan_sha256'],runner_sha256=runnerpin,
        source_sha256=plan['source_sha256'],source_archive_sha256=plan['source_archive_sha256'],original_source_modified=False,simulation_started=False))


def admit(artifact,pin):
    from experiments.supplement_admission import validate_full_admission
    certificate=validate_full_admission(artifact,pin)
    code=r'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['worker']['project_root']);sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.quantification_study import validate_owned,isolated_environment
from experiments.run_rcp_perturbation import atomic
original=json.loads((root/'outputs/quantification_study/plan.json').read_text());validate_owned(root,original)
out=root/'outputs/prospective_supplement';p=json.loads((out/'plan.json').read_text());runner=Path(c['runner']);certificate=c['certificate']
if (hashlib.sha256(runner.read_bytes()).hexdigest()!=p['runner_sha256']or certificate.get('runner_sha256')!=p['runner_sha256']
 or p['supplement_plan_sha256']!=certificate['supplement_plan_sha256']):raise ValueError('Independent wrapper/plan pin differs')
if original['source_sha256']!=certificate['source_sha256']or original['source_archive_sha256']!=certificate['source_archive_sha256']:raise ValueError('Frozen source differs from certificate')
marker=out/'precontrol-admission.json'
if marker.exists():
 if json.loads(marker.read_text())['sha256']!=certificate['sha256']:raise ValueError('Retain existing independent certificate')
else:
 if any((out/'jobs').glob('*/data/request.json')):raise ValueError('Supplement request predates prospective prediction seal')
 atomic(marker,certificate)
p['precontrol_admission_sha256']=certificate['sha256'];atomic(out/'plan.json',p)
status=json.loads((out/'study_status.json').read_text())if(out/'study_status.json').exists()else{};pid=status.get('worker_pid');alive=False
if isinstance(pid,int):
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:pass
if alive:print(json.dumps(dict(pid=pid,existing=True)))
else:
 with(out/'worker.log').open('ab')as stream:
  child=subprocess.Popen([c['worker']['python'],'-u',str(runner),'--worker-plan',str(out/'plan.json')],cwd=root,env=isolated_environment(root),stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
 print(json.dumps(dict(pid=child.pid,existing=False,starts_only_after_original_matrix=True)))
'''
    result=[]
    for worker in validation.workers(artifact):
        reply=json.loads(validation.ssh(worker,code,dict(worker=worker,runner=str(RUNNER),certificate=certificate)))
        result.append(dict(worker_id=worker['id'],**reply));atomic(artifact/'deployment'/worker['id']/'runner.json',result[-1])
    atomic(artifact/'admission.json',dict(certificate=certificate,workers=result,admitted_unix_s=time.time()));print(json.dumps(result),flush=True)


def monitor(artifact):
    while True:
        with ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(lambda worker:validation.mirror(worker,artifact),validation.workers(artifact)))
        atomic(artifact/'coordinator-status.json',dict(updated_unix_s=time.time(),workers=rows));print(json.dumps(rows),flush=True)
        if all(w.get('completed')and w.get('done')==w.get('verified')for w in rows):break
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in('prepare','admit','monitor'):group.add_argument('--'+name,action='store_true')
    args=parser.parse_args();plan=verify_plan(read(args.plan),args.plan_sha256)
    if args.prepare:prepare(args.artifact,plan,args.plan_sha256)
    elif args.admit:admit(args.artifact,args.plan_sha256)
    else:monitor(args.artifact)
