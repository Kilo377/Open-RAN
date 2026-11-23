#!/usr/bin/env python3
"""Retain owned channel data and verify every sealed raw file; delete nothing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_rcp_perturbation import atomic


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while chunk:=stream.read(8*1024**2):h.update(chunk)
    return h.hexdigest()


def mirror_cache(worker,artifact,remote_root):
    dest=artifact/'channel-cache'/worker['id'];dest.mkdir(parents=True,exist_ok=True)
    shell=shlex.join(['ssh','-S',worker['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=10'])
    command=['rsync','-a','--partial','--exclude','*.tmp','-e',shell,
        worker['host']+':'+remote_root+'/channel_cache/context_identification/',str(dest)+'/']
    result=subprocess.run(command,capture_output=True,text=True,timeout=300)
    if result.returncode not in(0,24):raise RuntimeError('Owned channel cache mirror failed: '+worker['id'])
    files=[p for p in dest.rglob('*')if p.is_file()]
    return dict(worker=worker['id'],files=len(files),bytes=sum(p.stat().st_size for p in files),
                files_sha256={str(p.relative_to(dest)):digest(p)for p in files})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--cache-only',action='store_true');a=parser.parse_args()
    module=importlib.util.spec_from_file_location('context_coordinator',ROOT/'scripts/coordinate-context-identification.py')
    coordinator=importlib.util.module_from_spec(module);module.loader.exec_module(coordinator)
    workers=coordinator.workers(a.artifact)
    with ThreadPoolExecutor(max_workers=2)as pool:
        caches=list(pool.map(lambda w:mirror_cache(w,a.artifact,coordinator.REMOTE_ROOT),workers))
    cache_report=dict(copied_unix_s=time.time(),workers=caches,raw_data_deleted=False)
    atomic(a.artifact/'channel-cache-retention.json',cache_report)
    if a.cache_only:
        print(json.dumps([dict(worker=r['worker'],files=r['files'],bytes=r['bytes'])for r in caches]));return
    rows=[];runtime=[];model_pins=None;partial_outputs=[];unstarted_optional=[]
    for worker in workers:
        study=a.artifact/'workers'/worker['id'];plan=json.loads((study/'plan.json').read_text());received=json.loads((study/'received-checkpoints.json').read_text())
        status=json.loads((study/'study_status.json').read_text())
        if status.get('running'):raise ValueError('A primary experiment is still running')
        priority=json.loads((study/'priority-status.json').read_text())if(study/'priority-status.json').exists()else{}
        if priority.get('phase')in('running','waiting_for_case_boundary','waiting_for_primary_completion'):raise ValueError('A supplemental queue is still active')
        archive=a.artifact/'deployment'/worker['id']/'source.tgz'
        if digest(archive)!=plan['source_archive_sha256']:raise ValueError('Retained source archive changed')
        for case in plan['cases']:
            if not(study/'checkpoints'/(case['name']+'.json')).exists():
                if case.get('optional')is True:unstarted_optional.append(dict(worker=worker['id'],case=case['name']))
                else:raise ValueError('Missing required primary checkpoint: '+case['name'])
        if model_pins is not None and model_pins!=plan['frozen_models']:raise ValueError('Different model releases across workers')
        model_pins=plan['frozen_models']
        for path in sorted((study/'checkpoints').glob('*.json')):
            seal=json.loads(path.read_text());folder=study/'cases'/path.stem
            if path.stem not in received or received[path.stem]['checkpoint_sha256']!=digest(path):raise ValueError('Unverified durable checkpoint: '+path.stem)
            for name,value in seal['files_sha256'].items():
                if digest(folder/name)!=value:raise ValueError('Retained raw file changed: '+str(folder/name))
            rows.append(dict(worker=worker['id'],case=path.stem,files=len(seal['files_sha256']),
                bytes=sum((folder/name).stat().st_size for name in seal['files_sha256']),duration_s=seal['case']['duration_s'],checkpoint_sha256=digest(path)))
        for folder in(study/'cases').iterdir():
            if folder.is_dir()and not(study/'checkpoints'/(folder.name+'.json')).exists():
                files=[p for p in folder.rglob('*')if p.is_file()]
                partial_outputs.append(dict(worker=worker['id'],case=folder.name,files=len(files),bytes=sum(p.stat().st_size for p in files),used_as_completed_measurement=False))
        code=r'''
import json,sys
from pathlib import Path
root=Path('/dev/shm/oran_gain_context_20261119/source');sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import load,sha,source_hash
from experiments.context_identification import resources
p=load(root/'outputs/context/plan.json')
print(json.dumps(dict(source_sha256=source_hash(root),expected_source_sha256=p['source_sha256'],models_unchanged=all(sha(root/name)==value for name,value in p['frozen_models'].items()),free=resources(root))))
'''
        proof=json.loads(coordinator.ssh(worker,code))
        if proof['source_sha256']!=proof['expected_source_sha256']or not proof['models_unchanged']:raise ValueError('Frozen remote source/model drift')
        runtime.append(dict(worker=worker['id'],**proof))
    if any(digest(ROOT/name)!=value for name,value in model_pins.items()):raise ValueError('Local model release changed')
    report=dict(schema='oran.context-final-retention.v1',verified_unix_s=time.time(),sealed_cases=len(rows),
        files_verified=sum(r['files']for r in rows),raw_bytes=sum(r['bytes']for r in rows),
        cumulative_run_duration_s=sum(r['duration_s']for r in rows),cases=rows,workers=runtime,
        frozen_model_sha256s=model_pins,channel_cache_manifest_sha256=digest(a.artifact/'channel-cache-retention.json'),
        cache_bytes=sum(r['bytes']for r in caches),source_archives={str(p.relative_to(a.artifact)):digest(p)for p in a.artifact.rglob('*.tgz')},
        retained_partial_outputs=partial_outputs,unstarted_optional_cases=unstarted_optional,
        raw_data_deleted=False,trained_models_deleted=False,global_environment_modified=False)
    atomic(a.artifact/'retention-verification.json',report)
    print(json.dumps({k:report[k]for k in('sealed_cases','files_verified','raw_bytes','cache_bytes','cumulative_run_duration_s')}))


if __name__=='__main__':main()
