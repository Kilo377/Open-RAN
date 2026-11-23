#!/usr/bin/env python3
"""Replace the owned local UI after credential/source preflight; keep secrets off disk."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_rcp_perturbation import atomic


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--port',type=int,default=53505);p.add_argument('--metadata',type=Path,default=Path('/Users/kilobao/Desktop/ORAN/SMO/current-service.json'));a=p.parse_args()
    manifest=read(a.manifest);before={}
    for scene in manifest['scenarios']:
        path=Path(scene['intent_store'])if scene.get('intent_store')else None
        if path and path.exists():before[str(path)]=read(path)
    previous=read(a.metadata)if a.metadata.exists()else{}
    atomic(a.artifact/'ui-previous-service.json',previous)
    result=subprocess.run(['lsof','-nP',f'-iTCP:{a.port}','-sTCP:LISTEN','-Fp'],capture_output=True,text=True)
    if result.returncode not in(0,1):raise RuntimeError('Cannot identify the current local UI listener')
    listeners=sorted({int(line[1:])for line in result.stdout.splitlines()if line.startswith('p')})
    if len(listeners)>1:raise ValueError('More than one process owns the UI port')
    if listeners:
        # Preserve the operator's latest viewed run and model preference, not
        # an earlier template snapshot. This reads no credential material.
        with urlopen(f'http://127.0.0.1:{a.port}/api/graph',timeout=30)as response:current=json.load(response)
        for scene in manifest['scenarios']:
            if scene['id']!=manifest['default']:continue
            job=current['source'].get('deployment_job_id')
            if job:scene['retained_job_id']=job
            else:scene.pop('retained_job_id',None)
            origin=current.get('evaluation_origin',{})
            if origin.get('status')=='ready':scene['prepared_origin_slots']=origin['observe_slots']
            else:scene.pop('prepared_origin_slots',None)
        manifest['llm_model']=current['intent_service']['model'];atomic(a.manifest,manifest)
    key=getpass.getpass('DeepSeek session key (memory only): ')
    command=[sys.executable,'-u','-m','smo.scenario_service','--manifest',str(a.manifest),'--port',str(a.port),'--session-key-stdin','--verify-language']
    if listeners:command+=['--replace-pid',str(listeners[0])]
    logpath=a.artifact/('ui-service-'+str(time.time_ns())+'.log')
    with logpath.open('wb')as log:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=log,start_new_session=True,text=True)
        child.stdin.write(key+'\n');child.stdin.close();del key
    atomic(a.artifact/'ui-startup-process.json',dict(pid=child.pid,log=str(logpath),args=command,started_unix_s=time.time()))
    started=time.time();ready=None
    while time.time()-started<180:
        for line in logpath.read_text(errors='replace').splitlines():
            if not line.startswith('{'):continue
            try:row=json.loads(line)
            except ValueError:continue
            if row.get('ready')is True:ready=row
        if ready:break
        if child.poll()is not None:
            print(json.dumps(dict(ready=False,pid=child.pid,log=str(logpath))),flush=True)
            raise SystemExit(1)
        time.sleep(1)
    if ready is None:raise RuntimeError('New UI startup has not completed; inspect its owned log')
    with urlopen(f'http://127.0.0.1:{a.port}/api/intent/status',timeout=15)as response:language=json.load(response)
    if not language.get('verified')or language.get('conflict_input_schema')!='oran.trusted-control-chain-inputs.v1':raise ValueError('New language/risk boundary is not verified')
    preserved=0
    for path,old in before.items():
        current=read(path);records={r['id']:r for r in current['records']}
        if current['context']!=old['context']or any(records.get(r['id'])!=r for r in old['records']):raise ValueError('An existing frozen intent record changed')
        preserved+=len(old['records'])
    metadata={**previous,'pid':child.pid,'listener_pid':child.pid,'args':command,'url':f'http://127.0.0.1:{a.port}/',
        'canonical_ui':f'http://127.0.0.1:{a.port}/','artifact':str(a.artifact),'scenario_manifest':str(a.manifest),
        'scenario_gains':{s['id']:s['influences']for s in manifest['scenarios']},
        'scenario_replays':{s['id']:s['replay']for s in manifest['scenarios']},
        'derived_power_influences':next(s['influences']for s in manifest['scenarios']if s['id']==manifest['default']),
        'scenario_specific_influences':True,'traffic_contexts_pooled':False,
        'conflict_input_schema':language['conflict_input_schema'],'llm_session_verified':True,
        'conflict_risk_source_sha256':hashlib.sha256((ROOT/'smo/web/app.js').read_bytes()).hexdigest(),
        'conflict_risk_validation':str(a.artifact/'risk-audit/final/risk-coverage-summary.json'),
        'previous_service_metadata':str(a.artifact/'ui-previous-service.json'),'ui_service_log':str(logpath),
        'llm_key_storage':'service_memory_only','existing_frozen_records_preserved':preserved,
        'automatic_selection':False,'automatic_deployment':False,'updated_unix_s':time.time()}
    atomic(a.metadata,metadata);atomic(a.artifact/'ui-publication.json',dict(ready=True,pid=child.pid,url=metadata['url'],
        scenarios=ready['scenarios'],language_verified=True,frozen_records_preserved=preserved,
        preset_revision=read(manifest['preset_store'])['revision']if manifest.get('preset_store')else None,
        credential_written_to_file=False,published_unix_s=time.time()))
    print(json.dumps(dict(ready=True,pid=child.pid,url=metadata['url'],scenarios=ready['scenarios'],frozen_records_preserved=preserved)),flush=True)


if __name__=='__main__':main()
