"""Frozen X-01 DRX unlock: matched Base, native-DRX and unlocked-DRX runs.

All completed raw recordings and the input forest are retained. The existing
acceptance thresholds are reported as historical guards, not a stopping rule:
tradeoffs are outcomes, and no seed is discarded to obtain a favorable result.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_ue_energy_study import compare_pair


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def action_audit(folder):
    path=Path(folder)/'energy_rf_events.jsonl';decisions=[]
    if path.exists():
        decisions=[r for r in map(json.loads,path.read_text().splitlines()) if r.get('kind')=='decision']
    actions=Counter(str(r['action']) for r in decisions)
    # Actual applied control receipts, not prediction or intent, establish use.
    import gzip
    trace=Path(folder)/'rt_sys_trace.jsonl.gz'
    applied=[];executed=Counter();non_native=0
    if trace.exists():
        with gzip.open(trace,'rt')as stream:
            for line in stream:
                row=json.loads(line)
                for uid,s in row.get('ue_services',{}).items():
                    policy=s.get('ue_service_policy') or s.get('policy')
                    if policy is not None:
                        applied.append((uid,policy))
                        if uid in ('ue_0','ue_4'):
                            drx=policy['drx'];profile=tuple(drx[k]for k in('cycle_ms','on_duration_ms','inactivity_ms'))
                            executed[str(profile)]+=1;non_native+=profile!=(40.,4.,8.)
    summary=load(Path(folder)/'final_summary.json')
    return dict(decision_count=len(decisions),chosen_actions=dict(actions),
                chosen_non_native_drx=sum(count for index,count in actions.items() if int(index)>=3),
                final_service_states=summary.get('ue_services',{}),
                executed_policy_samples=len(applied),executed_drx_profiles=dict(executed),executed_non_native_drx_samples=non_native)


def run(output,model,seed,duration,cache,retention=None):
    output=Path(output).resolve();model=Path(model).resolve();cache=Path(cache).resolve()
    if not output.is_relative_to(ROOT/'outputs') or output.exists():raise ValueError('Choose a fresh project output directory')
    if not model.is_relative_to(ROOT/'outputs') or not model.is_file():raise ValueError('Stage the pinned frozen model under project outputs')
    output.mkdir(parents=True);(output/'cases').mkdir();(output/'logs').mkdir()
    retention=Path(retention).resolve() if retention else None
    if retention:retention.mkdir(parents=True,exist_ok=False)
    def retain():
        if retention:shutil.copytree(output,retention,dirs_exist_ok=True)
    checkpoint=output/'input-model.json';shutil.copy2(model,checkpoint);digest=sha(checkpoint)
    commands={}
    for name in ('base','native_drx','unlocked_drx'):
        cmd=['bash','scripts/orin/run-six-ue-cuda-nr.sh','--model','fast','--duration-s',str(duration),'--seed',str(seed),
             '--traffic-load','1','--ue-services','drx-mec','--xapp-pool','chongyu','--record-compression','gzip',
             '--mec-stage-metrics','--channel-cache-root',str(cache),'--output',str(output/'cases'/name)]
        if name!='base':cmd+=['--energy-xapp-mode','frozen','--energy-xapp-model',str(checkpoint),
                              '--energy-xapp-deploy-drx' if name=='unlocked_drx' else '--no-energy-xapp-deploy-drx']
        commands[name]=cmd
    plan=dict(schema='oran.drx-unlock-study.v1',seed=seed,duration_s=duration,model_sha256=digest,
              forest_training='None; exact existing frozen forest, no artifact writes',commands=commands,
              traffic_and_radio='Native six-UE, 3-cell setting; matched seed, movement, traffic and radio',
              scoring='Full-run energy/goodput; task cohorts starting at 100 ms and mature by the same horizon')
    atomic(output/'plan.json',plan)
    state=dict(status='running',completed=[],active=None,model_sha256=digest,started_at=time.time())
    atomic(output/'status.json',state);retain()
    for name,cmd in commands.items():
        assert sha(checkpoint)==digest
        state.update(active=name);atomic(output/'status.json',state)
        with (output/'logs'/(name+'.log')).open('x')as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;atomic(output/'status.json',state)
            while child.poll() is None:
                state['updated_at']=time.time()
                progress=output/'cases'/name/'run_status.json'
                if progress.exists():
                    try:state['progress']=load(progress)
                    except (OSError,ValueError):pass
                atomic(output/'status.json',state)
                if retention:atomic(retention/'status.json',state)
                time.sleep(5)
        if child.returncode:
            state.update(status='failed',exit_code=child.returncode);atomic(output/'status.json',state);retain()
            raise RuntimeError('Case failed: '+name)
        assert sha(checkpoint)==digest
        manifest=load(output/'cases'/name/'config_snapshot.json')
        if name!='base':
            params=manifest['configs']['ric']['xapps'][0]['parameters']
            assert params['mode']=='frozen' and params['deploy_drx']==(name=='unlocked_drx')
        state['completed'].append(name);atomic(output/'status.json',state);retain()
    thresholds=dict(min_target_energy_saving_fraction=.05,max_task_rate_drop=.05,max_delivered_fraction_drop=.05)
    comparisons={name:compare_pair(output/'cases/base',output/'cases'/name,thresholds)for name in('native_drx','unlocked_drx')}
    comparisons['drx_increment']=compare_pair(output/'cases/native_drx',output/'cases/unlocked_drx',thresholds)
    atomic(output/'comparison.json',dict(schema='oran.drx-unlock-results.v1',comparisons=comparisons,
        note='Legacy guard flags are diagnostic only. Matched frozen-policy effect, not pure DRX attribution if offload also changes.',
        action_audits={name:action_audit(output/'cases'/name)for name in('native_drx','unlocked_drx')}))
    state.update(status='completed',active=None,finished_at=time.time());atomic(output/'status.json',state);retain()
    return output/'comparison.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--retention-root',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--duration-s',type=float,default=3.);p.add_argument('--channel-cache-root',type=Path,required=True)
    args=p.parse_args()
    if args.duration_s<=.3 or args.seed<0:raise ValueError('Need a positive seed and sufficient mature cohorts')
    print(run(args.output,args.model,args.seed,args.duration_s,args.channel_cache_root,args.retention_root),flush=True)


if __name__=='__main__':main()
