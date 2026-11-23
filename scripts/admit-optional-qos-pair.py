#!/usr/bin/env python3
"""Admit one preregistered extra seed only after all required AGX2 work."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.context_identification import make_case
from experiments.run_rcp_perturbation import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--keep-main-alive',action='store_true');a=p.parse_args()
    spec=importlib.util.spec_from_file_location('context_coordinator',ROOT/'scripts/coordinate-context-identification.py')
    coordinator=importlib.util.module_from_spec(spec);spec.loader.exec_module(coordinator)
    worker=next(w for w in coordinator.workers(a.artifact)if w['id']=='agx2')
    cases=[{**make_case('burst_2x',20261123,arm,stage='optional_qos_seed'),'optional':True}for arm in('native','qos_native')]
    code=r'''
import sys,json,time
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import atomic,load,source_hash
from experiments.context_identification import make_case
out=root/'outputs/context';path=out/'plan.json';plan=load(path);status=load(out/'study_status.json');priority=load(out/'priority-status.json')
if source_hash(root)!=plan['source_sha256']:raise ValueError('Frozen physical source changed')
left=plan['deadline_unix_s']-time.time()
result=dict(admitted=False,remaining_s=left,deadline_unix_s=plan['deadline_unix_s'])
if c['prepare']:
 if priority.get('phase')!='running':result['reason']='supplemental_queue_not_running'
 else:
  plan['analysis_reserve_s']=300;atomic(path,plan);result.update(prepared=True,reason='keep_idle_primary_worker_available_until_targeted_pair_budget_check')
elif status.get('running')or any(not(out/'checkpoints'/(case['name']+'.json')).exists()for case in plan['cases']if not case.get('optional')):
 result['reason']='required_primary_cases_not_complete'
elif priority.get('phase')!='complete'or not priority.get('models_unchanged')or any(not(out/'checkpoints'/(case['name']+'.json')).exists()for case in load(out/'priority-plan.json')['cases']):
 result['reason']='required_priority_component_cases_not_complete'
elif left<1400:
 result['reason']='insufficient_budget_for_complete_pair_and_analysis'
else:
 for case in c['cases']:
  canonical=make_case('burst_2x',20261123,case['arm'],stage='optional_qos_seed')
  if {k:v for k,v in case.items()if k!='optional'}!=canonical:raise ValueError('Optional pair differs from the fixed protocol')
 identities={(x['context'],x['seed'],x['arm'],x['duration_s'])for x in plan['cases']}
 added=[]
 for case in c['cases']:
  identity=(case['context'],case['seed'],case['arm'],case['duration_s'])
  if identity not in identities:plan['cases'].append(case);added.append(case['name']);identities.add(identity)
 plan['analysis_reserve_s']=300;plan['sealed']=False;atomic(path,plan)
 result.update(admitted=True,added=added,reason='all_required_data_complete_and_fixed_extra_pair_fits_budget')
print(json.dumps(dict(result=result,plan=plan)))
'''
    value=json.loads(coordinator.ssh(worker,code,dict(root=coordinator.REMOTE_ROOT,cases=cases,prepare=a.keep_main_alive)))
    atomic(a.artifact/'worker-plans/agx2.json',value['plan'])
    atomic(a.artifact/('optional-qos-admission-'+str(time.time_ns())+'.json'),value['result'])
    print(json.dumps(value['result'],indent=2))


if __name__=='__main__':main()
