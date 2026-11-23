#!/usr/bin/env python3
"""Append preregistered seed blocks without editing immutable worker source."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.context_identification import make_case
from experiments.run_rcp_perturbation import atomic
spec=importlib.util.spec_from_file_location('coordinator',ROOT/'scripts/coordinate-context-identification.py');coordinator=importlib.util.module_from_spec(spec);spec.loader.exec_module(coordinator)


def append(artifact,blocks,*,stage='grid',duration_s=1.8,arms=None,sealed=False):
    artifact=Path(artifact);design=json.loads((artifact/'design.json').read_text())
    ws=coordinator.workers(artifact)
    code=r'''
import sys,json
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['root']);sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import atomic,load
p=root/'outputs/context/plan.json';old=load(p)
if old['source_sha256']!=c['source_sha256']:raise ValueError('Source pin changed')
names={e['name']for e in old['cases']}
identities={(e['context'],e['seed'],e['arm'],e['duration_s'])for e in old['cases']}
for case in c['new_cases']:
 identity=(case['context'],case['seed'],case['arm'],case['duration_s'])
 if case['name']not in names and identity not in identities:
  old['cases'].append(case);names.add(case['name']);identities.add(identity)
old['sealed']=c['sealed'];atomic(p,old);print(json.dumps(old))
'''
    summary=[]
    for w in ws:
        base=json.loads((artifact/'worker-plans'/(w['id']+'.json')).read_text());new=[]
        for b in blocks:
            if b['worker_id']!=w['id']:continue
            for arm in arms or list(design['specs']):new.append(make_case(b['context'],b['seed'],arm,stage=stage,duration_s=duration_s))
        value=json.loads(coordinator.ssh(w,code,dict(root=coordinator.REMOTE_ROOT,source_sha256=base['source_sha256'],new_cases=new,sealed=sealed)))
        atomic(artifact/'worker-plans'/(w['id']+'.json'),value)
        summary.append(dict(worker=w['id'],requested=len(new),added=len(value['cases'])-len(base['cases']),total=len(value['cases'])))
    atomic(artifact/f'extension-{stage}-{time.time_ns()}.json',dict(blocks=blocks,arms=arms or list(design['specs']),duration_s=duration_s,sealed=sealed,workers=summary))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--blocks',type=Path,required=True);p.add_argument('--stage',default='grid');p.add_argument('--duration-s',type=float,default=1.8);p.add_argument('--arms',nargs='+');p.add_argument('--seal',action='store_true');a=p.parse_args()
    print(json.dumps(append(a.artifact,json.loads(a.blocks.read_text()),stage=a.stage,duration_s=a.duration_s,arms=a.arms,sealed=a.seal),indent=2))


if __name__=='__main__':main()
