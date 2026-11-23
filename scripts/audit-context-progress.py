#!/usr/bin/env python3
"""Audit every durable observation; fit a scene only after full seed blocks."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smo.context_identify import extract,analyze
from smo.identify import load,write


def once(artifact):
    artifact=Path(artifact);rows=extract(artifact);design=load(artifact/'design.json')
    blocks=load(artifact/'proposed-primary-blocks.json');summary=[]
    for context in ['burst_2x','native_1x','steady_2x']:
        own=[r for r in rows if r['context']==context]
        expected={(b['seed'],arm)for b in blocks if b['context']==context for arm in design['specs']}
        have={(r['seed'],r['arm'])for r in own};complete=expected<=have
        supplemental_grid=next((g for g in design['grids']if g.get('stimulus_protocol')=='fixed_scheduler_probe'),None)
        supplemental_complete=bool(supplemental_grid)and all((seed,arm)in have for seed in sorted({s for s,a in expected})for arm in supplemental_grid['arms'])
        version='v2_component'if supplemental_complete else'v1_core'
        output=artifact/'analysis'/version/context
        result=dict(context=context,audited=len(own),primary_complete=len(expected&have),primary_expected=len(expected),
            full_primary_blocks_ready=complete,identified_component_ready=supplemental_complete)
        if complete:
            if not(output/'identification-report.json').exists():analyze(artifact,context,output)
            result.update(analysis=str(output),report=load(output/'identification-report.json'))
        summary.append(result)
    value=dict(schema='oran.context-audit-progress.v1',audited_observations=len(rows),
        audited_scored_slots=sum(r['proof']['verified_held_parameter_slots']for r in rows),contexts=summary,updated_unix_s=time.time())
    write(artifact/'evidence-progress.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    try:print(json.dumps(once(a.artifact),indent=2))
    except Exception as error:
        write(a.artifact/'evidence-audit-error.json',dict(error=repr(error),updated_unix_s=time.time()))
        raise


if __name__=='__main__':main()
