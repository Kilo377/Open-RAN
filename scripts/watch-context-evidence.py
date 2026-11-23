#!/usr/bin/env python3
"""Bounded event watcher for the running, already authorized study."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('context_auditor',Path(__file__).with_name('audit-context-progress.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--until-done',type=int,required=True);a=p.parse_args()
    last=None;progress={}
    while True:
        status=json.loads((a.artifact/'coordinator-status.json').read_text());workers=status['workers']
        if time.time()-status['updated_unix_s']>180:
            raise RuntimeError('Durable mirror status is stale')
        for w in workers:
            if w.get('error')or w.get('failed')or w.get('mismatches'):raise RuntimeError('Worker or mirror failure: '+json.dumps(w))
            if w.get('phase')=='waiting_for_storage':raise RuntimeError('Worker storage guard engaged')
            for case in w.get('running',[]):
                key=(w['worker'],case['name']);slot=case.get('slot',0);old=progress.get(key)
                if old is None or old[0]!=slot:progress[key]=(slot,time.time())
                limit=600 if case.get('stage')=='initializing'else 240
                if time.time()-progress[key][1]>limit:
                    raise RuntimeError(f"Slot progress stalled: {w['worker']} {case['name']} slot {slot}")
                if time.time()-case['started_unix_s']>1200:
                    raise RuntimeError(f"Case exceeded the observed-cost safety window: {w['worker']} {case['name']}")
        count=sum(w['done']for w in workers);verified=sum(w['verified']for w in workers)
        if verified!=last:
            evidence=audit.once(a.artifact);last=verified
            brief=dict(done=count,verified=verified,audited=evidence['audited_observations'],
                contexts=[dict(id=c['context'],done=c['primary_complete'],expected=c['primary_expected'],ready=c['full_primary_blocks_ready'])for c in evidence['contexts']])
            (a.artifact/'event-watcher-status.json').write_text(json.dumps(brief,indent=2))
        if count>=a.until_done or all(w['completed']for w in workers):
            print(json.dumps(brief),flush=True);return
        time.sleep(30)


if __name__=='__main__':main()
