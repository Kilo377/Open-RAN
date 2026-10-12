"""Read cache footprint and protection references. Cleanup requires exact keys.

Any key referenced by saved output config/status or a PINNED marker is retained.
No automatic eviction of historical experimental evidence is performed.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phy.rt_sys.cache_storage import cache_lock


def inventory():
    references={}
    for saved in (ROOT/'outputs').rglob('run_status.json'):
        try:key=json.loads(saved.read_text()).get('cache_key')
        except (ValueError,OSError):continue
        if key:references.setdefault(key,[]).append(str(saved.relative_to(ROOT)))
    rows=[]
    for path in sorted((ROOT/'channel_cache').glob('*')):
        if not path.is_dir() or not (path/'manifest.json').is_file():continue
        metadata=json.loads((path/'manifest.json').read_text())
        files=list(path.rglob('*'))
        rows.append({'key':path.name,'bytes':sum(p.stat().st_size for p in files if p.is_file()),
            'format':metadata['manifest'].get('cache_format','dense_v1'),
            'referenced_outputs':references.get(path.name,[]),'pinned':(path/'PINNED').exists()})
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delete-key',action='append',default=[])
    p.add_argument('--output',type=Path);args=p.parse_args()
    root=ROOT/'channel_cache';root.mkdir(exist_ok=True)
    with cache_lock(root):
        rows=inventory();bykey={r['key']:r for r in rows}
        for key in args.delete_key:
            row=bykey.get(key)
            if row is None or row['pinned'] or row['referenced_outputs']:raise ValueError(f'Unknown or protected cache: {key}')
        for key in args.delete_key:shutil.rmtree(root/key)
    report={'total_bytes':sum(r['bytes'] for r in rows),'caches':rows,'deleted_keys':args.delete_key,
        'compact_quota_bytes':5*1024**3,'quota_scope':'paths_v2 files only; legacy dense caches retained'}
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
