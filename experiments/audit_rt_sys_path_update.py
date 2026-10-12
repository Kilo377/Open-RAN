"""Compare short GPU path-update probes and deterministic cached replay."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(root):
    names = ('path_dt1', 'path_dt10', 'path_replay')
    traces = {}; configs = {}; statuses = {}
    for name in names:
        path = root / name
        statuses[name] = json.loads((path / 'run_status.json').read_text())
        assert statuses[name]['completed'] and statuses[name]['spark_verified']
        configs[name] = json.loads((path / 'config_snapshot.json').read_text())
        traces[name] = [json.loads(line) for line in (path / 'rt_sys_trace.jsonl').read_text().splitlines()]
    assert traces['path_dt10'] == traces['path_replay']
    manifests = {name: configs[name]['manifest'] for name in names}
    assert manifests['path_dt1']['path_update_ms'] == 1
    assert manifests['path_dt10']['path_update_ms'] == 10
    a, b = manifests['path_dt1'].copy(), manifests['path_dt10'].copy()
    a.pop('path_update_ms'); b.pop('path_update_ms')
    assert a == b
    slots = a['total_slots']
    error_energy = reference_energy = 0.; max_gain_delta_db = 0.
    import sys
    sys.path.insert(0,str(ROOT))
    from phy.rt_sys.channel import reconstruct_paths,geometry_blocks
    def cached_slot(name,slot):
        manifest=manifests[name];folder=ROOT/'channel_cache'/statuses[name]['cache_key']
        if manifest.get('cache_format')=='paths_v2':
            blocks,_=geometry_blocks(manifest)
            block=blocks[np.searchsorted(blocks,slot,side='right')-1]
            with np.load(folder/f'paths_{block:09d}.npz') as arrays:
                return reconstruct_paths(arrays,manifest,(slot-block)*manifest['slot_duration_ms']/1000)
        chunk=slot//manifest['chunk_slots'];index=slot%manifest['chunk_slots']
        with np.load(folder/f'chunk_{chunk:06d}.npz') as arrays:return arrays['cfr'][index]
    for slot in range(1,slots+1):
        x,y=[cached_slot(name,slot) for name in names[:2]]
        error_energy += float(np.sum(np.abs(x-y)**2, dtype=np.float64))
        reference_energy += float(np.sum(np.abs(x)**2, dtype=np.float64))
        gx=np.mean(abs(x)**2,axis=(2,3,4,5));gy=np.mean(abs(y)**2,axis=(2,3,4,5))
        valid=(gx>1e-25)&(gy>1e-25)
        if valid.any():
            max_gain_delta_db=max(max_gain_delta_db,float(np.max(abs(10*np.log10(gx[valid]/gy[valid])))))
    return {'completed': True, 'device': 'cuda', 'scene': a['scene'],
        'cells': len(a['cells']), 'ues': len(a['ues']), 'slots': slots,
        'replay_trace_exact_match': True,
        'delivered_bytes': {name: sum(u['delivered_bytes'] for t in trace for u in t['ues'])
                            for name, trace in traces.items()},
        'relative_cfr_l2_error': (error_energy / max(reference_energy, 1e-30))**.5,
        'max_link_mean_gain_difference_db': max_gain_delta_db,
        'scope': 'Short integration check; SYS-reported SINR may saturate. Does not establish long-term approximation accuracy.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/rt_sys_acceptance_grid')
    args = parser.parse_args()
    report = audit(args.root)
    (args.root/'path_update_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
