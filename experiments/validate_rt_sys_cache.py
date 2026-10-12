"""Compare compact path reconstruction against actual Sionna RT CFR."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))


def validate(scene,device):
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider,SionnaRayTracer
    m,_,_=build_case(scene,total_slots=40,device=device)
    provider=RtChannelProvider(m,ROOT/'channel_cache',device=device)
    fresh=SionnaRayTracer(m,device); fresh.validate_geometry()
    slots=(1,2,19,20,21,39,40); errors=[]; dense_bytes=0
    for slot in slots:
        compact=provider.snapshot(slot).cfr
        reference=fresh.generate(slot,slot+1,provider.update_slots)['cfr'][0]
        error=float(np.linalg.norm(compact-reference)/max(np.linalg.norm(reference),1e-30))
        assert error<1e-4,(scene,slot,error)
        errors.append(error); dense_bytes+=reference.nbytes
    replay=RtChannelProvider(m,ROOT/'channel_cache',device=device)
    for slot in slots:np.testing.assert_array_equal(provider.snapshot(slot).cfr,replay.snapshot(slot).cfr)
    assert replay.misses==0
    saved_bytes=sum(p.stat().st_size for p in provider.root.glob('paths*.npz'))
    return {'scene':scene,'device':device,'completed':True,'cache_key':provider.key,
        'max_relative_l2_error':max(errors),'cache_replay_exact':True,
        'tested_slots':list(slots),'compact_cache_bytes':saved_bytes,
        'dense_tested_cfr_bytes':dense_bytes,'cache_misses':provider.misses}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    args=p.parse_args(); reports=[validate(s,args.device) for s in ('simple_street_canyon','munich')]
    out=ROOT/'outputs/rt_sys_repairs/cache_validation.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
