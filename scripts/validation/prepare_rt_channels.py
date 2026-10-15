"""Measure isolated cold RT generation, then prime identical canonical channels."""
import argparse,json,os,sys,time,shutil
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main(a):
    import numpy as np
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    a.output.mkdir(parents=True,exist_ok=False)
    manifest,configs,_=build_case(a.scene,3,a.ues,512,512,device='cuda',config_path=ROOT/'configs/rt_sys/paper_interfaces.json',initial_queue_bytes=0,ue_speed_mps=0)
    start=time.perf_counter();cold=RtChannelProvider(manifest,a.output/'cold_cache',device='cuda');init=time.perf_counter()-start
    start=time.perf_counter();x=cold.snapshot(0);cold_s=time.perf_counter()-start
    canonical=ROOT/'channel_cache'/cold.key;preexisting=(canonical/'paths_000000000.npz').exists()
    canonical.mkdir(parents=True,exist_ok=True)
    for p in cold.root.iterdir():
        target=canonical/p.name
        if not target.exists():shutil.copy2(p,target)
    warm=RtChannelProvider(manifest,ROOT/'channel_cache',device='cuda')
    start=time.perf_counter();y=warm.snapshot(0);warm_s=time.perf_counter()-start
    assert x.cache_key==y.cache_key and warm.hits==1 and warm.misses==0
    np.testing.assert_allclose(x.cfr,y.cfr,rtol=1e-6,atol=1e-12)
    report=dict(scene=a.scene,ues=a.ues,slots=512,cache_key=cold.key,initialization_s=init,
        cold_generation_and_cfr_s=cold_s,warm_read_and_cfr_s=warm_s,
        canonical_preexisted=preexisting,primed=True,cold_misses=cold.misses,warm_hits=warm.hits,
        scope='RT channel generation only; no NR encoding/decoding or link probes.')
    (a.output/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--ues',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
