"""Full closed-loop replay on both maps; volatile timing/cache counters excluded."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.run_rt_sys import run_case


if __name__=='__main__':
    root=ROOT/'outputs/rt_sys_targeted_repairs/reference_replay';root.mkdir(parents=True,exist_ok=True);reports=[]
    for scene in ('simple_street_canyon','munich'):
        for label in ('fresh','replay'):
            args=argparse.Namespace(scene=scene,cells=2,ues=4,seed=307,slots=40,path_update_ms=None,device='cuda',config=None,
                output=root/(scene+'_'+label),dry_run=False,generate_only=False,phase='integration',selection_mode='reflective',
                model_state=None,warmup_ms=0,deployment_id=None,traffic_load=1,csi_error_nmse_db=None,harq_combining='none')
            run_case(args)
        a=root/(scene+'_fresh');b=root/(scene+'_replay')
        assert (a/'ue_metrics.csv').read_bytes()==(b/'ue_metrics.csv').read_bytes()
        assert (a/'cell_metrics.csv').read_bytes()==(b/'cell_metrics.csv').read_bytes()
        status=json.loads((b/'run_status.json').read_text());assert status['spark_verified']
        reports.append({'scene':scene,'ue_csv_exact':True,'cell_csv_exact':True,'spark_verified':True,
            'source_sha256':status['source_sha256'],'cache_key':status['cache_key']})
    (root.parent/'reference_replay_validation.json').write_text(json.dumps({'passed':True,'cases':reports},indent=2)+'\n')
    print(json.dumps(reports,indent=2))
