"""Capture real post-LMMSE frequency profiles without changing PHY results."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from phy.rt_sys.abstraction import SionnaAbstraction
from phy.rt_sys.calibration import curve_key
import phy.rt_sys.adapter as adapter_module
from experiments.run_rt_sys import run_case


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slots',type=int,default=100);args=p.parse_args();profiles={};records=[]
    config=json.loads((ROOT/'configs/rt_sys/default.json').read_text())
    config['rt_sys'].update(feedback_mode='sys_abstraction',allow_unvalidated_sys=True)
    config_file=ROOT/'outputs/rt_sys_targeted_repairs/profile_diagnostic_config.json'
    config_file.parent.mkdir(parents=True,exist_ok=True);config_file.write_text(json.dumps(config))
    class Capture(SionnaAbstraction):
        def evaluate(self,sinr,table,mcs,*args,**kwargs):
            result=super().evaluate(sinr,table,mcs,*args,**kwargs)
            values=sinr[sinr>0].astype(float);profile=values/values.mean();key=curve_key({'qam64':1,'qam256':2}[table],result['mcs'],len(profile))
            score=float(np.std(profile))
            if key not in profiles or score>profiles[key]['std']:
                profiles[key]={'profile':profile.tolist(),'std':score,'scene':scene,'seed':101}
            return result
    adapter_module.SionnaAbstraction=Capture
    for scene in ('simple_street_canyon','munich'):
        namespace=argparse.Namespace(scene=scene,cells=3,ues=12,seed=101,slots=args.slots,path_update_ms=None,device='cuda',config=config_file,
            output=ROOT/f'outputs/rt_sys_targeted_repairs/profile_{scene}',dry_run=False,generate_only=False,phase='integration',selection_mode='fixed_plan',
            model_state=None,warmup_ms=0,deployment_id=2,traffic_load=1,csi_error_nmse_db=None,harq_combining='none')
        run_case(namespace);records.append({'scene':scene,'seed':101,'deployment_id':2,'slots':args.slots})
    out=ROOT/'outputs/rt_sys_targeted_repairs/rt_profiles.json';out.write_text(json.dumps({'profiles':profiles,'runs':records,
        'scope':'Real RT/RZF/LMMSE profiles; results use uncalibrated feedback and are calibration probes, not algorithm performance results'},indent=2)+'\n');print(json.dumps({'captured_profiles':len(profiles)}),flush=True)
