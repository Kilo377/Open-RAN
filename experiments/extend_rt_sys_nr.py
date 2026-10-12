"""Extend fitting tuples; retained RT profile values are never used to fit."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.fit_rt_sys_nr import fit_curve,fit_beta
from phy.rt_sys.calibration import curve_key,artifact_digest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--profiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.input.read_text());profiles=json.loads(args.profiles.read_text())['profiles'];cases=set(profiles)
    for key in data.get('excluded_nr_tuples',{}):
        t,m,re=map(int,key.split(':'));cases.add(curve_key(t,m-1,re))
    for key in sorted(cases):
        if key in data['curves'] or key in data.get('excluded_nr_tuples',{}):continue
        table,mcs,re=map(int,key.split(':'))
        try:c=fit_curve((table,mcs,re),256);beta,evidence=fit_beta(table,mcs,c,256)
        except ValueError as exc:
            if 'Repetition coding is currently not supported' not in str(exc):raise
            data.setdefault('excluded_nr_tuples',{})[key]=str(exc);continue
        data['curves'][key]=c;data['beta'][key]=beta;data['beta_fit_evidence'][key]=evidence;print(json.dumps({'fitted_new_tuple':key}),flush=True)
    data['digest']=artifact_digest(data);data['validation']={'passed':False};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2)+'\n')
