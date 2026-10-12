"""Development-only adaptive EESM fit; fresh final validation remains required."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import least_squares
from phy.rt_sys.calibration import NRWaveform,artifact_digest,beta_value


def fit_adaptive(table,mcs,curve,initial_beta,samples=256):
    re=curve['data_re'];wave=NRWaveform(table,mcs,re);parameters=[];knots=[];evidence=[]
    for index,low in enumerate((.4,.1)):
        profile=np.ones(re);profile[:re//2]=low;profile[re//2:]=2-low
        grid=np.arange(-15,46,.25)
        eff=-initial_beta*(logsumexp(-profile[None,:]*10**(grid[:,None]/10)/initial_beta,axis=1)-np.log(re))
        pred=np.interp(10*np.log10(np.maximum(eff,1e-30)),curve['snr_db'],curve['tbler'])
        center=float(grid[np.argmin(abs(pred-.5))]);snrs=np.array([center+x for x in (-2,-1,-.5,.5,1,2)])
        seed=2800001+table*100000+mcs*1000+re+index*300000
        errors=wave.simulate_fast(snrs,profile,samples,seed)
        cv=float(profile.std()/profile.mean());reference=center+10*np.log10(profile.mean())
        def residual(p):
            spec={'cv_knots':[0,cv],'parameters':[dict(log_beta=p[0],slope=p[1],reference_db=reference)]*2}
            predictions=[]
            for db in snrs:
                values=profile*10**(db/10);beta=beta_value(spec,values)
                effective=-beta*(logsumexp(-values/beta)-np.log(re))
                predictions.append(np.interp(10*np.log10(max(effective,1e-30)),curve['snr_db'],curve['tbler']))
            return np.asarray(predictions)-errors/samples
        solutions=[least_squares(residual,[np.log(initial_beta),slope],bounds=([-12,0],[16,4])) for slope in (0,.5,1,2,3.5)]
        opt=min(solutions,key=lambda s:float(np.sum(s.fun**2)))
        parameters.append(dict(log_beta=float(opt.x[0]),slope=float(opt.x[1]),reference_db=reference));knots.append(cv)
        evidence.append(dict(cv=cv,snr_db=snrs.tolist(),errors=errors.tolist(),samples=samples,seed=seed,mse=float(np.mean(opt.fun**2))))
    return {'kind':'snr_variance_power_law','cv_knots':[0.]+knots,'parameters':[parameters[0]]+parameters},evidence


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mse-threshold',type=float,default=0);args=p.parse_args()
    data=json.loads(args.input.read_text())
    recipe={'input_digest':artifact_digest(data),'revision':1,'samples':256,'seed_namespace':2800001,'profiles':[.4,.1]}
    recipe_id=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()[:16]
    root=args.output.parent/(args.output.stem+'_adaptive')/recipe_id
    root.mkdir(parents=True,exist_ok=True);(root/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    start=time.monotonic();evidence={}
    for index,(key,c) in enumerate(data['curves'].items()):
        if data.get('beta_fit_evidence',{}).get(key,{}).get('fit_mse',1)<args.mse_threshold:continue
        table,mcs,re=map(int,key.split(':'));file=root/(key.replace(':','_')+'.json')
        if file.exists():record=json.loads(file.read_text())
        else:
            spec,points=fit_adaptive(table,mcs,c,float(data['beta'][key]));record={'spec':spec,'fit_points':points};file.write_text(json.dumps(record)+'\n')
        data['beta'][key]=record['spec'];evidence[key]=record['fit_points'];print(json.dumps(dict(stage='adaptive_beta_fit',done=index+1,total=len(data['curves']),case=key,elapsed_s=time.monotonic()-start)),flush=True)
    data['adaptive_beta_fit_evidence']=evidence;data['waveform']['effective_sinr']='EESM with SNR/variance-conditioned beta; fixed-profile inputs interpolated, arbitrary profiles require validation'
    data['digest']=artifact_digest(data);data['validation']={'passed':False};args.output.write_text(json.dumps(data,indent=2)+'\n')
