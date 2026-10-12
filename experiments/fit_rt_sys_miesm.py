"""Fit MIESM on independent artificial profiles; RT values stay held out."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import minimize_scalar,least_squares,minimize
from scipy.special import ndtr,ndtri
from phy.rt_sys.calibration import NRWaveform,monotone,artifact_digest,effective_sinr_value


def smooth_flat_curve(curve):
    """Fit binomial likelihood, rather than linear interpolation across a cliff."""
    x=np.asarray(curve['snr_db']);counts=np.asarray(curve['errors']);n=curve['samples_per_point'];num_cb=curve['num_cb']
    center=float(np.interp(.5,np.asarray(curve['tbler'])[::-1],x[::-1]))
    def probability(p,db):
        cb=ndtr((p[0]-db)/np.exp(p[1]))
        return -np.expm1(num_cb*np.log1p(-np.minimum(cb,1-1e-15)))
    def objective(p):
        pred=np.clip(probability(p,x),1e-12,1-1e-12)
        return float(-np.sum(counts*np.log(pred)+(n-counts)*np.log1p(-pred)))
    solutions=[minimize(objective,[center,np.log(s)],bounds=[(center-3,center+3),(-4,2)]) for s in (.15,.3,.6,1.)]
    opt=min(solutions,key=lambda s:s.fun);grid=np.arange(-20,40.001,.05)
    return dict(curve,snr_db=grid.tolist(),tbler=probability(opt.x,grid).tolist(),
        curve_model='binomial_CB_probit_TB',curve_parameters=opt.x.tolist(),
        fit_measurements={'snr_db':curve['snr_db'],'errors':curve['errors'],'samples':n})


def training_profiles(re):
    for low in (.4,.1):
        x=np.ones(re);x[:re//2]=low;x[re//2:]=2-low;yield x
    z=ndtri((np.arange(re)+.5)/re)
    for target in (1.5,3.,6.,9.):
        lo,hi=0.,10.
        for _ in range(40):
            sigma=(lo+hi)/2;x=np.exp(sigma*(z-z.max()));x/=x.mean()
            if x.std()<target:lo=sigma
            else:hi=sigma
        yield x


def fit(table,mcs,curve,samples=256):
    re=curve['data_re'];wave=NRWaveform(table,mcs,re)
    center=float(np.interp(.5,np.asarray(curve['tbler'])[::-1],np.asarray(curve['snr_db'])[::-1]))
    seed=3800001+table*100000+mcs*1000+re
    c=smooth_flat_curve(curve)
    parameters=[];scales=[];knots=[];evidence=[]
    for index,profile in enumerate(training_profiles(re)):
        spec={'kind':'qam_bicm_mi','qam_bits':wave.q,'cv_knots':[0.],'shift_db':[0.]}
        grid=np.arange(-15,81,.25)
        pred=[np.interp(10*np.log10(effective_sinr_value(spec,profile*10**(db/10))),c['snr_db'],c['tbler']) for db in grid]
        median=float(grid[np.argmin(abs(np.asarray(pred)-.5))]);test_snrs=np.asarray([median+x for x in (-2,-1,-.5,.5,1,2)])
        fit_seed=seed+500000+index*300000
        measured=wave.simulate_fast(test_snrs,profile,samples,fit_seed)/samples
        effective=np.asarray([10*np.log10(effective_sinr_value(spec,profile*10**(db/10))) for db in test_snrs])
        def residual(p):return np.interp(center+p[1]*(effective-center)+p[0],c['snr_db'],c['tbler'])-measured
        solutions=[least_squares(residual,[shift,1.],bounds=([-4,.25],[4,4])) for shift in (-1.,0.,1.)]
        opt=min(solutions,key=lambda s:np.sum(s.fun**2));cv=float(profile.std()/profile.mean())
        knots.append(cv);parameters.append(float(opt.x[0]));scales.append(float(opt.x[1]));evidence.append({'cv':cv,'profile_family':'contiguous_two_level' if index<2 else 'sorted_lognormal','fit_seed':fit_seed,'snr_db':test_snrs.tolist(),'measured':measured.tolist(),'mse':float(np.mean(opt.fun**2))})
    spec={'kind':'qam_bicm_mi','qam_bits':wave.q,'cv_knots':[0.]+knots,'shift_db':[0.]+parameters,'slope':[1.]+scales,'reference_db':center}
    return c,spec,evidence


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.input.read_text());recipe={'input_digest':artifact_digest(data),'revision':3,'samples':256,'seed':3800001}
    recipe_id=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()[:16]
    root=args.output.parent/(args.output.stem+'_shards')/recipe_id;root.mkdir(parents=True,exist_ok=True);start=time.monotonic();evidence={}
    for index,(key,c) in enumerate(data['curves'].items()):
        file=root/(key.replace(':','_')+'.json')
        if file.exists():record=json.loads(file.read_text())
        else:
            t,m,re=map(int,key.split(':'));curve,spec,points=fit(t,m,c);record={'curve':curve,'spec':spec,'fit_points':points};file.write_text(json.dumps(record)+'\n')
        data['curves'][key]=record['curve'];data['beta'][key]=record['spec'];evidence[key]=record['fit_points']
        print(json.dumps({'stage':'miesm_fit','done':index+1,'total':len(data['curves']),'case':key,'elapsed_s':time.monotonic()-start}),flush=True)
    data['miesm_fit_evidence']=evidence;data['waveform']['effective_sinr']='Gray-QAM BICM-MIESM with per-tuple CV-conditioned SNR correction; not EESM'
    data['digest']=artifact_digest(data);data['validation']={'passed':False};args.output.write_text(json.dumps(data,indent=2)+'\n')
