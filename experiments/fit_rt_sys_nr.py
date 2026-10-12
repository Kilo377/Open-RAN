"""Fit exact-resource NR TB curves and EESM; validation uses separate seeds.

Resume shards are immutable inputs keyed by the complete waveform settings.
Only independently validated artifacts can be selected by the production PHY.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,json,sys,math,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from phy.rt_sys.calibration import NRWaveform,monotone,curve_key,artifact_digest


def transition(wave,profile,seed):
    center=10*np.log10(2**(wave.q*wave.rate)-1)
    lo,hi=center-5,center+12
    for _ in range(7):
        mid=(lo+hi)/2;error=wave.simulate([mid],profile,64,seed)[0]/64
        if error>.5:lo=mid
        else:hi=mid
    return round((lo+hi)/2*2)/2


def fit_curve(case,samples):
    table,mcs,re=case;w=NRWaveform(*case);seed=800001+table*100000+mcs*1000+re
    center=transition(w,np.ones(re),seed)
    snrs=sorted(set([-20.,40.]+[center+x/2 for x in range(-6,7)]))
    errors=w.simulate(snrs,np.ones(re),samples,seed+31)
    return dict(table=table,mcs=mcs,data_re=re,information_bits=w.bits,cb_size=w.cb,num_cb=w.num_cb,
        snr_db=snrs,tbler=monotone(errors/samples),samples_per_point=samples,errors=errors.tolist(),fit_seed=seed+31)


def profiles(re):
    out=[]
    for low in (.1,.4):
        x=np.ones(re);x[:re//2]=low;x[re//2:]=2-low;out.append(x)
    return out


def fit_beta(table,mcs,curve,samples):
    from scipy.optimize import minimize_scalar
    re=curve['data_re'];w=NRWaveform(table,mcs,re);observations=[]
    for i,profile in enumerate(profiles(re)):
        seed=1800001+table*100000+mcs*1000+i*100
        center=transition(w,profile,seed)
        snrs=[center+x for x in (-1.5,-.5,.5,1.5)]
        errors=w.simulate(snrs,profile,samples,seed+41)
        observations.extend((db,profile,int(error)/samples) for db,error in zip(snrs,errors))
    def objective(logbeta):
        from scipy.special import logsumexp
        beta=np.exp(logbeta);loss=0.
        for db,profile,emp in observations:
            eff=-beta*(logsumexp(-profile*10**(db/10)/beta)-np.log(re))
            pred=np.interp(10*np.log10(max(eff,1e-30)),curve['snr_db'],curve['tbler']);loss+=(pred-emp)**2
        return loss/len(observations)
    grid=np.linspace(-5,10,121);losses=[objective(x) for x in grid];best=int(np.argmin(losses))
    opt=minimize_scalar(objective,bounds=(grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]),method='bounded')
    return float(np.exp(opt.x)),dict(fit_mse=float(opt.fun),profiles=['contiguous_0.1_1.9','contiguous_0.4_1.6'],samples_per_point=samples)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-re',type=int,nargs='+',default=[324]);p.add_argument('--observed-root',type=Path)
    p.add_argument('--samples',type=int,default=256);p.add_argument('--output',type=Path,default=ROOT/'outputs/rt_sys_calibration_repair/nr_model.json');args=p.parse_args()
    if args.samples<128 or any(re<24 for re in args.data_re):raise ValueError('Fit requires >=128 samples and >=24 data RE')
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):raise ValueError('Fitted state must stay within copy outputs/')
    output.parent.mkdir(parents=True,exist_ok=True);shards=output.parent/(output.stem+'_curves');shards.mkdir(exist_ok=True)
    from phy.rt_sys.abstraction import SionnaAbstraction
    model=SionnaAbstraction('cuda');valid={table:sorted(model.phy.bler_table['category'][1]['index'][table]['MCS']) for table in (1,2)}
    cases={(table,mcs,re) for table in valid for mcs in valid[table] for re in args.data_re}
    if args.observed_root:
        for path in args.observed_root.glob('*/rt_sys_trace.jsonl'):
            for line in path.open():
                for u in json.loads(line)['ues']:
                    if u['scheduled'] and u['data_re']>0:
                        for table in valid:
                            if u['mcs'] in valid[table]:cases.add((table,u['mcs'],u['data_re']))
        for re in {case[2] for case in cases}:cases.update(((1,min(valid[1]),re),(2,min(valid[2]),re)))
    revision=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'phy/rt_sys/calibration.py').read_bytes()).hexdigest()[:12]
    print(json.dumps({'planned_curves':len(cases),'samples':args.samples,'revision':revision}),flush=True)
    curves={};excluded={};start=time.monotonic()
    for i,case in enumerate(sorted(cases)):
        key=curve_key(*case);file=shards/(key.replace(':','_')+f'_n{args.samples}_{revision}.json')
        if file.exists():c=json.loads(file.read_text())
        else:
            try:c=fit_curve(case,args.samples)
            except ValueError as exc:
                if 'Repetition coding is currently not supported' not in str(exc):raise
                excluded[key]=str(exc);print(json.dumps(dict(stage='unsupported_nr_tuple',case=key,reason=str(exc))),flush=True);continue
            file.write_text(json.dumps(c)+'\n')
        curves[key]=c;print(json.dumps(dict(stage='flat_fit',done=i+1,total=len(cases),case=key,elapsed_s=time.monotonic()-start)),flush=True)
    betas={};beta_evidence={}
    for key,c in curves.items():
        table,mcs,re=map(int,key.split(':'));file=shards/f'beta_{table}_{mcs}_{re}_n{args.samples}_{revision}.json'
        if file.exists():b=json.loads(file.read_text())
        else:
            beta,evidence=fit_beta(table,mcs,c,args.samples);b={'beta':beta,'evidence':evidence};file.write_text(json.dumps(b)+'\n')
        betas[key]=b['beta'];beta_evidence[key]=b['evidence'];print(json.dumps(dict(stage='beta_fit',case=key,beta=b['beta'],elapsed_s=time.monotonic()-start)),flush=True)
    artifact={'schema':'nr_tb_exact_re_v1','curves':curves,'beta':betas,'beta_fit_evidence':beta_evidence,'excluded_nr_tuples':excluded,
        'waveform':{'sionna':'2.2.0','channel':'PDSCH','decoder_iterations':20,'check_node':'boxplus-phi','rv':0,'fit_seed_namespace':'800001/1800001','coded_resource_index':'exact_data_re'},
        'validation':{'passed':False},'fit_elapsed_s':time.monotonic()-start}
    artifact['digest']=artifact_digest(artifact);output.write_text(json.dumps(artifact,indent=2)+'\n');print(json.dumps({'fitted':True,'curves':len(curves),'output':str(output),'elapsed_s':artifact['fit_elapsed_s']}),flush=True)
