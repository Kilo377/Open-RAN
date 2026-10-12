"""Independent held-out-seed and held-out-profile NR-TB validation."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from phy.rt_sys.calibration import NRWaveform,load_artifact,curve_key,effective_sinr_value
from phy.rt_sys.abstraction import SionnaAbstraction
from experiments.calibrate_rt_sys_phy import wilson


def validation(artifact_file,samples=512,rt_profiles=None):
    model=SionnaAbstraction('cuda',calibration_file=artifact_file,_allow_unvalidated=True);a=model.calibration;records=[];start=time.monotonic()
    for index,(key,c) in enumerate(a['curves'].items()):
        table,mcs,re=map(int,key.split(':'));wave=NRWaveform(table,mcs,re);profiles=[('flat',np.ones(re))]
        for label,low in [('independent_seed_training_profile',.1),('held_out_profile',.25)]:
            profile=np.ones(re);profile[:re//2]=low;profile[re//2:]=2-low;profiles.append((label,profile))
        if rt_profiles and key in rt_profiles:profiles.append(('held_out_rt_lmmse_profile',np.asarray(rt_profiles[key]['profile'])))
        for pi,(label,profile) in enumerate(profiles):
            grid=np.arange(-15,101,.25)
            values=profile[None,:]*10**(grid[:,None]/10)
            eff=np.asarray([effective_sinr_value(a['beta'][key],v) for v in values])
            pred=np.interp(10*np.log10(np.maximum(eff,1e-30)),c['snr_db'],c['tbler'])
            center=float(grid[np.argmin(abs(pred-.5))])
            offsets=(-2.75,-1.25,-.25,.25,1.25,2.75) if re==324 and pi==0 else (-1.75,-.75,.25,1.25)
            snrs=[center+x for x in offsets];seed=6800001+table*100000+mcs*1000+re+pi*300000
            errors=wave.simulate_fast(snrs,profile,samples,seed)
            for db,error in zip(snrs,errors):
                r=model.evaluate((profile*10**(db/10))[None,:,None],{1:'qam64',2:'qam256'}[table],mcs,1,0,'validation',True)
                ci=wilson(int(error),samples);ok=ci[0]-.1<=r['tbler']<=ci[1]+.1
                records.append(dict(case=key,profile=label,snr_db=db,errors=int(error),samples=samples,empirical_bler=int(error)/samples,
                    sys_bler=r['tbler'],wilson_95=ci,passed=bool(ok),seed=seed))
                if not ok:print(json.dumps({'failed_point':records[-1]}),flush=True)
        print(json.dumps(dict(stage='independent_validation',done=index+1,total=len(a['curves']),failed=sum(not r['passed'] for r in records),elapsed_s=time.monotonic()-start)),flush=True)
    return dict(passed=all(r['passed'] for r in records),fitted_digest=a['digest'],device='cuda:0',actual_nr_crc_ldpc_qam=True,
        samples_per_point=samples,points=records,fit_seeds_disjoint=True,held_out_frequency_profile=True,
        criterion='SYS TBLER in empirical Wilson 95% interval expanded by absolute 0.10',
        validated_tuples=list(a['curves']),validated_profiles=['flat','contiguous_0.1_1.9','contiguous_0.25_1.75']+(['held_out_rt_lmmse_profiles'] if rt_profiles else []),
        limitations=['Not a URLLC tail guarantee','Other coded-resource tuples fail explicitly','Other fading profiles require validation','No waveform-level HARQ/CSI guarantee'],elapsed_s=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--samples',type=int,default=512);p.add_argument('--output',type=Path)
    p.add_argument('--profiles',type=Path,default=ROOT/'outputs/rt_sys_targeted_repairs/rt_profiles.json');p.add_argument('--allow-fixed-beta',action='store_true');args=p.parse_args()
    if args.samples<256:raise ValueError('Independent validation requires at least 256 samples per point')
    rt=json.loads(args.profiles.read_text())['profiles'] if args.profiles.exists() else None
    data=json.loads(args.artifact.read_text())
    if not args.allow_fixed_beta and all(not isinstance(b,dict) for b in data['beta'].values()):
        raise SystemExit('Fixed EESM beta failed the development pilot; fit adaptive beta before final validation')
    if rt:
        missing=set(rt)-set(data['curves'])-set(data.get('excluded_nr_tuples',{}))
        if missing:raise ValueError(f'Fit the missing resource tuples first: {sorted(missing)}')
        # A projected NR code is separately validated on the same physical profile.
        for excluded in data.get('excluded_nr_tuples',{}):
            if excluded not in rt:continue
            table,mcs,re=map(int,excluded.split(':'))
            for lower in range(mcs-1,-1,-1):
                key=curve_key(table,lower,re)
                if key in data['curves']:
                    rt[key]=rt[excluded];break
    report=validation(args.artifact,args.samples,rt);target=args.output or args.artifact.with_name('nr_independent_validation.json');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    if report['passed']:
        data=json.loads(args.artifact.read_text());data['validation']={k:v for k,v in report.items() if k!='points'};data['validation']['report_file']=str(target);args.artifact.write_text(json.dumps(data,indent=2)+'\n')
    else:raise SystemExit('Independent NR validation failed; calibration artifact remains disabled')
