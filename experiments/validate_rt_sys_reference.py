"""Actual NR reference vs separate non-vectorised decoder execution on GPU.

No fitted SINR/BLER model participates. This verifies the new execution path,
including CRC/bit failures, SYS TBS/probability bridging and keyed replay.
It does not certify waveform interference, CSI estimation or HARQ combining.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from phy.rt_sys.reference import NrReferenceAbstraction
from phy.rt_sys.calibration import NRWaveform
from phy.rt_sys.contracts import source_fingerprint


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cpu','cuda']);args=p.parse_args()
    baseline=json.loads((ROOT/'outputs/rt_sys_calibration_repair/nr_model.json').read_text())
    rt=json.loads((ROOT/'outputs/rt_sys_targeted_repairs/rt_profiles.json').read_text())['profiles']
    model=NrReferenceAbstraction(args.device);records=[];start=time.monotonic();cases=[]
    for key,c in baseline['curves'].items():
        table,mcs,count=map(int,key.split(':'))
        if count!=324:continue
        center=float(np.interp(.5,np.asarray(c['tbler'])[::-1],np.asarray(c['snr_db'])[::-1]))
        for offset in (-2.75,-1.25,-.25,.25,1.25,2.75):cases.append((key,'flat',np.ones(count)*10**((center+offset)/10)))
    for key in ('1:20:1056','2:26:1056','1:14:990','1:24:924'):
        profile=np.asarray(rt[key]['profile'])
        for db in (31.,37.,50.,64.):cases.append((key,'rt_development_stress',profile*10**(db/10)))
    for db in (23.,24.,25.):cases.append(('2:26:3888','multi_cb',np.ones(3888)*10**(db/10)))
    for index,(key,profile_label,values) in enumerate(cases):
        table,mcs,count=map(int,key.split(':'));table_name={1:'qam64',2:'qam256'}[table]
        result=model.evaluate(values[None,:,None],table_name,mcs,211,index,'validation',fixed=True)
        wave=NRWaveform(table,mcs,count,model.device)
        errors=int(wave.simulate([0.],values,model.bler_samples,result['bler_probe_seed'])[0])
        failure=int(wave.simulate([0.],values,1,result['ack_seed'])[0])
        ok=errors==result['bler_probe_errors'] and failure==result['reference_tb_errors'] and result['capacity_bytes']==wave.bits//8 and abs(result['tbler']-errors/model.bler_samples)<1e-6
        replay=model.evaluate(values[None,:,None],table_name,mcs,211,index,'validation',fixed=True)
        ok=ok and replay==result
        records.append({'case':key,'profile':profile_label,'passed':bool(ok),'probe_errors':errors,'samples':model.bler_samples,
            'reference_tb_errors':failure,'num_cb':wave.num_cb,'capacity_bytes':result['capacity_bytes']})
        if not ok:raise AssertionError(records[-1])
        if (index+1)%26==0:print(json.dumps({'checked':index+1,'total':len(cases),'elapsed_s':time.monotonic()-start}),flush=True)
    output=ROOT/'outputs/rt_sys_targeted_repairs/reference_validation.json'
    output.write_text(json.dumps({'passed':True,'device':model.device,'source_sha256':source_fingerprint(ROOT),'cases':records,
        'flat_points':sum(r['profile']=='flat' for r in records),'default_bler_eesm_used':False,
        'vectorised_vs_nonvectorised_exact':True,'keyed_replay_exact':True,'elapsed_s':time.monotonic()-start,
        'scope':'NR CRC/LDPC/rate matching/APP QAM over independent post-LMMSE Gaussian resources; not full OFDM/interference/HARQ calibration'},indent=2)+'\n')
    print(json.dumps({'passed':True,'points':len(records),'elapsed_s':time.monotonic()-start}),flush=True)
