"""Actual NR CRC/LDPC/QAM/AWGN validation of the SYS abstraction domain.

The report includes sampling uncertainty. A finite AWGN check does not
certify URLLC tails, arbitrary code blocks, or frequency-selective EESM beta.
"""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def wilson(errors,samples):
    p=errors/samples;z=1.959963984540054;den=1+z*z/samples
    center=(p+z*z/(2*samples))/den
    radius=z*math.sqrt(p*(1-p)/samples+z*z/(4*samples*samples))/den
    return [max(0,center-radius),min(1,center+radius)]


def calibrate(device,samples,batch,data_re,selective=False,full_mcs=False,bler_files=None,beta_file=None,calibration_file=None):
    import numpy as np
    import torch
    from phy.rt_sys.abstraction import SionnaAbstraction
    from sionna.phy.nr import TBEncoder,TBDecoder
    from sionna.phy.mapping import Mapper,Demapper
    model=SionnaAbstraction(device,bler_table_files=bler_files,beta_table_file=beta_file,calibration_file=calibration_file); dev=model.device; results=[]
    cases=[('qam64',3),('qam64',10),('qam64',20),('qam256',20)]
    if full_mcs:
        cases=[(table,int(mcs)) for table,index in [('qam64',1),('qam256',2)] if index in model.phy.bler_table['category'][1]['index']
            for mcs in model.phy.bler_table['category'][1]['index'][index]['MCS']]
    profile=np.ones(data_re);profile[:data_re//2]=.1 if selective else 1
    profile[data_re//2:]=1.9 if selective else 1
    for table,mcs in sorted(cases):
        metrics=model.evaluate(np.ones((1,data_re,1))*10,table,mcs,1,0,'probe',fixed=True)
        q=metrics['qm'];bits=metrics['capacity_bytes']*8;coded=data_re*q
        encoder=TBEncoder(bits,coded,metrics['rate'],q,channel_type='PDSCH',device=dev)
        decoder=TBDecoder(encoder,num_bp_iter=20,device=dev)
        mapper=Mapper('qam',q,device=dev);demapper=Demapper('app','qam',q,device=dev)
        snrs=(-5,0,5,10,15,20)
        if full_mcs:
            search=[(abs(model.evaluate((profile*10**(db/10))[None,:,None],table,mcs,1,0,'search',fixed=True)['tbler']-.5),db) for db in range(-5,31)]
            center=min(search)[1];snrs=[center+x for x in (-3,-1.5,0,.5,1.5,3)]
        for snr_db in snrs:
            generator=torch.Generator(device=dev);generator.manual_seed(int(1701+mcs*100+snr_db*10))
            no=10**(-snr_db/10); errors=crc_errors=0
            with torch.no_grad():
                for start in range(0,samples,batch):
                    n=min(batch,samples-start)
                    u=torch.randint(0,2,(n,bits),device=dev,generator=generator).float()
                    x=mapper(encoder(u))
                    noise=(torch.randn(x.shape,device=dev,generator=generator)+1j*torch.randn(x.shape,device=dev,generator=generator))*math.sqrt(no/2)
                    gain=torch.as_tensor(np.sqrt(profile),dtype=torch.float32,device=dev)
                    llr=demapper((x*gain+noise)/gain,torch.as_tensor(no/profile,dtype=torch.float32,device=dev))
                    decoded,valid=decoder(llr)
                    errors+=int((torch.any(decoded!=u,dim=-1)|~valid.bool()).sum())
                    crc_errors+=int((~valid.bool()).sum())
            sys=model.evaluate((profile*10**(snr_db/10))[None,:,None],table,mcs,1,1,'probe',fixed=True)
            interval=wilson(errors,samples); predicted=sys['tbler']
            results.append({'table':table,'mcs':mcs,'data_re':data_re,'information_bits':bits,
                'coded_bits':coded,'snr_db':snr_db,'samples':samples,'errors':errors,'crc_errors':crc_errors,
                'empirical_bler':errors/samples,'wilson_95':interval,'sys_bler':predicted,
                'within_interval_plus_0_10':interval[0]-.10<=predicted<=interval[1]+.10})
            print(json.dumps(results[-1]),flush=True)
    return {'device':dev,'actual_nr_crc_ldpc_qam_awgn':True,'samples_per_point':samples,
        'passed':all(r['within_interval_plus_0_10'] for r in results),'points':results,
        'validated_domain':{'channel':'two-level frequency selective AWGN' if selective else 'flat AWGN','data_re':data_re,'tested_tables_mcs':cases},
        'criterion':'SYS TBLER inside empirical 95% interval expanded by absolute 0.10',
        'limitations':['Not an URLLC tail guarantee','No EESM beta fit for arbitrary frequency selective channels','Other MCS/code-block sizes remain unvalidated']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--samples',type=int,default=512);p.add_argument('--batch',type=int,default=64);p.add_argument('--data-re',type=int,default=324)
    p.add_argument('--frequency-selective',action='store_true');p.add_argument('--full-mcs',action='store_true')
    p.add_argument('--bler-table',action='append');p.add_argument('--beta-table')
    p.add_argument('--nr-calibration')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/rt_sys_repairs/phy_calibration.json');args=p.parse_args()
    if min(args.samples,args.batch,args.data_re)<=0:raise ValueError('Positive sample/resource budgets required')
    report=calibrate(args.device,args.samples,args.batch,args.data_re,args.frequency_selective,args.full_mcs,args.bler_table,args.beta_table,args.nr_calibration);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit('PHY calibration exposes domain mismatch; see report')
