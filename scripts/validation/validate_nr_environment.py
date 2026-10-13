"""Independent scalar/batched decoder checks on tables 1/2/3 and RT profiles."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from phy.rt_sys.reference import NrReferenceAbstraction
from phy.rt_sys.calibration import NRWaveform
from phy.rt_sys.contracts import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/rt_environment_repairs_20261013/nr_reference_validation.json')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'outputs'):
        raise ValueError('Output must remain inside copy')
    model=NrReferenceAbstraction(args.device)
    plan=[]
    for table,indices in ((1,(0,3,14,27)),(2,(0,3,14,26)),(3,(0,3,8,16))):
        for mcs in indices:
            plan.append((table,mcs,'flat',np.ones(432)*10**(5/10)))
            plan.append((table,mcs,'strong_selectivity',np.tile([1e-3,1e3],216)))
    profiles=json.loads((ROOT/'outputs/rt_sys_targeted_repairs/rt_profiles.json').read_text())['profiles']
    for key in ('1:20:1056','2:26:1056','1:14:990','1:24:924'):
        table,mcs,_=map(int,key.split(':'))
        plan.append((table,mcs,'saved_rt_profile',np.asarray(profiles[key]['profile'])*10**(37/10)))
    results=[]
    for slot,(table,mcs,label,values) in enumerate(plan):
        name={1:'qam64',2:'qam256',3:'qam64_low_se'}[table]
        r=model.evaluate(values[None,:,None],name,mcs,412,slot,'probe',fixed=True)
        assert r['ack_source']=='actual_nr_crc_and_bit_check'
        w=NRWaveform(table,r['mcs'],len(values),model.device)
        errors=int(w.simulate([0.],values,model.bler_samples,r['bler_probe_seed'])[0])
        nack=int(w.simulate([0.],values,1,r['ack_seed'])[0])
        assert errors==r['bler_probe_errors'] and nack==r['reference_tb_errors']
        assert w.bits//8==r['capacity_bytes']
        assert r==model.evaluate(values[None,:,None],name,mcs,412,slot,'probe',fixed=True)
        results.append(dict(table=table,mcs=r['mcs'],profile=label,data_re=len(values),
            coding_backend=w.coding_backend,probe_errors=errors,passed=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(passed=True,points=len(results),device=model.device,cases=results,
        source_sha256=source_fingerprint(ROOT),scope='Equivalent-link decoder execution, not full waveform or strict BLER-tail calibration')
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'points':len(results)}))


if __name__=='__main__':main()
