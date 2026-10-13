"""Independent GPU checks of local BLER lookup, actual feedback, and Chase."""
import json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.reference import NrReferenceAbstraction,CachedNrReferenceAbstraction,StatisticalBankAbstraction,probability_interval
from phy.rt_sys.contracts import source_fingerprint
from phy.rt_sys.nr_harq import NrHarqReceiver
from phy.rt_sys.calibration import NRWaveform


def main():
    out=ROOT/'outputs/rt_research_readiness_20261013/final_ran';out.mkdir(parents=True,exist_ok=True)
    fast=CachedNrReferenceAbstraction('cuda',config={'probability_bank':{'calibration_samples':512,'validation_samples':512}});normal=NrReferenceAbstraction('cuda')
    import torch
    cases=[]
    profiles=json.loads((ROOT/'outputs/rt_sys_targeted_repairs/rt_profiles.json').read_text())['profiles']
    plan=[('qam64',3,np.ones(324)*100,'flat'),('qam64_low_se',0,np.ones(324)*100,'low_rate'),
          ('qam64',3,np.tile([.01,1000.],162),'selective'),
          ('qam64',14,np.asarray(profiles['1:14:990']['profile'])*10**(37/10),'held_out_rt')]
    for table,mcs,v,label in plan:
        start=time.perf_counter();r=fast.evaluate(v,table,mcs,511,0,label,fixed=True);torch.cuda.synchronize()
        admission=time.perf_counter()-start
        w=fast.waveform({'qam64':1,'qam64_low_se':3}[table],mcs,len(v))
        held=int(w.simulate_fast([0.],v,1024,971+mcs)[0]);p=held/1024
        assert abs(p-r['tbler'])<=.1,(label,p,r['tbler'])
        torch.cuda.synchronize();start=time.perf_counter()
        old=[normal.evaluate(v,table,mcs,511,slot,label,fixed=True) for slot in range(1,21)]
        torch.cuda.synchronize();slow_s=time.perf_counter()-start
        start=time.perf_counter();new=[fast.evaluate(v,table,mcs,511,slot,label,fixed=True) for slot in range(1,21)]
        torch.cuda.synchronize();fast_s=time.perf_counter()-start
        assert all(a['ack']==b['ack'] and a['capacity_bytes']==b['capacity_bytes'] and a['mcs']==b['mcs'] for a,b in zip(old,new))
        assert all(r['bler_estimate_source']=='independently_validated_local_bler_bank' for r in new)
        cases.append(dict(profile=label,table=table,mcs=mcs,passed=True,independent_samples=1024,
            independent_bler=p,predicted_bler=r['tbler'],absolute_error=abs(p-r['tbler']),
            admission_s=admission,reference_s=slow_s,cached_s=fast_s,warm_speedup=slow_s/fast_s))
    # Include an interior BLER point; saturated 0/1 points alone cannot
    # validate probability interpolation or sampled feedback.
    wmid=NRWaveform(1,3,324,'cuda:0');lo,hi=-10.,10.
    for _ in range(8):
        midpoint=(lo+hi)/2;errors=int(wmid.simulate_fast([midpoint],np.ones(324),128,711)[0])
        if .15<errors/128<.85:break
        if errors/128>=.85:lo=midpoint
        else:hi=midpoint
    assert .15<errors/128<.85,'Interior BLER calibration point not found'
    vector=np.ones(324)*10**(midpoint/10)
    stat=StatisticalBankAbstraction('cuda',config={'probability_bank':{'calibration_samples':512,'validation_samples':512}})
    calibrated=stat.evaluate(vector,'qam64',3,911,0,'interior',fixed=True)
    assert calibrated['ack_source']=='validated_bler_bank_bernoulli'
    independent=int(wmid.simulate_fast([0.],vector,2048,8191)[0])/2048
    assert abs(independent-calibrated['tbler'])<=.08
    sampled=[stat.evaluate(vector,'qam64',3,911,slot,'interior',fixed=True) for slot in range(1,1025)]
    empirical=sum(not r['ack'] for r in sampled)/len(sampled)
    assert abs(empirical-independent)<=.08
    assert all(r['reference_tb_ack'] is None for r in sampled)
    h=NrHarqReceiver('chase');w=NRWaveform(1,3,324,'cuda:0')
    context=dict(process_id=1,ndi=1,is_retx=False,cell_id='cell_0')
    first=h.transmit(w,np.ones(324)*.1,501,1,'u',context)
    bits=h.pending[('u',1)]['bits'].clone();prior=h.pending[('u',1)]['llr'].clone()
    context['is_retx']=True
    second=h.transmit(w,np.ones(324)*100,501,2,'u',context)
    torch.testing.assert_close(bits,h.pending[('u',1)]['bits'])
    assert first['codeword_id']==second['codeword_id'] and second['ack']
    assert second['reason']=='chase_llr_combined'
    # Reproduce second-round demapper LLR without combining, using same word/noise.
    single=NrHarqReceiver('none');single.pending[('u',1)]=dict(h.pending[('u',1)],llr=None,rounds=1)
    single.transmit(w,np.ones(324)*100,501,2,'u',context)
    # Capture the raw second LLR independently through a decoder wrapper.
    original=w.decoder;captured=[]
    def record(llr):captured.append(llr.clone());return original(llr)
    w.decoder=record;single.transmit(w,np.ones(324)*100,501,2,'u',context);w.decoder=original
    torch.testing.assert_close(h.pending[('u',1)]['llr'],prior+captured[0])
    h.acknowledge('u',1);assert not h.pending
    report=dict(passed=True,source_sha256=source_fingerprint(ROOT),cases=cases,
        chase={'passed':True,'actual_gpu_crc':True,'same_codeword':True,'llr_sum_exact':True,'first':first,'second':second},
        statistical_feedback={'passed':True,'sinr_db':midpoint,'predicted_bler':calibrated['tbler'],
            'independent_bler':independent,'independent_nr_samples':2048,'sampled_feedback_bler':empirical,'feedback_samples':len(sampled)},
        bank_statistics=dict(fast.bank.stats),scope='Local ordered-SINR bin domain; actual TB ACK; no extrapolated EESM, no strict URLLC tail guarantee')
    (out/'link_validation_optimized.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
