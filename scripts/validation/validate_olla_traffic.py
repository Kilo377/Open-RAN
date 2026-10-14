"""Finite actual business-TB OLLA tests; zero probability probes or fitted tables.

Fixed flat/selective equivalent NR channels isolate the DU adaptation loop.
Every decoder invocation carries reserved business bytes; retransmissions retain
bits/format. These tests do not claim full OFDM or RT scene execution.
"""
import argparse,json,sys,time,os
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from types import SimpleNamespace as NS
from phy.rt_sys.direct import DirectNrAbstraction
from phy.rt_sys.olla import OllaState,HarqFeedbackEvent
from phy.rt_sys.tx_contract import nr_budget,codec_format_supported
from phy.rt_sys.reference import probability_interval
from phy.rt_sys.contracts import source_fingerprint
from phy.mcs_profiles import select_mcs_index_from_sinr
from traffic.packet_ledger import PacketLedger


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();sha=source_fingerprint(ROOT)
    results=[]
    for profile,db in [('flat',0.),('flat',8.),('selective',0.),('selective',8.)]:
        name=f'{profile}_{db:g}dB';folder=out/name;folder.mkdir()
        model=DirectNrAbstraction(args.device,config={'harq_combining':'chase'})
        olla=OllaState();cell='fixed';table='qam64';count=648
        shape=np.ones(count) if profile=='flat' else np.resize(np.array([.03,1.97]),count)
        values=10**(db/10)*shape
        lc=NS(logical_channel_id='lc',queue_bytes=4_000_000,max_queue_bytes=None,packet_delay_budget_ms=100000)
        ue=NS(ue_id=name,logical_channels=[lc]);ledger=PacketLedger(ue,.5)
        slot=0;errors=samples=saturated=attempts=0;mcs_counts={};history=[]
        with (folder/'actual_tbs.jsonl').open('w') as log:
            for first in range(args.warmup+args.samples):
                slot+=1;olla.consume(slot,cell)
                offset=olla.offsets.get(table,0.)
                mcs=select_mcs_index_from_sinr(db+offset,mcs_table_id=table,max_mcs_index=27)
                if not codec_format_supported(table,mcs,count):raise ValueError('Unsupported fixed test format')
                _,_,bits=nr_budget(table,mcs,count)
                payload=min(bits//8,ledger.available('lc'))
                if payload<=0:raise RuntimeError('Fixed traffic buffer exhausted')
                tb=f'{name}:{first}';ledger.reserve(tb,[('lc',payload)])
                context=dict(process_id=0,ndi=first%2,is_retx=False,cell_id=cell)
                ack=False
                for round_id in range(4):
                    result=model.evaluate(values,table,mcs,10117,slot,name,harq_context=context)
                    if result['reference_tb_ack'] is None:raise AssertionError('Real business TB did not execute')
                    ack=bool(result['reference_tb_ack']);attempts+=1
                    olla.pending.append(HarqFeedbackEvent(tb,cell,table,mcs,ack,round_id>0,slot+1))
                    log.write(json.dumps(dict(tb_id=tb,slot_id=slot,mcs=mcs,round=round_id,ack=ack,
                        payload_bytes=payload,offset_db=offset,first_index=first,post_warmup=first>=args.warmup))+'\n')
                    if round_id==0 and first>=args.warmup:
                        samples+=1;errors+=not ack
                        saturated+=mcs in (0,27) or abs(offset)>=10-1e-9
                        mcs_counts[mcs]=mcs_counts.get(mcs,0)+1
                        history.append(offset)
                    if ack:
                        ledger.acknowledge(tb,slot);model.harq.acknowledge(name,0);break
                    slot+=1;olla.consume(slot,cell);context['is_retx']=True
                if not ack:
                    ledger.release(tb);model.harq.acknowledge(name,0)
            olla.consume(slot+1,cell)
        statistics=model.statistics
        assert statistics['amc_probe_trials']==statistics['bler_probe_trials']==statistics['calibration_decode_trials']==0
        assert model.decode_trials==attempts
        rate=errors/samples
        eligible=samples>=2000 and saturated/samples<.05
        report=dict(case=name,first_samples=samples,first_errors=errors,first_bler=rate,
            wilson_95=probability_interval(errors,samples),saturated_fraction=saturated/samples,
            eligible_for_target_acceptance=eligible,target_passed=eligible and .07<=rate<=.13,
            mcs_counts=mcs_counts,final_olla=olla.summary(),offset_range_db=[min(history),max(history)],
            actual_decode_attempts=attempts,statistics=statistics,packet_metrics=ledger.summary(),
            source_sha256=sha,reference_sinr_db=db,channel_profile=profile,data_re_streams=count)
        (folder/'validation.json').write_text(json.dumps(report,indent=2));results.append(report)
        print(json.dumps({k:report[k] for k in ('case','first_bler','saturated_fraction','target_passed')}),flush=True)
    report=dict(passed=all(r['target_passed'] for r in results),cases=results,source_sha256=sha,
        elapsed_s=time.perf_counter()-start,scope=__doc__)
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    if source_fingerprint(ROOT)!=sha:raise RuntimeError('Source changed during validation')
    if not report['passed']:raise SystemExit('OLLA target gate failed; preserve actual outcomes')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=3000)
    p.add_argument('--warmup',type=int,default=1500);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args)
