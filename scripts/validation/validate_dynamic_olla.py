"""Actual business-TB step/delay/recovery tests, never link probes or BLER fitting."""
import argparse,json,sys,time
from collections import deque
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.direct import DirectNrAbstraction
from phy.rt_sys.olla import OllaState,HarqFeedbackEvent
from phy.rt_sys.tx_contract import nr_budget
from phy.rt_sys.reference import probability_interval
from phy.rt_sys.contracts import source_fingerprint
from phy.mcs_profiles import select_mcs_index_from_sinr
from traffic.packet_ledger import PacketLedger
from types import SimpleNamespace as NS


def run_case(out,profile,delay,seed,device,warmup,samples):
    out.mkdir(exist_ok=False);start=time.perf_counter();model=DirectNrAbstraction(device,config={'harq_combining':'none'})
    olla=OllaState();uid=f'{profile}_delay{delay}_s{seed}';table='qam64';cell='fixed';count=648
    shape=np.ones(count) if profile=='flat' else np.resize([.03,1.97],count)
    lc=NS(logical_channel_id='lc',queue_bytes=4096,max_queue_bytes=None,packet_delay_budget_ms=10000000.)
    ue=NS(ue_id=uid,logical_channels=[lc]);ledger=PacketLedger(ue,.5)
    clock=0;reports=deque();latest=None;first_id=0;attempts=0;phases=[]
    def tick(db):
        nonlocal clock,latest
        clock+=1;olla.consume(clock,cell)
        reports.append(dict(produced_slot=clock,available_slot=clock+delay,reference_sinr_db=db))
        while reports and reports[0]['available_slot']<=clock:latest=reports.popleft()
    plan=[('initial',0.,warmup+samples,True),('step_up',8.,warmup+samples,True),
          ('step_down',0.,warmup+samples,True),('short_dip',-12.,50,False),
          ('recovery',0.,warmup+samples,True)]
    with (out/'actual_tbs.jsonl').open('w') as stream:
        for phase,db,length,stable in plan:
            errors=n=saturated=0;history=[];mcs_counts={};window=[]
            for index in range(length):
                tick(db);offset=olla.offsets.get(table,0.)
                measured=latest['reference_sinr_db'] if latest else None
                mcs=select_mcs_index_from_sinr(measured+offset,mcs_table_id=table,max_mcs_index=27) if latest else 0
                _,_,bits=nr_budget(table,mcs,count)
                if ledger.available('lc')<bits//8:ledger.add(lc,4096,clock)
                payload=min(bits//8,ledger.available('lc'));assert payload>0
                tb=f'{uid}:{first_id}';first_id+=1;ledger.reserve(tb,[('lc',payload)])
                context=dict(process_id=0,ndi=first_id%2,is_retx=False,cell_id=cell)
                ack=False
                for round_id in range(4):
                    r=model.evaluate(10**(db/10)*shape,table,mcs,seed,clock,uid,harq_context=context)
                    assert r['reference_tb_ack'] is not None
                    ack=bool(r['reference_tb_ack']);attempts+=1
                    event=HarqFeedbackEvent(tb,cell,table,mcs,ack,round_id>0,clock+1)
                    olla.pending.append(event)
                    report=dict(tb_id=tb,slot_id=clock,phase=phase,phase_first_index=index,
                        reference_sinr_db=measured,measurement_produced_slot=latest['produced_slot'] if latest else None,
                        measurement_delivered_slot=latest['available_slot'] if latest else None,
                        actual_channel_db=db,offset_db=offset,mcs=mcs,round=round_id,ack=ack,
                        payload_bytes=payload,feedback_available_slot=clock+1,
                        post_warmup=stable and index>=warmup)
                    stream.write(json.dumps(report)+'\n')
                    if round_id==0:
                        history.append(dict(index=index,slot_id=clock,offset_db=offset,ack=ack,mcs=mcs))
                        if stable and index>=warmup:
                            n+=1;errors+=not ack;saturated+=mcs in (0,27) or abs(offset)>=10-1e-9
                            mcs_counts[mcs]=mcs_counts.get(mcs,0)+1
                        window.append(not ack)
                    if ack:
                        ledger.acknowledge(tb,clock);model.harq.acknowledge(uid,0);break
                    if round_id<3:tick(db);context['is_retx']=True
                if not ack:ledger.release(tb);model.harq.acknowledge(uid,0)
                if index%1000==0:
                    (out/'progress.json').write_text(json.dumps(dict(phase=phase,first_index=index,total_firsts=first_id,actual_decodes=attempts,elapsed_s=time.perf_counter()-start)))
                ledger.drain_events()
            eligible=stable and n>=2000 and saturated/max(n,1)<.05
            windows=[dict(first_start=i,first_end=min(i+200,len(window)),first_bler=sum(window[i:i+200])/len(window[i:i+200])) for i in range(0,len(window),200)]
            recovery=next((w['first_end'] for j,w in enumerate(windows[:-1]) if .07<=w['first_bler']<=.13 and .07<=windows[j+1]['first_bler']<=.13),None)
            phases.append(dict(phase=phase,channel_db=db,first_samples=n,first_errors=errors,
                first_bler=errors/n if n else None,wilson_95=probability_interval(errors,n) if n else None,
                saturated_fraction=saturated/max(n,1),eligible=eligible,
                target_status=('passed' if .07<=errors/n<=.13 else 'failed') if eligible else 'transient_or_saturated_or_insufficient',
                offset_start_db=history[0]['offset_db'],offset_end_db=history[-1]['offset_db'],
                mcs_counts=mcs_counts,windows=windows,recovery_first_index=recovery,
                recovery_ms=(history[recovery-1]['slot_id']-history[0]['slot_id'])*.5 if recovery else None,
                recovery_definition='First of two consecutive nonoverlapping 200-first-TB windows within 0.07-0.13; descriptive only'))
    olla.consume(clock+1,cell)
    assert attempts==model.decode_trials
    assert all(model.statistics.get(k,0)==0 for k in ('amc_probe_trials','bler_probe_trials','calibration_decode_trials'))
    result=dict(case=uid,phases=phases,actual_decodes=attempts,statistics=model.statistics,
        elapsed_s=time.perf_counter()-start,structural_passed=True,
        passed=all(p['target_status']=='passed' for p in phases if p['phase']!='short_dip'),
        final_olla=olla.summary(),packet_metrics=ledger.summary(),measurement_delay_slots=delay,
        scope='Backlogged business-TB fixture; noiseless delayed reference reports isolate adaptation. Not an RT capacity test.',
        confidence_scope='Descriptive Wilson interval; correlated adaptation samples do not establish independent error trials.')
    (out/'validation.json').write_text(json.dumps(result,indent=2));return result


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);sha=source_fingerprint(ROOT);cases=[]
    for profile in ('flat','selective'):
        for delay in (2,8):
            for seed in (512,513):
                label=f'{profile}_delay{delay}_s{seed}'
                r=run_case(a.output/label,profile,delay,seed,a.device,a.warmup,a.samples);cases.append(r)
                (a.output/'progress.json').write_text(json.dumps(dict(completed=len(cases),total=8,cases=cases),indent=2))
                print(label,r['passed'],r['elapsed_s'],flush=True)
                if not r['passed']:break
            if cases and not cases[-1]['passed']:break
        if cases and not cases[-1]['passed']:break
    result=dict(completed=len(cases)==8,passed=len(cases)==8 and all(c['passed'] for c in cases),source_sha256=sha,cases=cases)
    assert source_fingerprint(ROOT)==sha
    (a.output/'validation.json').write_text(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('Stable eligible segment failed; preserve data and inspect before expansion')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--warmup',type=int,default=1500);p.add_argument('--samples',type=int,default=3000)
    a=p.parse_args();main(a)
