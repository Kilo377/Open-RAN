"""Audit each slot of completed matrix artifacts; fail on any invariant violation."""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]


def audit(path):
    status=json.loads((path/'run_status.json').read_text())
    saved=json.loads((path/'config_snapshot.json').read_text());m=saved['manifest']
    assert status['completed'] and status['spark_verified'],path
    assert saved['configs']['phy']['rt_sys']['device']=='cuda'
    totals=defaultdict(int);perue=defaultdict(lambda:defaultdict(int));percell=defaultdict(lambda:defaultdict(float))
    max_power_error=0.;max_norm_error=0.;delays=[];unsupported=defaultdict(int);last_cell={}
    with (path/'rt_sys_trace.jsonl').open() as trace,(path/'slot_metrics.csv').open() as metrics:
        metric_reader=csv.DictReader(metrics)
        for slot,line in enumerate(trace,1):
            t=json.loads(line);row=next(metric_reader)
            assert t['slot_id']==slot==int(row['slot_id'])
            assert t['cache_key']==status['cache_key']
            assert len(t['ues'])==len(m['ues']) and len({u['ue_id'] for u in t['ues']})==len(m['ues'])
            transmitted_energy=defaultdict(float);emitted=defaultdict(int);occupied=defaultdict(int)
            for group in t['precoders']:
                cid=group['cell_id'];streams=group['streams'];count=group['subcarrier_count']
                assert len(streams)<=group['beam_subspace_dimension']
                transmitted_energy[cid]+=count*sum(s['power_per_re_w'] for s in streams)
                emitted[cid]+=count*len(streams);occupied[cid]+=count
                if len(streams)>1:totals['mu_resource_groups']+=1
                for k in range(len(streams)):
                    norm=sum(re[k]**2+im[k]**2 for re,im in zip(group['weights_real'],group['weights_imag']))
                    error=abs(norm-1);max_norm_error=max(max_norm_error,error)
                    assert error<1e-4,(path,slot,'precoder norm',norm)
            for cid,cell in t['cells'].items():
                used=cell['average_used_w'];budget=cell['budget_w']
                assert 0<=cell['occupied_re']<=cell['grid_re']
                assert 0<=used<=budget+1e-6
                assert occupied[cid]==cell['occupied_re']
                assert emitted[cid]==cell['emitted_re_streams']
                error=abs(transmitted_energy[cid]/m['num_symbols']-used)
                max_power_error=max(max_power_error,error);assert error<1e-6
                assert used==0 or abs(used-budget)<1e-6
                percell[cid]['power_w_slot_sum']+=used
                percell[cid]['occupied_re_sum']+=cell['occupied_re']
                percell[cid]['grid_re_sum']+=cell['grid_re']
            delivered=sum(u['delivered_bytes'] for u in t['ues'])
            assert delivered==int(row['true_delivered_new_bytes_this_slot'])+int(row['true_delivered_retx_bytes_this_slot'])
            assert sum(u['tb_bytes'] for u in t['ues'])<=int(row['true_grant_bytes_budget_total'])
            assert float(row['true_queue_bytes'])>=0 and float(row['true_total_pending_retx_bytes'])>=0
            for u in t['ues']:
                assert 0<=u['data_re']<=u['allocated_re']
                assert 0<=u['delivered_bytes']<=u['tb_bytes']<=u['queue_bytes']
                assert u['delivered_bytes']==(u['tb_bytes'] if u['ack'] else 0)
                assert 0<=u['bler']<=1 and math.isfinite(u['sinr_db'])
                if not u['scheduled']:assert u['delivered_bytes']==u['tb_bytes']==u['allocated_re']==0
                if u['precoder_column_norm_mean']==0:assert u['delivered_bytes']==0
                perue[u['ue_id']]['delivered_bytes']+=u['delivered_bytes']
                if u['ue_id'] in last_cell and last_cell[u['ue_id']]!=u['cell_id']:
                    totals['association_changes']+=1
                    perue[u['ue_id']]['association_changes']+=1
                last_cell[u['ue_id']]=u['cell_id']
                perue[u['ue_id']]['scheduled_slots']+=int(u['scheduled'])
                perue[u['ue_id']]['ack']+=int(u['ack'])
                perue[u['ue_id']]['nack']+=int(u['scheduled'] and not u['ack'] and u['tb_bytes']>0)
                percell[u['cell_id']]['delivered_bytes']+=u['delivered_bytes']
            age=int(row['control_input_measurement_age_slots']);delays.append(age)
            if slot>5:assert age>=saved['configs']['sim']['obs_delay_slots']
            for action in t.get('control_audit',[]):
                for field in action['fields']:
                    if field['route']=='unsupported':
                        assert field['reason'];unsupported[field['field']]+=1
            totals['slots']+=1;totals['ue_rows']+=len(t['ues']);totals['delivered_bytes']+=delivered
        assert next(metric_reader,None) is None
    assert totals['slots']==m['total_slots']
    duration_s=m['total_slots']*m['slot_duration_ms']/1000
    for stats in perue.values():stats['throughput_mbps']=stats['delivered_bytes']*8/duration_s/1e6
    for stats in percell.values():stats['throughput_mbps']=stats['delivered_bytes']*8/duration_s/1e6
    return {'case':path.name,'scene':m['scene'],'ues':len(m['ues']),'seed':m['rng_seed'],
        'passed':True,'totals':dict(totals),'per_ue':{k:dict(v) for k,v in perue.items()},
        'per_cell':{k:dict(v) for k,v in percell.items()},'max_power_error_w':max_power_error,
        'max_precoder_norm_error':max_norm_error,'control_measurement_age_max':max(delays),
        'unsupported_actions':dict(unsupported),'status':status}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'outputs/rt_sys_matrix')
    args=parser.parse_args();plan=json.loads((args.root/'run_plan.json').read_text())
    reports=[];errors=[]
    for case in plan:
        path=args.root/f'{case["scene"]}_u{case["ues"]}_s{case["seed"]}_dt{case["path_update_ms"]}'
        try:reports.append(audit(path))
        except Exception as exc:errors.append({'case':path.name,'error':repr(exc)})
    target=ROOT/'outputs/rt_sys_acceptance/matrix_audit.json';target.parent.mkdir(parents=True,exist_ok=True)
    report={'completed':not errors and len(reports)==len(plan),'cases':reports,'errors':errors,
        'checks':'GPU completion, all slots/UEs, precoder normalization, resource/energy accounting, TB/ACK/queue budgets, delayed controller observations',
        'harq_scope':'ACK/TB accounting here; per-process queue/HARQ state checked by control probes and regression'}
    target.write_text(json.dumps(report,indent=2));print(json.dumps({'completed':report['completed'],'passed_cases':len(reports),'errors':errors},indent=2))
    if not report['completed']:raise SystemExit(1)
