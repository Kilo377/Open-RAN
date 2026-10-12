"""Audit each slot of completed matrix artifacts; fail on any invariant violation."""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]


def audit(path, require_resource_metrics=False):
    status=json.loads((path/'run_status.json').read_text())
    saved=json.loads((path/'config_snapshot.json').read_text());m=saved['manifest']
    final=json.loads((path/'final_summary.json').read_text())
    active_re=m['num_symbols']*int(final['n_prbs_total'])*12
    assert status['completed'] and status['spark_verified'],path
    assert saved['configs']['phy']['rt_sys']['device']=='cuda'
    totals=defaultdict(int);perue=defaultdict(lambda:defaultdict(int));percell=defaultdict(lambda:defaultdict(float))
    max_power_error=0.;max_norm_error=0.;delays=[];unsupported=defaultdict(int);last_cell={}
    previous_counts={};last_completed={};last_association_change={}
    with (path/'rt_sys_trace.jsonl').open() as trace,(path/'slot_metrics.csv').open() as metrics:
        metric_reader=csv.DictReader(metrics)
        for slot,line in enumerate(trace,1):
            t=json.loads(line);row=next(metric_reader)
            assert t['slot_id']==slot==int(row['slot_id'])
            assert t['cache_key']==status['cache_key']
            if 'handover_counts' in t:
                for uid,count in t['handover_counts'].items():
                    if uid in previous_counts:
                        assert count-previous_counts[uid]==sum(e['ue_id']==uid and e['handover_completed'] for e in t['mobility_events'])
                previous_counts=t['handover_counts']
                for event in t['mobility_events']:
                    if event['handover_completed']:
                        uid=event['ue_id'];dwell=saved['configs']['phy']['rt_sys']['mobility']['min_dwell_ms']/m['slot_duration_ms']
                        if uid in last_completed and t['cells'][event['from_cell']]['budget_w']>0:
                            assert slot-last_completed[uid]>=dwell
                        last_completed[uid]=slot
                        totals['completed_handovers']+=1
            assert len(t['ues'])==len(m['ues']) and len({u['ue_id'] for u in t['ues']})==len(m['ues'])
            transmitted_energy=defaultdict(float);emitted=defaultdict(int);occupied=defaultdict(int)
            for group in t['precoders']:
                cid=group['cell_id'];streams=group['streams'];count=group['subcarrier_count']
                assert len(streams)<=group['beam_subspace_dimension']
                transmitted_energy[cid]+=count*sum(s['power_per_re_w'] for s in streams)
                emitted[cid]+=count*len(streams);occupied[cid]+=count
                if len({stream['ue_id'] for stream in streams})>1:totals['mu_resource_groups']+=1
                if len(streams)>1:totals['multi_layer_resource_groups']+=1
                for k in range(len(streams)):
                    norm=sum(re[k]**2+im[k]**2 for re,im in zip(group['weights_real'],group['weights_imag']))
                    error=abs(norm-1);max_norm_error=max(max_norm_error,error)
                    assert error<1e-4,(path,slot,'precoder norm',norm)
            for cid,cell in t['cells'].items():
                used=cell['average_used_w'];budget=cell['budget_w']
                assert cell['grid_re']==m['num_symbols']*m['fft_size']
                assert 0<=cell['occupied_re']<=active_re<=cell['grid_re']
                assert 0<=used<=budget+1e-6
                assert occupied[cid]==cell['occupied_re']
                assert emitted[cid]==cell['emitted_re_streams']
                error=abs(transmitted_energy[cid]/m['num_symbols']-used)
                max_power_error=max(max_power_error,error);assert error<1e-6
                assert used==0 or abs(used-budget)<1e-6
                percell[cid]['power_w_slot_sum']+=used
                percell[cid]['occupied_re_sum']+=cell['occupied_re']
                percell[cid]['grid_re_sum']+=cell['grid_re']
                percell[cid]['active_grid_re_sum']+=active_re
            delivered=sum(u['delivered_bytes'] for u in t['ues'])
            if require_resource_metrics:
                layers=sum(u['layers'] for u in t['ues'] if u['scheduled'])
                assert int(row['true_scheduled_layer_count_total'])==layers,(path,slot,'layer telemetry')
                grant_utilization=sum(c['occupied_re'] for c in t['cells'].values())/(active_re*len(m['cells']))
                assert abs(float(row['true_resource_grid_utilization'])-grant_utilization)<1e-6,(path,slot,'resource telemetry accounting')
                assert 0<=grant_utilization<=1
                if any(cell['occupied_re'] for cell in t['cells'].values()):
                    assert float(row['true_resource_grid_utilization'])>0,(path,slot,'resource telemetry')
            assert delivered==int(row['true_delivered_new_bytes_this_slot'])+int(row['true_delivered_retx_bytes_this_slot'])
            assert sum(u['tb_bytes'] for u in t['ues'])<=int(row['true_grant_bytes_budget_total'])
            assert float(row['true_queue_bytes'])>=0 and float(row['true_total_pending_retx_bytes'])>=0
            for u in t['ues']:
                assert 0<=u['data_re']<=u['allocated_re']
                assert 0<=u['delivered_bytes']<=u['tb_bytes']<=u['queue_bytes']
                assert u['delivered_bytes']==(u['tb_bytes'] if u['ack'] else 0)
                assert 0<=u['bler']<=1 and math.isfinite(u['sinr_db'])
                if 'physical_tb_capacity_bytes' in u:
                    assert u['tb_bytes']<=u['physical_tb_capacity_bytes']
                    assert u['padding_bytes']==max(0,u['physical_tb_capacity_bytes']-u['tb_bytes'])
                if not u['scheduled']:assert u['delivered_bytes']==u['tb_bytes']==u['allocated_re']==0
                if u['precoder_column_norm_mean']==0:assert u['delivered_bytes']==0
                perue[u['ue_id']]['delivered_bytes']+=u['delivered_bytes']
                if u['ue_id'] in last_cell and last_cell[u['ue_id']]!=u['cell_id']:
                    uid=u['ue_id'];dwell=saved['configs']['phy']['rt_sys']['mobility']['min_dwell_ms']/m['slot_duration_ms']
                    if uid in last_association_change and t['cells'][last_cell[uid]]['budget_w']>0:
                        assert slot-last_association_change[uid]>=dwell,(path,slot,'association dwell')
                    last_association_change[uid]=slot
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
    evaluation_path=path/'evaluation_summary.json'
    if evaluation_path.exists():
        evaluation=json.loads(evaluation_path.read_text())
        if evaluation['warmup_ms']==0:
            # The earliest repair matrix serialized a shallow trace copy before
            # adding mobility fields. Its live counter remains in this summary.
            # Keep the provenance explicit; never invent missing event records.
            totals['live_completed_handovers']=evaluation['completed_handovers']
            assert evaluation['completed_handovers']==totals['association_changes']
    duration_s=m['total_slots']*m['slot_duration_ms']/1000
    for stats in perue.values():stats['throughput_mbps']=stats['delivered_bytes']*8/duration_s/1e6
    for stats in percell.values():
        stats['throughput_mbps']=stats['delivered_bytes']*8/duration_s/1e6
        stats['active_resource_utilization']=stats['occupied_re_sum']/stats['active_grid_re_sum']
    return {'case':path.name,'scene':m['scene'],'ues':len(m['ues']),'seed':m['rng_seed'],
        'passed':True,'totals':dict(totals),'per_ue':{k:dict(v) for k,v in perue.items()},
        'mobility_events_recorded':bool(previous_counts),
        'handover_count_source':'trace_and_live_summary' if previous_counts else 'live_summary; per-slot events absent',
        'per_cell':{k:dict(v) for k,v in percell.items()},'max_power_error_w':max_power_error,
        'max_precoder_norm_error':max_norm_error,'control_measurement_age_max':max(delays),
        'unsupported_actions':dict(unsupported),'status':status}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'outputs/rt_sys_matrix')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/rt_sys_acceptance/matrix_audit.json')
    parser.add_argument('--require-resource-metrics',action='store_true')
    args=parser.parse_args();plan=json.loads((args.root/'run_plan.json').read_text())
    reports=[];errors=[]
    for case in plan:
        path=args.root/f'{case["scene"]}_u{case["ues"]}_s{case["seed"]}_dt{case["path_update_ms"]}'
        try:reports.append(audit(path,args.require_resource_metrics))
        except Exception as exc:errors.append({'case':path.name,'error':repr(exc)})
    target=args.output;target.parent.mkdir(parents=True,exist_ok=True)
    report={'completed':not errors and len(reports)==len(plan),'cases':reports,'errors':errors,
        'resource_metrics_checked':args.require_resource_metrics,
        'checks':'GPU completion, all slots/UEs, precoder normalization, resource/energy accounting, TB/ACK/queue budgets, delayed controller observations',
        'harq_scope':'ACK/TB accounting here; per-process queue/HARQ state checked by control probes and regression'}
    target.write_text(json.dumps(report,indent=2));print(json.dumps({'completed':report['completed'],'passed_cases':len(reports),'errors':errors},indent=2))
    if not report['completed']:raise SystemExit(1)
