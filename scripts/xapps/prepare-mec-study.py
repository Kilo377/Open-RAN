"""Recompute unchanged UI metrics and retain task identities for paired audits."""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import sys
import argparse

REPO=Path(__file__).resolve().parents[2]
pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--study',type=Path)
initial,_=pre.parse_known_args();OUT=initial.study.resolve() if initial.study else Path.cwd()
KEYS=['mec_offloaded_latency_ms','network_goodput_mbps','ue_power_w','ap_power_w',
      'urllc_failure_fraction','urllc_packet_latency_ms','task_latency_ms','task_success_ratio']

def file_sha256(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()

def read_lines(path):
    gz=path.with_suffix(path.suffix+'.gz')
    target=path if path.is_file() else gz
    if not target.is_file():return []
    with (gzip.open(target,'rt') if target.suffix=='.gz' else target.open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]

def cell_tasks(index,start,end,uid):
    cutoff=end*index.dt-.1
    tasks=[dict(t,task_key=str(key)) for key,t in index.tasks.items() if t['ue_id']==uid
           and start<t['born_slot']<=end and start*index.dt-1e-10<=t['created_s']<cutoff-1e-10]
    off=[t for t in tasks if t['offloaded']]
    completed=[t for t in off if t.get('terminal',{}).get('kind')=='task_completed' and t['terminal']['recorded_slot']<=end]
    unresolved=[t for t in off if not t.get('terminal') or t['terminal']['recorded_slot']>end]
    lat=[t['terminal']['latency_ms'] for t in completed]
    return dict(value=sum(lat)/len(lat) if lat and not unresolved else None,sample_count=len(lat),
                generated=len(off),unresolved=len(unresolved),failed=sum(t.get('terminal',{}).get('kind') in ('task_queue_overflow','task_result_overflow') and t['terminal']['recorded_slot']<=end for t in off),tasks=off)

def execution_audit(run,decisions,last_slot=None):
    wanted={(d['slot_id']+1,d['cell_id']):d for d in decisions
            if d.get('kind')=='decision' and isinstance(d.get('ue_scheduler_weights'),dict)
            and (last_slot is None or d['slot_id']+1<=last_slot)}
    matched={};rejected=[]
    path=run/'control_receipts.jsonl.gz'
    if not wanted:return dict(applicable=False)
    with gzip.open(path,'rt') as stream:
        for line in stream:
            r=json.loads(line)
            if not any(p.get('instance_id')=='latency_scheduler_rf' for p in r.get('provenance',[])):continue
            if r.get('status')!='applied':rejected.append({k:r.get(k) for k in ('slot_id','status','reason','target_cell_id')})
            for cid,cfg in r.get('after',{}).items():
                key=(r.get('apply_slot'),cid);d=wanted.get(key)
                if d is None:continue
                if cfg.get('scheduler_mode')!=d['scheduler_mode']:continue
                if any(cfg.get('ue_scheduler_weights',{}).get(uid)!=v for uid,v in d['ue_scheduler_weights'].items()):continue
                matched[key]=r.get('control_request_id')
    return dict(applicable=True,proposed_decisions=len(wanted),matched_normal_execution=len(matched),
        all_proposed_actions_verified=len(matched)==len(wanted) and not rejected,rejected=rejected,
        missing=[dict(slot=s,cell_id=c) for s,c in wanted if (s,c) not in matched],
        first_apply_slot=min((s for s,c in matched),default=None),
        fields=['scheduler_mode','ue_scheduler_weights'],ordinary_ric_and_delayed_ran_path=True)

def one(folder,origin_override=None):
    sys.path.insert(0,str(REPO))
    from smo.window_metrics import WindowIndex
    from smo.runtime import trace_users
    cfg=json.loads((folder/'run/config_snapshot.json').read_text())
    case=json.loads((folder/'case.json').read_text());original_case=dict(case)
    if origin_override is not None:
        if case['arm']!='OFF' or case['split'] not in ('training','diagnostic'):raise ValueError('Only genuine training/native diagnostic OFF records can be reused')
        case.update(name='derived_native__'+case['name']+'__o'+str(origin_override),origin_slot=origin_override,split='training',record_kind='derived_native_reference')
    origin=case.get('origin_slot',2040)
    index=WindowIndex(cfg);prefix=hashlib.sha256()
    with gzip.open(folder/'run/rt_sys_trace.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);slot=row['slot_id']
            if origin_override is not None and row.get('xapp_bindings'):raise ValueError('Reused native record contains a deployed xApp')
            index.observe(slot,trace_users(row),row.get('ue_services',{}),row.get('ap_services',{}).get('cells',{}),row.get('ue_service_events',[]))
            if slot<=origin:
                # Stable network evidence excludes file paths, actor identities and timing.
                reduced={k:row.get(k) for k in ('slot_id','ue_service_events','ue_services','ap_services','cells','ues','traffic_arrivals')}
                prefix.update(json.dumps(reduced,sort_keys=True,separators=(',',':')).encode())
    if index.task_errors:raise ValueError(index.task_errors[:3])
    if len(index.slots)!=case['slots'] or any(slot!=i+1 for i,slot in enumerate(index.slots)):
        raise ValueError('Completed case has a missing or duplicated physical time slot')
    lengths=[500]
    for length in (1000,2000,4000,6000,10000,20000):
        if origin+length<=index.slots[-1]:lengths.append(length)
    windows={str(n):dict(aggregate=index.window(KEYS,origin,origin+n,through=index.slots[-1]),
                        per_ue={uid:cell_tasks(index,origin,origin+n,uid) for uid in ('ue_0','ue_4')}) for n in lengths}
    before=index.window(KEYS,origin-2000,origin,through=index.slots[-1])
    decisions=read_lines(folder/'run/latency_rf_events.jsonl')
    perf=json.loads((folder/'run/performance.json').read_text())
    result=dict(case=case,folder=str(folder),before=before,windows=windows,decisions=decisions,
        contiguous_slots_verified=len(index.slots),
        execution_audit=execution_audit(folder/'run',decisions,index.slots[-1]),
        network_prefix_sha256=prefix.hexdigest(),simulation_s=perf['simulation_s'],
        source_sha256=cfg.get('source_sha256'),trace_sha256=file_sha256(folder/'run/rt_sys_trace.jsonl.gz'))
    if origin_override is None:
        temp=folder/'metrics.tmp';temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(folder/'metrics.json')
    else:
        result.update(derived_from_case=original_case,new_simulations=0,source_simulation_s=result.pop('simulation_s'))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--study",type=Path,required=True);parser.parse_args()
    folders=[p.parent for p in OUT.glob('workers/*/*/archive-verified.json')
             if (p.parent/'complete.json').is_file()
             and json.loads((p.parent/'complete.json').read_text()).get('state')=='complete']
    todo=[p for p in folders if not (p/'metrics.json').is_file()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for v in pool.map(one,todo):print(json.dumps(dict(case=v['case']['name'],short=v['windows']['500']['aggregate']['states']['kpi:mec_offloaded_latency_ms'],seconds=v['simulation_s'])),flush=True)
    rows=[json.loads((p/'metrics.json').read_text()) for p in folders if (p/'metrics.json').is_file()]
    temp=OUT/'analysis/case-index.tmp';temp.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n');temp.replace(OUT/'analysis/case-index.json')
    print(json.dumps(dict(complete=len(rows),new=len(todo))))

if __name__=='__main__':main()
