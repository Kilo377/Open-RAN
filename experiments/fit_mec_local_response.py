"""Reuse owned completed runs for a scoped observational MEC response model.

This extracts actual executions, not enabled-xApp labels or shadow actions.
It never edits source runs, launches a simulator or changes a trained xApp.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from smo.runtime import records,trace_users
from smo.window_metrics import WindowIndex

SCHEMA='oran.mec-response-dataset.v1'
VERSION=4


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n');tmp.replace(path)


def feature_names(cells,users):
    return [f'mode:{c}:{m}'for c in cells for m in('latency_aware','qos_deadline')]+['weight:'+u for u in users]


def encode_scheduler(state,cells,users):
    modes=state['scheduler_mode'];weights=state['ue_scheduler_weights']
    if set(modes)!=set(cells)or set(weights)!=set(users):raise ValueError('incomplete_scheduler_observation')
    if any(v not in('round_robin','latency_aware','qos_deadline')for v in modes.values()):raise ValueError('unknown_scheduler_mode')
    return [float(modes[c]==m)for c in cells for m in('latency_aware','qos_deadline')]+[float(weights[u])for u in users]


def extract_job(path,window_slots=500):
    path=Path(path);job=json.loads(path.read_text());run=path.parent/'data/run'
    out=dict(job_id=job['id'],run=str(run),app_ids=job['app_ids'],worker_id=job.get('worker_id'),
        status='excluded',extraction_version=VERSION,window_slots=window_slots)
    if job.get('phase')!='complete' or not run.is_dir():return dict(out,reason='incomplete_run')
    source_files=[run/'config_snapshot.json',run/'rt_sys_trace.jsonl.gz',run/'control_receipts.jsonl.gz',run/'xapp_decision_previews.jsonl']
    if not all(p.is_file()for p in source_files[:3]):return dict(out,reason='source_files_unavailable')
    # Receipt-window metadata and completion state are inputs too. A changed
    # job/window must invalidate a checkpoint even if its trace is unchanged.
    out['files_sha256']={str(p):sha(p)for p in source_files+[path,run/'run_status.json']if p.is_file()}
    cfg=json.loads(source_files[0].read_text());dt=cfg['manifest']['slot_duration_ms']
    applied=job.get('comparison',{}).get('application_slot')
    if type(applied)is not int:return dict(out,reason='application_receipt_unavailable')
    start,end=applied-1,applied-1+window_slots
    status=json.loads((run/'run_status.json').read_text())
    if end>status.get('last_completed_slot',0):return dict(out,reason='insufficient_retained_followup')
    cells=sorted(c['cell_id']for c in cfg['manifest']['cells']);users=sorted(u['ue_id']for u in cfg['manifest']['ues'])
    services=cfg['configs']['phy']['rt_sys']['ue_services']
    native_drx={u:{**services['drx'],**services.get('drx_by_ue',{}).get(u,{})}for u in users}
    native_offload={u:p['offload_fraction']for u,p in services['mec']['tasks_by_ue'].items()}
    assignments=None # actual RT association can differ from config placeholders
    # Compact actual receipts are authoritative. First-before states establish
    # the native value before the first recorded modification of a cell.
    initial=dict(scheduler_mode={},ue_scheduler_weights={});changes=[]
    for r in records(run,'control_receipts'):
        at=r.get('apply_slot',r.get('slot_id'))
        if r.get('status')!='applied'or type(at)is not int or at>end:continue
        for cid,before in r.get('before',{}).items():
            if not isinstance(before,dict):continue
            if 'scheduler_mode'in before:initial['scheduler_mode'].setdefault(cid,before['scheduler_mode'])
            for u,v in before.get('ue_scheduler_weights',{}).items():initial['ue_scheduler_weights'].setdefault(u,v)
        after={c:{k:deepcopy(v)for k,v in s.items()if k in('scheduler_mode','ue_scheduler_weights')}
            for c,s in r.get('after',{}).items()if isinstance(s,dict)}
        if after:changes.append((at,after))
    # Some untouched cells have no receipts. A captured native public report
    # before the deployment is an observation, not an assumed default.
    for preview in records(run,'xapp_decision_previews'):
        if preview.get('slot',end)>start:break
        for a in preview.get('apps',[]):
            for r in a.get('actions',[]):
                p=r['parameter'];entity=r.get('ue_id')or r.get('cell_id')
                if p in initial and r.get('current')is not None:initial[p].setdefault(entity,r['current'])
        if set(initial['scheduler_mode'])==set(cells)and set(initial['ue_scheduler_weights'])==set(users):break
    try:encode_scheduler(initial,cells,users)
    except ValueError as error:return dict(out,reason=str(error))
    state=deepcopy(initial);changes.sort(key=lambda r:r[0]);cursor=0;index=WindowIndex(cfg)
    sums=[0.] * len(feature_names(cells,users));segments=[];last=None;exogenous=hashlib.sha256();prefix=hashlib.sha256()
    seen_slots=[]
    for row in records(run,'rt_sys_trace'):
        slot=row['slot_id']
        if slot>end:break
        if assignments is None:assignments={u['ue_id']:u['cell_id']for u in row['ues']}
        seen_slots.append(slot)
        for u in row['ues']:
            if assignments.get(u['ue_id'])!=u['cell_id']:return dict(out,reason='association_changed')
        while cursor<len(changes)and changes[cursor][0]<=slot:
            for cid,values in changes[cursor][1].items():
                if 'scheduler_mode'in values:state['scheduler_mode'][cid]=values['scheduler_mode']
                state['ue_scheduler_weights'].update(values.get('ue_scheduler_weights',{}))
            cursor+=1
        # All non-scheduler controls must remain native through the complete
        # interval AND its preceding history; no hidden DRX/AP/offload mixture.
        for u in users:
            policy=row.get('ue_services',{}).get(u,{}).get('ue_service_policy',{})
            if policy.get('drx')!=native_drx[u]:return dict(out,reason='non_native_drx')
            if u in native_offload and policy.get('offload_fraction')!=native_offload[u]:return dict(out,reason='non_native_offload')
        ap=row.get('ap_services',{}).get('cells',{})
        if set(ap)!=set(cells)or any(v.get('ap_policy')!='active'for v in ap.values()):return dict(out,reason='non_native_ap_policy')
        births=[{k:e.get(k)for k in('ue_id','task_id','created_s','deadline_s')}for e in row.get('ue_service_events',[])if e['kind']=='task_generated']
        exogenous.update(canonical([slot,row.get('traffic_arrivals'),births]).encode())
        if slot<=start:prefix.update(canonical([slot,[(u['ue_id'],u.get('delivered_bytes'),u.get('packet_pending_bytes',u.get('queue_bytes')))for u in row['ues']]]).encode())
        index.observe(slot,trace_users(row),row.get('ue_services',{}),ap,row.get('ue_service_events',[]))
        if start<slot<=end:
            try:x=encode_scheduler(state,cells,users)
            except ValueError as error:return dict(out,reason=str(error))
            sums=[a+b for a,b in zip(sums,x)]
            signature=canonical(state)
            if signature!=last:segments.append(dict(relative_slot=slot-start,state=deepcopy(state)));last=signature
    if seen_slots!=list(range(1,end+1)):return dict(out,reason='incomplete_trace')
    metric=index.window(['mec_offloaded_latency_ms'],start,end,through=end)['states']['kpi:mec_offloaded_latency_ms']
    if metric.get('reason')or metric.get('value')is None:return dict(out,reason=metric.get('reason')or'missing_latency',metric=metric)
    if metric['sample_count']<3:return dict(out,reason='too_few_mature_tasks',metric=metric)
    context=dict(cells=cells,users=users,assignments=assignments,native_drx=native_drx,native_offload=native_offload,
        ue_services=services,cell_radio=cfg['configs']['cell']['cell_radio_profiles'],ue_config=cfg['configs']['ue']['ues'],
        seed=job['seed'],slot_duration_ms=dt,worker_source_archive_sha256=job.get('expected_source_archive_sha256'),source_sha256=cfg['source_sha256'],
        phy_model=cfg['configs']['phy']['rt_sys'].get('model'),fast_profile=cfg['configs']['phy']['rt_sys'].get('fast_profile'))
    stratum=digest(dict(seed=job['seed'],start=start,end=end,exogenous=exogenous.hexdigest(),prefix=prefix.hexdigest()))
    trajectory=digest(segments)
    return dict(out,status='usable',start_slot=start,end_slot=end,duration_ms=window_slots*dt,
        context=context,context_sha256=digest(context),stratum=stratum,trajectory_sha256=trajectory,
        exogenous_sha256=exogenous.hexdigest(),prefix_sha256=prefix.hexdigest(),features=feature_names(cells,users),
        x=[v/window_slots for v in sums],y=metric['value'],metric=metric,segments=segments,
        observation_unit='one_complete_run_window_not_a_slot_or_independent_seed')


def dataset(jobs_root,output,workers=4,supplement_roots=()):
    root=Path(jobs_root);output=Path(output);cache=output/'extracted';cache.mkdir(parents=True,exist_ok=True)
    paths=sorted(root.glob('*/job.json'))+[p for extra in supplement_roots for p in sorted(Path(extra).glob('*/job.json'))];rows=[];pending=[]
    for p in paths:
        saved=cache/(p.parent.name+'.json')
        cached=json.loads(saved.read_text())if saved.exists()else None
        if cached and cached.get('extraction_version')==VERSION and cached.get('files_sha256')and all(Path(f).is_file()and sha(f)==h for f,h in cached['files_sha256'].items()):rows.append(cached)
        else:pending.append(p)
    with ProcessPoolExecutor(max_workers=workers)as pool:
        futures={pool.submit(extract_job,str(p)):p for p in pending}
        for i,future in enumerate(as_completed(futures),1):
            p=futures[future]
            try:row=future.result()
            except Exception as error:row=dict(job_id=p.parent.name,status='excluded',reason='extraction_error',error=type(error).__name__+': '+str(error))
            save(cache/(p.parent.name+'.json'),row);rows.append(row)
            if i%10==0 or i==len(futures):print(canonical(dict(extracted=i,total=len(futures),usable=sum(r['status']=='usable'for r in rows))),flush=True)
    # Deduplicate executed trajectories within a matched time/exogenous stratum.
    unique={};conflicts=[]
    for row in sorted(rows,key=lambda r:r['job_id']):
        if row['status']!='usable':continue
        key=(row['context_sha256'],row['stratum'],row['trajectory_sha256'])
        if key not in unique:unique[key]=dict(row,source_job_ids=[row['job_id']]);continue
        old=unique[key]
        if abs(old['y']-row['y'])>1e-6:
            conflicts.append(dict(key=key,jobs=[old['job_id'],row['job_id']],values=[old['y'],row['y']]))
        old['source_job_ids'].append(row['job_id'])
    result=dict(schema=SCHEMA,extraction_version=VERSION,created_unix_s=time.time(),jobs_root=str(root),
        window_slots=500,rows=list(unique.values()),duplicate_response_discrepancies=conflicts,
        audit=[{k:r.get(k)for k in('job_id','status','reason','error')}for r in rows],
        raw_job_count=len(rows),usable_before_dedup=sum(r['status']=='usable'for r in rows),
        unique_rows=len(unique),independent_seeds=sorted({r['context']['seed']for r in unique.values()}),
        retention='source runs untouched; every extraction is checkpointed')
    save(output/'dataset.json',result)
    return result


def fit_and_report(data,output):
    from smo.mec_empirical import fit_dataset
    output=Path(output);model=fit_dataset(data)
    save(output/'model.json',model)
    save(output/'checkpoints'/('model-'+model['dataset_content_sha256'][:16]+'.json'),model)
    summary=dict(status=model['status'],coverage=model['coverage'],validation={k:v for k,v in model['validation'].items()if k!='predictions'},
        independent_seeds=model['independent_seeds'],causal_effect_identified=False,
        excluded_reasons={k:sum(r.get('reason')==k for r in data['audit'])for k in sorted({r.get('reason')for r in data['audit']if r.get('reason')})})
    save(output/'report.json',summary)
    with(output/'held-out-validation.csv').open('w')as stream:
        fields=['trajectory_sha256','actual','predicted','baseline','absolute_error','baseline_error','supported_direction','direction_correct']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(model['validation']['predictions'])
    p=model['validation']['predictions'];v=model['validation'];c=model['coverage']
    hi=max([1.]+[r[k]for r in p for k in('actual','predicted')])*1.1
    lo=min([0.]+[r['predicted']for r in p])
    px=lambda x:65+(x-lo)/(hi-lo)*450
    py=lambda y:405-(y-lo)/(hi-lo)*350
    points=''.join(f'<circle cx="{px(r["actual"]):.2f}" cy="{py(r["predicted"]):.2f}" r="5" fill="'+('#17665d'if r['supported_direction']else'#b07821')+f'"><title>Actual {r["actual"]:.2f} ms; predicted {r["predicted"]:.2f} ms</title></circle>'for r in p)
    plot=f'<svg viewBox="0 0 560 460" role="img" aria-label="Whole-trajectory held-out predictions"><path d="M65 55 V405 H515" fill="none" stroke="#63756d"/><path d="M65 405 L515 55" fill="none" stroke="#9aaba3" stroke-dasharray="5 5"/>{points}<text x="250" y="447">Observed latency (ms)</text><text x="16" y="265" transform="rotate(-90 16 265)">Held-out prediction (ms)</text></svg>'
    rows=''.join(f'<tr><td>{i+1}</td><td>{r["actual"]:.3f}</td><td>{r["predicted"]:.3f}</td><td>{r["absolute_error"]:.3f}</td><td>{"Yes"if r["supported_direction"]else"No"}</td></tr>'for i,r in enumerate(p))
    html=f'''<!doctype html><meta charset="utf-8"><title>MEC response approximation</title><style>body{{max-width:1000px;margin:40px auto;padding:0 24px;font:15px/1.6 system-ui;color:#19332d;background:#f7faf8}}h1{{font-size:26px}}.stats{{display:flex;gap:28px;flex-wrap:wrap;background:white;padding:20px}}strong{{display:block;font-size:24px}}svg{{max-width:580px;background:white;margin:20px 0}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dde6e0}}small{{color:#5a6d65}}</style>
    <h1>MEC response approximation</h1><div class="stats"><span>Retained runs<strong>{c['raw_jobs']}</strong></span><span>Unique windows<strong>{c['unique_run_windows']}</strong></span><span>Identified rank<strong>{c['identified_rank']} / {c['feature_dimension']}</strong></span><span>Held-out MAE<strong>{v['mae_ms']:.2f} ms</strong></span><span>Direction accuracy<strong>{100*v['direction_accuracy']:.1f}%</strong></span></div>
    <p>Target: mean offloaded MEC task end-to-end latency. Scheduler mode and actual UE weights; native DRX, offloading and AP policy. Nested validation leaves complete executed trajectories out.</p>{plot}
    <p><small>Teal: direction supported by the fold's observed subspace. Amber: unsupported direction in that fold. One exogenous seed; observational approximation, not independent-seed causal validation. Error envelopes are descriptive, not confidence intervals.</small></p>
    <table><thead><tr><th>Condition</th><th>Observed (ms)</th><th>Held-out prediction (ms)</th><th>Error (ms)</th><th>Fold direction supported</th></tr></thead><tbody>{rows}</tbody></table>'''
    (output/'report.html').write_text(html)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--supplement-root',type=Path,action='append',default=[])
    a=p.parse_args();r=dataset(a.jobs_root,a.output,a.workers,a.supplement_root)
    print(canonical({k:r[k]for k in('raw_job_count','usable_before_dedup','unique_rows','independent_seeds')}),flush=True)
    if len(r['rows'])>=4:print(canonical(fit_and_report(r,a.output)),flush=True)


if __name__=='__main__':main()
