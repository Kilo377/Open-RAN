"""Read-only proof of frozen singleton selection, consumption and changed/noop controls."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

LOGS={'ue_energy_rf':'energy_rf_events','ap_energy_rf':'ap_rf_events',
      'throughput_scheduler_mlp':'throughput_mlp_events','latency_scheduler_rf':'latency_rf_events',
      'latency_urllc_rf':'urllc_rf_events','mec_offload_budget_qgbdt':'budget_qgbdt_events'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def records(run,name):
    for suffix,opener in(('.jsonl.gz',gzip.open),('.jsonl',open)):
        path=Path(run)/(name+suffix)
        if path.is_file():
            with opener(path,'rt')as f:
                for line in f:
                    if line.strip():yield json.loads(line)
            return


def _subset_equal(actual,desired):
    if isinstance(desired,dict):
        return isinstance(actual,dict)and all(k in actual and _subset_equal(actual[k],v)for k,v in desired.items())
    return actual==desired


def controlled_value(config,field,desired):
    """Project actual executed state to precisely the requested control scope."""
    if field in('ue_drx_configs','ue_offload_fraction'):
        key='drx'if field=='ue_drx_configs'else'offload_fraction'
        policies=config.get('ue_service_policies',{})
        if not isinstance(desired,dict)or any(uid not in policies or key not in policies[uid]for uid in desired):return None,False
        return {uid:policies[uid][key]for uid in desired},True
    if field not in config:return None,False
    value=config[field]
    if isinstance(desired,dict):
        if not isinstance(value,dict)or any(k not in value for k in desired):return None,False
        value={k:value[k]for k in desired}
    return value,True


def receipt_evidence(row,implementation):
    owners=[p for p in row.get('provenance',[])if p.get('implementation_id')==implementation]
    if not owners:return None
    fields=[];changed=False;resolved=unresolved=0;mismatches=[]
    for cid,after in row.get('after',{}).items():
        before=row.get('before',{}).get(cid,{})
        for f in row.get('fields',[]):
            name=f['field'];desired=f.get('value')
            actual,ok=controlled_value(after,name,desired);prior,prior_ok=controlled_value(before,name,desired)
            matched=ok and _subset_equal(actual,desired)
            delta=prior_ok and ok and prior!=actual
            changed|=delta
            resolved+=bool(ok and prior_ok);unresolved+=not(ok and prior_ok)
            if not matched:mismatches.append(dict(cell_id=cid,field=name,reason='requested_value_not_in_executed_after'))
            fields.append(dict(cell_id=cid,field=name,requested=desired,before=prior,after=actual,
                consumer_resolved=bool(ok and prior_ok),consumer_matches_request=matched,changed=delta,
                route=f.get('route'),reason=f.get('reason')))
    return dict(slot=row.get('apply_slot',row.get('slot_id')),issued_at_slot=row.get('issued_at_slot'),
        request_id=row.get('control_request_id'),status=row.get('status'),owner_provenance=owners,
        recorded_ap_policy_owners={cid:a.get('ap_policy_owner')for cid,a in row.get('after',{}).items()
            if any(f['field']=='cell_sleep_mode'for f in row.get('fields',[]))},
        fields=fields,changed=changed,resolved=resolved,unresolved=unresolved,mismatches=mismatches)


def _bindings(value):
    return {a for groups in value.values()for apps in groups.values()
            for a in([apps]if isinstance(apps,str)else apps)}


def planned_selection(hot,implementation,baseline):
    timeline=hot.get('timeline',[])
    return (hot.get('initial')==[]and len(timeline)==1
        and timeline[0].get('operation')=='set'and timeline[0].get('apps')==[implementation]
        and timeline[0].get('scheduled_slot')==baseline)


def audit_case(case,plan,root):
    base=dict(job_id=case['id'],case_id=case['id'],worker_id=plan['worker_id'],stage=plan['stage'],
        arm=case['arm'],app_ids=case['app_ids'],source_sha256=plan['source_sha256'],
        source_archive_sha256=plan['source_archive_sha256'],runner_sha256=plan['runner_sha256'])
    root=Path(root);cp=root/'checkpoints'/(case['id']+'.json')
    if not case['app_ids']:
        return dict(**base,status='not_applicable',checkpoint_sha256=sha(cp)if cp.exists()else None,
                    recording_completed=cp.exists(),reason='OFF reference has no selected xApp')
    if not cp.is_file():return dict(**base,status='pending',reason='sealed_recording_required')
    seal=read(cp);job=root/'jobs'/case['id'];run=job/'data/run'
    body={k:v for k,v in plan.items()if k!='plan_sha256'}
    if (not seal.get('completed')or seal.get('case')!=case or seal.get('plan_sha256')!=plan['plan_sha256']
            or canonical(body)!=plan['plan_sha256']or seal.get('source_sha256')!=plan['source_sha256']
            or seal.get('source_archive_sha256')!=plan['source_archive_sha256']
            or seal.get('runner_sha256')!=plan['runner_sha256']or seal.get('frozen_models')!=plan['frozen_models']):
        raise ValueError('sealed_activation_identity_mismatch')
    pins=seal.get('files_sha256',{})
    if not pins:raise ValueError('sealed_raw_manifest_required')
    for relative,pin in pins.items():
        p=(job/relative).resolve()
        if not p.is_relative_to(job.resolve())or not p.is_file()or sha(p)!=pin:raise ValueError('activation_raw_sha_mismatch:'+relative)
    impl=case['app_ids'][0].removeprefix('xapp:');baseline=case['baseline_slot'];end=baseline+case['observe_slots']
    hot=read(run/'xapp_hot_start.json');models=hot.get('models',{});expected=set(plan['frozen_models'].values())
    frozen=len(models)==6 and {m.get('sha256')for m in models.values()}==expected and all(
        m.get('mode')=='frozen'and sha(run/m['checkpoint'])==m['sha256']for m in models.values())
    timeline_ok=planned_selection(hot,impl,baseline)
    own=[];nonapplied=Counter()
    for r in records(run,'control_receipts'):
        at=r.get('apply_slot',r.get('slot_id'))
        if type(at)is not int or not baseline<at<=end:continue
        proof=receipt_evidence(r,impl)
        if proof is None:continue
        if r.get('status')=='applied'and r.get('fields'):own.append(proof)
        else:nonapplied[r.get('status','missing')]+=1
    counts=Counter();first=None;last=None;unexpected=[];pre=[];trace_slots=[];trace_checks=[]
    by_slot={r['slot']:[]for r in own}
    for r in own:by_slot[r['slot']].append(r)
    for row in records(run,'rt_sys_trace'):
        slot=row['slot_id'];trace_slots.append(slot);ids=_bindings(row.get('xapp_bindings',{}))
        if slot<=baseline and ids:pre.append(slot)
        if baseline<slot<=end:
            if impl in ids:first=slot if first is None else first;last=slot;counts['binding_slots']+=1
            if ids-{impl}:unexpected.append(dict(slot=slot,apps=sorted(ids-{impl})))
        for receipt in by_slot.get(slot,[]):
            for f in receipt['fields']:
                if f['field']in('ue_drx_configs','ue_offload_fraction'):
                    key='drx'if f['field']=='ue_drx_configs'else'offload_fraction'
                    actual={uid:row.get('ue_services',{}).get(uid,{}).get('ue_service_policy',{}).get(key)for uid in f['requested']}
                elif f['field']=='cell_sleep_mode':
                    cell=row.get('ap_services',{}).get('cells',{}).get(f['cell_id'],{})
                    actual=cell.get('ap_policy')
                    if cell.get('ap_policy_owner')!=impl:trace_checks.append(dict(slot=slot,field=f['field'],reason='trace_AP_owner_mismatch'))
                else:continue # Scheduler ACK is the executor's actual before/after state, not the requested payload.
                if not _subset_equal(actual,f['requested']):trace_checks.append(dict(slot=slot,field=f['field'],reason='trace_consumer_policy_mismatch'))
    if trace_slots!=list(range(1,case['slots']+1)):raise ValueError('complete_activation_trace_required')
    decisions=[r for r in records(run,LOGS[impl])if r.get('kind')=='decision'and baseline<r.get('slot_id',-1)<=end]
    frozen_decisions=bool(decisions)and all(r.get('mode')=='frozen'for r in decisions)
    compact_keys=('slot_id','kind','mode','action','action_name','p','scheduler_mode','ue_scheduler_weight','ue_id','cell_id','status','reason','implementation_id')
    samples=[{k:r.get(k)for k in compact_keys if k in r}for r in(decisions[:2]+decisions[-2:]if len(decisions)>4 else decisions)]
    changed=[r for r in own if r['changed']];noop=[r for r in own if not r['changed']and not r['unresolved']and not r['mismatches']]
    resolved=sum(r['resolved']for r in own);unresolved=sum(r['unresolved']for r in own)
    mismatches=[m for r in own for m in r['mismatches']]
    activation=bool(frozen and timeline_ok and frozen_decisions and first is not None and baseline<first<=end and not pre and not unexpected)
    consumer=bool(own and not unresolved and not mismatches and not trace_checks)
    actual_changed=bool(changed and consumer)
    status='invalid'if not activation or mismatches or trace_checks or unresolved else'ready'
    return dict(**base,status=status,checkpoint_sha256=sha(cp),frozen_verified=frozen,
        activation_verified=activation,consumer_verified=consumer,first_binding_slot=first,last_binding_slot=last,
        first_decision_slot=min((r['slot_id']for r in decisions),default=None),
        first_applied_receipt_slot=min((r['slot']for r in own),default=None),
        first_changed_receipt_slot=min((r['slot']for r in changed),default=None),
        own_applied_receipt_count=len(own),changed_receipt_count=len(changed),noop_receipt_count=len(noop),
        consumer_fields_resolved=resolved,consumer_fields_unresolved=unresolved,consumer_mismatches=mismatches,
        trace_consumer_mismatches=trace_checks,nonapplied_owner_receipts=dict(nonapplied),
        decision_count=len(decisions),decision_samples=samples,observed_binding_slots=counts['binding_slots'],
        actual_changed=actual_changed,effect_provenance_status='changed_control'if actual_changed else'deployed_noop'if consumer else'selected_without_consumer_receipt',
        before_baseline_binding_slots=pre,unexpected_active_apps=unexpected,
        first_consumer_receipt=own[0]if own else None,first_changed_consumer_receipt=changed[0]if changed else None,
        counts_scope=dict(start_slot_exclusive=baseline,end_slot_inclusive=end,tail_excluded=True,
            persistent_repeated_receipts_are_not_independent_policy_decisions=True),
        input_refs={name:dict(path=str(job/name),sha256=pin)for name,pin in pins.items()
            if any(token in name for token in ('control_receipts','rt_sys_trace','xapp_hot_start',LOGS[impl],'config_snapshot'))})


def audit(plans,roots,output):
    entries=[]
    for path in plans:
        p=read(path);root=roots.get(p['stage']+':'+p['worker_id'],roots.get(p['worker_id']))
        if root is None:raise ValueError('worker_root_required')
        for case in p['cases']:
            try:entries.append(audit_case(case,p,root))
            except (ValueError,KeyError,OSError,TypeError)as error:
                entries.append(dict(status='invalid',job_id=case['id'],case_id=case['id'],worker_id=p['worker_id'],
                    stage=p['stage'],arm=case['arm'],error=type(error).__name__+': '+str(error)))
    result=dict(schema='oran.environment-activation-audit.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        audit_code_sha256=sha(__file__),new_simulations=0,training_performed=False,entries=entries,
        counts=dict(Counter(e['status']for e in entries)),
        interpretation='Activation/consumer authenticity is separate from KPI improvement. Applied unchanged controls are noops. All applied counts are AFTER-window scoped; repeated persistent receipts are not independent actions.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(output)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,action='append',required=True)
    p.add_argument('--worker-root',action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();roots={}
    for value in a.worker_root:
        worker,sep,path=value.partition('=')
        if not sep or worker in roots:raise ValueError('unique_worker_root_required')
        roots[worker]=path
    result=audit(a.plan,roots,a.output);print(json.dumps(dict(output=str(a.output),counts=result['counts'])))


if __name__=='__main__':main()
