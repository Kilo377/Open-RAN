"""Read-only, same-origin evaluation of explicitly isolated pressure scenarios.

Only sealed complete recordings enter results. Application effects compare ON
AFTER with OFF AFTER on the same host, seed, native prefix and observation
window. Environment contrasts and BEFORE/AFTER diagnostics remain separate.
No simulator, training, selection policy or deployed service is invoked.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from smo.runtime import records,trace_users
from smo.window_metrics import CounterColumn,InstantColumn,WindowIndex,numeric
from smo.evaluation_reference import comparable

APPS={
    'A':('ue_energy_rf','UE power','ue_power_w',-1,'W'),
    'B':('ap_energy_rf','BS power','ap_power_w',-1,'W'),
    'C':('throughput_scheduler_mlp','Downlink goodput','network_goodput_mbps',1,'Mbit/s'),
    'D':('latency_scheduler_rf','MEC task delay','mec_offloaded_latency_ms',-1,'ms'),
    'E':('latency_urllc_rf','URLLC failure rate','urllc_failure_fraction',-1,'ratio'),
    'F':('mec_offload_budget_qgbdt','Expected MEC demand','mec_expected_gflops',-1,'GFLOP/s'),
}
KEYS=tuple(dict.fromkeys([v[2]for v in APPS.values()]+[
    'urllc_packet_latency_ms','packet_latency_ms','task_latency_ms','task_success_ratio',
    'queue_bytes','hol_delay_ms','queue_overflow_bytes','mec_realized_gflops']))
SOURCE='cbe114446814007b275856e4bae92a2fe609abf4f4bcbbf8c32394a722a36af0'


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def effect(after,reference,direction,*,ratio=False):
    a,b=after.get('value'),reference.get('value')
    if not numeric(a)or not numeric(b):
        return dict(status='unavailable',change=None,target_improvement=None,improvement_percent=None,
            after_reason=after.get('reason'),reference_reason=reference.get('reason'))
    delta=a-b;improvement=direction*delta
    tolerance=max(1e-9,1e-10*max(abs(a),abs(b)))
    resolutions=[m.get('resolution')for m in (after,reference)if numeric(m.get('resolution'))and m['resolution']>0]
    resolution=max(resolutions)if resolutions else None
    # Slot duration is not the resolution of a many-completion latency mean.
    # Only delivered-bit counter metrics carry an admitted byte-quantum floor.
    goodput_counter=all(m.get('aggregation')=='counter_delta_over_equal_window'and numeric(m.get('resolution'))for m in (after,reference))
    if goodput_counter and resolution:tolerance=max(tolerance,resolution*.5)
    basis='absolute_and_relative_numeric_tolerance';sign=1 if improvement>tolerance else-1 if improvement<-tolerance else 0
    count_fields=('failure_events','offered_packets')
    has_counts=ratio and any(any(k in m for k in count_fields)for m in (after,reference))
    valid_counts=all(all(numeric(m.get(k))and m[k]>=0 and float(m[k]).is_integer()for k in count_fields)and m['offered_packets']>0 for m in (after,reference))
    if has_counts and not valid_counts:
        return dict(status='unavailable',change=delta,target_improvement=improvement,improvement_percent=None,
            reason='valid_failure_event_ratio_counts_required')
    if has_counts:
        if any(not math.isclose(m['value'],m['failure_events']/m['offered_packets'],rel_tol=1e-12,abs_tol=1e-15)for m in (after,reference)):
            return dict(status='unavailable',change=delta,target_improvement=improvement,improvement_percent=None,
                reason='metric_value_disagrees_with_event_counts')
        numerator=direction*(int(after['failure_events'])*int(reference['offered_packets'])-int(reference['failure_events'])*int(after['offered_packets']))
        sign=1 if numerator>0 else-1 if numerator<0 else 0
        tolerance=0.;basis='exact_integer_failure_event_ratio_comparison'
    return dict(status='improved'if sign>0 else'worsened'if sign<0 else'unchanged',
        change=delta,target_improvement=improvement,
        improvement_percent=100*improvement/abs(b)if b else None,
        improvement_percentage_points=100*improvement if ratio else None,
        numerical_tolerance=tolerance,classification_basis=basis,
        counter_resolution=resolution if goodput_counter else None,
        change_in_counter_quanta=delta/resolution if goodput_counter and resolution else None,
        practical_significance_claimed=False)


def latency_diagnostic(metric,profiles):
    """Keep conditional means separate from a fixed-deadline cost diagnostic."""
    deadlines={p.get('deadline_ms')for p in profiles.values()}
    deadline=next(iter(deadlines))if len(deadlines)==1 else None
    counts={k:metric.get(k)for k in ('generated','sample_count','failed','pending')}
    n,c,f,p=(counts[k]for k in ('generated','sample_count','failed','pending'))
    reason=None;penalized=None
    if not numeric(deadline)or deadline<=0:reason='one_common_positive_task_deadline_required'
    elif not all(type(x)is int and x>=0 for x in (n,c,f,p))or n!=c+f+p:reason='complete_cohort_counts_required'
    elif not n:reason='no_mature_task_samples'
    elif c and not numeric(metric.get('value')):reason='completed_latency_sum_unavailable'
    else:penalized=((metric['value']*c if c else 0)+deadline*(f+p))/n
    return dict(**counts,conditional_completed_latency_ms=metric.get('value'),
        fixed_deadline_penalty_ms=deadline,deadline_penalized_mean_ms=penalized,reason=reason,
        interpretation='Diagnostic completion latency plus one configured deadline for each failed/pending mature task; not an imputed physical completion time.')


def full_birth_followup(index,start,end,through,*,offloaded_only=False):
    """Additional diagnostic: all AFTER births followed through the sealed tail."""
    if index.task_errors:return dict(value=None,reason='incomplete_task_event_log')
    tasks=[t for t in index.tasks.values()if start<t['born_slot']<=end]
    a,b=bisect_left(index.slots,start),bisect_left(index.slots,end)
    if a>=len(index.slots)or b>=len(index.slots)or index.slots[a]!=start or index.slots[b]!=end:
        return dict(value=None,reason='exact_birth_window_unavailable')
    if index.counters['task_generated'].delta(a,b)!=len(tasks)or index.counters['mec_generated'].delta(a,b)!=sum(t['offloaded']is True for t in tasks):
        return dict(value=None,reason='incomplete_task_birth_log')
    tail=bisect_left(index.slots,through)
    if tail>=len(index.slots)or index.slots[tail]!=through:return dict(value=None,reason='exact_followup_tail_unavailable')
    for key,kinds in [('task_completed',{'task_completed'}),('task_failed',{'task_queue_overflow','task_result_overflow'})]:
        count=sum(start<t.get('terminal',{}).get('recorded_slot',-1)<=through and t['terminal']['kind']in kinds for t in index.tasks.values())
        if index.counters[key].delta(a,tail)!=count:return dict(value=None,reason='incomplete_task_terminal_log')
    if offloaded_only:tasks=[t for t in tasks if t['offloaded']is True]
    completed=[];failed=[];pending=[]
    for task in tasks:
        terminal=task.get('terminal')
        if not terminal or terminal['recorded_slot']>through:pending.append(task)
        elif terminal['kind']=='task_completed':completed.append((task,terminal))
        else:failed.append(task)
    count=len(tasks);total=sum(t['latency_ms']for _,t in completed)
    return dict(value=total/len(completed)if completed else None,reason=None if completed else'no_completed_tasks',
        generated=count,sample_count=len(completed),failed=len(failed),pending=len(pending),
        fully_resolved=not pending,on_time_completed=sum(t['latency_ms']<=b['deadline_s']*1000+1e-9 for b,t in completed),
        deadline_success_fraction=sum(t['latency_ms']<=b['deadline_s']*1000+1e-9 for b,t in completed)/count if count else None,
        deadline_penalized_mean_ms=(total+sum(t['deadline_s']*1000 for t in failed+pending))/count if count else None,
        birth_start_slot_exclusive=start,birth_end_slot_inclusive=end,terminal_followup_through_slot=through,
        exposure_ms=(end-start)*index.dt_ms,aggregation='all_interval_births_followed_through_sealed_tail',
        conditional_on_completion=True,changes_primary_guarded_target=False,
        interpretation='Separate follow-up diagnostic. Failed/pending tasks remain explicit; deadline penalties are costs, not invented completion latencies.')


class ExtraWindows:
    """Parallel exact scalar index; exogenous arrivals never include MEC results."""
    def __init__(self,config):
        self.uids=tuple(u['ue_id']for u in config['configs']['ue']['ues'])
        self.dt=config['manifest'].get('slot_duration_ms',.5)/1000
        self.profiles=config.get('configs',{}).get('phy',{}).get('rt_sys',{}).get('ue_services',{}).get('mec',{}).get('tasks_by_ue',{})
        self.slots=[]
        self.exogenous_rows=[]
        self.task_birth_rows=[]
        self.packet={k:CounterColumn()for k in ('offered_bytes','delivered_bytes','overflow_bytes')}
        self.instant={k:InstantColumn()for k in ('pending_bytes','exogenous_bytes','mec_result_bytes')}
        self.demand={uid:InstantColumn()for uid in self.profiles}
        users=config['configs']['ue']['ues']
        self.scopes={'embb':[u['ue_id']for u in users if u.get('slice_id')=='embb'],
            'non_urllc':[u['ue_id']for u in users if u.get('slice_id')!='urllc'],
            'urllc':[u['ue_id']for u in users if u.get('slice_id')=='urllc']}
        self.scope_packet={scope:{k:CounterColumn()for k in self.packet}for scope in self.scopes}
        self.scope_instant={scope:{k:InstantColumn()for k in ('pending_bytes','exogenous_bytes')}
            for scope in self.scopes}

    def observe(self,row,users,services):
        self.slots.append(row['slot_id']);by={u['ue_id']:u for u in users}
        for key,column in self.packet.items():column.append({u:by.get(u,{}).get('packet',{}).get(key)for u in self.uids})
        pending=[by.get(u,{}).get('packet',{}).get('pending_bytes')for u in self.uids]
        self.instant['pending_bytes'].append(sum(pending)if all(numeric(v)for v in pending)else None)
        arrivals=row.get('traffic_arrivals',{})
        self.exogenous_rows.append({u:arrivals.get(u,{}).get('exogenous_bytes')for u in self.uids})
        for event in row.get('ue_service_events',[]):
            if event.get('kind')!='task_generated':continue
            profile=self.profiles.get(event.get('ue_id'),{})
            birth={k:event.get(k)for k in ('ue_id','task_id','created_s','time_s','deadline_s')}
            birth.update(compute_gflop=profile.get('compute_gflop'),input_bytes=profile.get('input_bytes'),
                result_bytes=profile.get('result_bytes'),interval_ms=profile.get('interval_ms'))
            valid=all(birth.get(k)is not None for k in ('ue_id','task_id','created_s','time_s','deadline_s','compute_gflop'))
            self.task_birth_rows.append((row['slot_id'],birth,valid))
        for key in ('exogenous_bytes','mec_result_bytes'):
            values=[arrivals.get(u,{}).get(key)for u in self.uids]
            self.instant[key].append(sum(values)if all(numeric(v)and v>=0 for v in values)else None)
        for uid,column in self.demand.items():
            p=self.profiles[uid];fraction=services.get(uid,{}).get('ue_service_policy',{}).get('offload_fraction')
            column.append(1000/p['interval_ms']*p['compute_gflop']*fraction
                if numeric(p.get('interval_ms'))and p['interval_ms']>0 and numeric(p.get('compute_gflop'))
                and numeric(fraction)and 0<=fraction<=1 else None)
        for scope,uids in self.scopes.items():
            for key,column in self.scope_packet[scope].items():column.append({u:by.get(u,{}).get('packet',{}).get(key)for u in uids})
            for key,column in self.scope_instant[scope].items():
                values=[by.get(u,{}).get('packet',{}).get(key)if key=='pending_bytes'else arrivals.get(u,{}).get(key)for u in uids]
                column.append(sum(values)if values and all(numeric(v)and v>=0 for v in values)else None)

    def _scope_flow(self,scope,a,b):
        uids=self.scopes[scope]
        if not uids:return dict(status='unavailable',reason='no_configured_scope_members')
        seconds=(b-a)*self.dt;bits=lambda v:v*8/1e6/seconds if numeric(v)else None
        values={k:c.delta(a,b)for k,c in self.scope_packet[scope].items()}
        pending=self.scope_instant[scope]['pending_bytes'];q0,q1=pending.values[a],pending.values[b]
        change=q1-q0 if math.isfinite(q0)and math.isfinite(q1)else None
        stats=pending.interval(a,b);exo=self.scope_instant[scope]['exogenous_bytes'].interval(a,b)
        exo_total=exo['value']*(b-a)if exo else None
        offered,delivered,overflow=(values[k]for k in ('offered_bytes','delivered_bytes','overflow_bytes'))
        return dict(status='observed',member_ue_ids=uids,**values,
            exogenous_offered_load_mbps=bits(exo_total),all_packet_offered_load_mbps=bits(offered),
            delivered_goodput_mbps=bits(delivered),arrival_minus_delivery_bytes=offered-delivered if offered is not None and delivered is not None else None,
            pending_bytes_change=change,pending_bytes_at_start=q0 if math.isfinite(q0)else None,
            pending_bytes_at_end=q1 if math.isfinite(q1)else None,pending_bytes_time_stats=stats,
            backlogged_slot_fraction=sum(v>0 for v in pending.values[a+1:b+1])/(b-a)if stats else None,
            empty_queue_slot_fraction=sum(v==0 for v in pending.values[a+1:b+1])/(b-a)if stats else None,
            overflow_fraction=overflow/offered if overflow is not None and offered else None,
            conservation_residual_bytes=offered-delivered-overflow-change if all(numeric(v)for v in (offered,delivered,overflow,change))else None,
            interpretation='Packet-flow deficit decomposes into overflow and signed backlog growth. Existing or periodic backlog can exist with zero endpoint growth. Includes any MEC result packets in this UE service class.')

    def window(self,start,end):
        a,b=bisect_left(self.slots,start),bisect_left(self.slots,end)
        if a>=len(self.slots)or b>=len(self.slots)or self.slots[a]!=start or self.slots[b]!=end or b-a!=end-start:
            return dict(status='unavailable',reason='exact_window_data_unavailable')
        seconds=(end-start)*self.dt
        changes={k:c.delta(a,b)for k,c in self.packet.items()}
        bits=lambda value:value*8/1e6/seconds if numeric(value)else None
        pending=self.instant['pending_bytes'].values
        queue_delta=pending[b]-pending[a]if math.isfinite(pending[a])and math.isfinite(pending[b])else None
        arrivals={k:c.interval(a,b)for k,c in self.instant.items()if k!='pending_bytes'}
        exo=arrivals['exogenous_bytes'];mec=arrivals['mec_result_bytes']
        exo_bytes=exo['value']*(b-a)if exo else None;mec_bytes=mec['value']*(b-a)if mec else None
        offered,delivered,overflow=(changes[k]for k in ('offered_bytes','delivered_bytes','overflow_bytes'))
        balance=(offered-delivered-overflow-queue_delta)if all(numeric(v)for v in (offered,delivered,overflow,queue_delta))else None
        arrival_balance=offered-exo_bytes-mec_bytes if all(numeric(v)for v in (offered,exo_bytes,mec_bytes))else None
        births=[(slot,birth)for slot,birth,valid in self.task_birth_rows if start<slot<=end]
        births_valid=all(valid for slot,_,valid in self.task_birth_rows if start<slot<=end)
        scoped={scope:self._scope_flow(scope,a,b)for scope in self.scopes}
        width=max(1,round(.1/self.dt));cuts=[start];next_cut=(start//width+1)*width
        while next_cut<end:cuts.append(next_cut);next_cut+=width
        cuts.append(end);subwindows=[]
        for left,right in zip(cuts,cuts[1:]):
            x,y=bisect_left(self.slots,left),bisect_left(self.slots,right)
            subwindows.append(dict(start_slot_exclusive=left,end_slot_inclusive=right,
                exposure_ms=(right-left)*self.dt*1000,full_natural_100ms_bin=right-left==width,
                partial_edge=right-left!=width,service_class_flows={scope:self._scope_flow(scope,x,y)for scope in self.scopes}))
        return dict(status='observed',offered_packet_bytes=offered,delivered_packet_bytes=delivered,
            overflow_bytes=overflow,pending_bytes_change=queue_delta,packet_conservation_residual_bytes=balance,
            offered_input_conservation_residual_bytes=arrival_balance,
            exogenous_arrivals_sha256=digest(list(zip(self.slots[a+1:b+1],self.exogenous_rows[a+1:b+1])))if exo else None,
            task_births_sha256=digest(births)if births_valid else None,task_birth_count=len(births),
            task_birth_identity_fields=['slot','ue_id','task_id','created_s','time_s','deadline_s','compute_gflop','input_bytes','result_bytes','interval_ms'],
            task_compute_basis='Compute/input/result/interval use the frozen config task profile; source task_generated events do not serialize compute. Offloaded route and serving-cell fields are excluded as endogenous.',
            exogenous_offered_bytes=exo_bytes,mec_result_offered_bytes=mec_bytes,
            exogenous_offered_load_mbps=bits(exo_bytes),mec_result_offered_load_mbps=bits(mec_bytes),
            all_packet_offered_load_mbps=bits(offered),delivered_goodput_mbps=bits(delivered),
            arrival_minus_delivery_mbps=bits(offered-delivered)if offered is not None and delivered is not None else None,
            delivered_to_offered_ratio=delivered/offered if offered else None,
            ratio_semantics='Interval flow ratio; may exceed one when serving an earlier backlog, not a success probability.',
            service_class_flows=scoped,
            natural_100ms_subwindows=subwindows,
            subwindow_definition='Absolute virtual-time 100ms boundaries; first/last clipped edges keep their true exposure. No outcome-dependent phase or cutpoint is selected.',
            per_workload_expected_demand_gflops={u:c.interval(a,b)or dict(value=None,reason='missing_workload_policy')for u,c in self.demand.items()})


def _actions(app):
    result={}
    for a in app.get('actions',[]):
        if a.get('parameter')not in ('scheduler_mode','ue_scheduler_weights'):continue
        key=(a['parameter'],a.get('cell_id'),a.get('ue_id'))
        if key in result and result[key]!=a.get('proposed'):raise ValueError('contradictory_preview_action')
        result[key]=deepcopy(a.get('proposed'))
    return result


def preview_comparison(preview):
    """One OFF observation, two dry-run policies: never two independent paths."""
    apps=preview.get('apps',[]);apps=list(apps.values())if isinstance(apps,dict)else apps
    by={a.get('app_id'):a for a in apps}
    c=by.get('xapp:throughput_scheduler_mlp');d=by.get('xapp:latency_scheduler_rf')
    if not c or not d or any(a.get('status')!='proposed'or a.get('inference_invoked')is not True for a in (c,d)):
        return dict(slot=preview.get('slot'),status='unavailable',reason='both_real_policy_proposals_required')
    observed=c.get('observation_slot');captured=preview.get('slot')
    if type(observed)is not int or type(captured)is not int or observed>captured or observed!=d.get('observation_slot'):
        return dict(slot=preview.get('slot'),status='unavailable',reason='noncontemporaneous_observations')
    public=c.get('input_sha256')
    if not isinstance(public,str)or not public or public!=d.get('input_sha256'):
        return dict(slot=preview.get('slot'),status='unavailable',reason='same_public_observation_hash_required')
    x,y=_actions(c),_actions(d);shared=x.keys()&y.keys()
    if not shared:return dict(slot=preview.get('slot'),status='unavailable',reason='no_shared_scheduler_resources')
    changes=lambda a:{(x['parameter'],x.get('cell_id'),x.get('ue_id')):x.get('proposed')
        for x in a.get('actions',[])if x.get('parameter')in ('scheduler_mode','ue_scheduler_weights')and x.get('changed')is True}
    cx,dy=changes(c),changes(d)
    differences=[dict(parameter=k[0],cell_id=k[1],ue_id=k[2],throughput=x[k],mec_latency=y[k])for k in sorted(shared,key=str)if x[k]!=y[k]]
    return dict(slot=preview.get('slot'),observation_slot=c.get('observation_slot'),status='observed',
        public_input_sha_equal=True,public_input_sha256=public,
        shared_resource_equal=all(x[k]==y[k]for k in shared),whole_bundle_equal=x==y,
        changed_shared_equal={k:v for k,v in cx.items()if k in shared}=={k:v for k,v in dy.items()if k in shared},
        shared_fields=len(shared),shared_equal_fields=sum(x[k]==y[k]for k in shared),differences=differences,
        throughput_only_fields=len(x.keys()-y.keys()),mec_only_fields=len(y.keys()-x.keys()),
        throughput_changed_fields=len(cx),mec_changed_fields=len(dy),
        throughput_maintained_fields=len(x)-len(cx),mec_maintained_fields=len(y)-len(dy))


def summarize_previews(run,baseline,end):
    rows=[preview_comparison(p)for p in records(run,'xapp_decision_previews')
        if type(p.get('slot'))is int and baseline<p['slot']<=end]
    valid=[r for r in rows if r['status']=='observed'];n=len(valid)
    return dict(scope='Contemporaneous C/D shadow decisions from the same OFF run and observation; AFTER interval only.',
        samples=n,unavailable_samples=len(rows)-n,
        shared_resource_equal=sum(r['shared_resource_equal']for r in valid),
        shared_resource_equal_fraction=sum(r['shared_resource_equal']for r in valid)/n if n else None,
        whole_bundle_equal=sum(r['whole_bundle_equal']for r in valid),
        changed_shared_equal=sum(r['changed_shared_equal']for r in valid),
        samples_by_slot=rows)


def _case_root(roots,worker,stage):
    value=roots.get(stage+':'+worker,roots.get(worker))
    if value is None:raise ValueError('worker_root_required:'+stage+':'+worker)
    base=Path(value)
    choices=[p for p in (base,base/stage,base/'outputs'/stage)if(p/'checkpoints').is_dir()]
    if len(choices)!=1:raise ValueError('unique_stage_checkpoint_root_required:'+str(base))
    return choices[0]


def inspect_case(case,plan,plan_path,roots,*,require_receiver_seals=False):
    stage=case.get('stage',plan.get('stage'));worker=case.get('worker_id',plan.get('worker_id'))
    if stage not in ('pilot','validation')or not isinstance(worker,str):raise ValueError('stage_and_worker_required')
    root=_case_root(roots,worker,stage);ident=case['id']
    if not ident or ident in ('.','..')or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-.'for c in ident):raise ValueError('unsafe_case_id')
    cp=root/'checkpoints'/(ident+'.json')
    if not cp.is_file():return dict(status='pending',case=case,stage=stage,worker_id=worker,reason='checkpoint_pending')
    # A checkpoint may be mirrored before its recording files finish copying.
    # Bind one exact checkpoint read to the receiver's completed publication proof.
    # This additional gate never replaces the all-file checks below.
    checkpoint_bytes=cp.read_bytes();checkpoint_sha=hashlib.sha256(checkpoint_bytes).hexdigest()
    receiver_admission=None
    if require_receiver_seals:
        receiver_path=root/'received-checkpoints.json'
        receiver_bytes=receiver_path.read_bytes()if receiver_path.is_file()else None
        received=json.loads(receiver_bytes)if receiver_bytes is not None else{}
        receiver=received.get(ident,{})
        if receiver.get('seal_sha256')!=checkpoint_sha:
            return dict(status='pending',case=case,stage=stage,worker_id=worker,reason='receiver_verified_checkpoint_pending')
        receiver_admission=dict(required=True,path=str(receiver_path),manifest_sha256=hashlib.sha256(receiver_bytes).hexdigest(),
            checkpoint_sha256=checkpoint_sha,verification=receiver)
    seal=json.loads(checkpoint_bytes)
    if seal.get('completed')is not True:return dict(status='pending',case=case,stage=stage,worker_id=worker,reason='completed_checkpoint_required')
    if seal.get('case')!=case or seal.get('worker_id')!=worker or seal.get('stage')!=stage:raise ValueError('checkpoint_case_identity_mismatch')
    plan_body=deepcopy(plan);plan_pin=plan_body.pop('plan_sha256',None)
    if not plan_pin or digest(plan_body)!=plan_pin or seal.get('plan_sha256')!=plan_pin:
        raise ValueError('checkpoint_plan_sha256_mismatch')
    if len(plan.get('frozen_models',{}))!=6 or seal.get('frozen_models')!=plan['frozen_models']:
        raise ValueError('six_frozen_model_pins_required')
    if seal.get('source_archive_sha256')!=plan.get('source_archive_sha256')or seal.get('runner_sha256')!=plan.get('runner_sha256'):
        raise ValueError('checkpoint_archive_runner_identity_mismatch')
    if seal.get('models_unchanged')is not True or seal.get('training_performed')is not False:
        raise ValueError('frozen_untrained_validation_required')
    if seal.get('post_deployment_kpis_used_for_model_training')is not False:
        raise ValueError('deployment_feedback_training_not_permitted')
    if not isinstance(seal.get('actual_system_hostname'),str)or not seal['actual_system_hostname']:
        raise ValueError('physical_hostname_evidence_required')
    run=root/'jobs'/ident/'data/run';job=run.parents[1];pins=seal.get('files_sha256',{})
    required={'data/run/config_snapshot.json','data/run/run_status.json'}
    trace={p for p in pins if p in ('data/run/rt_sys_trace.jsonl','data/run/rt_sys_trace.jsonl.gz')}
    if not required<=pins.keys()or len(trace)!=1:raise ValueError('sealed_trace_config_status_required')
    verified={}
    # Verify every published pin, rather than silently trusting a completed flag.
    for relative,pin in pins.items():
        path=(job/relative).resolve()
        if not path.is_relative_to(job.resolve())or not path.is_file()or file_sha(path)!=pin:raise ValueError('sealed_recording_changed:'+relative)
        verified[relative]=pin
    config=read(run/'config_snapshot.json');status=read(run/'run_status.json')
    expected=plan.get('source_sha256',SOURCE)
    if expected!=SOURCE:raise ValueError('frozen_environment_source_required')
    if status.get('completed')is not True or seal.get('source_sha256')!=expected or config.get('source_sha256')!=expected:
        raise ValueError('completed_frozen_source_required')
    if config.get('manifest',{}).get('rng_seed')!=case['seed']:raise ValueError('recorded_seed_mismatch')
    baseline=case['baseline_slot'];observe=case['observe_slots'];before=case['origin_observe_slots'];end=baseline+observe
    if config['manifest']['total_slots']!=case['slots']or end>case['slots']:raise ValueError('recorded_timing_mismatch')
    arm=case['arm'];apps=case.get('app_ids',[])
    if arm=='OFF'and apps or arm in APPS and apps!=['xapp:'+APPS[arm][0]]:raise ValueError('single_app_arm_mismatch')
    guard=None
    if apps:
        guard=read(run/'evaluation_origin_verified.json')
        if guard.get('origin_id')!=seal.get('origin_id')or not all(guard.get(k)is True for k in (
            'verified_before_activation','observed_prefix_equal','configuration_equal','models_equal','network_dynamics_code_equal')):
            raise ValueError('same_native_prefix_guard_required')
    index=WindowIndex(config,population=True);extra=ExtraWindows(config);through=0
    for row in records(run,'rt_sys_trace'):
        slot=row['slot_id']
        if slot>case['slots']:break
        users=trace_users(row);services=row.get('ue_services',{});ap=row.get('ap_services',{}).get('cells',{})
        index.observe(slot,users,services,ap,row.get('ue_service_events',[]));extra.observe(row,users,services);through=slot
    if through!=case['slots']:raise ValueError('complete_sealed_trace_through_tail_required')
    windows={};flows={}
    quota_apps=[a for a in config.get('configs',{}).get('ric',{}).get('xapps',[])
        if a.get('implementation_id')=='mec_offload_budget_qgbdt']
    quota=quota_apps[0].get('parameters',{}).get('budget_gflops')if len(quota_apps)==1 else None
    for label,start,stop in (('before',baseline-before,baseline),('after',baseline,end)):
        windows[label]=index.window(KEYS,start,stop,through=through)
        if any(s.get('reason')in ('invalid_or_future_window','exact_window_data_unavailable')for s in windows[label]['states'].values()):
            raise ValueError('complete_exact_windows_required')
        flows[label]=extra.window(start,stop)
        d=flows[label].get('per_workload_expected_demand_gflops',{}).get('ue_4',{})
        flows[label]['ue4_budget']=dict(target_ue_id='ue_4',budget_gflops=quota,expected_demand_gflops=d,
            mean_compliant=d.get('value')<=quota if numeric(quota)and numeric(d.get('value'))else None,
            maximum_compliant=d.get('maximum')<=quota if numeric(quota)and numeric(d.get('maximum'))else None,
            scope='Released F contract: UE4 expected-work quota, separate from population total demand.')
    if file_sha(cp)!=checkpoint_sha:raise ValueError('checkpoint_changed_during_analysis')
    return dict(status='complete',case=case,stage=stage,worker_id=worker,origin_id=seal.get('origin_id'),
        actual_system_hostname=seal.get('actual_system_hostname'),frozen_models=seal['frozen_models'],
        runner_sha256=seal['runner_sha256'],source_archive_sha256=seal['source_archive_sha256'],
        run_path=str(run),checkpoint=dict(path=str(cp),sha256=checkpoint_sha),receiver_admission=receiver_admission,files_sha256=verified,
        source_sha256=expected,input_model_sha256=config.get('input_model_sha256'),
        config_sha256=file_sha(run/'config_snapshot.json'),windows=windows,flows=flows,
        full_after_birth_followup=dict(all_tasks=full_birth_followup(index,baseline,end,through),
            offloaded_tasks=full_birth_followup(index,baseline,end,through,offloaded_only=True)),
        origin_guard=guard,proposal_overlap=summarize_previews(run,baseline,end)if not apps else None,
        network_config=dict(slot_duration_ms=config['manifest'].get('slot_duration_ms',.5),
            carrier_frequency_hz=config['manifest'].get('carrier_frequency_hz'),bandwidth_hz=config['manifest'].get('bandwidth_hz'),
            cells=len(config['manifest'].get('cells',[])),ues=len(config['manifest'].get('ues',[])),
            urllc_scenario=config['manifest'].get('urllc_scenario'),
            task_profiles=config['configs']['phy']['rt_sys'].get('ue_services',{}).get('mec',{}).get('tasks_by_ue',{})))


def pair_results(cases):
    grouped=defaultdict(list)
    for r in cases:
        if r['status']=='complete':grouped[(r['stage'],r['case']['origin_key'],r['worker_id'])].append(r)
    paired=[];baselines=[]
    for key,members in sorted(grouped.items()):
        off=[m for m in members if not m['case'].get('app_ids')]
        if len(off)!=1:
            if not off:continue
            raise ValueError('one_unique_same_host_off_required')
        off=off[0];baselines.append(off)
        for row in members:
            if row is off:continue
            if row['origin_id']!=off['origin_id']or not row['origin_id']:raise ValueError('different_origin_cannot_pair')
            if row['case']['seed']!=off['case']['seed']or row['input_model_sha256']!=off['input_model_sha256']or row.get('frozen_models')!=off.get('frozen_models'):
                raise ValueError('same_seed_and_models_required')
            if row.get('actual_system_hostname')!=off.get('actual_system_hostname'):raise ValueError('different_physical_host_cannot_pair')
            if not comparable(row['windows']['before'],off['windows']['before']):raise ValueError('before_windows_differ')
            if any(row['case'][k]!=off['case'][k]for k in ('baseline_slot','observe_slots','origin_observe_slots')):
                raise ValueError('paired_window_timing_differs')
            for interval in ('before','after'):
                for field,reason in [('exogenous_arrivals_sha256','same_origin_exogenous_arrivals_differ'),
                    ('task_births_sha256','same_origin_exogenous_task_births_differ')]:
                    r_hash=row['flows'][interval].get(field);o_hash=off['flows'][interval].get(field)
                    if not r_hash or not o_hash:raise ValueError('complete_external_input_evidence_required:'+field)
                    if r_hash!=o_hash:raise ValueError(reason)
            arm=row['case']['arm'];target=APPS[arm][2]
            a=row['windows']['after']['states'];b=off['windows']['after']['states']
            all_effects={k:effect(a['kpi:'+k],b['kpi:'+k],direction,ratio=k=='urllc_failure_fraction')for _,_,k,direction,_ in APPS.values()}
            cost_specs={'urllc_packet_latency_ms':-1,'packet_latency_ms':-1,'task_latency_ms':-1,
                'task_success_ratio':1,'queue_overflow_bytes':-1}
            cost_effects={k:effect(a['kpi:'+k],b['kpi:'+k],direction,ratio=k=='task_success_ratio')
                for k,direction in cost_specs.items()if'kpi:'+k in a and'kpi:'+k in b}
            profiles=row.get('network_config',{}).get('task_profiles',{})
            mec_on=latency_diagnostic(a['kpi:mec_offloaded_latency_ms'],profiles)
            mec_off=latency_diagnostic(b['kpi:mec_offloaded_latency_ms'],profiles)
            concern=all_effects['mec_offloaded_latency_ms']['status']=='improved'and any(
                numeric(mec_on.get(k))and numeric(mec_off.get(k))and direction*(mec_on[k]-mec_off[k])>0
                for k,direction in [('sample_count',-1),('failed',1),('pending',1)])
            paired.append(dict(stage=row['stage'],scene=row['case']['scene'],seed=row['case']['seed'],worker_id=row['worker_id'],
                origin_key=row['case']['origin_key'],arm=arm,app_id=APPS[arm][0],target_kpi=target,
                job_id=row['case']['id'],off_job_id=off['case']['id'],target=all_effects[target],effects=all_effects,
                control_provenance=row.get('control_provenance',dict(status='unverified',reason='independent_activation_audit_not_provided')),
                post_first_apply_ue4_quota=row.get('post_first_apply_ue4_quota'),
                additional_cost_effects=cost_effects,
                before=row['windows']['before'],after=row['windows']['after'],off_after=off['windows']['after'],
                before_comparison=dict(exact_equal=row['windows']['before']==off['windows']['before'],
                    comparable_under_origin_protocol=True,relative_tolerance=1e-7,absolute_tolerance=1e-8),
                external_input_comparison=dict(before_exogenous_arrivals_equal=True,after_exogenous_arrivals_equal=True,
                    before_task_births_equal=True,after_task_births_equal=True,
                    equality='Exact slot/UE byte and birth-identity hashes; controlled MEC-result arrivals and offload routes are excluded.'),
                mec_completion_diagnostic=dict(after=mec_on,off_after=mec_off,selection_effect_concern=concern,
                    offloaded_cohort_count_changed=mec_on['generated']!=mec_off['generated'],
                    offloaded_task_membership_comparison='Not established by equal counts; route/completion composition may differ. Endogenous cohort membership is not an external-input equality gate.',
                    full_after_birth_followup=row.get('full_after_birth_followup'),
                    off_full_after_birth_followup=off.get('full_after_birth_followup')),
                before_after_diagnostic=effect(a['kpi:'+target],row['windows']['before']['states']['kpi:'+target],APPS[arm][3],ratio=arm=='E'),
                flows=row['flows'],off_flows=off['flows']))
    return paired,baselines


def bind_environment_exclusions(cases,path,plan_values,worker_roots):
    """Explicit authorized pre-admission rejection is not failed telemetry."""
    if path is None:return None
    manifest=read(path)
    if manifest.get('schema')!='oran.environment-excluded-cases.v1':raise ValueError('environment_exclusion_schema_mismatch')
    receipt_ref=manifest.get('rejection_receipt',{});receipt=read(receipt_ref['path'])
    if file_sha(receipt_ref['path'])!=receipt_ref.get('sha256')or receipt.get('schema')!='oran.environment-off-causal-rejection.v1':
        raise ValueError('environment_rejection_receipt_changed')
    if receipt.get('source_sha256')!=SOURCE or receipt.get('models_or_actions_modified')is not False:
        raise ValueError('rejection_cannot_change_physics_or_models')
    multi_plan='plan_files'in manifest
    pinned_plans={}
    if multi_plan:
        # A rejection may stop unstarted arms on both hosts. Preserve each
        # original plan identity rather than pretending there was one plan.
        refs=manifest['plan_files']
        workers={p.get('worker_id')for p in plan_values}
        if (not isinstance(refs,dict)or len(workers)!=len(plan_values)
            or set(refs)!=workers or 'plan_file'in manifest or 'plan_sha256'in manifest):
            raise ValueError('excluded_case_plan_identity_mismatch')
        for worker,plan_ref in refs.items():
            if file_sha(plan_ref['path'])!=plan_ref.get('sha256'):raise ValueError('excluded_original_plan_changed')
            plan=read(plan_ref['path']);body=deepcopy(plan);pin=body.pop('plan_sha256',None)
            if not pin or digest(body)!=pin:raise ValueError('excluded_original_plan_digest_mismatch')
            if (plan not in plan_values or plan.get('worker_id')!=worker
                or plan_ref.get('plan_sha256')!=pin):
                raise ValueError('excluded_case_plan_identity_mismatch')
            pinned_plans[worker]=plan
        pins={worker:p['plan_sha256']for worker,p in pinned_plans.items()}
        if receipt.get('plan_sha256_by_worker')!=pins:
            raise ValueError('excluded_case_plan_identity_mismatch')
    else:
        plan_ref=manifest.get('plan_file',{})
        if file_sha(plan_ref['path'])!=plan_ref.get('sha256'):raise ValueError('excluded_original_plan_changed')
        plan=read(plan_ref['path'])
        body=deepcopy(plan);pin=body.pop('plan_sha256',None)
        if not pin or digest(body)!=pin:raise ValueError('excluded_original_plan_digest_mismatch')
        if plan not in plan_values or plan.get('plan_sha256')!=manifest.get('plan_sha256')or receipt.get('plan_sha256')!=manifest.get('plan_sha256'):
            raise ValueError('excluded_case_plan_identity_mismatch')
    by={r['case']['id']:r for r in cases};entries=manifest.get('excluded_cases',[]);seen=set()
    for entry in entries:
        ident=entry.get('case_id');row=by.get(ident)
        if not row or ident in seen or entry.get('job_id')!=ident:raise ValueError('unique_planned_excluded_case_required')
        seen.add(ident);case=row['case']
        if any(entry.get(k)!=case.get(k)for k in ('arm','scene','seed','worker_id','origin_key'))or entry.get('rejection_receipt_sha256')!=receipt_ref['sha256']:
            raise ValueError('excluded_case_identity_mismatch')
        if multi_plan:
            plan=pinned_plans.get(row['worker_id'])
            if (not plan or entry.get('plan_sha256')!=plan['plan_sha256']
                or case not in plan['cases']):
                raise ValueError('excluded_case_plan_identity_mismatch')
        if entry.get('status')!='not_run_environment_rejected'or row['status']=='complete':raise ValueError('cannot_hide_a_completed_or_unapproved_case')
        root=_case_root(worker_roots,row['worker_id'],row['stage']);job=root/'jobs'/ident
        if(root/'checkpoints'/(ident+'.json')).exists()or job.exists()and any(p.is_file()for p in job.rglob('*')):
            raise ValueError('excluded_case_has_actual_run_evidence')
        row.update(status='not_run_environment_rejected',reason=receipt['reason'],exclusion_evidence=entry)
    if manifest.get('planned')!=len(cases)or manifest.get('actual_target')!=len(cases)-len(seen)or manifest.get('excluded_count')!=len(seen):
        raise ValueError('environment_exclusion_counts_mismatch')
    return dict(path=str(Path(path).resolve()),sha256=file_sha(path),rejection_receipt=receipt_ref,
        planned=len(cases),actual_target=len(cases)-len(seen),excluded_count=len(seen),reason=receipt['reason'])


def bind_activation_audit(cases,path):
    if path is None or not Path(path).is_file():
        for case in cases:
            if case['status']=='not_run_environment_rejected':
                case['control_provenance']=dict(status='not_run_environment_rejected',reason='explicit_pre_admission_candidate_rejection',qualified=False);continue
            case['control_provenance']=dict(status='not_applicable'if not case['case'].get('app_ids')else'unverified',
                reason='independent_activation_audit_pending',qualified=False)
        return None
    audit=read(path)
    if audit.get('schema')!='oran.environment-activation-audit.v1':raise ValueError('activation_audit_schema_mismatch')
    entries=audit.get('entries',[])
    if not isinstance(entries,list):raise ValueError('activation_audit_entries_required')
    by={e.get('job_id',e.get('case_id')):e for e in entries}
    if len(by)!=len(entries):raise ValueError('duplicate_activation_audit_job')
    for row in cases:
        if row['status']=='not_run_environment_rejected':
            row['control_provenance']=dict(status='not_run_environment_rejected',reason='explicit_pre_admission_candidate_rejection',qualified=False);continue
        if not row['case'].get('app_ids'):
            row['control_provenance']=dict(status='not_applicable',qualified=False);continue
        entry=by.get(row['case']['id'])
        if not entry or row['status']!='complete':
            row['control_provenance']=dict(status='unverified',reason='completed_activation_evidence_pending',qualified=False);continue
        if entry.get('status')!='ready':
            row['control_provenance']=dict(status=entry.get('status','unverified'),qualified=False,evidence=entry);continue
        if any(entry.get(key)!=value for key,value in (
            ('checkpoint_sha256',row['checkpoint']['sha256']),('source_sha256',row['source_sha256']),
            ('source_archive_sha256',row['source_archive_sha256']),
            ('runner_sha256',row['runner_sha256']),('worker_id',row['worker_id']))):
            raise ValueError('activation_audit_recording_identity_mismatch')
        unresolved=entry.get('consumer_fields_unresolved');mismatches=entry.get('consumer_mismatches')
        clear=lambda value:value==0 or isinstance(value,(list,dict))and not value
        qualified=(entry.get('activation_verified')is True and entry.get('frozen_verified')is True
            and entry.get('consumer_verified')is True and entry.get('actual_changed')is True
            and clear(unresolved)and clear(mismatches)and clear(entry.get('trace_consumer_mismatches')))
        if row['case']['arm']=='F':row['post_first_apply_ue4_quota']=post_first_apply_quota(row,entry)
        row['control_provenance']=dict(status=entry.get('effect_provenance_status','unverified'),qualified=qualified,evidence=entry)
    return dict(path=str(Path(path).resolve()),sha256=file_sha(path),schema=audit['schema'])


def post_first_apply_quota(row,entry):
    """Start quota review at the first own ACK, even if noop or infeasible."""
    first=entry.get('first_applied_receipt_slot');binding=entry.get('first_binding_slot')
    base=row['case']['baseline_slot'];end=base+row['case']['observe_slots']
    quota=row['flows']['after'].get('ue4_budget',{}).get('budget_gflops')
    result=dict(start_basis='first_actual_own_applied_consumer_action_even_if_noop',first_applied_slot=first,
        first_binding_slot=binding,after_end_slot=end,budget_gflops=quota,
        target_ue_id='ue_4',primary_whole_after_metric_changed=False)
    if type(first)is not int or not base<first<=end:
        return dict(result,status='unavailable',reason='first_own_applied_action_required')
    profile=row.get('network_config',{}).get('task_profiles',{}).get('ue_4',{})
    if not numeric(quota)or quota<0 or not numeric(profile.get('interval_ms'))or profile['interval_ms']<=0 or not numeric(profile.get('compute_gflop')):
        return dict(result,status='unavailable',reason='frozen_ue4_profile_and_quota_required')
    dt=row['network_config']['slot_duration_ms'];values=[];expected=first
    for trace in records(row['run_path'],'rt_sys_trace'):
        slot=trace['slot_id']
        if slot<first:continue
        if slot>end:break
        if slot!=expected:return dict(result,status='unavailable',reason='exact_post_apply_window_required')
        expected+=1
        p=trace.get('ue_services',{}).get('ue_4',{}).get('ue_service_policy',{}).get('offload_fraction')
        if not numeric(p)or not 0<=p<=1:return dict(result,status='unavailable',reason='actual_ue4_policy_missing')
        values.append(1000/profile['interval_ms']*profile['compute_gflop']*p)
    if len(values)!=end-first+1:return dict(result,status='unavailable',reason='exact_post_apply_window_required')
    mean=sum(values)/len(values);maximum=max(values);minimum=min(values)
    return dict(result,status='observed',reason=None,mean_gflops=mean,minimum_gflops=minimum,maximum_gflops=maximum,
        mean_compliant=mean<=quota,maximum_compliant=maximum<=quota,
        activation_to_first_apply_slots=first-binding if type(binding)is int else None,
        activation_to_first_apply_ms=(first-binding)*dt if type(binding)is int else None,
        post_apply_exposure_slots=len(values),post_apply_exposure_ms=len(values)*dt,
        interpretation='Separate executed-control quota diagnostic, including the first applied slot. The whole AFTER mean/max and startup transient remain recorded.')


def environment_contrasts(baselines):
    """Explicit native reference versus selected OFF, never an app comparator."""
    by=defaultdict(list)
    for b in baselines:by[(b['stage'],b['case']['seed'],b['worker_id'])].append(b)
    result=[]
    def native(case):
        return case.get('scene')in ('native_reference','native_reference_1x')or any(
            case.get(k)=='native_reference'for k in ('role','environment_role','reference_kind'))
    for (stage,seed,worker),members in sorted(by.items()):
        refs=[r for r in members if native(r['case'])];selected=[r for r in members if not native(r['case'])]
        if not refs:continue
        if len(refs)!=1:raise ValueError('one_native_reference_per_host_seed_required')
        ref=refs[0]
        for row in selected:
            if row['source_sha256']!=ref['source_sha256']or row['input_model_sha256']!=ref['input_model_sha256']:
                raise ValueError('environment_reference_source_models_differ')
            if any(row['case'][k]!=ref['case'][k]for k in ('baseline_slot','observe_slots','origin_observe_slots')):
                raise ValueError('environment_reference_timing_differs')
            a=row['windows']['after']['states'];b=ref['windows']['after']['states']
            result.append(dict(stage=stage,seed=seed,worker_id=worker,selected_scene=row['case']['scene'],
                selected_off_job_id=row['case']['id'],native_off_job_id=ref['case']['id'],
                comparison='selected_OFF_after_vs_native_reference_OFF_after',
                interpretation='Joint environment contrast (traffic and initial service policy); not an xApp effect or a single-factor attribution.',
                metrics={k:dict(native=b['kpi:'+k],selected=a['kpi:'+k],
                    target_direction_change=effect(a['kpi:'+k],b['kpi:'+k],direction,ratio=k=='urllc_failure_fraction'))
                    for _,_,k,direction,_ in APPS.values()},
                native_flows=ref['flows']['after'],selected_flows=row['flows']['after']))
    return result


def _value(state,unit=''):
    value=state.get('value')
    if not numeric(value):return 'Unavailable'
    if unit=='ratio':return f'{value*100:.2f}%'
    return f'{value:.3f}'+(' '+unit if unit else'')


def make_plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7));colors={'improved':'#24826c','worsened':'#be6152','unchanged':'#7d8e95'}
    pairs=report['paired_results']
    if pairs:
        for ax,(arm,(_,name,key,_,unit))in zip(axes.flat,APPS.items()):
            rows=[r for r in pairs if r['arm']==arm];x=range(len(rows))
            if not rows:ax.text(.5,.5,'Pending',transform=ax.transAxes,ha='center',va='center',color='#637b74')
            off=[r['off_after']['states']['kpi:'+key].get('value')for r in rows]
            on=[r['after']['states']['kpi:'+key].get('value')for r in rows];scale=100 if unit=='ratio'else 1
            for n,(o,v,r)in enumerate(zip(off,on,rows)):
                if numeric(o):ax.bar(n-.17,o*scale,.32,color='#d4dcdf',edgecolor='#647780',linewidth=.5)
                if numeric(v):ax.bar(n+.17,v*scale,.32,color=colors.get(r['target']['status'],'#7d8e95'),edgecolor='#365b54',linewidth=.5)
                else:ax.text(n,.5,'Unavailable',transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=8)
            ax.set_xticks(list(x),[r['scene']+'\n'+str(r['seed'])for r in rows],fontsize=7)
            ax.set_xlim(-.7,max(.7,len(rows)-.3));ax.set_ylim(bottom=0)
            ax.set_title(name);ax.set_ylabel('%'if unit=='ratio'else unit);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        label='discovery'if report.get('analysis_role')=='discovery'else'validation'
        fig.suptitle(f'Trained xApp {label} · paired OFF and ON',fontsize=14)
        note=('Each recipe is single-seed discovery, not a two-seed validation.'if report.get('analysis_role')=='discovery'else'No confidence interval is inferred from two seeds.')
        fig.text(.5,.012,'Grey: same-origin OFF AFTER · Color: single xApp AFTER · '+note,ha='center',fontsize=9)
    else:
        base=report['baselines']
        for ax,(_,name,key,_,unit)in zip(axes.flat,APPS.values()):
            for n,b in enumerate(base):
                v=b['windows']['after']['states']['kpi:'+key].get('value')
                if numeric(v):ax.bar(n,v*(100 if unit=='ratio'else 1),color='#4a847b')
                else:ax.text(n,.5,'Unavailable',transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=8)
            ax.set_xticks(range(len(base)),[b['case']['scene']+'\n'+str(b['case']['seed'])for b in base],fontsize=7)
            ax.set_xlim(-.7,max(.7,len(base)-.3));ax.set_ylim(bottom=0)
            ax.set_title(name);ax.set_ylabel('%'if unit=='ratio'else unit);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        fig.suptitle('OFF network pressure',fontsize=14)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(Path(output)/'environment-validation.png',dpi=180);plt.close(fig)


def make_html(report,output):
    esc=lambda s:html.escape(str(s));parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Environment validation</title><style>body{font:15px system-ui;color:#173b35;background:#f4f7f6;margin:0;padding:32px}main{max-width:1260px;margin:auto}h1{font-size:28px}h2{font-size:19px;margin-top:28px}p{color:#526861;line-height:1.5}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce5e1;font-size:13px}th{background:#e8efec}.improved{color:#16705d}.worsened{color:#ae4b3e}.unavailable{color:#8b6a33}img{width:100%;margin-top:20px;background:white;border-radius:8px}small{color:#60766f}</style><main><h1>Environment validation</h1>',
        '<p>Each single xApp is compared with the same-host OFF run, using the same seed, native prefix and AFTER window. Native-reference contrasts describe the environment change separately.</p>',
        f'<p>{len(report["complete_cases"])} completed recordings · {len(report["pending_cases"])} pending · {len(report.get("intentionally_not_run_cases",[]))} intentionally not run · No retraining</p>',
        '<h2>OFF network pressure</h2><table><tr><th>Scene / seed / host</th><th>Exogenous offered / goodput</th><th>eMBB queue growth / mean queue / overflow</th><th>URLLC failure</th><th>MEC delay / completed</th><th>Total queue / overflow</th><th>Expected MEC mean / max</th></tr>']
    if report.get('analysis_role')=='discovery':
        parts[0]=parts[0].replace('Environment validation','Environment discovery')
        parts.insert(1,'<p>Each recipe is a separate single-seed discovery case. Different recipes are not pooled as two-seed validation of one environment; a selected recipe still requires fresh independent validation.</p>')
    if report.get('environment_exclusions'):
        parts.insert(2,'<p>Intentional OFF-only candidate rejection: '+esc(report['environment_exclusions']['reason'])+' Unstarted excluded cases are neither failed simulations nor successful xApp validations.</p>')
    for b in report['baselines']:
        s=b['windows']['after']['states'];f=b['flows']['after'];d=s['kpi:mec_offloaded_latency_ms'];ex=s['kpi:mec_expected_gflops']
        n=b['case'];load=f.get('exogenous_offered_load_mbps');load=f'{load:.3f}'if numeric(load)else'Unavailable'
        bulk=f.get('service_class_flows',{}).get('embb',{});mean_queue=(bulk.get('pending_bytes_time_stats')or{}).get('value')
        bulk_text=' / '.join(f'{v/1000:.1f} kB'if numeric(v)else'Unavailable'for v in (bulk.get('pending_bytes_change'),mean_queue,bulk.get('overflow_bytes')))
        if numeric(bulk.get('backlogged_slot_fraction')):bulk_text+=f'; backlog present {100*bulk["backlogged_slot_fraction"]:.1f}% of slots'
        parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>'for v in (
            f'{n["scene"]} / {n["seed"]} / {b["worker_id"]}',f'{load} / {_value(s["kpi:network_goodput_mbps"])} Mbit/s',
            bulk_text,_value(s['kpi:urllc_failure_fraction'],'ratio'),f'{_value(d,"ms")} / {d.get("sample_count","—")}',
            f'{_value(s["kpi:queue_bytes"])} / {_value(s["kpi:queue_overflow_bytes"])} bytes',
            f'{_value(ex)} / {ex.get("maximum","Unavailable")} GFLOP/s'))+'</tr>')
    parts.append('</table>')
    if report['environment_contrasts']:
        parts.append('<h2>Environment change: native OFF → selected OFF</h2><p>Same seed and host; traffic and initial service policy change together. This is separate from xApp effectiveness.</p><table><tr><th>Seed / selected scene</th><th>Goodput</th><th>URLLC failure</th><th>MEC delay</th></tr>')
        for row in report['environment_contrasts']:
            cells=[str(row['seed'])+' / '+row['selected_scene']]
            for key,unit in [('network_goodput_mbps','Mbit/s'),('urllc_failure_fraction','ratio'),('mec_offloaded_latency_ms','ms')]:
                item=row['metrics'][key];cells.append(_value(item['native'],unit)+' → '+_value(item['selected'],unit))
            parts.append('<tr>'+''.join('<td>'+esc(c)+'</td>'for c in cells)+'</tr>')
        parts.append('</table>')
    if report['paired_results']:
        parts.append('<h2>Single xApp target and trade-offs</h2><table><tr><th>xApp / seed / scene</th><th>Target: OFF → ON</th><th>Target improvement</th><th>Other target-KPI regressions</th></tr>')
        for r in report['paired_results']:
            arm=r['arm'];_,name,key,_,unit=APPS[arm];e=r['target'];p=e.get('improvement_percent')
            text=f'{p:+.4g}%'if numeric(p)else f'{e["improvement_percentage_points"]:+.4g} percentage points'if numeric(e.get('improvement_percentage_points'))else'Unavailable'
            if e['status']=='unchanged':text='Unchanged'
            losses=[]
            for _,label,k,_,u in APPS.values():
                if k!=key and r['effects'][k]['status']=='worsened':
                    q=r['effects'][k];z=q.get('improvement_percent');losses.append(label+': '+(f'{z:.2f}%'if numeric(z)else f'{q.get("improvement_percentage_points",0):.2f} pp'))
                elif k!=key and r['effects'][k]['status']=='unavailable':losses.append(label+': unavailable')
            for k,label,cost_unit in [('urllc_packet_latency_ms','URLLC packet delay','ms'),
                ('task_success_ratio','On-time task completion','pp'),('queue_overflow_bytes','Queue overflow','bytes')]:
                q=r['additional_cost_effects'].get(k,{})
                if q.get('status')=='worsened':
                    change=q['change']*(100 if cost_unit=='pp'else 1);losses.append(f'{label}: {change:+.3f} {cost_unit}')
            if r['mec_completion_diagnostic']['selection_effect_concern']:losses.append('MEC selection-effect concern: faster conditional mean with fewer completions or more failed/pending tasks')
            values=_value(r['off_after']['states']['kpi:'+key],unit)+' → '+_value(r['after']['states']['kpi:'+key],unit)
            if arm in ('D','E'):
                old=r['off_after']['states']['kpi:'+key];new=r['after']['states']['kpi:'+key]
                values+='; '+('completed'if arm=='D'else'offered')+f' n={old.get("sample_count","—")} → {new.get("sample_count","—")}'
            if arm=='F':
                old_max=r['off_after']['states']['kpi:'+key].get('maximum');new_max=r['after']['states']['kpi:'+key].get('maximum')
                values+='; population max '+(f'{old_max:.3f}'if numeric(old_max)else'Unavailable')+' → '+(f'{new_max:.3f}'if numeric(new_max)else'Unavailable')+' GFLOP/s'
                quota=r.get('post_first_apply_ue4_quota')or{}
                if quota.get('status')=='observed':
                    values+=f'; UE4 post-apply max {quota["maximum_gflops"]:.3f} / quota {quota["budget_gflops"]:.3f} GFLOP/s'
                else:values+='; UE4 post-apply quota: unverified'
            control=r.get('control_provenance',{});control_label=control.get('status','unverified')
            parts.append('<tr><td>'+esc(f'{name} / {r["seed"]} / {r["scene"]}')+'<br><small>'+esc(control_label)+'</small></td><td>'+esc(values)+'</td><td class="'+esc(e['status'])+'">'+esc(text)+'</td><td>'+esc('; '.join(losses)or'None observed among the other five target KPIs')+'</td></tr>')
        parts.append('</table>')
    parts.append('<h2>Policy action diversity</h2><table><tr><th>OFF scene / seed</th><th>Same-observation C/D samples</th><th>Equal shared scheduler resources</th><th>Equal whole bundles</th></tr>')
    for b in report['baselines']:
        o=b['proposal_overlap'];parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>'for v in(
            f'{b["case"]["scene"]} / {b["case"]["seed"]}',o['samples'],f'{o["shared_resource_equal"]}/{o["samples"]}',f'{o["whole_bundle_equal"]}/{o["samples"]}'))+'</tr>')
    parts+=['</table>','<img src="environment-validation.png" alt="Paired network KPI measurements">'if report.get('plot_created')else'',
        '<p><small>Population totals use all configured UEs/cells/workloads. MEC delay is a completed-task mean over matched mature birth cohorts; missing samples stay unavailable. URLLC is an interval failure-event ratio including carryover. This descriptive experiment does not establish statistical generalization or a guarantee. Full values, denominators, mean/max demand, side effects and source hashes are in <a href="results.json">results.json</a>.</small></p></main></html>']
    (Path(output)/'index.html').write_text(''.join(parts))


def pressure_summary(baselines,rules=None):
    band=(rules or{}).get('pilot_selection',{}).get('preferred_urllc_failure_fraction')
    result=[]
    for b in baselines:
        states=b['windows']['after']['states'];flow=b['flows']['after'];url=states['kpi:urllc_failure_fraction']
        value=url.get('value');band_status=None
        if band is not None:
            if not isinstance(band,list)or len(band)!=2 or not all(numeric(v)for v in band)or not 0<=band[0]<=band[1]:
                raise ValueError('invalid_preregistered_urllc_screening_band')
            band_status='unavailable'if not numeric(value)else'below_band'if value<band[0]else'above_band'if value>band[1]else'in_band'
        result.append(dict(stage=b['stage'],scene=b['case']['scene'],seed=b['case']['seed'],worker_id=b['worker_id'],
            url_failure=url,url_preferred_band=band,url_band_status=band_status,
            exogenous_offered_load_mbps=flow.get('exogenous_offered_load_mbps'),
            delivered_goodput_mbps=flow.get('delivered_goodput_mbps'),
            all_packet_arrival_minus_delivery_mbps=flow.get('arrival_minus_delivery_mbps'),
            overflow_bytes=flow.get('overflow_bytes'),pending_bytes_change=flow.get('pending_bytes_change'),
            service_class_flows=flow.get('service_class_flows'),
            natural_100ms_subwindows=flow.get('natural_100ms_subwindows'),
            subwindow_definition=flow.get('subwindow_definition'),
            mec_latency=states['kpi:mec_offloaded_latency_ms'],task_success=states['kpi:task_success_ratio'],
            ue4_budget=flow.get('ue4_budget'),population_expected_mec_demand=states['kpi:mec_expected_gflops'],
            full_after_birth_followup=b.get('full_after_birth_followup'),
            action_overlap={k:v for k,v in b['proposal_overlap'].items()if k!='samples_by_slot'},
            decision='Report OFF pressure only; URL band membership alone does not establish bulk-throughput headroom or viable task service.'))
    return result


def _singleton_summary_for_scene(pairs,cases,scene):
    result=[]
    for arm,(_,name,key,_,unit)in APPS.items():
        planned=[r['case']for r in cases if r['case'].get('arm')==arm]
        if not planned:continue
        observed=[r for r in pairs if r['arm']==arm]
        ids={r['job_id']for r in observed};expected={r['id']for r in planned}
        excluded={r['case']['id']for r in cases if r['case'].get('arm')==arm and r['status']=='not_run_environment_rejected'}
        statuses=Counter(r['target']['status']for r in observed)
        all_positive=bool(expected)and ids==expected and statuses['improved']==len(expected)
        quota_ok=(all((r.get('post_first_apply_ue4_quota')or{}).get('mean_compliant')is True
            and(r.get('post_first_apply_ue4_quota')or{}).get('maximum_compliant')is True for r in observed)
            and ids==expected)if arm=='F'else None
        whole_quota_ok=(all(r['flows']['after'].get('ue4_budget',{}).get('mean_compliant')is True
            and r['flows']['after'].get('ue4_budget',{}).get('maximum_compliant')is True for r in observed)
            and ids==expected)if arm=='F'else None
        status='pending'if ids!=expected else'positive_all_planned_pairs'if all_positive else'requires_environment_iteration'
        if expected and expected<=excluded:status='not_run_environment_rejected'
        if arm=='F'and all_positive and not quota_ok:status='positive_population_effect_but_ue4_quota_not_verified'
        result.append(dict(scene=scene,arm=arm,name=name,target_kpi=key,unit=unit,planned_pairs=len(expected),paired=len(ids),
            status=status,counts=dict(statuses),all_planned_pairs_numerically_positive=all_positive,
            control_provenance_qualified_all_pairs=bool(expected)and ids==expected and all(r.get('control_provenance',{}).get('qualified')is True for r in observed),
            released_ue4_quota_satisfied_all_pairs=quota_ok,
            ue4_quota_review_window='From first actual own applied consumer action (including noop) through AFTER end; no feasible-action cherry-picking.'if arm=='F'else None,
            whole_after_ue4_quota_compliant_all_pairs=whole_quota_ok,
            missing_job_ids=sorted(expected-ids-excluded),intentionally_not_run_job_ids=sorted(excluded),
            per_seed=[dict(seed=r['seed'],worker_id=r['worker_id'],scene=r['scene'],
                off_after_value=r['off_after']['states']['kpi:'+key].get('value'),
                after_value=r['after']['states']['kpi:'+key].get('value'),
                sample_count=r['after']['states']['kpi:'+key].get('sample_count'),
                off_sample_count=r['off_after']['states']['kpi:'+key].get('sample_count'),
                primary_window_slots=r['after']['window_slots'],effect=r['target'],
                control_provenance=r.get('control_provenance'),
                post_first_apply_ue4_quota=r.get('post_first_apply_ue4_quota'),
                mec_selection_effect_concern=r['mec_completion_diagnostic']['selection_effect_concern'])for r in observed],
            interpretation='Strictly positive measured target direction is required; a zero-error URLLC 0→0 or any unchanged target is not a successful positive gain. Numerical positivity is not a significance or generalization claim.'))
    return result


def singleton_summary(pairs,cases):
    scenes=sorted({r['case']['scene']for r in cases if r['case'].get('arm')in APPS})
    return [entry for scene in scenes for entry in _singleton_summary_for_scene(
        [p for p in pairs if p['scene']==scene],[c for c in cases if c['case']['scene']==scene],scene)]


def recipe_assessments(summary,role):
    result=[]
    for scene in sorted({s['scene']for s in summary}):
        rows=[s for s in summary if s['scene']==scene]
        seeds=sorted({p['seed']for s in rows for p in s['per_seed']})
        complete_seed_sets=[{p['seed']for p in s['per_seed']}for s in rows]
        common_seeds=sorted(set.intersection(*complete_seed_sets))if complete_seed_sets else[]
        numeric_positive=len(rows)==6 and all(s['all_planned_pairs_numerically_positive']for s in rows)
        controls=len(rows)==6 and all(s['control_provenance_qualified_all_pairs']for s in rows)
        f_rows=[s for s in rows if s['arm']=='F']
        quota=bool(f_rows)and all(s['released_ue4_quota_satisfied_all_pairs']is True for s in f_rows)
        validated=role=='validation'and len(common_seeds)>=2 and numeric_positive and controls and quota
        rejected_apps=[s['arm']for s in rows if s['intentionally_not_run_job_ids']]
        result.append(dict(scene=scene,analysis_role=role,observed_seeds=seeds,
            common_complete_seeds_across_all_apps=common_seeds,
            all_six_target_directions_positive_in_planned_cases=numeric_positive,
            all_controls_verified=controls,F_post_apply_quota_verified=quota,
            intentionally_not_run_apps=rejected_apps,
            same_environment_independent_two_seed_validation_passed=validated,
            status='intentionally_rejected_candidate'if rejected_apps else'validated_positive_directions'if validated else'requires_fresh_two_seed_validation'if role=='discovery'and numeric_positive else'not_yet_verified',
            interpretation='Different recipes and seeds are not pooled into one environment. Single-seed discovery cannot establish final two-seed validation.'))
    return result


def analyze(plans,worker_roots,output,*,plots=True,decision_rules=None,activation_audit=None,analysis_role='auto',environment_exclusions=None,require_receiver_seals=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cases=[];inputs=[]
    plan_values=[]
    for path in plans:
        plan=read(path);inputs.append(dict(path=str(Path(path).resolve()),sha256=file_sha(path)))
        plan_values.append(plan)
        if not isinstance(plan.get('cases'),list):raise ValueError('flat_cases_required')
        cases.extend(inspect_case(c,plan,path,worker_roots,require_receiver_seals=require_receiver_seals)for c in plan['cases'])
    ids=[(c['stage'],c['worker_id'],c['case']['id'])for c in cases]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate_physical_case')
    exclusion_ref=bind_environment_exclusions(cases,environment_exclusions,plan_values,worker_roots)
    activation_ref=bind_activation_audit(cases,activation_audit)
    paired,baselines=pair_results(cases);rules=read(decision_rules)if decision_rules else None
    summary=singleton_summary(paired,cases)
    if analysis_role=='auto':
        analysis_role=('validation'if all(p.get('stage')=='validation'for p in plan_values)else
            'discovery'if any(c['case'].get('app_ids')for c in cases)else'pilot')
    if analysis_role not in ('pilot','discovery','validation'):raise ValueError('unknown_analysis_role')
    recipes=recipe_assessments(summary,analysis_role)
    result=dict(schema='oran.environment-validation-analysis.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        plan_inputs=inputs,analysis_code=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__)),
        comparison='single_app_after_minus_same_origin_off_after',intent_evaluation_used=False,
        recording_admission=dict(receiver_seals_required=require_receiver_seals,all_published_raw_sha256_required=True),
        new_simulations=0,training_performed=False,network_modified=False,
        complete_cases=[c for c in cases if c['status']=='complete'],pending_cases=[c for c in cases if c['status']=='pending'],
        intentionally_not_run_cases=[c for c in cases if c['status']=='not_run_environment_rejected'],environment_exclusions=exclusion_ref,
        paired_results=paired,baselines=baselines,environment_contrasts=environment_contrasts(baselines),
        baseline_pressure_summary=pressure_summary(baselines,rules),
        singleton_summary=summary,
        analysis_role=analysis_role,recipe_assessments=recipes,
        same_environment_two_seed_validation_passed=(len(recipes)==1 and recipes[0]['same_environment_independent_two_seed_validation_passed']),
        activation_audit=activation_ref,
        all_six_targets_numerically_positive=(len(summary)==6 and all(s['all_planned_pairs_numerically_positive']for s in summary)),
        all_six_controls_verified_and_numerically_positive=(len(summary)==6 and all(s['all_planned_pairs_numerically_positive']
            and s['control_provenance_qualified_all_pairs']and(s['arm']!='F'or s['released_ue4_quota_satisfied_all_pairs'])for s in summary)),
        decision_rules=dict(path=str(Path(decision_rules).resolve()),sha256=file_sha(decision_rules),value=rules)if decision_rules else None,
        environment_selection_performed_by_analyzer=False,default_adoption_performed=False,
        definitions=dict(power='Population total mean power from energy-counter differences, W.',
            goodput='Delivered downlink bits per second; packet offered load and exogenous arrival load are separate.',
            service_gap='Offered minus delivered flow; may be negative while draining backlog. Overflow and pending growth are reported separately.',
            mec_latency='Completed-only mean for same-window mature births; counts/failed/pending/guard retained; no zero imputation.',
            urllc='Interval failure events divided by offered packets, including carryover; not cohort failure probability.',
            mec_demand='Expected offered compute over all configured MEC sources, time mean/min/max plus per-source breakdown; not server utilization.',
            action_overlap='C/D shadow policies evaluated from the same OFF observation; never compare independent ON trajectories.'),
        limitations=['Seed-wise descriptive effects; no fabricated confidence intervals or significance claims.',
            'Environment contrasts jointly change configured traffic and initial service policy; not single-factor causal attribution.',
            'Missing task observations stay unavailable; conditional completion means must be read alongside failures and pending tasks.'])
    result['plot_created']=bool(plots)
    if analysis_role=='discovery':result['limitations'].append('Each recipe is independent single-seed discovery. Different recipes are never pooled as two seeds validating one environment; fresh same-recipe independent seeds remain necessary.')
    if plots:make_plot(result,output)
    write(output/'results.json',result);make_html(result,output)
    write(output/'compact-results.json',dict(schema=result['schema'],plan_inputs=inputs,
        complete=len(result['complete_cases']),pending=len(result['pending_cases']),
        planned=len(cases),intentionally_not_run=len(result['intentionally_not_run_cases']),environment_exclusions=exclusion_ref,
        baseline_pressure=result['baseline_pressure_summary'],
        analysis_role=analysis_role,recipe_assessments=recipes,
        same_environment_two_seed_validation_passed=result['same_environment_two_seed_validation_passed'],
        singleton_summary=summary,all_six_targets_numerically_positive=result['all_six_targets_numerically_positive'],
        all_six_controls_verified_and_numerically_positive=result['all_six_controls_verified_and_numerically_positive'],
        activation_audit=activation_ref,
        paired_results=[{k:v for k,v in r.items()if k not in ('before','after','off_after','flows','off_flows')}for r in paired],
        environment_contrasts=result['environment_contrasts'],decision_rules=result['decision_rules']))
    files=['results.json','compact-results.json','index.html']+(['environment-validation.png']if plots else[])
    write(output/'analysis-manifest.json',dict(schema=result['schema'],plans=inputs,
        complete=len(result['complete_cases']),pending=len(result['pending_cases']),paired_single_apps=len(paired),
        planned=len(cases),intentionally_not_run=len(result['intentionally_not_run_cases']),
        files={n:dict(path=str((output/n).resolve()),sha256=file_sha(output/n))for n in files}))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,action='append',required=True)
    p.add_argument('--worker-root',action='append',default=[],help='worker=directory or stage:worker=directory')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--no-plots',action='store_true')
    p.add_argument('--decision-rules',type=Path,help='Pinned OFF-only screening rules; annotate, never automatically choose or adopt a scene.')
    p.add_argument('--activation-audit',type=Path,help='Independent per-job deployment/consumer audit. Missing or unverified evidence cannot establish control attribution.')
    p.add_argument('--analysis-role',choices=('auto','pilot','discovery','validation'),default='auto')
    p.add_argument('--environment-exclusions',type=Path,help='Explicit receipt-bound cases rejected before admission; never inferred from missing telemetry.')
    p.add_argument('--require-receiver-seals',action='store_true',help='Require receiver binding of the exact mirrored checkpoint bytes before reading recordings; raw SHA checks remain mandatory.')
    a=p.parse_args(argv);roots={}
    for value in a.worker_root:
        key,sep,path=value.partition('=')
        if not sep or key in roots:raise ValueError('unique_worker_root_assignments_required')
        roots[key]=path
    r=analyze(a.plan,roots,a.output,plots=not a.no_plots,decision_rules=a.decision_rules,activation_audit=a.activation_audit,analysis_role=a.analysis_role,environment_exclusions=a.environment_exclusions,require_receiver_seals=a.require_receiver_seals)
    print(json.dumps(dict(output=str(a.output.resolve()),complete=len(r['complete_cases']),pending=len(r['pending_cases']),paired_single_apps=len(r['paired_results']))))


if __name__=='__main__':main()
