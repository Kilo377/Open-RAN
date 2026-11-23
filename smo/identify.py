"""Audit fixed interventions and fit seed-blocked local first-order responses.

The seed is the independent replication unit. Slot samples are never counted
as independent experiments. The numerical slope uses symmetric low/high arms;
the center arm checks local linearity. Categorical actions use named contrasts.
"""
from collections import Counter
from copy import deepcopy
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.rcp_perturbation import vector,USERS
from experiments.task_event_metrics import TaskEventCollector
from smo.contracts import build_graph,KPIS
from smo.runtime import records,FrameBuilder,trace_users
from smo.population import SCOPE


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def numeric(v):return type(v)in(int,float)and math.isfinite(v)


def response_delta(low,high):
    # Preserve full raw pairs; suppress only IEEE counter-ratio roundoff.
    return 0. if math.isclose(low,high,rel_tol=1e-11,abs_tol=1e-12)else high-low


def counter_delta(end,start):
    if not numeric(end)or not numeric(start)or end<start-1e-9:
        return None
    return max(0.,end-start)


def audit_case(study,case,plan, *, population=False):
    folder=Path(study)/'cases'/case['name']
    status=load(folder/'run_status.json');summary=load(folder/'final_summary.json')
    config=load(folder/'config_snapshot.json');initial=load(folder/'initial_conditions.json')
    if not status.get('completed')or not status.get('replay',{}).get('audits_passed'):
        raise ValueError('Incomplete or unaudited case: '+case['name'])
    if status['source_sha256']!=plan['source_sha256'] or config['source_sha256']!=plan['source_sha256']:
        raise ValueError('Mismatched simulator sources')
    spec=plan['specs'][case['arm']]
    timeline_probe=plan.get('stimulus_kind')=='fixed_scheduler_probe'
    if timeline_probe:
        if load(folder/'control_timeline.json')!=spec or sha(folder/'stimulus_driver.py')!=plan['stimulus_driver_sha256']:
            raise ValueError('Recorded fixed scheduler probe differs from its pinned plan/driver')
    elif load(folder/'perturbation_spec.json')!=spec or config.get('rcp_perturbation')!=spec:
        raise ValueError('Recorded intervention differs from the preregistered plan')
    if config['configs']['ric']['rapps']or config['configs']['ric']['xapps']:
        raise ValueError('Unrelated learned controllers were enabled')
    if not config['configs']['ric']['freeze_learning']:
        raise ValueError('Identification must not train models')
    start=round(plan['scoring_start_ms']*2);end=round(plan['scoring_end_ms']*2)
    target=deepcopy(plan['executed_target_vector'])if timeline_probe else vector(spec,perturbed=True)
    receipts=iter(records(folder,'control_receipts'));receipt=next(receipts,None)
    builder=FrameBuilder(config,window_ms=plan['scoring_end_ms']-plan['scoring_start_ms'],population=population)
    if not timeline_probe:
        requests=list(records(folder,'perturbation_requests'))
        if not requests or any(r['requested_vector']!=vector(spec,perturbed=r['slot_id']*.5>=spec['switch_ms'])for r in requests):
            raise ValueError('Stimulus log does not match its fixed arm')
    prefix=hashlib.sha256();exogenous=hashlib.sha256();births=hashlib.sha256()
    history=TaskEventCollector(plan['duration_s'],start_s=plan['scoring_start_ms']/1000)
    first=None;last=None;final=None;rows=0;checked=0;rejections=Counter();applied=0;executed_weights={}
    for row in records(folder,'rt_sys_trace'):
        rows+=1
        if row['slot_id']!=rows or {u['ue_id']for u in row['ues']}!=set(USERS):
            raise ValueError('Incomplete/duplicate slot or UE records')
        while receipt is not None and receipt.get('slot_id',receipt.get('apply_slot',0))<=rows:
            if receipt.get('status')=='rejected':rejections[receipt.get('reason')]+=1
            if receipt.get('status')=='applied':
                applied+=1
                for cell in receipt.get('after',{}).values():executed_weights.update(cell.get('ue_scheduler_weights',{}))
            builder.observe_receipt(receipt)
            receipt=next(receipts,None)
        services=row.get('ue_services',{});ap=row.get('ap_services',{}).get('cells',{})
        users=trace_users(row)
        exogenous.update(canonical(dict(slot=rows,arrivals={uid:v['exogenous_bytes']for uid,v in row['traffic_arrivals'].items()})))
        for event in row.get('ue_service_events',[]):
            history.observe(event)
            if event.get('kind')=='task_generated':
                if event['offloaded'] is None:
                    raise ValueError('Admission changed RNG consumption before routing')
                births.update(canonical({k:event[k]for k in ('ue_id','task_id','created_s','deadline_s','cohort_index')}))
        if rows<=round(plan['switch_ms']*2):
            prefix.update(canonical(dict(slot=rows,users=users,services={uid:{k:v for k,v in s.items()if k!='ue_task_cohorts'}for uid,s in services.items()},
                ap={cid:{k:a.get(k)for k in ('ap_energy_total_j','ap_policy','ap_available_fraction')}for cid,a in ap.items()})))
        if start<rows<=end:
            for uid,p in target['ue_drx_configs'].items():
                actual=services[uid]['ue_service_policy']
                if actual['drx']!=p:
                    raise ValueError('Executed DRX differs from its held arm')
                if uid in target['ue_offload_fraction']and actual['offload_fraction']!=target['ue_offload_fraction'][uid]:
                    raise ValueError('Executed offload differs from its held arm')
            membership={u['ue_id']:u['cell_id']for u in users}
            for uid,w in target['ue_scheduler_weights'].items():
                # Weight belongs to UE state and follows its handover. A
                # per-cell receipt snapshot may lag the new cell by one slot.
                if executed_weights.get(uid)!=w:
                    raise ValueError(f"Executed scheduling weight differs from its held arm: slot={rows} uid={uid} expected={w} actual={executed_weights.get(uid)}")
            for cid in (c['cell_id']for c in config['manifest']['cells']):
                if builder.executed.get(cid,{}).get('scheduler_mode')!=target['scheduler_mode']:
                    raise ValueError('Executed scheduler differs from its fixed context')
                if ap[cid]['ap_policy']!=target['cell_sleep_mode']:
                    raise ValueError('Executed AP policy differs from its held arm')
            checked+=1
        if rows==start:
            first=deepcopy(row)
            builder.frame(rows,users,services,ap,{})
        if rows==end:
            last=deepcopy(row)
            measured=builder.frame(rows,users,services,ap,{})
        final=row
    while receipt is not None:
        if receipt.get('status')=='rejected':rejections[receipt.get('reason')]+=1
        receipt=next(receipts,None)
    # A delayed refresh can be rejected during a handover while its already
    # applied persistent target remains unchanged. New longer studies may
    # explicitly permit this exact reason, but only AFTER verifying every
    # scored slot above. All other rejection reasons remain blocking.
    unexpected={r:n for r,n in rejections.items()if r not in plan.get('allowed_refresh_rejection_reasons',[])}
    if rows!=plan['slots']or checked!=end-start or first is None or last is None or unexpected:
        raise ValueError('Missing score interval or rejected experimental controls: '+str(rejections))
    for uid,p in summary['packet_metrics'].items():
        if p['initial_bytes']+p['offered_bytes']!=p['delivered_bytes']+p['pending_bytes']+p['overflow_bytes']:
            raise ValueError('Packet-byte conservation failed: '+uid)
    values={k[len('kpi:'):]:v['value']for k,v in measured['states'].items()if k.startswith('kpi:')}
    if not population:
        values['per_ue_goodput_mbps']=measured['states']['kpi:per_ue_goodput_mbps']['by_entity']
    energy_ids=USERS if population else('ue_0','ue_4')
    values['ue_energy_j']=sum(last['ue_services'][uid]['ue_energy_total_j']-first['ue_services'][uid]['ue_energy_total_j']for uid in energy_ids)
    values['ue_active_fraction']=sum((last['ue_services'][uid]['ue_active_fraction']*end-first['ue_services'][uid]['ue_active_fraction']*start)/(end-start)for uid in energy_ids)/len(energy_ids)
    values['ap_available_fraction']=sum((last['ap_services']['cells'][cid]['ap_available_fraction']*end-first['ap_services']['cells'][cid]['ap_available_fraction']*start)/(end-start)for cid in last['ap_services']['cells'])/len(last['ap_services']['cells'])
    before={u['ue_id']:u for u in first['ues']};after={u['ue_id']:u for u in last['ues']}
    values['queue_overflow_bytes']=sum(after[uid]['packet_overflow_bytes']-before[uid]['packet_overflow_bytes']for uid in USERS)
    tasks=history.summary()
    if 'cohort_end_ms' in plan:
        for u in tasks['per_ue'].values():
            u['cohorts']=[c for c in u['cohorts']if c['end_s']<=plan['cohort_end_ms']/1000+1e-9]
            u['eligible_cohorts']=len(u['cohorts']);u['eligible_generated']=sum(c['generated']for c in u['cohorts'])
    cohorts=[c for u in tasks['per_ue'].values()for c in u['cohorts']]
    totals={k:sum(c[k]for c in cohorts)for k in ('generated','completed','on_time_completed','failed','pending',
        'completion_latency_ms_sum','offloaded_generated','offloaded_completed','offloaded_pending','offloaded_completion_latency_ms_sum')}
    values['mec_offloaded_latency_ms']=totals['offloaded_completion_latency_ms_sum']/totals['offloaded_completed']if totals['offloaded_completed']else None
    values['task_latency_ms']=totals['completion_latency_ms_sum']/totals['completed']if totals['completed']else None
    values['task_success_ratio']=totals['on_time_completed']/totals['generated']if totals['generated']else None
    if totals['generated']!=plan.get('expected_task_count',20) or len(cohorts)!=plan.get('expected_task_cohort_count',10):
        raise ValueError('Recorded mature task cohorts differ from the preregistered scoring window')
    # Check exactly what was preserved, not just a derived metric fingerprint.
    checkpoint=load(Path(study)/'checkpoints'/(case['name']+'.json'))
    for rel,digest in checkpoint['files_sha256'].items():
        if sha(folder/rel)!=digest:raise ValueError('Raw case changed after checkpoint: '+rel)
    return dict(name=case['name'],seed=case['seed'],arm=case['arm'],metrics=values,
        task_counts=totals,per_ue_task_cohorts=tasks['per_ue'],
        proof=dict(source_sha256=status['source_sha256'],exogenous_environment_sha256=initial['exogenous_environment_sha256'],
            cache_key=status['cache_key'],native_prefix_sha256=prefix.hexdigest(),
            exogenous_arrivals_sha256=exogenous.hexdigest(),task_births_sha256=births.hexdigest(),
            verified_held_parameter_slots=checked,applied_receipts=applied,rejected_receipts=dict(rejections),
            tolerated_refresh_rejections=bool(rejections)and not unexpected,weight_state_basis='last_applied_persistent_per_UE_value_across_handover',
            files_sha256=checkpoint['files_sha256'],stimulus_protocol=plan.get('stimulus_kind','population_rcp_hold')),
        performance=load(folder/'performance.json'))


def paired_summary(values):
    """Two-seed Student interval, not slot-level pseudoreplication.

    This intentionally wide pilot interval has df=1. Zero observed variation
    does not establish a globally zero effect. Longer/more-seed validation is
    needed before using the local approximation as a guarantee.
    """
    mean=statistics.mean(values)
    se=statistics.stdev(values)/math.sqrt(len(values))if len(values)>1 else None
    critical={2:12.706204736432095,3:4.302652729696142,4:3.182446305284263,5:2.776445105197799,6:2.570581835636305,7:2.446911851144969,8:2.3646242510103,9:2.306004135204166,10:2.2621571628540993}.get(len(values))
    ci=[mean-critical*se,mean+critical*se]if critical is not None else None
    sign_agreement=sum((v>0)==(mean>0)and v!=0 for v in values)if mean else sum(v==0 for v in values)
    return dict(estimate=mean,standard_error=se,confidence_interval=ci,replicate_range=[min(values),max(values)],
        sign_agreement=sign_agreement,zero_in_both=all(v==0 for v in values))


def fit_contrasts(plan,cases,data_sha256, *, population=False):
    lookup={(c['seed'],c['arm']):c for c in cases}
    result=[]
    for c in plan['contrasts']:
        for k,(_,unit,_,_)in KPIS.items():
            vectors=k=='per_ue_goodput_mbps' and not population
            per={};raw=[];centers=[];base_values=[];pairs=[]
            for seed in plan['seeds']:
                low=lookup[seed,c['low']]['metrics'][k];high=lookup[seed,c['high']]['metrics'][k]
                ref=lookup[seed,c['reference']]['metrics'][k]
                pairs.append(dict(seed=seed,low=low,high=high,reference=ref))
                if vectors:
                    for uid in USERS:
                        if all(numeric(v.get(uid))for v in (low,high,ref)):
                            per.setdefault(uid,[]).append(response_delta(low[uid],high[uid])/c['step'])
                elif all(numeric(v)for v in(low,high,ref)):
                    raw.append(response_delta(low,high)/c['step']);base_values.append(ref)
                    if c['kind']=='local_derivative':centers.append((high+low)/2-ref)
            count=min((len(v)for v in per.values()),default=0)if vectors else len(raw)
            if count<2:continue
            summary=paired_summary(raw)if not vectors else dict(estimate=None,
                entity_estimates={uid:paired_summary(v)['estimate']for uid,v in per.items()},
                entity_uncertainty={uid:paired_summary(v)for uid,v in per.items()})
            row=dict(parameter=c['parameter'],kpi=k,contrast_id=c['id'],kind=c['kind'],
                parameter_component=c.get('parameter_component',{'offload_ue4':'ue_4','drx_ue4':'ue_4.inactivity_ms',
                    'weight_mec_ue4':'ue_4','weight_urllc_ue6':'ue_6'}.get(c['id'],'all_cells')),
                unit=unit+(' / fraction'if c['parameter']=='ue_offload_fraction'else' / weight'if c['parameter']=='ue_scheduler_weights'else' / contrast'),
                sample_count=count,observed_run_count=count*(3 if c['kind']=='local_derivative'else 2),
                horizon_ms=plan['scoring_end_ms']-plan['switch_ms'],
                scope=c['scope'],context=c['context'],design='controlled_perturbation',data_sha256=data_sha256,
                method='seed-blocked local linear regression; symmetric central difference'if c['kind']=='local_derivative'else'seed-paired binary-action contrast',
                scoring_start_ms=plan['scoring_start_ms'],scoring_end_ms=plan['scoring_end_ms'],
                scoring_window_ms=plan['scoring_end_ms']-plan['scoring_start_ms'],
                finite_step_high_minus_low=c['step'],seed_pairs=pairs,
                numerical_difference_tolerance=dict(relative=1e-11,absolute=1e-12,raw_pairs_preserved=True),
                confidence_level=.95,independent_unit='seed_pair',
                interpretation='Short two-seed local pilot. Zero/noisy response does not exclude an effect in another load, operating point or horizon.',
                **summary)
            if population:
                row['metric_scope']=SCOPE
                row['population_control_eligible']=(row['parameter_component']=='all_cells' or
                    plan.get('population')is True and c.get('population_control_eligible')is True)
                if row['population_control_eligible']:
                    row['context']=row['context'].replace('p4=.5','native MEC offload')
            field=c.get('level_field')or{'ue_offload_fraction':'offload_ue4','ue_drx_configs':'drx_ue4_inactivity_ms',
                'ue_scheduler_weights':'weight_ue6'if c['id']=='weight_urllc_ue6'else'weight_ue4',
                'scheduler_mode':'scheduler_mode','cell_sleep_mode':'cell_sleep_mode'}[c['parameter']]
            row['parameter_levels']={name:plan['specs'][c[arm]][field]for name,arm in(('low','low'),('reference','reference'),('high','high'))}if'specs'in plan else{}
            if c['kind']=='finite_contrast':row.update({'from':c['from'],'to':c['to']})
            if centers:row['center_linearity_residual']=statistics.mean(centers)
            if base_values:
                ref=statistics.mean(base_values)
                row['reference_kpi_mean']=ref
                row['high_minus_low_percent']=row['estimate']*c['step']/abs(ref)*100 if ref else None
            result.append(row)
    return result


def analyze(study,output, *, population=False):
    study=Path(study);output=Path(output);plan=load(study/'plan.json')
    if len(plan['seeds'])!=2 or plan['duration_s']!=.8:
        raise ValueError('This preregistered pilot analyzer currently requires two seeds × 0.8s')
    state=load(study/'study_status.json')
    if not state.get('completed')or state.get('failed'):
        raise ValueError('Wait for the complete checkpointed sweep')
    cases=[audit_case(study,c,plan,population=population)for c in plan['cases']]
    for seed in plan['seeds']:
        same=[c['proof']for c in cases if c['seed']==seed]
        for key in('source_sha256','exogenous_environment_sha256','cache_key','native_prefix_sha256','exogenous_arrivals_sha256','task_births_sha256'):
            if len({p[key]for p in same})!=1:
                raise ValueError('Paired exogenous/warmup evidence differs: '+str(seed)+':'+key)
    dataset=dict(schema='oran.rcp-perturbation-measurements.v1',plan=plan,cases=cases,
        metric_notes=dict(ue_energy_j='Score-window all-UE energy counter delta.' if population else 'Score-window UE0+UE4 energy counter delta, not accumulated startup energy.',
            task_latency='Mature creation cohorts born 200–700ms, terminals through 800ms; completion means conditional on completion.',
            fractions='URLLC failure fraction is the cumulative end-of-score event/offered ratio; not mature-cohort SLA. UE/AP availability are score-window counter ratios.',
            per_ue_goodput='Arithmetic mean across the complete UE population, including zero rates.' if population else 'Full six-UE vector; no mean-of-means substitution.',
            response='100ms wash-in; 500ms score window; response ends 600ms after the intervention.'))
    if population:
        dataset['metric_scope']=SCOPE
        dataset['metric_notes']['mec_demand']='Sum across all configured task-generating UEs.'
    output.mkdir(parents=True,exist_ok=True);write(output/'measurement-dataset.json',dataset)
    estimates=fit_contrasts(plan,cases,sha(output/'measurement-dataset.json'),population=population)
    write(output/'all-effects.json',dict(schema='oran.intervention-edges.v1',estimates=estimates,
        metric_scope=SCOPE if population else 'legacy_target_subset'))
    graph=build_graph();candidates={(e['source'][4:],e['target'][4:])for e in graph['edges']if e['kind']=='influences'}
    # Preserve all prior candidates, including unresolved/zero estimates. New
    # exploratory links enter the graph only with an interval excluding zero;
    # the full matrix is retained separately, with no multiple-test claim.
    def resolved(r):
        intervals=[r['confidence_interval']]if r.get('confidence_interval')is not None else [v['confidence_interval']for v in r.get('entity_uncertainty',{}).values()]
        return any(ci is not None and (ci[0]>0 or ci[1]<0)for ci in intervals)
    eligible=[r for r in estimates if not population or r['population_control_eligible']]
    discovered={(r['parameter'],r['kpi'])for r in eligible if resolved(r)}-candidates
    published=[r for r in eligible if(r['parameter'],r['kpi'])in candidates|discovered]
    write(output/'influences.json',dict(schema='oran.intervention-edges.v1',estimates=published,
        study='Two-seed fixed-RCP pilot; intervals are descriptive, exploratory additions not multiplicity-corrected.',
        complete_matrix='all-effects.json',measurement_dataset_sha256=sha(output/'measurement-dataset.json'),
        metric_scope=SCOPE if population else 'legacy_target_subset'))
    with(output/'measurements.csv').open('w',newline='')as f:
        fields=['case','seed','arm']+[k for k in KPIS if population or k!='per_ue_goodput_mbps']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cases:writer.writerow(dict(case=c['name'],seed=c['seed'],arm=c['arm'],**{k:v for k,v in c['metrics'].items()if population or k!='per_ue_goodput_mbps'}))
    report=dict(completed_cases=len(cases),simulated_seconds=len(cases)*.8,independent_seed_pairs=2,
        contrasts=len(plan['contrasts']),estimated_responses=len(estimates),published_estimates=len(published),
        graph_links_measured=len({(r['parameter'],r['kpi'])for r in published}),
        new_exploratory_links=[list(x)for x in sorted(discovered)],
        measurements_sha256=sha(output/'measurement-dataset.json'),all_raw_data_preserved=True,
        source_sha256=plan['source_sha256'],models_trained=False,
        observed_network_wall_seconds=sum(c['performance']['simulation_s']for c in cases))
    if population:
        report.update(metric_scope=SCOPE,excluded_single_member_estimates=len(estimates)-len(eligible),
            note=('Every intervention covers its complete applicable group; outcomes aggregate the UE population.'if plan.get('population')else
                  'Historical single-member probes are retained in all-effects.json but do not identify a common population control response.'))
    write(output/'identification-report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view',choices=('entity','population'),default='entity')
    args=p.parse_args();print(json.dumps(analyze(args.study,args.output,population=args.view=='population'),indent=2))


if __name__=='__main__':main()
