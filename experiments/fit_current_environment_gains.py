"""Fit only audited parameter-intervention records from the current environment."""
from copy import deepcopy
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.current_environment_calibration import atomic, digest, read, sha
from smo.contracts import KPIS, build_graph, validate_influences
from smo.grid_identify import line_fit
from smo.identify import paired_summary, response_delta
from smo.population import SCOPE, graph_view
from smo.runtime import FrameBuilder, records, trace_users
from smo.window_metrics import RecordedWindows

MEASUREMENT_FILES=('case.json','stimulus.json','command.json','simulation.log',
    'run/config_snapshot.json','run/run_status.json','run/control_timeline.json',
    'run/final_summary.json','run/initial_conditions.json',
    'run/rt_sys_trace.jsonl.gz','run/control_receipts.jsonl.gz')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def numeric(x):
    return type(x) in (int, float) and math.isfinite(x)


def audit(job, plan, case):
    """Verify admission, held state, input hashes and full paired traffic prefix."""
    job = Path(job)
    seal = read(job/'seal.json')
    mirror_path=job/'mirror-verified.json'if(job/'mirror-verified.json').exists()else job/'analysis-ready.json'
    mirror=read(mirror_path)
    if not mirror.get('verified') or mirror.get('seal_sha256')!=sha(job/'seal.json') or seal.get('case') != case:
        raise ValueError('A verified, same-case local mirror is required')
    if mirror_path.name=='analysis-ready.json':
        required=set(MEASUREMENT_FILES)
        if mirror.get('schema')!='oran.parameter-measurement-mirror.v1' or set(mirror.get('files_sha256',{}))!=required:
            raise ValueError('Incomplete essential measurement mirror')
        if any(mirror['files_sha256'][n]!=seal['files_sha256'].get(n)or sha(job/n)!=mirror['files_sha256'][n]for n in required):
            raise ValueError('Essential measurement bytes differ from the sealed physical run')
    if seal['plan_sha256'] != plan['plan_sha256'] or seal['source_pins'] != plan['source_pins']:
        raise ValueError('Raw evidence belongs to another experiment')
    run = job/'run'
    config, status = read(run/'config_snapshot.json'), read(run/'run_status.json')
    if not status.get('completed') or not config['configs']['ric']['freeze_learning']:
        raise ValueError('Incomplete or learning-enabled parameter run')
    if config['configs']['ric']['xapps'] or config['configs']['ric']['rapps']:
        raise ValueError('Trained controller results cannot identify P–K gains')
    if config.get('source_sha256') != status.get('source_sha256'):
        raise ValueError('Recorded physical source identities disagree')
    summary=read(run/'final_summary.json')
    for uid,packet in summary['packet_metrics'].items():
        if packet['initial_bytes']+packet['offered_bytes'] != packet['delivered_bytes']+packet['pending_bytes']+packet['overflow_bytes']:
            raise ValueError('Packet-byte conservation failed: '+uid)
    if read(run/'control_timeline.json') != read(job/'stimulus.json'):
        raise ValueError('Recorded stimulus differs')
    vector = case['vector']
    builder = FrameBuilder(config, window_ms=1000, population=True)
    receipt_iter = iter(records(run, 'control_receipts'))
    receipt = next(receipt_iter, None)
    prefix, arrivals, births = [hashlib.sha256() for _ in range(3)]
    weights = {}
    rejections = []
    checked = 0
    first = last = None
    for count, row in enumerate(records(run, 'rt_sys_trace'), 1):
        if row['slot_id'] != count:
            raise ValueError('Missing or duplicate physical slot')
        while receipt is not None and receipt.get('apply_slot', receipt.get('slot_id', 0)) <= count:
            if receipt.get('status') == 'rejected':
                rejections.append(receipt)
            if receipt.get('status') == 'applied':
                for state in receipt.get('after', {}).values():
                    weights.update(state.get('ue_scheduler_weights', {}))
            builder.observe_receipt(receipt)
            receipt = next(receipt_iter, None)
        users = trace_users(row)
        services = row['ue_services']
        aps = row['ap_services']['cells']
        arrivals.update(canonical(dict(slot=count, exogenous={u:v['exogenous_bytes'] for u,v in row['traffic_arrivals'].items()})))
        for event in row.get('ue_service_events', []):
            if event['kind'] == 'task_generated':
                births.update(canonical({k:event[k] for k in ('ue_id','task_id','created_s','deadline_s','cohort_index')}))
        if count <= plan['switch_slot']:
            prefix.update(canonical(dict(slot=count, users=users,
                services={u:{k:v for k,v in s.items() if k!='ue_task_cohorts'} for u,s in services.items()},
                ap={c:{k:a.get(k) for k in ('ap_energy_total_j','ap_policy','ap_available_fraction')} for c,a in aps.items()})))
        if plan['scoring_start_slot'] < count <= plan['scoring_end_slot']:
            for uid, target in vector['ue_drx_configs'].items():
                policy = services[uid]['ue_service_policy']
                if policy['drx'] != target:
                    raise ValueError(f'Executed DRX differs at {count}/{uid}')
                if uid in vector['ue_offload_fraction'] and policy['offload_fraction'] != vector['ue_offload_fraction'][uid]:
                    raise ValueError(f'Executed offload differs at {count}/{uid}')
            for uid, target in vector['ue_scheduler_weights'].items():
                if weights.get(uid) != target:
                    raise ValueError(f'Executed weight differs at {count}/{uid}')
            for cid, target in vector['scheduler_mode'].items():
                if builder.executed.get(cid, {}).get('scheduler_mode') != target:
                    raise ValueError(f'Executed scheduler differs at {count}/{cid}')
                if aps[cid]['ap_policy'] != vector['cell_sleep_mode'][cid]:
                    raise ValueError(f'Executed sleep policy differs at {count}/{cid}')
            checked += 1
        if count == plan['scoring_start_slot']: first = row
        if count == plan['scoring_end_slot']: last = row
    # The manager refreshes its last cached command each slot. The probe's
    # finite admission lease can expire after the original write has applied;
    # persistent executed configuration does not roll back on that expiry.
    # Accept only these late duplicate refreshes, after checking the complete
    # executed target on EVERY scored slot above. Initial or other rejections
    # are still fatal and are never interpreted as an applied intervention.
    unexpected=[r for r in rejections if r.get('reason')!='expired_before_actuation' or
        r.get('issued_at_slot',0)<plan['switch_slot']+39]
    if count != plan['total_slots'] or checked != plan['scoring_end_slot']-plan['scoring_start_slot'] or unexpected:
        raise ValueError('Incomplete interval or rejected fixed stimulus')
    index = RecordedWindows(run, config, population=True)
    index.sync()
    window = index.window(list(KPIS), plan['scoring_start_slot'], plan['scoring_end_slot'], through=index.slots[-1])
    # Cumulative energy is an historical auxiliary KPI, measured as an interval
    # counter difference for this controlled study rather than a cumulative age.
    energy = sum(last['ue_services'][u]['ue_energy_total_j']-first['ue_services'][u]['ue_energy_total_j'] for u in services)
    window['states']['kpi:ue_energy_j'].update(value=energy, reason=None, aggregation='interval_energy_counter_difference')
    return dict(case=case, metrics={k:v['value'] for k,v in window['states'].items()}, metric_details=window['states'],
        proof=dict(prefix_sha256=prefix.hexdigest(), exogenous_sha256=arrivals.hexdigest(), task_births_sha256=births.hexdigest(),
            verified_scored_slots=checked, rejected_requests=len(rejections), unexpected_rejections=0,
            rejected_refresh_reason='expired_before_actuation' if rejections else None,
            rejected_refreshes_do_not_undo_verified_executed_configuration=True,seal_sha256=sha(job/'seal.json'),
            local_mirror_kind='full_raw'if mirror_path.name=='mirror-verified.json'else'complete_numerical_evidence',
            physical_source_sha256=status['source_sha256'], source_archive_sha256=plan['source_archive_sha256']),
        observation=dict(start_slot=plan['scoring_start_slot'],end_slot=plan['scoring_end_slot'],through_slot=index.slots[-1]))


def summarize_contrast(observations):
    valid = [o for o in observations if numeric(o.get('low')) and numeric(o.get('high'))]
    if len(valid) < 2:
        return None
    return dict(**paired_summary([response_delta(o['low'],o['high']) for o in valid]),
        sample_count=len(valid), seed_pairs=valid, reference_kpi_mean=statistics.mean(o['low'] for o in valid))


def fit_scalar(points, seeds):
    """Each seed contributes one affine slope, not one sample per slot/UE."""
    kept = [p for p in points if all(numeric(p['values'].get(str(s))) for s in seeds)]
    if len(kept) < 3:
        return None
    xs = [p['x'] for p in kept]
    fits = {s:line_fit(xs,[p['values'][str(s)] for p in kept]) for s in seeds}
    summary = paired_summary([fits[s]['beta'] for s in seeds])
    beta = summary['estimate']
    residual = [p['values'][str(s)]-(fits[s]['mean']+beta*(p['x']-statistics.mean(xs))) for s in seeds for p in kept]
    total = sum((p['values'][str(s)]-fits[s]['mean'])**2 for s in seeds for p in kept)
    r2 = None if total == 0 else 1-sum(e*e for e in residual)/total
    return dict(**summary, kind='range_linear_fit' if len(kept)>=4 else 'local_derivative',
        sample_count=len(seeds), fit_domain=[min(xs),max(xs)], sampled_domain=[min(p['x'] for p in points),max(p['x'] for p in points)],
        fit_levels=xs, sampled_levels=[p['x'] for p in points], fit_point_count=len(kept), grid_point_count=len(points),
        grid_points=[dict(x=p['x'], by_seed=[dict(seed=int(s),value=v) for s,v in p['values'].items()],
            mean=statistics.mean(p['values'].values()) if all(numeric(v) for v in p['values'].values()) else None,
            valid_seed_count=sum(numeric(v) for v in p['values'].values()),used_in_fit=p in kept) for p in points],
        seed_fits=[dict(seed=s,**fits[s]) for s in seeds],
        fit_diagnostics=dict(within_seed_r2=r2,rmse=math.sqrt(statistics.mean(e*e for e in residual)),
            assessment='constant_response_observed' if total==0 else 'limited_first_order_approximation' if r2<.7 else 'first_order_approximation_supported'))


def estimate(design, measurements):
    lookup={(r['worker_id'],r['case']['seed'],r['case']['arm']):r for r in measurements}
    def value(worker,seed,arm,key):return lookup[worker,seed,arm]['metrics']['kpi:'+key]
    rows, unavailable = [], []
    groups={}
    for arm in design['arms']:
        if arm['kind']=='grid':
            group=(arm['worker'],arm['parameter'],arm['component'],arm.get('scalar_component'),arm['reference'])
            groups.setdefault(group,[]).append(arm)
    contrasts=[]
    for arm in design['arms']:
        if arm['kind'] in ('profile','category') or arm['kind']=='reference' and arm['id'] in ('latency','qos'):
            # Duplicate scheduler references on two hosts are independent input
            # parity evidence, not two independent samples for the same seed.
            if arm['id'] in ('latency','qos') and arm['worker']!='agx2':continue
            contrasts.append(arm)
    data_pin=digest(measurements)
    for key,(_,unit,_,_) in KPIS.items():
        for (worker,parameter,component,scalar,reference),arms in groups.items():
            native=next(a for a in design['arms'] if a['worker']==worker and a['id']==reference)
            if parameter=='ue_offload_fraction':
                levels={a['level']:a['id'] for a in arms}
            else:
                base_level=8 if scalar=='inactivity_ms' else 40 if scalar=='cycle_ms' else 1
                levels={base_level:reference,**{a['level']:a['id'] for a in arms}}
            points=[dict(x=x,values={str(s):value(worker,s,arm,key) for s in design['seeds']}) for x,arm in sorted(levels.items())]
            fitted=fit_scalar(points,design['seeds'])
            if fitted is None:
                unavailable.append(dict(parameter=parameter,kpi=key,component=component,reason='fewer_than_three_complete_numeric_levels',grid_points=points));continue
            denom='fraction' if parameter=='ue_offload_fraction' else 'ms' if parameter=='ue_drx_configs' else 'weight'
            refs=[value(worker,s,reference,key) for s in design['seeds']]
            row=dict(parameter=parameter,kpi=key,contrast_id='current:'+parameter+':'+component+':'+str(scalar)+':'+key,
                parameter_component=component,members=arms[0]['members'],scalar_component=scalar,
                reference_scheduler_mode=next(iter(native['vector']['scheduler_mode'].values())),unit=unit+' / '+denom,x_unit=denom,
                reference_parameter=statistics.mean(native['vector'][parameter][u] for u in arms[0]['members']) if parameter!='ue_drx_configs' else native['vector'][parameter][arms[0]['members'][0]][scalar],
                reference_kpi_mean=statistics.mean(refs) if all(numeric(v) for v in refs) else None,
                **fitted)
            row['intercept_at_reference']=statistics.mean(f['intercept']+f['beta']*row['reference_parameter'] for f in row['seed_fits'])
            row['contrast_label']={'mec_workload':'MEC offloading','mec_priority':'MEC priority · Latency-aware',
                'urllc_priority':'URLLC priority · QoS-deadline','efficiency_priority':'Throughput priority · Latency-aware',
                'drx_enabled_population':'DRX cycle' if scalar=='cycle_ms' else 'DRX inactivity'}[component]
            if parameter=='ue_scheduler_weights':
                row['conditioning']={'scheduler_mode':row['reference_scheduler_mode']}
            rows.append(row)
        for arm in contrasts:
            worker=arm['worker'];ref=next(a for a in design['arms'] if a['worker']==worker and a['id']==arm['reference'])
            parameter=arm['parameter'];observations=[dict(seed=s,low=value(worker,s,arm['reference'],key),high=value(worker,s,arm['id'],key)) for s in design['seeds']]
            summary=summarize_contrast(observations)
            if summary is None:
                unavailable.append(dict(parameter=parameter,kpi=key,arm=arm['id'],reason='fewer_than_two_numeric_pairs',seed_pairs=observations));continue
            if parameter=='ue_drx_configs':
                left=ref['vector'][parameter][arm['members'][0]];right=arm['vector'][parameter][arm['members'][0]]
                component='mec_drx_population';members=arm['members']
            else:
                left=next(iter(ref['vector'][parameter].values()));right=next(iter(arm['vector'][parameter].values()))
                component='all_cells';members=sorted(ref['vector'][parameter])
            rows.append(dict(parameter=parameter,kpi=key,contrast_id='current:'+arm['id']+':'+key,
                kind='finite_contrast',parameter_component=component,members=members,
                contrast_label=('DRX 40/4/8 → '+str(right['cycle_ms'])+'/4/1 ms') if parameter=='ue_drx_configs' else
                    str(left).replace('_',' ').capitalize()+' → '+str(right).replace('_',' ').capitalize(),
                **{'from':left,'to':right},unit=unit+' / contrast',reference_scheduler_mode='round_robin',**summary))
    for row in rows:
        ci=row['confidence_interval'];significant=ci is not None and (ci[0]>0 or ci[1]<0)
        zero=row['estimate']==0 and ci==[0,0]
        span=row['fit_domain'][1]-row['fit_domain'][0] if row['kind']!='finite_contrast' else 1.
        baseline=row.get('reference_kpi_mean')
        scale=1. if row['kpi'] in ('urllc_failure_fraction','task_success_ratio','ue_active_fraction','ap_available_fraction') else abs(baseline) if numeric(baseline) and baseline else 1.
        row.update(population_control_eligible=True,metric_scope=SCOPE,scope='Population KPI; stated parameter-member direction',
            context=design['environment_id'],design='controlled_perturbation',method='Same-host paired fixed interventions; independent-seed first-order response',
            data_sha256=data_pin,confidence_level=.95,
            independent_unit='seed_pair'if row['kind']=='finite_contrast'else'seed_block',horizon_ms=1200,
            scoring_start_ms=design['scoring_start_slot']*.5,scoring_end_ms=design['scoring_end_slot']*.5,scoring_window_ms=1000,
            calibration_environment_id=design['environment_id'],combination_evaluation_data_used=False,
            importance=dict(signed_reference_importance=row['estimate']*span/scale,parameter_span=span,kpi_scale=scale,
                normalization_basis='canonical_probability_range' if scale==1. and row['kpi'].endswith(('fraction','ratio')) else 'absolute_paired_reference_kpi'),
            statistical_context=dict(direction_significant=significant,confidence_level=.95,
                retained_as='significant_reference' if significant else 'observed_zero_reference' if zero else 'uncertain_reference',
                screening_reason=None if significant else 'observed_zero' if zero else 'interval_includes_zero'),
            caveat='Finite statistical influence conditional on this environment, population direction and horizon; not a prediction or a global linear model.')
    return dict(schema='oran.intervention-edges.v1',metric_scope=SCOPE,estimates=rows,unavailable=unavailable,
        current_environment_numeric_gains_available=True,historical_statistics_separate=True,
        scenario=dict(id=design['environment_id'],calibration_status='controlled_parameter_calibration',traffic_load=design['traffic_load'],
            dynamics_sha256=design['source_pins']['sha256'],source_archive_sha256=design['source_archive_sha256'],
            profiles={k:{'sha256':v['sha256']} for k,v in design['profiles'].items()},ue_count=6,cells=3,seeds=design['seeds']),
        provenance=dict(design_sha256=digest(design),measurement_sha256=data_pin,combination_evaluation_data_used=False,learned_xapps_invoked=False))


def _collect(folder):
    folder=Path(folder);design=read(folder/'design.json');measurements=[]
    for worker in ('agx1','agx2'):
        plan=read(folder/(worker+'-plan.json'))
        if digest({k:v for k,v in plan.items() if k!='plan_sha256'}) != plan['plan_sha256']:
            raise ValueError('Changed experimental plan')
        for case in plan['cases']:
            job=folder/'workers'/worker/'jobs'/case['name']
            if not any((job/n).exists()for n in('mirror-verified.json','analysis-ready.json')):continue
            cached=job/'parameter-audit.json'
            if cached.exists() and read(cached).get('auditor_sha256')==sha(__file__):
                measured=read(cached)['measurement']
            else:
                measured=audit(job,plan,case);atomic(cached,dict(auditor_sha256=sha(__file__),measurement=measured))
            measurements.append(dict(worker_id=worker,**measured))
    blocks={}
    for measured in measurements:
        key=(measured['worker_id'],measured['case']['seed']);proof=measured['proof']
        identities={k:proof[k] for k in ('prefix_sha256','exogenous_sha256','task_births_sha256','physical_source_sha256','source_archive_sha256')}
        if key in blocks and blocks[key]!=identities:
            raise ValueError('Unmatched input/prefix or source within paired seed: '+str(key))
        blocks[key]=identities
    return design,measurements


def collect(folder):
    # A manual audit and the completion watcher can meet at a checkpoint.
    # Serialize cache writes without locking any simulator or UI process.
    with (Path(folder)/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _collect(folder)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    design,measurements=collect(args.study)
    print(json.dumps(dict(audited=len(measurements),expected=sum(len(read(args.study/(w+'-plan.json'))['cases']) for w in ('agx1','agx2')))))
    if args.audit_only:return
    expected=sum(len(read(args.study/(w+'-plan.json'))['cases']) for w in ('agx1','agx2'))
    if len(measurements)!=expected:raise ValueError('Full current-environment design is not durably complete')
    result=estimate(design,measurements)
    validate_influences(graph_view(build_graph()),_write_result(args.study,result))
    atomic(args.study/'analysis/measurement-dataset.json',dict(design_sha256=sha(args.study/'design.json'),measurements=measurements))


def _write_result(study,result):
    path=study/'analysis/influences.json'
    if path.exists():raise ValueError('Keep prior fits; select a new analysis version')
    atomic(path,result);return path


if __name__=='__main__':main()
