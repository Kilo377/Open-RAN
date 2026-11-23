"""Scene-separated, independently seed-blocked RCP-to-KPI evidence.

Raw runs and their sealed checkpoints remain the source of every coefficient.
No slots, UEs or grid levels are counted as independent seed replications.
"""
from collections import defaultdict
from copy import deepcopy
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from smo.identify import audit_case,fit_contrasts,load,sha,write,numeric
from smo.grid_identify import fit_grid
from smo.contracts import KPIS,build_graph,validate_influences
from smo.population import graph_view,SCOPE


PROOF_KEYS=('source_sha256','exogenous_environment_sha256','cache_key','native_prefix_sha256','exogenous_arrivals_sha256','task_births_sha256')


def discover(artifact,context=None,duration_s=1.8):
    artifact=Path(artifact);result=[]
    for study in sorted((artifact/'workers').glob('agx*')):
        if not(study/'plan.json').exists():continue
        plan=load(study/'plan.json')
        received=load(study/'received-checkpoints.json')if(study/'received-checkpoints.json').exists()else{}
        entries=[(case,plan)for case in plan['cases']]
        supplement=study/'priority-plan.json'
        if supplement.exists():
            for case in load(supplement)['cases']:
                seal=study/'checkpoints'/(case['name']+'.json')
                if not seal.exists():continue
                info=load(seal)
                entries.append((case,{**plan,'specs':{case['arm']:case['timeline_spec']},
                    'stimulus_kind':'fixed_scheduler_probe','stimulus_driver_sha256':info['stimulus_driver_sha256'],
                    'executed_target_vector':info['executed_target_vector']}))
        for case,protocol in entries:
            if context is not None and case['context']!=context or case['duration_s']!=duration_s:continue
            if case['name']not in received:continue
            merged={**protocol,**case,'cohort_end_ms':case['cohort_end_ms'],
                'allowed_refresh_rejection_reasons':['ue_scheduler_weight_target_outside_cell_scope']}
            result.append((study,case,merged))
    return result


def extract(artifact,context=None,duration_s=1.8):
    artifact=Path(artifact);cache=artifact/'analysis-cache';cache.mkdir(exist_ok=True);cases=[]
    auditor_sha=sha(Path(__file__).with_name('identify.py'))
    for study,case,plan in discover(artifact,context,duration_s):
        checkpoint=study/'checkpoints'/(case['name']+'.json');seal_sha=sha(checkpoint)
        path=cache/(study.name+'_'+case['name']+'.json')
        old=load(path)if path.exists()else{}
        if old.get('checkpoint_sha256')==seal_sha and old.get('auditor_sha256')==auditor_sha:
            row=old['measurement']
        else:
            row=audit_case(study,case,plan,population=True)
            row.update(context=case['context'],duration_s=case['duration_s'],stage=case['stage'],
                scoring_start_ms=case['scoring_start_ms'],scoring_end_ms=case['scoring_end_ms'],
                stimulus_protocol=plan.get('stimulus_kind','population_rcp_hold'),
                origin=dict(worker_id=study.name,study=str(study),case=case['name'],checkpoint_sha256=seal_sha,
                    source_archive_sha256=plan.get('source_archive_sha256')))
            write(path,dict(checkpoint_sha256=seal_sha,auditor_sha256=auditor_sha,measurement=row))
        cases.append(row)
    blocks=defaultdict(list)
    for row in cases:blocks[row['context'],row['seed'],row.get('stimulus_protocol','population_rcp_hold')].append(row)
    for (ctx,seed,protocol),rows in blocks.items():
        if len({r['origin']['worker_id']for r in rows})!=1:raise ValueError('A context/seed block crossed hardware')
        for key in PROOF_KEYS:
            if len({r['proof'][key]for r in rows})!=1:raise ValueError(f'Unmatched exogenous block {ctx}/{seed}: {key}')
    return cases


def context_metadata(artifact,context,cases):
    design=load(Path(artifact)/'design.json');info=deepcopy(design['scenarios'][context])
    info.update(id=context,source_sha256s=sorted({c['proof']['source_sha256']for c in cases}),
        duration_s=cases[0]['duration_s'],scoring_start_ms=cases[0]['scoring_start_ms'],scoring_end_ms=cases[0]['scoring_end_ms'],
        metric_scope=SCOPE,ue_count=6,cells=3,phy_model='accepted fast/system; unchanged radio and moving UEs',
        metric_windows=measurement_windows(cases[0]['scoring_start_ms'],cases[0]['scoring_end_ms'],cases[0]['duration_s']))
    return info


def measurement_windows(start_ms,end_ms,duration_s):
    """Expose each metric's actual estimator; cumulative ratios are not deltas."""
    counter={'ue_power_w','ue_energy_j','ap_power_w','network_goodput_mbps','per_ue_goodput_mbps',
        'urllc_failure_events_per_s','urllc_packet_latency_ms','packet_latency_ms','queue_overflow_bytes',
        'ue_active_fraction','ap_available_fraction','mec_realized_gflops'}
    cohorts={'mec_offloaded_latency_ms','task_latency_ms','task_success_ratio'}
    result={}
    for k in KPIS:
        if k in counter:result[k]=dict(kind='counter_difference',start_ms=start_ms,end_ms=end_ms)
        elif k in cohorts:result[k]=dict(kind='matched_mature_birth_cohorts',birth_start_ms=start_ms,birth_end_ms=end_ms,followup_end_ms=duration_s*1000,
            conditional_on_completion=k!='task_success_ratio',pending_in_denominator=k=='task_success_ratio')
        elif k=='urllc_failure_fraction':result[k]=dict(kind='cumulative_ratio_including_common_warmup',start_ms=0,end_ms=end_ms,
            mature_packet_cohort=False)
        else:result[k]=dict(kind='instantaneous',at_ms=end_ms)
    return result


def importance(row):
    span=(row['fit_domain'][1]-row['fit_domain'][0])if row['kind']=='range_linear_fit'else 1.
    baseline=row.get('reference_kpi_mean');scale=abs(baseline)if numeric(baseline)and baseline else None
    basis='absolute_reference_kpi'
    if not scale:
        vals=[p.get('mean')for p in row.get('grid_points',[])if numeric(p.get('mean'))]
        vals += [p[k]for p in row.get('seed_pairs',[])for k in('low','high')if numeric(p.get(k))]
        scale=max(map(abs,vals),default=0.)or 1.;basis='observed_magnitude_or_one_unit_zero_reference_fallback'
    return dict(signed_reference_importance=row['estimate']*span/scale,parameter_span=span,kpi_scale=scale,normalization_basis=basis)


def analyze(artifact,context,output,*,duration_s=1.8,minimum_seeds=3):
    artifact=Path(artifact);output=Path(output)
    if output.exists()and(output/'identification-report.json').exists():raise ValueError('Use a new versioned analysis directory')
    design=load(artifact/'design.json');cases=extract(artifact,context,duration_s)
    if not cases:raise ValueError('No durably mirrored, audited observations for this context')
    if len({c['proof']['source_sha256']for c in cases})!=1:raise ValueError('Do not mix source versions')
    lookup={(c['seed'],c['arm']):c for c in cases}
    if len(lookup)!=len(cases):raise ValueError('Repeated seed/arm is not independent replication; select one preregistered study')
    seeds=sorted({c['seed']for c in cases});metadata=context_metadata(artifact,context,cases)
    module=Path(__file__).resolve();code_sha256s={p.name:sha(p)for p in [module,module.with_name('grid_identify.py'),module.with_name('identify.py'),module.with_name('contracts.py'),module.with_name('population.py')]}
    model_pins={}
    for worker in {c['origin']['worker_id']for c in cases}:
        path=artifact/'workers'/worker/'plan.json'
        if not path.exists():continue  # Synthetic estimator fixtures have no deployed worker.
        pins=load(path)['frozen_models']
        if model_pins and pins!=model_pins:raise ValueError('Workers used different released model pins')
        model_pins=pins
    if any(sha(module.parents[1]/name)!=digest for name,digest in model_pins.items()):
        raise ValueError('Local graph models differ from the frozen experimental release')
    dataset=dict(schema='oran.context-measurements.v1',metric_scope=SCOPE,scenario=metadata,seeds=seeds,cases=cases,
        analysis_code_sha256s=code_sha256s,frozen_model_sha256s=model_pins,
        independent_unit='seed_block_within_context',
        notes=['Individual context/seed blocks stay on one hardware environment.',
            'Every fixed control vector is verified against executed per-slot state and real delayed control receipts.',
            'Power uses scored counter differences / elapsed time; task latency uses matched mature birth cohorts.',
            'Missing completed tasks remain unavailable; all-zero observed failures describe this regime only.',
            'Traffic contexts and observation horizons are never pooled into one coefficient.'])
    output.mkdir(parents=True,exist_ok=True);write(output/'measurement-dataset.json',dataset);digest=sha(output/'measurement-dataset.json')
    estimates=[];unavailable=[];not_tested_designs=[]
    for grid in design['grids']:
        if grid.get('contexts')and context not in grid['contexts']:
            not_tested_designs.append(dict(contrast_id=grid['id'],parameter=grid['parameter'],reason='supplemental_component_not_planned_in_this_context',planned_contexts=grid['contexts']))
            continue
        complete=[s for s in seeds if all((s,a)in lookup for a in grid['arms'])]
        if len(complete)<minimum_seeds:
            unavailable.extend(dict(contrast_id=grid['id'],parameter=grid['parameter'],kpi=k,reason='insufficient_independent_complete_seed_blocks',available_seeds=complete)for k in KPIS)
            continue
        for k in KPIS:
            row=fit_grid(grid,complete,lookup,k,digest)
            if row is None:unavailable.append(dict(contrast_id=grid['id'],parameter=grid['parameter'],kpi=k,reason='fewer_than_four_complete_numeric_levels'));continue
            row['members']=grid.get('members')or design['specs']['native']['groups'][{'mec_workload':'mec','mec_priority':'mec','urllc_priority':'urllc','drx_enabled_population':'drx'}[grid['component']]]
            row['reference_scheduler_mode']=grid['mode']
            if grid.get('stimulus_protocol'):row['stimulus_protocol']=grid['stimulus_protocol']
            estimates.append(row)
    for contrast in design['contrasts']:
        complete=[s for s in seeds if all((s,contrast[a])in lookup for a in('low','high','reference'))]
        if len(complete)<minimum_seeds:
            unavailable.extend(dict(contrast_id=contrast['id'],parameter=contrast['parameter'],kpi=k,reason='insufficient_independent_paired_seed_blocks',available_seeds=complete)for k in KPIS)
            continue
        # Structured joint profiles deliberately keep their full from/to map.
        cp=dict(population=True,seeds=complete,contrasts=[contrast],switch_ms=100,scoring_start_ms=metadata['scoring_start_ms'],scoring_end_ms=metadata['scoring_end_ms'])
        rows=fit_contrasts(cp,[c for c in cases if c['seed']in complete],digest,population=True)
        for row in rows:
            row['parameter_levels']=dict(low=contrast['from'],reference=contrast['from'],high=contrast['to'])
            row['members']=design['specs']['native']['groups']['drx']if contrast['parameter']=='ue_drx_configs'else['cell_0','cell_1','cell_2']
        estimates.extend(rows)
        have={r['kpi']for r in rows}
        unavailable.extend(dict(contrast_id=contrast['id'],parameter=contrast['parameter'],kpi=k,reason='missing_outcome_in_paired_conditions')for k in KPIS if k not in have)
    for row in estimates:
        row.update(scenario_id=context,scenario=metadata,horizon_ms=metadata['scoring_end_ms']-100,
            scoring_start_ms=metadata['scoring_start_ms'],scoring_end_ms=metadata['scoring_end_ms'],
            scoring_window_ms=metadata['scoring_end_ms']-metadata['scoring_start_ms'],importance=importance(row))
        row['metric_window']=metadata['metric_windows'][row['kpi']]
        row['identified_control_basis']='explicit_member_component'if row.get('parameter_component')=='throughput_priority'else'common_scoped_population_direction'
        row['context']=metadata['description']+'; '+row['context']
        row['interpretation']='Signed first-order influence weight conditional on this scenario and scoring window, not a prediction of the deployed KPI or a global causal law.'
    matrix=dict(schema='oran.intervention-edges.v1',metric_scope=SCOPE,scenario=metadata,estimates=estimates,unavailable=unavailable,not_tested_designs=not_tested_designs,
        analysis_code_sha256s=code_sha256s,frozen_model_sha256s=model_pins,
        measurement_dataset_sha256=digest,study='Multi-context independently seed-blocked fixed interventions',
        multiple_testing='Exploratory per-link intervals; no familywise significance claim')
    write(output/'all-effects.json',matrix)
    # Publish every measured parameter/KPI relationship, not only favorable,
    # statistically resolved or predeclared contract edges.
    write(output/'influences.json',matrix)
    validate_influences(graph_view(build_graph()),output/'influences.json')
    fields=['scenario','parameter','component','members','reference_scheduler_mode','kpi','contrast','kind','estimate','ci95_low','ci95_high','unit','seeds','signed_importance','r2','assessment','metric_window_kind','scoring_window_ms']
    with(output/'response-coefficients.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in estimates:
            ci=r.get('confidence_interval')or[None,None];d=r.get('fit_diagnostics',{})
            writer.writerow(dict(scenario=context,parameter=r['parameter'],component=r.get('scalar_component')or r['parameter_component'],members='|'.join(r['members']),reference_scheduler_mode=r.get('reference_scheduler_mode'),kpi=r['kpi'],contrast=r['contrast_id'],kind=r['kind'],
                estimate=r['estimate'],ci95_low=ci[0],ci95_high=ci[1],unit=r['unit'],seeds=r['sample_count'],signed_importance=r['importance']['signed_reference_importance'],r2=d.get('within_seed_r2'),assessment=d.get('assessment'),metric_window_kind=r['metric_window']['kind'],scoring_window_ms=r['scoring_window_ms']))
    with(output/'measurements.csv').open('w',newline='')as stream:
        fields=['scenario','case','seed','arm','worker']+list(KPIS);writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for c in cases:writer.writerow(dict(scenario=context,case=c['name'],seed=c['seed'],arm=c['arm'],worker=c['origin']['worker_id'],**{k:c['metrics'].get(k)for k in KPIS}))
    coverage=defaultdict(set)
    for r in estimates:coverage[r['parameter']].add(r['kpi'])
    native=[c for c in cases if c['arm']=='native']
    report=dict(scenario=context,observations=len(cases),seeds=seeds,complete_seed_blocks=[s for s in seeds if all((s,a)in lookup for a in design['specs'])],source_sha256s=metadata['source_sha256s'],
        parameter_kpi_links=sum(len(v)for v in coverage.values()),estimated_responses=len(estimates),unavailable_responses=len(unavailable),
        signed_resolved=sum(bool((r.get('confidence_interval')or[0,0])[0]>0 or(r.get('confidence_interval')or[0,0])[1]<0)for r in estimates),
        constant_observed=sum(r.get('fit_diagnostics',{}).get('assessment')=='constant_response_observed'for r in estimates),not_tested_designs=not_tested_designs,
        coverage={p:dict(measured_kpis=sorted(v),count=len(v),total_kpis=len(KPIS))for p,v in coverage.items()},
        baseline_urllc=[dict(seed=c['seed'],events_per_s=c['metrics']['urllc_failure_events_per_s'],failure_fraction=c['metrics']['urllc_failure_fraction'])for c in native],
        measurement_dataset_sha256=digest,models_trained=False,xapp_models_trained=False,influence_weights_fitted=True,
        frozen_models_locally_verified=bool(model_pins),raw_data_retained=True,graph_validation_passed=True)
    write(output/'identification-report.json',report)
    return report


def pilot_report(artifact):
    cases=extract(artifact);rows=[]
    for c in cases:
        if c['stage']!='pilot':continue
        rows.append(dict(context=c['context'],seed=c['seed'],arm=c['arm'],worker=c['origin']['worker_id'],
            goodput_mbps=c['metrics']['network_goodput_mbps'],urllc_events_per_s=c['metrics']['urllc_failure_events_per_s'],
            urllc_failure_fraction=c['metrics']['urllc_failure_fraction'],urllc_delay_ms=c['metrics']['urllc_packet_latency_ms'],
            mec_delay_ms=c['metrics']['mec_offloaded_latency_ms'],wall_s=c['performance'].get('wall_s')))
    return dict(schema='oran.context-pilot-report.v1',audited_cases=len(rows),cases=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--context');p.add_argument('--output',type=Path);p.add_argument('--pilot',action='store_true');p.add_argument('--duration-s',type=float,default=1.8);a=p.parse_args()
    if a.pilot:
        result=pilot_report(a.artifact);write(a.output or a.artifact/'pilot-report.json',result)
    else:
        if not a.context or not a.output:p.error('--context and --output required')
        result=analyze(a.artifact,a.context,a.output,duration_s=a.duration_s)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
