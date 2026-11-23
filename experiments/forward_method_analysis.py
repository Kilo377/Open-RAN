"""Fixed future-only subset views and independent five-method comparisons.

Subset views copy frozen predictions, never recompute them. Eligibility comes
only from the two immutable remote request scans; no AFTER value chooses a
case, intent, scene, candidate flag or method.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path

from experiments import study_quantification as study
from experiments.aggregation_aware_study import METHOD,addendum,ref,pinned
from experiments.prospective_request_inventory import paired_eligibility

OLD_METHODS=('goal_obstruction_v2','net_intent_obstruction_v1','temporal_net_intent_obstruction_v1','net_profile_gain_obstruction_v1')
VIEW_SCHEMA='oran.frozen-prediction-subset-view.v1'


def _analysis_plan(reference,value):
    plan=pinned(reference)
    expected=[('joint_five_matched',list(OLD_METHODS)+[METHOD]),
        ('profileG_pairwise_ablation',['net_profile_gain_obstruction_v1',METHOD])]
    if (plan.get('schema')!='oran.aggregation-forward-analysis-plan.v1'
        or study.digest({k:v for k,v in plan.items()if k!='plan_sha256'})!=plan.get('plan_sha256')
        or [(c.get('id'),c.get('methods'))for c in plan.get('comparisons',[])]!=expected
        or plan.get('partitions')!=['original_matrix','prospective_supplement','combined_diagnostic']
        or plan.get('addendum')!=value['seal']['addendum']or plan.get('before_ledger')!=value['seal']['ledger']
        or plan.get('after_KPIs_used_to_choose_analysis')is not False or plan.get('new_physical_runs')!=0):
        raise ValueError('both_preregistered_forward_comparisons_required')
    return plan


def protocol(path):
    seal=study.read(path)
    if seal.get('schema')!='oran.aggregation-forward-score-seal.v1'or seal.get('method')!=METHOD or seal.get('after_kpis_used_for_scoring')is not False:
        raise ValueError('independent_forward_fifth_method_seal_required')
    ledger=pinned(seal['ledger']);code=pinned(seal['code_freeze']);plan=pinned(seal['addendum']);addendum(seal['addendum']['path'])
    if code['ledger']!=seal['ledger']or code['code_bundle']!=seal['code_bundle']or code['addendum']!=seal['addendum']:
        raise ValueError('forward_code_ledger_addendum_binding_differs')
    if study.file_sha(seal['code_bundle']['path'])!=seal['code_bundle']['sha256']:raise ValueError('forward_frozen_bundle_changed')
    for component in code['components'].values():
        if study.file_sha(component['path'])!=component['sha256']:raise ValueError('forward_component_archive_changed')
    eligibility=pinned(seal['forward_eligibility'])
    repeated=paired_eligibility(seal['scan_before'],seal['scan_after'],code_bundle=seal['code_bundle'],
        scoring_started_unix_s=seal['scoring_started_unix_s'],scoring_finished_unix_s=seal['scoring_finished_unix_s'])
    if repeated!=eligibility:raise ValueError('forward_cohort_differs_from_two_remote_scans')
    cases={c['case_id']:c for c in ledger['cases']};jobs={j['job_id']:j for j in ledger['jobs']}
    cohort=set(eligibility['eligible_job_ids'])
    if (len(cases)!=1980 or len(jobs)!=132 or not cohort<=set(jobs)
        or seal.get('all_case_count')!=1980 or len(seal['blocks'])!=12):raise ValueError('forward_ledger_grid_differs')
    for old in seal['original_four_sets'].values():pinned(old)
    if datetime.fromisoformat(code['created_utc']).timestamp()>study.read(seal['scan_before']['path'])['local_started_unix_s']:
        raise ValueError('forward_code_was_not_frozen_before_remote_scan')
    return dict(seal=seal,ledger=ledger,code=code,eligibility=eligibility,cases=cases,jobs=jobs,cohort=cohort,method_seal=ref(path))


def fifth_parent(spec,verified_protocol=None):
    value=verified_protocol or protocol(spec['seal_path']);seal=value['seal'];ledger=value['ledger']
    selected=[b for b in seal['blocks']if Path(b['score']['path']).resolve()==Path(spec['score_path']).resolve()]
    if len(selected)!=1:raise ValueError('fifth_parent_score_not_in_global_method_seal')
    block=selected[0];data=pinned(block['input']);score=pinned(block['score']);sources=pinned(block['label_sources'])['sources']
    original=next(b for b in ledger['blocks']if b['id']==block['block_id']);parent_data=pinned(original['profile_input'])
    if data['graphs']!=parent_data['graphs']:raise ValueError('fifth_method_changed_profileG')
    before={c['case_id']:study.materialize_case(parent_data,c)for c in parent_data['cases']}
    after={c['case_id']:study.materialize_case(data,c)for c in data['cases']}
    if before!=after or set(after)!=set(sources):raise ValueError('fifth_before_geometry_or_source_grid_changed')
    if score.get('graph_variant')or score.get('method')!=METHOD or score.get('stage')!='scores_before_label_join'or score.get('after_values_used')is not False:
        raise ValueError('fifth_method_identity_or_before_only_protocol_changed')
    if score.get('input_sha256')!=block['input']['sha256']or score.get('label_sources_sha256')!=block['label_sources']['sha256']:
        raise ValueError('fifth_parent_input_source_pins_differ')
    if {r['case_id']for r in score['scores']}!=set(after)or len(score['scores'])!=len(after):raise ValueError('fifth_score_case_grid_changed')
    for name,pin in value['code']['unchanged_parent_app_goal_sha256'].items():
        if score['module_sha256'].get(name)!=pin:raise ValueError('fifth_projection_release_changed')
    if score['module_sha256'].get(seal['code_bundle']['path'])!=seal['code_bundle']['sha256']:
        raise ValueError('fifth_math_bundle_release_changed')
    for name,pin in score['module_sha256'].items():
        if study.file_sha(score['archived_modules'][name])!=pin:raise ValueError('fifth_archived_engine_changed')
    return dict(score=score,data=data,sources=sources,input_ref=block['input'],source_ref=block['label_sources'],
        method_seal=value['method_seal'],protocol=value,block=block)


def _parent(spec,verified_protocol=None):
    seal=study.read(spec['seal_path'])
    if seal.get('schema')=='oran.aggregation-forward-score-seal.v1':return fifth_parent(spec,verified_protocol)
    from experiments.study_evaluation_cache import verify_prediction
    verified=verify_prediction(spec);score=verified['scores'];inp=Path(score['input_path'])
    return dict(score=score,data=study.read(inp),sources=verified['sources'],input_ref=ref(inp),source_ref=ref(inp.with_name('label-sources.json')))


def write_subset_view(parent_spec,method_seal_path,output,*,verified_protocol=None):
    value=verified_protocol or protocol(method_seal_path);parent=_parent(parent_spec,value);data=deepcopy(parent['data']);original=parent['score']
    ids={c['case_id']for c in data['cases']if c['provenance']['job_id']in value['cohort']}
    if not ids:return None
    data['cases']=[c for c in data['cases']if c['case_id']in ids]
    output=Path(output);inp=output/'scoring-inputs.json';source_path=output/'label-sources.json';score_path=output/'scores-view.json'
    binding=dict(parent_score=ref(parent_spec['score_path']),parent_seal=ref(parent_spec['seal_path']),forward_method_seal=value['method_seal'],
        selected_case_ids=sorted(ids),new_predictions_computed=False,candidate_flags_changed=False,after_KPI_values_used_for_filtering=False)
    data['frozen_subset_view']=binding
    study.write_immutable(inp,data)
    source_payload=study.read(parent['source_ref']['path']);source_payload['sources']={k:v for k,v in source_payload['sources'].items()if k in ids}
    study.write_immutable(source_path,source_payload)
    score=deepcopy(original);score.update(input_path=str(inp),input_sha256=study.file_sha(inp),label_sources_sha256=study.file_sha(source_path),
        scores=[r for r in original['scores']if r['case_id']in ids],frozen_subset_view=binding)
    study.write_immutable(score_path,score)
    view=dict(schema=VIEW_SCHEMA,method=original['method'],created_utc=datetime.now(timezone.utc).isoformat(),parent_spec=deepcopy(parent_spec),
        parent_score=ref(parent_spec['score_path']),parent_seal=ref(parent_spec['seal_path']),forward_method_seal=value['method_seal'],
        input=ref(inp),label_sources=ref(source_path),score=ref(score_path),case_ids=sorted(ids),
        new_predictions_computed=False,after_KPI_values_used_for_filtering=False,original_candidates_or_scores_changed=False)
    path=output/'subset-view.json';study.write_immutable(path,view)
    return dict(method=score['method'],score_path=str(score_path),seal_path=str(path),score_sha256=study.file_sha(score_path),seal_sha256=study.file_sha(path),cases=len(ids))


def verify_subset_view(spec):
    """Return ordinary verified-score metadata without opening any run data."""
    view=study.read(spec['seal_path'])
    if (view.get('schema')!=VIEW_SCHEMA or view.get('new_predictions_computed')is not False
        or view.get('after_KPI_values_used_for_filtering')is not False):raise ValueError('frozen_forward_subset_view_required')
    if view['parent_score']!=ref(view['parent_spec']['score_path'])or view['parent_seal']!=ref(view['parent_spec']['seal_path']):
        raise ValueError('subset_parent_prediction_changed')
    value=protocol(view['forward_method_seal']['path'])
    if value['method_seal']!=view['forward_method_seal']:raise ValueError('subset_forward_protocol_changed')
    parent=_parent(view['parent_spec'],value);data=pinned(view['input']);score=pinned(view['score']);sources=pinned(view['label_sources'])['sources']
    expected={c['case_id']for c in parent['data']['cases']if c['provenance']['job_id']in value['cohort']}
    if set(view['case_ids'])!=expected:raise ValueError('subset_cherry_picked_future_cases_or_included_started_jobs')
    if data['cases']!=[c for c in parent['data']['cases']if c['case_id']in expected]:raise ValueError('subset_input_cases_changed')
    a={k:v for k,v in data.items()if k not in('cases','frozen_subset_view')};b={k:v for k,v in parent['data'].items()if k!='cases'}
    if a!=b:raise ValueError('subset_changed_frozen_graph_context_or_options')
    if sources!={k:v for k,v in parent['sources'].items()if k in expected}:raise ValueError('subset_actual_label_paths_ranges_or_units_changed')
    if score['scores']!=[r for r in parent['score']['scores']if r['case_id']in expected]:raise ValueError('subset_recomputed_or_changed_frozen_risks')
    before={k:v for k,v in score.items()if k not in('input_path','input_sha256','label_sources_sha256','scores','frozen_subset_view')}
    original={k:v for k,v in parent['score'].items()if k not in('input_path','input_sha256','label_sources_sha256','scores')}
    if before!=original:raise ValueError('subset_changed_frozen_engine_or_method_metadata')
    if (Path(spec['score_path']).resolve()!=Path(view['score']['path']).resolve()or score['input_sha256']!=view['input']['sha256']
        or score['label_sources_sha256']!=view['label_sources']['sha256']):raise ValueError('subset_score_input_pin_differs')
    from experiments.study_evaluation_cache import geometry_signature
    cases={c['case_id']:study.materialize_case(data,c)for c in data['cases']};apps={c['provenance']['job_id']:sorted(c['app_ids'])for c in cases.values()}
    return dict(spec=deepcopy(spec),scores=score,sources=sources,source_path=Path(view['label_sources']['path']),source_sha256=view['label_sources']['sha256'],
        geometry_sha256=geometry_signature(cases),case_ids=expected,planned_apps=apps)


def prepare_views(method_seal_path,analysis_plan_path,output):
    value=protocol(method_seal_path);_analysis_plan(ref(analysis_plan_path),value);output=Path(output)
    analysis=pinned(value['ledger']['analysis_manifest']);blocks=[]
    for block in analysis['blocks']:
        newer=next(b for b in value['seal']['blocks']if b['block_id']==block['id'])
        specs=[dict(score_path=s['score_path'],seal_path=s['seal_path'])for s in block['methods']]
        specs.append(dict(score_path=newer['score']['path'],seal_path=str(method_seal_path)))
        views=[]
        for spec in specs:
            method=study.read(spec['score_path'])['method']
            view=write_subset_view(spec,method_seal_path,output/'views'/block['partition']/block['origin_key']/method,verified_protocol=value)
            if view:views.append(view)
        if views:blocks.append(dict(id=block['id'],partition=block['partition'],origin_key=block['origin_key'],methods=views))
    declared=sorted({'|'.join((c['intent_id'],c['scene'],str(c['seed'])))for c in value['ledger']['cases']})
    result=dict(schema='oran.forward-five-analysis-inputs.v1',created_utc=datetime.now(timezone.utc).isoformat(),method_seal=value['method_seal'],
        analysis_plan=ref(analysis_plan_path),blocks=blocks,declared_strata=declared,eligible_job_ids=sorted(value['cohort']),
        eligible_physical_jobs=len(value['cohort']),diagnostic_physical_jobs=132-len(value['cohort']),future_case_count=sum(c['job_id']in value['cohort']for c in value['ledger']['cases']),
        original_four_full_reports_unchanged=True,after_KPI_values_used_for_view_selection=False)
    study.write_immutable(output/'forward-analysis-inputs.json',result);return result


def run_forward_analysis(manifest_path,output,cache_dir,*,interim=False,check_only=False):
    from experiments.study_evaluation_cache import verify_prediction,completed_physical_proofs,_physical_key,join_with_label_reuse
    from experiments.report_quantification_study import build_report
    from experiments.report_study_guide import write_landing_page
    from experiments.report_execution_provenance import source_refs
    manifest=study.read(manifest_path);value=protocol(manifest['method_seal']['path']);plan=_analysis_plan(manifest['analysis_plan'],value)
    if (manifest.get('schema')!='oran.forward-five-analysis-inputs.v1'or value['method_seal']!=manifest['method_seal']
        or set(manifest['eligible_job_ids'])!=value['cohort']):raise ValueError('fixed_forward_analysis_manifest_required')
    evidence_path=Path(manifest['method_seal']['path']).with_name('evidence-binding.json');evidence=None
    if evidence_path.exists():
        evidence=study.read(evidence_path)
        if evidence.get('method_seal')!=manifest['method_seal']or evidence.get('local_forward_analysis')!=manifest['analysis_plan']:
            raise ValueError('forward_evidence_binding_differs')
        for name in('root_forward_analysis','runtime','driver_source'):
            if study.file_sha(evidence[name]['path'])!=evidence[name]['sha256']:raise ValueError('forward_evidence_archive_changed')
    seen=set();ready=set()
    for block in manifest['blocks']:
        if [s['method']for s in block['methods']]!=list(OLD_METHODS)+[METHOD]:raise ValueError('all_five_future_views_required')
        verified=[verify_prediction(spec)for spec in block['methods']];first=verified[0]
        if any(x['source_sha256']!=first['source_sha256']or x['geometry_sha256']!=first['geometry_sha256']or x['case_ids']!=first['case_ids']for x in verified):
            raise ValueError('forward_methods_do_not_share_the_identical_future_scope')
        proofs,_=completed_physical_proofs(first['sources'],first['planned_apps'])
        for source in first['sources'].values():
            job=source['provenance']['job_id']
            if job not in value['cohort']:raise ValueError('started_diagnostic_job_entered_forward_validation')
            seen.add(job)
            if proofs[_physical_key(source)]['ready']:ready.add(job)
    if seen!=value['cohort']:raise ValueError('forward_view_dropped_or_added_physical_jobs')
    status=dict(eligible_physical_jobs=len(seen),verified_physical_jobs=len(ready),pending_job_ids=sorted(seen-ready),
        diagnostic_jobs_excluded=132-len(seen),future_intent_cases=manifest['future_case_count'],
        unique_seed_count=len({value['jobs'][j]['seed']for j in seen}),scene_seed_blocks=len({(value['jobs'][j]['scene'],value['jobs'][j]['seed'])for j in seen}))
    if check_only or(not interim and ready!=seen):return dict(stage='ready_for_forward_report'if ready==seen else'waiting_for_forward_physical_cohort',output_written=False,status=status)
    output=Path(output)
    if output.exists():raise ValueError('forward_analysis_requires_a_new_output_directory')
    output.mkdir(parents=True);mode='interim'if interim else'final';joined=[]
    for block in manifest['blocks']:
        result=join_with_label_reuse(block['methods'],output/'evaluations'/block['partition']/block['origin_key'],cache_dir,refresh_id=mode)
        joined.append(dict(block=block,result=result))
    report_refs=[]
    partitions=[('original_matrix',{'original_matrix'}),('prospective_supplement',{'prospective_supplement'}),
        ('combined_diagnostic',{'original_matrix','prospective_supplement'})]
    for comparison in plan['comparisons']:
        for partition,included in partitions:
            selected=[j for j in joined if j['block']['partition']in included]
            if not selected:continue
            methods=[]
            for method in comparison['methods']:
                entries=[]
                for entry in selected:
                    spec=next(s for s in entry['block']['methods']if s['method']==method)
                    evaluation=next(s for s in entry['result']['outputs']if s['method']==method)
                    entries.append(dict(score_path=spec['score_path'],seal_path=spec['seal_path'],evaluation_path=evaluation['path']))
                methods.append(dict(id=method,blocks=entries))
            report_manifest=dict(schema='oran.quantification-report-inputs.v1',methods=methods,analysis_status=mode,
                study_partition=partition,title='Forward validation · '+comparison['id'].replace('_',' ')+' · '+partition.replace('_',' '),
                population_note=f'Only the fixed {len(seen)}-job future cohort absent in BOTH remote scans. The fifth method was registered after original outcome availability; independent validation therefore excludes all previously/during-started jobs. Original four-method full-matrix reports remain separate and unchanged. '+comparison['rule'],
                status_note=f'{mode.upper()} forward analysis: {len(ready)}/{len(seen)} eligible physical jobs verified; {132-len(seen)} previously/during-started jobs are diagnostic only. No new independent seeds.',
                declared_strata=manifest['declared_strata'],
                forward_protocol=dict(method_seal=manifest['method_seal'],analysis_plan=manifest['analysis_plan'],addendum=value['seal']['addendum'],
                    scan_before=value['seal']['scan_before'],scan_after=value['seal']['scan_after'],code_bundle=value['seal']['code_bundle'],
                    forward_eligibility=value['seal']['forward_eligibility'],scope=status,comparison_id=comparison['id']))
            if evidence is not None:
                report_manifest['forward_protocol']['evidence_binding']=ref(evidence_path)
                report_manifest['forward_protocol']['root_forward_analysis']=evidence['root_forward_analysis']
            # Use the same BEFORE-only applicability inventory, never an AFTER
            # screen. It remains explicitly labelled as the originalG inventory.
            analysis=pinned(value['ledger']['analysis_manifest'])
            report_manifest['before_action_coverage_inventory']=analysis['before_action_coverage_inventory']['path']
            report_manifest['execution_label_sources']=source_refs(methods)
            destination=output/'reports'/comparison['id']/partition;path=output/'report-manifests'/(comparison['id']+'-'+partition+'.json')
            study.write_immutable(path,report_manifest);build_report(path,destination)
            report_refs.append(dict(comparison=comparison['id'],partition=partition,html=str(destination/'index.html'),summary=ref(destination/'report-summary.json')))
    completion=dict(schema='oran.forward-analysis-completion.v1',analysis_status=mode,method_seal=manifest['method_seal'],analysis_plan=manifest['analysis_plan'],
        status=status,reports=report_refs,original_four_reports_unchanged=True,cohort_selected_without_after_KPIs=True,
        joint_five_and_sameG_pairwise_both_reported=True,finished_utc=datetime.now(timezone.utc).isoformat())
    study.write_immutable(output/'forward-analysis-completion.json',completion)
    write_landing_page(output,title='Forward validation',status=mode.upper()+' · '+str(len(ready))+'/'+str(len(seen))+
        ' eligible physical jobs verified · 2 independent seeds, 6 scene × seed blocks',pages=report_refs,forward=True)
    return completion


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare-views');prep.add_argument('--method-seal',type=Path,required=True);prep.add_argument('--analysis-plan',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    run=commands.add_parser('run');run.add_argument('--manifest',type=Path,required=True);run.add_argument('--output',type=Path,required=True);run.add_argument('--cache',type=Path,required=True)
    run.add_argument('--interim',action='store_true');run.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare-views':
        value=prepare_views(args.method_seal,args.analysis_plan,args.output);print(json.dumps(dict(eligible_jobs=value['eligible_physical_jobs'],cases=value['future_case_count'],manifest=str(args.output/'forward-analysis-inputs.json'))));return
    result=run_forward_analysis(args.manifest,args.output,args.cache,interim=args.interim,check_only=args.check_only);print(json.dumps(result))
    if result.get('stage')=='waiting_for_forward_physical_cohort'and not args.check_only:raise SystemExit(2)


if __name__=='__main__':main()
