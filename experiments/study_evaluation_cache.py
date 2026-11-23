"""Reuse independent actual labels across frozen methods, never risks.

The unchanged full Evaluation join remains the canonical label producer.
Persistent caches contain only labels from completed, checksum-verified runs;
pending states are shared only within one refresh and are never cached.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import threading

from experiments import study_quantification as study

RISK_FIELDS=('risk','raw_risk','risk_status','risk_reason','reference_effect','reference_effect_score')
WINDOW_IO_LOCK=threading.RLock()
CODE_FILES=('experiments/study_quantification.py','smo/result_evaluation.py','smo/intent_quality.py',
    'smo/window_metrics.py','smo/runtime.py','smo/population.py','smo/contracts.py')


def evaluation_code_pins():
    return {name:study.file_sha(study.ROOT/name)for name in CODE_FILES}|{'label_reuse_helper':study.file_sha(__file__)}


def geometry_signature(cases):
    geometry={}
    for ident,case in cases.items():
        geometry[ident]={k:deepcopy(case.get(k))for k in('case_id','intent_id','group_id','app_ids','ranges','provenance',
            'candidate','protocol','timing','frame','preview','cohort_id','independence_group','intent_resolution')}
        geometry[ident]['normalization']=deepcopy(case.get('options',{}).get('normalization'))
    return study.digest(geometry)


def verify_prediction(spec):
    score_path=Path(spec['score_path']);seal=study.read(spec['seal_path']);scores=study.read(score_path)
    if seal.get('schema')=='oran.frozen-prediction-subset-view.v1':
        from experiments.forward_method_analysis import verify_subset_view
        return verify_subset_view(spec)
    if (scores.get('stage')!='scores_before_label_join'or scores.get('after_values_used')is not False
        or seal.get('after_values_used')is not False or seal.get('nonempty_arm_started_before_scoring')is not False
        or seal.get('algorithm_tuned_to_test_labels')is not False):raise ValueError('frozen_predeployment_prediction_required')
    if seal.get('schema')not in('oran.prospective-score-seal.v1','oran.prospective-supplement-score-seal.v1'):
        raise ValueError('independent_prediction_seal_required')
    method=scores['method'];ref=seal.get('scores',{}).get(method,{})
    if ref.get('sha256')!=study.file_sha(score_path)or Path(ref.get('path','')).resolve()!=score_path.resolve():
        raise ValueError('frozen_score_pin_changed')
    expected=(seal.get('method_inputs',{}).get(method,{})if seal['schema']=='oran.prospective-supplement-score-seal.v1'else
        dict(path=seal['input_path'],sha256=seal['input_sha256']))
    inp=Path(scores['input_path'])
    if expected.get('sha256')!=scores['input_sha256']or study.file_sha(inp)!=scores['input_sha256']or Path(expected.get('path','')).resolve()!=inp.resolve():
        raise ValueError('frozen_input_pin_changed')
    inputs=study.read(inp);by_id={c['case_id']:study.materialize_case(inputs,c)for c in inputs['cases']}
    ids=[r['case_id']for r in scores['scores']]
    if (inputs.get('after_values_included')is not False or len(ids)!=len(set(ids))or set(ids)!=set(by_id)
        or ref.get('count')!=len(ids)):raise ValueError('frozen_case_grid_changed')
    if not scores.get('module_sha256'):raise ValueError('frozen_engine_archives_required')
    for source,pin in scores['module_sha256'].items():
        if study.file_sha(scores['archived_modules'][source])!=pin:raise ValueError('frozen_engine_archive_changed')
    source_path=inp.with_name('label-sources.json');source_pin=study.file_sha(source_path)
    if scores.get('label_sources_sha256')!=source_pin:raise ValueError('frozen_label_sources_changed')
    sources=study.read(source_path)['sources']
    if set(sources)!=set(ids):raise ValueError('label_source_case_grid_changed')
    geometry={}
    by_score={r['case_id']:r for r in scores['scores']};planned_apps={}
    for ident,case in by_id.items():
        source=sources[ident]
        if any(source.get(k)!=case.get(k)for k in('ranges','provenance','candidate','protocol','intent_id','cohort_id','independence_group')):
            raise ValueError('score_input_and_label_source_geometry_differ')
        if any(by_score[ident].get(k)!=case.get(k)for k in('intent_id','provenance','candidate','protocol','cohort_id','independence_group','scene')):
            raise ValueError('frozen_score_and_input_metadata_differ')
        groups=by_score[ident].get('result',{}).get('groups')or[]
        if groups and(len(groups)!=1 or groups[0].get('group_id')!=case['group_id']or groups[0].get('app_ids')!=case['app_ids']):
            raise ValueError('frozen_score_result_identifies_another_control_arm')
        job_id=source['provenance']['job_id'];apps=sorted(case['app_ids'])
        if job_id in planned_apps and planned_apps[job_id]!=apps:raise ValueError('one_actual_run_assigned_to_different_control_arms')
        planned_apps[job_id]=apps
        geometry[ident]={k:deepcopy(case.get(k))for k in('case_id','intent_id','group_id','app_ids','ranges','provenance',
            'candidate','protocol','timing','frame','preview','cohort_id','independence_group','intent_resolution')}
        geometry[ident]['normalization']=deepcopy(case.get('options',{}).get('normalization'))
    return dict(spec=deepcopy(spec),scores=scores,sources=sources,source_path=source_path,source_sha256=source_pin,
        geometry_sha256=geometry_signature(by_id),case_ids=set(ids),planned_apps=planned_apps)


def completed_physical_proofs(sources,planned_apps):
    """Revalidate real completion and raw/config/checkpoint pins every refresh."""
    proofs={};all_ready=True
    for source in sources.values():
        ident=study.digest({k:source.get(k)for k in('run_path','seal_path','expected_origin_id','requires_actual_origin_guard')})
        if ident in proofs:continue
        if not source.get('seal_path'):raise ValueError('label_reuse_requires_completed_physical_checkpoint')
        ready,reason=study._actual_arm_ready(source)
        if not ready:
            if reason!='awaiting_completed_sealed_run':raise ValueError('certified_physical_data_changed:'+str(reason))
            all_ready=False;proofs[ident]=dict(ready=False,reason=reason);continue
        seal_path=Path(source['seal_path']);seal=study.read(seal_path);run=Path(source['run_path']);config=study.read(run/'config_snapshot.json')
        status=study.read(run/'run_status.json');pins=seal.get('files_sha256',{});required={'data/run/config_snapshot.json','data/run/run_status.json'}
        trace={p for p in pins if p in('data/run/rt_sys_trace.jsonl','data/run/rt_sys_trace.jsonl.gz')}
        if source.get('requires_actual_origin_guard'):required.add('data/run/evaluation_origin_verified.json')
        if (not required<=pins.keys()or len(trace)!=1 or seal.get('case',{}).get('id')!=source['provenance']['job_id']
            or seal.get('physical_plan_sha256')!=source['provenance']['physical_plan_sha256']
            or seal.get('source_sha256')!=config.get('source_sha256')
            or status.get('completed')is not True or status.get('source_sha256')!=config.get('source_sha256')
            or sorted(seal.get('case',{}).get('app_ids',[]))!=planned_apps[source['provenance']['job_id']]
            or config.get('manifest',{}).get('rng_seed')!=source['provenance']['seed']):
            raise ValueError('physical_checkpoint_lacks_complete_matched_evidence')
        proofs[ident]=dict(ready=True,checkpoint_path=str(seal_path),checkpoint_sha256=study.file_sha(seal_path),
            origin_id=seal['origin_id'],job_id=seal['case']['id'],physical_plan_sha256=seal['physical_plan_sha256'],
            source_sha256=seal['source_sha256'],files_sha256=deepcopy(pins),
            before_window=deepcopy(source['before_window']),after_window=deepcopy(source['after_window']))
    return proofs,all_ready


def _physical_key(source):
    return study.digest({k:source.get(k)for k in('run_path','seal_path','expected_origin_id','requires_actual_origin_guard')})


def _safe_window_pair(pair,source):
    forbidden={'invalid_or_future_window','exact_window_data_unavailable'}
    if not isinstance(pair,(tuple,list))or len(pair)!=2:return False
    keys={'kpi:'+r['kpi']for r in source['ranges']}
    for frame,window in zip(pair,(source['before_window'],source['after_window'])):
        if (not isinstance(frame,dict)or frame.get('slot')!=window['slot']or frame.get('window_start_slot')!=window['window_start_slot']
            or frame.get('window_slots')!=window['slot']-window['window_start_slot']or set(frame.get('states',{}))!=keys):return False
        for metric in frame['states'].values():
            if (metric.get('reason')in forbidden or metric.get('sampled_at_slot')!=window['slot']
                or metric.get('window_start_slot')!=window['window_start_slot']):return False
    return True


def join_with_label_reuse(specs,output_root,cache_dir,*,refresh_id=None):
    """One coherent physical/intent block, several independently frozen risks."""
    if not specs:raise ValueError('at_least_one_frozen_method_required')
    verified=[verify_prediction(spec)for spec in specs];first=verified[0]
    if len({x['scores']['method']for x in verified})!=len(verified):raise ValueError('duplicate_method_in_label_reuse_group')
    if any(x['source_sha256']!=first['source_sha256']or x['geometry_sha256']!=first['geometry_sha256']or x['case_ids']!=first['case_ids']for x in verified):
        raise ValueError('label_reuse_requires_identical_sources_ranges_units_provenance_and_BEFORE_geometry')
    pins=evaluation_code_pins();proofs,all_ready=completed_physical_proofs(first['sources'],first['planned_apps'])
    identity=dict(label_sources_sha256=first['source_sha256'],geometry_sha256=first['geometry_sha256'],physical_proofs=proofs,evaluation_code_sha256=pins)
    key=study.digest(identity)
    refresh_id=refresh_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    if not refresh_id or any(c not in'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.T'for c in refresh_id):raise ValueError('invalid_refresh_id')
    output=Path(output_root)/refresh_id
    if output.exists():raise ValueError('each_label_refresh_requires_a_new_output_version')
    output.mkdir(parents=True);memory={};staged={};cache_refs={};native=[]
    stats=dict(raw_parses=0,memory_hits=0,disk_hits=0,persistent_writes=0,uncertified_bypasses=0)
    # This scoped adapter changes only the observation I/O, not the unchanged
    # Evaluation function. A lock serializes adapters in this process; unknown
    # sources delegate to the original loader. Restoration is unconditional.
    with WINDOW_IO_LOCK:
        original_loader=study._observed_windows
        def observations(source):
            physical=proofs.get(_physical_key(source),{})
            if not physical.get('ready'):
                stats['uncertified_bypasses']+=1;return original_loader(source)
            window_identity=dict(label_sources_sha256=first['source_sha256'],geometry_sha256=first['geometry_sha256'],
                physical_proof=physical,evaluation_code_sha256=pins,before_window=source['before_window'],
                after_window=source['after_window'],kpis=[r['kpi']for r in source['ranges']])
            window_key=study.digest(window_identity);cache_path=Path(cache_dir)/'windows'/(window_key+'.json')
            if window_key in memory:
                stats['memory_hits']+=1;return deepcopy(memory[window_key])
            if cache_path.exists():
                saved=study.read(cache_path);pair=saved['windows']
                if (saved.get('schema')!='oran.actual-window-cache.v1'or saved.get('identity')!=window_identity
                    or saved.get('windows_sha256')!=study.digest(pair)or not _safe_window_pair(pair,source)
                    or any(set(frame)&set(RISK_FIELDS)for frame in pair)):
                    raise ValueError('actual_window_cache_changed')
                stats['disk_hits']+=1
            else:
                pair=original_loader(source);stats['raw_parses']+=1
                if _safe_window_pair(pair,source):
                    staged[window_key]=(cache_path,dict(schema='oran.actual-window-cache.v1',identity=window_identity,windows=pair,
                        windows_sha256=study.digest(pair),physical_run_sealed_complete=True,pending_states_cached=False,risks_cached=False))
            memory[window_key]=deepcopy(pair);cache_refs[window_key]=str(cache_path)
            return deepcopy(pair)
        study._observed_windows=observations
        try:
            for item in verified:
                path=output/'native-full-joins'/('evaluation-'+item['scores']['method']+'.json')
                value=study.join_evaluation(item['spec']['score_path'],item['source_path'],path)
                native.append((item,path,value))
        finally:study._observed_windows=original_loader
        after_proofs,_=completed_physical_proofs(first['sources'],first['planned_apps'])
        if after_proofs!=proofs:raise ValueError('physical_evidence_changed_during_label_join_refresh_again')
        if evaluation_code_pins()!=pins:raise ValueError('evaluation_code_changed_during_label_join')
        for item,_,value in native:
            if study.file_sha(item['spec']['score_path'])!=value['scores_sha256']:
                raise ValueError('frozen_score_changed_during_label_join')
        # Commit only after post-read physical/checksum verification succeeds.
        for cache_path,package in staged.values():study.write_immutable(cache_path,package);stats['persistent_writes']+=1
    hit=stats['raw_parses']==0 and stats['disk_hits']>0;outputs=[]
    for item,native_path,value in native:
        value['label_reuse']=dict(cache_key=key,persistent_cache_hit=hit,pending_states_persisted=False,risks_recomputed_or_reused_from_labels=False,
            evaluation_code_sha256=pins,label_sources_sha256=first['source_sha256'],physical_proofs_sha256=study.digest(proofs),
            native_full_join_path=str(native_path),window_cache_refs=cache_refs,window_cache_stats=stats)
        path=output/('evaluation-'+item['scores']['method']+'.json');study.write_immutable(path,value)
        outputs.append(dict(method=value['method'],path=str(path),sha256=study.file_sha(path)))
    result=dict(schema='oran.actual-label-refresh.v1',refresh_id=refresh_id,outputs=outputs,cache_key=key,persistent_cache_hit=hit,
        persistent_cache_created=stats['persistent_writes']>0,all_physical_runs_sealed_complete=all_ready,window_cache_stats=stats,
        raw_label_producer_unchanged=True,risk_scores_from_frozen_files_only=True,evaluation_code_sha256=pins)
    study.write_immutable(output/'refresh.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True,help='JSON list of {score_path,seal_path}')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--refresh')
    args=parser.parse_args();result=join_with_label_reuse(study.read(args.manifest),args.output,args.cache,refresh_id=args.refresh)
    print(__import__('json').dumps(result))


if __name__=='__main__':main()
