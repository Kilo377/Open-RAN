"""One-command analysis of the fixed120-run study and12-run supplement.

Default final analysis starts only after all132 physical checkpoints pass
checksum/origin verification. --interim creates a separately marked progress
directory. Neither mode changes a formula, gain, candidate or prediction.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path

from experiments import study_quantification as study
from experiments import study_evaluation_cache as reuse
from experiments.report_quantification_study import build_report
from experiments.report_study_guide import write_landing_page
from experiments.report_execution_provenance import source_refs

METHODS=('goal_obstruction_v2','net_intent_obstruction_v1','temporal_net_intent_obstruction_v1','net_profile_gain_obstruction_v1')
PARTITIONS={'original_matrix':120,'prospective_supplement':12}


def _ref(path):
    return dict(path=str(path),sha256=study.file_sha(path))


def _pinned(ref):
    if study.file_sha(ref['path'])!=ref['sha256']:raise ValueError('analysis_evidence_pin_changed:'+str(ref['path']))
    return study.read(ref['path'])


def prepare_manifest(artifact,output):
    artifact=Path(artifact);v2=artifact/'versions/v2'
    set_paths=dict(original_matrix=v2/'predictions-v2/six-block-score-seal.json',profile_variant=v2/'predictions-profile-v1/six-block-score-seal.json',
        prospective_supplement=v2/'supplement-predictions/six-block-score-seal.json')
    sets={k:study.read(path)for k,path in set_paths.items()};blocks=[]
    profile={b['origin_key']:b for b in sets['profile_variant']['blocks']}
    for old in sets['original_matrix']['blocks']:
        key=old['origin_key'];main=_pinned({k:old[k]for k in('path','sha256')});newref=profile[key];new=_pinned({k:newref[k]for k in('path','sha256')})
        specs=[]
        for method in METHODS:
            seal_path=old['path']if method!=METHODS[-1]else newref['path'];parent=main if method!=METHODS[-1]else new
            specs.append(dict(method=method,score_path=parent['scores'][method]['path'],score_sha256=parent['scores'][method]['sha256'],
                seal_path=seal_path,seal_sha256=study.file_sha(seal_path)))
        blocks.append(dict(id='original_matrix:'+key,partition='original_matrix',origin_key=key,methods=specs))
    for block in sets['prospective_supplement']['blocks']:
        seal=_pinned({k:block[k]for k in('path','sha256')})
        specs=[dict(method=m,score_path=seal['scores'][m]['path'],score_sha256=seal['scores'][m]['sha256'],
            seal_path=block['path'],seal_sha256=block['sha256'])for m in METHODS]
        blocks.append(dict(id='prospective_supplement:'+block['origin_key'],partition='prospective_supplement',origin_key=block['origin_key'],methods=specs))
    inventory=artifact/'before-action-coverage-inventory.json'
    manifest=dict(schema='oran.quantification-analysis-plan.v1',created_utc=datetime.now(timezone.utc).isoformat(),source_version='v2',
        artifact_root=str(artifact),expected_physical_runs=PARTITIONS,expected_unique_seeds=[20261201,20261202],expected_scene_seed_blocks=6,
        methods=list(METHODS),source_sets={k:_ref(p)for k,p in set_paths.items()},blocks=blocks,
        before_action_coverage_inventory=_ref(inventory),
        primary_pool_policy='Original20-arm primary results remain separate. Supplement uses its own frozen flags. Combined results are diagnostic; no additional independent seeds.',
        scores_gains_candidates_and_Evaluation_modified=False)
    manifest['manifest_sha256']=study.digest(manifest);study.write_immutable(output,manifest);return manifest


def validate_manifest(path):
    manifest=study.read(path)
    if (manifest.get('schema')!='oran.quantification-analysis-plan.v1'or manifest.get('source_version')!='v2'
        or manifest.get('expected_physical_runs')!=PARTITIONS or manifest.get('methods')!=list(METHODS)
        or study.digest({k:v for k,v in manifest.items()if k!='manifest_sha256'})!=manifest.get('manifest_sha256')):
        raise ValueError('fixed132_run_analysis_manifest_required')
    if Counter(b['partition']for b in manifest['blocks'])!={'original_matrix':6,'prospective_supplement':6}:
        raise ValueError('six_original_and_six_supplement_blocks_required')
    if len({b['id']for b in manifest['blocks']})!=12:raise ValueError('duplicate_analysis_block')
    origins={p:{b['origin_key']for b in manifest['blocks']if b['partition']==p}for p in PARTITIONS}
    if len(origins['original_matrix'])!=6 or origins['original_matrix']!=origins['prospective_supplement']:
        raise ValueError('supplement_must_share_the_original_six_origins')
    for ref in manifest['source_sets'].values():
        if _pinned(ref).get('after_values_used')is not False:raise ValueError('analysis_requires_frozen_prediction_sets')
    _pinned(manifest['before_action_coverage_inventory'])
    return manifest


def preflight(manifest):
    seen=set();counts=Counter();ready=Counter();pending=[];seed_values=set();scene_seed=set();blocks=[]
    for block in manifest['blocks']:
        specs=block['methods']
        if [s['method']for s in specs]!=list(METHODS):raise ValueError('each_analysis_block_requires_all_four_methods')
        for spec in specs:
            if study.file_sha(spec['score_path'])!=spec['score_sha256']or study.file_sha(spec['seal_path'])!=spec['seal_sha256']:
                raise ValueError('analysis_prediction_reference_changed')
        verified=[reuse.verify_prediction(spec)for spec in specs];first=verified[0]
        if any(v['source_sha256']!=first['source_sha256']or v['geometry_sha256']!=first['geometry_sha256']or v['case_ids']!=first['case_ids']for v in verified):
            raise ValueError('analysis_methods_changed_common_actual_label_geometry')
        proofs,all_ready=reuse.completed_physical_proofs(first['sources'],first['planned_apps'])
        unique={}
        for source in first['sources'].values():
            job=source['provenance']['job_id'];unique.setdefault(job,source)
            seed_values.add(source['provenance']['seed']);scene_seed.add((source['scene'],source['provenance']['seed']))
        expected=20 if block['partition']=='original_matrix'else 2
        if len(unique)!=expected:raise ValueError('analysis_physical_arms_per_block_differ')
        for job,source in unique.items():
            if job in seen:raise ValueError('physical_run_reused_as_an_independent_experiment')
            seen.add(job);counts[block['partition']]+=1;proof=proofs[reuse._physical_key(source)]
            if proof['ready']:ready[block['partition']]+=1
            else:pending.append(dict(job_id=job,partition=block['partition'],origin_key=block['origin_key'],reason=proof['reason']))
        blocks.append(dict(id=block['id'],all_physical_runs_verified=all_ready))
    if dict(counts)!=PARTITIONS or len(seen)!=132:raise ValueError('analysis_must_cover_all132_distinct_physical_runs')
    if sorted(seed_values)!=manifest['expected_unique_seeds']or len(scene_seed)!=manifest['expected_scene_seed_blocks']:
        raise ValueError('supplement_created_or_dropped_seed_blocks')
    return dict(expected_physical_runs=132,verified_physical_runs=sum(ready.values()),ready_by_partition=dict(ready),pending=pending,
        all_physical_runs_verified=not pending,unique_seed_count=len(seed_values),scene_seed_blocks=len(scene_seed),blocks=blocks)


def run_analysis(manifest_path,output,cache_dir,*,interim=False,check_only=False):
    manifest=validate_manifest(manifest_path);status=preflight(manifest);output=Path(output)
    if check_only or(not interim and not status['all_physical_runs_verified']):
        return dict(stage='ready_for_final'if status['all_physical_runs_verified']else'waiting_for_completed_physical_matrix',
            output_written=False,status=status,manifest_sha256=manifest['manifest_sha256'])
    if output.exists():raise ValueError('analysis_requires_a_new_output_directory')
    output.mkdir(parents=True);mode='interim'if interim else'final';evaluations=[]
    for block in manifest['blocks']:
        joined=reuse.join_with_label_reuse(block['methods'],output/'evaluations'/block['partition']/block['origin_key'],cache_dir,refresh_id=mode)
        evaluations.append(dict(block_id=block['id'],partition=block['partition'],origin_key=block['origin_key'],outputs=joined['outputs']))
    report_refs=[]
    definitions=[('original_matrix','Original20-arm matrix',{'original_matrix'},
        'Original20-arm study only. This is the separately retained main-study primary pool.'),
        ('prospective_supplement','Prospective two-arm supplement',{'prospective_supplement'},
        'BCE/BCDE supplement only. The same two seeds and six scene/seed blocks are reused; these are not new independent seeds.'),
        ('combined_diagnostic','Combined diagnostic',set(PARTITIONS),
        'Original matrix plus prospective supplement, shown diagnostically. The original primary results remain separate; no candidate flags or seed counts are changed.')]
    for partition,title,included,note in definitions:
        methods=[]
        for method in METHODS:
            blocks=[]
            for evaluated in evaluations:
                if evaluated['partition']not in included:continue
                parent=next(b for b in manifest['blocks']if b['id']==evaluated['block_id'])
                spec=next(s for s in parent['methods']if s['method']==method);result=next(v for v in evaluated['outputs']if v['method']==method)
                blocks.append(dict(score_path=spec['score_path'],seal_path=spec['seal_path'],evaluation_path=result['path']))
            methods.append(dict(id=method,blocks=blocks))
        report_dir=output/'reports'/partition
        report_manifest=dict(schema='oran.quantification-report-inputs.v1',methods=methods,title=title,study_partition=partition,
            analysis_status=mode,population_note=note,status_note=(f'Interim analysis: {status["verified_physical_runs"]}/132 physical runs verified. No method conclusion is drawn.'if interim else
                'Final descriptive analysis: all132 physical checkpoints verified. Derived intents are not independent replications.'),
            before_action_coverage_inventory=manifest['before_action_coverage_inventory']['path'])
        report_manifest['execution_label_sources']=source_refs(methods)
        path=output/'report-manifests'/(partition+'.json');study.write_immutable(path,report_manifest)
        build_report(path,report_dir);report_refs.append(dict(partition=partition,html=str(report_dir/'index.html'),summary=_ref(report_dir/'report-summary.json')))
    summary=dict(schema='oran.quantification-analysis-completion.v1',analysis_status=mode,manifest=_ref(manifest_path),status=status,reports=report_refs,
        original_primary_results_reported_separately=True,supplement_adds_no_new_seeds=True,candidate_flags_modified=False,
        formulas_gains_or_weights_modified=False,finished_utc=datetime.now(timezone.utc).isoformat())
    study.write_immutable(output/'analysis-completion.json',summary)
    write_landing_page(output,title='O-RAN quantification study',
        status=mode.upper()+' · '+str(status['verified_physical_runs'])+'/132 verified physical runs · 2 independent seeds, 6 scene × seed blocks',pages=report_refs)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare-manifest');prep.add_argument('--artifact',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    run=commands.add_parser('run');run.add_argument('--manifest',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--cache',type=Path,required=True);run.add_argument('--interim',action='store_true');run.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare-manifest':value=prepare_manifest(args.artifact,args.output);print(json.dumps({'manifest':str(args.output),'manifest_sha256':value['manifest_sha256']}));return
    result=run_analysis(args.manifest,args.output,args.cache,interim=args.interim,check_only=args.check_only)
    print(json.dumps(result))
    if result.get('stage')=='waiting_for_completed_physical_matrix'and not args.check_only:raise SystemExit(2)


if __name__=='__main__':main()
