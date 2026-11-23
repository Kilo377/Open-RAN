"""Independent fifth-method ledger and genuinely forward-only validation.

The original four methods and profileG are immutable. All1980 scores may be
retained diagnostically; only physical jobs absent in both remote start scans
qualify for independent validation of this later comparator.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from experiments import study_quantification as study
from experiments.analyze_quantification_study import validate_manifest

METHOD='aggregation_aware_obstruction_v1'
ENGINE='SmoAggregationAwareObstruction'
PROFILE_METHOD='net_profile_gain_obstruction_v1'


def ref(path):return dict(path=str(path),sha256=study.file_sha(path))


def pinned(value):
    if study.file_sha(value['path'])!=value['sha256']:raise ValueError('aggregation_study_evidence_pin_changed:'+str(value['path']))
    return study.read(value['path'])


def addendum(path):
    value=study.read(path)
    if (value.get('schema')!='oran.aggregation-aware-method-addendum.v1'or value.get('method')!=METHOD
        or study.digest({k:v for k,v in value.items()if k!='plan_sha256'})!=value.get('plan_sha256')
        or value.get('scoring_parameters_fit_to_after_labels')is not False or value.get('additional_physical_runs')!=0):
        raise ValueError('independent_fifth_method_addendum_required')
    return value


def prepare_ledger(analysis_manifest_path,addendum_path,output):
    manifest=validate_manifest(analysis_manifest_path);plan=addendum(addendum_path);output=Path(output)
    blocks=[];cases=[];jobs={};parent_pins={}
    for block in manifest['blocks']:
        spec=next(s for s in block['methods']if s['method']==PROFILE_METHOD)
        seal=pinned(dict(path=spec['seal_path'],sha256=spec['seal_sha256']))
        source_ref=(seal['method_inputs'][PROFILE_METHOD]if seal['schema']=='oran.prospective-supplement-score-seal.v1'else
            dict(path=seal['input_path'],sha256=seal['input_sha256']))
        original=pinned(source_ref);copied=deepcopy(original)
        # graph_variant.id was historically also treated as a method alias by
        # Evaluation. Separate G provenance so the new method keeps its own ID.
        gain=copied.pop('graph_variant',None)
        copied['gain_model_provenance']=dict(profile_graph_variant=gain,parent_input=source_ref,parent_seal=ref(spec['seal_path']),
            graph_sha256={k:study.digest(g)for k,g in original['graphs'].items()},new_gain_model=False)
        copied['method_context']=dict(method_id=METHOD,addendum=ref(addendum_path),source_before_only=True,options_recomputed=False)
        directory=output/'inputs'/block['partition']/block['origin_key'];inp=directory/'scoring-inputs.json'
        proof=[]
        for before,after in zip(original['cases'],copied['cases']):
            a=study.materialize_case(original,before);b=study.materialize_case(copied,after)
            if a!=b:raise ValueError('fifth_method_changed_frozen_case')
            digest=study.digest(a);proof.append(dict(case_id=a['case_id'],materialized_sha256=digest))
            provenance=a['provenance'];job_id=provenance['job_id'];partition='main'if block['partition']=='original_matrix'else'supplement'
            job=dict(job_id=job_id,origin_key=block['origin_key'],partition=partition,worker_id=provenance['worker_id'],
                app_ids=a['app_ids'],scene=a['scene'],seed=provenance['seed'])
            if job_id in jobs and jobs[job_id]!=job:raise ValueError('one_physical_job_has_inconsistent_frozen_identity')
            jobs[job_id]=job
            cases.append(dict(case_id=a['case_id'],job_id=job_id,block_id=block['id'],partition=partition,analysis_partition=block['partition'],
                origin_key=block['origin_key'],group_id=a['group_id'],intent_id=a['intent_id'],scene=a['scene'],seed=provenance['seed'],
                materialized_input_sha256=digest,candidate_flags_sha256=study.digest(a['candidate']),prospective_eligibility='pending_two_remote_scans'))
        if copied['graphs']!=original['graphs']:raise ValueError('fifth_method_changed_profile_G')
        study.write_immutable(inp,copied)
        source_path=Path(source_ref['path']).with_name('label-sources.json');dest=directory/'label-sources.json'
        if dest.exists():
            if dest.read_bytes()!=source_path.read_bytes():raise ValueError('fifth_method_changed_label_source_bytes')
        else:dest.write_bytes(source_path.read_bytes())
        blocks.append(dict(id=block['id'],partition=block['partition'],origin_key=block['origin_key'],cases=len(copied['cases']),
            input=ref(inp),label_sources=ref(dest),profile_input=source_ref,profile_parent_seal=ref(spec['seal_path']),
            materialized_case_proofs=proof,graph_sha256=copied['gain_model_provenance']['graph_sha256']))
        # All fifth scores must use the unchanged app/goal release of profileG.
        parent_score=pinned(seal['scores'][PROFILE_METHOD])
        for name,pin in parent_score['module_sha256'].items():
            if Path(name).name in('app.js','goal_obstruction.js'):
                if name in parent_pins and parent_pins[name]!=pin:raise ValueError('profile_parent_projection_releases_differ')
                parent_pins[name]=pin
    if len(blocks)!=12 or len(cases)!=1980 or len({c['case_id']for c in cases})!=1980 or len(jobs)!=132:
        raise ValueError('complete1980_case132_job_fifth_ledger_required')
    if {Path(name).name for name in parent_pins}!={'app.js','goal_obstruction.js'}:
        raise ValueError('both_frozen_parent_projection_module_pins_required')
    ledger=dict(schema='oran.aggregation-before-only-ledger.v1',method=METHOD,created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_manifest=ref(analysis_manifest_path),addendum=ref(addendum_path),addendum_plan_sha256=plan['plan_sha256'],
        blocks=blocks,cases=cases,jobs=list(jobs.values()),parent_app_goal_sha256=parent_pins,
        old_four_prediction_sets=manifest['source_sets'],after_kpis_used=False,new_physical_runs=0,original_methods_graphs_or_flags_modified=False)
    study.write_immutable(output/'before-ledger.json',ledger);return ledger


def freeze_bundle(ledger_path,output):
    ledger=study.read(ledger_path);output=Path(output);parts=[];components={}
    for name in('smo/web/observation_normalization.js','smo/web/aggregation_aware_obstruction.js'):
        source=study.ROOT/name;pin=study.file_sha(source);archive=output/(pin+'-'+source.name);archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():archive.write_bytes(source.read_bytes())
        if study.file_sha(archive)!=pin:raise ValueError('aggregation_component_archive_changed')
        components[name]=ref(archive);parts.append(b'\n;\n/* '+name.encode()+b' sha256='+pin.encode()+b' */\n'+source.read_bytes())
    for name,pin in ledger['parent_app_goal_sha256'].items():
        if study.file_sha(name)!=pin:raise ValueError('app_or_goal_changed_since_frozen_profile_parent')
        source=Path(name);archive=output/(pin+'-'+source.name)
        if not archive.exists():archive.write_bytes(source.read_bytes())
        if study.file_sha(archive)!=pin:raise ValueError('parent_projection_archive_changed')
        components[name]=ref(archive)
    bundle=output/'aggregation-aware-bundle.js';payload=b''.join(parts)
    if bundle.exists():
        if bundle.read_bytes()!=payload:raise ValueError('aggregation_bundle_is_immutable')
    else:bundle.write_bytes(payload)
    package=dict(schema='oran.aggregation-code-freeze.v1',method=METHOD,created_utc=datetime.now(timezone.utc).isoformat(),created_unix_s=time.time(),
        ledger=ref(ledger_path),addendum=ledger['addendum'],components=components,code_bundle=ref(bundle),bundle_mtime_ns=bundle.stat().st_mtime_ns,
        unchanged_parent_app_goal_sha256=ledger['parent_app_goal_sha256'],old_four_sets=ledger['old_four_prediction_sets'],after_kpis_used=False)
    study.write_immutable(output/'code-freeze.json',package)
    for path in[*(Path(v['path'])for v in components.values()),bundle,output/'code-freeze.json']:path.chmod(0o444)
    return package


def score_between_scans(ledger_path,code_freeze_path,v2_artifact,output):
    from experiments.prospective_request_inventory import scan_requests,paired_eligibility
    ledger=study.read(ledger_path);code=study.read(code_freeze_path);output=Path(output)
    if code['ledger']!=ref(ledger_path):raise ValueError('frozen_code_and_ledger_differ')
    pinned(code['addendum']);bundle=code['code_bundle']
    if study.file_sha(bundle['path'])!=bundle['sha256']:raise ValueError('frozen_aggregation_bundle_changed')
    if output.exists():raise ValueError('scoring_attempt_requires_a_new_directory')
    output.mkdir(parents=True)
    before=scan_requests(v2_artifact,output/'scan-before.json',code_bundle=bundle)
    started=time.time();started_utc=datetime.now(timezone.utc).isoformat()
    def score_block(block):
        for name,pin in ledger['parent_app_goal_sha256'].items():
            if study.file_sha(name)!=pin:raise ValueError('parent_projection_changed_before_fifth_scoring')
        directory=output/'scores'/block['partition']/block['origin_key'];path=directory/'scores-aggregation_aware_obstruction_v1.json'
        result=study.score_cases(block['input']['path'],path,method=METHOD,module=bundle['path'],engine=ENGINE)
        for name,pin in ledger['parent_app_goal_sha256'].items():
            if result['module_sha256'].get(name)!=pin:raise ValueError('fifth_score_uses_a_different_parent_projection')
        return dict(block_id=block['id'],partition=block['partition'],origin_key=block['origin_key'],input=block['input'],
            label_sources=block['label_sources'],score=ref(path),cases=len(result['scores']))
    # The bundled local Node runtime showed a native allocator/JSON-parser
    # crash under parallel scoring. The frozen mathematical bundle is unchanged.
    scored=[score_block(block)for block in ledger['blocks']]
    finished=time.time();finished_utc=datetime.now(timezone.utc).isoformat()
    after=scan_requests(v2_artifact,output/'scan-after.json',code_bundle=bundle)
    eligibility=paired_eligibility(before,after,code_bundle=bundle,scoring_started_unix_s=started,scoring_finished_unix_s=finished)
    study.write_immutable(output/'forward-eligibility.json',eligibility)
    value=dict(schema='oran.aggregation-forward-score-seal.v1',method=METHOD,created_utc=datetime.now(timezone.utc).isoformat(),
        ledger=ref(ledger_path),code_freeze=ref(code_freeze_path),addendum=ledger['addendum'],code_bundle=bundle,scan_before=before,scan_after=after,
        scoring_started_unix_s=started,scoring_finished_unix_s=finished,scoring_started_utc=started_utc,scoring_finished_utc=finished_utc,
        forward_eligibility=ref(output/'forward-eligibility.json'),blocks=scored,all_case_count=1980,new_physical_runs=0,
        previously_started_jobs_diagnostic_only=True,after_kpis_used_for_scoring=False,original_four_sets=ledger['old_four_prediction_sets'])
    study.write_immutable(output/'method-seal.json',value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--analysis-manifest',type=Path,required=True);prep.add_argument('--addendum',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    freeze=commands.add_parser('freeze-code');freeze.add_argument('--ledger',type=Path,required=True);freeze.add_argument('--output',type=Path,required=True)
    score=commands.add_parser('score');score.add_argument('--ledger',type=Path,required=True);score.add_argument('--code-freeze',type=Path,required=True)
    score.add_argument('--v2-artifact',type=Path,required=True);score.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':value=prepare_ledger(args.analysis_manifest,args.addendum,args.output);print(json.dumps(dict(cases=len(value['cases']),jobs=len(value['jobs']))));return
    if args.command=='freeze-code':value=freeze_bundle(args.ledger,args.output);print(json.dumps(value['code_bundle']));return
    value=score_between_scans(args.ledger,args.code_freeze,args.v2_artifact,args.output);print(json.dumps(dict(method_seal=str(args.output/'method-seal.json'),cases=value['all_case_count'])))


if __name__=='__main__':main()
