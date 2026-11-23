"""Outcome-blind predictions for the separately preregistered BCE/BCDE grid.

Only immutable OFF-derived inputs and frozen scoring releases are read. The
original20-arm plan, predictions and physical source are never rewritten.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from experiments import study_quantification as study
from experiments.quantification_supplement import assignment,verify_plan

METHODS=('goal_obstruction_v2','net_intent_obstruction_v1','temporal_net_intent_obstruction_v1','net_profile_gain_obstruction_v1')


def pinned(ref):
    if study.file_sha(ref['path'])!=ref['sha256']:raise ValueError('supplement_evidence_pin_changed:'+str(ref['path']))
    return study.read(ref['path'])


def selection_settings(base_seal_path,source_archive,archive_pin):
    """Recover original settings without executing the old harness."""
    pipeline=study.read(Path(base_seal_path).parent/'pipeline-sources/manifest.json')
    source=pipeline['files']['experiments/study_quantification.py']
    if study.file_sha(source['archived_path'])!=source['sha256']:raise ValueError('original_harness_archive_changed')
    trees=ast.parse(Path(source['archived_path']).read_text());settings=None
    adapter=lambda tree:next(ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='candidate_flags')
    if adapter(trees)!=adapter(ast.parse(Path(study.__file__).read_text())):
        raise ValueError('candidate_adapter_changed_since_original_predictions')
    for node in trees.body:
        if isinstance(node,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='DEFAULT_SELECTION'for x in node.targets):
            value=node.value
            if not isinstance(value,ast.Call)or not isinstance(value.func,ast.Name)or value.func.id!='dict'or value.args:
                raise ValueError('unsupported_frozen_selector_settings')
            settings={kw.arg:ast.literal_eval(kw.value)for kw in value.keywords}
    if settings is None:raise ValueError('frozen_selector_settings_missing')
    selector=pipeline['files']['smo/candidate_selection.py']
    if study.file_sha(selector['archived_path'])!=selector['sha256']or study.file_sha(study.ROOT/'smo/candidate_selection.py')!=selector['sha256']:
        raise ValueError('candidate_selector_changed_since_original_predictions')
    if study.file_sha(source_archive)!=archive_pin:raise ValueError('supplement_physical_archive_changed')
    dependencies=('smo/graph_combinations.py','smo/xapp_capabilities.py','smo/direct_action_filter.py')
    dependency_pins={}
    with tarfile.open(source_archive,'r:*')as archive:
        members={m.name.removeprefix('./'):m for m in archive if m.isfile()}
        for dependency in dependencies:
            if dependency not in members:raise ValueError('original_selector_dependency_missing:'+dependency)
            pin=hashlib.sha256(archive.extractfile(members[dependency]).read()).hexdigest()
            if study.file_sha(study.ROOT/dependency)!=pin:raise ValueError('selector_dependency_changed_since_original_physical_source:'+dependency)
            dependency_pins[dependency]=pin
    runtime=pipeline['files']['smo/runtime.py']
    if study.file_sha(study.ROOT/'smo/runtime.py')!=runtime['sha256']:raise ValueError('selector_numeric_helper_changed')
    return settings,dict(settings=settings,selector_sha256=selector['sha256'],selector_dependency_sha256=dependency_pins,
        frozen_settings_source_sha256=source['sha256'])


def _load_context(base_seal_path,plan_path,manifest_path,origin_key):
    plan=study.read(plan_path);verify_plan(plan,plan['plan_sha256']);manifest=study.read(manifest_path)
    if (manifest.get('schema')!='oran.prospective-supplement-cases.v1'or manifest.get('supplement_plan_sha256')!=plan['plan_sha256']
        or manifest.get('parent_physical_plan_sha256')!=plan['parent_physical_plan_sha256']or manifest.get('source_version')!='v2'
        or manifest.get('after_values_used')is not False):raise ValueError('supplement_cases_manifest_mismatch')
    expected={case['id']:{**case,'worker_id':worker}for worker,cases in assignment(plan,plan['plan_sha256']).items()for case in cases}
    if len(manifest['cases'])!=12 or {c['id']for c in manifest['cases']}!=set(expected):raise ValueError('supplement_manifest_grid_incomplete')
    artifact=Path(manifest_path).parent
    for case in manifest['cases']:
        if any(case.get(k)!=v for k,v in expected[case['id']].items()):raise ValueError('supplement_stable_case_identity_changed')
        if any(case.get(k)!=plan[k]for k in('source_sha256','source_archive_sha256')):raise ValueError('supplement_case_physical_source_changed')
        path=artifact/'workers'/case['worker_id']/'jobs'/case['id']/'data/run'
        if Path(case['run_path']).resolve()!=path.resolve():raise ValueError('supplement_future_run_path_changed')
    origins=[o for o in plan['origins']if o['origin_key']==origin_key]
    if len(origins)!=1:raise ValueError('supplement_origin_not_registered')
    origin=origins[0];proof=pinned(origin['origin_proof']);shadow=pinned(origin['before_preview'])
    if proof['origin_id']!=origin['origin_id']or shadow.get('origin_id')!=origin['origin_id']or shadow.get('after_kpis_used')is not False:
        raise ValueError('supplement_OFF_origin_evidence_changed')
    original_set=pinned(plan['original_score_set']);base_ref=next(b for b in original_set['blocks']if b['origin_key']==origin_key)
    if Path(base_ref['path']).resolve()!=Path(base_seal_path).resolve():raise ValueError('supplement_must_reuse_original_sealed_origin')
    base=pinned(base_ref)
    if (base.get('schema')!='oran.prospective-score-seal.v1'or base.get('after_values_used')is not False
        or base['blocks'][0]['origin_id']!=origin['origin_id']or base['blocks'][0]['off_job_id']!=origin['off_job_id']):
        raise ValueError('supplement_base_origin_changed')
    inputs=pinned(dict(path=base['input_path'],sha256=base['input_sha256']))
    sources_path=Path(base['input_path']).with_name('label-sources.json')
    sources=pinned(dict(path=str(sources_path),sha256=base['label_sources_sha256']))['sources']
    selected=[c for c in manifest['cases']if c['origin_key']==origin_key]
    if {c['arm']for c in selected}!={'BCE','BCDE'}:raise ValueError('supplement_requires_both_registered_arms')
    for case in selected:
        if(Path(case['run_path']).parent/'request.json').exists():raise ValueError('supplement_arm_started_before_predictions')
    return plan,manifest,origin,base,inputs,sources,selected


def prepare_origin_inputs(base_seal_path,plan_path,manifest_path,output,*,origin_key):
    plan,manifest,origin,base,original,original_sources,jobs=_load_context(base_seal_path,plan_path,manifest_path,origin_key)
    settings,selection=selection_settings(base_seal_path,Path(manifest_path).parent.parent/'source.tgz',plan['source_archive_sha256'])
    templates={}
    for saved in original['cases']:
        if not saved['app_ids']:
            if saved['intent_id']in templates:raise ValueError('duplicate_original_OFF_intent_template')
            templates[saved['intent_id']]=study.materialize_case(original,saved)
    if len(templates)!=plan['intents']:raise ValueError('all_fifteen_original_OFF_intents_required')
    data={k:deepcopy(v)for k,v in original.items()if k not in('cases','contexts','context_storage')};data['cases']=[];sources={}
    for job in sorted(jobs,key=lambda c:c['arm']):
        for intent_id,template in templates.items():
            case=deepcopy(template);old_id=case['case_id'];case.pop('context_id',None)
            case.update(case_id=':'.join((origin_key,job['id'],intent_id)),group_id=job['group_id'],app_ids=deepcopy(job['app_ids']))
            provenance=case['provenance']
            expected=dict(seed=origin['seed'],worker_id=origin['worker_id'],scoring_origin_id=origin['origin_id'],
                scoring_origin_job_id=origin['off_job_id'],physical_plan_sha256=plan['parent_physical_plan_sha256'],source_archive_sha256=plan['source_archive_sha256'],after_kpis_used=False)
            if any(provenance.get(k)!=v for k,v in expected.items()):raise ValueError('original_OFF_template_physical_origin_mismatch')
            provenance.update(job_id=job['id'],batch_id='supplement:'+origin_key,supplement_plan_sha256=plan['plan_sha256'],grid_partition='prospective_supplement')
            case['protocol'].update(primary_validation_eligible=True,actual_arm_origin_guard_required=True,actual_arm_origin_guard_pending=True,
                reasons=[r for r in case['protocol'].get('reasons',[])if r!='OFF_diagnostic'])
            if case.get('intent_resolution',{}).get('status')=='unavailable':
                case['candidate']=dict(pool='unresolved_intent',eligible=False,reason='intent_reference_unavailable')
            else:
                case['candidate']=study.candidate_flags(data['graphs'][case['graph_id']],case['frame'],case['ranges'],case['preview'],case['app_ids'],settings)
            case['independence_group']=study.digest(dict(cohort=case['cohort_id'],apps=sorted(case['app_ids']),prefix=provenance['prefix_exogenous_sha256']))
            source=deepcopy(original_sources[old_id]);source.update(run_path=job['run_path'],
                seal_path=str(Path(job['run_path']).parents[3]/'checkpoints'/(job['id']+'.json')),requires_actual_origin_guard=True,
                protocol=deepcopy(case['protocol']),candidate=deepcopy(case['candidate']),provenance=deepcopy(provenance),independence_group=case['independence_group'])
            data['cases'].append(case);sources[case['case_id']]=source
    data.update(stage='predeployment_inputs',after_values_included=False,supplement=dict(plan_path=str(plan_path),plan_sha256=plan['plan_sha256'],
        manifest_path=str(manifest_path),manifest_sha256=study.file_sha(manifest_path),base_seal=dict(path=str(base_seal_path),sha256=study.file_sha(base_seal_path)),
        original_matrix_modified=False,selection=selection))
    for block in data.get('prospective_blocks',[]):block.update(arms=2,intents=plan['intents'])
    study.compact_contexts(data);output=Path(output)
    study.write_immutable(output/'scoring-inputs.json',data)
    study.write_immutable(output/'label-sources.json',dict(schema=study.SCHEMA,sources=sources,after_values_included=False))
    return data


def _score_release(inputs_path,output,release_score_path,*,method,dispatch_method=None):
    """Use original archived JS bytes; do not replace comparator releases."""
    data=study.read(inputs_path);release=study.read(release_score_path);output=Path(output)
    archive=output.parent/'engine-sources';archive.mkdir(parents=True,exist_ok=True);modules={}
    for source,pin in release['module_sha256'].items():
        old=Path(release['archived_modules'][source])
        if study.file_sha(old)!=pin:raise ValueError('original_scoring_release_changed')
        dest=archive/(pin+'-'+Path(source).name)
        if not dest.exists():dest.write_bytes(old.read_bytes())
        if study.file_sha(dest)!=pin:raise ValueError('supplement_scoring_archive_changed')
        modules[source]=str(dest)
    identity=dict(method=method,dispatch_method=dispatch_method,input_path=str(inputs_path),input_sha256=study.file_sha(inputs_path),
        module_sha256=release['module_sha256'],archived_modules=modules,engine=release['engine'],
        release_score_path=str(release_score_path),release_score_sha256=study.file_sha(release_score_path),
        label_sources_sha256=study.file_sha(Path(inputs_path).with_name('label-sources.json')))
    if output.exists():
        saved=study.read(output)
        if any(saved.get(k)!=v for k,v in identity.items()):raise ValueError('refusing_to_overwrite_supplement_scores')
        return saved
    driver="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
const d=JSON.parse(fs.readFileSync(0,'utf8')),original=JSON.stringify(d);
for(const p of d.modules)vm.runInContext(fs.readFileSync(p,'utf8'),c);
const engine=c[d.engine];if(!engine?.review)throw Error('Frozen scoring engine unavailable');
const compact=r=>({method:r.method,schema:r.schema,status:r.status,reason:r.reason,
 kpi_forecast_performed:r.kpi_forecast_performed,observed_after_kpis_used:r.observed_after_kpis_used,
 groups:r.groups?.map(g=>Object.fromEntries(['group_id','app_ids','score','score_status','reason','raw_obstruction_strength',
  'reference_effect','reference_effect_score','goal_components','projection_coverage','model_coverage_complete','history_sample_count','no_active_push']
  .filter(k=>k in g).map(k=>[k,g[k]])))});
const scores=d.data.cases.map(x=>({case_id:x.case_id,intent_id:x.intent_id,cohort_id:x.cohort_id,independence_group:x.independence_group,
 protocol:x.protocol,provenance:x.provenance,candidate:x.candidate,scene:x.scene,result:(()=>{
 if(x.intent_resolution?.status==='unavailable')return {method:d.method,status:'unavailable',reason:'intent_reference_unavailable',
  groups:[{group_id:x.group_id,app_ids:x.app_ids,score:null,raw_obstruction_strength:null,score_status:'unavailable',reason:'intent_reference_unavailable'}],observed_after_kpis_used:false};
 const context=d.data.contexts[x.context_id];return compact(engine.review(d.data.graphs[x.graph_id],context.frame,
 [{id:x.group_id,app_ids:x.app_ids}],x.ranges,context.preview,{...context.options,...x.options,method:d.dispatch_method||d.method}));})()}));
if(original!==JSON.stringify(d))throw Error('Predeployment inputs mutated');process.stdout.write(JSON.stringify(scores));"""
    node=shutil.which('node')
    if not node:raise ValueError('Node is required for archived scoring releases')
    values=json.loads(subprocess.check_output([node,'--input-type=commonjs','-e',driver],text=True,
        input=json.dumps(dict(data=data,modules=list(modules.values()),engine=release['engine'],method=method,dispatch_method=dispatch_method))))
    driver_pin=study.file_sha(__file__);driver_copy=archive/(driver_pin+'-study_supplement.py')
    if not driver_copy.exists():driver_copy.write_bytes(Path(__file__).read_bytes())
    if study.file_sha(driver_copy)!=driver_pin:raise ValueError('supplement_driver_archive_changed')
    result=dict(schema=study.SCHEMA,stage='scores_before_label_join',**identity,scores=values,after_values_used=False,new_simulations=0,
        engine_method=dispatch_method or method,driver_sha256=driver_pin,archived_driver=str(driver_copy),full_projection_results=False)
    if data.get('graph_variant'):result['graph_variant']=deepcopy(data['graph_variant'])
    study.write_immutable(output,result);return result


def seal_supplement_origin(base_seal_path,profile_parent_seal_path,plan_path,manifest_path,output,*,origin_key):
    output=Path(output);data=prepare_origin_inputs(base_seal_path,plan_path,manifest_path,output,origin_key=origin_key)
    plan,manifest,origin,base,_,_,jobs=_load_context(base_seal_path,plan_path,manifest_path,origin_key)
    profile=study.read(profile_parent_seal_path)
    if (profile.get('origin_key')!=origin_key or profile.get('after_values_used')is not False
        or profile.get('base_seal_sha256')!=study.file_sha(base_seal_path)):
        raise ValueError('supplement_profile_parent_must_match_original_origin')
    profile_data=pinned(dict(path=profile['input_path'],sha256=profile['input_sha256']))
    study.derive_graph_variant_inputs(output/'scoring-inputs.json',profile_data['graphs'],output/'profile',
        variant_id=METHODS[-1],provenance_paths=[profile_parent_seal_path,plan_path,manifest_path])
    scores={};method_inputs={}
    for method in METHODS:
        parent=profile if method==METHODS[-1]else base
        ref=parent['scores'][method];pinned(ref)
        inp=output/'profile/scoring-inputs.json'if method==METHODS[-1]else output/'scoring-inputs.json'
        scored_path=output/('scores-'+method+'.json')
        scored=_score_release(inp,scored_path,ref['path'],method=method,
            dispatch_method='net_intent_obstruction_v1'if method==METHODS[-1]else None)
        scores[method]=dict(path=str(scored_path),sha256=study.file_sha(scored_path),count=len(scored['scores']),
            valid_scores=sum((x['result'].get('groups')or[{}])[0].get('score')is not None for x in scored['scores']))
        method_inputs[method]=dict(path=str(inp),sha256=study.file_sha(inp))
    if any((Path(job['run_path']).parent/'request.json').exists()for job in jobs):raise ValueError('supplement_started_during_scoring')
    helper=output/'helper-code'/(study.file_sha(__file__)+'-study_supplement.py');helper.parent.mkdir(parents=True,exist_ok=True)
    if not helper.exists():helper.write_bytes(Path(__file__).read_bytes())
    pins=dict(supplement_plan_sha256=plan['plan_sha256'],parent_physical_plan_sha256=plan['parent_physical_plan_sha256'],source_version='v2',
        source_sha256=plan['source_sha256'],source_archive_sha256=plan['source_archive_sha256'],origin_key=origin_key,
        blocks=deepcopy(data['prospective_blocks']),physical_arms=2,intents=plan['intents'],cases=len(data['cases']),
        base_seal=dict(path=str(base_seal_path),sha256=study.file_sha(base_seal_path)),
        profile_parent_seal=dict(path=str(profile_parent_seal_path),sha256=study.file_sha(profile_parent_seal_path)),
        cases_manifest=dict(path=str(manifest_path),sha256=study.file_sha(manifest_path)),
        input_path=str(output/'scoring-inputs.json'),input_sha256=study.file_sha(output/'scoring-inputs.json'),
        profile_input=method_inputs[METHODS[-1]],method_inputs=method_inputs,scores=scores,
        label_sources=dict(path=str(output/'label-sources.json'),sha256=study.file_sha(output/'label-sources.json')),
        profile_label_sources=dict(path=str(output/'profile/label-sources.json'),sha256=study.file_sha(output/'profile/label-sources.json')),
        helper_code=dict(path=str(helper),sha256=study.file_sha(helper)))
    path=output/'prediction-seal.json'
    if path.exists():
        saved=study.read(path)
        if any(saved.get(k)!=v for k,v in pins.items()):raise ValueError('refusing_to_overwrite_supplement_seal')
        return saved
    seal=dict(schema='oran.prospective-supplement-score-seal.v1',created_utc=datetime.now(timezone.utc).isoformat(),**pins,
        after_values_used=False,nonempty_arm_started_before_scoring=False,gate_opened=False,algorithm_tuned_to_test_labels=False,
        original_matrix_modified=False)
    study.write_immutable(path,seal);return seal


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base-seal',type=Path,required=True)
    parser.add_argument('--profile-parent-seal',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--origin',required=True)
    args=parser.parse_args();seal=seal_supplement_origin(args.base_seal,args.profile_parent_seal,args.plan,args.manifest,args.output,origin_key=args.origin)
    print(json.dumps(seal,ensure_ascii=False))


if __name__=='__main__':main()
