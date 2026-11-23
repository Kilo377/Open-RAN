"""Offline, provenance-separated conflict-score validation on retained runs.

The score phase receives only a frozen predeployment state, trained-policy
proposals, declared intent ranges, and the selected KG. Observed AFTER values
are read only by the separate label-join phase, after scores are persisted.
This module never launches a simulator or writes to the user's intent ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

from smo.__main__ import GraphService
from smo.candidate_selection import select_combinations
from smo.contracts import KPIS
from smo.evaluation_reference import archived_dynamics_fingerprint, comparable, origin_configuration, origin_frame, prepare_origin
from smo.influence_units import normalization
from smo.intent_language import resolve_ranges
from smo.intent_quality import assess
from smo.result_evaluation import evaluate
from smo.runtime import FrameBuilder, records, trace_users
from smo.window_metrics import WindowIndex
from smo.xapp_preview import PreviewStore

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='oran.quantification-study.v1'
DEFAULT_SELECTION=dict(method='semantic_influence_coverage_v2',coverage_threshold=.05,marginal_tolerance=.05)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


def write_immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('refusing_to_overwrite_frozen_study_artifact:'+str(path))
        return
    write(path,value)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def freeze_pipeline(output):
    """Content-addressed source copies, independent of any observed outcome."""
    output=Path(output);files={}
    names=['smo/web/app.js','smo/web/goal_obstruction.js','smo/web/intent_obstruction.js',
        'smo/influence_units.py','smo/intent_language.py','smo/candidate_selection.py','smo/window_metrics.py',
        'smo/evaluation_reference.py','smo/result_evaluation.py','smo/runtime.py','experiments/study_quantification.py']
    for name in names:
        source=ROOT/name
        if not source.is_file():continue
        pin=file_sha(source);target=output/(pin+'-'+source.name);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(source.read_bytes())
        if file_sha(target)!=pin:raise ValueError('pipeline_source_pin_mismatch')
        files[name]=dict(sha256=pin,archived_path=str(target))
    manifest=dict(schema=SCHEMA,stage='frozen_pipeline_sources',files=files,observed_after_values_used=False)
    manifest['pipeline_sha256']=digest(files);write_immutable(output/'manifest.json',manifest)
    return manifest


def _trace_exists(run):
    return any((Path(run)/('rt_sys_trace'+suffix)).is_file()for suffix in('.jsonl','.jsonl.gz'))


def member_protocol(batch,member):
    """Metadata eligibility only; prefix verification happens during extraction."""
    c=member.get('comparison')or{};n=batch.get('observe_slots');baseline=c.get('baseline',{}).get('slot')
    before=c.get('baseline_window')or{};after=c.get('after_window')or{};guard=c.get('evaluation_origin')or{}
    origin=(batch.get('context')or{}).get('evaluation_origin')or{}
    checks=dict(fresh_off_prefix=c.get('baseline_origin')=='fresh_run_all_xapps_off',
        exact_before_window=type(baseline)is int and type(n)is int and before.get('window_start_slot')==baseline-n and before.get('slot')==baseline,
        fixed_after_window=c.get('fixed_combination_observation_window')is True and after.get('window_start_slot')==baseline and after.get('slot')==baseline+n if type(baseline)is int and type(n)is int else False,
        origin_pinned=bool(origin.get('origin_id')) and origin.get('origin_id')==guard.get('origin_id'),
        verified_before_activation=all(guard.get(k)is True for k in('verified_before_activation','observed_prefix_equal','configuration_equal','models_equal','network_dynamics_code_equal')) and guard.get('after_kpis_used')is False)
    return dict(name='common_origin_fixed_window'if all(checks.values())else'legacy_or_unverified',
        metadata_primary_candidate=all(checks.values()),checks=checks,
        reasons=[key for key,value in checks.items()if not value],primary_validation_eligible=False)


def inventory_batches(jobs_root):
    root=Path(jobs_root);batches=[]
    for path in sorted((root/'batches').glob('*/batch.json')):
        batch=read(path);members=[]
        for member in batch.get('members',[]):
            run=root/str(member.get('job_id','missing'))/'data/run'
            members.append(dict(job_id=member.get('job_id'),group_id=member.get('group_id'),app_ids=member.get('app_ids',[]),
                phase=member.get('phase'),comparison_available=bool(member.get('comparison')),run_path=str(run),
                config_available=(run/'config_snapshot.json').is_file(),trace_available=_trace_exists(run),
                preview_available=any((run/('xapp_decision_previews'+s)).is_file()for s in('.jsonl','.jsonl.gz')),
                protocol=member_protocol(batch,member)))
        batches.append(dict(batch_id=batch['id'],path=str(path),phase=batch.get('phase'),seed=batch.get('seed'),
            observe_slots=batch.get('observe_slots'),members=members))
    all_members=[m for b in batches for m in b['members']]
    return dict(schema=SCHEMA,stage='inventory',jobs_root=str(root),batch_count=len(batches),member_count=len(all_members),
        phase_counts=dict(Counter(m['phase']for m in all_members)),seeds=sorted({b['seed']for b in batches if b['seed']is not None}),
        unique_job_ids=len({m['job_id']for m in all_members if m.get('job_id')}),
        unique_completed_job_ids=len({m['job_id']for m in all_members if m.get('job_id')and m['phase']=='complete'}),
        complete_retained_count=sum(m['phase']=='complete'and m['comparison_available']and m['config_available']and m['trace_available']for m in all_members),
        primary_metadata_candidates=sum(m['phase']=='complete'and m['protocol']['metadata_primary_candidate']for m in all_members),
        new_simulations=0,raw_archives_modified=False,batches=batches)


def graph_for_study(influences,*,confidence_level=.90):
    """Exactly the current service merge/screen/unit policy, without any run."""
    return GraphService(influences=influences,influence_selection='significant_relations',
        influence_confidence_level=confidence_level).graph


def calibration_seeds(influences):
    seeds=set()
    for row in read(influences).get('estimates',[]):
        for fit in row.get('seed_fits',[]):
            if type(fit.get('seed'))is int:seeds.add(fit['seed'])
    return sorted(seeds)


def _hydrate_current_controls(frame,preview):
    frame=deepcopy(frame);observed={};ambiguous=set()
    for app in preview.get('apps',[]):
        for action in app.get('actions',[]):
            key=(action.get('parameter'),action.get('ue_id')or action.get('cell_id'))
            if not all(key)or action.get('current')is None:continue
            if key in observed and not comparable(observed[key],action['current']):ambiguous.add(key)
            observed[key]=deepcopy(action['current'])
    for (parameter,entity),value in observed.items():
        if (parameter,entity)in ambiguous:continue
        state=frame['states'].setdefault('rcp:'+parameter,{'by_entity':{}})
        state.setdefault('by_entity',{}).setdefault(entity,value)
    return frame


def _exogenous(row):
    return dict(slot=row['slot_id'],traffic_arrivals=row.get('traffic_arrivals'),
        task_births=[{k:v for k,v in e.items()if k!='latency_ms'}for e in row.get('ue_service_events',[])if e.get('kind')=='task_generated'])


def _predeployment_run(run,baseline,observe_slots,keys):
    """Stop at baseline; no AFTER metric or future decision enters this return."""
    config=read(Path(run)/'config_snapshot.json')
    previews={}
    for item in records(run,'xapp_decision_previews'):
        if type(item.get('slot'))is not int:continue
        if item['slot']>baseline:break
        previews[item['slot']]=item
    if not previews:raise ValueError('predeployment_preview_missing')
    decision=max(previews);preview=PreviewStore(run).read(decision)
    index=WindowIndex(config,population=True);builder=FrameBuilder(config,population=True)
    prefix=hashlib.sha256();frame=None;first=None;active=False;history=[]
    for row in records(run,'rt_sys_trace'):
        slot=row['slot_id']
        if slot>decision:break
        if first is None:first=slot
        users=trace_users(row);services=row.get('ue_services',{});ap=row.get('ap_services',{}).get('cells',{})
        index.observe(slot,users,services,ap,row.get('ue_service_events',[]))
        active|=bool(row.get('xapp_bindings'))
        frame=builder.frame(slot,users,services,ap,row.get('xapp_bindings')or{})
        prefix.update(json.dumps(_exogenous(row),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        if slot in previews and decision-500<slot<=decision and slot>first:
            historic=index.window(keys,max(first,slot-observe_slots),slot,through=slot)
            historic['slot_duration_ms']=config.get('manifest',{}).get('slot_duration_ms',.5)
            history.append(dict(frame=_hydrate_current_controls(origin_frame(frame,historic),previews[slot]),preview=deepcopy(previews[slot])))
    if frame is None or frame['slot']!=decision:raise ValueError('exact_decision_frame_missing')
    start=max(first,decision-observe_slots)
    if start>=decision:raise ValueError('before_observation_window_missing')
    window=index.window(keys,start,decision,through=decision)
    window['slot_duration_ms']=config.get('manifest',{}).get('slot_duration_ms',.5)
    frame=_hydrate_current_controls(origin_frame(frame,window),preview)
    return dict(frame=frame,preview=preview,config=config,prefix_exogenous_sha256=prefix.hexdigest(),
        prefix_controllers_off=not active,decision_slot=decision,scoring_window_start=start,
        scoring_window_slots=decision-start,exact_baseline_decision=decision==baseline,
        exact_before_window=start==baseline-observe_slots and decision==baseline,history=history)


def _verified_models(run):
    entries=read(Path(run)/'xapp_hot_start.json').get('models',{});pins={}
    if not entries:raise ValueError('model_pins_missing')
    for name,item in entries.items():
        path=(Path(run)/item['checkpoint']).resolve()
        if not path.is_relative_to(Path(run).resolve())or not path.is_file()or file_sha(path)!=item.get('sha256'):
            raise ValueError('model_pin_mismatch')
        pins[name]=item['sha256']
    return pins


def collect_predeployment_case(batch,member,jobs_root,*,kpis,common_origin=None):
    """Extract scoreable BEFORE inputs, labeling historical protocol limitations."""
    if member.get('phase')!='complete'or not member.get('comparison'):raise ValueError('completed_comparison_required')
    comparison=member['comparison'];baseline=comparison.get('baseline',{}).get('slot')
    application=comparison.get('application_slot');n=batch['observe_slots']
    fixed=comparison.get('fixed_combination_observation_window')is True
    if type(baseline)is not int or (type(application)is not int and not fixed)or (type(application)is int and application<=baseline):
        raise ValueError('predeployment_boundary_invalid')
    run=Path(jobs_root)/member['job_id']/'data/run'
    pre=_predeployment_run(run,baseline,n,kpis)
    if type(application)is int and pre['decision_slot']>=application:raise ValueError('future_proposal_rejected')
    protocol=member_protocol(batch,member);verification=[]
    if protocol['metadata_primary_candidate']:
        try:
            expected=batch['context']['evaluation_origin'];guard=read(run/'evaluation_origin_verified.json')
            checks=dict(exact_predeployment_decision=pre['exact_baseline_decision'],exact_scoring_window=pre['exact_before_window'],
                native_prefix=pre['prefix_controllers_off'],cold_proposals=all(a.get('mode')=='cold_activation'for a in pre['preview']['apps']),
                source_archive_verified=bool(common_origin and common_origin.get('status')=='ready' and common_origin['proof']['dynamics_sha256']==expected['dynamics_sha256']),
                guard_matches=guard.get('origin_id')==expected['origin_id'] and all(guard.get(k)is True for k in('verified_before_activation','observed_prefix_equal','configuration_equal','models_equal','network_dynamics_code_equal')),
                configuration_matches=comparable(origin_configuration(pre['config']),expected['configuration']),
                exogenous_prefix_matches=pre['prefix_exogenous_sha256']==expected['prefix_exogenous_sha256'],
                model_pins_match=_verified_models(run)==expected['model_sha256'],
                before_readings_match=all(comparable(pre['frame']['states'].get('kpi:'+k,{}).get('value'),expected['before_frame']['states'].get('kpi:'+k,{}).get('value'))for k in kpis))
            protocol['verification_checks']=checks
            verification=[key for key,value in checks.items()if not value]
            protocol['primary_validation_eligible']=not verification
        except (OSError,ValueError,KeyError,TypeError)as error:verification=['origin_verification_unavailable:'+type(error).__name__]
    protocol['reasons']+=verification
    if not pre['exact_baseline_decision']:protocol['reasons'].append('lagged_predeployment_preview')
    if not pre['exact_before_window']:protocol['reasons'].append('different_scoring_and_evaluation_before_windows')
    config=pre.pop('config');ric=config.get('configs',{}).get('ric',{})
    return dict(group_id=member['group_id'],app_ids=deepcopy(member['app_ids']),frame=pre['frame'],preview=pre['preview'],
        protocol=protocol,timing=dict(decision_slot=pre['decision_slot'],baseline_slot=baseline,application_slot=application,
            observe_slots=n,scoring_window_start=pre['scoring_window_start'],scoring_window_slots=pre['scoring_window_slots'],
            lag_to_baseline_ms=(baseline-pre['decision_slot'])*config.get('manifest',{}).get('slot_duration_ms',.5)),
        arbitration=dict(mode=ric.get('direct_conflict_mitigation','uniform_random'),window_ms=ric.get('mitigation_window_ms',100),held_state_reconstructed=False),
        history=pre['history'],
        provenance=dict(batch_id=batch['id'],job_id=member['job_id'],seed=batch.get('seed'),
            configuration_sha256=digest(origin_configuration(config)),prefix_exogenous_sha256=pre['prefix_exogenous_sha256'],
            source_archive_sha256=batch.get('worker_source_archive_sha256'),after_kpis_used=False),
        label_source=dict(run_path=str(run),before_window=deepcopy(comparison.get('baseline_window')or dict(window_start_slot=baseline-n,slot=baseline,window_slots=n)),
            after_window=deepcopy(comparison.get('after_window')or dict(window_start_slot=application-1,slot=application+n-1,window_slots=n))))


def candidate_flags(graph,frame,ranges,preview,app_ids,settings=None):
    """Eligibility is frozen before reading labels; OFF is a diagnostic arm."""
    if not app_ids:return dict(pool='off_diagnostic',eligible=False,reason='OFF_control_not_a_candidate_combination')
    catalog=select_combinations(graph,frame,dict(ranges=ranges,questions=[],snapshot={'slot':frame['slot']}),
        settings=settings or DEFAULT_SELECTION,decision_context=dict(frame=frame,preview=preview,requested_slot=frame['slot'],lag_ms=0))
    target=frozenset(app_ids)
    for key,pool in(('groups','primary'),('reserve_groups','reserve'),('excluded_groups','excluded_direct_action')):
        match=next((g for g in catalog.get(key,[])if frozenset(g['app_ids'])==target),None)
        if match:return dict(pool=pool,eligible=pool=='primary',reason=(match.get('initial_selection')or{}).get('reason'),
            selection_algorithm=catalog.get('selection_algorithm'),metadata=deepcopy(match.get('initial_selection')))
    return dict(pool='not_candidate',eligible=False,reason='not_in_current_predeployment_candidate_catalog',
        selection_algorithm=catalog.get('selection_algorithm'),missing_kpis=catalog.get('missing_kpis',[]))


def make_scoring_case(pre,spec,graph,*,case_id,scene,graph_id=None,candidate_settings=None):
    # A relative request has no numeric interval without its genuine BEFORE
    # reference. Keep the preregistered case in the ledger, but do not invent a
    # bound, borrow another seed's reference, or inspect an AFTER observation.
    missing=[]
    if not spec.get('use_archived_ranges'):
        nodes={n['key']:n for n in graph['nodes']if n['kind']=='kpi'}
        for row in spec['ranges']:
            if any(row.get(k)=='current'or isinstance(row.get(k),dict)for k in('lower','upper')):
                current=pre['frame'].get('states',{}).get(nodes.get(row['kpi'],{}).get('id',''),{}).get('value')
                if type(current)not in(int,float)or not math.isfinite(current):missing.append(row['kpi'])
    resolution=dict(status='unavailable',reason='intent_reference_unavailable',missing_kpis=list(dict.fromkeys(missing)),
        requested_ranges=deepcopy(spec['ranges']))if missing else None
    ranges=[]if resolution else(deepcopy(spec['ranges'])if spec.get('use_archived_ranges')else resolve_study_intent(spec,graph,pre['frame']))
    case={k:deepcopy(pre[k])for k in('group_id','app_ids','frame','preview','protocol','timing','provenance')}
    source_identity=('study-origin:'+pre['provenance']['scoring_origin_id']if pre['provenance'].get('scoring_origin_id')else
        'study-run:'+pre['provenance']['job_id']);graph_sha=digest(graph)
    history=[dict(**deepcopy(x),source_identity=source_identity,graph_context_sha256=graph_sha)for x in pre.get('history',[])]
    case.update(case_id=case_id,intent_id=spec['id'],scene=scene,graph_id=graph_id or scene,ranges=ranges,
        candidate=(dict(pool='unresolved_intent',eligible=False,reason='intent_reference_unavailable')if resolution else
            candidate_flags(graph,pre['frame'],ranges,pre['preview'],pre['app_ids'],candidate_settings)),
        options=dict(method='goal_obstruction_v2',showDashed=False,arbitration=pre['arbitration'],normalization=normalization(ranges,pre['frame']),
            sourceIdentity=source_identity,graphContextSha256=graph_sha,history=history))
    case['cohort_id']=digest(dict(scene=scene,seed=pre['provenance']['seed'],configuration=pre['provenance']['configuration_sha256'],
        window=pre['timing']['observe_slots'],protocol=pre['protocol']['name']))
    case['independence_group']=digest(dict(cohort=case['cohort_id'],apps=sorted(case['app_ids']),prefix=pre['provenance']['prefix_exogenous_sha256']))
    source=dict(**pre['label_source'],ranges=ranges,protocol=pre['protocol'],candidate=case['candidate'],scene=scene,
        intent_id=spec['id'],cohort_id=case['cohort_id'],independence_group=case['independence_group'],provenance=pre['provenance'])
    if resolution:case['intent_resolution']=deepcopy(resolution);source['intent_resolution']=deepcopy(resolution)
    return case,source


def compact_contexts(data):
    """Store one BEFORE context per physical run, shared by derived intents."""
    contexts={}
    for case in data['cases']:
        options=case['options']
        context=dict(frame=case.pop('frame'),preview=case.pop('preview'),
            options={k:options.pop(k)for k in('arbitration','sourceIdentity','graphContextSha256','history')if k in options})
        ident=digest(context);contexts.setdefault(ident,context);case['context_id']=ident
    data['contexts']=contexts;data['context_storage']='deduplicated_predeployment_only'
    return data


def materialize_case(data,case):
    context=data.get('contexts',{}).get(case.get('context_id'),{})
    return {**context,**case,'options':{**context.get('options',{}),**case.get('options',{})}}


def resolve_study_intent(spec,graph,frame):
    """An explicit experimental intent; never insert it into the user ledger."""
    if not isinstance(spec.get('id'),str)or not spec['id']:raise ValueError('study_intent_id_required')
    result=resolve_ranges(dict(ranges=deepcopy(spec['ranges']),questions=[]),graph,frame)
    if result.get('questions'):raise ValueError('study_intent_cannot_be_resolved')
    ranges=result['ranges']
    for row in ranges:
        row['anchors']=[dict(intent_id='study:'+spec['id'],snapshot={k:frame.get(k)for k in('slot','time_ms')},
            current=row.get('current'),lower=row.get('lower'),upper=row.get('upper'),
            lower_inclusive=row.get('lower_inclusive',True),upper_inclusive=row.get('upper_inclusive',True),
            declared_experimental_intent=True)]
    return ranges


def extract_inputs(jobs_root,influences,output,*,intents=None,batch_ids=None,include_legacy=False,confidence_level=.90,candidate_settings=None):
    output=Path(output);inventory=inventory_batches(jobs_root);write(output/'inventory.json',inventory)
    graph=graph_for_study(influences,confidence_level=confidence_level);cases=[];sources={};excluded=[];selected=set(batch_ids or [])
    scene=read(influences).get('scenario',{}).get('id','operator_selected_context')
    fitted_seeds=calibration_seeds(influences)
    for entry in inventory['batches']:
        if selected and entry['batch_id']not in selected:continue
        batch=read(entry['path']);common=None
        if any(m['protocol']['metadata_primary_candidate']for m in entry['members']):
            common=prepare_origin(batch,jobs_root,observe_slots=batch['observe_slots'])
        specs=intents or[dict(id='archived:'+batch['id'],ranges=deepcopy(batch['ranges']),use_archived_ranges=True)]
        keys=list(dict.fromkeys(r['kpi']for spec in specs for r in spec['ranges']))
        for member in batch['members']:
            try:
                if not include_legacy and not member_protocol(batch,member)['metadata_primary_candidate']:
                    raise ValueError('legacy_protocol_excluded_from_primary_extraction')
                pre=collect_predeployment_case(batch,member,jobs_root,kpis=keys,common_origin=common)
                pre['provenance'].update(evaluation_partition='archived_development',calibration_seed_overlap=batch.get('seed')in fitted_seeds)
                if not include_legacy and not pre['protocol']['primary_validation_eligible']:raise ValueError('predeployment_origin_not_verified')
                if batch.get('context',{}).get('contracts')!=graph.get('contract_sha256'):raise ValueError('controller_contract_version_mismatch')
                for spec in specs:
                    case_id=':'.join((batch['id'],member['job_id'],spec['id']))
                    case,source=make_scoring_case(pre,spec,graph,case_id=case_id,scene=scene,candidate_settings=candidate_settings)
                    cases.append(case);sources[case_id]=source
            except (OSError,ValueError,KeyError,TypeError)as error:excluded.append(dict(batch_id=batch['id'],job_id=member.get('job_id'),reason=str(error)))
    data=dict(schema=SCHEMA,stage='predeployment_inputs',graph=graph,cases=cases,
        calibration=dict(path=str(influences),sha256=file_sha(influences),graph_sha256=digest(graph),policy='significant_relations',confidence_level=confidence_level,seeds=fitted_seeds),
        declared_experimental_intents=deepcopy(intents),excluded=excluded,after_values_included=False,new_simulations=0)
    compact_contexts(data);write_immutable(output/'scoring-inputs.json',data)
    write_immutable(output/'label-sources.json',dict(schema=SCHEMA,sources=sources,after_values_included=False))
    return data


def extract_worker_inputs(artifact,influences_by_scene,output,*,intents,confidence_level=.90,candidate_settings=None):
    """Read the new sealed physical-study layout without touching live queues.

    ``influences_by_scene`` explicitly maps scene IDs to their own calibration
    files. No slope is borrowed from a different traffic scene. OFF arms are
    scored as diagnostics but never enter primary candidate-order statistics.
    """
    artifact=Path(artifact);output=Path(output);graphs={scene:graph_for_study(path,confidence_level=confidence_level)for scene,path in influences_by_scene.items()}
    plan_path=next((p/'physical-plan.json'for p in(artifact,*list(artifact.parents)[:3])if (p/'physical-plan.json').is_file()),None)
    if plan_path is None:raise ValueError('preregistered_physical_plan_missing')
    plan=read(plan_path);expected_plan=plan['plan_sha256'];arms={x['id']:x['app_ids']for x in plan['arms']}
    cases=[];sources={};excluded=[];keys=list(dict.fromkeys(r['kpi']for spec in intents for r in spec['ranges']))
    archive=artifact/'source.tgz';archive_cache={}
    for seal_path in sorted((artifact/'workers').glob('*/checkpoints/*.json')):
        seal=read(seal_path)
        if seal.get('completed')is not True:continue
        case_spec=seal['case'];worker=seal_path.parents[1];ident=case_spec['id'];scene=case_spec['scene']
        try:
            if scene not in graphs:raise ValueError('scene_specific_calibration_missing')
            if seal.get('physical_plan_sha256')!=expected_plan:raise ValueError('preregistered_plan_mismatch')
            if scene not in plan['scenarios']or case_spec['seed']not in plan['seeds']or arms.get(case_spec['arm'])!=case_spec.get('app_ids'):
                raise ValueError('case_not_in_preregistered_design')
            jobs=worker/'jobs';run=jobs/ident/'data/run';request=read(jobs/ident/'data/request.json')
            if request.get('case')!=case_spec:raise ValueError('sealed_case_request_mismatch')
            if request.get('physical_plan_sha256')!=seal.get('physical_plan_sha256'):raise ValueError('sealed_plan_mismatch')
            pin=seal.get('source_sha256')or request.get('source_archive_sha256')
            if pin not in archive_cache:archive_cache[pin]=archived_dynamics_fingerprint(archive,pin)
            # Hash verification parses no AFTER KPI values and prevents a
            # silently replaced trace/preview/model from becoming test data.
            for relative,pinned in seal.get('files_sha256',{}).items():
                path=(jobs/ident/relative).resolve()
                if not path.is_relative_to((jobs/ident).resolve())or not path.is_file()or file_sha(path)!=pinned:
                    raise ValueError('sealed_run_file_mismatch')
            origin=read(worker/'origins'/(case_spec['origin_key']+'.json'))
            if digest({k:v for k,v in origin.items()if k!='origin_id'})!=origin.get('origin_id'):raise ValueError('origin_proof_digest_invalid')
            if origin['dynamics_sha256']!=archive_cache[pin]:raise ValueError('origin_dynamics_mismatch')
            if origin.get('origin_id')!=seal.get('origin_id')or origin.get('seed')!=case_spec['seed']:raise ValueError('sealed_origin_mismatch')
            n=request['observe_slots'];baseline=origin['baseline_slot'];off=not case_spec.get('app_ids')
            if n!=plan['observe_slots']or baseline!=plan['baseline_slot']or origin.get('observe_slots')!=n:raise ValueError('preregistered_window_mismatch')
            guard=read(run/'evaluation_origin_verified.json')if not off else{}
            comparison=dict(baseline={'slot':baseline},baseline_origin='fresh_run_all_xapps_off',
                baseline_window=dict(window_start_slot=baseline-n,slot=baseline,window_slots=n),
                after_window=dict(window_start_slot=baseline,slot=baseline+n,window_slots=n),
                fixed_combination_observation_window=True,evaluation_origin=guard)
            batch=dict(id='physical-study:'+case_spec['origin_key'],observe_slots=n,seed=case_spec['seed'],
                context=dict(evaluation_origin=origin),worker_source_archive_sha256=pin)
            member=dict(phase='complete',job_id=ident,group_id='study-arm:'+case_spec['arm'],app_ids=case_spec.get('app_ids',[]),comparison=comparison)
            pre=collect_predeployment_case(batch,member,jobs,kpis=keys,common_origin={'status':'ready','proof':origin})
            pre['provenance'].update(physical_plan_sha256=seal['physical_plan_sha256'],worker_id=seal.get('worker_id'),
                calibration_scenario=scene,sealed_source_archive_sha256=pin,evaluation_partition='preregistered_held_out',
                calibration_seed_overlap=case_spec['seed']in calibration_seeds(influences_by_scene[scene]))
            if off:
                pre['protocol'].update(name='off_control_diagnostic',primary_validation_eligible=False)
                pre['protocol']['reasons'].append('OFF_control_has_no_controller_activation')
            elif not pre['protocol']['primary_validation_eligible']:raise ValueError('physical_study_origin_not_verified:'+','.join(pre['protocol']['reasons']))
            for spec in intents:
                case_id=':'.join((case_spec['origin_key'],ident,spec['id']))
                case,source=make_scoring_case(pre,spec,graphs[scene],case_id=case_id,scene=scene,candidate_settings=candidate_settings)
                cases.append(case);sources[case_id]=source
        except (OSError,ValueError,KeyError,TypeError)as error:excluded.append(dict(case_id=ident,scene=scene,reason=str(error)))
    data=dict(schema=SCHEMA,stage='predeployment_inputs',graphs=graphs,cases=cases,
        calibration={scene:dict(path=str(path),sha256=file_sha(path),graph_sha256=digest(graphs[scene]),seeds=calibration_seeds(path))for scene,path in influences_by_scene.items()},
        declared_experimental_intents=deepcopy(intents),excluded=excluded,after_values_included=False,new_simulations=0)
    compact_contexts(data);write_immutable(output/'scoring-inputs.json',data);write_immutable(output/'label-sources.json',dict(schema=SCHEMA,sources=sources,after_values_included=False))
    return data


def extract_prospective_inputs(artifact,influences_by_scene,output,*,intents,origin_keys=None,confidence_level=.90,candidate_settings=None):
    """Score all planned arms from a sealed OFF run's genuine BEFORE prefix.

    Nonempty arm runs need not exist. Their later OriginGuard is checked only
    during label joining; no proposed action or state is guessed from outcomes.
    """
    artifact=Path(artifact);output=Path(output);wanted=set(origin_keys or [])
    graphs={scene:graph_for_study(path,confidence_level=confidence_level)for scene,path in influences_by_scene.items()}
    cases=[];sources={};excluded=[];blocks=[];keys=list(dict.fromkeys(r['kpi']for spec in intents for r in spec['ranges']))
    archive_cache={}
    for worker_plan in sorted((artifact/'workers').glob('*/plan.json')):
        worker=worker_plan.parent;plan=read(worker_plan);physical=plan['physical'];pin=plan['source_archive_sha256']
        if pin not in archive_cache:archive_cache[pin]=archived_dynamics_fingerprint(artifact/'source.tgz',pin)
        by_origin=defaultdict(list)
        for arm in plan['cases']:by_origin[arm['origin_key']].append(arm)
        for origin_key,arms in sorted(by_origin.items()):
            if wanted and origin_key not in wanted:continue
            off=next((a for a in arms if not a['app_ids']),None)
            if off is None:continue
            if sorted(a['arm']for a in arms)!=sorted(a['id']for a in physical['arms']):
                excluded.append(dict(origin_key=origin_key,reason='preregistered_physical_arm_grid_incomplete'));continue
            seal_path=worker/'checkpoints'/(off['id']+'.json')
            if not seal_path.is_file():continue
            block_case_start=len(cases)
            try:
                seal=read(seal_path)
                if seal.get('completed')is not True:continue
                if seal.get('case')!=off or seal.get('physical_plan_sha256')!=plan['physical_plan_sha256']:raise ValueError('OFF_plan_seal_mismatch')
                off_job=worker/'jobs'/off['id'];run=off_job/'data/run'
                for relative,pinned in seal.get('files_sha256',{}).items():
                    path=(off_job/relative).resolve()
                    if not path.is_relative_to(off_job.resolve())or not path.is_file()or file_sha(path)!=pinned:raise ValueError('OFF_sealed_file_mismatch')
                origin=read(worker/'origins'/(origin_key+'.json'))
                if digest({k:v for k,v in origin.items()if k!='origin_id'})!=origin['origin_id']:raise ValueError('OFF_origin_digest_invalid')
                if origin['dynamics_sha256']!=archive_cache[pin]or origin['origin_id']!=seal.get('origin_id'):raise ValueError('OFF_origin_source_mismatch')
                scene=off['scene'];graph=graphs[scene];n=physical['observe_slots'];baseline=physical['baseline_slot']
                if origin['baseline_slot']!=baseline or origin['observe_slots']!=n or origin['seed']!=off['seed']:raise ValueError('OFF_origin_recipe_mismatch')
                prefix=_predeployment_run(run,baseline,n,keys)
                checks=dict(exact_baseline=prefix['exact_baseline_decision'],matched_window=prefix['exact_before_window'],
                    controllers_off=prefix['prefix_controllers_off'],cold_proposals=all(a.get('mode')=='cold_activation'for a in prefix['preview']['apps']),
                    exogenous_prefix=prefix['prefix_exogenous_sha256']==origin['prefix_exogenous_sha256'],
                    configuration=comparable(origin_configuration(prefix['config']),origin['configuration']),
                    model_pins=_verified_models(run)==origin['model_sha256'],
                    before_metrics=all(comparable(prefix['frame']['states'].get('kpi:'+k,{}).get('value'),origin['before_frame']['states'].get('kpi:'+k,{}).get('value'))for k in keys))
                if not all(checks.values()):raise ValueError('OFF_prefix_not_verified:'+','.join(k for k,v in checks.items()if not v))
                ric=prefix['config'].get('configs',{}).get('ric',{})
                for arm in arms:
                    is_off=not arm['app_ids'];run_path=worker/'jobs'/arm['id']/'data/run'
                    pre=dict(group_id='study-arm:'+arm['arm'],app_ids=arm['app_ids'],frame=prefix['frame'],preview=prefix['preview'],history=prefix['history'],
                        protocol=dict(name='prospective_certified_OFF_prefix',primary_validation_eligible=not is_off,
                            actual_arm_origin_guard_required=not is_off,actual_arm_origin_guard_pending=not is_off,
                            origin_checks=checks,reasons=['OFF_diagnostic']if is_off else[]),
                        timing=dict(decision_slot=baseline,baseline_slot=baseline,application_slot=None,observe_slots=n,
                            scoring_window_start=baseline-n,scoring_window_slots=n,lag_to_baseline_ms=0),
                        arbitration=dict(mode=ric.get('direct_conflict_mitigation','uniform_random'),window_ms=ric.get('mitigation_window_ms',100),held_state_reconstructed=False),
                        provenance=dict(batch_id='prospective:'+origin_key,job_id=arm['id'],seed=arm['seed'],
                            configuration_sha256=digest(origin_configuration(prefix['config'])),prefix_exogenous_sha256=prefix['prefix_exogenous_sha256'],
                            scoring_origin_job_id=off['id'],scoring_origin_id=origin['origin_id'],source_archive_sha256=pin,
                            evaluation_partition='preregistered_prospective_held_out',calibration_seed_overlap=arm['seed']in calibration_seeds(influences_by_scene[scene]),
                            physical_plan_sha256=plan['physical_plan_sha256'],worker_id=plan['worker_id'],after_kpis_used=False),
                        label_source=dict(run_path=str(run_path),seal_path=str(worker/'checkpoints'/(arm['id']+'.json')),
                            expected_origin_id=origin['origin_id'],requires_actual_origin_guard=not is_off,
                            before_window=dict(window_start_slot=baseline-n,slot=baseline,window_slots=n),
                            after_window=dict(window_start_slot=baseline,slot=baseline+n,window_slots=n)))
                    for spec in intents:
                        case_id=':'.join((origin_key,arm['id'],spec['id']))
                        case,source=make_scoring_case(pre,spec,graph,case_id=case_id,scene=scene,candidate_settings=candidate_settings)
                        cases.append(case);sources[case_id]=source
                blocks.append(dict(origin_key=origin_key,origin_id=origin['origin_id'],off_job_id=off['id'],arms=len(arms),intents=len(intents),
                    after_values_used=False,previews_from='one_certified_OFF_prefix_shared_by_all_arms'))
            except (OSError,ValueError,KeyError,TypeError)as error:
                for partial in cases[block_case_start:]:sources.pop(partial['case_id'],None)
                del cases[block_case_start:]
                excluded.append(dict(origin_key=origin_key,reason=str(error)))
    if wanted and({b['origin_key']for b in blocks}!=wanted or excluded):
        raise ValueError('requested_prospective_blocks_not_ready:'+json.dumps(excluded,sort_keys=True))
    data=dict(schema=SCHEMA,stage='predeployment_inputs',graphs=graphs,cases=cases,prospective_blocks=blocks,
        calibration={scene:dict(path=str(path),sha256=file_sha(path),graph_sha256=digest(graphs[scene]),seeds=calibration_seeds(path))for scene,path in influences_by_scene.items()},
        declared_experimental_intents=deepcopy(intents),excluded=excluded,after_values_included=False,new_simulations=0)
    compact_contexts(data);write_immutable(output/'scoring-inputs.json',data);write_immutable(output/'label-sources.json',dict(schema=SCHEMA,sources=sources,after_values_included=False))
    return data


def score_cases(inputs_path,output,*,method='goal_obstruction_v2',module=None,engine='SmoGoalObstruction',node=None,full_results=False,dispatch_method=None):
    """Persist scores before any label join; the VM has no filesystem API."""
    node=node or shutil.which('node')
    if not node:raise ValueError('Node is required')
    inputs_path=Path(inputs_path);data=read(inputs_path)
    if data.get('after_values_included')is not False:raise ValueError('predeployment_only_inputs_required')
    modules=[ROOT/'smo/web/app.js',ROOT/'smo/web/goal_obstruction.js']
    if module and Path(module).resolve()not in modules:modules.append(Path(module).resolve())
    input_sha=file_sha(inputs_path);module_sha={str(p):file_sha(p)for p in modules}
    output=Path(output)
    if output.exists():
        saved=read(output)
        if saved.get('input_sha256')==input_sha and saved.get('module_sha256')==module_sha and saved.get('method')==method and saved.get('engine')==engine and saved.get('full_projection_results')==full_results and saved.get('dispatch_method')==dispatch_method:return saved
        raise ValueError('refusing_to_overwrite_frozen_scores:'+str(output))
    archive=output.parent/'engine-sources';archive.mkdir(parents=True,exist_ok=True);pinned_modules=[]
    for path in modules:
        target=archive/(module_sha[str(path)]+'-'+path.name)
        if not target.exists():target.write_bytes(path.read_bytes())
        if file_sha(target)!=module_sha[str(path)]:raise ValueError('scoring_module_pin_mismatch')
        pinned_modules.append(target)
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
const input=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(input);
for(const p of input.modules)vm.runInContext(fs.readFileSync(p,'utf8'),c);
const engine=c[input.engine];if(!engine?.review)throw Error('Requested scoring engine missing');
const compact=r=>input.full_results?r:{method:r.method,schema:r.schema,status:r.status,reason:r.reason,
 kpi_forecast_performed:r.kpi_forecast_performed,observed_after_kpis_used:r.observed_after_kpis_used,
 groups:r.groups?.map(g=>Object.fromEntries(['group_id','app_ids','score','score_status','reason','raw_obstruction_strength',
  'reference_effect','reference_effect_score','goal_components','projection_coverage','model_coverage_complete',
  'history_sample_count','reference_subset_count','reference_outcome_count','no_active_push'].filter(k=>k in g).map(k=>[k,g[k]])))};
const scores=input.data.cases.map(x=>({case_id:x.case_id,intent_id:x.intent_id,cohort_id:x.cohort_id,
 independence_group:x.independence_group,protocol:x.protocol,provenance:x.provenance,
 candidate:x.candidate,scene:x.scene,
 result:(()=>{if(x.intent_resolution?.status==='unavailable')return {method:input.method,status:'unavailable',reason:x.intent_resolution.reason,
 groups:[{group_id:x.group_id,app_ids:x.app_ids,score:null,raw_obstruction_strength:null,score_status:'unavailable',reason:x.intent_resolution.reason}],observed_after_kpis_used:false};
 const context=input.data.contexts?.[x.context_id]||{};return compact(engine.review(input.data.graphs?.[x.graph_id]||input.data.graph,
 x.frame||context.frame,[{id:x.group_id,app_ids:x.app_ids}],x.ranges,x.preview||context.preview,{...context.options,...x.options,method:input.dispatch_method||input.method}));})()}));
if(before!==JSON.stringify(input))throw Error('Scoring input mutated');
process.stdout.write(JSON.stringify(scores));"""
    values=json.loads(subprocess.check_output([node,'-e',script],cwd=ROOT,text=True,
        input=json.dumps(dict(data=data,modules=[str(p)for p in pinned_modules],engine=engine,method=method,dispatch_method=dispatch_method,full_results=full_results))))
    result=dict(schema=SCHEMA,stage='scores_before_label_join',method=method,engine=engine,
        input_path=str(inputs_path),input_sha256=input_sha,module_sha256=module_sha,
        archived_modules={str(p):str(q)for p,q in zip(modules,pinned_modules)},
        label_sources_sha256=file_sha(inputs_path.with_name('label-sources.json'))if inputs_path.with_name('label-sources.json').is_file()else None,
        scores=values,after_values_used=False,new_simulations=0,full_projection_results=full_results,
        reproduce_full_projections='Run the same immutable inputs and pinned module hashes with full_results=True.')
    if data.get('graph_variant'):result['graph_variant']=deepcopy(data['graph_variant'])
    if dispatch_method:result.update(dispatch_method=dispatch_method,engine_method=dispatch_method)
    write_immutable(output,result);return result


def derive_graph_variant_inputs(base_inputs,graphs,output,*,variant_id,provenance_paths):
    """Copy a sealed BEFORE ledger, replacing only admitted-edge statistics.

    Graph/history pins are updated together. Existing proposal/state/range,
    candidate, normalization and label-source bytes remain frozen. A variant
    cannot add or remove an xApp, RCP, KPI or causal edge.
    """
    base_inputs=Path(base_inputs);output=Path(output);original=read(base_inputs)
    if original.get('after_values_included')is not False:raise ValueError('predeployment_only_inputs_required')
    if set(graphs)!=set(original['graphs']):raise ValueError('variant_graph_contexts_must_match')
    for key,graph in graphs.items():
        base=original['graphs'][key]
        if sorted((n['id'],n['kind'])for n in graph['nodes'])!=sorted((n['id'],n['kind'])for n in base['nodes']):
            raise ValueError('graph_variant_changed_nodes:'+key)
        topology=lambda g:sorted((e['id'],e['source'],e['target'],e['kind'])for e in g['edges'])
        if topology(graph)!=topology(base):raise ValueError('graph_variant_changed_admitted_edges:'+key)
        if any(n!=next(x for x in base['nodes']if x['id']==n['id'])for n in graph['nodes']):
            raise ValueError('graph_variant_changed_node_contracts:'+key)
    variant=dict(id=variant_id,base_input_path=str(base_inputs),base_input_sha256=file_sha(base_inputs),
        base_label_sources_sha256=file_sha(base_inputs.with_name('label-sources.json')),
        base_graph_sha256={k:digest(v)for k,v in original['graphs'].items()},graph_sha256={k:digest(v)for k,v in graphs.items()},
        provenance_files_sha256={str(Path(p)):file_sha(p)for p in provenance_paths},
        original_methods_and_seals_modified=False,topology_unchanged=True,after_validation_kpis_used=False)
    data=deepcopy(original);data['graphs']=deepcopy(graphs);data['graph_variant']=variant
    data.pop('contexts',None);data.pop('context_storage',None);data['cases']=[]
    for saved in original['cases']:
        case=deepcopy(materialize_case(original,saved));case.pop('context_id',None)
        pin=digest(graphs[case['graph_id']]);case['options']['graphContextSha256']=pin
        for envelope in case['options'].get('history',[]):envelope['graph_context_sha256']=pin
        data['cases'].append(case)
    compact_contexts(data);write_immutable(output/'scoring-inputs.json',data)
    source=base_inputs.with_name('label-sources.json');target=output/'label-sources.json'
    if target.exists():
        if target.read_bytes()!=source.read_bytes():raise ValueError('variant_label_sources_changed')
    else:target.write_bytes(source.read_bytes())
    return data


def seal_graph_variant(base_seal_path,graph_paths,output,*,profile_plan_path,provenance_paths=()):
    """Seal the separately preregistered gain variant before control admission."""
    base_seal_path=Path(base_seal_path);base=read(base_seal_path);output=Path(output);plan=read(profile_plan_path)
    if base.get('after_values_used')is not False or base.get('nonempty_arm_started_before_scoring')is not False:
        raise ValueError('prospective_base_seal_required')
    if digest({k:v for k,v in plan.items()if k!='plan_sha256'})!=plan.get('plan_sha256'):
        raise ValueError('profile_calibration_plan_digest_mismatch')
    variant_id=plan['extra_variant']['id']
    if variant_id!='net_profile_gain_obstruction_v1':raise ValueError('unregistered_graph_variant')
    original=read(base['input_path'])
    if file_sha(base['input_path'])!=base['input_sha256']:raise ValueError('prospective_base_inputs_changed')
    sources_path=Path(base['input_path']).with_name('label-sources.json')
    if file_sha(sources_path)!=base['label_sources_sha256']:raise ValueError('prospective_base_sources_changed')
    for scored in base['scores'].values():
        if file_sha(scored['path'])!=scored['sha256']:raise ValueError('prospective_base_scores_changed')
    sources=read(sources_path)['sources']
    for case in original['cases']:
        if case['app_ids']and(Path(sources[case['case_id']]['run_path']).parent/'request.json').exists():
            raise ValueError('nonempty_arm_already_started_before_variant_seal')
        if case['provenance'].get('seed')not in plan['validation_seeds_excluded']:
            raise ValueError('variant_validation_seed_not_registered')
    graphs={key:read(path)for key,path in graph_paths.items()}
    provenance=[profile_plan_path,*graph_paths.values(),*provenance_paths]
    data=derive_graph_variant_inputs(base['input_path'],graphs,output,variant_id=variant_id,provenance_paths=provenance)
    pipeline=freeze_pipeline(output/'pipeline-sources')
    score_path=output/('scores-'+variant_id+'.json')
    result=score_cases(output/'scoring-inputs.json',score_path,method=variant_id,dispatch_method='net_intent_obstruction_v1',
        module=ROOT/'smo/web/intent_obstruction.js',engine='SmoIntentObstruction')
    for name,pin in result['module_sha256'].items():
        if pipeline['files'][str(Path(name).relative_to(ROOT))]['sha256']!=pin:
            raise ValueError('projection_code_changed_during_variant_scoring')
    if any(case['app_ids']and(Path(sources[case['case_id']]['run_path']).parent/'request.json').exists()for case in original['cases']):
        raise ValueError('nonempty_arm_started_during_variant_scoring')
    pins=dict(base_seal_path=str(base_seal_path),base_seal_sha256=file_sha(base_seal_path),
        origin_key=base['origin_key'],blocks=deepcopy(base['blocks']),variant_id=variant_id,engine_method='net_intent_obstruction_v1',
        input_path=str(output/'scoring-inputs.json'),input_sha256=file_sha(output/'scoring-inputs.json'),
        label_sources_sha256=file_sha(output/'label-sources.json'),graph_variant=data['graph_variant'],
        score_path=str(score_path),score_sha256=file_sha(score_path),
        scores={variant_id:dict(path=str(score_path),sha256=file_sha(score_path),count=len(result['scores']),
            valid_scores=sum((x['result'].get('groups')or[{}])[0].get('score')is not None for x in result['scores']))},
        physical_arms=base['physical_arms'],intents=base['intents'],contexts=len(data['contexts']),
        module_sha256=result['module_sha256'],
        pipeline_sha256=pipeline['pipeline_sha256'],cases=len(data['cases']),
        valid_scores=sum((x['result'].get('groups')or[{}])[0].get('score')is not None for x in result['scores']))
    seal_path=output/'prediction-seal.json'
    if seal_path.exists():
        old=read(seal_path)
        if any(old.get(k)!=v for k,v in pins.items()):raise ValueError('prospective_graph_variant_seal_changed')
        return old
    seal=dict(schema='oran.prospective-score-seal.v1',created_utc=datetime.now(timezone.utc).isoformat(),**pins,
        after_values_used=False,observed_validation_after_kpis_used=False,nonempty_arm_started_before_scoring=False,
        gate_opened=False,algorithm_tuned_to_test_labels=False,
        original_methods_and_seals_modified=False,score_parameters_fitted_to_validation_labels=False)
    write_immutable(seal_path,seal);return seal


def seal_prospective_block(artifact,influences_by_scene,intents_path,output,*,origin_key,preregistration_path,numeric_policy_path):
    """Freeze all three declared methods while nonempty arms remain gated."""
    artifact=Path(artifact);output=Path(output);catalog=read(intents_path);specs=catalog['intents']if isinstance(catalog,dict)else catalog
    before_started=[]
    for path in (artifact/'workers').glob('*/plan.json'):
        plan=read(path)
        for arm in plan['cases']:
            if arm['origin_key']==origin_key and arm['app_ids']and (path.parent/'jobs'/arm['id']/'data/request.json').exists():before_started.append(arm['id'])
    if before_started:raise ValueError('nonempty_arm_already_started_before_prospective_seal')
    data=extract_prospective_inputs(artifact,influences_by_scene,output,intents=specs,origin_keys=[origin_key])
    if len(data['prospective_blocks'])!=1 or data['excluded']:raise ValueError('one_complete_certified_OFF_block_required')
    block=data['prospective_blocks'][0]
    if len(data['cases'])!=block['arms']*len(specs):raise ValueError('prospective_case_grid_incomplete')
    prereg=read(preregistration_path);allowed={x['id']for x in prereg['methods']};scores={}
    methods=('goal_obstruction_v2','net_intent_obstruction_v1','temporal_net_intent_obstruction_v1')
    if set(methods)!=allowed:raise ValueError('preregistered_method_set_mismatch')
    pipeline=freeze_pipeline(output/'pipeline-sources');module_set={}
    for method in methods:
        dest=output/('scores-'+method+'.json')
        result=score_cases(output/'scoring-inputs.json',dest,method=method,
            module=None if method=='goal_obstruction_v2'else ROOT/'smo/web/intent_obstruction.js',
            engine='SmoGoalObstruction'if method=='goal_obstruction_v2'else'SmoIntentObstruction')
        for name,pin in result['module_sha256'].items():
            if name in module_set and module_set[name]!=pin:raise ValueError('scoring_module_changed_between_methods')
            module_set[name]=pin
        scores[method]=dict(path=str(dest),sha256=file_sha(dest),count=len(result['scores']),
            valid_scores=sum((x['result'].get('groups')or[{}])[0].get('score')is not None for x in result['scores']))
    seal_path=output/'prediction-seal.json'
    pins=dict(input_sha256=file_sha(output/'scoring-inputs.json'),label_sources_sha256=file_sha(output/'label-sources.json'),
        scores=scores,module_sha256=module_set,pipeline_sha256=pipeline['pipeline_sha256'],
        preregistration_sha256=file_sha(preregistration_path),intent_catalog_sha256=file_sha(intents_path),numeric_policy_sha256=file_sha(numeric_policy_path))
    if seal_path.exists():
        old=read(seal_path)
        if any(old.get(k)!=v for k,v in pins.items()):raise ValueError('prospective_prediction_seal_changed')
        return old
    seal=dict(schema='oran.prospective-score-seal.v1',origin_key=origin_key,created_utc=datetime.now(timezone.utc).isoformat(),
        blocks=data['prospective_blocks'],physical_arms=block['arms'],intents=len(specs),cases=len(data['cases']),contexts=len(data['contexts']),
        input_path=str(output/'scoring-inputs.json'),**pins,after_values_used=False,
        nonempty_arm_started_before_scoring=False,gate_opened=False,algorithm_tuned_to_test_labels=False)
    write_immutable(seal_path,seal);return seal


def _observed_windows(source):
    run=Path(source['run_path']);config=read(run/'config_snapshot.json');index=WindowIndex(config,population=True)
    before,after=source['before_window'],source['after_window'];end=after['slot'];through=None
    for row in records(run,'rt_sys_trace'):
        slot=row['slot_id']
        if slot>end:break
        index.observe(slot,trace_users(row),row.get('ue_services',{}),row.get('ap_services',{}).get('cells',{}),row.get('ue_service_events',[]));through=slot
    keys=[r['kpi']for r in source['ranges']]
    return (index.window(keys,before['window_start_slot'],before['slot'],through=through or-1),
        index.window(keys,after['window_start_slot'],after['slot'],through=through or-1))


def _actual_arm_ready(source):
    """Check completion/origin before opening an arm's AFTER telemetry."""
    if not source.get('seal_path'):return True,None
    seal_path=Path(source['seal_path'])
    if not seal_path.is_file():return False,'awaiting_completed_sealed_run'
    seal=read(seal_path)
    if seal.get('completed')is not True:return False,'awaiting_completed_sealed_run'
    if seal.get('origin_id')!=source.get('expected_origin_id'):return False,'actual_arm_origin_mismatch'
    job=Path(source['run_path']).parents[1]
    for relative,pin in seal.get('files_sha256',{}).items():
        path=(job/relative).resolve()
        if not path.is_relative_to(job.resolve())or not path.is_file()or file_sha(path)!=pin:return False,'actual_arm_sealed_file_mismatch'
    if source.get('requires_actual_origin_guard'):
        guard=read(Path(source['run_path'])/'evaluation_origin_verified.json')
        if guard.get('origin_id')!=source['expected_origin_id']or guard.get('after_kpis_used')is not False or not all(
            guard.get(k)is True for k in('verified_before_activation','observed_prefix_equal','configuration_equal','models_equal','network_dynamics_code_equal')):
            return False,'actual_arm_origin_guard_failed'
    return True,None


def ordering(rows,*,risk_key='raw_risk',target_key='observed_deterioration'):
    valid=[r for r in rows if type(r.get(risk_key))in(int,float)and math.isfinite(r[risk_key])and type(r.get(target_key))in(int,float)and math.isfinite(r[target_key])]
    # Preregistered numeric-policy addendum: descriptive ties only. The raw
    # scores, strict interval attainment and Evaluation are never rounded.
    x=[round(r[risk_key]/(100 if risk_key=='reference_effect_score'else 1),12)for r in valid]
    y=[round(r[target_key]/(100 if target_key=='observed_metric'else 1),12)for r in valid]
    def ranks(values):return [1+sum(v<a for v in values)+(sum(v==a for v in values)-1)/2 for a in values]
    def corr(a,b):
        if len(a)<2:return None
        ma,mb=sum(a)/len(a),sum(b)/len(b);d=math.sqrt(sum((v-ma)**2 for v in a)*sum((v-mb)**2 for v in b))
        return sum((u-ma)*(v-mb)for u,v in zip(a,b))/d if d else None
    concordant=discordant=risk_ties=0
    for i in range(len(x)):
        for j in range(i):
            if y[i]==y[j]:continue
            value=(x[i]-x[j])*(y[i]-y[j])
            if value>0:concordant+=1
            elif value<0:discordant+=1
            else:risk_ties+=1
    pairs=concordant+discordant+risk_ties
    rho=corr(ranks(x),ranks(y))
    reason='too_few_valid_rows'if len(valid)<2 else'constant_risk_and_target'if len(set(x))<2 and len(set(y))<2 else'constant_risk'if len(set(x))<2 else'constant_target'if len(set(y))<2 else None
    return dict(n=len(valid),missing=len(rows)-len(valid),spearman=rho,spearman_unavailable_reason=reason,pearson=corr(x,y),
        pair_concordance=(concordant+.5*risk_ties)/pairs if pairs else None,
        concordant=concordant,discordant=discordant,risk_ties=risk_ties,comparable_pairs=pairs,
        distinct_risk_values=len(set(x)),distinct_target_values=len(set(y)),
        predeployment_equivalence_groups=len({r.get('independence_group',r['case_id'])for r in valid}),
        seed_count=len({r['provenance']['seed']for r in valid if r.get('provenance',{}).get('seed')is not None}),
        tie_policy='round_dimensionless_descriptive_ordering_values_to_12_decimals',risk_field=risk_key,target_field=target_key)


def summarize(rows,*,risk_key='raw_risk',target_key='observed_deterioration'):
    per_intent=defaultdict(list);per_cohort=defaultdict(list);strata=defaultdict(list)
    for row in rows:
        per_intent[row['intent_id']].append(row);per_cohort[row['intent_id']+'|'+row['cohort_id']].append(row)
        strata['|'.join((row['intent_id'],row.get('scene','unknown'),str(row.get('provenance',{}).get('seed'))))].append(row)
    metric=lambda value:ordering(value,risk_key=risk_key,target_key=target_key)
    intent_metrics={key:metric(value)for key,value in sorted(per_intent.items())}
    stratum_metrics={key:metric(value)for key,value in sorted(strata.items())}
    macro=lambda key:sum(r[key]for r in stratum_metrics.values()if r[key]is not None)/sum(r[key]is not None for r in stratum_metrics.values())if any(r[key]is not None for r in stratum_metrics.values())else None
    return dict(rows=len(rows),per_intent=intent_metrics,per_intent_cohort={key:metric(value)for key,value in sorted(per_cohort.items())},
        per_intent_scene_seed=stratum_metrics,unique_physical_runs=len({r['provenance']['job_id']for r in rows if r.get('provenance',{}).get('job_id')}),
        seed_blocks=len({(r.get('scene'),r.get('provenance',{}).get('seed'))for r in rows}),
        unique_seed_count=len({r['provenance']['seed']for r in rows if r.get('provenance',{}).get('seed')is not None}),
        valid_spearman_strata=[k for k,v in stratum_metrics.items()if v['spearman']is not None],
        undefined_spearman_strata={k:v['spearman_unavailable_reason']for k,v in stratum_metrics.items()if v['spearman']is None},
        calibration_seed_overlap_rows=sum(r.get('provenance',{}).get('calibration_seed_overlap',False)for r in rows),
        partition_counts=dict(Counter(r.get('provenance',{}).get('evaluation_partition','unspecified')for r in rows)),
        macro_spearman=macro('spearman'),macro_pair_concordance=macro('pair_concordance'),
        macro_weighting='equal_valid_intent_scene_seed_stratum_weight; no IID pooling of derived intents',
        limitations=['Repeated intents on one retained run are not independent simulations.',
            'Repeated same-seed/predeployment groups are reported; no independent-seed confidence interval is claimed.',
            'Constant-score strata have undefined Spearman and remain visible; use matched strata and pairwise ties for comparisons.'])


def join_evaluation(scores_path,sources_path,output):
    """Read AFTER traces only after the score artifact is durably available."""
    scores_path=Path(scores_path)
    if not scores_path.is_file():raise ValueError('persist_scores_before_loading_labels')
    scored=read(scores_path)
    if scored.get('stage')!='scores_before_label_join'or scored.get('after_values_used')is not False:
        raise ValueError('untrusted_scoring_artifact')
    input_path=Path(scored['input_path'])
    if not input_path.is_file()or file_sha(input_path)!=scored['input_sha256']:raise ValueError('frozen_scoring_inputs_changed')
    if scored.get('label_sources_sha256')and file_sha(sources_path)!=scored['label_sources_sha256']:raise ValueError('frozen_label_sources_changed')
    sources=read(sources_path)['sources'];rows=[];windows={};readiness={}
    for item in scored['scores']:
        source=sources[item['case_id']];group=(item.get('result',{}).get('groups')or[{}])[0]
        if source.get('intent_resolution',{}).get('status')=='unavailable':
            rows.append(dict(case_id=item['case_id'],intent_id=source['intent_id'],cohort_id=source['cohort_id'],
                independence_group=source['independence_group'],provenance=source['provenance'],protocol=deepcopy(source['protocol']),
                candidate=source.get('candidate',{}),scene=source.get('scene','unknown'),risk=None,raw_risk=None,risk_status='unavailable',
                reference_effect=None,reference_effect_score=None,risk_reason='intent_reference_unavailable',label_status='intent_reference_unavailable',
                observed_metric=None,observed_deterioration=None,all_satisfied=None,evaluation=None,before_after=[],intent_resolution=deepcopy(source['intent_resolution'])))
            continue
        ready_key=(source.get('run_path'),source.get('seal_path'),source.get('expected_origin_id'))
        if ready_key not in readiness:readiness[ready_key]=_actual_arm_ready(source)
        ready,reason=readiness[ready_key]
        protocol=deepcopy(source['protocol'])
        if source.get('requires_actual_origin_guard'):
            protocol['actual_arm_origin_guard_pending']=not ready
            protocol['primary_validation_eligible']=protocol.get('primary_validation_eligible',False)and ready
        base=dict(case_id=item['case_id'],intent_id=source['intent_id'],cohort_id=source['cohort_id'],
            independence_group=source['independence_group'],provenance=source['provenance'],protocol=protocol,
            candidate=source.get('candidate',{}),scene=source.get('scene','unknown'),
            risk=group.get('score'),raw_risk=group.get('raw_obstruction_strength'),risk_status=group.get('score_status'),
            reference_effect=group.get('reference_effect'),reference_effect_score=group.get('reference_effect_score'),risk_reason=group.get('reason'))
        if not ready:
            rows.append(dict(**base,label_status=reason,observed_metric=None,observed_deterioration=None,all_satisfied=None,evaluation=None,before_after=[]));continue
        key=(source['run_path'],digest(source['before_window']),digest(source['after_window']))
        if key not in windows:windows[key]=_observed_windows({**source,'ranges':[{'kpi':k}for k in KPIS]})
        before,after=windows[key];ranges=source['ranges']
        comparison=dict(rows=[dict(**deepcopy(r),before=assess(r,before),after=assess(r,after))for r in ranges])
        # Evaluation's baseline is the actual common BEFORE window. It is never
        # passed back into score normalization or used to tune graph weights.
        result=evaluate(comparison,ranges=ranges,reference_values={r['kpi']:before['states']['kpi:'+r['kpi']].get('value')for r in ranges})
        rows.append(dict(**base,label_status='complete'if result['after']['complete']else'incomplete_requested_kpis',observed_metric=result['after']['score'],
            observed_deterioration=max(0.,-result['after']['score'])/100 if result['after']['score']is not None else None,
            all_satisfied=result['after']['all_satisfied'],evaluation=result,before_after=comparison['rows']))
    primary=[r for r in rows if r['protocol']['primary_validation_eligible']and r.get('candidate',{}).get('eligible')]
    diagnostics=rows
    legacy=[r for r in rows if r['protocol'].get('name')=='legacy_or_unverified']
    result=dict(schema=SCHEMA,stage='labels_joined_after_scoring',method=scored.get('graph_variant',{}).get('id',scored['method']),rows=rows,
        primary=summarize(primary),primary_signed_effect=summarize(primary,risk_key='reference_effect_score',target_key='observed_metric'),
        all_arms_diagnostics=summarize(diagnostics),candidate_pool_counts=dict(Counter(r.get('candidate',{}).get('pool','unknown')for r in rows)),
        label_status_counts=dict(Counter(r['label_status']for r in rows)),
        legacy_exploratory=summarize(legacy),scores_sha256=file_sha(scores_path),
        scoring_input_sha256=scored['input_sha256'],score_label_separation=True,new_simulations=0,
        observed_results_used_for_scoring=False,raw_archives_modified=False)
    if scored.get('graph_variant'):result.update(engine_method=scored.get('engine_method',scored['method']),graph_variant=deepcopy(scored['graph_variant']))
    write(output,result);return result


def compare_evaluation_reports(report_paths,output):
    """Compare methods on the same eligible physical cases and defined strata.

    This explicitly separate AFTER analysis accepts only already joined
    reports; it does not feed results back into quantification or calibration.
    """
    reports=[read(p)for p in report_paths]
    if any(r.get('stage')!='labels_joined_after_scoring'or r.get('observed_results_used_for_scoring')is not False for r in reports):
        raise ValueError('separately_joined_evaluation_reports_required')
    methods=[r['method']for r in reports]
    if len(methods)<2 or len(set(methods))!=len(methods):raise ValueError('distinct_method_reports_required')
    indexed={r['method']:{row['case_id']:row for row in r['rows']}for r in reports}
    common=set.intersection(*(set(rows)for rows in indexed.values()))
    for case_id in common:
        reference=indexed[methods[0]][case_id]
        for method in methods[1:]:
            row=indexed[method][case_id]
            if (row.get('candidate')!=reference.get('candidate')or row['intent_id']!=reference['intent_id']
                or not comparable(row.get('observed_metric'),reference.get('observed_metric'))
                or not comparable(row.get('observed_deterioration'),reference.get('observed_deterioration'))):
                raise ValueError('methods_do_not_share_the_same_frozen_candidate_and_labels:'+case_id)
    primary={case_id for case_id in common if all(indexed[m][case_id]['protocol'].get('primary_validation_eligible')
        and indexed[m][case_id].get('candidate',{}).get('eligible')for m in methods)}
    matched={case_id for case_id in primary if all(type(indexed[m][case_id].get('raw_risk'))in(int,float)
        and math.isfinite(indexed[m][case_id]['raw_risk'])and type(indexed[m][case_id].get('observed_deterioration'))in(int,float)
        and math.isfinite(indexed[m][case_id]['observed_deterioration'])for m in methods)}
    summaries={m:summarize([indexed[m][case_id]for case_id in sorted(matched)])for m in methods}
    rho_strata=set.intersection(*(set(s['valid_spearman_strata'])for s in summaries.values()))
    result=dict(schema='oran.quantification-method-comparison.v1',stage='descriptive_after_analysis_only',
        report_files_sha256={str(Path(p)):file_sha(p)for p in report_paths},methods=methods,
        common_cases=len(common),common_primary_cases=len(primary),common_numeric_primary_cases=len(matched),
        unmatched_or_unavailable_primary_case_ids=sorted(primary-matched),matched_primary=summaries,
        same_defined_spearman_strata=sorted(rho_strata),
        macro_spearman_same_defined_strata={m:sum(summaries[m]['per_intent_scene_seed'][k]['spearman']for k in rho_strata)/len(rho_strata)if rho_strata else None for m in methods},
        method_native_primary={r['method']:r['primary']for r in reports},
        label_status_counts={r['method']:r['label_status_counts']for r in reports},
        scoring_or_selection_modified_from_results=False,
        limitations=['Matched comparisons exclude missing method scores transparently; availability is separately reported.',
            'Derived intents and arms sharing a seed are not independent replications.',
            'Constant scores have undefined Spearman; pair concordance retains score ties.',
            'These are descriptive associations, without an independent-seed confidence interval.'])
    write(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    inv=sub.add_parser('inventory');inv.add_argument('--jobs',type=Path,required=True);inv.add_argument('--output',type=Path,required=True)
    for name in('extract','run'):
        p=sub.add_parser(name);p.add_argument('--jobs',type=Path,required=True);p.add_argument('--influences',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True);p.add_argument('--intents',type=Path);p.add_argument('--batch',action='append')
        p.add_argument('--include-legacy',action='store_true');p.add_argument('--confidence',type=float,default=.90)
        if name=='run':
            p.add_argument('--method',default='goal_obstruction_v2');p.add_argument('--module',type=Path);p.add_argument('--engine',default='SmoGoalObstruction')
    score=sub.add_parser('score');score.add_argument('--inputs',type=Path,required=True);score.add_argument('--output',type=Path,required=True)
    score.add_argument('--method',default='goal_obstruction_v2');score.add_argument('--module',type=Path);score.add_argument('--engine',default='SmoGoalObstruction');score.add_argument('--full-projections',action='store_true')
    join=sub.add_parser('evaluate');join.add_argument('--scores',type=Path,required=True);join.add_argument('--sources',type=Path,required=True);join.add_argument('--output',type=Path,required=True)
    workers=sub.add_parser('extract-workers');workers.add_argument('--artifact',type=Path,required=True);workers.add_argument('--calibrations',type=Path,required=True)
    workers.add_argument('--intents',type=Path,required=True);workers.add_argument('--output',type=Path,required=True)
    prospective=sub.add_parser('extract-prospective');prospective.add_argument('--artifact',type=Path,required=True);prospective.add_argument('--calibrations',type=Path,required=True)
    prospective.add_argument('--intents',type=Path,required=True);prospective.add_argument('--output',type=Path,required=True);prospective.add_argument('--origin',action='append')
    seal=sub.add_parser('seal-prospective');seal.add_argument('--artifact',type=Path,required=True);seal.add_argument('--calibrations',type=Path,required=True)
    seal.add_argument('--intents',type=Path,required=True);seal.add_argument('--output',type=Path,required=True);seal.add_argument('--origin',required=True)
    seal.add_argument('--preregistration',type=Path,required=True);seal.add_argument('--numeric-policy',type=Path,required=True)
    variant=sub.add_parser('seal-graph-variant');variant.add_argument('--base-seal',type=Path,required=True)
    variant.add_argument('--graphs',type=Path,required=True,help='JSON scene-to-complete-graph-path mapping')
    variant.add_argument('--profile-plan',type=Path,required=True);variant.add_argument('--provenance',type=Path,action='append',default=[])
    variant.add_argument('--output',type=Path,required=True)
    comparison=sub.add_parser('compare-reports');comparison.add_argument('--report',type=Path,action='append',required=True)
    comparison.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='compare-reports':
        value=compare_evaluation_reports(args.report,args.output);print(json.dumps(value,ensure_ascii=False));return
    if args.command=='seal-graph-variant':
        value=seal_graph_variant(args.base_seal,read(args.graphs),args.output,profile_plan_path=args.profile_plan,provenance_paths=args.provenance)
        print(json.dumps(value,ensure_ascii=False));return
    if args.command=='inventory':value=inventory_batches(args.jobs);write(args.output,value);print(json.dumps({k:v for k,v in value.items()if k!='batches'}));return
    if args.command=='extract-workers':
        specs=read(args.intents);specs=specs['intents']if isinstance(specs,dict)else specs
        value=extract_worker_inputs(args.artifact,read(args.calibrations),args.output,intents=specs)
        print(json.dumps(dict(cases=len(value['cases']),excluded=value['excluded'])));return
    if args.command=='extract-prospective':
        specs=read(args.intents);specs=specs['intents']if isinstance(specs,dict)else specs
        value=extract_prospective_inputs(args.artifact,read(args.calibrations),args.output,intents=specs,origin_keys=args.origin)
        print(json.dumps(dict(cases=len(value['cases']),blocks=value['prospective_blocks'],excluded=value['excluded'])));return
    if args.command=='seal-prospective':
        value=seal_prospective_block(args.artifact,read(args.calibrations),args.intents,args.output,origin_key=args.origin,
            preregistration_path=args.preregistration,numeric_policy_path=args.numeric_policy)
        print(json.dumps(value,ensure_ascii=False));return
    if args.command in('extract','run'):
        specs=read(args.intents)if args.intents else None
        if isinstance(specs,dict):specs=specs['intents']
        value=extract_inputs(args.jobs,args.influences,args.output,intents=specs,batch_ids=args.batch,
            include_legacy=args.include_legacy,confidence_level=args.confidence)
        if args.command=='extract':print(json.dumps(dict(cases=len(value['cases']),excluded=len(value['excluded']))));return
        score_cases(args.output/'scoring-inputs.json',args.output/'scores-before-label-join.json',method=args.method,module=args.module,engine=args.engine)
        value=join_evaluation(args.output/'scores-before-label-join.json',args.output/'label-sources.json',args.output/'results.json')
    elif args.command=='score':value=score_cases(args.inputs,args.output,method=args.method,module=args.module,engine=args.engine,full_results=args.full_projections)
    else:value=join_evaluation(args.scores,args.sources,args.output)
    print(json.dumps({k:v for k,v in value.items()if k not in('rows','scores')},ensure_ascii=False))


if __name__=='__main__':main()
