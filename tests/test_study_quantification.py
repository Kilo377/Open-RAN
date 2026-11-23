"""Offline protocol/score-label separation; analytic fixtures, no simulation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import study_quantification as study


def test_metadata_inventory_does_not_promote_legacy_windows(tmp_path):
    comparison=dict(baseline={'slot':520},application_slot=561,baseline_origin='fresh_run_all_xapps_off',
        baseline_window={'window_start_slot':20,'slot':520},after_window={'window_start_slot':560,'slot':1060})
    member=dict(job_id='j',group_id='g',app_ids=['a'],phase='complete',comparison=comparison)
    batch=dict(id='b',seed=7,observe_slots=500,members=[member])
    study.write(tmp_path/'batches/b/batch.json',batch)
    inventory=study.inventory_batches(tmp_path)
    assert inventory['primary_metadata_candidates']==0
    assert inventory['phase_counts']=={'complete':1}
    assert 'fixed_after_window'in inventory['batches'][0]['members'][0]['protocol']['reasons']


def test_predeployment_extraction_never_uses_after_values(tmp_path,monkeypatch):
    config=dict(manifest=dict(slot_duration_ms=1,ues=[dict(ue_id='u')],cells=[dict(cell_id='c')]),configs={})
    study.write(tmp_path/'config_snapshot.json',config)
    previews=[dict(slot=slot,apps=[dict(app_id='a',actions=[dict(parameter='ue_offload_fraction',ue_id='u',cell_id='c',current=.5,proposed=1)])])for slot in(2,4,6)]
    rows=[dict(slot_id=s,ues=[dict(ue_id='u',cell_id='c',packet_delivered_bytes=s*100)],
        ue_services={'u':dict(ue_energy_total_j=s*.001)},ap_services={})for s in range(1,5)]
    # Only its slot may be inspected to stop; no future metric is available.
    class Future(dict):
        def get(self,*a):raise AssertionError('AFTER content read during predeployment extraction')
        def __getitem__(self,key):
            if key=='slot_id':return 5
            raise AssertionError('AFTER content read during predeployment extraction')
    rows.append(Future())
    monkeypatch.setattr(study,'records',lambda run,name:iter(previews if name=='xapp_decision_previews'else rows))
    monkeypatch.setattr(study.PreviewStore,'read',lambda self,slot:deepcopy(next(p for p in previews if p['slot']==slot)))
    class Builder:
        def __init__(self,*a,**kw):pass
        def frame(self,slot,*a,**kw):return dict(slot=slot,time_ms=slot,states={})
    monkeypatch.setattr(study,'FrameBuilder',Builder)
    before=deepcopy(previews)
    pre=study._predeployment_run(tmp_path,4,2,['network_goodput_mbps'])
    assert pre['decision_slot']==4 and pre['exact_before_window']
    assert pre['frame']['states']['kpi:network_goodput_mbps']['value']==pytest.approx(.8)
    assert pre['frame']['states']['rcp:ue_offload_fraction']['by_entity']['u']==.5
    assert [x['frame']['slot']for x in pre['history']]==[2,4]
    assert previews==before


def test_study_intents_anchor_to_actual_before_without_mutating_the_ledger_inputs():
    graph=dict(nodes=[dict(kind='kpi',key='network_goodput_mbps',id='kpi:network_goodput_mbps',
        label='Throughput',unit='Mbit/s',scope='All UEs')])
    spec=dict(id='new',ranges=[dict(kpi='network_goodput_mbps',lower={'current_multiplier':.95},upper=None,
        lower_inclusive=True,upper_inclusive=True)])
    frame=dict(slot=520,time_ms=260,states={'kpi:network_goodput_mbps':{'value':20}})
    original=deepcopy((spec,frame))
    row=study.resolve_study_intent(spec,graph,frame)[0]
    assert row['lower']==19 and row['anchors'][0]['current']==20
    assert row['anchors'][0]['declared_experimental_intent']
    assert (spec,frame)==original


def test_primary_order_population_excludes_non_candidates_and_off_controls():
    rows=[]
    for ident,eligible,scene,seed in [('a',True,'native',1),('b',True,'native',1),('c',False,'native',1)]:
        rows.append(dict(case_id=ident,intent_id='i',cohort_id='c',scene=scene,provenance={'seed':seed,'job_id':ident},
            independence_group=ident,risk=len(rows),raw_risk=len(rows),observed_deterioration=len(rows),candidate={'eligible':eligible}))
    summary=study.summarize([r for r in rows if r['candidate']['eligible']])
    assert summary['macro_spearman']==pytest.approx(1)
    assert summary['rows']==2 and summary['unique_physical_runs']==2 and summary['seed_blocks']==1


def test_macro_correlation_is_stratified_not_pooled_across_intents():
    rows=[]
    for intent,seed,pairs in [('i',1,[(1,1),(2,2)]),('j',2,[(10,20),(20,10)])]:
        for i,(risk,actual)in enumerate(pairs):rows.append(dict(case_id=f'{intent}{i}',intent_id=intent,cohort_id=intent,
            scene='native',provenance={'seed':seed,'job_id':str(i)},independence_group=str(i),risk=risk,raw_risk=risk,observed_deterioration=actual))
    summary=study.summarize(rows)
    assert summary['macro_spearman']==pytest.approx(0)
    assert summary['macro_pair_concordance']==pytest.approx(.5)
    assert len(summary['per_intent_scene_seed'])==2


def test_scores_must_be_persisted_before_labels_are_accessed(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'_observed_windows',lambda *a:pytest.fail('Labels read before a score artifact exists'))
    with pytest.raises(ValueError,match='persist_scores'):study.join_evaluation(tmp_path/'missing.json',tmp_path/'sources.json',tmp_path/'results.json')


def test_context_deduplication_and_generic_engine_do_not_include_labels(tmp_path):
    from test_smo_goal_obstruction import fixture
    from smo.influence_units import align_graph
    graph,frame,catalog,preview,_=fixture();graph=align_graph(graph)
    case=dict(case_id='case',intent_id='i',group_id='ab',app_ids=['a','b'],frame=frame,preview=preview,
        ranges=catalog['ranges'],options={'method':'goal_obstruction_v2'},protocol={},provenance={},cohort_id='c',independence_group='j')
    second=deepcopy(case);second['case_id']='case2';second['intent_id']='j'
    data=study.compact_contexts(dict(schema=study.SCHEMA,graph=graph,cases=[case,second],after_values_included=False))
    assert len(data['contexts'])==1 and 'frame'not in data['cases'][0]
    assert study.materialize_case(data,data['cases'][0])['frame']==frame
    source=tmp_path/'inputs.json';study.write(source,data)
    result=study.score_cases(source,tmp_path/'scores.json')
    assert result['after_values_used']is False and len(result['scores'])==2
    first=result['scores'][0]['result']['groups'][0]
    assert first['score']==pytest.approx(100*.1/1.1)
    assert 'actual_outcomes'not in first
    assert result['input_sha256']==study.file_sha(source)


def test_graph_variant_id_dispatches_the_identical_registered_net_formula(tmp_path):
    from test_smo_goal_obstruction import fixture
    from smo.influence_units import align_graph
    graph,frame,catalog,preview,_=fixture()
    case=dict(case_id='c',intent_id='i',group_id='ab',app_ids=['a','b'],frame=frame,preview=preview,ranges=catalog['ranges'],
        options={},protocol={},provenance={},cohort_id='c',independence_group='j')
    source=tmp_path/'inputs.json';study.write(source,study.compact_contexts(dict(graph=align_graph(graph),cases=[case],after_values_included=False)))
    common=dict(module=study.ROOT/'smo/web/intent_obstruction.js',engine='SmoIntentObstruction')
    direct=study.score_cases(source,tmp_path/'net.json',method='net_intent_obstruction_v1',**common)
    variant=study.score_cases(source,tmp_path/'variant.json',method='net_profile_gain_obstruction_v1',dispatch_method='net_intent_obstruction_v1',**common)
    assert variant['method']=='net_profile_gain_obstruction_v1'and variant['engine_method']=='net_intent_obstruction_v1'
    assert variant['scores']==direct['scores']


def test_label_join_reassesses_only_requested_kpis_and_exports_nonnegative_d(tmp_path,monkeypatch):
    source=dict(run_path='unread',ranges=[dict(kpi='network_goodput_mbps',unit='Mbit/s',domain=[0,None],lower=10,upper=None,
        lower_inclusive=True,upper_inclusive=True)],before_window={},after_window={},intent_id='i',cohort_id='c',
        independence_group='j',provenance={'job_id':'j','seed':1},scene='native',protocol={'primary_validation_eligible':True},candidate={'eligible':True,'pool':'primary'})
    def window(value):return dict(states={'kpi:network_goodput_mbps':dict(value=value),'kpi:unrequested':dict(value=-1e9)})
    monkeypatch.setattr(study,'_observed_windows',lambda _: (window(10),window(8)))
    study.write(tmp_path/'sources.json',dict(sources={'case':source}))
    study.write(tmp_path/'inputs.json',dict(after_values_included=False))
    study.write(tmp_path/'scores.json',dict(stage='scores_before_label_join',after_values_used=False,input_path=str(tmp_path/'inputs.json'),input_sha256=study.file_sha(tmp_path/'inputs.json'),method='test',scores=[
        dict(case_id='case',result={'groups':[dict(score=20,raw_obstruction_strength=.25,reference_effect_score=-20)]})]))
    result=study.join_evaluation(tmp_path/'scores.json',tmp_path/'sources.json',tmp_path/'results.json')
    assert result['rows'][0]['observed_metric']==-20
    assert result['rows'][0]['observed_deterioration']==pytest.approx(.2)
    assert [m['kpi']for m in result['rows'][0]['evaluation']['metrics']]==['network_goodput_mbps']
    assert result['primary']['rows']==1


def test_frozen_inputs_and_scores_cannot_be_silently_rewritten(tmp_path):
    study.write_immutable(tmp_path/'input.json',{'value':1})
    study.write_immutable(tmp_path/'input.json',{'value':1})
    with pytest.raises(ValueError,match='frozen_study_artifact'):study.write_immutable(tmp_path/'input.json',{'value':2})


def test_pending_prospective_arm_does_not_open_after_telemetry(tmp_path,monkeypatch):
    source=dict(run_path='future-run',seal_path=str(tmp_path/'not-finished.json'),expected_origin_id='origin',requires_actual_origin_guard=True,
        intent_id='i',cohort_id='c',independence_group='j',scene='native',provenance={'seed':1,'job_id':'future'},
        candidate={'eligible':True,'pool':'primary'},protocol={'name':'prospective_certified_OFF_prefix','primary_validation_eligible':True})
    study.write(tmp_path/'sources.json',{'sources':{'case':source}});study.write(tmp_path/'inputs.json',{'after_values_included':False})
    study.write(tmp_path/'scores.json',dict(stage='scores_before_label_join',after_values_used=False,input_path=str(tmp_path/'inputs.json'),
        input_sha256=study.file_sha(tmp_path/'inputs.json'),method='test',scores=[dict(case_id='case',result={'groups':[{'score':20,'raw_obstruction_strength':.25}]})]))
    monkeypatch.setattr(study,'_observed_windows',lambda _:pytest.fail('A prospective arm has not completed; no AFTER access'))
    result=study.join_evaluation(tmp_path/'scores.json',tmp_path/'sources.json',tmp_path/'results.json')
    assert result['rows'][0]['label_status']=='awaiting_completed_sealed_run'
    assert result['rows'][0]['risk']==20 and result['rows'][0]['observed_metric']is None
    assert result['primary']['rows']==0


def test_only_descriptive_ordering_quantizes_numerical_noise():
    rows=[dict(case_id=str(i),raw_risk=x,observed_deterioration=y)for i,(x,y)in enumerate([(0.,0.),(1e-17,.1)])]
    original=deepcopy(rows);result=study.ordering(rows)
    assert result['risk_ties']==1 and result['pair_concordance']==.5
    assert result['spearman_unavailable_reason']=='constant_risk'
    assert rows==original


def test_prospective_all_arms_are_scored_from_off_prefix_before_future_runs_exist(tmp_path,monkeypatch):
    frame=dict(slot=4,time_ms=4,states={'kpi:network_goodput_mbps':{'value':10}})
    preview=dict(slot=4,apps=[dict(app_id='a',mode='cold_activation',actions=[])])
    config=dict(manifest={'rng_seed':1},configs={})
    prefix=dict(frame=frame,preview=preview,config=config,history=[{'frame':frame,'preview':preview}],
        exact_baseline_decision=True,exact_before_window=True,prefix_controllers_off=True,prefix_exogenous_sha256='prefix')
    origin=dict(seed=1,baseline_slot=4,observe_slots=2,dynamics_sha256='dynamics',prefix_exogenous_sha256='prefix',
        configuration=study.origin_configuration(config),model_sha256={'a':'model'},before_frame=frame)
    origin['origin_id']=study.digest(origin)
    off=dict(id='off',origin_key='native_s1',scene='native',seed=1,arm='OFF',app_ids=[])
    future=dict(id='future',origin_key='native_s1',scene='native',seed=1,arm='A',app_ids=['a'])
    physical=dict(observe_slots=2,baseline_slot=4,arms=[{'id':'OFF'},{'id':'A'}])
    worker=tmp_path/'workers/agx';study.write(worker/'plan.json',dict(worker_id='agx',source_archive_sha256='source',physical_plan_sha256='plan',physical=physical,cases=[off,future]))
    study.write(worker/'checkpoints/off.json',dict(completed=True,case=off,physical_plan_sha256='plan',origin_id=origin['origin_id']))
    study.write(worker/'origins/native_s1.json',origin)
    graph=dict(nodes=[dict(kind='kpi',key='network_goodput_mbps',id='kpi:network_goodput_mbps',label='Throughput',unit='Mbit/s',scope='All UEs')],edges=[])
    monkeypatch.setattr(study,'graph_for_study',lambda *a,**kw:graph)
    monkeypatch.setattr(study,'archived_dynamics_fingerprint',lambda *a:'dynamics')
    monkeypatch.setattr(study,'_predeployment_run',lambda *a:deepcopy(prefix))
    monkeypatch.setattr(study,'_verified_models',lambda *a:{'a':'model'})
    monkeypatch.setattr(study,'candidate_flags',lambda *a:dict(pool='primary'if a[4]else'off_diagnostic',eligible=bool(a[4])))
    spec=dict(id='new',ranges=[dict(kpi='network_goodput_mbps',lower='current',upper=None,lower_inclusive=False,upper_inclusive=True)])
    calibration=tmp_path/'influences.json';study.write(calibration,dict(estimates=[]))
    data=study.extract_prospective_inputs(tmp_path,{'native':calibration},tmp_path/'predictions',intents=[spec])
    assert data['excluded']==[] and len(data['cases'])==2 and len(data['contexts'])==1
    assert data['prospective_blocks'][0]['after_values_used']is False
    assert not (worker/'jobs/future').exists()
    context=next(iter(data['contexts'].values()))
    assert context['options']['sourceIdentity']=='study-origin:'+origin['origin_id']
    assert all(x['source_identity']==context['options']['sourceIdentity']for x in context['options']['history'])


def test_relative_intent_without_before_reference_is_preserved_without_scoring_or_labels(tmp_path,monkeypatch):
    graph=dict(nodes=[dict(kind='kpi',key='mec_task_latency_ms',id='kpi:mec_task_latency_ms',
        label='MEC latency',unit='ms',scope='MEC workload')],edges=[])
    frame=dict(slot=2040,time_ms=1020,states={'kpi:mec_task_latency_ms':{'value':None}})
    spec=dict(id='relative-mec',ranges=[dict(kpi='mec_task_latency_ms',lower=0,upper={'current_multiplier':.9},
        lower_inclusive=True,upper_inclusive=True)])
    pre=dict(group_id='g',app_ids=['a'],frame=frame,preview={'slot':2040,'apps':[]},history=[],
        protocol={'name':'prospective_certified_OFF_prefix','primary_validation_eligible':True},timing={'observe_slots':2000},arbitration={},
        provenance=dict(scoring_origin_id='o',job_id='j',seed=1,configuration_sha256='config',prefix_exogenous_sha256='prefix'),
        label_source={'run_path':'does-not-exist'})
    monkeypatch.setattr(study,'candidate_flags',lambda *a,**kw:pytest.fail('Unresolved intent must not enter candidate selection'))
    case,source=study.make_scoring_case(pre,spec,graph,case_id='case',scene='steady')
    assert case['ranges']==[] and source['ranges']==[]
    assert case['candidate']==dict(pool='unresolved_intent',eligible=False,reason='intent_reference_unavailable')
    assert case['intent_resolution']['requested_ranges']==spec['ranges']
    assert case['intent_resolution']['missing_kpis']==['mec_task_latency_ms']
    inputs=tmp_path/'scoring-inputs.json';sources=tmp_path/'label-sources.json'
    study.write(inputs,study.compact_contexts(dict(graph=graph,cases=[case],after_values_included=False)))
    study.write(sources,dict(sources={'case':source}))
    engine=tmp_path/'throwing-engine.js';engine.write_text('globalThis.Never={review(){throw Error("Unresolved intent scored")}};')
    scores=tmp_path/'scores.json';scored=study.score_cases(inputs,scores,module=engine,engine='Never')
    assert scored['scores'][0]['result']['groups'][0]['score']is None
    monkeypatch.setattr(study,'_actual_arm_ready',lambda *a:pytest.fail('Unresolved interval must not inspect label readiness'))
    monkeypatch.setattr(study,'_observed_windows',lambda *a:pytest.fail('Missing BEFORE must not resolve from AFTER'))
    joined=study.join_evaluation(scores,sources,tmp_path/'results.json')
    assert joined['rows'][0]['observed_metric']is None
    assert joined['rows'][0]['label_status']=='intent_reference_unavailable'


def test_unready_explicit_prospective_origin_does_not_freeze_empty_inputs(tmp_path):
    with pytest.raises(ValueError,match='requested_prospective_blocks_not_ready'):
        study.extract_prospective_inputs(tmp_path,{},tmp_path/'predictions',intents=[],origin_keys=['not-ready'])
    assert not (tmp_path/'predictions/scoring-inputs.json').exists()


def variant_fixture(tmp_path):
    graph=dict(nodes=[dict(id='rcp:p',kind='parameter'),dict(id='kpi:k',kind='kpi',unit='W')],
        edges=[dict(id='pk',source='rcp:p',target='kpi:k',kind='influences',sensitivities=[{'estimate':1}])])
    frame=dict(slot=2040,states={'kpi:k':{'value':10}});preview={'slot':2040,'apps':[]}
    context=dict(frame=frame,preview=preview,source_identity='origin',graph_context_sha256=study.digest(graph))
    case=dict(case_id='case',intent_id='i',group_id='g',app_ids=['a'],graph_id='native',frame=frame,preview=preview,
        candidate={'pool':'primary','eligible':True},ranges=[dict(kpi='k',lower=0,upper=9)],provenance={'seed':2},
        options=dict(arbitration={'mode':'uniform_random'},sourceIdentity='origin',graphContextSha256=study.digest(graph),
            history=[context],normalization={'rows':[{'scale':10}]}))
    data=study.compact_contexts(dict(graphs={'native':graph},cases=[case],after_values_included=False))
    inp=tmp_path/'base/scoring-inputs.json';study.write(inp,data)
    source=inp.with_name('label-sources.json');source.write_text('{"sources":{"case":{"run_path":"'+str(tmp_path/'worker/jobs/a/data/run')+'"}}}\n')
    provenance=tmp_path/'training-calibration.json';study.write(provenance,{'training_seed':25})
    return inp,graph,provenance


def test_graph_variant_preserves_every_before_value_range_candidate_and_label_byte(tmp_path):
    inp,graph,provenance=variant_fixture(tmp_path);original=study.read(inp);before=study.materialize_case(original,original['cases'][0])
    overlay=deepcopy(graph);overlay['edges'][0]['sensitivities'].append({'estimate':-2,'kind':'finite_contrast'})
    data=study.derive_graph_variant_inputs(inp,{'native':overlay},tmp_path/'variant',variant_id='net_profile_gain_obstruction_v1',provenance_paths=[provenance])
    actual=study.materialize_case(data,data['cases'][0])
    assert all(actual[k]==before[k]for k in('frame','preview','ranges','candidate','provenance'))
    assert actual['options']['normalization']==before['options']['normalization']
    assert actual['options']['sourceIdentity']==before['options']['sourceIdentity']
    assert actual['options']['graphContextSha256']==study.digest(overlay)
    assert actual['options']['history'][0]['graph_context_sha256']==study.digest(overlay)
    assert actual['options']['history'][0]['frame']==before['options']['history'][0]['frame']
    assert (tmp_path/'variant/label-sources.json').read_bytes()==inp.with_name('label-sources.json').read_bytes()
    assert study.read(inp)==original


@pytest.mark.parametrize('change',['edge','node_contract'])
def test_graph_gain_variant_refuses_new_topology_or_changed_xapp_contracts(tmp_path,change):
    inp,graph,provenance=variant_fixture(tmp_path);overlay=deepcopy(graph)
    if change=='edge':overlay['edges'].append(dict(id='new',source='rcp:p',target='kpi:k',kind='influences'))
    else:overlay['nodes'][1]['unit']='ms'
    with pytest.raises(ValueError,match='changed_admitted_edges|changed_node_contracts'):
        study.derive_graph_variant_inputs(inp,{'native':overlay},tmp_path/'variant',variant_id='v',provenance_paths=[provenance])
    assert not (tmp_path/'variant/scoring-inputs.json').exists()


def test_variant_seal_refuses_started_nonempty_arm_before_derivation(tmp_path):
    inp,graph,provenance=variant_fixture(tmp_path)
    request=tmp_path/'worker/jobs/a/data/request.json';study.write(request,{})
    seal=tmp_path/'base/prediction-seal.json'
    study.write(seal,dict(input_path=str(inp),input_sha256=study.file_sha(inp),label_sources_sha256=study.file_sha(inp.with_name('label-sources.json')),
        scores={},after_values_used=False,nonempty_arm_started_before_scoring=False))
    plan=dict(extra_variant={'id':'net_profile_gain_obstruction_v1'},validation_seeds_excluded=[2]);plan['plan_sha256']=study.digest(plan)
    plan_path=tmp_path/'plan.json';study.write(plan_path,plan)
    with pytest.raises(ValueError,match='already_started'):
        study.seal_graph_variant(seal,{},tmp_path/'variant',profile_plan_path=plan_path)
    assert not (tmp_path/'variant').exists()


def test_method_comparison_keeps_constant_score_strata_and_compares_identical_labels(tmp_path):
    paths=[]
    for method,risks in(('a',[0,0]),('b',[0,.1])):
        rows=[dict(case_id=str(i),intent_id='i',cohort_id='c',independence_group=str(i),scene='native',provenance={'seed':2,'job_id':str(i)},
            protocol={'primary_validation_eligible':True},candidate={'eligible':True},raw_risk=risk,
            observed_deterioration=i*.1,observed_metric=-10*i)for i,risk in enumerate(risks)]
        report=dict(stage='labels_joined_after_scoring',method=method,rows=rows,observed_results_used_for_scoring=False,
            primary=study.summarize(rows),label_status_counts={'complete':2})
        path=tmp_path/(method+'.json');study.write(path,report);paths.append(path)
    result=study.compare_evaluation_reports(paths,tmp_path/'comparison.json')
    assert result['common_numeric_primary_cases']==2 and result['same_defined_spearman_strata']==[]
    assert result['macro_spearman_same_defined_strata']=={'a':None,'b':None}
    assert result['matched_primary']['a']['macro_pair_concordance']==.5
    assert result['matched_primary']['a']['undefined_spearman_strata']=={'i|native|2':'constant_risk'}
    changed=study.read(paths[1]);changed['rows'][0]['observed_metric']=1;study.write(paths[1],changed)
    with pytest.raises(ValueError,match='same_frozen_candidate_and_labels'):
        study.compare_evaluation_reports(paths,tmp_path/'bad.json')
