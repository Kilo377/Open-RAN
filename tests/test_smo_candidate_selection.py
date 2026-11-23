"""Influence screening must not masquerade as action or feasibility inference."""
from copy import deepcopy

import pytest

from smo.candidate_selection import SelectionConfig, filter_catalog, options, select_combinations
from smo.graph_combinations import enumerate_combinations
from smo.__main__ import graph_intent_review, GraphService


def fixture():
    nodes = [dict(id='xapp:'+a,kind='xapp',label=a,target_kpis={}) for a in 'ABCD']
    controls = [dict(id='control:'+a,kind='controls',source='xapp:'+a,target='rcp:'+p,
        deployment='controllable') for a,p in zip('ABCD','pqrq')]
    weights = {'p':(1.,.02),'q':(.02,1.),'r':(.01,.01)}
    effects = [dict(id='influence:'+p+':'+k,kind='influences',source='rcp:'+p,target='kpi:'+k,
        sensitivities=[dict(contrast_id='test',estimate=v,confidence_interval=[-2,2],
            population_control_eligible=True,importance=dict(signed_reference_importance=v,
                normalization_basis='absolute_reference_kpi'))])
        for p,vs in weights.items() for k,v in zip(('power','delay'),vs)]
    graph = dict(metric_scope='ue_population.v1',nodes=nodes,edges=controls+effects)
    frame = dict(slot=10,time_ms=5,states={})
    plan = dict(snapshot=dict(slot=10),ranges=[dict(kpi=k,lower=0,upper=9) for k in ('power','delay')],questions=[])
    return graph,frame,plan


def groups(catalog, field='groups'):
    return {frozenset(g['app_ids']) for g in catalog[field]}


def test_two_kpis_require_meaningful_coverage_and_weak_additions_have_witnesses():
    graph,frame,plan=fixture(); original=deepcopy((graph,frame,plan))
    raw=enumerate_combinations(graph,frame,plan)
    result=select_combinations(graph,frame,plan,settings=dict(method='influence_coverage_v1'))
    assert groups(result)=={frozenset(('xapp:A','xapp:B')),frozenset(('xapp:A','xapp:D')),
        frozenset(('xapp:A','xapp:B','xapp:D'))}
    assert result['structural_group_count']==15 and result['group_count']==3 and result['reserve_group_count']==12
    assert groups(result)|groups(result,'reserve_groups')==groups(raw)
    assert not groups(result)&groups(result,'reserve_groups')
    for g in result['reserve_groups']:
        info=g['initial_selection']
        if info['reason']=='low_marginal_contribution':
            witness=next(x for x in result['groups']if x['id']==info['witness_group_id'])
            assert set(witness['app_ids'])<set(g['app_ids'])
            assert info['maximum_coverage_loss']<=.05
    assert (graph,frame,plan)==original


def test_uncertain_coefficients_and_different_signs_are_not_removed_as_missing():
    graph,frame,plan=fixture()
    result=select_combinations(graph,frame,plan,settings=dict(method='influence_coverage_v1'))
    # Every fixture confidence interval crosses zero. Magnitudes still work.
    for e in graph['edges']:
        for r in e.get('sensitivities',[]):
            r['estimate'] *= -1
            r['importance']['signed_reference_importance'] *= -1
    changed=select_combinations(graph,frame,plan,settings=dict(method='influence_coverage_v1'))
    assert groups(result)==groups(changed)


def test_meaningful_distinct_controls_and_shared_trained_policies_are_preserved():
    graph,frame,plan=fixture()
    e=next(e for e in graph['edges']if e['id']=='influence:r:delay')
    e['sensitivities'][0]['importance']['signed_reference_importance']=.2
    result=select_combinations(graph,frame,plan,settings=dict(method='influence_coverage_v1'))
    assert frozenset(('xapp:A','xapp:B','xapp:C')) in groups(result)
    assert frozenset(('xapp:A','xapp:B','xapp:D')) in groups(result)
    raw=enumerate_combinations(graph,frame,plan)
    for g in result['groups']:
        original=next(x for x in raw['groups']if x['id']==g['id'])
        assert g['conflicts']==original['conflicts'] and g['raw_conflicts']==original['raw_conflicts']


def test_legacy_method_is_exact_and_no_observed_results_or_proposals_are_needed():
    graph,frame,plan=fixture()
    assert select_combinations(graph,frame,plan)==enumerate_combinations(graph,frame,plan)
    raw=enumerate_combinations(graph,frame,plan)
    raw['observed_after_kpis']={'power':999}
    original=deepcopy(raw)
    result=filter_catalog(graph,raw,method='influence_coverage_v1')
    assert raw==original
    assert not result['initial_selection']['observed_outcomes_used']
    assert not result['initial_selection']['action_values_used']
    assert not result['initial_selection']['conflict_scores_used']
    assert not result['initial_selection']['final_strategy_selected']
    assert not result['attainment_verified'] and not result['automatic_selection']


def test_missing_normalized_calibration_is_preserved_not_treated_as_zero():
    graph,frame,plan=fixture()
    for e in graph['edges']:
        if e['kind']=='influences':
            e['sensitivities'][0].pop('importance')
    result=select_combinations(graph,frame,plan,settings=dict(method='influence_coverage_v1'))
    assert result['group_count']==15 and result['reserve_group_count']==0
    assert all(g['initial_selection']['reason']=='unquantified_evidence_preserved' for g in result['groups'])


def test_zero_reference_capability_is_not_a_reason_to_remove_every_group():
    graph,frame,plan=fixture()
    raw=enumerate_combinations(graph,frame,plan)
    for e in graph['edges']:
        if e['kind']=='influences':
            e['sensitivities'][0]['importance']['signed_reference_importance']=0
    result=filter_catalog(graph,raw,method='influence_coverage_v1')
    assert result['group_count']==15
    assert result['initial_selection']['per_kpi_scales']==dict(power=0,delay=0)


def test_conflicting_intent_bounds_and_blocked_reviews_cannot_create_candidates():
    graph,frame,plan=fixture();plan['ranges'][0].update(lower=11,upper=9)
    result=select_combinations(graph,frame,plan,settings=dict(method='influence_coverage_v1'))
    assert result['empty_intersection'] and not result['groups'] and not result['reserve_groups']


def test_service_dispatches_only_initial_selection_and_leaves_default_exhaustive():
    graph,frame,plan=fixture()
    a=graph_intent_review(graph,frame,plan)
    b=graph_intent_review(graph,frame,plan,dict(method='influence_coverage_v1'))
    assert a['combination_catalog']['group_count']==15
    assert b['combination_catalog']['group_count']==3
    assert not b['model_inference_invoked'] and not b['automatic_selection']
    assert GraphService().candidate_selection['method']=='reverse_graph'
    assert GraphService(candidate_selection=dict(method='influence_coverage_v1')).candidate_selection['method']=='influence_coverage_v1'


def test_explicit_batch_evaluation_uses_primary_pool_without_scheduling_reserve(monkeypatch):
    graph,frame,plan=fixture()
    service=GraphService(candidate_selection=dict(method='influence_coverage_v1'))
    service.graph=graph
    monkeypatch.setattr(service,'intent_context',lambda slot:(dict(source=dict(name='fixture')),frame))
    monkeypatch.setattr(service.intent_ledger,'select',lambda ids:[dict(id='fixture')])
    monkeypatch.setattr('smo.__main__.intersect_intents',lambda *args:plan)
    service.evaluation_origin=dict(origin_id='fixture-origin',observe_slots=500)
    frame['evaluation_origin']=dict(origin_id='fixture-origin')
    captured={}
    monkeypatch.setattr(service.deployments.batch,'start',lambda **kwargs:captured.update(kwargs) or kwargs)
    service.evaluate_all_combinations(['fixture'])
    assert len(captured['groups'])==3
    assert all(g['initial_selection']['pool']=='primary' for g in captured['groups'])
    assert all('xapp:C'not in g['app_ids'] for g in captured['groups'])
    # Switching methods restores all groups; no graph/intent edits are needed.
    service.candidate_selection=options(dict(method='reverse_graph'))
    service.evaluate_all_combinations(['fixture'])
    assert len(captured['groups'])==15


@pytest.mark.parametrize('settings',[
    dict(method='invented'),dict(coverage_threshold=float('nan')),
    dict(coverage_threshold=True),dict(marginal_tolerance=-.1),
    dict(preserve_shared_policies='yes'),dict(unrecognized=2),[],
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises((ValueError,TypeError)):
        options(settings)
