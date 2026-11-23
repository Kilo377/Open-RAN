"""Static policy capability semantics are separate from dynamic proposals."""
from copy import deepcopy
import hashlib
import json

import pytest

from smo.candidate_selection import filter_catalog, select_combinations
from smo.graph_combinations import enumerate_combinations
from smo.xapp_capabilities import build_capabilities, required_directions
from test_smo_candidate_selection import fixture, groups


def semantic_fixture():
    graph,frame,plan=fixture()
    for node,targets in zip(graph['nodes'],[
        {'power':'minimize'},{'delay':'maximize'},{'ue_power':'minimize'},{'delay':'minimize'}]):
        node['target_kpis']=targets
        node['model_sha256']='a'*64
        node['runtime_parameters']={}
    # 'delay' is an arbitrary KPI key here: the intent asks it to increase.
    # The selector must follow the interval, not a preferred KPI direction.
    plan['ranges'][1]=dict(kpi='delay',lower=10.,upper=None,lower_inclusive=False,upper_inclusive=True)
    for e in graph['edges']:
        if e['id']=='influence:p:delay':e['sensitivities'][0]['importance']['signed_reference_importance']=1.
        if e['id']=='influence:r:delay':e['sensitivities'][0]['importance']['signed_reference_importance']=.4
    return graph,frame,plan


def select(graph,frame,plan):
    return select_combinations(graph,frame,plan,settings=dict(method='semantic_influence_coverage_v2'))


def test_single_energy_app_or_two_energy_apps_do_not_cover_increasing_other_kpi():
    graph,frame,plan=semantic_fixture();before=deepcopy((graph,frame,plan))
    old=select_combinations(graph,frame,plan,settings=dict(method='influence_coverage_v1'))
    assert frozenset(['xapp:A'])in groups(old)
    assert frozenset(['xapp:A','xapp:C'])in groups(old)
    result=select(graph,frame,plan)
    assert frozenset(['xapp:A'])not in groups(result)
    assert frozenset(['xapp:A','xapp:C'])not in groups(result)
    assert frozenset(['xapp:A','xapp:B'])in groups(result)
    assert groups(result)|groups(result,'reserve_groups')==groups(enumerate_combinations(graph,frame,plan))
    for g in result['groups']:
        assert all(v['supporting_apps'] for v in g['initial_selection']['direction_coverage'].values())
    assert (graph,frame,plan)==before


def test_adding_an_opposed_controller_does_not_veto_a_supported_combination():
    graph,frame,plan=semantic_fixture();result=select(graph,frame,plan)
    assert frozenset(['xapp:A','xapp:B','xapp:D'])in groups(result)
    raw=enumerate_combinations(graph,frame,plan)
    for g in result['groups']:
        prior=next(x for x in raw['groups']if x['id']==g['id'])
        assert prior['raw_conflicts']==g['raw_conflicts']


def test_gain_signs_current_state_and_actions_cannot_change_capability_directions(monkeypatch):
    graph,frame,plan=semantic_fixture();expected=groups(select(graph,frame,plan))
    for e in graph['edges']:
        for s in e.get('sensitivities',[]):
            s['estimate']*=-1;s['importance']['signed_reference_importance']*=-1
    frame['states']={'xapp:B':dict(current_proposals=[dict(parameter='p',proposed=-999)]),'kpi:delay':dict(value=999)}
    graph['nodes'][1]['label']='Reduce something else; labels are not semantics'
    result=select(graph,frame,plan)
    assert groups(result)==expected
    assert not result['initial_selection']['current_actions_used']
    assert not result['initial_selection']['model_inference_invoked']
    assert not result['initial_selection']['gain_signs_used_for_capability']


def test_only_matching_direction_sources_set_the_strength_denominator():
    graph,frame,plan=semantic_fixture()
    next(e for e in graph['edges']if e['id']=='influence:p:delay')['sensitivities'][0]['importance']['signed_reference_importance']=100.
    result=select(graph,frame,plan)
    assert result['initial_selection']['per_kpi_scales']['delay']==1.
    assert frozenset(['xapp:A','xapp:B'])in groups(result)


def test_unknown_directions_do_not_fall_back_into_primary():
    graph,frame,plan=semantic_fixture()
    graph['nodes'][1]['target_kpis']={}
    result=select(graph,frame,plan)
    assert result['group_count']==0 and result['reserve_group_count']==result['structural_group_count']
    assert result['direction_missing_kpis']==['delay']
    assert not result['initial_selection']['exhaustive_fallback']


@pytest.mark.parametrize('row,expected',[
    (dict(lower=20,upper=None,lower_inclusive=False),['increase']),
    (dict(lower=0,upper=10),['decrease']),
    (dict(lower=0,upper=0),['decrease']),
    (dict(lower=.8,upper=1,domain=[0,1]),['increase']),
    (dict(lower=20,upper=30),['increase','decrease','maintain']),
    (dict(lower=0,upper=None),['increase','decrease','maintain']),
])
def test_interval_semantics_are_stable_even_when_current_value_changes(row,expected):
    a=required_directions(dict(row,current=1));b=required_directions(dict(row,current=100))
    assert a==b and a['directions']==expected


def test_upper_bound_controller_cannot_claim_increasing_capability():
    graph,frame,plan=semantic_fixture();graph['nodes'][1]['target_kpis']={'delay':'upper_bound'}
    assert not select(graph,frame,plan)['groups']
    plan['ranges'][1]=dict(kpi='delay',lower=0,upper=15)
    assert select(graph,frame,plan)['groups']


def test_declared_direction_still_requires_an_unlocked_graph_path():
    graph,frame,plan=semantic_fixture()
    graph['edges'][1]['deployment']='native_locked'
    assert not select(graph,frame,plan)['groups']


def secondary_catalog(tmp_path,graph,**changes):
    for e in graph['edges']:
        for s in e.get('sensitivities',[]):s['scenario_id']='native_1x'
    source=tmp_path/'audit.json';source.write_text('audited frozen-policy comparison fixture')
    row=dict(app_id='xapp:C',kpi='delay',direction='increase',model_sha256='a'*64,
        runtime_parameters={},metric_scope='ue_population.v1',scenario_id='native_1x',
        kind='matched_frozen_policy_on_off',samples=[dict(seed=1,before=1.,after=3.),dict(seed=2,before=2.,after=4.)],
        confidence_interval=[1.,3.],source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    row.update(changes);path=tmp_path/'capabilities.json';path.write_text(json.dumps(dict(schema='oran.xapp-capabilities.v1',estimates=[row])))
    return path


def test_audited_secondary_direction_is_eligible_without_being_a_primary_objective(tmp_path):
    graph,frame,plan=semantic_fixture();path=secondary_catalog(tmp_path,graph)
    result=select_combinations(graph,frame,plan,settings=dict(method='semantic_influence_coverage_v2',capability_catalog=str(path)))
    assert frozenset(['xapp:A','xapp:C'])in groups(result)
    cell=result['initial_selection']['xapp_direction_capabilities']['profiles']['xapp:C']['delay']
    assert cell['status']=='validated_secondary' and cell['directions']==['increase']
    assert result['initial_selection']['historical_capability_evidence_used']


@pytest.mark.parametrize('changes',[
    dict(model_sha256='b'*64),dict(runtime_parameters=dict(changed=True)),
    dict(metric_scope='legacy_target'),dict(scenario_id='burst_2x'),dict(confidence_interval=[-1.,3.]),
])
def test_unmatched_or_uncertain_secondary_evidence_does_not_create_a_capability(tmp_path,changes):
    graph,frame,plan=semantic_fixture();path=secondary_catalog(tmp_path,graph,**changes)
    caps=build_capabilities(graph,str(path))
    assert 'delay'not in caps['profiles']['xapp:C']
    assert caps['rejected_secondary'] and not caps['applied_secondary']


def test_cannot_use_parameter_interventions_or_unpinned_outcomes_as_capabilities(tmp_path):
    graph,frame,plan=semantic_fixture();path=secondary_catalog(tmp_path,graph,kind='controlled_parameter_perturbation')
    with pytest.raises(ValueError,match='frozen-policy'):build_capabilities(graph,str(path))
    path=secondary_catalog(tmp_path,graph);(tmp_path/'audit.json').write_text('changed')
    with pytest.raises(ValueError,match='hash'):build_capabilities(graph,str(path))
