"""A significant relation can have uncertain, explicitly measured actions."""
from copy import deepcopy

import pytest

from smo.statistical_filter import filter_graph


def graph():
    def row(name,value,ci,**extra):
        return dict(parameter='cell_sleep_mode',contrast_id=name,kind='finite_contrast',
            population_control_eligible=True,members=['c'],confidence_level=.95,
            estimate=value,confidence_interval=ci,unit='W / contrast',
            **{'from':'active','to':name},**extra)
    return dict(metric_scope='ue_population.v1',nodes=[dict(id='kpi:k',kind='kpi',unit='W')],edges=[
        dict(id='control',kind='controls',source='x',target='rcp:cell_sleep_mode'),
        dict(id='sleep:k',kind='influences',source='rcp:cell_sleep_mode',target='kpi:k',
            sensitivities=[row('deep_sleep',-8,[-9,-7]),row('light_sleep',-1,[-2,1]),
                row('micro_sleep',0,[0,0],zero_in_both=True)])])


def test_relation_admission_keeps_exact_measured_conditional_gains_and_uncertainty():
    raw=graph();before=deepcopy(raw)
    strict=filter_graph(raw)
    result=filter_graph(raw,retain_measured_contexts=True)
    edge=result['edges'][1]
    assert [e['id']for e in result['edges']]==[e['id']for e in strict['edges']]
    assert result['edges'][0]==raw['edges'][0]
    assert [r['contrast_id']for r in edge['sensitivities']]==['deep_sleep','light_sleep','micro_sleep']
    for old,new in zip(raw['edges'][1]['sensitivities'],edge['sensitivities']):
        assert all(new[k]==value for k,value in old.items())
    assert edge['statistical_selection']['significant_contrasts']==['deep_sleep']
    assert edge['statistical_selection']['uncertain_contrasts']==['light_sleep']
    assert edge['statistical_selection']['observed_zero_contrasts']==['micro_sleep']
    assert edge['statistical_selection']['action_context_policy']=='retain_valid_measured'
    assert not edge['sensitivities'][1]['statistical_context']['direction_significant']
    assert result['statistical_filter']['retained_uncertain_estimate_count']==1
    assert result['statistical_filter']['retained_observed_zero_estimate_count']==1
    assert result['statistical_filter']['significant_estimate_count']==1
    assert not result['statistical_filter']['partially_filtered_relations']
    assert raw==before


def test_legacy_default_remains_strict_per_context():
    result=filter_graph(graph())
    assert [r['contrast_id']for r in result['edges'][1]['sensitivities']]==['deep_sleep']
    assert 'action_context_policy'not in result['statistical_filter']
    assert result['statistical_filter']['excluded_estimate_count']==2


def test_unestablished_relation_remains_absent_and_cannot_be_restored_by_uncertain_rows():
    raw=graph();raw['edges'][1]['sensitivities']=raw['edges'][1]['sensitivities'][1:]
    result=filter_graph(raw,retain_measured_contexts=True)
    assert [e['id']for e in result['edges']]==['control']
    assert result['statistical_filter']['active_edge_count']==0
    assert result['statistical_filter']['active_estimate_count']==0
    assert result['statistical_filter']['retained_uncertain_estimate_count']==0
    assert result['statistical_filter']['excluded_relations'][0]['reason']=='no_significant_estimates'


@pytest.mark.parametrize('changes,reason',[
    (dict(unit='ms / contrast'),'gain_units_invalid'),
    (dict(population_control_eligible=False),'population_evidence_unavailable'),
    (dict(confidence_level=None),'confidence_level_unverified'),
    (dict(confidence_interval=None),'confidence_interval_unavailable'),
    (dict(estimate=3),'estimate_outside_interval'),
    (dict(estimate=1,confidence_interval=[.5,2],zero_in_both=True),'observed_zero_inconsistent'),
])
def test_invalid_conditional_measurement_is_not_reintroduced(changes,reason):
    raw=graph();raw['edges'][1]['sensitivities'][1].update(changes)
    result=filter_graph(raw,retain_measured_contexts=True)
    edge=result['edges'][1]
    assert 'light_sleep'not in [r['contrast_id']for r in edge['sensitivities']]
    audit=result['statistical_filter']['partially_filtered_relations'][0]
    assert audit['excluded_estimates'][0]['reason']==reason
    assert not audit['excluded_estimates'][0]['screened_effect_eligible']


def test_incompatible_significant_gain_cannot_admit_a_relation():
    raw=graph();raw['edges'][1]['sensitivities'][0]['unit']='ms / contrast'
    result=filter_graph(raw,retain_measured_contexts=True)
    assert result['statistical_filter']['active_edge_count']==0
    assert result['statistical_filter']['reasons']['gain_units_invalid']==1


def test_gain_value_sign_does_not_resolve_an_uncertain_direction_by_fiat():
    raw=graph();raw['edges'][1]['sensitivities'][1].update(estimate=1,confidence_interval=[-2,2])
    result=filter_graph(raw,retain_measured_contexts=True)
    row=result['edges'][1]['sensitivities'][1]
    assert row['estimate']==1 and row['confidence_interval']==[-2,2]
    assert row['statistical_context']['screening_reason']=='interval_includes_zero'
    assert row['statistical_context']['retained_as']=='uncertain_reference'


@pytest.mark.parametrize('value',[None,1,'yes'])
def test_context_policy_requires_explicit_boolean(value):
    with pytest.raises(ValueError,match='must be boolean'):
        filter_graph(graph(),retain_measured_contexts=value)
