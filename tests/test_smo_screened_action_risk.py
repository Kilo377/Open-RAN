"""Screened measurements are omitted; unmeasured actions remain unknown."""
from copy import deepcopy

import pytest

from smo.statistical_filter import filter_graph
from test_smo_goal_obstruction import fixture, strength


def mixed_scalar_fixture():
    g,f,c,p,a=fixture()
    g['metric_scope']='ue_population.v1'
    for edge in g['edges']:
        for row in edge.get('sensitivities',[]):
            row.update(confidence_level=.95,members=['u'])
    drx=next(e for e in g['edges'] if e['kind']=='influences' and e['source']=='rcp:ue_drx_configs')
    drx['sensitivities'][0].update(estimate=-100,confidence_interval=[-1000,1000])
    cycle=deepcopy(drx['sensitivities'][0])
    cycle.update(contrast_id='cycle',scalar_component='cycle_ms',estimate=2,confidence_interval=[1,3],fit_domain=[0,100])
    drx['sensitivities'].append(cycle)
    return g,f,c,p,a


def test_partial_scalar_screening_does_not_invalidate_retained_model():
    g,f,c,p,_=mixed_scalar_fixture()
    result=strength(filter_graph(g),f,c,p)
    q=result['groups'][0]
    assert q['score']==0 and q['score_status']=='partial_model_score'
    assert not result['screened_gain_values_used_for_score']
    assert not result['observed_after_kpis_used']
    assert not q['projection_coverage']['unknown_links']
    omitted=q['projection_coverage']['excluded_links']
    assert omitted[0]['action_context']['calibrations']==['ue_drx_configs']
    assert all(not any(x['parameter']=='ue_drx_configs' for x in goal['chains'])
        for outcome in q['actual_outcomes'] for goal in outcome['goals'])


def test_screened_coefficient_value_cannot_change_score():
    g,f,c,p,_=mixed_scalar_fixture()
    first=strength(filter_graph(g),f,c,p)['groups'][0]
    g['edges'][1]['sensitivities'][0].update(estimate=999,confidence_interval=[-10000,10000])
    second=strength(filter_graph(g),f,c,p)['groups'][0]
    assert first['score']==second['score']
    assert first['raw_obstruction_strength']==second['raw_obstruction_strength']


def test_missing_active_goal_coverage_is_explicit_even_with_numeric_model_score():
    g,f,c,p,_=mixed_scalar_fixture()
    g['nodes'].append(dict(id='kpi:k2',kind='kpi',unit='W'))
    f['states']['kpi:k2']=dict(value=10.)
    c['ranges'].append(dict(kpi='k2',lower=10.,upper=None,lower_inclusive=False))
    for original in list(g['edges']):
        if original['kind']!='influences':continue
        edge=deepcopy(original)
        edge.update(id=edge['id']+':k2',target='kpi:k2')
        for row in edge['sensitivities']:
            row.update(estimate=1,confidence_interval=[-2,2])
        g['edges'].append(edge)
    q=strength(filter_graph(g),f,c,p)['groups'][0]
    assert q['score']==0 and q['score_status']=='partial_model_score'
    assert q['model_coverage_complete']is False
    assert q['projection_coverage']['covered_kpis']==1
    assert q['projection_coverage']['required_kpis']==2


@pytest.mark.parametrize('changes',[dict(on_duration_ms=5),dict(inactivity_ms=41)])
def test_unmeasured_component_or_out_of_domain_action_is_not_screened_zero(changes):
    g,f,c,p,_=mixed_scalar_fixture()
    p['apps'][1]['actions'][0]['proposed'].update(changes)
    q=strength(filter_graph(g),f,c,p)['groups'][0]
    assert q['score']is None and q['reason']=='reference_projection_incomplete'
    assert any(x['parameter']=='ue_drx_configs' for x in q['projection_coverage']['unknown_links'])


def sleep_fixture(target='light_sleep'):
    g,f,c,p,a=mixed_scalar_fixture()
    edge=dict(id='sleep:k',kind='influences',source='rcp:cell_sleep_mode',target='kpi:k',sensitivities=[])
    for mode,estimate,ci in [('deep_sleep',-8,[-9,-7]),('light_sleep',-1,[-2,1])]:
        edge['sensitivities'].append(dict(kind='finite_contrast',parameter='cell_sleep_mode',
            parameter_component='all_cells',contrast_id=mode,members=['c'],population_control_eligible=True,
            confidence_level=.95,unit='W / contrast',estimate=estimate,confidence_interval=ci,
            **{'from':'active','to':mode}))
    g['edges'].append(edge)
    p['apps'][1]['actions']=[a('cell_sleep_mode',target)]
    return g,f,c,p


def test_partial_sleep_screening_is_exact_to_recorded_profile():
    g,f,c,p=sleep_fixture()
    q=strength(filter_graph(g),f,c,p)['groups'][0]
    assert q['score']==0
    omitted=q['projection_coverage']['excluded_links']
    assert omitted[0]['action_context']['calibrations']==['light_sleep']
    g,f,c,p=sleep_fixture('never_measured_sleep')
    q=strength(filter_graph(g),f,c,p)['groups'][0]
    assert q['score']is None
    assert q['projection_coverage']['unknown_links'][0]['parameter']=='cell_sleep_mode'


def test_non_significant_but_incompatible_units_do_not_become_omissions():
    g,f,c,p=sleep_fixture()
    g['edges'][-1]['sensitivities'][1]['unit']='ms / contrast'
    q=strength(filter_graph(g),f,c,p)['groups'][0]
    assert q['score']is None
