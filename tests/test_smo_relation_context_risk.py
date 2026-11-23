"""Admitted relations keep the exact measured action, including its uncertainty."""
from copy import deepcopy

import pytest

from smo.statistical_filter import filter_graph
from test_smo_goal_obstruction import fixture, strength


def measured_fixture():
    g,f,c,p,a=fixture()
    g['metric_scope']='ue_population.v1'
    for edge in g['edges']:
        for row in edge.get('sensitivities',[]):
            row.update(confidence_level=.95,members=['u'])
    return g,f,c,p,a


def test_exact_light_sleep_estimate_is_used_without_borrowing_deep_sleep():
    g,f,c,p,a=measured_fixture()
    edge=dict(id='sleep:k',kind='influences',source='rcp:cell_sleep_mode',target='kpi:k',sensitivities=[])
    for mode,gain,ci in [('deep_sleep',-8,[-9,-7]),('light_sleep',-1,[-2,1])]:
        edge['sensitivities'].append(dict(kind='finite_contrast',parameter='cell_sleep_mode',
            parameter_component='all_cells',contrast_id=mode,members=['c'],population_control_eligible=True,
            confidence_level=.95,unit='W / contrast',estimate=gain,confidence_interval=ci,
            **{'from':'active','to':mode}))
    g['edges'].append(edge);p['apps'][1]['actions']=[a('cell_sleep_mode','light_sleep')]
    graph=filter_graph(g,retain_measured_contexts=True)
    result=strength(graph,f,c,p);q=result['groups'][0]
    assert q['score']==pytest.approx(100*.1/1.1)
    assert q['model_coverage_complete'] and q['score_status']=='reference_approximation'
    assert q['projection_coverage']['covered_kpis']==1
    chains=q['actual_outcomes'][0]['goals'][0]['chains']
    sleep=next(x for x in chains if x['parameter']=='cell_sleep_mode')
    assert sleep['drive']==-1 and sleep['calibrations']==['light_sleep']
    assert 'uncertain_gain_direction'in sleep['notes']
    assert not result['observed_after_kpis_used']
    # Changing an inapplicable significant profile cannot change this action.
    edge['sensitivities'][0].update(estimate=-800,confidence_interval=[-900,-700])
    assert strength(filter_graph(g,retain_measured_contexts=True),f,c,p)['groups'][0]['score']==pytest.approx(q['score'])
    p['apps'][1]['actions']=[a('cell_sleep_mode','never_measured_sleep')]
    assert strength(graph,f,c,p)['groups'][0]['score']is None


def test_weight_action_uses_its_measured_population_not_another_group():
    g,f,c,p,a=measured_fixture()
    weak=dict(kind='range_linear_fit',parameter='ue_scheduler_weights',parameter_component='mec_priority',
        contrast_id='mec_weights',members=['u'],population_control_eligible=True,confidence_level=.95,
        unit='W / weight',x_unit='weight',fit_domain=[.5,4],estimate=-.5,confidence_interval=[-1,1])
    strong=deepcopy(weak);strong.update(contrast_id='other_population',members=['v'],estimate=10,confidence_interval=[9,11])
    g['edges'].append(dict(id='weights:k',kind='influences',source='rcp:ue_scheduler_weights',target='kpi:k',sensitivities=[weak,strong]))
    p['apps'][1]['actions']=[a('scheduler_mode','round_robin'),a('ue_scheduler_weights',2)]
    result=strength(filter_graph(g,retain_measured_contexts=True),f,c,p);q=result['groups'][0]
    assert q['score']==pytest.approx(100*.05/1.05)
    chain=next(x for x in q['actual_outcomes'][0]['goals'][0]['chains']if x['parameter']=='ue_scheduler_weights')
    assert chain['drive']==-.5 and chain['calibrations']==['mec_weights']
    assert q['model_coverage_complete']
