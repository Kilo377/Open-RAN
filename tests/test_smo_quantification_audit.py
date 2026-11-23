"""Regressions for control-chain coverage, frozen references and ratio scales."""
from copy import deepcopy

import pytest

from test_smo_goal_obstruction import fixture, strength


def test_partially_measured_population_cannot_be_a_complete_v2_score():
    g,f,c,p,a=fixture()
    for edge in g['edges']:
        for row in edge.get('sensitivities',[]):row['members']=['u']
    extra=a('ue_offload_fraction',.6)
    extra.update(ue_id='never_measured',current=.5)
    p['apps'][0]['actions'].append(extra)
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']is None and q['reason']=='reference_projection_incomplete'
    assert not q['model_coverage_complete']
    missing=q['projection_coverage']['unknown_links']
    assert missing[0]['action_coverage']['uncovered_entities']==['never_measured']


def test_unknown_drx_component_does_not_hide_behind_a_known_inactivity_fit():
    g,f,c,p,_=fixture()
    p['apps'][1]['actions'][0]['proposed']['on_duration_ms']=5
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']is None and not q['model_coverage_complete']
    missing=q['projection_coverage']['unknown_links'][0]
    assert missing['parameter']=='ue_drx_configs'
    assert missing['action_coverage']['uncovered_components']=={'u':['on_duration_ms']}


def test_neutral_physical_domain_has_explicit_zero_strength():
    g,f,c,p,_=fixture()
    c['ranges'][0].update(lower=0,upper=None,domain=[0,None],lower_inclusive=True)
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']==0 and q['raw_obstruction_strength']==0
    goal=q['goal_components'][0]
    assert goal['neutral_goal'] and goal['intent_margin_factor']==0
    assert goal['obstruction_strength']==0


def ratio_fixture():
    g,f,c,p,_=fixture();key='urllc_failure_fraction'
    g['nodes'][0].update(id='kpi:'+key,unit='ratio')
    for edge in g['edges']:
        if edge['kind']!='influences':continue
        edge['target']='kpi:'+key
        for row in edge['sensitivities']:
            coordinate='ms'if row['parameter']=='ue_drx_configs'else'fraction'
            row.update(kpi=key,estimate=.1,confidence_interval=[.1,.1],unit='ratio / '+coordinate)
    f['states']['kpi:'+key]=dict(value=0.)
    c['ranges']=[dict(kpi=key,current=0.,unit='ratio',lower=0.,upper=.01,domain=[0.,1.],lower_inclusive=True,upper_inclusive=True)]
    c['groups']=[dict(id='a',app_ids=['a'])]
    return g,f,c,p


def test_zero_failure_fraction_has_canonical_scale_and_a_finite_risk():
    g,f,c,p=ratio_fixture();r=strength(g,f,c,p)
    assert r['status']=='ready' and r['intent_contexts'][0]['scale']==1
    assert r['groups'][0]['score']==pytest.approx(100*(.02/1.01)/(1+.02/1.01))


def test_failure_fraction_and_percentage_coordinates_are_equivalent():
    g,f,c,p=ratio_fixture();before=strength(g,f,c,p)['groups'][0]['score']
    g['nodes'][0]['unit']='%'
    for edge in g['edges']:
        for row in edge.get('sensitivities',[]):
            row['unit']=row['unit'].replace('ratio / ','% / ')
            row['estimate']*=100;row['confidence_interval']=[x*100 for x in row['confidence_interval']]
    row=c['ranges'][0];row.update(unit='%',upper=1.,domain=[0.,100.])
    assert strength(g,f,c,p)['groups'][0]['score']==pytest.approx(before)


def test_missing_current_observation_is_not_misreported_as_missing_gain():
    g,f,c,p,_=fixture();f['states'].pop('kpi:k');c['ranges'][0]['current']=10
    r=strength(g,f,c,p)
    assert r['status']=='unavailable' and r['reason']=='kpi_observation_unavailable'


def test_intent_anchor_order_cannot_change_the_score():
    g,f,c,p,_=fixture()
    c['ranges'][0]['anchors']=[dict(intent_id='later',current=20,snapshot=dict(slot=200,time_ms=100)),
        dict(intent_id='earlier',current=10,snapshot=dict(slot=100,time_ms=50))]
    before=strength(g,f,c,p)
    c['ranges'][0]['anchors'].reverse()
    after=strength(g,f,c,p)
    assert before['groups'][0]['score']==after['groups'][0]['score']
    assert before['intent_contexts'][0]['scale']==after['intent_contexts'][0]['scale']==10
