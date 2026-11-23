"""Analytic checks of the actual goal-obstruction engine, not xApp results."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from test_smo_conflict_quantification import fixture as base_fixture
from smo.influence_units import align_graph

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    g,f,c,p,a=base_fixture()
    g['nodes']=[dict(id='kpi:k',kind='kpi',unit='W')]
    for e in g['edges']:
        for r in e.get('sensitivities',[]):
            coordinate='ms'if r['parameter']=='ue_drx_configs'else'fraction'
            r.update(unit='W / '+coordinate,x_unit=coordinate)
    return g,f,c,p,a


def review(g,f,c,p,**options):
    # Retained v1 regression cases below explicitly exercise the old formula.
    options.setdefault('method','goal_obstruction_v1')
    node=shutil.which('node')
    if not node:pytest.skip('Node required for production goal-obstruction engine')
    code="""
const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
vm.runInContext(fs.readFileSync('smo/web/goal_obstruction.js','utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8')),original=JSON.stringify(d);
const result=c.SmoGoalObstruction.review(d.g,d.f,d.c.groups,d.c.ranges,d.p,d.options);
if(original!==JSON.stringify(d))throw Error('Risk input mutated');
process.stdout.write(JSON.stringify(result));
"""
    return json.loads(subprocess.check_output([node,'-e',code],cwd=ROOT,text=True,
        input=json.dumps(dict(g=align_graph(g),f=f,c=c,p=p,options=options))))


def test_compatible_opposition_measures_positive_capability_loss():
    g,f,c,p,_=fixture();r=review(g,f,c,p);q=r['groups'][0]
    assert r['method']=='goal_obstruction_v1' and q['score']==pytest.approx(50)
    goal=q['goal_components'][0]
    assert goal['potential']==pytest.approx(.2)
    assert goal['expected_goal_drive']==pytest.approx(.1)
    assert goal['expected_lost_positive_capability']==pytest.approx(.1)
    assert not r['kpi_forecast_performed'] and not r['observed_after_kpis_used']
    assert not r['selection_performed'] and not r['deployment_enabled']


def test_single_policy_is_not_split_to_invent_self_conflict():
    g,f,c,p,_=fixture();p['apps'][0]['actions']+=p['apps'][1]['actions'];c['groups']=[dict(id='a',app_ids=['a'])]
    q=review(g,f,c,p)['groups'][0]
    assert q['score']==0 and q['goal_components'][0]['potential']==pytest.approx(.1)
    assert q['reference_subset_count']==2


def test_same_parameter_uses_random_winner_not_additive_cancellation():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    q=review(g,f,c,p)['groups'][0]
    assert len(q['actual_outcomes'])==2 and q['score']==pytest.approx(50)
    assert q['arbitration_probability_mass']==pytest.approx(1)
    assert sorted(x['probability']for x in q['actual_outcomes'])==[.5,.5]
    # Both improve, but selecting the weaker request loses half the capability.
    p['apps'][1]['actions']=[a('ue_offload_fraction',.6)]
    assert review(g,f,c,p)['groups'][0]['score']==pytest.approx(25)


def test_equal_writes_are_not_added_or_counted_as_conflicting():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.7)]
    q=review(g,f,c,p)['groups'][0]
    assert q['score']==0 and len(q['actual_outcomes'])==1
    assert q['goal_components'][0]['potential']==pytest.approx(.2)


def test_source_probabilities_preserve_two_equal_requests_vs_one_other():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    p['apps'].append(dict(app_id='z',status='proposed',actions=[a('ue_offload_fraction',.7)]))
    g['edges'].append(dict(id='z:off',kind='controls',source='z',target='rcp:ue_offload_fraction'))
    c['groups'][0]['app_ids'].append('z');q=review(g,f,c,p)['groups'][0]
    assert q['score']==pytest.approx(100/3)


def test_same_direction_harm_is_not_zero_risk():
    g,f,c,p,a=fixture();p['apps'][0]['actions']=[a('ue_offload_fraction',.3)]
    q=review(g,f,c,p)['groups'][0]
    assert q['legacy_score']==0 and q['score']==100
    assert q['goal_components'][0]['potential']==0


def test_no_action_is_explicit_and_missing_evidence_is_not_zero():
    g,f,c,p,_=fixture()
    for app in p['apps']:app.update(status='no_action',inference_invoked=True,actions=[])
    q=review(g,f,c,p)['groups'][0]
    assert q['score']==0 and q['no_active_push'] and q['score_status']=='no_active_push'
    p['apps'][0].update(status='unavailable',inference_invoked=False)
    assert review(g,f,c,p)['groups'][0]['score']is None


def test_domain_floor_is_not_a_second_power_optimization_target():
    g,f,c,p,a=fixture();c['groups']=[dict(id='a',app_ids=['a'])]
    c['ranges'][0].update(lower=0,upper=90,lower_inclusive=True,upper_inclusive=False,domain=[0,None])
    f['states']['kpi:k']['value']=10;p['apps'][0]['actions']=[a('ue_offload_fraction',.3)]
    r=review(g,f,c,p);assert r['groups'][0]['score']==0
    assert [b['side']for b in r['intent_contexts'][0]['boundaries']]==['upper']


def test_bounded_band_centre_uses_directional_derivative_and_exact_hold_is_not_progress():
    g,f,c,p,_=fixture();c['ranges'][0].update(lower=5,upper=15,lower_inclusive=True)
    q=review(g,f,c,p)['groups'][0]
    assert q['score']==100 # Any nonzero movement consumes centre margin.
    c['ranges'][0].update(lower=10,upper=10)
    assert review(g,f,c,p)['groups'][0]['score']==100


def test_necessary_kpis_are_not_averaged_away():
    g,f,c,p,_=fixture()
    g['nodes'].append(dict(id='kpi:k2',kind='kpi',unit='W'))
    for param in ('ue_offload_fraction','ue_drx_configs'):
        e=deepcopy(next(e for e in g['edges']if e['kind']=='influences'and e['source']=='rcp:'+param))
        e.update(id=e['id']+'2',target='kpi:k2');e['sensitivities'][0].update(estimate=-10,confidence_interval=[-10,-10]);g['edges'].append(e)
    c['ranges'].append(dict(kpi='k2',lower=0,upper=10,lower_inclusive=True,upper_inclusive=False,domain=[0,None]));f['states']['kpi:k2']=dict(value=10)
    q=review(g,f,c,p)['groups'][0]
    assert q['score']==pytest.approx(100*max(x['expected_loss_fraction']for x in q['goal_components']))


def test_atomic_scheduler_actions_are_never_spliced_between_apps():
    g,f,c,p,a=fixture()
    for mode,effect in [('latency_aware',2),('qos_deadline',1)]:
        g['edges'].append(dict(id='mode:'+mode,kind='influences',source='rcp:scheduler_mode',target='kpi:k',
            sensitivities=[dict(kind='finite_contrast',parameter_component='all_cells',contrast_id=mode,
                parameter='scheduler_mode',unit='W / contrast',estimate=effect,confidence_interval=[effect,effect],**{'from':'round_robin','to':mode})]))
    # One influence edge must carry all category contrasts.
    mode_edges=[e for e in g['edges']if e['source']=='rcp:scheduler_mode'and e['kind']=='influences']
    mode_edges[0]['sensitivities']+=mode_edges[1]['sensitivities'];g['edges'].remove(mode_edges[1])
    g['edges'].append(dict(id='weights',kind='influences',source='rcp:ue_scheduler_weights',target='kpi:k',
        sensitivities=[dict(kind='range_linear_fit',parameter='ue_scheduler_weights',unit='W / weight',x_unit='weight',parameter_component='mec_priority',members=['u'],fit_domain=[0,10],estimate=1,confidence_interval=[1,1])]))
    for app,mode,weight in zip(p['apps'],['latency_aware','qos_deadline'],[2,3]):
        app['actions']=[a('scheduler_mode',mode),a('ue_scheduler_weights',weight)]
    q=review(g,f,c,p)['groups'][0]
    assert len(q['actual_outcomes'])==2
    assert all(len(x['winners'])==1 for x in q['actual_outcomes'])
    assert q['arbitration_resources'][0]['resource']==['scheduler_bundle','c']
    assert q['score']==0 and all(x['goals'][0]['goal_drive']==pytest.approx(.3)for x in q['actual_outcomes'])


def test_real_kpi_unit_conversion_is_automatic_and_invariant():
    g,f,c,p,_=fixture();before=review(g,f,c,p)
    g['nodes'][0]['unit']='mW';f['states']['kpi:k']['value']*=1000;c['ranges'][0]['lower']*=1000
    after=review(g,f,c,p)
    assert after['groups'][0]['score']==pytest.approx(before['groups'][0]['score'])
    assert after['groups'][0]['goal_components']==before['groups'][0]['goal_components']
    assert after['intent_contexts'][0]['weight']==1


def test_parameter_coordinate_conversion_and_bad_dimensions():
    g,f,c,p,_=fixture();before=review(g,f,c,p)['groups'][0]
    row=g['edges'][0]['sensitivities'][0]
    row.update(unit='W / %',x_unit='%',estimate=.1,confidence_interval=[.1,.1],fit_domain=[0,100])
    converted=review(g,f,c,p)['groups'][0]
    assert converted['score']==pytest.approx(before['score'])
    assert converted['goal_components']==before['goal_components']
    row['unit']='ms / %'
    assert review(g,f,c,p)['groups'][0]['score']is None


def test_public_reference_is_fixed_independent_of_target_distance_and_equal_kpi_value():
    g,f,c,p,_=fixture();c['ranges'][0]['lower']=12
    options=dict(normalization=dict(kpis=[dict(kpi='k',scale=25,scale_basis='batch_baseline_reference_magnitude',weight=999)]))
    r=review(g,f,c,p,**options)
    assert r['intent_contexts'][0]['scale']==25 and r['intent_contexts'][0]['weight']==1
    assert r['groups'][0]['goal_components'][0]['expected_goal_drive']==pytest.approx(1/25)
    f['states']['kpi:k']['value']=20
    assert review(g,f,c,p,**options)['intent_contexts'][0]['scale']==25
    options['normalization']['kpis'][0]['scale']=None
    assert review(g,f,c,p,**options)['reason']=='kpi_scale_unavailable'


def test_unknown_projection_and_outcome_caps_do_not_fabricate_scores():
    g,f,c,p,a=fixture();g['edges']=[e for e in g['edges']if e['kind']!='influences']
    assert review(g,f,c,p)['groups'][0]['score']is None
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    assert review(g,f,c,p,maxOutcomes=1)['groups'][0]['reason']=='arbitration_outcome_limit'
    assert review(g,f,c,p,maxReferenceOutcomes=1)['groups'][0]['reason']=='reference_space_limit'


def test_unit_conversion_order_and_future_outcomes_do_not_change_score():
    g,f,c,p,_=fixture();original=review(g,f,c,p)['groups'][0]['score']
    f['states']['kpi:k']['value']*=1000;c['ranges'][0]['lower']*=1000
    for e in g['edges']:
        for r in e.get('sensitivities',[]):r['estimate']*=1000;r['confidence_interval']=[x*1000 for x in r['confidence_interval']]
    c['groups'][0]['app_ids'].reverse();f['states']['future_actual_kpi']=dict(value=-99999)
    assert review(g,f,c,p)['groups'][0]['score']==pytest.approx(original)


def strength(g,f,c,p,**options):
    return review(g,f,c,p,method='goal_obstruction_v2',**options)


def test_v2_is_the_production_default_and_uses_absolute_reference_loss():
    g,f,c,p,_=fixture();r=review(g,f,c,p,method=None);q=r['groups'][0]
    assert r['method']=='goal_obstruction_v2' and r['schema']=='oran.goal-obstruction.v2'
    assert q['raw_obstruction_strength']==pytest.approx(.1)
    assert q['score']==pytest.approx(100*.1/1.1)
    assert q['fractional_score']==pytest.approx(50)
    assert q['goal_components'][0]['expected_reference_loss']==pytest.approx(.1)
    assert r['kpi_value_weights']=='equal' and not r['observed_after_kpis_used']


def test_same_fractional_loss_but_larger_reference_magnitude_is_more_severe():
    g,f,c,p,_=fixture();a=strength(g,f,c,p)['groups'][0]
    for e in g['edges']:
        for row in e.get('sensitivities',[]):
            row['estimate']*=2;row['confidence_interval']=[v*2 for v in row['confidence_interval']]
    b=strength(g,f,c,p)['groups'][0]
    assert b['fractional_score']==pytest.approx(a['fractional_score'])
    assert b['raw_obstruction_strength']==pytest.approx(2*a['raw_obstruction_strength'])
    assert b['score']>a['score']


def test_negative_goal_drive_is_not_clipped_at_available_positive_capability():
    g,f,c,p,a=fixture();p['apps'][0]['actions']=[a('ue_offload_fraction',.3)]
    q=strength(g,f,c,p)['groups'][0]
    assert q['goal_components'][0]['potential']==0
    assert q['raw_obstruction_strength']==pytest.approx(.3)
    assert q['score']==pytest.approx(100*.3/1.3)
    assert q['fractional_score']==100
    # There is no pre-display ceiling at one unit of reference influence.
    for e in g['edges']:
        for row in e.get('sensitivities',[]):
            row['estimate']*=10;row['confidence_interval']=[v*10 for v in row['confidence_interval']]
    q=strength(g,f,c,p)['groups'][0]
    assert q['raw_obstruction_strength']==pytest.approx(3)
    assert q['score']==pytest.approx(75)


def test_uniform_arbiter_expectation_uses_whole_execution_states():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    q=strength(g,f,c,p)['groups'][0]
    assert q['goal_components'][0]['expected_reference_loss']==pytest.approx(.2)
    assert q['raw_obstruction_strength']==pytest.approx(.2)
    assert sorted(x['goals'][0]['reference_loss_amount']for x in q['actual_outcomes'])==pytest.approx([0,.4])


def test_intent_safety_margin_is_separate_from_equal_kpi_value_weights():
    g,f,c,p,_=fixture();opts=dict(normalization=dict(kpis=[dict(kpi='k',scale=10,scale_basis='fixed')]))
    a=strength(g,f,c,p,**opts);f['states']['kpi:k']['value']=20
    b=strength(g,f,c,p,**opts)
    assert a['intent_contexts'][0]['weight']==b['intent_contexts'][0]['weight']==1
    assert a['intent_contexts'][0]['intent_margin_factor']==1
    assert b['intent_contexts'][0]['intent_margin_factor']==.5
    assert b['groups'][0]['raw_obstruction_strength']==pytest.approx(a['groups'][0]['raw_obstruction_strength']/2)


def test_v2_unit_invariance_policy_integrity_and_unknowns():
    g,f,c,p,_=fixture();a=strength(g,f,c,p)['groups'][0]
    g['nodes'][0]['unit']='mW';f['states']['kpi:k']['value']*=1000;c['ranges'][0]['lower']*=1000
    b=strength(g,f,c,p)['groups'][0]
    assert b['score']==pytest.approx(a['score']) and b['raw_obstruction_strength']==pytest.approx(a['raw_obstruction_strength'])
    p['apps'][0]['actions']+=p['apps'][1]['actions'];c['groups']=[dict(id='a',app_ids=['a'])]
    assert strength(g,f,c,p)['groups'][0]['score']==0
    p['apps'][0]['status']='unavailable'
    assert strength(g,f,c,p)['groups'][0]['score']is None


def test_v2_takes_worst_goal_strength_without_compensating_other_goals():
    g,f,c,p,_=fixture();g['nodes'].append(dict(id='kpi:k2',kind='kpi',unit='W'))
    for param in ('ue_offload_fraction','ue_drx_configs'):
        e=deepcopy(next(e for e in g['edges']if e['kind']=='influences'and e['source']=='rcp:'+param))
        e.update(id=e['id']+'2',target='kpi:k2');e['sensitivities'][0].update(estimate=-10,confidence_interval=[-10,-10]);g['edges'].append(e)
    c['ranges'].append(dict(kpi='k2',lower=0,upper=10,lower_inclusive=True,upper_inclusive=False,domain=[0,None]));f['states']['kpi:k2']=dict(value=10)
    q=strength(g,f,c,p)['groups'][0]
    assert q['raw_obstruction_strength']==pytest.approx(1)
    assert q['score']==pytest.approx(50)
    assert q['raw_obstruction_strength']==max(m['obstruction_strength']for m in q['goal_components'])
    assert 'contributors'not in q and 'interactions'not in q


def test_v2_no_action_is_zero_without_claiming_goal_satisfaction():
    g,f,c,p,_=fixture()
    for app in p['apps']:app.update(status='no_action',inference_invoked=True,actions=[])
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']==0 and q['raw_obstruction_strength']==0 and q['no_active_push']
