"""Analytic preregistered reference effects, never observed deployment labels."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from smo.influence_units import align_graph,normalization
from test_smo_goal_obstruction import fixture

ROOT=Path(__file__).resolve().parents[1]


def review(g,f,c,p,**options):
    node=shutil.which('node')
    if not node:pytest.skip('Node required for the production reference-obstruction module')
    if options.get('history') is not None:
        options.setdefault('method','temporal_net_intent_obstruction_v1')
    options.setdefault('normalization',normalization(c['ranges'],f))
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
for(const f of ['smo/web/app.js','smo/web/goal_obstruction.js','smo/web/intent_obstruction.js'])
 vm.runInContext(fs.readFileSync(f,'utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(d);
const result=c.SmoIntentObstruction.review(d.graph,d.frame,d.catalog.groups,d.catalog.ranges,d.preview,d.options);
if(JSON.stringify(d)!==before)throw Error('Read-only research inputs mutated');
process.stdout.write(JSON.stringify(result));"""
    return json.loads(subprocess.check_output([node,'-e',script],cwd=ROOT,text=True,input=json.dumps(dict(
        graph=align_graph(g),frame=f,catalog=c,preview=p,options=options))))


def one_app():
    g,f,c,p,a=fixture()
    c['groups']=[dict(id='a',app_ids=['a'])]
    return g,f,c,p,a


def test_positive_but_foregone_reference_capability_is_not_regression():
    g,f,c,p,_=fixture();r=review(g,f,c,p);q=r['groups'][0]
    assert q['score']==0 and q['reference_effect']==pytest.approx(.1)
    assert q['reference_effect_score']==pytest.approx(10)
    assert q['reference_subset_count']==0
    assert not r['observed_after_kpis_used'] and not r['kpi_forecast_performed']
    assert not r['selection_performed'] and not r['deployment_enabled']


def test_reference_effect_is_invariant_to_kpi_and_parameter_coordinate_units():
    g,f,c,p,_=fixture();a=review(g,f,c,p)['groups'][0]
    g['nodes'][0]['unit']='mW';f['states']['kpi:k']['value']*=1000
    c['ranges'][0]['lower']*=1000
    gain=g['edges'][0]['sensitivities'][0]
    gain.update(unit='W / %',x_unit='%',estimate=.1,confidence_interval=[.1,.1],fit_domain=[0,100])
    b=review(g,f,c,p)['groups'][0]
    assert b['reference_effect']==pytest.approx(a['reference_effect'])
    assert b['score']==pytest.approx(a['score'])


def test_worst_negative_progress_cannot_be_compensated_by_another_kpi():
    g,f,c,p,_=fixture();g['nodes'].append(dict(id='kpi:k2',kind='kpi',unit='W'))
    f['states']['kpi:k2']=dict(value=10)
    c['ranges'].append(dict(kpi='k2',lower=10,upper=None,lower_inclusive=False))
    for original in list(g['edges']):
        if original['kind']!='influences':continue
        edge=deepcopy(original);edge.update(id=edge['id']+'2',target='kpi:k2')
        for row in edge['sensitivities']:
            row['estimate']*=-1;row['confidence_interval']=[-row['confidence_interval'][1],-row['confidence_interval'][0]]
        g['edges'].append(edge)
    q=review(g,f,c,p)['groups'][0]
    assert q['reference_effect']==pytest.approx(-.1)
    assert q['score']==pytest.approx(100*.1/1.1)


def test_random_arbitration_vectors_are_averaged_before_the_risk_hinge():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    q=review(g,f,c,p)['groups'][0]
    assert q['arbitration_outcome_count']==2
    assert q['reference_effect']==pytest.approx(0,abs=1e-14)
    assert q['score']==pytest.approx(0,abs=1e-12)


@pytest.mark.parametrize('low,high,gain,expected',[(5,15,10,0),(12,14,10,.1),(12,14,20,-.4),(10,10,10,-.5)])
def test_bands_and_points_use_distance_to_the_requested_set(low,high,gain,expected):
    g,f,c,p,a=one_app();c['ranges'][0].update(lower=low,upper=high,lower_inclusive=True)
    g['edges'][0]['sensitivities'][0].update(estimate=gain,confidence_interval=[gain,gain])
    p['apps'][0]['actions']=[a('ue_offload_fraction',1)]
    q=review(g,f,c,p)['groups'][0]
    assert q['reference_effect']==pytest.approx(expected)
    assert q['score']==pytest.approx(100*max(0,-expected)/(1+max(0,-expected)))


def test_physical_domain_projection_applies_after_the_joint_mean():
    g,f,c,p,a=one_app();g['edges'][0]['sensitivities'][0].update(estimate=100,confidence_interval=[100,100])
    p['apps'][0]['actions']=[a('ue_offload_fraction',0)]
    q=review(g,f,c,p)['groups'][0];goal=q['goal_components'][0]
    assert goal['expected_normalized_reference_drive']==pytest.approx(-5)
    assert goal['physical_projected_reference_drive']==-1 and goal['physical_saturation_applied']
    assert q['reference_effect']==-1 and q['score']==50


def ratio_fixture(proposed):
    g,f,c,p,a=one_app();g['nodes'][0].update(id='kpi:urllc_failure_fraction',unit='ratio')
    f['states'].pop('kpi:k');f['states']['kpi:urllc_failure_fraction']=dict(value=0)
    for edge in g['edges']:
        if edge['kind']!='influences':continue
        edge['target']='kpi:urllc_failure_fraction'
        for row in edge['sensitivities']:
            denominator='ms'if row['parameter']=='ue_drx_configs'else'fraction'
            row.update(unit='ratio / '+denominator,estimate=.1,confidence_interval=[.1,.1])
    c['ranges']=[dict(kpi='urllc_failure_fraction',unit='ratio',lower=0,upper=.01,
        domain=[0,1],lower_inclusive=True,upper_inclusive=True,current=0)]
    p['apps'][0]['actions']=[a('ue_offload_fraction',proposed)]
    return g,f,c,p


def test_zero_failure_baseline_uses_percentage_point_scale_without_negative_failure_credit():
    q=review(*ratio_fixture(.7))['groups'][0]
    assert q['reference_effect']==pytest.approx(-.02)
    assert q['score']==pytest.approx(100*.02/1.02)
    q=review(*ratio_fixture(.3))['groups'][0]
    assert q['reference_effect']==0 and q['score']==0
    assert q['goal_components'][0]['physical_saturation_applied']


def test_off_control_is_a_real_no_action_zero_not_missing_coverage():
    g,f,c,p,_=fixture();c['groups']=[dict(id='off',app_ids=[])]
    q=review(g,f,c,p)['groups'][0]
    assert q['score']==0 and q['reference_effect']==0 and q['no_active_push']
    assert q['model_coverage_complete']


def histories(frame,preview):
    rows=[]
    for slot in (50,100):
        f=deepcopy(frame);f.update(slot=slot,time_ms=slot/2)
        p=deepcopy(preview);p['slot']=slot
        rows.append(dict(frame=f,preview=p,source_identity='same-run-model-scene',graph_context_sha256='same-frozen-kg'))
    return rows


def temporal_options(history):
    return dict(history=history,sourceIdentity='same-run-model-scene',graphContextSha256='same-frozen-kg')


def test_temporal_averages_joint_vectors_before_physical_projection_and_hinge():
    g,f,c,p,a=one_app();history=histories(f,p)
    history[0]['preview']['apps'][0]['actions']=[a('ue_offload_fraction',.3)]
    r=review(g,f,c,p,**temporal_options(history));q=r['groups'][0]
    assert r['history_slots']==[50,100] and r['history_sample_count']==2
    assert q['reference_effect']==pytest.approx(0,abs=1e-14)
    assert q['score']==pytest.approx(0,abs=1e-12)
    assert r['history_lookback_slots']==500 and not r['horizon_scaling_performed']


@pytest.mark.parametrize('old_value',[None,0.,1000.])
def test_temporal_uses_frozen_final_kpi_geometry_not_historical_kpi_readings(old_value):
    g,f,c,p,_=fixture();history=histories(f,p)
    baseline=review(g,f,c,p,**temporal_options(history))
    if old_value is None:history[0]['frame']['states'].pop('kpi:k')
    else:history[0]['frame']['states']['kpi:k']['value']=old_value
    actual=review(g,f,c,p,**temporal_options(history))
    assert actual['status']=='ready'
    assert actual['intent_contexts'][0]['current']==10
    assert actual['groups'][0]['reference_effect']==baseline['groups'][0]['reference_effect']
    assert actual['groups'][0]['score']==baseline['groups'][0]['score']


def test_physical_saturation_is_after_random_joint_vector_averaging():
    g,f,c,p,a=fixture();g['edges'][0]['sensitivities'][0].update(estimate=100,confidence_interval=[100,100])
    p['apps'][1]['actions']=[a('ue_offload_fraction',.3)]
    q=review(g,f,c,p)['groups'][0]
    # The two requests have normalized vectors +2 and -2. Clamping each
    # branch first would create a fictitious +0.5 credit at the lower bound.
    assert q['reference_effect']==pytest.approx(0,abs=1e-14)
    assert q['score']==pytest.approx(0,abs=1e-12)


def test_single_temporal_sample_is_explicit_and_matches_primary():
    g,f,c,p,_=fixture();h=histories(f,p)[-1:]
    a=review(g,f,c,p);b=review(g,f,c,p,**temporal_options(h))
    assert b['history_sample_status']=='single_sample'
    assert b['groups'][0]['reference_effect']==a['groups'][0]['reference_effect']
    assert b['groups'][0]['score']==a['groups'][0]['score']


@pytest.mark.parametrize('mutation,reason',[
    ('future','history_future_sample'),('noncontemporaneous','history_snapshot_mismatch'),
    ('source','history_context_mismatch'),('graph','history_context_mismatch'),
    ('duplicate','history_duplicate_snapshot'),('after','history_untrusted_fields'),
    ('old','history_outside_lookback'),('model','history_model_mismatch')])
def test_temporal_rejects_mixed_future_unverified_or_after_label_inputs(mutation,reason):
    g,f,c,p,_=fixture();history=histories(f,p)
    if mutation=='future':history[-1]['frame'].update(slot=101,time_ms=50.5);history[-1]['preview']['slot']=101
    elif mutation=='noncontemporaneous':history[0]['preview']['slot']=49
    elif mutation=='source':history[0]['source_identity']='other-run'
    elif mutation=='graph':history[0]['graph_context_sha256']='other-kg'
    elif mutation=='duplicate':history.append(deepcopy(history[0]))
    elif mutation=='after':history[0]['actual_after_kpis']={'k':-999}
    elif mutation=='old':history[0]['frame'].update(slot=-400,time_ms=-200);history[0]['preview']['slot']=-400
    elif mutation=='model':p['apps'][0]['model_sha256']='current-model';history[0]['preview']['apps'][0]['model_sha256']='other-model'
    r=review(g,f,c,p,**temporal_options(history))
    assert r['status']=='unavailable' and r['reason']==reason


def test_unknown_gain_cannot_be_converted_to_zero_obstruction():
    g,f,c,p,_=fixture();g['edges']=[e for e in g['edges']if e['kind']!='influences']
    q=review(g,f,c,p)['groups'][0]
    assert q['score']is None and q['reason']=='reference_projection_incomplete'


def test_future_result_fields_do_not_enter_the_reference_effect():
    g,f,c,p,_=fixture();baseline=review(g,f,c,p)
    f['states']['future_actual_kpi']={'value':-999999}
    actual=review(g,f,c,p,post_deployment_results={'after':{'k':999999}})
    assert actual['groups'][0]['reference_effect']==baseline['groups'][0]['reference_effect']
    assert actual['groups'][0]['score']==baseline['groups'][0]['score']
