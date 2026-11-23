"""Forward-only aggregation reference geometry; all outcomes are synthetic."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from smo.influence_units import align_graph
from smo.intent_quality import assess
from smo.result_evaluation import evaluate

ROOT=Path(__file__).resolve().parents[1]
METHOD='aggregation_aware_obstruction_v1'


def run(graph,frame,catalog,preview,*,projected=None,**options):
    if not shutil.which('node'):pytest.skip('Node required for production aggregation-aware method')
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
for(const f of ['app.js','goal_obstruction.js','observation_normalization.js','aggregation_aware_obstruction.js'])
 vm.runInContext(fs.readFileSync('smo/web/'+f,'utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(d);
const options={...d.options};if(d.projected)options.projectionEngine={projectOutcomes:()=>structuredClone(d.projected)};
const result=c.SmoAggregationAwareObstruction.review(d.graph,d.frame,d.catalog.groups,d.catalog.ranges,d.preview,options);
if(before!==JSON.stringify(d))throw Error('Frozen inputs mutated');
process.stdout.write(JSON.stringify(result));"""
    return json.loads(subprocess.check_output(['node','-e',script],cwd=ROOT,text=True,input=json.dumps(dict(
        graph=align_graph(graph),frame=frame,catalog=catalog,preview=preview,projected=projected,options=options))))


def analytic(*,current=1.,low=0.,high=1.,domain=None,extrema=None,nested=False,
             drives=(0.,1.),probabilities=(.5,.5),kpi='k',unit='W'):
    row=dict(kpi=kpi,unit=unit,current=999,anchors=[dict(current=999)],lower=low,upper=high,
        lower_inclusive=True,upper_inclusive=True,domain=domain if domain is not None else[0,None])
    state=dict(value=current)
    if extrema is not None:
        metadata=dict(constraint_uses_extrema=True,minimum=extrema[0],maximum=extrema[1])
        if nested:state['measurement']=metadata
        else:state.update(metadata)
    frame=dict(slot=100,time_ms=50,states={'kpi:'+kpi:state})
    catalog=dict(groups=[dict(id='g',app_ids=['a','b'])],ranges=[row])
    projected=dict(status='ready',groups=[dict(group_id='g',model_coverage_complete=True,
        actual_outcomes=[dict(probability=p,goals=[dict(kpi=kpi,reference_drive=d,chains=[])])
                         for p,d in zip(probabilities,drives)])])
    return dict(graph={},frame=frame,catalog=catalog,preview={'slot':100},projected=projected)


def one(**kwargs):
    result=run(**analytic(**kwargs))
    return result,result['groups'][0],result['groups'][0]['goal_components'][0]


def test_mean_and_extrema_follow_the_declared_observation_operator():
    mean,q,m=one();extrema,r,e=one(extrema=(1,1))
    assert q['raw_obstruction_strength']==pytest.approx(.5)
    assert q['score']==pytest.approx(100*.5/1.5)
    assert m['reference_interval']==[1.5,1.5]
    assert r['raw_obstruction_strength']==1 and r['score']==50
    assert e['reference_interval']==[1,2]
    assert e['aggregation']=='baseline_range_by_joint_action_support_envelope'
    assert extrema['finite_window_endpoint_occurrence_guaranteed']is False
    assert extrema['intent_failure_probability_computed']is False
    assert mean['history_ensemble_used']is False and mean['horizon_scaling_performed']is False


@pytest.mark.parametrize('low,high',[(0,1),(.9,1.1),(1,1),(0,None)])
@pytest.mark.parametrize('nested',[False,True])
def test_noop_preserves_an_existing_variable_window_exactly(low,high,nested):
    result,q,c=one(low=low,high=high,extrema=(.6,1.7),nested=nested,drives=(0,0),probabilities=(.3,.7))
    assert c['baseline_reference_interval']==c['reference_interval']==[.6,1.7]
    assert c['reference_progress']==0 and q['reference_effect']==0 and q['score']==0
    assert c['no_active_push'] and q['no_active_push']


def test_mean_noop_is_exact_despite_probability_summation():
    _,q,c=one(current=1.7,drives=(0,0,0),probabilities=(.1,.2,.7))
    assert c['reference_interval']==[1.7,1.7] and c['reference_progress']==0 and q['score']==0


def test_per_joint_physical_projection_precedes_probability_averaging():
    _,q,c=one(current=0,kpi='urllc_failure_fraction',unit='ratio',domain=[0,1],
        high=0,drives=(-.01,.01))
    assert c['reference_interval']==pytest.approx([.005,.005])
    assert q['raw_obstruction_strength']==pytest.approx(.005)
    assert c['physically_clipped_joint_outcomes']==1
    _,q,c=one(current=0,kpi='urllc_failure_fraction',unit='ratio',domain=[0,1],
        high=0,extrema=(0,0),drives=(-.01,.01))
    assert c['reference_interval']==[0,.01] and q['raw_obstruction_strength']==.01


def test_reference_is_whole_joint_net_influence_not_per_controller_clipping():
    data=analytic(current=0,kpi='urllc_failure_fraction',unit='ratio',domain=[0,1],high=0,
        drives=(0,),probabilities=(1,))
    data['projected']['groups'][0]['actual_outcomes'][0]['goals'][0]['chains']=[
        dict(drive=-10,action_coverage={'complete':True}),dict(drive=10,action_coverage={'complete':True})]
    q=run(**data)['groups'][0]
    assert q['score']==0 and q['goal_components'][0]['physically_clipped_joint_outcomes']==0


def test_positive_support_ignores_zero_probability_unknown_or_extreme_branch():
    _,q,c=one(extrema=(1,1),drives=(0,None),probabilities=(1,0))
    assert c['reference_interval']==[1,1] and c['positive_probability_outcomes']==1 and q['score']==0


def test_positive_probability_weights_matter_for_mean_but_not_support_endpoints():
    _,q,c=one(drives=(0,1),probabilities=(.9,.1))
    assert c['reference_interval']==[1.1,1.1] and q['raw_obstruction_strength']==pytest.approx(.1)
    _,q,c=one(extrema=(1,1),drives=(0,1),probabilities=(.9,.1))
    assert c['reference_interval']==[1,2] and q['raw_obstruction_strength']==1


def test_band_envelope_can_fail_when_the_mean_is_unchanged():
    _,q,c=one(low=.9,high=1.1,extrema=(1,1),drives=(-.2,.2))
    assert c['reference_interval']==[.8,1.2]
    assert q['raw_obstruction_strength']==pytest.approx(.1)
    assert c['mode']=='interval_distance_reduction'


@pytest.mark.parametrize('low,high,delta,progress',[(2,3,.5,.5),(2,3,1.5,1),(2,3,3.5,-.5),(1,1,.1,-.1)])
def test_band_and_point_use_reduction_in_distance(low,high,delta,progress):
    _,q,c=one(low=low,high=high,drives=(delta,),probabilities=(1,))
    assert c['reference_progress']==pytest.approx(progress)
    assert q['raw_obstruction_strength']==pytest.approx(max(0,-progress))


def test_single_boundary_rewards_direction_not_only_attainment():
    _,q,c=one(high=2,drives=(.5,),probabilities=(1,))
    assert c['before_reference_margin']==1 and c['after_reference_margin']==.5
    assert c['reference_progress']==-.5 and q['raw_obstruction_strength']==.5


def test_full_physical_domain_is_neutral_under_nonzero_reference_push():
    _,q,c=one(high=None,drives=(1,),probabilities=(1,))
    assert c['reference_progress']==0 and q['score']==0 and c['mode']=='neutral_physical_domain'


def test_current_coordinates_override_any_old_anchor_normalization_option():
    data=analytic(current=100,low=100,high=None,drives=(-1,),probabilities=(1,))
    data['catalog']['ranges'][0]['anchors']=[dict(current=10)]
    r=run(**data,normalization={'kpis':[dict(kpi='k',scale=10)]})
    assert r['normalization']['policy']=='current_observation_v1'
    assert r['normalization']['kpis'][0]['scale']==100
    assert r['groups'][0]['raw_obstruction_strength']==.01


@pytest.mark.parametrize('mutation,reason',[
    ('missing_min','observed_window_extrema_unavailable'),('missing_max','observed_window_extrema_unavailable'),
    ('inverted','observed_window_extrema_unavailable'),('conflict','contradictory_observed_window_extrema'),
    ('bad_flag','invalid_observed_aggregation_flag'),('outside','observed_reference_outside_physical_domain'),
    ('missing_current','kpi_observation_unavailable')])
def test_missing_or_invalid_observed_extrema_never_become_invented_values(mutation,reason):
    data=analytic(extrema=(.5,1.5));state=data['frame']['states']['kpi:k']
    if mutation=='missing_min':state.pop('minimum')
    elif mutation=='missing_max':state['maximum']=None
    elif mutation=='inverted':state.update(minimum=2,maximum=1)
    elif mutation=='conflict':state['measurement']=dict(constraint_uses_extrema=True,minimum=.4,maximum=1.5)
    elif mutation=='bad_flag':state['constraint_uses_extrema']='true'
    elif mutation=='outside':state['minimum']=-1
    else:state['value']=None
    r=run(**data)
    assert r['status']=='unavailable' and r['reason']==reason and not r['groups']


def test_nonratio_current_zero_is_unavailable_instead_of_anchor_fallback():
    r=run(**analytic(current=0))
    assert r['status']=='unavailable' and r['reason']=='kpi_scale_unavailable'


@pytest.mark.parametrize('mutation', ['unknown_drive','entity_coverage','model_coverage','unknown_link','excluded_context'])
def test_unknown_action_or_entity_projection_is_not_zero_imputed(mutation):
    data=analytic();record=data['projected']['groups'][0];goal=record['actual_outcomes'][0]['goals'][0]
    if mutation=='unknown_drive':goal['reference_drive']=None
    elif mutation=='entity_coverage':goal['chains']=[dict(drive=0,action_coverage={'complete':False})]
    elif mutation=='model_coverage':record['model_coverage_complete']=False
    elif mutation=='unknown_link':record['projection_coverage']={'unknown_links':[{'parameter':'p'}]}
    else:goal['excluded_action_contexts']=[{'parameter':'p'}]
    q=run(**data)['groups'][0]
    assert q['score']is None and q['score_status']=='insufficient_evidence' and q['reason']


@pytest.mark.parametrize('probabilities',[(0,0),(-1,2)])
def test_invalid_probability_mass_is_unavailable(probabilities):
    q=run(**analytic(probabilities=probabilities))['groups'][0]
    assert q['score']is None and q['reason']=='invalid_arbitration_probability_mass'


def test_uncertain_measured_coefficients_remain_reference_approximations():
    data=analytic();goal=data['projected']['groups'][0]['actual_outcomes'][1]['goals'][0]
    goal['chains']=[dict(drive=1,notes=['uncertain_gain_direction'])]
    q=run(**data)['groups'][0]
    assert q['score']is not None and q['score_status']=='reference_approximation'


def test_multikpi_negative_progress_has_a_noncompensable_veto():
    data=analytic(drives=(-.5,),probabilities=(1,));r2=deepcopy(data['catalog']['ranges'][0]);r2['kpi']='k2'
    data['catalog']['ranges'].append(r2);data['frame']['states']['kpi:k2']={'value':1}
    data['projected']['groups'][0]['actual_outcomes'][0]['goals'].append(dict(kpi='k2',reference_drive=.1,chains=[]))
    q=run(**data)['groups'][0]
    assert q['reference_effect']==pytest.approx(-.1) and q['raw_obstruction_strength']==pytest.approx(.1)
    data['projected']['groups'][0]['actual_outcomes'][0]['goals'][1]['reference_drive']=-.3
    assert run(**data)['groups'][0]['reference_effect']==pytest.approx(.4)


def test_units_preserve_the_same_normalized_envelope_geometry():
    for kpi,units,factors in [('k',['W','mW'],[1,1000]),('urllc_failure_fraction',['ratio','%'],[1,100])]:
        results=[]
        for unit,factor in zip(units,factors):
            domain=[0,1*factor]if kpi=='urllc_failure_fraction'else[0,None]
            data=analytic(kpi=kpi,unit=unit,current=.5*factor,high=.5*factor,domain=domain,
                extrema=(.4*factor,.6*factor),drives=(0,.1*factor))
            results.append(run(**data)['groups'][0])
        assert results[0]['reference_effect']==pytest.approx(results[1]['reference_effect'])
        assert results[0]['score']==pytest.approx(results[1]['score'])


def test_observed_mean_score_is_separate_from_the_static_reference_envelope():
    data=analytic(extrema=(.7,1.4),drives=(-.1,.2));r=run(**data);q=r['groups'][0];c=q['goal_components'][0]
    target=data['catalog']['ranges'][0];after=dict(states={'kpi:k':dict(value=1.1,minimum=c['reference_interval'][0],
        maximum=c['reference_interval'][1],constraint_uses_extrema=True)})
    comparison=dict(rows=[dict(**target,before=assess(target,data['frame']),after=assess(target,after))])
    observed=evaluate(comparison,ranges=[target])
    # The frozen static method still measures its conservative range. The
    # observed result now scores the recorded mean and checks peaks separately.
    assert q['reference_effect_score']==pytest.approx(-20)
    assert observed['after']['score']==pytest.approx(-10)
    assert observed['violation']['after']['max']==pytest.approx(60)
    assert not observed['after']['all_satisfied']


def test_future_history_and_after_named_options_are_not_inputs_to_static_reference():
    data=analytic(extrema=(1,1));plain=run(**data)
    other=run(**data,history=[dict(frame={'slot':999999},after={'value':999})],after_kpis={'k':999},report_slots=999999)
    assert plain==other


def test_real_projector_uses_complete_joint_actions_and_preserves_missing_proposals():
    from test_smo_goal_obstruction import fixture
    g,f,c,p,_=fixture();q=run(g,f,c,p)['groups'][0]
    assert q['score']==0 and q['reference_effect']==pytest.approx(.1)
    assert q['reference_subset_count']==0 and q['model_coverage_complete']
    p['apps'][0].update(status='unavailable',inference_invoked=False)
    assert run(g,f,c,p)['groups'][0]['score']is None


def test_real_empty_scheduler_bundle_recovery_is_reused_with_its_provenance():
    from test_smo_empty_scheduler_bundle import empty_bundle_fixture
    g,f,c,p=empty_bundle_fixture();q=run(g,f,c,p)['groups'][0]
    assert q['reason']is None and len(q['recovered_empty_scheduler_bundles'])==1
    assert q['recovered_empty_scheduler_bundles'][0]['flat_ue_actions_added']==0
    p['apps'][0]['decisions'][0]['slot_id']-=1
    q=run(g,f,c,p)['groups'][0]
    assert q['score']is None and q['reason']=='incomplete_scheduler_bundle'


def test_real_joint_sleep_profile_gain_is_used_once_without_constituent_addition():
    from test_smo_goal_obstruction import fixture
    from test_smo_joint_cell_sleep_profile import edge,BASE,MIXED
    g,f,c,p,_=fixture();g['edges'].append(edge())
    f['states']['rcp:cell_sleep_mode']['by_entity']=deepcopy(BASE)
    c['groups']=[dict(id='joint',app_ids=['a'])]
    p['apps'][0]['actions']=[dict(parameter='cell_sleep_mode',cell_id=cid,ue_id=None,current='active',proposed=mode)
        for cid,mode in MIXED.items()]
    q=run(g,f,c,p)['groups'][0]
    assert q['reference_effect']==pytest.approx(-.45)
    assert q['raw_obstruction_strength']==pytest.approx(.45)


def test_fresh_matching_preview_and_no_deployment_are_required():
    data=analytic();data['preview']['slot']=99
    assert run(**data)['reason']=='matching_decision_preview_unavailable'
    data['preview'].update(slot=100,deployment_enabled=True)
    assert run(**data)['reason']=='invalid_shadow_preview'
