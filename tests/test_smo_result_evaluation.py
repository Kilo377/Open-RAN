"""Intent evaluation axioms and analytic examples; no invented performance."""
from copy import deepcopy
import pytest
from smo.intent_quality import assess, enrich_comparison
from smo.result_evaluation import enrich_batch, evaluate


def target(k, lower=0, upper=None, **kw):
    result=dict(kpi=k,lower=lower,upper=upper,lower_inclusive=True,upper_inclusive=True,domain=[0,None],unit='test-unit')
    result.update(kw)
    return result


def comparison(targets,before,after):
    def read(t,value):return assess(t,dict(states={'kpi:'+t['kpi']:dict(value=value)}))
    return dict(rows=[dict(**t,before=read(t,b),after=read(t,a))for t,b,a in zip(targets,before,after)])


def test_all_unchanged_is_zero_even_when_intent_is_not_met():
    ts=[target('throughput',30),target('power',upper=88.2)]
    r=evaluate(comparison(ts,[30,90],[30,90]))
    assert r['before']['score']==r['after']['score']==r['change']==0
    assert r['admissible'] and r['state']=='unchanged' and not r['after']['all_satisfied']
    assert r['violation']['after']['max']==pytest.approx(2) and r['violation']['change']==0


def test_finite_partial_improvement_is_positive_before_full_attainment():
    ts=[target('throughput',30),target('power',upper=88.2)]
    r=evaluate(comparison(ts,[30,90],[31.5,89.1]))
    assert [m['improvement_score']for m in r['metrics']]==pytest.approx([5,1])
    assert r['before']['score']==0 and r['after']['score']==pytest.approx(3)
    assert r['admissible'] and not r['after']['all_satisfied']
    assert r['violation']['after']['max']==pytest.approx(1)
    assert r['metrics'][1]['target_progress_percent']==pytest.approx(50)


def test_any_regression_cannot_be_compensated_by_another_targets_gain():
    ts=[target('throughput',30),target('power',upper=88.2)]
    r=evaluate(comparison(ts,[30,90],[31.5,90.9]))
    assert r['mean_improvement']==pytest.approx(2) and r['after']['score']==pytest.approx(-1)
    assert r['regression_kpis']==['power'] and not r['admissible']
    assert r['after']['bottleneck_kpis']==['power']
    assert not r['risk_inputs_used'] and not r['graph_gains_used']


def test_one_target_held_does_not_block_other_targets_benefit():
    ts=[target('throughput',30),target('power',upper=88.2)]
    r=evaluate(comparison(ts,[30,90],[30,88.2]))
    assert r['after']['score']==pytest.approx(1) and r['admissible'] and r['after']['all_satisfied']


@pytest.mark.parametrize('after,score,met',[(100,0,False),(99,1,False),(97,3,True),(103,-3,False)])
def test_power_target_distinguishes_gain_and_attainment(after,score,met):
    r=evaluate(comparison([target('power',upper=98)],[100],[after]))
    assert r['after']['score']==pytest.approx(score) and r['after']['all_satisfied']is met


def test_halfline_keeps_directional_gain_inside_target():
    r=evaluate(comparison([target('power',upper=90)],[90],[10]))
    assert r['after']['score']==pytest.approx(100*80/90)
    assert r['metrics'][0]['active_boundaries']==[('upper',90)]
    assert r['metrics'][0]['mode']=='directional_change'
    r=evaluate(comparison([target('throughput',30,lower_inclusive=False)],[30],[30]))
    assert r['after']['score']==0 and not r['after']['all_satisfied']
    assert r['metrics'][0]['after']['strict_boundary_violation']


@pytest.mark.parametrize('after,score,violation',[(20,0,25),(16,20,5),(15,25,0),(10,25,0),(5,25,0),(0,0,25)])
def test_bounded_interval_does_not_reward_its_center(after,score,violation):
    r=evaluate(comparison([target('latency',5,15)],[20],[after]))
    assert r['before']['score']==0 and r['after']['score']==pytest.approx(score)
    assert r['violation']['after']['max']==pytest.approx(violation)
    assert r['metrics'][0]['mode']=='interval_distance_reduction'


def test_maintained_band_neutral_but_does_not_block_power_benefit():
    ts=[target('throughput',20,40),target('power',upper=98)]
    r=evaluate(comparison(ts,[30,100],[25,98]))
    assert r['after']['score']==1 and r['admissible']
    r=evaluate(comparison(ts,[30,100],[19,90]))
    assert r['after']['score']==pytest.approx(-100/30) and r['regression_kpis']==['throughput']


def test_point_target_uses_absolute_distance():
    t=target('point',10,10)
    assert evaluate(comparison([t],[10],[10]))['after']['score']==0
    assert evaluate(comparison([t],[10],[11]))['after']['score']==-10
    assert evaluate(comparison([t],[20],[15]))['after']['score']==25


def test_open_endpoint_regression_is_flagged_without_invented_epsilon():
    r=evaluate(comparison([target('band',5,15,lower_inclusive=False)],[10],[5]))
    assert r['after']['score']==0
    assert r['metrics'][0]['strict_boundary_regression']
    assert not r['admissible'] and not r['after']['all_satisfied']
    assert r['violation']['after']['max']==0


@pytest.mark.parametrize('before,after,score',[(0,0,0),(0,.07,-7),(.07,.05,2)])
def test_urlcc_ratio_uses_percentage_points_including_zero(before,after,score):
    t=target('urllc_failure_fraction',upper=0,domain=[0,1],unit='ratio')
    r=evaluate(comparison([t],[before],[after]))
    assert r['after']['score']==pytest.approx(score)
    assert r['metrics'][0]['score_unit']=='percentage_points' and r['metrics'][0]['scale']==1
    assert r['violation']['after']['max']==pytest.approx(100*after)
    assert r['admissible']is (score>=0)


def test_ratio_unit_and_physical_domain_are_checked_without_clamping():
    t=target('urllc_failure_fraction',upper=0,domain=[0,1],unit='ratio')
    assert evaluate(comparison([t],[0],[1.1]))['after']['score']is None
    t.update(domain=[0,100],unit='%')
    assert evaluate(comparison([t],[7],[5]))['after']['score']==2


def test_other_zero_baseline_requires_scale_never_uses_after():
    c=comparison([target('failures',upper=1)],[0],[.5]);r=evaluate(c)
    assert r['after']['score']is None and r['metrics'][0]['after']['reason']=='explicit_scale_required'
    r=evaluate(c,reference_scales={'failures':1})
    assert r['before']['score']==0 and r['after']['score']==-50
    c['rows'][0]['after']['value']=1000
    assert evaluate(c)['after']['score']is None
    c['rows'][0]['kpi']='urllc_failure_events_per_s'
    assert evaluate(c)['after']['score']is None # events/s are never silently converted into ratios


def test_missing_or_invalid_target_cannot_be_dropped():
    ts=[target('throughput',30),target('power',upper=88.2)]
    c=comparison(ts,[30,90],[31,87]);c['rows'][1]['after']['status']='unavailable'
    assert evaluate(c)['after']['score']is None
    c['rows'].pop();assert evaluate(c,ranges=ts)['after']['score']is None
    assert evaluate(comparison([target('power',upper=90)],[90],[-1]))['after']['score']is None
    assert evaluate(comparison([target('p',10,5,empty=True)],[7],[8]))['before']['score']is None
    c=comparison([target('power',upper=90)],[90],[80]);c['rows'][0]['after']['measurement']={'reason':'no_mature_task_samples'}
    assert evaluate(c)['after']['score']is None


def test_extrema_contract_applies_to_gain_and_violation():
    c=comparison([target('load',upper=2)],[2.5],[1.5])
    c['rows'][0]['after'].update(status='above_range',measurement=dict(constraint_uses_extrema=True,minimum=1,maximum=3))
    r=evaluate(c)
    assert r['after']['score']==-20 and r['violation']['after']['max']==40
    c['rows'][0]['after']['measurement'].pop('maximum')
    assert evaluate(c)['after']['score']is None


def test_batch_scales_frozen_order_independent_and_copied():
    ts=[target('throughput',30),target('power',upper=88.2)]
    a=comparison(ts,[30,90],[30.6,87.3]);b=comparison(ts,[30,90],[31.5,89.1])
    batch=dict(ranges=ts,baseline_reference={'throughput':30,'power':90},members=[dict(group_id='a',comparison=a),dict(group_id='b',comparison=b)])
    original=deepcopy(batch);r=enrich_batch(batch);scores=[m['comparison']['result_evaluation']for m in r['members']]
    assert [m['scale']for m in scores[1]['metrics']]==[30,90]
    assert scores[0]['after']['score']==pytest.approx(2.5)
    batch['members'].reverse();assert enrich_batch(batch)['members'][0]['comparison']['result_evaluation']==scores[1]
    batch['members'].reverse();assert batch==original
    batch.pop('baseline_reference')
    assert all(m['comparison']['result_evaluation']['after']['score']is None for m in enrich_batch(batch)['members'])


def test_nonintent_kpis_graph_and_conflict_scores_never_enter_evaluation():
    t=target('throughput',30);c=comparison([t],[30],[31]);original=deepcopy(c)
    c.update(conflict_score=100,graph_gains={'throughput':-999})
    c['rows'].append(dict(kpi='power',before={'value':0},after={'value':1e9}))
    assert evaluate(c,ranges=[t])['after']['score']==evaluate(original)['after']['score']
    c['rows'].pop();r=enrich_comparison(c)
    assert 'result_evaluation'in r and 'result_evaluation'not in c and c['rows']==original['rows']


def test_duplicate_targets_are_rejected():
    t=target('p',upper=1)
    with pytest.raises(ValueError,match='unique'):evaluate(comparison([t,t],[1,1],[.5,.5]))


def test_near_zero_scale_does_not_emit_nonfinite_json_numbers():
    import json
    r=evaluate(comparison([target('power',upper=1)],[1e-300],[1e300]))
    assert r['after']['score']is None
    assert r['metrics'][0]['after']['reason']=='nonfinite_normalized_violation'
    json.dumps(r,allow_nan=False)


def test_full_physical_domain_is_a_neutral_valid_interval():
    t=target('urllc_failure_fraction',upper=1,domain=[0,1],unit='ratio')
    r=evaluate(comparison([t],[0],[.7]))
    assert r['after']['score']==0 and r['before']['score']==0
    assert r['admissible'] and r['after']['all_satisfied']
    assert r['metrics'][0]['active_boundaries']==[]
    assert r['violation']['after']['max']==0
    # Physical impossibilities are still invalid despite neutral preferences.
    assert evaluate(comparison([t],[0],[1.1]))['after']['score']is None


def test_overflowing_violation_cannot_be_presented_as_a_complete_zero_change():
    r=evaluate(comparison([target('p',upper=1)],[1e300],[1e300]),reference_scales={'p':1e-300})
    assert r['after']['score']is None and r['before']['score']is None
    assert r['admissible']is None and not r['violation']['after']['complete']
