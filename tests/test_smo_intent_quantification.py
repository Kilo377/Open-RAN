"""Analytic intent/measurement examples; no claimed xApp performance."""
from copy import deepcopy
import pytest
from smo.intent_quality import assess, quantify, enrich_comparison


def evaluate(b, y, low=0., high=None, *, strict=None, current=None, requested=None, domain=None):
    row=dict(kpi='example',lower=low,upper=high,lower_inclusive=strict!='lower',
        upper_inclusive=strict!='upper',current=b if current is None else current,
        domain=domain or [0.,None])
    if requested is not None:row['requested']=requested
    reading=lambda v:assess(row,dict(states={'kpi:example':dict(value=v)}))
    a,z=reading(b),reading(y)
    return quantify(row,a,z),row,a,z


@pytest.mark.parametrize('b,y,low,high,strict,gain',[
    (20,22,20,None,'lower',10),(10,9.5,0,10,'upper',5),
    (20,18,20,None,'lower',-10),(10,10,0,10,'upper',0)])
def test_pure_direction_uses_actual_before_after_no_fabricated_completion(b,y,low,high,strict,gain):
    q,_,_,_=evaluate(b,y,low,high,strict=strict)
    assert q['mode']=='directional_change'
    assert q['favorable_change_percent']==pytest.approx(gain)
    assert q['target_progress_percent'] is None
    assert q['satisfied']==(gain>0)
    assert q['statistical_significance']=='not_estimated'


@pytest.mark.parametrize('y,progress,passed',[(9.9,50,False),(9.8,100,True),(9.7,150,True),(10.1,-50,False)])
def test_percent_reduction_target_keeps_uncapped_progress(y,progress,passed):
    q,_,_,_=evaluate(10,y,high=9.8,requested=[dict(lower='minimum',upper={'current_multiplier':.98},upper_inclusive=True)])
    assert q['target_progress_percent']==pytest.approx(progress)
    assert q['satisfied']==passed


def test_actual_baseline_changes_do_not_rebase_original_intent():
    req=[dict(lower='current',upper='maximum',lower_inclusive=False)]
    q,_,_,_=evaluate(22,21,20,None,strict='lower',current=20,requested=req)
    assert q['mode']=='directional_change'
    assert q['favorable_change_percent']==pytest.approx(-100/22)
    assert q['satisfied'] # original frozen target is >20, despite deterioration
    assert q['after_margin']==1


def test_bounded_interval_overshoot_can_have_high_progress_but_still_fail():
    q,_,_,_=evaluate(10,22,15,20)
    assert q['target_progress_percent']==240
    assert not q['satisfied'] and q['after_margin']==-2
    q,_,_,_=evaluate(17,18,15,20)
    assert q['mode']=='maintain_range' and q['target_progress_percent'] is None


def test_zero_baseline_and_strict_endpoint_do_not_divide_by_zero_or_pass_equality():
    q,_,_,_=evaluate(0,2,0,None,strict='lower')
    assert q['mode']=='directional_change' and q['favorable_delta']==2
    assert q['favorable_change_percent'] is None and q['satisfied']
    q,_,_,_=evaluate(10,10,0,10,strict='upper')
    assert q['after_margin']==0 and not q['satisfied']


def test_physical_maximum_does_not_turn_directional_ratio_into_bounded_goal():
    req=[dict(lower='current',upper='maximum',lower_inclusive=False)]
    q,_,_,_=evaluate(.5,.6,.5,1,strict='lower',domain=[0,1],requested=req)
    assert q['mode']=='directional_change'
    assert q['favorable_change_percent']==pytest.approx(20)


def test_missing_data_and_archive_enrichment_preserve_original_readings():
    q,row,before,after=evaluate(20,None,20,None,strict='lower')
    assert not q['complete'] and q['target_progress_percent'] is None
    assert q['satisfied'] is None
    archive=dict(rows=[dict(**row,before=before,after=after)])
    original=deepcopy(archive)
    enriched=enrich_comparison(archive)
    assert archive==original and enriched['quantification_version']==1
    assert not enriched['rows'][0]['quantification']['complete']


def test_population_extrema_control_interval_margin():
    q,row,before,after=evaluate(2,1.7,0,1.8)
    after['measurement']=dict(constraint_uses_extrema=True,minimum=1.6,maximum=1.9)
    after['status']='above_range'
    q=quantify(row,before,after)
    assert q['after_margin']==pytest.approx(-.1) and not q['satisfied']
    assert q['target_progress_percent']==pytest.approx(50)
    assert q['progress_statistic']=='window_maximum'
