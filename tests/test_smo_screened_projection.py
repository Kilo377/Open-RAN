"""Screened action applicability is audit coverage, never a replacement gain."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from smo.statistical_filter import filter_graph

ROOT=Path(__file__).resolve().parents[1]


def screened(parameter, rows, base, target, frame=None):
    graph=dict(metric_scope='ue_population.v1',nodes=[dict(id='kpi:k',kind='kpi',unit='W')],edges=[
        dict(id='e',kind='influences',source='rcp:'+parameter,target='kpi:k',sensitivities=rows)])
    graph=filter_graph(graph)
    audit=graph['statistical_filter']['excluded_relations'][0]
    code="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(d);
const result=c.SmoConflictRisk.screenedProjection(d.audit,d.edge,d.base,d.target,d.frame);
if(before!==JSON.stringify(d))throw Error('Applicability audit mutated input');
process.stdout.write(JSON.stringify(result));"""
    result=json.loads(subprocess.check_output(['node','-e',code],cwd=ROOT,text=True,input=json.dumps(dict(
        audit=audit,edge=graph.get('edges',[{}])[0] if graph.get('edges') else dict(source='rcp:'+parameter,target='kpi:k'),
        base=base,target=target,frame=frame or {'states':{}}))))
    return result,audit


def row(parameter, **changes):
    return dict(parameter=parameter,kind='range_linear_fit',contrast_id='measured',
        members=['u'],population_control_eligible=True,estimate=.1,
        confidence_interval=[-1,1],confidence_level=.95,unit='W / weight',x_unit='weight',
        fit_domain=[.5,4],**changes)


def sleep_row(**changes):
    value=row('cell_sleep_mode')
    value.update(kind='finite_contrast',unit='W / contrast',x_unit='contrast',
        **{'from':'active','to':'light_sleep'})
    value.update(changes)
    return value


@pytest.mark.parametrize('base,target,expected',[
    ('active','light_sleep',True),('light_sleep','active',True),
    ('active','deep_sleep',False),('deep_sleep','light_sleep',False)])
def test_only_the_exact_measured_category_transition_or_reverse_is_screened(base,target,expected):
    result,_=screened('cell_sleep_mode',[sleep_row()],{'u':base},{'u':target})
    assert result['excluded'] is expected
    assert 'drive' not in result and not result['coefficients_used'] and not result['physical_zero_assumed']


def test_range_requires_every_changed_member_and_measured_parameter_domain():
    evidence=row('ue_scheduler_weights',reference_scheduler_mode='latency_aware')
    frame={'states':{'rcp:scheduler_mode':{'by_entity':{'c':'round_robin'}}}}
    good,audit=screened('ue_scheduler_weights',[evidence],{'u':1},{'u':2},frame)
    assert good['excluded'] and good['calibrations']==['measured']
    assert good['context_notes']==['reference_context_shift']
    assert audit['excluded_estimates'][0]['calibration']['members']==['u']
    for base,target in [({'v':1},{'v':2}),({'u':.25},{'u':2}),({'u':1,'v':1},{'u':2,'v':2})]:
        assert not screened('ue_scheduler_weights',[evidence],base,target)[0]['excluded']


def test_rejected_coefficient_magnitude_does_not_enter_applicability_result():
    evidence=row('ue_scheduler_weights')
    first,_=screened('ue_scheduler_weights',[evidence],{'u':1},{'u':2})
    evidence.update(estimate=-1e6,confidence_interval=[-2e6,2e6])
    second,_=screened('ue_scheduler_weights',[evidence],{'u':1},{'u':2})
    assert first==second


def test_range_applicability_uses_canonical_parameter_coordinate():
    evidence=row('ue_offload_fraction')
    evidence.update(unit='W / %',x_unit='%',fit_domain=[10,90])
    assert screened('ue_offload_fraction',[evidence],{'u':.5},{'u':.6})[0]['excluded']
    assert not screened('ue_offload_fraction',[evidence],{'u':.5},{'u':.95})[0]['excluded']


@pytest.mark.parametrize('changes,why',[
    ({'unit':'ms / weight'},'interval_includes_zero'),
    ({'population_control_eligible':False},'population_evidence_unavailable'),
    ({'confidence_level':None},'confidence_level_unverified'),
    ({'confidence_interval':None},'confidence_interval_unavailable')])
def test_invalid_or_unverified_measurement_is_not_a_screened_effect(changes,why):
    evidence=row('ue_scheduler_weights');evidence.update(changes)
    result,audit=screened('ue_scheduler_weights',[evidence],{'u':1},{'u':2})
    assert not result['excluded']
    assert audit['excluded_estimates'][0]['reason']==why
    assert not audit['excluded_estimates'][0]['screened_effect_eligible']


def test_structured_drx_requires_coverage_of_every_changed_field():
    baseline=dict(enabled=True,cycle_ms=40,inactivity_ms=8,on_duration_ms=4)
    scalar=row('ue_drx_configs');scalar.update(unit='W / ms',x_unit='ms',scalar_component='inactivity_ms',fit_domain=[1,16])
    target=dict(baseline,inactivity_ms=4)
    assert screened('ue_drx_configs',[scalar],{'u':baseline},{'u':target})[0]['excluded']
    target['cycle_ms']=80
    assert not screened('ue_drx_configs',[scalar],{'u':baseline},{'u':target})[0]['excluded']
    profile=sleep_row()
    profile.update(parameter='ue_drx_configs',**{'from':dict(cycle_ms=40,inactivity_ms=8),'to':dict(cycle_ms=80,inactivity_ms=1)})
    target=dict(baseline,cycle_ms=80,inactivity_ms=1)
    assert screened('ue_drx_configs',[profile],{'u':baseline},{'u':target})[0]['excluded']
    assert screened('ue_drx_configs',[profile],{'u':target},{'u':baseline})[0]['excluded']
    target['on_duration_ms']=8
    assert not screened('ue_drx_configs',[profile],{'u':baseline},{'u':target})[0]['excluded']


def test_observed_zero_is_audited_but_does_not_claim_zero_physical_effect():
    evidence=row('ue_scheduler_weights');evidence.update(estimate=0,confidence_interval=[0,0])
    original=deepcopy(evidence)
    result,audit=screened('ue_scheduler_weights',[evidence],{'u':1},{'u':2})
    assert result['excluded'] and not result['physical_zero_assumed']
    assert audit['excluded_estimates'][0]['reason']=='observed_zero' and evidence==original


def test_contradictory_zero_flag_is_not_valid_omission_provenance():
    evidence=row('ue_scheduler_weights')
    evidence.update(estimate=1,confidence_interval=[.5,1.5],zero_in_both=True)
    result,audit=screened('ue_scheduler_weights',[evidence],{'u':1},{'u':2})
    assert not result['excluded'] and not audit['excluded_estimates'][0]['screened_effect_eligible']
