"""Opt-in current coordinates; no deployment data or default-policy changes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from smo.influence_units import normalization as frozen_normalization
from smo.observation_normalization import POLICY, POLICY_VERSION, normalization
from smo.result_evaluation import evaluate


ROOT=Path(__file__).resolve().parents[1]


def packet(value=100):
    ranges=[dict(kpi='power',unit='W',current=40,lower=20,upper=80,domain=[0,None],
        anchors=[dict(current=10,intent_id='old',snapshot=dict(slot=1,time_ms=.5)),
                 dict(current=30,intent_id='new',snapshot=dict(slot=2,time_ms=1))])]
    frame=dict(slot=2040,time_ms=1020,source_id='network',run_id='run',source_sha256='source',
        evaluation_origin_id='origin',evaluation_window=dict(start_slot=40,end_slot=2040),
        states={'kpi:power':dict(value=value,source='recorded_counter_window',window_slots=2000,
            sampled_at_slot=2040,measurement=dict(window='matched',sample_count=2000))})
    return ranges,frame


def browser(packets):
    if not shutil.which('node'):pytest.skip('Node required for production JS parity')
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/observation_normalization.js','utf8'),c);
const inputs=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(inputs);
const values=inputs.map(x=>c.SmoObservationNormalization.normalization(x.ranges,x.frame));
if(JSON.stringify(inputs)!==before)throw Error('Input mutated');
process.stdout.write(JSON.stringify(values));"""
    return json.loads(subprocess.check_output(['node','-e',script],cwd=ROOT,text=True,
        input=json.dumps([dict(ranges=r,frame=f)for r,f in packets])))


def test_current_frame_is_the_only_reference_and_old_policy_is_unchanged():
    ranges,frame=packet();original=deepcopy((ranges,frame))
    result=normalization(ranges,frame)
    assert result['policy']==POLICY=='current_observation_v1' and result['policy_version']==POLICY_VERSION==1
    assert result['kpis'][0]['scale']==100 and result['kpis'][0]['reference_value']==100
    assert result['reference_basis']=='current_observed_frame'
    assert result['reference_snapshot']==dict(slot=2040,time_ms=1020)
    assert result['reference_frame_provenance']['run_id']=='run'
    assert result['kpis'][0]['observation_provenance']['metadata']['window_slots']==2000
    assert result['deployment_batch_used']is False and result['intent_anchors_used_for_scale']is False
    assert result['observed_after_kpis_used']is False and result['interval_endpoints_modified']is False
    assert (ranges,frame)==original
    legacy=frozen_normalization(ranges,frame)
    assert legacy['kpis'][0]['scale']==10 and legacy.get('policy')is None


def test_persistent_anchor_order_and_intent_ids_do_not_affect_current_coordinates():
    ranges,frame=packet();reference=normalization(ranges,frame)
    ranges[0]['anchors'].reverse()
    for a in ranges[0]['anchors']:a.update(current=10000,intent_id='changed',snapshot={})
    ranges[0]['current']=50000
    assert normalization(ranges,frame)==reference
    frame['states']['kpi:power']['value']=120
    assert normalization(ranges,frame)['kpis'][0]['scale']==120
    assert ranges[0]['lower']==20 and ranges[0]['upper']==80


@pytest.mark.parametrize('value',[None,True,'123'])
def test_missing_or_non_numeric_current_never_uses_row_anchor_or_bound(value):
    ranges,frame=packet(value)
    row=normalization(ranges,frame)['kpis'][0]
    assert row['scale']is None and row['reference_value']is None
    assert row['status']=='unavailable' and row['reason']=='current_observation_unavailable'


@pytest.mark.parametrize('value',[0,-0.])
def test_nonratio_zero_requires_an_explicit_future_policy_instead_of_one_unit(value):
    ranges,frame=packet(value)
    row=normalization(ranges,frame)['kpis'][0]
    assert row['scale']is None and row['reason']=='explicit_scale_required'
    assert row['reference_observation_available']is True
    assert row['scale_available']is False


def test_absent_state_and_frame_provenance_are_explicit():
    ranges,_=packet()
    result=normalization(ranges,{})
    assert result['reference_snapshot']==dict(slot=None,time_ms=None)
    assert result['kpis'][0]['scale']is None
    assert result['kpis'][0]['observation_provenance']['state_present']is False


@pytest.mark.parametrize('unit,scale',[('ratio',1),('fraction',1),('%',100),('W',None),(None,None)])
def test_zero_url_uses_the_canonical_probability_range(unit,scale):
    ranges=[dict(kpi='urllc_failure_fraction',unit=unit,lower=0,upper=0,anchors=[dict(current=.07)])]
    frame=dict(slot=20,time_ms=10,states={'kpi:urllc_failure_fraction':{'value':0}})
    row=normalization(ranges,frame)['kpis'][0]
    assert row['scale']==scale
    assert row['reason']==('incompatible_ratio_unit'if scale is None else None)


def test_missing_url_reading_has_known_units_but_is_explicitly_unavailable():
    result=normalization([dict(kpi='urllc_failure_fraction')],dict(states={}))['kpis'][0]
    assert result['scale']==1 and result['scale_available']is True
    assert result['reference_value']is None and result['reference_observation_available']is False
    assert result['status']=='unavailable' and result['reason']=='current_observation_unavailable'


def test_failure_event_rate_is_not_misidentified_as_a_ratio():
    row=normalization([dict(kpi='urllc_failure_events_per_s',unit='events/s')],
        dict(states={'kpi:urllc_failure_events_per_s':{'value':0}}))['kpis'][0]
    assert row['scale']is None and row['reason']=='explicit_scale_required'


def test_unit_equivalence_for_power_and_canonical_ratio():
    for key,units,values,drives in [
        ('power',['W','mW'],[5,5000],[-.5,-500]),
        ('urllc_failure_fraction',['ratio','%'],[.07,7],[.01,1])]:
        coordinates=[]
        for unit,value,drive in zip(units,values,drives):
            row=normalization([dict(kpi=key,unit=unit)],dict(states={'kpi:'+key:{'value':value}}))['kpis'][0]
            coordinates.append(drive/row['scale'])
        assert coordinates[0]==pytest.approx(coordinates[1])


def test_python_browser_parity_for_missing_units_zero_and_public_provenance():
    packets=[packet(v)for v in (100,0,None,True,'123',-5)]
    packets.append((packet()[0],{}))
    for unit in ('ratio','fraction','%',None,'W'):
        for current in (0,None,.07):
            packets.append(([dict(kpi='urllc_failure_fraction',unit=unit,anchors=[dict(current=999)])],
                dict(slot=40,time_ms=20,states={'kpi:urllc_failure_fraction':{'value':current,'source':'public'}})))
    assert browser(packets)==[normalization(r,f)for r,f in packets]


def test_outputs_copy_public_provenance_without_aliasing_inputs():
    ranges,frame=packet();saved=deepcopy(frame);result=normalization(ranges,frame)
    result['reference_frame_provenance']['evaluation_window']['start_slot']=99
    result['kpis'][0]['observation_provenance']['metadata']['measurement']['sample_count']=99
    assert frame==saved


def test_old_anchor_multikpi_ordering_now_matches_shared_current_before_coordinates():
    from test_smo_intent_obstruction_linear_reference import abstract_case,run_reference_cases
    a=abstract_case([dict(before=100,anchor=10,delta=-1,lower=100,upper=None),
                    dict(before=100,anchor=100,delta=0,lower=100,upper=None)],'A')
    b=abstract_case([dict(before=100,anchor=10,delta=0,lower=100,upper=None),
                    dict(before=100,anchor=100,delta=-2,lower=100,upper=None)],'B')
    old=run_reference_cases([a,b]);frozen=deepcopy([a['ranges'],b['ranges']])
    assert old[0]['raw_obstruction_strength']>old[1]['raw_obstruction_strength']
    for case in (a,b):case['normalization']=normalization(case['ranges'],case['frame'])
    current=run_reference_cases([a,b])
    for case,result in zip((a,b),current):
        abstract_metric=evaluate(case['comparison'],ranges=case['ranges'])['after']['score']
        assert result['raw_obstruction_strength']==pytest.approx(max(0,-abstract_metric)/100)
    assert current[0]['raw_obstruction_strength']==pytest.approx(.01)
    assert current[1]['raw_obstruction_strength']==pytest.approx(.02)
    assert [a['ranges'],b['ranges']]==frozen


def test_fresh_intent_scales_equal_the_existing_policy_without_redefining_it():
    ranges,frame=packet();ranges[0]['anchors'][0]['current']=100
    current=normalization(ranges,frame);legacy=frozen_normalization(ranges,frame)
    assert current['kpis'][0]['scale']==legacy['kpis'][0]['scale']
    assert current['kpis'][0]['reference_value']==legacy['kpis'][0]['reference_value']
