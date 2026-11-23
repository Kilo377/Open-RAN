"""URLLC percentage objectives preserve canonical ratios and event-rate history."""
from copy import deepcopy
import json

import pytest

from smo.contracts import build_graph,validate_influences
from smo.intent_language import IntentLanguage,resolve_ranges
from smo.window_metrics import WindowIndex


def test_primary_urlcc_objective_is_fraction_without_relabelling_event_rate():
    graph=build_graph();nodes={n['id']:n for n in graph['nodes']}
    fraction=nodes['kpi:urllc_failure_fraction'];events=nodes['kpi:urllc_failure_events_per_s']
    assert fraction['primary'] and fraction['unit']=='ratio'
    assert fraction['display_unit']=='%' and fraction['display_multiplier']==100
    assert fraction['improvement_unit']=='percentage_points'
    assert not events['primary'] and events['unit']=='events/s'
    app=nodes['xapp:latency_urllc_rf']
    assert app['target_kpis']['urllc_failure_fraction']=='minimize'
    assert 'urllc_failure_events_per_s'not in app['target_kpis']


def test_existing_ratio_and_event_rate_gains_are_independent_and_unchanged(tmp_path):
    shared=dict(parameter='scheduler_mode',kind='finite_contrast',sample_count=3,horizon_ms=500,
        scope='All URLLC UEs',context='Analytic fixture only',method='seed-paired contrast',
        data_sha256='a'*64,design='controlled_perturbation',**{'from':'round_robin','to':'qos_deadline'})
    rows=[dict(shared,kpi='urllc_failure_fraction',estimate=-.02,unit='ratio / contrast',confidence_interval=[-.03,-.01]),
          dict(shared,kpi='urllc_failure_events_per_s',estimate=-7.,unit='events/s / contrast',confidence_interval=[-9.,-5.])]
    p=tmp_path/'gains.json';p.write_text(json.dumps(dict(schema='oran.intervention-edges.v1',estimates=rows)))
    raw=p.read_bytes();original=build_graph();copy=deepcopy(original)
    graph=validate_influences(original,p)
    for row in rows:
        edge=next(e for e in graph['edges']if e['target']=='kpi:'+row['kpi']and e['source']=='rcp:scheduler_mode')
        assert edge['sensitivity']['estimate']==row['estimate']
        assert edge['sensitivity']['unit']==row['unit']
    assert original==copy and p.read_bytes()==raw


def ratio_index(*,offered=100,failures=7):
    cfg=dict(manifest=dict(slot_duration_ms=1,ues=[dict(ue_id=u)for u in('a','b')],cells=[]),
        configs=dict(ue=dict(ues=[dict(ue_id=u,slice_id='urllc')for u in('a','b')])))
    index=WindowIndex(cfg)
    for s in range(101):
        users=[dict(ue_id=u,packet=dict(admitted_packets=offered if s==100 else 0,
            deadline_failed_packets=failures if s==100 else 0,overflow_packets=0))for u in('a','b')]
        index.observe(s,users,{}, {})
    return index


def test_ratio_uses_the_same_window_numerator_and_denominator():
    states=ratio_index().window(['urllc_failure_fraction','urllc_failure_events_per_s'],0,100,through=100)['states']
    ratio=states['kpi:urllc_failure_fraction']
    assert ratio['value']==pytest.approx(.07) # 14 failure events / 200 offered packets
    assert ratio['value']*ratio['display_multiplier']==pytest.approx(7.)
    assert ratio['failure_events']==14 and ratio['offered_packets']==200
    assert ratio['includes_carryover'] and not ratio['mature_cohort_probability']
    assert states['kpi:urllc_failure_events_per_s']['value']==pytest.approx(140.)


def test_zero_exposure_is_unavailable_and_carryover_ratio_is_never_capped():
    no_packets=ratio_index(offered=0,failures=0).window(['urllc_failure_fraction'],0,100,through=100)['states']['kpi:urllc_failure_fraction']
    assert no_packets['value']is None and no_packets['reason']=='no_valid_offered_packet_samples'
    carry=ratio_index(offered=2,failures=9).window(['urllc_failure_fraction'],0,100,through=100)['states']['kpi:urllc_failure_fraction']
    assert carry['value']==4.5 and carry['includes_carryover']


def test_llm_uses_ratio_for_error_percent_but_keeps_explicit_event_rate_separate():
    captured={}
    class Client:
        def __init__(self,**kwargs):pass
        def generate_json_or_raise(self,**kwargs):
            captured.update(kwargs)
            return dict(ranges=[dict(kpi='urllc_failure_fraction',lower=0,upper=.07,
                lower_inclusive=True,upper_inclusive=True)],questions=[])
    graph=build_graph();frame=dict(states={'kpi:urllc_failure_fraction':dict(value=.1)})
    language=IntentLanguage(client_factory=Client)
    result,provenance=language.parse('URLLC 错误率不超过 7%',graph,
        validator=lambda raw:resolve_ranges(raw,graph,frame))
    assert result['ranges'][0]['upper']==.07
    assert '7% is 0.07'in captured['system_prompt']
    assert 'Never relabel events/s as percent'in captured['system_prompt']
    assert provenance['urllc_semantics_version']=='urllc-fraction-objective.v1'
