"""Active intent units and legacy-constraint handling for UE power."""
from copy import deepcopy
import json

import pytest

from smo.__main__ import GraphService
from smo.contracts import build_graph
from smo.population import graph_view
from smo.intent_language import IntentError, IntentLanguage, resolve_ranges
from test_smo_intent import interval,snapshot


def app():
    service=GraphService()
    frame=snapshot()
    frame['states']['kpi:ue_power_w'].update(value=5.8,window='counter_delta',window_start_slot=1400,sampled_at_slot=1600,source='recorded_execution')
    frame['states']['kpi:ap_power_w']=dict(value=89.97)
    service.source=dict(mode='recorded',name='Analytic fixture',frames=[frame])
    return service,frame


def test_power_is_primary_and_cumulative_energy_remains_historical():
    graph=graph_view(build_graph())
    nodes={n['id']:n for n in graph['nodes']}
    assert nodes['xapp:ue_energy_rf']['target_kpis']=={'ue_power_w':'minimize'}
    assert nodes['kpi:ue_power_w']['primary'] and nodes['kpi:ue_power_w']['unit']=='W'
    assert nodes['kpi:ap_power_w']['unit']=='W'
    assert not nodes['kpi:ue_energy_j']['primary'] and nodes['kpi:ue_energy_j']['unit']=='J'
    assert nodes['kpi:ue_energy_j']['preferred_direction'] is None


def test_lower_power_resolves_from_window_value_not_cumulative_energy():
    service,frame=app()
    result=resolve_ranges(dict(ranges=[interval('ue_power_w')],questions=[]),service.graph,frame)['ranges'][0]
    assert result['lower']==0 and result['upper']==5.8 and not result['upper_inclusive']
    assert result['unit']=='W' and result['scope']=='All UEs'
    assert result['measurement']['window_start_slot']==1400
    assert result['upper']!=frame['states']['kpi:ue_energy_j']['value']


def test_old_joule_anchor_is_preserved_and_cannot_be_silently_relabelled():
    service,frame=app()
    parsed=resolve_ranges(dict(ranges=[interval('ue_energy_j')],questions=[]),service.graph,frame)
    old=service.intent_ledger.add('Lower UE energy (old schema)',frame,parsed,{})
    before=deepcopy(old)
    reviewed=service.review_intents([old['id']],1600)
    assert reviewed['invalidated_kpis']==['ue_energy_j'] and reviewed['questions']==[]
    assert reviewed['planning_blocked']
    issue=reviewed['constraint_issues'][0]
    assert issue['kind']=='legacy_energy_requires_reparse' and issue['intent_ids']==[old['id']]
    assert not reviewed['combination_catalog']['groups']
    assert reviewed['active_intents'][0]==before
    assert reviewed['ranges'][0]['unit']=='J' and reviewed['ranges'][0]['upper']==3
    assert reviewed['ranges'][0]['upper']!=5.8


def test_old_energy_issue_does_not_become_an_error_on_a_valid_bs_power_parse():
    service,frame=app()
    old=service.intent_ledger.add('Old UE energy',frame,
        resolve_ranges(dict(ranges=[interval('ue_energy_j')],questions=[]),service.graph,frame),{})
    before=deepcopy(old)
    class Language:
        def status(self): return dict(configured=True)
        def parse(self,text,graph,validator):
            return validator(dict(ranges=[interval('network_goodput_mbps','current','maximum',False,True),interval('ap_power_w')],questions=[])),{}
    service.language=Language()
    result=service.analyze_intent('Increase throughput and reduce BS power',1600,[old['id']])
    assert result['questions']==[] and len(result['active_intents'])==2
    assert result['constraint_issues'][0]['intent_ids']==[old['id']]
    assert result['planning_blocked'] and not result['combination_catalog']['groups']
    assert service.intent_ledger.records[old['id']]==before
    new=result['active_intents'][-1]
    assert {r['kpi'] for r in new['ranges']}=={'network_goodput_mbps','ap_power_w'}
    reviewed=service.review_intents([new['id']],1600)
    assert reviewed['questions']==[] and reviewed['constraint_issues']==[]
    assert not reviewed['planning_blocked'] and reviewed['combination_catalog']['groups']
    assert service.intent_ledger.records[old['id']]==before


def test_empty_saved_interval_has_its_own_issue_without_poisoning_new_parse():
    service,frame=app()
    frame['states']['kpi:urllc_failure_events_per_s']=dict(value=0)
    saved=service.intent_ledger.add('Reduce URLLC failure from zero',frame,
        resolve_ranges(dict(ranges=[interval('urllc_failure_events_per_s')],questions=[]),service.graph,frame),{})
    result=service.review_intents([saved['id']],1600)
    assert result['questions']==[] and result['planning_blocked']
    assert result['constraint_issues']==[dict(kind='empty_intersection',
        message=result['constraint_issues'][0]['message'],kpi='urllc_failure_events_per_s',
        intent_ids=[saved['id']],blocking=True)]
    assert result['combination_catalog']['empty_intersection']


def test_active_intent_rejects_cumulative_counter_even_if_model_returns_it():
    service,frame=app()
    class OldLanguage:
        def status(self): return dict(configured=True)
        def parse(self,text,graph,validator):
            return validator(dict(ranges=[interval('ue_energy_j')],questions=[])),{}
    service.language=OldLanguage()
    with pytest.raises(IntentError,match='ue_power_w'):
        service.analyze_intent('Lower power',1600)
    assert not service.intent_ledger.records


def test_native_language_catalog_and_examples_use_watts_and_explicit_scope():
    service,frame=app();seen={}
    class Client:
        def generate_json_or_raise(self,**kw):
            seen.update(kw)
            return dict(ranges=[interval('ue_power_w')],questions=[])
    language=IntentLanguage(lambda **kw:Client())
    parsed,_=language.parse('降低所有 UE 的总能耗',service.graph,
        validator=lambda raw:resolve_ranges(raw,service.graph,frame))
    catalog={r['kpi']:r for r in json.loads(seen['user_prompt'])['kpi_catalog']}
    assert catalog['ue_power_w']['unit']=='W' and 'counter deltas' in catalog['ue_power_w']['aggregation']
    assert 'ue_energy_j' not in catalog
    assert 'ue_power_w' in seen['system_prompt'] and 'joule budget' in seen['system_prompt']
    assert parsed['ranges'][0]['upper']==5.8
