"""Active SMO path uses KPI ownership and graph edges, never action templates."""
from copy import deepcopy
from pathlib import Path

from smo.__main__ import GraphService, export_html
from test_smo_intent import interval, snapshot


class Language:
    def __init__(self): self.calls=0
    def status(self): return dict(configured=True)
    def parse(self,text,graph,validator):
        self.calls+=1
        return validator(dict(ranges=[interval('ue_power_w'),interval('mec_offloaded_latency_ms')],questions=[])), dict(interpreted_by='fixture-only')


def test_active_service_never_calls_profiles_or_action_matcher(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('An action-space function was called by graph-only SMO')
    monkeypatch.setattr('smo.intent_match.profiles',forbidden)
    monkeypatch.setattr('smo.intent_match.match_intent',forbidden)
    service=GraphService(language=Language())
    frame=snapshot();frame['states']['kpi:mec_offloaded_latency_ms']=dict(value=9.9)
    # Even actual logged proposals must not become hypothetical policy choices.
    frame['states']['xapp:ue_energy_rf']=dict(enabled=False,current_proposals=[dict(action=999)])
    service.source=dict(mode='recorded',name='Analytic fixture',frames=[frame])
    before=deepcopy((service.graph,service.source))
    result=service.analyze_intent('Reduce UE energy and offloaded-task delay',1600)
    assert result['analysis_basis']=='knowledge_graph'
    assert result['model_inference_invoked'] is False
    assert result['runtime_proposals_used'] is False
    assert result['hypothetical_actions_used'] is False
    assert result['automatic_selection'] is False and result['deployment_enabled'] is False
    assert {r['kpi'] for r in result['ranges']}=={'ue_power_w','mec_offloaded_latency_ms'}
    assert all('variants' not in c and 'actions' not in c for c in result['candidates'])
    expected={'xapp:ue_energy_rf','xapp:latency_scheduler_rf'}
    assert {c['app_id'] for c in result['candidates']}==expected
    assert len(result['combination_catalog']['groups'])==1
    assert set(result['combination_catalog']['groups'][0]['app_ids'])==expected
    assert 'assignment_count' not in result['combination_catalog']
    ids=[r['id'] for r in result['active_intents']]
    reviewed=service.review_intents(ids,1600)
    assert service.language.calls==1
    assert reviewed['ranges']==result['ranges']
    assert (service.graph,service.source)==before


def test_browser_bundle_has_no_action_catalog_or_hypothetical_controls(tmp_path):
    web=Path(__file__).resolve().parents[1]/'smo/web'
    js=(web/'app.js').read_text();html=(web/'index.html').read_text()
    assert 'ACTION_CATALOG_START' not in js
    assert 'SmoConflicts.enumerateCombinations(' not in js
    assert 'SmoConflicts.review(' not in js
    assert '假设动作模板' not in html+js
    assert 'id="intent-variant"' not in html+js
    artifact=export_html(GraphService().bundle(),tmp_path/'graph.html').read_text()
    assert 'ACTION_CATALOG_START' not in artifact
    assert '<script src="./legacy_action_review.js"' not in artifact
