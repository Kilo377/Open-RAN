"""Frozen baseline and interval algebra tests, not network performance data."""
from copy import deepcopy
import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from smo.__main__ import GraphService, handler
from smo.intent_language import IntentError, resolve_ranges
from smo.intent_ledger import IntentLedger, intersect_intents
from smo.contracts import build_graph
from smo.population import graph_view
from test_smo_intent import snapshot, interval


KPI = 'network_goodput_mbps'


def at(value, slot):
    frame = snapshot()
    frame.update(slot=slot, time_ms=slot/2)
    frame['states']['kpi:'+KPI]['value'] = value
    return frame


class Language:
    def __init__(self): self.calls = []
    def status(self): return dict(configured=True)
    def parse(self, text, graph, validator):
        self.calls.append(text)
        raw = dict(ranges=[interval(KPI,'current','maximum',False,True) if text=='up'
            else interval(KPI)],questions=[])
        return validator(raw), dict(interpreted_by='llm-fixture')


def service(path=None):
    app = GraphService(language=Language(),intent_store=path)
    app.source = dict(mode='recorded', name='Analytic fixture',frames=[at(20,1),at(30,2),at(25,3),at(10,4)])
    return app


def test_two_intents_keep_their_original_anchors_and_intersect_without_label_rejection(tmp_path):
    app = service(tmp_path/'ledger.json')
    original = deepcopy(app.source)
    first = app.analyze_intent('up',1)
    old = first['active_intents'][0]
    assert old['ranges'][0]['lower']==20
    second = app.analyze_intent('down',2,[old['id']])
    row = second['ranges'][0]
    assert (row['lower'],row['upper']) == (20,30) and not row['empty']
    assert not row['lower_inclusive'] and not row['upper_inclusive']
    assert [r['current'] for r in row['anchors']] == [20,30]
    assert second['active_intents'][0] == old
    assert app.source == original and second['deployment_enabled'] is False
    # The throughput controller remains related despite its maximize label.
    assert any(c['app_id']=='xapp:throughput_scheduler_mlp' for c in second['candidates'])
    assert app.language.calls == ['up','down']
    refreshed = app.review_intents([r['id'] for r in second['active_intents']],3)
    assert refreshed['ranges'][0]['current']==25
    assert (refreshed['ranges'][0]['lower'],refreshed['ranges'][0]['upper'])==(20,30)
    assert refreshed['ranges'][0]['status']=='within_range'
    assert refreshed['candidates']  # Related controls are still reviewable.
    assert app.language.calls == ['up','down']
    assert app.review_intents([old['id']],4)['ranges'][0]['lower']==20


def test_empty_intersection_is_retained_and_removing_one_constraint_restores_the_other():
    app=service()
    first=app.analyze_intent('up',2)
    ids=[first['active_intents'][0]['id']]
    second=app.analyze_intent('down',1,ids)
    assert second['ranges'][0]['empty'] and not second['candidates']
    assert len(second['active_intents'])==2
    restored=app.review_intents(ids,1)
    assert not restored['ranges'][0]['empty'] and restored['ranges'][0]['lower']==30
    assert len(app.intent_ledger.records)==2  # Removal never deletes the audit.


def test_saved_numeric_anchors_survive_service_restart(tmp_path):
    path=tmp_path/'ledger.json';app=service(path)
    first=app.analyze_intent('up',1)
    fresh=service(path)
    result=fresh.analyze_intent('down',2,[first['active_intents'][0]['id']])
    assert (result['ranges'][0]['lower'],result['ranges'][0]['upper'])==(20,30)
    saved=json.loads(path.read_text())
    assert len(saved['records'])==2 and 'api_key' not in path.read_text()
    assert saved['records'][0]['ranges'][0]['requested'][0]['lower']=='current'
    assert saved['records'][0]['ranges'][0]['lower']==20


@pytest.mark.parametrize('lower,upper,li,ui,empty', [
    (20,30,False,False,False),(30,20,False,False,True),
    (20,20,False,False,True),(20,20,True,True,False),
    (20,20,True,False,True),
])
def test_frozen_intersection_preserves_endpoint_semantics(lower,upper,li,ui,empty):
    graph=graph_view(build_graph());ledger=IntentLedger({})
    up=resolve_ranges(dict(ranges=[interval(KPI,'current','maximum',li,True)],questions=[]),graph,at(lower,1))
    down=resolve_ranges(dict(ranges=[interval(KPI,'minimum','current',True,ui)],questions=[]),graph,at(upper,2))
    records=[ledger.add('up',at(lower,1),up,{}),ledger.add('down',at(upper,2),down,{})]
    row=intersect_intents(records,graph,at(25,3))['ranges'][0]
    assert row['empty'] is empty and row['lower']==lower and row['upper']==upper


@pytest.mark.parametrize('ids',[['unknown'],['unknown','unknown'],{},'unknown'])
def test_unknown_duplicate_or_forged_ids_fail_before_language_call(ids):
    app=service()
    with pytest.raises(IntentError):app.analyze_intent('up',1,ids)
    assert not app.language.calls and not app.intent_ledger.records


def test_ledger_cannot_be_reused_for_a_different_network_context(tmp_path):
    path=tmp_path/'ledger.json';app=service(path);app.analyze_intent('up',1)
    with pytest.raises(ValueError):IntentLedger(dict(other='network'),path)


def test_missing_observation_never_reanchors():
    app=service();first=app.analyze_intent('up',1)
    frame=at(30,2);frame['states'].pop('kpi:'+KPI)
    app.source['frames'].append(frame|dict(slot=5))
    result=app.review_intents([first['active_intents'][0]['id']],5)
    assert result['ranges'][0]['lower']==20 and result['ranges'][0]['current'] is None
    assert result['ranges'][0]['anchors'][0]['current']==20


def test_different_kpis_are_intersected_independently():
    graph=graph_view(build_graph());ledger=IntentLedger({})
    first=resolve_ranges(dict(ranges=[interval(KPI,'current','maximum',False,True),interval('ue_energy_j')],questions=[]),graph,at(20,1))
    second=resolve_ranges(dict(ranges=[interval(KPI),interval('ue_energy_j',1,2,True,True)],questions=[]),graph,at(30,2))
    records=[ledger.add('first',at(20,1),first,{}),ledger.add('second',at(30,2),second,{})]
    rows={r['kpi']:r for r in intersect_intents(records,graph,at(25,3))['ranges']}
    assert (rows[KPI]['lower'],rows[KPI]['upper'])==(20,30)
    assert (rows['ue_energy_j']['lower'],rows['ue_energy_j']['upper'])==(1,2)
    assert all(len(r['anchors'])==2 for r in rows.values())


def test_http_accepts_saved_ids_but_not_client_supplied_frozen_ranges():
    app=service();server=ThreadingHTTPServer(('127.0.0.1',0),handler(app))
    worker=Thread(target=server.serve_forever,daemon=True);worker.start()
    base='http://127.0.0.1:'+str(server.server_port)
    def post(path,data):
        with urlopen(Request(base+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})) as reply:
            return json.load(reply)
    try:
        first=post('/api/intent',dict(text='up',slot=1,active_intent_ids=[]))
        second=post('/api/intent',dict(text='down',slot=2,active_intent_ids=[first['active_intents'][0]['id']]))
        ids=[r['id'] for r in second['active_intents']]
        reviewed=post('/api/intent/review',dict(slot=3,active_intent_ids=ids))
        assert (reviewed['ranges'][0]['lower'],reviewed['ranges'][0]['upper'])==(20,30)
        assert app.language.calls==['up','down']
        for path,data in [('/api/intent',dict(text='up',ranges=[])),('/api/intent/review',dict(slot=3,active_intent_ids=ids,ranges=[])),('/api/intent/review',dict(slot=3,active_intent_ids=['unknown']))]:
            with pytest.raises(HTTPError) as error:post(path,data)
            assert error.value.code==400
    finally:server.shutdown();server.server_close();worker.join()
