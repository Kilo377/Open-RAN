"""The broker preserves only the existing LLM interval boundary, never actions."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
import pytest
from smo.intent_relay import LocalIntentRelay
from smo.intent_language import (resolve_ranges,IntentError,MEC_LATENCY_GLOSSARY,
    MEC_LATENCY_GLOSSARY_VERSION,MEC_LATENCY_GLOSSARY_SHA256)
from smo.contracts import build_graph
from smo.population import graph_view
from test_smo_intent import snapshot,interval


@pytest.mark.parametrize('url',['https://127.0.0.1:1','http://remote.example:1','http://user:pass@localhost:1','http://localhost:1/other'])
def test_broker_refuses_non_loopback_or_credential_urls(url):
    with pytest.raises(ValueError):LocalIntentRelay(url)


def test_broker_uses_original_llm_ranges_resolves_current_locally_and_rejects_catalog_drift():
    seen=[];graph=graph_view(build_graph());source=dict(mode='recorded',frames=[snapshot()]);bundle=dict(graph=graph,source=source)
    class Endpoint(BaseHTTPRequestHandler):
        def do_GET(self):
            data=bundle if self.path=='/api/graph'else dict(provider='DeepSeek',configured=True,verified=True,model='fixture')
            body=json.dumps(data).encode();self.send_response(200);self.end_headers();self.wfile.write(body)
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));seen.append((self.path,data))
            body=json.dumps(dict(ranges=[dict(lower=0,upper=999,requested=[interval('ue_energy_j')])],questions=[],language=dict(interpreted_by='llm'),candidates=[dict(app_id='untrusted_old_candidate')],deployment_enabled=True)).encode()
            self.send_response(200);self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Endpoint);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        relay=LocalIntentRelay('http://127.0.0.1:'+str(server.server_port));relay.verify_context(bundle)
        parsed,provenance=relay.parse_at_snapshot('Lower UE energy',graph,snapshot=snapshot(),validator=lambda raw:resolve_ranges(raw,graph,snapshot()))
        assert set(parsed)=={'ranges','questions'} and parsed['ranges'][0]['upper']==3.
        assert seen==[('/api/intent',dict(text=MEC_LATENCY_GLOSSARY+'\n\nOriginal user intent:\nLower UE energy'))]
        assert provenance['kpi_semantics_version']==MEC_LATENCY_GLOSSARY_VERSION
        assert provenance['kpi_semantics_sha256']==MEC_LATENCY_GLOSSARY_SHA256
        assert provenance['interpreted_by']=='llm' and relay.status()['deployment_enabled']is False
        relay.verify_context(dict(graph=graph,source={**source,'mode':'different'}))
        from copy import deepcopy
        changed=deepcopy(graph);next(n for n in changed['nodes']if n['kind']=='kpi')['unit']='wrong'
        with pytest.raises(ValueError):relay.verify_context(dict(graph=changed,source=source))
    finally:server.shutdown();server.server_close();thread.join()


@pytest.mark.parametrize('key,text',[
    ('mec_offloaded_latency_ms','降低 MEC 卸载任务的平均端到端时延。'),
    ('task_latency_ms','降低全部任务（本地执行和卸载）的平均端到端时延。'),
])
def test_relay_passes_both_model_selected_kpis_unchanged_and_preserves_original_text(monkeypatch,key,text):
    graph=graph_view(build_graph());frame=snapshot()
    frame['states']['kpi:'+key]=dict(value=12.)
    seen=[];relay=LocalIntentRelay('http://127.0.0.1:1');relay._verified_source='fixture'
    def request(path,data=None):
        seen.append((path,data))
        if path=='/api/intent/status':return dict(configured=True)
        return dict(ranges=[dict(requested=[interval(key)])],questions=[],language=dict(interpreted_by='llm-fixture'))
    monkeypatch.setattr(relay,'_request',request)
    from smo.__main__ import GraphService
    service=GraphService(language=relay);service.graph=graph
    service.source=dict(mode='recorded',name='Analytic fixture',frames=[frame])
    result=service.analyze_intent(text,1600)
    assert result['ranges'][0]['kpi']==key and result['ranges'][0]['upper']==12.
    assert result['active_intents'][0]['text']==text
    sent=next(data['text']for path,data in seen if path=='/api/intent')
    assert sent==MEC_LATENCY_GLOSSARY+'\n\nOriginal user intent:\n'+text
    assert result['language']['original_text_sha256']==__import__('hashlib').sha256(text.encode()).hexdigest()


def test_relay_rejects_oversize_semantic_request_without_truncation_or_network_call(monkeypatch):
    relay=LocalIntentRelay('http://127.0.0.1:1');relay._verified_source='fixture';calls=[]
    monkeypatch.setattr(relay,'_request',lambda *args,**kwargs:calls.append(args))
    with pytest.raises(IntentError,match='请缩短意图'):
        relay.parse_at_snapshot('x'*2000,graph_view(build_graph()))
    assert calls==[]
