"""Reuse a running loopback intent service without extracting its session key.

Only the LLM's validated original interval requests are consumed. The older
service's candidates, actions and graph are never used by the new matcher.
Only the KPI catalog must agree. Symbolic bounds are resolved again against
the new service snapshot; the broker's numerical anchors are discarded.
"""
import hashlib
import json
from urllib.parse import urlparse
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from smo.intent_language import (IntentError,LlmUnavailable,MEC_LATENCY_GLOSSARY,
    MEC_LATENCY_GLOSSARY_VERSION,MEC_LATENCY_GLOSSARY_SHA256)


def source_signature(source):
    return hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class LocalIntentRelay:
    def __init__(self,url):
        parsed=urlparse(url)
        if (parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or not parsed.port
                or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment):
            raise ValueError('The intent broker must be an explicit loopback HTTP service.')
        self.url=url.rstrip('/');self._verified_source=None

    def _request(self,path,data=None):
        payload=json.dumps(data,allow_nan=False).encode() if data is not None else None
        request=Request(self.url+path,data=payload,headers={'Content-Type':'application/json'},method='POST'if payload is not None else'GET')
        try:
            with urlopen(request,timeout=55)as response:
                body=response.read(8*1024*1024+1)
                if len(body)>8*1024*1024:raise LlmUnavailable('The local intent broker returned an oversized response.')
                return json.loads(body)
        except HTTPError as error:
            raise LlmUnavailable('The local DeepSeek session could not interpret this request. Check its connection or KPI bounds.')from error
        except (URLError,TimeoutError,ValueError)as error:
            raise LlmUnavailable('The local DeepSeek session is unavailable.')from error

    def status(self):
        try:status=self._request('/api/intent/status')
        except LlmUnavailable:status=dict(provider='DeepSeek',model='deepseek-v4-pro',configured=False,verified=False,broker_available=False)
        return {**status,'deployment_enabled':False,'credential_storage':'existing_loopback_service_memory','language_broker':self.url}

    def configure(self,key,model=None):
        self._request('/api/intent/session-key',dict(api_key=key,model=model))
        return self.status()

    def clear_key(self):
        self._request('/api/intent/clear-key',{})
        return self.status()

    def verify_context(self,bundle):
        remote=self._request('/api/graph')
        catalog=lambda g:sorted((n['key'],n['unit'],n['scope'])for n in g['nodes']if n['kind']=='kpi')
        if catalog(remote['graph'])!=catalog(bundle['graph']):
            raise ValueError('The intent broker KPI catalog differs.')
        self._verified_source=source_signature(bundle['source'])

    def parse_at_snapshot(self,text,graph,validator=None,snapshot=None):
        if not self._verified_source:raise LlmUnavailable('The intent broker snapshot context is not verified.')
        if not isinstance(text,str)or not 1<=len(text.strip())<=2000:raise IntentError('Describe a KPI intent in 1–2,000 characters.')
        # The existing broker retains its API credential in memory. Supply the
        # same catalog semantics as the new native parser without extracting
        # credentials or replacing the model's KPI selection after the call.
        broker_text=MEC_LATENCY_GLOSSARY+'\n\nOriginal user intent:\n'+text
        if len(broker_text)>2000:
            raise IntentError('加入 KPI 口径说明后超过 2,000 字符，请缩短意图后重试。')
        reply=self._request('/api/intent',dict(text=broker_text))
        raw=dict(ranges=[r for row in reply['ranges'] for r in row['requested']],questions=reply['questions'])
        interpreted=validator(raw)if validator else raw
        return interpreted,{**reply['language'],'transport':'verified_loopback_interval_relay',
            'kpi_semantics_version':MEC_LATENCY_GLOSSARY_VERSION,
            'kpi_semantics_sha256':MEC_LATENCY_GLOSSARY_SHA256,
            'original_text_sha256':hashlib.sha256(text.encode()).hexdigest()}

    def parse(self,*args,**kwargs):
        return self.parse_at_snapshot(*args,**kwargs)
