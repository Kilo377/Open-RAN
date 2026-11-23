"""User memo persistence and optimistic writes never touch network intents."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from smo.presets import PresetStore,PresetConflict,IDS
from smo.intent_language import IntentError


def test_text_tags_and_deletions_survive_reload_with_history(tmp_path):
    path=tmp_path/'presets.json';s=PresetStore(path)
    content={'c04':dict(text='提高全网下行吞吐量，同时降低基站功率。',tag='[无冲突] 自己的备注')}
    one=s.save(0,content,['lc02']);assert one['revision']==1
    assert PresetStore(path).read()==one
    two=s.save(1,{'c04':dict(text=content['c04']['text'],tag='新的自由 tag')},['lc02'])
    assert two['revision']==2
    history=[json.loads(x)for x in path.with_suffix('.history.jsonl').read_text().splitlines()]
    assert history[-1]==one
    assert s.save(2,two['edits'],two['deleted_ids'])==two # no-op save preserves revision


def test_stale_or_empty_browser_cannot_erase_saved_tags(tmp_path):
    s=PresetStore(tmp_path/'presets.json');saved=s.save(0,{'c04':dict(text='降低功率。',tag='[间接]')},[])
    with pytest.raises(PresetConflict):s.save(0,{},[])
    assert s.read()==saved


@pytest.mark.parametrize('edits,deleted',[
    ({'unknown':dict(text='x',tag='y')},[]),({'c04':dict(text=' ',tag='x')},[]),
    ({'c04':dict(text='x',tag=123)},[]),({'c04':dict(text='x',tag='x'*121)},[]),
    ({'c04':dict(text='x',tag='memo',kpi='fake')},[]),({},['unknown'])])
def test_invalid_user_profile_never_replaces_good_data(tmp_path,edits,deleted):
    s=PresetStore(tmp_path/'presets.json');before=s.read()
    with pytest.raises(IntentError):s.save(0,edits,deleted)
    assert s.read()==before


def test_profile_http_roundtrip_does_not_create_intents_or_deploy(tmp_path):
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    from smo.__main__ import GraphService,handler
    service=GraphService(preset_store=tmp_path/'presets.json');before=deepcopy(service.intent_ledger.records)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(service));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/api/intent/presets'
    def post(data,origin=None):
        h={'Content-Type':'application/json'}
        if origin:h['Origin']=origin
        return json.load(urlopen(Request(url,data=json.dumps(data).encode(),headers=h)))
    payload=dict(revision=0,edits={'c04':dict(text='减少 UE 功率。',tag='直接？自由备注')},deleted_ids=['lc02'])
    try:
        assert json.load(urlopen(url))['revision']==0
        saved=post(payload);assert saved['edits']==payload['edits']and saved['revision']==1
        with pytest.raises(HTTPError)as error:post(payload)
        assert error.value.code==409
        with pytest.raises(HTTPError)as error:post({**payload,'revision':1},'https://untrusted.example')
        assert error.value.code==403
        assert service.intent_ledger.records==before and service.deployments.status()['latest']is None
    finally:server.shutdown();server.server_close();thread.join()


def test_known_ids_match_frontend_and_legacy_entry_routes_to_canonical():
    from test_smo_preset_edits import probe
    assert set(probe('return graph.intentCases.map(p=>p.id);'))==IDS
    source=(Path(__file__).resolve().parents[1]/'smo/web/app.js').read_text()
    assert "location.port==='53504'"in source and "target.port='53505'"in source
    assert "syncPresetStore();"in source and "await persistPresetData("in source
