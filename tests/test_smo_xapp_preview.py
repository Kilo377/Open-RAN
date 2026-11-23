"""Shadow execution must not train, select, arbitrate, or modify live state."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest

from smo.xapp_preview import PreviewStore, preview_one

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def released(tmp_path, monkeypatch):
    from ric.near_rt.pools import build_pool_xapp
    import shutil
    bundle = ROOT/'artifacts/xapps/chongyu/20261117'
    release = json.loads((bundle/'release.json').read_text())
    target = tmp_path/'outputs/models'
    shutil.copytree(bundle, target)
    for mod in ('ue_energy_rf','ap_energy_rf','throughput_scheduler_mlp',
                'latency_scheduler_rf','latency_urllc_rf','latency_urllc_qos_rf','mec_offload_budget_qgbdt'):
        monkeypatch.setattr(importlib.import_module('ric.near_rt.pools.chongyu.'+mod), 'ROOT', tmp_path)
    def build(name):
        entry = release['models'][name]
        params = dict(entry['runtime_parameters'], mode='frozen', model_path=str(target/entry['path']))
        if name == 'x12_mec': params['complete_scheduler_bundle'] = True
        if name == 'x12_urllc': params['balanced_start_windows'] = 0
        return build_pool_xapp(dict(implementation_id=entry['implementation_id'],
            cell_ids=['cell_0','cell_1','cell_2'], parameters=params), 'chongyu')
    return build


def observed(name):
    # Explicit synthetic public-report fixtures only; no inferred radio values.
    modules = dict(x01=('test_ue_energy_xapp','observation'), x02=('test_ap_energy_rf','obs'),
        x08=('test_throughput_scheduler_mlp','obs'), x12_mec=('test_latency_scheduler_rf','obs'),
        x12_urllc=('test_urllc_qos_rf','obs'), x13=('test_mec_offload_budget_qgbdt','obs'))
    module, method = modules[name]
    result = getattr(importlib.import_module(module), method)()
    if not hasattr(result, 'cell_observations'):
        result.cell_observations = [NS(cell_id='cell_0', executed_config={})]
    result.paper_interfaces_enabled = True
    return result


@pytest.mark.parametrize('name', ['x01','x02','x08','x12_mec','x12_urllc','x13'])
def test_real_released_inference_is_deterministic_and_preserves_every_original(released, name, tmp_path):
    app = released(name); view = observed(name); policy = NS(epoch=3)
    app.log_path = tmp_path/'must-not-be-written.jsonl'
    original = pickle.dumps((app, view, policy))
    py_rng, np_rng = random.getstate(), pickle.dumps(np.random.get_state())
    model_before = app.model_path.read_bytes()
    first, second = preview_one(app, view, policy), preview_one(app, view, policy)
    assert first == second
    assert first['status'] == 'proposed', first
    assert first['inference_invoked'] is True and first['mode'] == 'cold_activation'
    assert pickle.dumps((app, view, policy)) == original
    assert app.model_path.read_bytes() == model_before
    assert first['model_sha256'] == hashlib.sha256(model_before).hexdigest()
    assert random.getstate() == py_rng and pickle.dumps(np.random.get_state()) == np_rng
    assert not app.log_path.exists()
    assert first['actions'] and all(r['current'] is not None for r in first['actions'])


def test_active_pending_model_is_cloned_without_resetting_or_forcing_an_action(released):
    app = released('x01'); app.select(['cell_0'], {'cell_0':7})
    view = observed('x01'); app.run(view, NS())
    before = pickle.dumps(app)
    result = preview_one(app, view)
    assert result['mode'] == 'state_clone' and result['status'] == 'no_action'
    assert result['actions'] == [] and pickle.dumps(app) == before


def test_invalid_contract_is_not_called_or_reported_as_unchanged(released):
    app = released('x01')
    result = preview_one(app, NS(), failures=['ue_0:queue_bytes:stale'])
    assert result['status'] == 'unavailable' and not result['inference_invoked']
    assert result['actions'] == [] and result['failures'] == ['ue_0:queue_bytes:stale']


def test_numpy_public_scalars_are_preserved_and_missing_current_values_are_not_assumed_zero():
    from smo.xapp_preview import public_json, current_value
    assert public_json(dict(x=np.float64(.5),n=np.int64(2))) == dict(x=.5,n=2)
    assert current_value(NS(cell_observations=[],ue_observations=[]),'cell_0','ue_offload_fraction','ue_4') is None


def test_online_training_is_rejected_before_any_execution(released):
    app = released('x01'); app.mode = 'online'
    result = preview_one(app, NS())
    assert not result['inference_invoked'] and result['reason'] == 'frozen_trained_model_required'


def test_replay_never_substitutes_another_slot_and_returns_independent_values(tmp_path):
    row = dict(schema='oran.xapp-decision-preview.v1', slot=200,
        deployment_enabled=False, apps=[dict(actions=[dict(proposed=.5)])])
    (tmp_path/'xapp_decision_previews.jsonl').write_text(json.dumps(row)+'\n')
    store = PreviewStore(tmp_path)
    assert store.read(201)['status'] == 'unavailable'
    read = store.read(200);read['apps'][0]['actions'][0]['proposed'] = 1
    assert store.read(200) == row


def test_live_preview_explicitly_reports_its_captured_time_separate_from_current_graph(tmp_path):
    path=tmp_path/'live.json'
    path.write_text(json.dumps(dict(decision_preview=dict(schema='oran.xapp-decision-preview.v1',
        slot=200,deployment_enabled=False,apps=[]))))
    result=PreviewStore(live=path).read(240)
    assert result['slot']==200 and result['requested_slot']==240
    assert result['snapshot_note']


def test_graph_groups_equal_actions_without_losing_ue_or_cell_scope():
    from test_smo_preset_edits import probe
    rows=[dict(parameter='ue_scheduler_weights',current=1,proposed=v,changed=v!=1,
               cell_id=c,ue_id=u) for u,c,v in [('ue_0','cell_0',2),('ue_4','cell_2',2),('ue_6','cell_0',1)]]
    result=probe('return graph.previewGroups({actions:input});',rows)
    assert len(result)==2
    assert result[0]['scopes']==[dict(cell_id='cell_0',ue_id='ue_0'),dict(cell_id='cell_2',ue_id='ue_4')]
    assert result[0]['current']==1 and result[0]['proposed']==2
    assert result[1]['changed'] is False and result[1]['scopes']==[dict(cell_id='cell_0',ue_id='ue_6')]


def test_graph_never_merges_different_current_values_or_drx_configurations():
    from test_smo_preset_edits import probe
    rows=[dict(parameter='ue_drx_configs',current=c,proposed=p,changed=True,cell_id='cell_0',ue_id=u)
          for u,c,p in [('ue_0',{'cycle_ms':40,'enabled':True},{'enabled':True,'cycle_ms':80}),
                        ('ue_4',{'enabled':True,'cycle_ms':40},{'cycle_ms':80,'enabled':True}),
                        ('ue_1',{'cycle_ms':20,'enabled':True},{'enabled':True,'cycle_ms':80}),
                        ('ue_8',{'cycle_ms':40,'enabled':True},{'enabled':False,'cycle_ms':80})]]
    result=probe('return graph.previewGroups({actions:input});',rows)
    assert len(result)==3
    assert len(result[0]['scopes'])==2
    assert result[1]['current']['cycle_ms']==20
    assert result[2]['proposed']['enabled'] is False


def test_endpoint_accepts_only_snapshot_and_never_parameter_overrides():
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from smo.__main__ import GraphService, handler
    service = GraphService()
    service.source = dict(mode='recorded', name='fixture', frames=[dict(slot=200,time_ms=100,states={})])
    server = ThreadingHTTPServer(('127.0.0.1',0), handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    try:
        def request(data):
            return urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/xapps/preview',
                data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}))
        assert json.load(request({'slot':200}))['status'] == 'unavailable'
        with pytest.raises(HTTPError) as error:
            request({'slot':200, 'ue_offload_fraction':1})
        assert error.value.code == 400
    finally:
        server.shutdown();server.server_close();thread.join()
