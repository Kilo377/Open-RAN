"""Risk inputs must not depend on old KPI-forecast or arbitration eligibility."""
import json
from threading import Thread
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError

import pytest
from types import SimpleNamespace

from smo.__main__ import GraphService,handler
from test_smo_conflict_quantification import fixture


def service_fixture(monkeypatch):
    graph,frame,catalog,preview,_=fixture();service=GraphService();service.graph=graph
    monkeypatch.setattr(service,'intent_context',lambda slot:(dict(source=dict(name='fixture',frames=[frame],slot_duration_ms=.5)),frame))
    monkeypatch.setattr(service.intent_ledger,'select',lambda ids:[dict(id='saved')])
    monkeypatch.setattr('smo.__main__.intersect_intents',lambda *args:{})
    monkeypatch.setattr('smo.__main__.graph_intent_review',lambda *args:dict(combination_catalog=catalog,combination_catalog_with_uncertain=catalog))
    monkeypatch.setattr(service,'preview_xapps',lambda slot:preview)
    monkeypatch.setattr('smo.conflict_quantification.quantify_catalog',lambda *a,**kw:pytest.fail('Legacy KPI forecast must not gate risk inputs'))
    return service,preview,catalog


def test_trusted_inputs_return_all_groups_frozen_ranges_without_arbiter_or_net_prediction(monkeypatch):
    service,preview,catalog=service_fixture(monkeypatch)
    first=service.conflict_inputs(['saved'],100)
    assert first['status']=='ready' and first['method']=='trusted_control_chain_inputs_v1'
    assert first['groups'][0]['app_ids']==['a','b']
    assert [r['range']for r in first['groups'][0]['kpis']]==catalog['ranges']
    assert not first['kpi_forecast_performed'] and not first['arbitration_performed']
    assert first['arbitration_reference']['mode']=='uniform_random'
    assert not first['arbitration_reference']['held_state_reconstructed']
    assert first['input_sha256']==service.conflict_inputs(['saved'],100)['input_sha256']
    preview['slot']=99
    result=service.conflict_inputs(['saved'],100)
    assert result['reason']=='matching_decision_preview_unavailable' and not result['groups']


def test_http_trust_boundary_rejects_user_supplied_actions_coefficients_or_group_subset(monkeypatch):
    service,_,_=service_fixture(monkeypatch)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(service));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    def post(data,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        return urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/conflicts/inputs',data=json.dumps(data).encode(),headers=headers))
    request=dict(active_intent_ids=['saved'],slot=100,observe_slots=500)
    try:
        assert json.load(post(request))['groups'][0]['group_id']=='ab'
        for extra in ({'actions':[]},{'gains':[1]},{'group_ids':['ab']},{'influence_normalization':{'k':999}}):
            with pytest.raises(HTTPError)as error:post({**request,**extra})
            assert error.value.code==400
        with pytest.raises(HTTPError)as error:post(request,'https://unrelated.example')
        assert error.value.code==403
    finally:server.shutdown();server.server_close();thread.join()


def test_quantification_does_not_read_before_after_results_or_batch_state(monkeypatch):
    service,_,catalog=service_fixture(monkeypatch)
    first=service.conflict_inputs(['saved'],100)
    batch=dict(active_intent_ids=['saved'],ranges=catalog['ranges'],baseline_reference={'k':25},
        members=[dict(comparison=dict(after={'value':-999}))])
    service.deployments.batch=SimpleNamespace(latest='b',items={'b':batch})
    assert service.conflict_inputs(['saved'],100)==first
    row=first['influence_normalization']['kpis'][0]
    assert row['scale']==10 and row['weight']==1 and row['scale_basis']=='frozen_intent_reference_magnitude'
    assert not first['deployment_results_used'] and not first['deployment_batch_used']
    batch['baseline_reference']['k']=123456
    batch['result_evaluation_scales']={'k':.001}
    batch['members'][0]['comparison']['after']['value']=123456
    assert service.conflict_inputs(['saved'],100)==first
    class NoBatchReads:
        def __getattribute__(self,name):raise AssertionError('Risk must not inspect deployments')
    service.deployments.batch=NoBatchReads()
    assert service.conflict_inputs(['saved'],100)==first


def test_ui_risk_key_uses_current_context_without_any_deployment_reads():
    from pathlib import Path
    import subprocess
    source=(Path(__file__).resolve().parents[1]/'smo/web/app.js').read_text()
    fn=source[source.index('  function quantificationKey()'):source.index('  const currentQuantification=')]
    js='''const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm'),c={structuredClone};
vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);const SmoGraph=c.SmoGraph;
const state={intent:{snapshot:{slot:100},ranges:[{kpi:'k',upper:10}]},intentBusy:false,
 bundle:{source:{run_id:'current-network'}},activeIntents:[{id:'saved'}],showDashed:false,
 combinationCatalog:{groups:[{id:'ab'}]}};
const quantificationMethod=()=> 'goal_obstruction_v2',frame=()=>({slot:100}),$=()=>({value:'500'});
const activeBatch=()=>{throw Error('Unexpected deployment dependency');};
const matchingBatch=activeBatch;
'''+fn+'''
const first=quantificationKey();assert.ok(first.includes('current-network'));
assert.equal(first,quantificationKey());
state.intent.ranges[0].upper=9;assert.notEqual(first,quantificationKey());
state.intent.ranges[0].upper=10;state.combinationCatalog.groups.push({id:'abc'});
assert.notEqual(first,quantificationKey());
'''
    subprocess.run(['node','-e',js],check=True,capture_output=True,text=True)


def test_trusted_packet_keeps_exact_frame_and_preview_after_source_changes(monkeypatch):
    from copy import deepcopy
    import hashlib
    service,preview,_=service_fixture(monkeypatch)
    _,frame=service.intent_context(100);original=deepcopy((frame,preview))
    result=service.conflict_inputs(['saved'],100)
    assert result['snapshot_packet_schema']=='oran.frozen-control-snapshot.v1'
    assert result['decision_frame']==frame and result['decision_preview']['apps']==preview['apps']
    sha=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert result['decision_frame_sha256']==sha(result['decision_frame'])
    assert result['decision_preview_sha256']==sha(result['decision_preview'])
    frame['states']['kpi:k']['value']=999;preview['apps'][0]['actions'][0]['proposed']=999
    assert result['decision_frame']==original[0]
    assert result['decision_preview']['apps']==original[1]['apps']


def test_same_slot_live_publication_during_capture_rejects_a_mixed_packet(monkeypatch,tmp_path):
    from types import SimpleNamespace
    path=tmp_path/'live.json';path.write_text('{}')
    service,preview,_=service_fixture(monkeypatch);service.live=SimpleNamespace(path=path)
    def changed(slot):
        path.write_text('{"changed":true}');return preview
    monkeypatch.setattr(service,'preview_xapps',changed)
    result=service.conflict_inputs(['saved'],100)
    assert result['reason']=='network_snapshot_changed_during_capture' and not result['groups']
    assert 'decision_frame'not in result
