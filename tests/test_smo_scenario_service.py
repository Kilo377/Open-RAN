"""Scene routes must never mix gains, frozen intents or deployment contexts."""
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import pytest

from smo.__main__ import GraphService
from smo.scenario_service import prepare,router,retire_service


def test_same_origin_routes_keep_scene_intents_and_sources_separate():
    native=GraphService();burst=GraphService(language=native.language)
    for service,name in [(native,'native'),(burst,'burst')]:
        service.source['name']=name
        service.scenario_view=dict(current=name)
    native.intent_ledger.records['native-only']=dict(id='native-only',text='saved',ranges=[],snapshot={})
    server=ThreadingHTTPServer(('127.0.0.1',0),router({'native':native,'burst':burst},'native'))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
    def post(path,data,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        return urlopen(Request(base+path,data=json.dumps(data).encode(),headers=headers))
    try:
        assert json.load(urlopen(base+'/api/graph'))['source']['name']=='native'
        assert json.load(urlopen(base+'/contexts/burst/api/graph'))['source']['name']=='burst'
        for status in [json.load(urlopen(base+'/contexts/burst/api/intent/status')),
                       json.load(post('/contexts/burst/api/intent/session-key',dict(api_key='fixture-only-unused-key'))),
                       json.load(post('/contexts/burst/api/intent/clear-key',{}))]:
            assert status['conflict_input_schema']=='oran.trusted-control-chain-inputs.v1'
        assert b'id="scenario-select"' in urlopen(base+'/contexts/burst/').read()
        assert json.load(post('/contexts/native/api/intent/review',dict(active_intent_ids=['native-only'])))['active_intents'][0]['id']=='native-only'
        with pytest.raises(HTTPError)as error:post('/contexts/burst/api/intent/review',dict(active_intent_ids=['native-only']))
        assert error.value.code==400
        with pytest.raises(HTTPError)as error:urlopen(base+'/contexts/missing/api/graph')
        assert error.value.code==404
        with pytest.raises(HTTPError)as error:post('/contexts/burst/api/intent/clear-key',{},'https://unrelated.example')
        assert error.value.code==403
    finally:server.shutdown();server.server_close();thread.join()


def test_manifest_rejects_cross_scene_gain_file_and_native_pool_in_pressure_view(tmp_path):
    path=tmp_path/'influences.json';path.write_text(json.dumps(dict(scenario=dict(id='burst_2x'))))
    manifest=dict(default='native_1x',scenarios=[dict(id='native_1x',label='Native',influences=str(path))])
    with pytest.raises(ValueError,match='borrow'):prepare(manifest)
    manifest=dict(default='burst_2x',scenarios=[dict(id='burst_2x',label='Burst',influences=str(path),deployment_config='native-pool.json')])
    with pytest.raises(ValueError,match='native deployment pool'):prepare(manifest)


def test_updated_graph_catalog_is_not_masked_by_a_retained_smaller_batch():
    node=shutil.which('node')
    if not node:pytest.skip('Node required for production catalog helper')
    code="""
const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const b={members:[{app_ids:['a']},{app_ids:['a','b']}]};
const same={groups:[{app_ids:['b','a']},{app_ids:['a']}]};
const newer={groups:[...same.groups,{app_ids:['b']}]};
process.stdout.write(JSON.stringify([c.SmoGraph.batchCatalogCompatible(b,same),c.SmoGraph.batchCatalogCompatible(b,newer)]));
"""
    assert json.loads(subprocess.check_output([node,'-e',code],cwd=Path(__file__).resolve().parents[1]))==[True,False]


def test_initial_selection_counts_distinguish_primary_reserve_and_old_method():
    node=shutil.which('node')
    if not node:pytest.skip('Node required for production catalog helper')
    code="""
const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const s={method:'influence_coverage_v1',config:{coverage_threshold:.05},input_group_count:62,primary_group_count:16,reserve_group_count:46};
const v2={...s,method:'semantic_influence_coverage_v2',primary_group_count:8,reserve_group_count:54};
process.stdout.write(JSON.stringify([c.SmoGraph.initialSelectionSummary({}),c.SmoGraph.initialSelectionSummary({initial_selection:s}),c.SmoGraph.initialSelectionSummary({initial_selection:v2})]));
"""
    old,new,semantic=json.loads(subprocess.check_output([node,'-e',code],cwd=Path(__file__).resolve().parents[1]))
    assert old is None
    assert (new['primary'],new['reserve'],new['total'])==(16,46,62)
    assert '门槛 5%' in new['text'] and '仅评估主候选' in new['text']
    assert (semantic['primary'],semantic['reserve'],semantic['total'])==(8,54,62)
    assert '能力方向＋统计影响' in semantic['text']


def test_retained_outcomes_require_same_intent_bounds_scope_and_contracts():
    node=shutil.which('node')
    if not node:pytest.skip('Node required for production catalog helper')
    code="""
const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const range={kpi:'power',lower:0,upper:9,lower_inclusive:true,upper_inclusive:false};
const graph={metric_scope:'population',contract_sha256:{model:'a'}},catalog={ranges:[range]};
const batch={context:{metric_scope:'population',contracts:{model:'a'}},active_intent_ids:['i'],ranges:[range]};
const f=(b=batch,g=graph,ids=['i'])=>c.SmoGraph.retainedEvidenceCompatible(b,catalog,g,ids);
process.stdout.write(JSON.stringify([f(),f({...batch,ranges:[{...range,upper:10}]}),f(batch,{...graph,contract_sha256:{model:'b'}}),f(batch,graph,['other']),f({...batch,context:{...batch.context,metric_scope:'legacy'}})]));
"""
    assert json.loads(subprocess.check_output([node,'-e',code],cwd=Path(__file__).resolve().parents[1]))==[True,False,False,False,False]


def test_staged_preflight_does_not_start_deployment_managers_or_attach_jobs(tmp_path,monkeypatch):
    matrix=tmp_path/'influences.json';matrix.write_text(json.dumps(dict(scenario=dict(id='native_1x'))))
    calls=[]
    def fake_service(**kw):
        calls.append(kw)
        return SimpleNamespace(graph=dict(influence_input=dict(sha256='a'*64)),
            deployments=SimpleNamespace(view=lambda *a:pytest.fail('Staged preflight cannot attach or resume jobs')))
    monkeypatch.setattr('smo.scenario_service.GraphService',fake_service)
    services,_=prepare(dict(default='native_1x',scenarios=[dict(id='native_1x',label='Native',replay='fixture',influences=str(matrix),deployment_config='existing-pool',retained_job_id='saved')]),staged_read_only=True)
    assert calls[0]['deployment_config']is None
    assert not services['native_1x'].scenario_view['manual_evaluation_enabled']


def test_language_preflight_failure_keeps_the_old_service_alive(tmp_path,monkeypatch):
    from smo import scenario_service
    from smo.intent_language import LlmUnavailable
    class Language:
        def configure(self,*args):pass
        def parse(self,*args,**kwargs):raise LlmUnavailable('Fixture connection unavailable')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}');stages=[]
    def staged(manifest,**kw):
        stages.append(kw['staged_read_only'])
        return {'native':SimpleNamespace(graph={},intent_context=lambda:({},{}))},'native'
    monkeypatch.setattr(scenario_service,'IntentLanguage',Language)
    monkeypatch.setattr(scenario_service,'prepare',staged)
    monkeypatch.setattr(scenario_service,'retire_service',lambda *a:pytest.fail('Old service retired before credentials verified'))
    monkeypatch.setattr(sys,'argv',['scenario_service','--manifest',str(manifest),'--verify-language','--replace-pid','999999','--session-key-stdin'])
    monkeypatch.setattr(sys,'stdin',io.StringIO('fixture-unused-key\n'))
    with pytest.raises(SystemExit)as error:scenario_service.main()
    assert error.value.code==1 and stages==[True]


@pytest.mark.parametrize('command',['python -m unrelated --port 53505','python -m smo --port 53506'])
def test_retirement_rejects_an_unrelated_process_or_wrong_port(monkeypatch,command):
    monkeypatch.setattr('smo.scenario_service.subprocess.check_output',lambda *a,**k:command)
    monkeypatch.setattr('smo.scenario_service.os.kill',lambda *a:pytest.fail('Unrelated process signalled'))
    with pytest.raises(ValueError,match='not the SMO'):retire_service(999999,53505)
