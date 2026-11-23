"""Deployment admission/orchestration tests. No fake measurements claim gain."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import pytest
from smo.__main__ import GraphService,handler
from smo.deployment import DeploymentManager
from smo.deployment_worker import plan
from smo.intent_language import IntentError
from test_smo_intent_comparison import setup


def config(tmp_path):
    p=tmp_path/'worker.json'
    p.write_text(json.dumps(dict(host='user@100.0.0.1',socket='/tmp/owned.sock',project_root='/dev/shm/owned/source',python='/owned/venv/bin/python',local_output=str(tmp_path/'jobs'),seed=1,label='Fixture worker')))
    return p


def test_worker_plans_frozen_hot_controllers_and_no_hypothetical_values(tmp_path):
    req=dict(id='a'*32,seed=1,app_ids=['xapp:ue_energy_rf','xapp:mec_offload_budget_qgbdt'],observe_slots=500)
    job,run,baseline,total,cmd=plan(req,tmp_path)
    assert baseline==520 and baseline-500>=1 and total==1420
    assert '--live-xapps'in cmd and cmd[cmd.index('--xapps-start')+1]==''
    assert '--xapp-toggle-timeline'in cmd and '--xapp-decision-preview'in cmd
    assert cmd[cmd.index('--xapp-preview-before-slot')+1]==str(baseline)
    assert '--energy-xapp-fixed-action'not in cmd and '--perturbation-spec'not in cmd
    assert run.parent==job and run.is_relative_to(tmp_path/'outputs')
    for overrides in ({'id':'../escape'},{'app_ids':['xapp:unknown']},{'observe_slots':True},{'seed':-1}):
        with pytest.raises(ValueError):plan({**req,**overrides},tmp_path)


def test_disabled_backend_and_busy_worker_do_not_silently_submit(tmp_path,monkeypatch):
    with pytest.raises(IntentError):DeploymentManager().start(records=[],group={},observe_slots=500,context={})
    monkeypatch.setattr(DeploymentManager,'watch',lambda *a,**k:None)
    m=DeploymentManager(config(tmp_path));records=[dict(id='intent',ranges=[])];group=dict(app_ids=['xapp:ue_energy_rf'],selected_names=['Reduce UE power'])
    original=deepcopy((records,group));row=m.start(records=records,group=group,observe_slots=500,context={'fixture':True})
    assert row['phase']=='queued' and not row['automatic_selection']
    assert (records,group)==original
    with pytest.raises(IntentError,match='正在运行'):m.start(records=records,group=group,observe_slots=500,context={})
    m.jobs[row['id']]['phase']='connection_lost'
    with pytest.raises(IntentError,match='正在运行'):m.start(records=records,group=group,observe_slots=500,context={})
    assert DeploymentManager(config(tmp_path)).status()['latest']['id']==row['id']


def test_stop_is_scoped_to_an_owned_job(tmp_path,monkeypatch):
    monkeypatch.setattr(DeploymentManager,'watch',lambda *a,**k:None)
    m=DeploymentManager(config(tmp_path));calls=[];monkeypatch.setattr(m,'ssh',lambda code,**k:calls.append(code))
    with pytest.raises(IntentError):m.cancel('../../another-job')
    row=m.start(records=[],group=dict(app_ids=['xapp:ue_energy_rf'],selected_names=['UE']),observe_slots=500,context={})
    m.cancel(row['id'])
    assert m.status()['latest']['stop_requested']
    assert len(calls)==1 and '/'+row['id']+'/STOP'in calls[0]
    assert 'kill'not in calls[0] # marker only, never arbitrary remote process kill


def test_archive_traversal_and_symlinks_rejected(tmp_path,monkeypatch):
    m=DeploymentManager(config(tmp_path));ident='a'*32;(m.output/ident).mkdir();m.jobs[ident]={}
    for name,symlink in [('../outside',False),('data/link',True)]:
        b=io.BytesIO()
        with tarfile.open(fileobj=b,mode='w:gz')as t:
            entry=tarfile.TarInfo(name)
            if symlink:entry.type=tarfile.SYMTYPE;entry.linkname='/etc/passwd'
            t.addfile(entry)
        monkeypatch.setattr(m,'ssh',lambda *a,**k:b.getvalue())
        with pytest.raises(ValueError,match='Invalid deployment archive'):m.download(ident,Path('/owned'))
    assert not (tmp_path/'outside').exists()


class Runner:
    source=None;run=None
    def __init__(self):self.calls=[]
    def status(self):return dict(enabled=True,latest=None)
    def start(self,**kwargs):self.calls.append(deepcopy(kwargs));return dict(id='a'*32,phase='queued')


def service_fixture():
    _,_,_,source,records=setup();s=GraphService();s.source=source
    s.intent_ledger.records={r['id']:r for r in records};s.deployments=Runner()
    s.evaluation_origin=dict(origin_id='fixture-origin',baseline_slot=600,observe_slots=500)
    source['frames'][0]['evaluation_origin']=deepcopy(s.evaluation_origin)
    review=s.review_intents(['test-intent'],600)
    return s,review['combination_catalog']['groups'][0]


def test_service_recomputes_all_target_coverage_never_accepts_supplied_actions():
    s,g=service_fixture();original=deepcopy(s.intent_ledger.records)
    job=s.deploy_combination(['test-intent'],g['id'],600,500)
    assert job['phase']=='queued' and len(s.deployments.calls)==1
    call=s.deployments.calls[0]
    assert call['group']['app_ids']==g['app_ids'] and call['observe_slots']==500
    assert call['records'][0]['ranges']==original['test-intent']['ranges']
    with pytest.raises(IntentError):s.deploy_combination(['test-intent'],'graph-group:xapp:unregistered',600)
    with pytest.raises(IntentError):s.deploy_combination([],g['id'],600)
    assert len(s.deployments.calls)==1 and s.intent_ledger.records==original


def test_http_rejects_injected_commands_and_foreign_origins():
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    s,g=service_fixture();server=ThreadingHTTPServer(('127.0.0.1',0),handler(s));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    def post(data,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        return json.load(urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/deployment/start',data=json.dumps(data).encode(),headers=headers)))
    data=dict(active_intent_ids=['test-intent'],group_id=g['id'],slot=600,observe_slots=500)
    try:
        for key in ('command','host','model_path','rcp_values','app_ids'):
            with pytest.raises(HTTPError)as e:post({**data,key:'untrusted'})
            assert e.value.code==400
        with pytest.raises(HTTPError)as e:post(data,'https://untrusted.example')
        assert e.value.code==403 and not s.deployments.calls
        assert post(data)['phase']=='queued' and len(s.deployments.calls)==1
    finally:server.shutdown();server.server_close();thread.join()
