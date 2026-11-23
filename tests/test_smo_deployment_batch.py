"""Queue coverage/checkpoint fixtures; no synthetic data are claimed as benefits."""
from copy import deepcopy
import csv
import io
import itertools
import json
import time
import pytest
from smo.deployment import DeploymentManager
from smo.intent_language import IntentError
from test_smo_deployment import config,service_fixture
from test_smo_intent_comparison import setup,review
from smo.runtime import APP_LOGS


def wait_until(condition):
    end=time.monotonic()+5
    while time.monotonic()<end:
        if condition():return
        time.sleep(.002)
    raise AssertionError('Queue did not reach expected state')


def groups(n=48):
    ids=['xapp:'+i for i in APP_LOGS];required={'xapp:ue_energy_rf','xapp:mec_offload_budget_qgbdt'}
    result=[]
    for size in range(1,len(ids)+1):
        for subset in itertools.combinations(ids,size):
            if set(subset)&required:result.append(dict(id='graph-group:'+'|'.join(subset),app_ids=list(subset),selected_names=list(subset),conflict_summaries=[]))
    assert len(result)==48
    return result[:n]


def fixture_comparison():
    store,ident,graph,source,_=setup()
    return review(store,ident,graph,source,source['frames'][3])


def manager(tmp_path,monkeypatch):
    monkeypatch.setattr('smo.deployment_batch.POLL_SECONDS',.001)
    monkeypatch.setattr(DeploymentManager,'watch',lambda *a,**k:None)
    return DeploymentManager(config(tmp_path))


def start(m,rows):
    _,_,_,_,records=setup()
    return m.batch.start(records=records,groups=rows,observe_slots=500,context={'fixture':True},snapshot={'slot':600},ranges=records[0]['ranges'],show_dashed=False)


def complete(m,phase='complete'):
    with m.lock:
        row=m.jobs[m.active]
        row.update(phase=phase,data_saved=True)
        if phase=='complete':row['comparison']=fixture_comparison()
        else:row['error']='Explicit fixture execution failure'
        m.persist(row)


def test_all_48_catalog_members_run_once_sequentially_without_selection(tmp_path,monkeypatch):
    m=manager(tmp_path,monkeypatch);catalog=groups();original=deepcopy(catalog);b=start(m,catalog);seen=set()
    while len(seen)<48:
        wait_until(lambda:m.active not in seen and m.active is not None)
        with m.lock:
            assert sum(j['phase']not in('complete','failed','cancelled')for j in m.jobs.values())==1
            seen.add(m.active)
        complete(m)
    wait_until(lambda:m.batch.status()['phase']=='complete')
    result=m.batch.status()
    assert result['total']==48 and result['counts']['complete']==48 and len(m.jobs)==48
    assert [r['app_ids']for r in result['members']]==[r['app_ids']for r in original]
    assert all(j['seed']==1 and j['observe_slots']==500 for j in m.jobs.values())
    assert all(len(r['attempts'])==1 for r in result['members'])
    assert not result['ranking_performed']and not result['automatic_selection']and not result['conflict_quantification_performed']
    assert not result['controlled_comparison']['risk_inputs_used']
    assert result['controlled_comparison']['purpose']=='independent_deployment_before_after_validation'
    assert catalog==original and m.source is None # never auto-view/choose last member
    raw,mime=m.batch.export(b['id'],'csv');table=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
    assert len(table)==1+48*2 and mime=='text/csv'
    assert set(r[2]for r in table[1:])=={r['label']for r in result['ranges']}
    assert 'Result score after'in table[0] and 'Common KPI scale'in table[0]
    assert float(table[1][table[0].index('Result score after')])==pytest.approx((100*.5/6+100*.3/2)/2)
    assert table[1][table[0].index('Origin alignment')]=='legacy_unaligned'
    archive=json.loads((m.output/'batches'/b['id']/'batch.json').read_text())
    # The common-batch derived score is not retroactively written into raw records.
    assert 'result_evaluation_method'not in archive
    assert 'controlled_comparison'not in archive


def test_failure_keeps_evidence_continues_queue_and_retry_preserves_successes(tmp_path,monkeypatch):
    m=manager(tmp_path,monkeypatch);b=start(m,groups(3));wait_until(lambda:len(m.jobs)==1);failed=m.active;complete(m,'failed')
    wait_until(lambda:len(m.jobs)==2);good=m.active;complete(m)
    wait_until(lambda:len(m.jobs)==3);complete(m)
    wait_until(lambda:m.batch.status()['phase']=='complete_with_failures')
    r=m.batch.status();assert r['counts']['complete']==2 and r['counts']['failed']==1
    raw,_=m.batch.export(b['id'],'csv');rows=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
    assert all(row[8:11]==['','','']for row in rows[1:3]) # no fabricated zero KPI
    m.batch.resume(b['id']);wait_until(lambda:len(m.jobs)==4);complete(m)
    wait_until(lambda:m.batch.status()['phase']=='complete')
    assert m.jobs[failed]['phase']=='failed'and m.jobs[good]['phase']=='complete'
    assert len(m.batch.status()['members'][0]['attempts'])==2
    assert len(m.batch.status()['members'][1]['attempts'])==1


def test_pause_reload_and_resume_keep_frozen_catalog_completed_data_and_attempts(tmp_path,monkeypatch):
    m=manager(tmp_path,monkeypatch);b=start(m,groups(3));wait_until(lambda:len(m.jobs)==1);first=m.active;complete(m)
    wait_until(lambda:len(m.jobs)==2);cancelled=m.active
    monkeypatch.setattr(m,'ssh',lambda *a,**k:b'')
    m.batch.pause(b['id']);complete(m,'cancelled')
    wait_until(lambda:m.batch.status()['phase']=='paused'and m.batch.thread is None)
    saved=m.batch.status();m2=DeploymentManager(config(tmp_path))
    assert m2.batch.status()['members']==saved['members']
    m2.batch.resume(b['id']);wait_until(lambda:len(m2.jobs)==3);complete(m2)
    wait_until(lambda:len(m2.jobs)==4);complete(m2)
    wait_until(lambda:m2.batch.status()['phase']=='complete')
    r=m2.batch.status();assert r['records']==saved['records']and r['catalog_sha256']==saved['catalog_sha256']
    assert r['members'][0]['job_id']==first and r['members'][1]['attempts'][0]==cancelled
    assert len(r['members'][1]['attempts'])==2


def test_connection_loss_blocks_new_jobs_and_does_not_skip_current_member(tmp_path,monkeypatch):
    m=manager(tmp_path,monkeypatch);start(m,groups(2));wait_until(lambda:len(m.jobs)==1)
    with m.lock:m.jobs[m.active]['phase']='connection_lost'
    wait_until(lambda:m.batch.status()['phase']=='connection_lost')
    with pytest.raises(IntentError):m.start(records=[],group=groups(1)[0],observe_slots=500,context={})
    assert len(m.jobs)==1
    complete(m);wait_until(lambda:len(m.jobs)==2);complete(m)
    wait_until(lambda:m.batch.status()['phase']=='complete')


def test_server_enumerates_entire_catalog_and_http_cannot_supply_subset():
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    from smo.__main__ import handler
    s,_=service_fixture()
    class Batch:
        calls=[]
        def start(self,**kwargs):self.calls.append(deepcopy(kwargs));return {'id':'fixture','total':len(kwargs['groups'])}
    s.deployments.batch=Batch()
    expected=s.review_intents(['test-intent'],600)['combination_catalog']['groups']
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(s));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    data=dict(active_intent_ids=['test-intent'],slot=600,observe_slots=500,show_dashed=False)
    def post(payload):return json.load(urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/deployment/batch/start',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})))
    try:
        for key in ('group_id','groups','app_ids','actions','ranking'):
            with pytest.raises(HTTPError)as error:post({**data,key:[]})
            assert error.value.code==400
        result=post(data)
        assert result['total']==len(expected)
        assert [g['app_ids']for g in s.deployments.batch.calls[0]['groups']]==[g['app_ids']for g in expected]
        assert len(s.deployments.batch.calls)==1
    finally:server.shutdown();server.server_close();thread.join()


def test_worker_source_hash_is_frozen_across_checkpoint_reload(tmp_path,monkeypatch):
    import hashlib
    p=tmp_path/'source.tgz';p.write_bytes(b'Fixture source archive version one')
    m=manager(tmp_path,monkeypatch);b=start(m,groups(1));wait_until(lambda:len(m.jobs)==1)
    expected=hashlib.sha256(p.read_bytes()).hexdigest()
    assert m.jobs[m.active]['expected_source_archive_sha256']==expected
    complete(m,'failed');wait_until(lambda:m.batch.status()['phase']=='complete_with_failures')
    p.write_bytes(b'Changed fixture archive must not silently rebase the old batch')
    restored=DeploymentManager(config(tmp_path));restored.batch.resume(b['id']);wait_until(lambda:len(restored.jobs)==2)
    assert restored.source_archive_sha256!=expected
    assert restored.jobs[restored.active]['expected_source_archive_sha256']==expected
    complete(restored);wait_until(lambda:restored.batch.status()['phase']=='complete')


def test_reload_recovers_admitted_child_before_queue_reference_was_saved(tmp_path,monkeypatch):
    from smo.deployment_batch import BatchEvaluator
    original_launch=BatchEvaluator.launch
    monkeypatch.setattr(BatchEvaluator,'launch',lambda self:None)
    m=manager(tmp_path,monkeypatch);b=start(m,groups(2))
    first=groups(1)[0]
    child=m.start(records=b['records'],group=first,observe_slots=500,context=b['context'],batch_id=b['id'],batch_index=0)
    complete(m)
    assert not m.batch.status()['members'][0]['attempts'] # crash before parent update
    monkeypatch.setattr(BatchEvaluator,'launch',original_launch)
    restored=DeploymentManager(config(tmp_path))
    wait_until(lambda:len(restored.jobs)==2)
    complete(restored)
    wait_until(lambda:restored.batch.status()['phase']=='complete')
    assert restored.batch.status()['members'][0]['job_id']==child['id']
    assert restored.batch.status()['members'][0]['attempts']==[child['id']]
    assert len(restored.jobs)==2
