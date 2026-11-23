"""Two-worker scheduling and recovery fixtures; no simulated gain claims."""
from copy import deepcopy
import json
import shutil
import pytest
from smo.deployment import DeploymentManager
from smo.intent_language import IntentError
from test_smo_deployment_batch import groups,start,wait_until,fixture_comparison


def pool_config(tmp_path,output=None):
    p=tmp_path/'pool.json'
    workers=[dict(id=f'agx{i}',host=f'user@100.0.0.{i}',socket=f'/tmp/owned-{i}.sock',project_root=f'/dev/shm/worker{i}/source',python=f'/venv{i}/bin/python',label=f'AGX {i}')for i in(2,1)]
    p.write_text(json.dumps(dict(workers=workers,local_output=str(output or tmp_path/'jobs'),seed=1,label='Two AGX workers')))
    return p


def manager(tmp_path,monkeypatch):
    monkeypatch.setattr('smo.deployment_batch.POLL_SECONDS',.001)
    monkeypatch.setattr(DeploymentManager,'watch',lambda *a,**k:None)
    return DeploymentManager(pool_config(tmp_path))


def finish(m,ident,phase='complete'):
    with m.lock:
        j=m.jobs[ident];j.update(phase=phase,data_saved=True)
        if phase=='complete':j['comparison']=fixture_comparison()
        m.persist(j)


def test_two_workers_fill_slots_and_healthy_worker_continues_after_other_disconnects(tmp_path,monkeypatch):
    m=manager(tmp_path,monkeypatch);b=start(m,groups(4));wait_until(lambda:len(m.jobs)==2)
    assert len(m.busy_jobs())==2 and len({j['worker_id']for j in m.busy_jobs()})==2
    with m.lock:
        stalled=next(j['id']for j in m.jobs.values()if j['worker_id']=='agx1')
        healthy=next(j['id']for j in m.jobs.values()if j['worker_id']=='agx2')
        m.jobs[stalled]['phase']='connection_lost'
    finish(m,healthy);wait_until(lambda:len(m.jobs)==3)
    third=next(j['id']for j in m.busy_jobs()if j['worker_id']=='agx2');finish(m,third)
    wait_until(lambda:len(m.jobs)==4)
    fourth=next(j['id']for j in m.busy_jobs()if j['worker_id']=='agx2');finish(m,fourth)
    wait_until(lambda:m.batch.status()['phase']=='connection_lost')
    assert m.batch.status()['counts']['complete']==3 and len(m.jobs)==4
    assert m.jobs[stalled]['phase']=='connection_lost'
    finish(m,stalled);wait_until(lambda:m.batch.status()['phase']=='complete')
    assert len(m.jobs)==4 and all(len(x['attempts'])==1 for x in m.batch.status()['members'])
    assert m.batch.status()['concurrency']==2 and not m.batch.status()['ranking_performed']


def test_pause_cancels_both_workers_and_resume_keeps_all_attempts(tmp_path,monkeypatch):
    m=manager(tmp_path,monkeypatch);b=start(m,groups(3));wait_until(lambda:len(m.jobs)==2)
    calls=[];monkeypatch.setattr(m,'ssh',lambda code,**kwargs:calls.append(kwargs['worker_id']))
    running=[j['id']for j in m.busy_jobs()];m.batch.pause(b['id'])
    assert set(calls)=={'agx1','agx2'}
    for ident in running:finish(m,ident,'cancelled')
    wait_until(lambda:m.batch.status()['phase']=='paused'and m.batch.thread is None)
    m2=DeploymentManager(pool_config(tmp_path));m2.batch.resume(b['id']);wait_until(lambda:len(m2.jobs)==4)
    for ident in list(j['id']for j in m2.busy_jobs()):finish(m2,ident)
    wait_until(lambda:len(m2.jobs)==5)
    for j in list(m2.busy_jobs()):finish(m2,j['id'])
    wait_until(lambda:m2.batch.status()['phase']=='complete')
    assert len(m2.jobs)==5 and all(m2.jobs[i]['phase']=='cancelled'for i in running)
    assert [len(x['attempts'])for x in m2.batch.status()['members']]==[2,2,1]


def test_recovery_watches_both_inflight_jobs_without_duplicate_dispatch(tmp_path,monkeypatch):
    m=manager(tmp_path,monkeypatch);start(m,groups(2));wait_until(lambda:len(m.jobs)==2)
    copied=tmp_path/'recovered';shutil.copytree(m.output,copied)
    watched=[];monkeypatch.setattr(DeploymentManager,'watch',lambda self,ident,launch=False:watched.append((ident,launch)))
    recovered=DeploymentManager(pool_config(tmp_path,copied))
    assert set(i for i,_ in watched)==set(m.jobs)and not any(launch for _,launch in watched)
    assert len(recovered.jobs)==2
    for ident in list(recovered.jobs):finish(recovered,ident)
    wait_until(lambda:recovered.batch.status()['phase']=='complete')
    assert len(recovered.jobs)==2
    for ident in list(m.jobs):finish(m,ident)
    wait_until(lambda:m.batch.status()['phase']=='complete')


def test_one_worker_never_accepts_two_concurrent_jobs_and_ssh_uses_assigned_host(tmp_path,monkeypatch):
    m=manager(tmp_path,monkeypatch)
    job=m.start(records=[],group=groups(1)[0],observe_slots=200,context={},worker_id='agx1')
    with pytest.raises(IntentError):m.start(records=[],group=groups(1)[0],observe_slots=200,context={},worker_id='agx1')
    m.start(records=[],group=groups(1)[0],observe_slots=200,context={},worker_id='agx2')
    assert len(m.busy_jobs())==2
    seen=[]
    class Reply:stdout=b'fixture'
    monkeypatch.setattr('smo.deployment.subprocess.run',lambda cmd,**kwargs:(seen.append(cmd)or Reply()))
    m.ssh('print(1)',worker_id='agx1');m.ssh('print(1)',worker_id='agx2')
    assert 'user@100.0.0.1'in seen[0]and 'user@100.0.0.2'in seen[1]


def test_duplicate_physical_host_cannot_be_configured_as_two_slots(tmp_path):
    p=pool_config(tmp_path);c=json.loads(p.read_text());c['workers'][1]['host']=c['workers'][0]['host'];p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='One slot per SSH host'):DeploymentManager(p)


def test_reconnect_at_watcher_exit_is_not_lost(tmp_path,monkeypatch):
    from threading import Event
    p=pool_config(tmp_path);m=DeploymentManager(p)
    ident='a'*32;m.jobs[ident]=dict(id=ident,phase='queued',worker_id='agx1');(m.output/ident).mkdir()
    blocked=Event();release=Event();calls=[]
    def watch(ident,launch):
        calls.append(ident)
        if len(calls)==1:
            m.jobs[ident]['phase']='connection_lost';blocked.set();release.wait(2)
        else:m.jobs[ident]['phase']='complete'
    monkeypatch.setattr(m,'_watch',watch)
    m.watch(ident);assert blocked.wait(2)
    m.reconnect();assert len(calls)==1
    release.set();wait_until(lambda:len(calls)==2 and m.jobs[ident]['phase']=='complete')


def test_paused_single_worker_batch_resumes_on_two_workers_without_rerunning_success(tmp_path,monkeypatch):
    from test_smo_deployment_batch import manager as single_manager
    m=single_manager(tmp_path,monkeypatch);b=start(m,groups(3));wait_until(lambda:len(m.jobs)==1)
    first=m.active;finish(m,first);wait_until(lambda:len(m.jobs)==2)
    second=m.active;monkeypatch.setattr(m,'ssh',lambda *a,**k:b'');m.batch.pause(b['id']);finish(m,second,'cancelled')
    wait_until(lambda:m.batch.status()['phase']=='paused'and m.batch.thread is None)
    upgraded=DeploymentManager(pool_config(tmp_path));upgraded.batch.resume(b['id'])
    wait_until(lambda:len(upgraded.jobs)==4)
    assert {j['worker_id']for j in upgraded.busy_jobs()}=={'agx1','agx2'}
    for j in list(upgraded.busy_jobs()):finish(upgraded,j['id'])
    wait_until(lambda:upgraded.batch.status()['phase']=='complete')
    assert upgraded.batch.status()['members'][0]['job_id']==first
    assert len(upgraded.batch.status()['members'][0]['attempts'])==1
