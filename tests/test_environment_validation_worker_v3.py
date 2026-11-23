"""Concurrency/ownership tests with synthetic cases and fake Popen only."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import sys
from threading import Event, RLock
from types import SimpleNamespace
import time

import pytest

from experiments import environment_validation_worker_v3 as worker
from experiments import environment_validation_worker_v2 as v2
from experiments import environment_validation_worker as v1


def seal(p):
    p=deepcopy(p);p.pop('plan_sha256',None);p['plan_sha256']=worker.digest(p);return p


@pytest.fixture
def plan(tmp_path,monkeypatch):
    scope=tmp_path/'owned';scope.mkdir();monkeypatch.setattr(worker,'SCOPE',scope)
    monkeypatch.setattr(v2,'SCOPE',scope)
    root=scope/'source';root.mkdir()
    case=dict(id='a'*32,name='p_OFF',origin_key='p_s1',scene='p',seed=1,arm='OFF',app_ids=[],
        worker_id='agx1',slots=6440,baseline_slot=2040,origin_observe_slots=2000,observe_slots=4000,
        traffic_load=1.5,traffic_profile=str(scope/'protocol/p.json'),traffic_profile_sha256='2'*64,
        ue_services_config=str(scope/'protocol/service.json'),ue_services_config_sha256='3'*64)
    on={**case,'id':'b'*32,'name':'p_A','arm':'A','app_ids':['xapp:ue_energy_rf']}
    p=seal(dict(schema=worker.SCHEMA,study_id='parallel-test',stage='validation',worker_id='agx1',
        source_root=str(root),source_archive_path=str(scope/'source-v2.tgz'),
        source_sha256=worker.SOURCE_SHA256,source_archive_sha256=worker.ARCHIVE_SHA256,
        frozen_models={f'model/{i}':str(i)*64 for i in range(6)},runner_sha256=worker.sha(worker.__file__),
        runner_path=str(scope/'runner/worker-v3.py'),active_plan_path=str(scope/'plans/parallel-test/validation.json'),
        output_root=str(root/'outputs/environment_validation/parallel-test/validation'),
        max_parallel=2,deadline_utc='2099-01-01T00:00:00+00:00',poll_seconds=.1,cases=[case,on]))
    worker.atomic(p['active_plan_path'],p)
    return p


def test_v1_v2_are_untouched_and_v3_command_only_changes_case_cache(plan):
    assert worker.sha(v1.__file__)=='ac4e7b94a51ba57d76952e55cddb8baa977efaf8a37be66f635554c807e4b410'
    assert worker.sha(v2.__file__)=='f5f65e1c987db290c3719f3ecd25501aae3dcc289dc79fc01abd5415cebe9b80'
    worker.validate_plan(plan,plan['plan_sha256'],plan_path=plan['active_plan_path'])
    case=plan['cases'][1];out=Path(plan['output_root'])
    new=worker.command(plan,case,out)
    old=v2.command({**plan,'runner_path':str(worker.SCOPE/'runner/worker-v2.py')},case,out)
    cache_idx=new.index('--channel-cache-root')+1
    assert new[:cache_idx]==old[:cache_idx] and new[cache_idx+1:]==old[cache_idx+1:]
    assert new[cache_idx].endswith('/'+case['id'])
    assert old[cache_idx].endswith('/'+case['origin_key'])
    assert worker.case_cache_root(plan,case)!=worker.case_cache_root(plan,plan['cases'][0])


@pytest.mark.parametrize('value',[0,4,True,1.5])
def test_parallel_limit_guard(plan,value):
    p=seal({**plan,'max_parallel':value})
    with pytest.raises(ValueError,match='max_parallel'):
        worker.validate_plan(p,p['plan_sha256'])


@pytest.mark.parametrize('value',[1,2,3])
def test_parallel_modes_allowed(plan,value):
    p=seal({**plan,'max_parallel':value})
    worker.validate_plan(p,p['plan_sha256'])


def test_v3_rejects_v2_runner_path_and_schema(plan):
    for change in ({'runner_path':str(worker.SCOPE/'runner/worker-v2.py')},{'schema':v2.SCHEMA}):
        p=seal({**plan,**change})
        with pytest.raises(ValueError):
            worker.validate_plan(p,p['plan_sha256'])


def cases(count=5):
    return [dict(id='OFF1',app_ids=[]),dict(id='A',app_ids=['A']),dict(id='OFF2',app_ids=[])]+[
        dict(id=str(i),app_ids=['A'])for i in range(count)]


@pytest.mark.parametrize('parallel',[2,3])
def test_off_barrier_and_singleton_concurrency_are_bounded(parallel):
    lock=RLock();active=set();done_off=set();peak=0;started=[]
    def run(case):
        nonlocal peak
        with lock:
            if not case['app_ids']:
                assert not active
            else:
                assert done_off=={'OFF1','OFF2'}
            active.add(case['id']);started.append(case['id']);peak=max(peak,len(active))
        time.sleep(.015)
        with lock:
            active.remove(case['id'])
            if not case['app_ids']:done_off.add(case['id'])
        return dict(id=case['id'],phase='complete')
    result=worker.schedule_cases(cases(6),max_parallel=parallel,execute=run,
        admission=lambda *a:dict(allowed=True),stop_reason=lambda:None,stop_event=Event(),
        stop_children=lambda:None,publish=lambda r:None,poll=.001)
    assert not result['failed'] and len(result['done'])==9
    assert started[:2]==['OFF1','OFF2'] and peak==parallel and not active


def test_failure_stops_new_admission_but_sibling_finishes():
    b_started=Event();started=[];stops=[]
    def run(case):
        started.append(case['id'])
        if case['id']=='A':
            assert b_started.wait(1)
            return dict(id='A',phase='failed',error='synthetic failure')
        b_started.set();time.sleep(.025)
        return dict(id=case['id'],phase='complete')
    queue=[dict(id=n,app_ids=['app'])for n in ('A','B','C')]
    result=worker.schedule_cases(queue,max_parallel=2,execute=run,
        admission=lambda *a:dict(allowed=True),stop_reason=lambda:None,stop_event=Event(),
        stop_children=lambda:stops.append(1),publish=lambda r:None,poll=.001)
    assert started==['A','B'] and result['done']==['B']
    assert result['failed'][0]['id']=='A' and result['queued']==['C'] and not stops


def test_ui_admission_only_blocks_new_entries_never_preempts_running_case():
    finished=Event();starts=[];stops=[]
    def run(case):
        starts.append(case['id']);time.sleep(.015);finished.set()
        return dict(id=case['id'],phase='complete')
    def gate(case,active):
        return dict(allowed=not active or finished.is_set(),reason='ui_pending')
    result=worker.schedule_cases([dict(id=n,app_ids=['app'])for n in ('A','B')],
        max_parallel=2,execute=run,admission=gate,stop_reason=lambda:None,stop_event=Event(),
        stop_children=lambda:stops.append(1),publish=lambda r:None,poll=.001)
    assert starts==['A','B'] and result['done']==['A','B'] and not stops


def test_stop_or_deadline_reaps_and_joins_all_owned_tasks_before_return():
    event=Event();begun=Event();finished=Event();stops=[]
    def run(case):
        begun.set();assert event.wait(1);finished.set()
        return dict(id=case['id'],phase='stopped')
    result=worker.schedule_cases([dict(id=n,app_ids=['app'])for n in ('A','B','C')],
        max_parallel=1,execute=run,admission=lambda *a:dict(allowed=True),
        stop_reason=lambda:'deadline' if begun.is_set()else None,stop_event=event,
        stop_children=lambda:stops.append(1),publish=lambda r:None,poll=.001)
    assert event.is_set() and finished.is_set() and stops
    assert len(result['stopped'])==1 and result['queued']==['B','C']


def test_manager_status_exception_stops_and_joins_thread_before_raising():
    event=Event();begun=Event();finished=Event()
    def run(case):
        begun.set();event.wait(1);finished.set();return dict(id='A',phase='stopped')
    def publish(result):
        assert begun.wait(1)
        raise OSError('synthetic manager status failure')
    with pytest.raises(OSError,match='manager status'):
        worker.schedule_cases([dict(id='A',app_ids=['app'])],max_parallel=2,execute=run,
            admission=lambda *a:dict(allowed=True),stop_reason=lambda:None,stop_event=event,
            stop_children=lambda:None,publish=publish,poll=.001)
    assert event.is_set() and finished.is_set()


class FakeChild:
    def __init__(self,pid=123,code=None):self.pid=pid;self.returncode=code
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode


def test_external_filter_excludes_only_registered_roots_and_descendants():
    registry=worker.OwnedChildren(Event());registry.children[100]=FakeChild(100)
    records=[dict(pid=p)for p in (100,200,300,999)]
    result=registry.external(lambda:records,parents={200:100,300:200,999:1})
    assert result==[dict(pid=999)]
    assert worker.is_owned_descendant(300,{100},{300:200,200:100})
    assert not worker.is_owned_descendant(777,{100},{777:888,888:777})


def test_stop_creation_registration_race_keeps_handle_owned(monkeypatch):
    event=Event();registry=worker.OwnedChildren(event);child=FakeChild();stops=[]
    def spawn(*a,**k):event.set();return child
    monkeypatch.setattr(worker.subprocess,'Popen',spawn)
    monkeypatch.setattr(worker,'stop_child',lambda c:(stops.append(c.pid),setattr(c,'returncode',-2)))
    assert registry.start([])is child and registry.children[child.pid]is child
    registry.stop_all();assert stops==[123] and child.poll()==-2
    with pytest.raises(worker.ManagerStopped):registry.start([])


def test_cleanup_attempts_every_owned_child_even_if_first_reap_raises(monkeypatch):
    registry=worker.OwnedChildren(Event());registry.children={i:FakeChild(i)for i in (1,2)};seen=[]
    def stop(child):
        seen.append(child.pid)
        if child.pid==1:raise OSError('first cleanup error')
    monkeypatch.setattr(worker,'stop_child',stop)
    with pytest.raises(OSError):registry.stop_all()
    assert seen==[1,2]


def test_simultaneous_manager_and_case_stop_serialize_signals_for_one_child(monkeypatch):
    registry=worker.OwnedChildren(Event());child=FakeChild();registry.children[child.pid]=child
    signals=[]
    def stop(c):
        if c.poll()is None:
            signals.append(c.pid);time.sleep(.01);c.returncode=-2
    monkeypatch.setattr(worker,'stop_child',stop)
    with ThreadPoolExecutor(max_workers=2)as executor:
        futures=[executor.submit(registry.stop,child)for _ in range(2)]
        for f in futures:f.result()
    assert signals==[123] and child.poll()==-2


def fake_case_setup(plan,monkeypatch,*,fail_running_status=False):
    out=Path(plan['output_root'])
    for name in ('jobs','checkpoints','partial','origins','locks','case_status'):
        (out/name).mkdir(parents=True,exist_ok=True)
    script_dir=Path(plan['source_root'])/'scripts/orin';script_dir.mkdir(parents=True)
    for name in ('run-six-ue-cuda-nr.sh','activate-gpu.sh'):
        (script_dir/name).write_bytes((Path(v1.__file__).parents[1]/'scripts/orin'/name).read_bytes())
    checks=[];children=[]
    monkeypatch.setattr(worker,'validate_owned',lambda *a:checks.append(1))
    monkeypatch.setattr(worker,'validate_installed_modelset',lambda *a:None)
    monkeypatch.setattr(worker,'origin_for',lambda *a:dict(origin_id='same-host-origin'))
    def env(cache):return {'TMPDIR':str(cache/'tmp'),'XDG_CACHE_HOME':str(cache/'.cache')}
    helpers=SimpleNamespace(isolated_environment=env)
    real_atomic=worker.atomic
    def spawn(args,**kwargs):
        run=Path(args[args.index('--output')+1]);run.mkdir(parents=True)
        real_atomic(run/'run_status.json',dict(completed=True,replay=dict(audits_passed=True),source_sha256=plan['source_sha256']))
        real_atomic(run/'evaluation_origin_verified.json',dict(verified_before_activation=True,origin_id='same-host-origin'))
        expected=worker.read(args[5])
        real_atomic(args[4],dict(case_cache_environment=expected,recorded_after_activation=True))
        child=FakeChild(code=None if fail_running_status else 0);children.append(child)
        return child
    monkeypatch.setattr(worker.subprocess,'Popen',spawn)
    if fail_running_status:
        def atomic(path,value):
            if Path(path).parent.name=='case_status' and value.get('phase')=='running':raise OSError('synthetic case status full')
            real_atomic(path,value)
        monkeypatch.setattr(worker,'atomic',atomic)
    return out,helpers,checks,children


def test_parallel_on_requires_completed_and_pinned_shared_model_install(plan):
    root=Path(plan['source_root']);artifact=root/'artifacts/xapps/chongyu/20261117'
    installed=root/'outputs/trained_xapps/20261117'
    models={};pins={}
    for i in range(6):
        path=installed/'models'/('x'+str(i)+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(str(i));pin=worker.sha(path)
        models[str(i)]=dict(path='models/'+path.name,sha256=pin)
        pins[str((artifact/'models'/path.name).relative_to(root))]=pin
    release=dict(models=models)
    worker.atomic(artifact/'release.json',release);worker.atomic(installed/'release.json',release)
    plan['frozen_models']=pins
    worker.validate_installed_modelset(plan)
    (installed/'models/x0.json').write_text('modified install')
    with pytest.raises(ValueError,match='installed frozen model'):
        worker.validate_installed_modelset(plan)


def test_real_shell_bootstrap_restores_only_cache_after_activation_without_a_simulator(plan):
    root=Path(plan['source_root']);case=plan['cases'][1];out=Path(plan['output_root'])
    scripts=root/'scripts/orin';scripts.mkdir(parents=True)
    (scripts/'run-six-ue-cuda-nr.sh').write_bytes((Path(v1.__file__).parents[1]/'scripts/orin/run-six-ue-cuda-nr.sh').read_bytes())
    # A stub activation and a stub Python argv recorder: no Sionna/PHY import.
    (scripts/'activate-gpu.sh').write_text('export VIRTUAL_ENV=activated-stub\nexport PATH="'+str(Path(sys.executable).parent)+':$PATH"\nexport DRJIT_CACHE_DIR=/shared/drjit\nexport CUDA_CACHE_PATH=/shared/cuda\nexport CUPY_CACHE_DIR=/shared/cupy\nexport MPLCONFIGDIR=/shared/mpl\n')
    data=out/'jobs'/case['id']/'data';data.mkdir(parents=True)
    cache=worker.case_cache_root(plan,case)
    env=os.environ.copy()
    for name in ('TMPDIR','XDG_CACHE_HOME','DRJIT_CACHE_DIR','CUDA_CACHE_PATH','CUPY_CACHE_DIR','MPLCONFIGDIR'):
        env[name]=str(cache/name)
    (root/'experiments').mkdir()
    (root/'experiments/run_rt_sys.py').write_text('import json,sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--output")+1]);p.mkdir(parents=True);(p/"stub-argv.json").write_text(json.dumps(sys.argv))\n')
    args,proof=worker.activated_command(plan,case,out,env)
    subprocess.run(args,cwd=root,env=env,check=True,capture_output=True)
    actual=worker.read(proof['effective_environment_path'])
    assert actual['recorded_after_activation'] is True
    assert actual['activation']['VIRTUAL_ENV']=='activated-stub'
    assert actual['case_cache_environment']=={k:env[k]for k in proof['case_cache_environment']}
    recorded=worker.read(data/'run/stub-argv.json')
    assert recorded[1:1+len(worker.SIX_UE_FIXED_ARGV)-1]==worker.SIX_UE_FIXED_ARGV[1:]
    (scripts/'run-six-ue-cuda-nr.sh').write_text('modified launcher')
    with pytest.raises(ValueError,match='launcher bytes'):
        worker.activated_command(plan,case,out,env)


def test_case_has_private_environment_real_wrapper_pin_origin_guard_and_complete_seal(plan,monkeypatch):
    out,helpers,checks,children=fake_case_setup(plan,monkeypatch)
    event=Event();registry=worker.OwnedChildren(event);case=plan['cases'][1]
    result=worker.execute_case(plan,case,out,helpers,registry,event,{},RLock())
    assert result['phase']=='complete' and len(checks)==2 and not registry.children
    assert worker.completed(out,case,plan)
    pin=worker.read(out/'checkpoints'/(case['id']+'.json'))
    assert pin['runner_sha256']==worker.sha(worker.__file__) and pin['origin_id']=='same-host-origin'
    command=worker.read(out/'jobs'/case['id']/'data/command.json')
    assert all(case['id']in v for v in command['cache_environment'].values())
    raw=out/'jobs'/case['id']/'data/run/run_status.json';raw.write_text('changed')
    assert not worker.completed(out,case,plan)


def test_case_status_exception_after_spawn_cleans_only_local_child(plan,monkeypatch):
    out,helpers,checks,children=fake_case_setup(plan,monkeypatch,fail_running_status=True)
    stopped=[]
    def stop(child):stopped.append(child.pid);child.returncode=-2
    monkeypatch.setattr(worker,'stop_child',stop)
    event=Event();registry=worker.OwnedChildren(event)
    result=worker.execute_case(plan,plan['cases'][1],out,helpers,registry,event,{},RLock())
    assert result['phase']=='failed' and stopped==[123] and not registry.children, result
    assert not (out/'checkpoints'/(plan['cases'][1]['id']+'.json')).exists()
