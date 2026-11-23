"""Synthetic protocol, command and admission checks; no physical simulations."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import signal
from types import SimpleNamespace

import pytest

from experiments import environment_validation_worker as worker


def seal(plan):
    plan = deepcopy(plan)
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = worker.digest(plan)
    return plan


@pytest.fixture
def plan(tmp_path, monkeypatch):
    scope = tmp_path/'owned'
    scope.mkdir()
    monkeypatch.setattr(worker, 'SCOPE', scope)
    root = scope/'source'
    root.mkdir()
    profile = scope/'protocol/traffic.json'
    profile.parent.mkdir()
    profile.write_text('{"ues":{}}')
    service = root/'configs/service.json'
    service.parent.mkdir()
    service.write_text('{}')
    case = dict(id='a'*32, name='pressure_s1_OFF', origin_key='pressure_s1', scene='pressure',
        seed=1, arm='OFF', app_ids=[], worker_id='agx1', slots=6440, baseline_slot=2040,
        origin_observe_slots=2000, observe_slots=4000, traffic_load=1.25,
        traffic_profile=str(profile), traffic_profile_sha256=worker.sha(profile),
        ue_services_config=str(service), ue_services_config_sha256=worker.sha(service))
    on = {**case, 'id':'b'*32, 'name':'pressure_s1_A', 'arm':'A', 'app_ids':['xapp:ue_energy_rf']}
    return seal(dict(schema=worker.SCHEMA, study_id='environment-test', stage='validation',
        worker_id='agx1', source_root=str(root), source_archive_path=str(scope/'source-v2.tgz'),
        source_archive_sha256=worker.ARCHIVE_SHA256, source_sha256=worker.SOURCE_SHA256,
        frozen_models={f'model/{i}.bin':str(i)*64 for i in range(6)},
        runner_sha256='1'*64, deadline_utc='2099-01-01T00:00:00+00:00', cases=[case, on]))


def validate(plan):
    return worker.validate_plan(plan, plan['plan_sha256'])


def test_explicit_long_after_window_keeps_short_origin_before_window(plan):
    assert validate(plan) is plan
    out = Path(plan['source_root'])/'outputs/environment_validation/validation'
    args = worker.command(plan, plan['cases'][1], out)
    assert args[args.index('--slots')+1] == '6440'
    assert args[args.index('--xapp-preview-before-slot')+1] == '2040'
    assert args[args.index('--traffic-load')+1] == '1.25'
    assert args[args.index('--traffic-profile')+1] == plan['cases'][1]['traffic_profile']
    assert args[args.index('--ue-services-config')+1] == plan['cases'][1]['ue_services_config']
    assert '--urllc-scenario' not in args
    assert '--perturbation-spec' not in args and '--fixed-control-timeline' not in args
    assert '--evaluation-origin-file' in args and '--xapp-decision-preview' in args
    assert Path(args[args.index('--output')+1]).is_relative_to(Path(plan['source_root'])/'outputs')
    assert args[args.index('--xapps-start')+1] == ''


def test_native_reference_has_no_arrival_overlay_and_keeps_its_own_service_recipe(plan):
    case = deepcopy(plan['cases'][0])
    case.update(traffic_profile=None, traffic_profile_sha256=None, scene='native_reference',
                origin_key='native_s1', traffic_load=1.)
    p = seal({**plan, 'cases':[case]})
    validate(p)
    args = worker.command(p, case, Path(p['source_root'])/'outputs/environment_validation/validation')
    assert '--traffic-profile' not in args and '--evaluation-origin-file' not in args
    assert '--xapp-decision-preview' in args


@pytest.mark.parametrize('change', [
    lambda p:p['cases'][1].update(worker_id='agx2'),
    lambda p:p['cases'][1].update(traffic_load=1.5),
    lambda p:p['cases'][1].update(ue_services_config_sha256='f'*64),
    lambda p:p['cases'][1].update(origin_key='another'),
    lambda p:p['cases'].reverse(),
    lambda p:p['cases'][1].update(app_ids=['xapp:ue_energy_rf','xapp:ap_energy_rf']),
    lambda p:p['cases'][0].update(origin_observe_slots=4000),
    lambda p:p.update(deadline_utc='2099-01-01T00:00:00'),
    lambda p:p.update(output_root='/tmp/escaped-output'),
    lambda p:p.update(source_archive_sha256='f'*64),
])
def test_invalid_host_pair_recipe_timing_or_scope_is_rejected(plan, change):
    change(plan)
    p = seal(plan)
    with pytest.raises(ValueError):
        validate(p)


def test_modified_plan_and_plan_inside_physical_source_are_rejected(plan):
    changed = deepcopy(plan)
    changed['cases'][0]['traffic_load'] = 2
    with pytest.raises(ValueError, match='digest'):
        worker.validate_plan(changed, plan['plan_sha256'])
    with pytest.raises(ValueError, match='outside immutable'):
        worker.validate_plan(plan, plan['plan_sha256'], plan_path=Path(plan['source_root'])/'plan.json')


def test_ui_snapshot_blocks_pending_batch_even_when_both_hosts_have_no_child(plan):
    snap = worker.ui_snapshot(dict(workers=[dict(id='agx1',busy=False),dict(id='agx2',busy=False)],
        latest_batch=dict(id='user-batch',phase='queued'),running_jobs=[]), now=100)
    assert snap['blocked_worker_ids'] == ['agx1','agx2']
    path = worker.SCOPE/'ui-admission.json'
    worker.atomic(path, snap)
    assert worker.ui_admission(plan, now=101)['allowed'] is False
    snap = worker.ui_snapshot(dict(workers=[dict(id='agx1',busy=False)],
        latest_batch=dict(phase='complete'),running_jobs=[]),now=100)
    worker.atomic(path,snap)
    assert worker.ui_admission(plan,now=101)['allowed'] is True
    assert worker.ui_admission(plan,now=191)['allowed'] is False
    assert worker.ui_admission(plan,now=99)['allowed'] is False
    path.unlink()
    assert worker.ui_admission(plan,now=101)['allowed'] is False


def test_ui_missing_worker_and_malformed_response_fail_closed(plan):
    with pytest.raises(ValueError):
        worker.ui_snapshot(dict(workers=[dict(id='agx1',busy='no')]))
    worker.atomic(worker.SCOPE/'ui-admission.json', worker.ui_snapshot(dict(
        workers=[dict(id='agx2',busy=False)], latest_batch=dict(phase='complete')),now=100))
    assert worker.ui_admission(plan,now=101)['allowed'] is False


def test_resource_admission_requires_each_budget():
    assert worker.resource_admission(worker.MINIMUM_FREE)
    for key in worker.MINIMUM_FREE:
        bad = {**worker.MINIMUM_FREE, key:worker.MINIMUM_FREE[key]-1}
        assert worker.resource_admission(bad) is False


def test_resume_rechecks_every_raw_sha_and_plan_identity(plan):
    out = worker.SCOPE/'results'
    case = plan['cases'][0]
    job = out/'jobs'/case['id']
    job.mkdir(parents=True)
    raw = job/'trace.jsonl'
    raw.write_text('retained physical trace\n')
    entry = dict(completed=True,case=case,plan_sha256=plan['plan_sha256'],
        source_sha256=plan['source_sha256'],runner_sha256=plan['runner_sha256'],
        files_sha256={'trace.jsonl':worker.sha(raw)})
    path = out/'checkpoints'/(case['id']+'.json')
    worker.atomic(path,entry)
    assert worker.completed(out,case,plan)
    raw.write_text('changed')
    assert not worker.completed(out,case,plan)
    raw.write_text('retained physical trace\n')
    entry['plan_sha256']='f'*64
    worker.atomic(path,entry)
    assert not worker.completed(out,case,plan)


def test_origin_builder_passes_before_length_and_rejects_cross_host_origin(plan):
    seen = {}
    def prepare(batch, output, **kwargs):
        seen.update(kwargs)
        return dict(status='ready',frame={},preview={},proof=dict(origin_id='origin',
            baseline_slot=2040,observe_slots=2000))
    out = worker.SCOPE/'results'
    assert worker.make_origin(out,plan['cases'][0],plan,SimpleNamespace(prepare_origin=prepare)) == 'origin'
    assert seen['observe_slots'] == 2000
    assert worker.origin_for(out,plan['cases'][1],plan)['origin_id'] == 'origin'
    owner = out/'origins'/(plan['cases'][0]['origin_key']+'-owner.json')
    data = worker.read(owner)
    data['actual_system_hostname'] = socket.gethostname()+'-other'
    worker.atomic(owner,data)
    with pytest.raises(ValueError,match='another host'):
        worker.origin_for(out,plan['cases'][1],plan)


def test_status_exception_reaps_only_its_own_child_and_restores_signal_handlers(monkeypatch):
    child = SimpleNamespace(pid=123, returncode=None)
    child.poll = lambda:child.returncode
    handlers = {}
    monkeypatch.setattr(worker.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(worker.signal,'getsignal',lambda n:'old-'+str(n))
    monkeypatch.setattr(worker.signal,'signal',lambda n,h:handlers.update({n:h}))
    stopped=[]
    def stop(c):
        assert c is child
        stopped.append(c.pid)
        c.returncode=-2
    monkeypatch.setattr(worker,'stop_owned_child',stop)
    def fail(c):raise OSError('owned status filesystem unavailable')
    with pytest.raises(OSError,match='owned status'):
        worker.run_owned_command([],cwd='/',env={},log=None,on_start=fail,
            tick=lambda:None,should_stop=lambda:False,poll=.1)
    assert stopped==[123]
    assert handlers=={signal.SIGINT:'old-'+str(signal.SIGINT),signal.SIGTERM:'old-'+str(signal.SIGTERM)}


def test_wrapper_signal_stops_its_child_without_polling_ui_midrun(monkeypatch):
    child=SimpleNamespace(pid=123,returncode=None)
    child.poll=lambda:child.returncode
    handlers={}
    monkeypatch.setattr(worker.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(worker.signal,'getsignal',lambda n:None)
    monkeypatch.setattr(worker.signal,'signal',lambda n,h:handlers.update({n:h}))
    monkeypatch.setattr(worker.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(worker,'stop_owned_child',lambda c:setattr(c,'returncode',-2))
    def tick():handlers[signal.SIGTERM](signal.SIGTERM,None)
    result=worker.run_owned_command([],cwd='/',env={},log=None,on_start=lambda c:None,
        tick=tick,should_stop=lambda:False,poll=.1)
    assert result.returncode==-2
