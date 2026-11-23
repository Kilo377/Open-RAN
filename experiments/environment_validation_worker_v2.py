"""Study-namespaced OFF/single-xApp experiments over the sealed simulator.

Copy this wrapper outside the sealed source tree. Plans, arrival/service
profiles, caches, results and locks belong to the dedicated RAM scope.
The v1 runner and all of its retained outputs remain untouched. Physical
commands are identical except for owned output paths and the study cache name.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import urlopen

SCOPE = Path('/dev/shm/oran_environment_20261122')
SCHEMA = 'oran.environment-validation-worker-plan.v2'
SOURCE_SHA256 = 'cbe114446814007b275856e4bae92a2fe609abf4f4bcbbf8c32394a722a36af0'
ARCHIVE_SHA256 = '4f915fd649e287b68465c4aed047a7b626959b471d532abd8a659e376e3641e1'
APPS = {'xapp:'+a for a in ('ue_energy_rf', 'ap_energy_rf', 'throughput_scheduler_mlp',
    'latency_scheduler_rf', 'latency_urllc_rf', 'mec_offload_budget_qgbdt')}
MINIMUM_FREE = dict(available_memory_bytes=8*1024**3, free_work_bytes=5*1024**3,
                    free_system_bytes=256*1024**2)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _owned(path):
    result = Path(path).resolve()
    if not result.is_relative_to(SCOPE.resolve()) or result == SCOPE.resolve():
        raise ValueError('Path is outside the dedicated environment scope')
    return result


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer '+name)
    return value


def _pin(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('A complete SHA256 pin is required')


def pairing_recipe(case):
    """These fields must be identical to the same-host OFF continuation."""
    names = ('worker_id', 'seed', 'scene', 'slots', 'baseline_slot', 'origin_observe_slots',
             'observe_slots', 'traffic_load', 'traffic_profile', 'traffic_profile_sha256',
             'ue_services_config', 'ue_services_config_sha256', 'ap_services_profile',
             'ap_services_profile_sha256')
    return {k: case.get(k) for k in names}


def study_paths(plan):
    """One study component; no caller may redirect v2 into a v1 output tree."""
    study_id = plan.get('study_id', '')
    if not isinstance(study_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', study_id):
        raise ValueError('Use a safe, non-traversing study identity')
    stage = plan.get('stage')
    if stage not in ('pilot', 'validation'):
        raise ValueError('Choose a separately recorded pilot or validation stage')
    root = _owned(plan['source_root'])
    expected = dict(output_root=root/'outputs/environment_validation'/study_id/stage,
                    active_plan_path=SCOPE.resolve()/'plans'/study_id/(stage+'.json'),
                    runner_path=SCOPE.resolve()/'runner/worker-v2.py')
    if 'output_root' not in plan:
        raise ValueError('The namespaced output_root must be explicitly pinned')
    for name, path in expected.items():
        declared = plan.get(name, str(path))
        if _owned(declared) != path:
            raise ValueError('The '+name+' must match the owned study namespace')
    return expected


def validate_plan(plan, pin, *, plan_path=None):
    body = deepcopy(plan)
    recorded = body.pop('plan_sha256', None)
    if recorded != pin or digest(body) != pin or plan.get('schema') != SCHEMA:
        raise ValueError('Environment plan schema or digest differs')
    if plan.get('stage') not in ('pilot', 'validation'):
        raise ValueError('Choose a separately recorded pilot or validation stage')
    for name in ('study_id', 'worker_id'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', plan.get(name, '')):
            raise ValueError('Invalid '+name)
    root = _owned(plan['source_root'])
    if root != SCOPE.resolve()/'source':
        raise ValueError('Use the immutable source copy under the new scope')
    paths = study_paths(plan)
    if plan_path is not None:
        path = _owned(plan_path)
        if path.is_relative_to(root):
            raise ValueError('Keep the worker plan outside immutable source')
        if path != paths['active_plan_path']:
            raise ValueError('The active plan must match this study and stage')
    _owned(plan['source_archive_path'])
    if plan.get('source_sha256') != SOURCE_SHA256 or plan.get('source_archive_sha256') != ARCHIVE_SHA256:
        raise ValueError('This experiment requires the verified cbe simulator release')
    _pin(plan['runner_sha256'])
    models = plan.get('frozen_models', {})
    if len(models) != 6:
        raise ValueError('Pin all six trained models')
    for relative, model_pin in models.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Use source-relative model paths')
        _pin(model_pin)
    deadline = datetime.fromisoformat(plan['deadline_utc'])
    if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0:
        raise ValueError('Use an explicit new UTC deadline')
    for field, default in (('analysis_reserve_s', 0), ('hard_stop_margin_s', 180)):
        _integer(plan.get(field, default), field, 0)
    poll = plan.get('poll_seconds', 2)
    if type(poll) not in (int, float) or not math.isfinite(poll) or not .1 <= poll <= 30:
        raise ValueError('Invalid status polling interval')
    lease_age = plan.get('ui_admission_max_age_s', 90)
    if type(lease_age) not in (int, float) or not math.isfinite(lease_age) or not 1 <= lease_age <= 300:
        raise ValueError('UI admission must be a short, refreshed lease')
    _owned(plan.get('ui_admission_path', SCOPE/'ui-admission.json'))
    if not isinstance(plan.get('cases'), list) or not plan['cases']:
        raise ValueError('An explicit nonempty case list is required')
    ids = set()
    origins = {}
    for case in plan['cases']:
        cid = case.get('id', '')
        if not re.fullmatch('[0-9a-f]{32}', cid) or cid in ids:
            raise ValueError('Case IDs must be unique, explicit research identities')
        ids.add(cid)
        for name in ('name', 'origin_key', 'scene', 'arm'):
            if not re.fullmatch(r'[A-Za-z0-9_.-]{1,200}', case.get(name, '')):
                raise ValueError('Invalid case '+name)
        if case.get('worker_id') != plan['worker_id']:
            raise ValueError('A case cannot silently move to another host')
        _integer(case['seed'], 'seed', 0)
        if case['seed'] >= 2**32:
            raise ValueError('Invalid seed')
        for name in ('slots', 'baseline_slot', 'origin_observe_slots', 'observe_slots'):
            _integer(case[name], name)
        if case['origin_observe_slots'] > 20000 or case['observe_slots'] > 20000:
            raise ValueError('Unsupported observation window')
        expected_baseline = ((case['origin_observe_slots']+1+39)//40)*40
        if case['baseline_slot'] != expected_baseline:
            raise ValueError('Native origin BEFORE window and RIC baseline disagree')
        if case['slots'] < case['baseline_slot']+case['observe_slots']:
            raise ValueError('The explicitly planned AFTER window must be complete')
        load = case['traffic_load']
        if type(load) not in (int, float) or not math.isfinite(load) or load <= 0:
            raise ValueError('Invalid traffic load')
        for path_field in ('traffic_profile', 'ue_services_config', 'ap_services_profile'):
            path = case.get(path_field)
            pin_value = case.get(path_field+'_sha256')
            if path is None:
                if path_field == 'ue_services_config' or pin_value is not None:
                    raise ValueError('Pin the explicit service recipe and nullable arrival recipe')
            else:
                _owned(path)
                _pin(pin_value)
        apps = case.get('app_ids')
        if not isinstance(apps, list) or len(apps) > 1 or any(a not in APPS for a in apps):
            raise ValueError('Use OFF or one registered frozen xApp')
        origin = case['origin_key']
        if not apps:
            if origin in origins:
                raise ValueError('Each same-host recipe needs one uniquely identified OFF reference')
            origins[origin] = pairing_recipe(case)
        elif origins.get(origin) != pairing_recipe(case):
            raise ValueError('A matching same-host OFF case must precede each treatment')
    return plan


def _runtime(root):
    """Import helpers from the sealed physical source, never copy them into it."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    perturbation = importlib.import_module('experiments.run_rcp_perturbation')
    study = importlib.import_module('experiments.quantification_study')
    context = importlib.import_module('experiments.context_identification')
    reference = importlib.import_module('smo.evaluation_reference')
    for module in (perturbation, study, context, reference):
        if not Path(module.__file__).resolve().is_relative_to(Path(root).resolve()):
            raise ValueError('A helper was imported from outside immutable source')
    return SimpleNamespace(source_hash=perturbation.source_hash,
        isolated_environment=study.isolated_environment, resources=context.resources,
        external_simulations=context.external_simulations, prepare_origin=reference.prepare_origin)


def validate_owned(plan, helpers):
    root = _owned(plan['source_root'])
    if sha(plan['source_archive_path']) != plan['source_archive_sha256']:
        raise ValueError('Immutable source archive differs')
    marker = read(root/'.smo-owned-worker.json')
    if marker.get('root') != str(root) or marker.get('sha256') != plan['source_archive_sha256']:
        raise ValueError('Owned source marker differs')
    if not (Path(marker['venv'])/'bin/python').is_file():
        raise ValueError('Read-only Python environment is unavailable')
    if helpers.source_hash(root) != plan['source_sha256']:
        raise ValueError('Immutable physical source differs')
    release = root/'artifacts/xapps/chongyu/20261117'
    expected = {str((release/e['path']).relative_to(root)): e['sha256']
                for e in read(release/'release.json')['models'].values()}
    if expected != plan['frozen_models'] or any(sha(root/p) != pin for p, pin in expected.items()):
        raise ValueError('A frozen trained model differs')
    for case in plan['cases']:
        for name in ('traffic_profile', 'ue_services_config', 'ap_services_profile'):
            if case.get(name) is not None and sha(case[name]) != case[name+'_sha256']:
                raise ValueError('An explicitly pinned run recipe changed: '+name)


def ui_snapshot(deployment, *, now=None):
    """Mac-side read-only API response -> short admission lease for both hosts."""
    workers = deployment.get('workers')
    if not isinstance(workers, list) or not workers:
        raise ValueError('UI worker state is unavailable')
    if any(not isinstance(w.get('id'), str) or type(w.get('busy')) is not bool for w in workers):
        raise ValueError('UI worker state is malformed')
    batch = deployment.get('latest_batch') or {}
    active = bool(batch and batch.get('phase') not in ('complete', 'complete_with_failures', 'paused'))
    blocked = [w['id'] for w in workers if w['busy'] or active]
    return dict(schema='oran.environment-ui-admission.v1', sampled_unix_s=time.time() if now is None else now,
        worker_ids=[w['id'] for w in workers], blocked_worker_ids=blocked,
        ui_batch_id=batch.get('id'), ui_batch_phase=batch.get('phase'), ui_batch_active=active,
        running_ui_jobs=len(deployment.get('running_jobs', [])), read_only=True)


def fetch_ui_snapshot(url='http://127.0.0.1:53505/api/deployment', *, timeout=10):
    with urlopen(url, timeout=timeout) as response:
        return ui_snapshot(json.load(response))


def ui_admission(plan, *, now=None):
    now = time.time() if now is None else now
    path = _owned(plan.get('ui_admission_path', SCOPE/'ui-admission.json'))
    try:
        snapshot = read(path)
        age = now-snapshot['sampled_unix_s']
        if (snapshot.get('schema') != 'oran.environment-ui-admission.v1'
                or not 0 <= age <= plan.get('ui_admission_max_age_s', 90)
                or plan['worker_id'] not in snapshot['worker_ids']):
            raise ValueError('Missing or expired UI admission lease')
        blocked = plan['worker_id'] in snapshot['blocked_worker_ids'] or snapshot.get('ui_batch_active') is not False
        return dict(allowed=not blocked, reason='user_ui_work_pending' if blocked else None,
                    snapshot_sha256=sha(path), age_s=age)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(allowed=False, reason='ui_admission_unavailable_or_expired')


def resource_admission(free):
    return all(type(free.get(k)) in (int, float) and free[k] >= v for k, v in MINIMUM_FREE.items())


def command(plan, case, out):
    if Path(out).resolve() != study_paths(plan)['output_root']:
        raise ValueError('Command output must use this study namespace')
    root = Path(plan['source_root'])
    data = Path(out)/'jobs'/case['id']/'data'
    run = data/'run'
    args = ['bash', str(root/'scripts/orin/run-six-ue-cuda-nr.sh'), '--model', 'fast',
        '--fast-profile', 'system', '--slots', str(case['slots']), '--seed', str(case['seed']),
        '--traffic-load', str(case['traffic_load']), '--ue-services', 'drx-mec',
        '--ue-services-config', case['ue_services_config'], '--xapp-pool', 'chongyu',
        '--ap-services-profile', case.get('ap_services_profile') or str(root/'configs/rt_sys/ap_services_reference.json'),
        '--record-compression', 'gzip', '--live-xapps', '--xapps-start', '',
        '--xapp-toggle-timeline', str(data/'timeline.json'), '--xapp-decision-preview',
        '--xapp-preview-before-slot', str(case['baseline_slot']),
        '--channel-cache-root', str(root/'channel_cache/environment_validation'/plan['study_id']/plan['stage']/case['origin_key']),
        '--replay-max-frames', '100', '--live-state', str(run/'live.json'), '--output', str(run)]
    if case['traffic_profile'] is not None:
        args += ['--traffic-profile', case['traffic_profile']]
    if case['app_ids']:
        args += ['--evaluation-origin-file', str(data/'expected_origin.json')]
    return args


def completed(out, case, plan):
    path = Path(out)/'checkpoints'/(case['id']+'.json')
    try:
        seal = read(path)
        if (seal.get('completed') is not True or seal.get('case') != case
                or seal.get('plan_sha256') != plan['plan_sha256']
                or seal.get('source_sha256') != plan['source_sha256']
                or seal.get('runner_sha256') != plan['runner_sha256']):
            return False
        job = Path(out)/'jobs'/case['id']
        files = seal['files_sha256']
        return bool(files) and all((job/name).resolve().is_relative_to(job.resolve())
            and (job/name).is_file() and sha(job/name) == pin for name, pin in files.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def make_origin(out, case, plan, helpers):
    batch = dict(seed=case['seed'], worker_source_archive_sha256=plan['source_archive_sha256'],
        members=[dict(job_id=case['id'], phase='complete', comparison=dict(
            baseline_origin='fresh_run_all_xapps_off', baseline=dict(slot=case['baseline_slot'])))])
    result = helpers.prepare_origin(batch, Path(out)/'jobs',
        observe_slots=case['origin_observe_slots'], source_archive=Path(plan['source_archive_path']))
    if result['status'] != 'ready':
        raise ValueError('OFF prefix could not be certified: '+result.get('reason', 'unknown'))
    proof = result['proof']
    atomic(Path(out)/'origins'/(case['origin_key']+'.json'), proof)
    atomic(Path(out)/'origins'/(case['origin_key']+'-owner.json'), dict(worker_id=plan['worker_id'],
        actual_system_hostname=socket.gethostname(), source_sha256=plan['source_sha256'],
        source_case_id=case['id'], origin_id=proof['origin_id'], recipe=pairing_recipe(case)))
    atomic(Path(out)/'origins'/(case['origin_key']+'-before-preview.json'), dict(
        frame=result['frame'], preview=result['preview'], origin_id=proof['origin_id'], after_kpis_used=False))
    return proof['origin_id']


def origin_for(out, case, plan):
    base = Path(out)/'origins'
    proof = read(base/(case['origin_key']+'.json'))
    owner = read(base/(case['origin_key']+'-owner.json'))
    if (owner['worker_id'] != plan['worker_id'] or owner['actual_system_hostname'] != socket.gethostname()
            or owner['source_sha256'] != plan['source_sha256'] or owner['origin_id'] != proof['origin_id']
            or owner['recipe'] != pairing_recipe(case)):
        raise ValueError('OFF origin belongs to another host or physical recipe')
    if proof['baseline_slot'] != case['baseline_slot'] or proof['observe_slots'] != case['origin_observe_slots']:
        raise ValueError('OFF origin timing differs')
    return proof


def stop_owned_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def run_owned_command(args, *, cwd, env, log, on_start, tick, should_stop, poll):
    """An audit/status error or wrapper signal must not orphan its own PHY."""
    child = None
    requested = []
    previous = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.getsignal(number)
        signal.signal(number, lambda number, frame: requested.append(number))
    try:
        child = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        on_start(child)
        while child.poll() is None:
            tick()
            if requested or should_stop():
                stop_owned_child(child)
            time.sleep(poll)
        return child
    finally:
        try:
            if child is not None and child.poll() is None:
                stop_owned_child(child)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def run_worker(plan_path, pin, *, retry_failed=False):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(read(plan_path), pin, plan_path=plan_path)
    paths = study_paths(plan)
    if Path(__file__).resolve() != paths['runner_path']:
        raise ValueError('Run the externally pinned worker-v2 path')
    if sha(__file__) != plan['runner_sha256']:
        raise ValueError('Actual external wrapper SHA differs from the plan')
    root = Path(plan['source_root'])
    helpers = _runtime(root)
    validate_owned(plan, helpers)
    lock = (SCOPE/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = paths['output_root']
    out.mkdir(parents=True, exist_ok=True)
    for name in ('jobs', 'checkpoints', 'partial', 'origins'):
        (out/name).mkdir(exist_ok=True)
    snapshot = out/'plan.json'
    if snapshot.exists() and read(snapshot) != plan:
        raise ValueError('Retain the earlier stage plan; use a new study identity instead of overwriting')
    if not snapshot.exists():
        atomic(snapshot, plan)
    for path in (SCOPE/'tmp', SCOPE/'.cache'):
        path.mkdir(exist_ok=True)
    env = helpers.isolated_environment(SCOPE)
    status_path = out/'study_status.json'
    old = read(status_path) if status_path.exists() else {}
    status = dict(schema='oran.environment-worker-status.v1', study_id=plan['study_id'], stage=plan['stage'],
        worker_id=plan['worker_id'], actual_system_hostname=socket.gethostname(), worker_pid=os.getpid(),
        plan_sha256=pin, source_sha256=plan['source_sha256'], source_archive_sha256=plan['source_archive_sha256'],
        runner_sha256=plan['runner_sha256'], done=[], failed=old.get('failed', []), running=[],
        completed=False, phase='initializing', models_unchanged=True, raw_data_retained=True,
        training_performed=False, post_deployment_kpis_used_for_model_training=False,
        environment_adoption_uses_validation_results=True)
    def publish(**values):
        status.update(values, updated_unix_s=time.time(), free=helpers.resources(SCOPE))
        atomic(status_path, status)
    if status['failed'] and not retry_failed:
        publish(phase='retained_failure_requires_review', completed=True)
        return status
    if retry_failed:
        atomic(out/('retained-failures-'+str(time.time_ns())+'.json'), status['failed'])
        status['failed'] = []
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    reserve = plan.get('analysis_reserve_s', 0)
    poll = plan.get('poll_seconds', 2)
    for case in plan['cases']:
        if completed(out, case, plan):
            status['done'].append(case['id'])
            publish()
            continue
        estimate = case.get('estimated_wall_s', max(180, case['slots']*.12))
        while True:
            if (SCOPE/'STOP').exists() or (out/'STOP').exists() or time.time()+estimate+reserve > deadline:
                publish(phase='budget_or_stop_complete', completed=True, running=[])
                return status
            lease = ui_admission(plan)
            if not (out/'ADMISSION_ALLOWED').is_file():
                publish(phase='waiting_for_stage_admission', ui_admission=lease)
            elif not lease['allowed']:
                publish(phase='waiting_for_user_ui', ui_admission=lease)
            elif helpers.external_simulations():
                publish(phase='waiting_for_external_simulation', ui_admission=lease)
            elif not resource_admission(helpers.resources(SCOPE)):
                publish(phase='waiting_for_storage_or_memory', ui_admission=lease)
            else:
                break
            time.sleep(poll)
        validate_owned(plan, helpers)
        if sha(__file__) != plan['runner_sha256'] or read(plan_path) != plan:
            raise ValueError('The active wrapper or external plan changed')
        job = out/'jobs'/case['id']
        old_seal = out/'checkpoints'/(case['id']+'.json')
        if job.exists() or old_seal.exists():
            partial = out/'partial'/(case['id']+'_'+str(time.time_ns()))
            partial.mkdir()
            if job.exists():
                job.rename(partial/'job')
            if old_seal.exists():
                old_seal.rename(partial/'checkpoint.json')
        data = job/'data'
        data.mkdir(parents=True)
        request = dict(schema='oran.environment-run-request.v1', study_id=plan['study_id'], stage=plan['stage'],
            case=case, plan_sha256=pin, source_sha256=plan['source_sha256'],
            source_archive_sha256=plan['source_archive_sha256'], runner_sha256=plan['runner_sha256'],
            frozen_models=plan['frozen_models'], actual_system_hostname=socket.gethostname(), training_performed=False)
        atomic(data/'request.json', request)
        atomic(data/'timeline.json', [dict(at_ms=case['baseline_slot']*.5, operation='set',
            apps=[a.removeprefix('xapp:') for a in case['app_ids']], cell='*')])
        if case['app_ids']:
            atomic(data/'expected_origin.json', origin_for(out, case, plan))
        args = command(plan, case, out)
        atomic(data/'command.json', dict(argv=args, cache_environment={k: v for k, v in env.items() if str(SCOPE) in v}))
        started = time.time()
        running = {**case, 'started_unix_s': started, 'slot': 0}
        def on_start(child):
            running['pid'] = child.pid
            publish(phase='running', running=[running])
        def tick():
            try:
                live = read(data/'run/live.json')
                running.update(slot=live.get('slot', 0), simulation_stage=live.get('stage'))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            publish()
        def should_stop():
            return ((SCOPE/'STOP').exists() or (out/'STOP').exists()
                    or time.time() >= deadline-plan.get('hard_stop_margin_s', 180))
        try:
            with (data/'simulation.log').open('wb') as log:
                child = run_owned_command(args, cwd=root, env=env, log=log,
                    on_start=on_start, tick=tick, should_stop=should_stop, poll=poll)
        except Exception as error:
            running.update(error=type(error).__name__+': '+str(error), wall_s=time.time()-started)
            status['failed'].append(running)
            publish(phase='failed_retained', running=[], completed=True)
            return status
        running.update(exit_code=child.returncode, wall_s=time.time()-started)
        try:
            result = read(data/'run/run_status.json')
            if child.returncode or not result.get('completed') or not result.get('replay', {}).get('audits_passed'):
                raise ValueError('Physical run is incomplete; raw data retained')
            if result.get('source_sha256') != plan['source_sha256']:
                raise ValueError('Recorded physical source differs')
            validate_owned(plan, helpers)
            if case['app_ids']:
                expected = origin_for(out, case, plan)
                verified = read(data/'run/evaluation_origin_verified.json')
                if verified.get('verified_before_activation') is not True or verified['origin_id'] != expected['origin_id']:
                    raise ValueError('Common OFF prefix was not verified before activation')
                origin_id = verified['origin_id']
            else:
                origin_id = make_origin(out, case, plan, helpers)
            seal = dict(schema='oran.environment-validation-run.v1', completed=True, case=case,
                worker_id=plan['worker_id'], actual_system_hostname=socket.gethostname(), stage=plan['stage'],
                study_id=plan['study_id'], plan_sha256=pin, source_sha256=plan['source_sha256'],
                source_archive_sha256=plan['source_archive_sha256'], runner_sha256=plan['runner_sha256'],
                frozen_models=plan['frozen_models'], origin_id=origin_id, wall_s=running['wall_s'],
                finished_unix_s=time.time(), training_performed=False,
                post_deployment_kpis_used_for_model_training=False,
                environment_adoption_uses_validation_results=True,
                models_unchanged=True, files_sha256={str(p.relative_to(job)): sha(p) for p in job.rglob('*') if p.is_file()})
            atomic(old_seal, seal)
            status['done'].append(case['id'])
            publish(running=[])
        except Exception as error:
            running['error'] = type(error).__name__+': '+str(error)
            status['failed'].append(running)
            publish(phase='failed_retained', running=[], completed=True)
            return status
    publish(phase='complete', running=[], completed=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    run_worker(args.worker_plan, args.plan_sha256, retry_failed=args.retry_failed)


if __name__ == '__main__':
    main()
