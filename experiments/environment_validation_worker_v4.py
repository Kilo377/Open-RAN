"""A pinned UE0 road phase over the unchanged sealed simulator.

Copy this wrapper outside the sealed source tree. Plans, arrival/service
profiles, caches, results and locks belong to the dedicated RAM scope.
One manager holds the global host lock; threads own independent physical
process groups. OFF references finish serially before singleton concurrency.
Real serial-versus-parallel PHY parity must be checked before adoption.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from threading import Event, RLock
from urllib.request import urlopen

SCOPE = Path('/dev/shm/oran_environment_20261122')
SCHEMA = 'oran.environment-validation-worker-plan.v4'
BASE_CONFIG = 'configs/rt_sys/agx_munich_6ue_depth2_fast.json'
ROUTE_OVERRIDE = {'ue_0': {'route_index': 0, 'start_fraction': .65}}
GEOMETRY_CHANGE = dict(kind='UE0_INITIAL_ROAD_PHASE', ue_route_overrides=ROUTE_OVERRIDE,
    source_files_changed=False, speed_radio_controls_and_other_five_routes_unchanged=True)
SOURCE_SHA256 = 'cbe114446814007b275856e4bae92a2fe609abf4f4bcbbf8c32394a722a36af0'
ARCHIVE_SHA256 = '4f915fd649e287b68465c4aed047a7b626959b471d532abd8a659e376e3641e1'
APPS = {'xapp:'+a for a in ('ue_energy_rf', 'ap_energy_rf', 'throughput_scheduler_mlp',
    'latency_scheduler_rf', 'latency_urllc_rf', 'mec_offload_budget_qgbdt')}
MINIMUM_FREE = dict(available_memory_bytes=8*1024**3, free_work_bytes=5*1024**3,
                    free_system_bytes=256*1024**2)
SIX_UE_LAUNCHER_SHA256 = '1a1e25637ccbfd2a6dd8a4ab7fcc7549e5655267a9bb4a79c895316bda030e6d'
SIX_UE_FIXED_ARGV = ['experiments/run_rt_sys.py', '--device', 'cuda', '--rt-device', 'cpu',
    '--scene', 'munich', '--cells', '3', '--ues', '6', '--config',
    'configs/rt_sys/agx_munich_6ue_depth2_fast.json', '--algebra-backend', 'torch',
    '--nr-execution', 'batched', '--recording-mode', 'experiment', '--diagnostic-window-slots', '40']
CACHE_ENV_KEYS = frozenset(('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','CUDA_CACHE_PATH','CUPY_CACHE_DIR',
    'TORCH_HOME','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','NUMBA_CACHE_DIR',
    'MPLCONFIGDIR','DRJIT_CACHE_DIR','MESA_SHADER_CACHE_DIR','HF_HOME','PIP_CACHE_DIR','UV_CACHE_DIR'))
ENVIRONMENT_SNAPSHOT_CODE = '''import json,os,sys
from pathlib import Path
expected=json.loads(Path(sys.argv[2]).read_text())
actual={k:os.environ.get(k)for k in expected}
if actual!=expected:raise RuntimeError("Activated case cache environment differs")
report=dict(case_cache_environment=actual,python_executable=sys.executable,
 activation={k:os.environ.get(k)for k in ("VIRTUAL_ENV","PATH","LD_LIBRARY_PATH","PYTHONPATH","CUDA_PATH")},
 recorded_after_activation=True)
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+"\\n")
'''
CASE_LAUNCHER = '''#!/usr/bin/env bash
set -euo pipefail
source "$1"
shift
source "$1"
shift
oran_effective_environment="$1"
shift
oran_expected_environment="$1"
shift
python -c '''+shlex.quote(ENVIRONMENT_SNAPSHOT_CODE)+''' "$oran_effective_environment" "$oran_expected_environment"
unset oran_effective_environment oran_expected_environment
exec python "$@"
'''


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
             'ap_services_profile_sha256', 'base_config', 'base_config_sha256')
    return {k: case.get(k) for k in names}


def base_config_path(plan):
    return _owned(SCOPE/'protocol/profiles'/(plan['study_id']+'_ue0_road0_phase065_base.json'))


def validate_base_config(plan, case):
    """Only a new road phase is admitted; no other physical/control edit."""
    path = _owned(case['base_config'])
    if path != base_config_path(plan) or sha(path) != case['base_config_sha256']:
        raise ValueError('Pinned study-owned base configuration differs')
    frozen = read(Path(plan['source_root'])/BASE_CONFIG)
    if 'ue_route_overrides' in frozen:
        raise ValueError('Frozen base unexpectedly already has route overrides')
    expected = {**frozen, 'ue_route_overrides': deepcopy(ROUTE_OVERRIDE)}
    # Canonical JSON also distinguishes booleans from numeric radio settings.
    if digest(read(path)) != digest(expected):
        raise ValueError('Only UE0 route0 start_fraction .65 may differ from frozen base')
    return path


def on_admission(plan, case, out):
    """OFF completion never implicitly authorizes any trained control action."""
    if case['app_ids'] and not (Path(out)/'ON_ADMISSION_ALLOWED').is_file():
        return dict(allowed=False, reason='waiting_for_explicit_on_admission')
    return dict(allowed=True)


def study_paths(plan):
    """One study component; no caller may redirect into retained outputs."""
    study_id = plan.get('study_id', '')
    if not isinstance(study_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', study_id):
        raise ValueError('Use a safe, non-traversing study identity')
    stage = plan.get('stage')
    if stage not in ('pilot', 'validation'):
        raise ValueError('Choose a separately recorded pilot or validation stage')
    root = _owned(plan['source_root'])
    expected = dict(output_root=root/'outputs/environment_validation'/study_id/stage,
                    active_plan_path=SCOPE.resolve()/'plans'/study_id/(stage+'.json'),
                    runner_path=SCOPE.resolve()/'runner/worker-v4.py')
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
    if plan.get('require_on_admission') is not True or plan.get('geometry_change') != GEOMETRY_CHANGE:
        raise ValueError('v4 requires explicit ON admission and the single pinned geometry change')
    parallel = plan.get('max_parallel', 2)
    if type(parallel) is not int or not 1 <= parallel <= 3:
        raise ValueError('max_parallel must be an integer from one to three')
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
        if _owned(case['base_config']) != base_config_path(plan):
            raise ValueError('The base configuration must use the pinned study-owned filename')
        _pin(case['base_config_sha256'])
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
        validate_base_config(plan, case)


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
        '--channel-cache-root', str(root/'channel_cache/environment_validation'/plan['study_id']/plan['stage']/case['id']),
        '--replay-max-frames', '100', '--live-state', str(run/'live.json'), '--output', str(run)]
    if case['traffic_profile'] is not None:
        args += ['--traffic-profile', case['traffic_profile']]
    if case['app_ids']:
        args += ['--evaluation-origin-file', str(data/'expected_origin.json')]
    # argparse uses the final occurrence; the sealed launcher's own --config
    # remains untouched and every other fixed/case argument stays identical.
    args += ['--config', str(validate_base_config(plan, case))]
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


def process_parents(proc=Path('/proc')):
    """Read Linux ancestry without matching or killing by command text."""
    result = {}
    for path in Path(proc).glob('[0-9]*/stat'):
        try:
            # comm may contain spaces or parentheses; PPID follows the final ).
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            pass
    return result


def is_owned_descendant(pid, roots, parents):
    seen = set()
    while pid not in seen:
        if pid in roots:
            return True
        seen.add(pid)
        if pid not in parents:
            return False
        pid = parents[pid]
    return False


class ManagerStopped(Exception):
    pass


class OwnedChildren:
    """Only direct Popen handles created by this manager enter the registry."""
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.lock = RLock()
        self.children = {}
        self.stop_locks = {}

    def start(self, args, **kwargs):
        with self.lock:
            if self.stop_event.is_set():
                raise ManagerStopped('Manager stopped before child creation')
            child = subprocess.Popen(args, **kwargs)
            try:
                self.children[child.pid] = child
                self.stop_locks[child.pid] = RLock()
            except BaseException:
                stop_child(child)
                raise
            # STOP may arrive while Popen is returning. The registered handle
            # remains visible to the manager and to the caller's finally block.
            return child

    def finish(self, child):
        with self.lock:
            if self.children.get(child.pid) is child:
                del self.children[child.pid]
                self.stop_locks.pop(child.pid, None)

    def stop(self, child):
        # Manager and owner thread may observe STOP simultaneously. Serialize
        # signals/wait for this one handle so a second SIGINT cannot interrupt
        # the simulator's first cleanup attempt.
        with self.lock:
            lock = self.stop_locks.setdefault(child.pid, RLock())
        with lock:
            stop_child(child)

    def external(self, inspect, parents=None):
        # Share the same lock as Popen+registration: no own child can appear in
        # /proc between inspection and registration and be treated as external.
        with self.lock:
            records = inspect()
            roots = {pid for pid, child in self.children.items() if child.poll() is None}
            ancestry = process_parents() if parents is None else parents
            return [r for r in records if not is_owned_descendant(r['pid'], roots, ancestry)]

    def stop_all(self):
        with self.lock:
            children = list(self.children.values())
        errors = []
        for child in children:
            try:
                self.stop(child)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]


def stop_child(child):
    """Reap the manager-created physical process group, never a foreign PID."""
    if child.poll() is None:
        for sig, timeout in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, None)):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                continue
    else:
        child.wait()


def case_cache_root(plan, case):
    return _owned(SCOPE/'cache'/plan['study_id']/plan['stage']/case['id'])


def activated_command(plan, case, out, env):
    """Restore only cache paths after the original activation script runs.

    activate-gpu/activate force CUDA, CuPy, DrJit and matplotlib caches back
    under shared source. Keep their PATH/venv/library setup, then restore the
    case-owned cache whitelist before the unchanged Python fixed/case argv.
    """
    root = Path(plan['source_root'])
    sealed = root/'scripts/orin/run-six-ue-cuda-nr.sh'
    if sha(sealed) != SIX_UE_LAUNCHER_SHA256:
        raise ValueError('Sealed six-UE launcher bytes differ')
    parsed = shlex.split(sealed.read_text().split('exec python ', 1)[1].replace('\\\n',' '))
    if parsed != SIX_UE_FIXED_ARGV+['$@']:
        raise ValueError('Sealed six-UE fixed Python argv differs')
    base = command(plan, case, out)
    if '--slots' not in base:
        raise ValueError('Every parallel case must have an explicit duration')
    data = Path(out)/'jobs'/case['id']/'data'
    cache = case_cache_root(plan, case)
    cache_env = {k:env[k] for k in sorted(CACHE_ENV_KEYS) if k in env}
    if not cache_env or any(not Path(v).resolve().is_relative_to(cache) for v in cache_env.values()):
        raise ValueError('All restored cache paths must belong to this case')
    restore = data/'case-cache-environment.sh'
    restore.write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in cache_env.items()))
    launcher = data/'launch-case.sh'
    launcher.write_text(CASE_LAUNCHER)
    expected = data/'case-cache-environment.json'
    atomic(expected, cache_env)
    effective = data/'effective-environment.json'
    args = ['bash',str(launcher),str(root/'scripts/orin/activate-gpu.sh'),str(restore),
            str(effective),str(expected),*SIX_UE_FIXED_ARGV,*base[2:]]
    return args, dict(case_cache_environment=cache_env, effective_environment_path=str(effective),
        launcher_sha256=sha(launcher), sealed_launcher_sha256=sha(sealed),
        activation_sha256=sha(root/'scripts/orin/activate-gpu.sh'), fixed_python_argv=SIX_UE_FIXED_ARGV,
        effective_environment_recorded_after_activation=True)


def validate_installed_modelset(plan):
    """OFF warm-up must finish the shared model install before parallel ON."""
    root = Path(plan['source_root'])
    artifact = root/'artifacts/xapps/chongyu/20261117'
    installed = root/'outputs/trained_xapps/20261117'
    release = read(artifact/'release.json')
    if sha(installed/'release.json') != sha(artifact/'release.json') or len(release.get('models', {})) != 6:
        raise ValueError('OFF warm-up did not leave a verified complete six-model install')
    for entry in release['models'].values():
        relative = Path(entry['path'])
        target = (installed/relative).resolve()
        source_relative = str((artifact/relative).relative_to(root))
        if (relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(installed.resolve())
                or entry['sha256'] != plan['frozen_models'].get(source_relative)
                or sha(target) != entry['sha256']):
            raise ValueError('The installed frozen model differs; do not race a new parallel install')


def execute_case(plan, case, out, helpers, registry, stop_event, progress, progress_lock):
    """One thread owns one case lock/status/log/cache and one physical child."""
    child = None
    job = out/'jobs'/case['id']
    case_status = out/'case_status'/(case['id']+'.json')
    state = dict(**case, phase='initializing', started_unix_s=time.time(), slot=0)
    def publish(**values):
        state.update(values, updated_unix_s=time.time())
        atomic(case_status, state)
        with progress_lock:
            progress[case['id']] = deepcopy(state)
    try:
        with (out/'locks'/(case['id']+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if completed(out, case, plan):
                return dict(id=case['id'], phase='complete', cached=True)
            if stop_event.is_set():
                raise ManagerStopped('Manager stopped before case admission')
            if not on_admission(plan, case, out)['allowed']:
                raise ManagerStopped('Explicit ON admission was withdrawn before case execution')
            validate_owned(plan, helpers)
            if sha(__file__) != plan['runner_sha256'] or read(study_paths(plan)['active_plan_path']) != plan:
                raise ValueError('The v4 wrapper or active plan changed')
            if case['app_ids']:
                validate_installed_modelset(plan)
            origin = origin_for(out, case, plan) if case['app_ids'] else None
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
            cache = case_cache_root(plan, case)
            for path in (cache/'tmp', cache/'.cache'):
                path.mkdir(parents=True, exist_ok=True)
            env = helpers.isolated_environment(cache)
            atomic(data/'request.json', dict(schema='oran.environment-run-request.v4',
                study_id=plan['study_id'], stage=plan['stage'], case=case,
                plan_sha256=plan['plan_sha256'], source_sha256=plan['source_sha256'],
                source_archive_sha256=plan['source_archive_sha256'], runner_sha256=plan['runner_sha256'],
                frozen_models=plan['frozen_models'], actual_system_hostname=socket.gethostname(),
                training_performed=False, scheduling_only_change=False,
                geometry_change=GEOMETRY_CHANGE, base_config_sha256=case['base_config_sha256'],
                require_on_admission=True))
            atomic(data/'timeline.json', [dict(at_ms=case['baseline_slot']*.5, operation='set',
                apps=[a.removeprefix('xapp:') for a in case['app_ids']], cell='*')])
            if origin is not None:
                atomic(data/'expected_origin.json', origin)
            args, activation = activated_command(plan, case, out, env)
            atomic(data/'command.json', dict(argv=args, cache_environment=activation['case_cache_environment'],
                activation=activation))
            publish()
            with (data/'simulation.log').open('wb') as log:
                child = registry.start(args, cwd=plan['source_root'], env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                publish(phase='running', pid=child.pid)
                while child.poll() is None:
                    if stop_event.is_set():
                        registry.stop(child)
                    try:
                        live = read(data/'run/live.json')
                    except (OSError, ValueError, TypeError, AttributeError):
                        live = None
                    if isinstance(live, dict):
                        state.update(slot=live.get('slot', 0), simulation_stage=live.get('stage'))
                    publish()
                    time.sleep(plan.get('poll_seconds', 2))
                code = child.wait()
            state.update(exit_code=code, wall_s=time.time()-state['started_unix_s'])
            if code and stop_event.is_set():
                raise ManagerStopped('Stopped physical case retained without a complete seal')
            result = read(data/'run/run_status.json')
            if code or not result.get('completed') or not result.get('replay', {}).get('audits_passed'):
                if stop_event.is_set():
                    raise ManagerStopped('Stopped physical case retained without a complete seal')
                raise ValueError('Physical run is incomplete; raw records retained')
            if result.get('source_sha256') != plan['source_sha256']:
                raise ValueError('Recorded physical source differs')
            effective = read(activation['effective_environment_path'])
            if (effective.get('recorded_after_activation') is not True
                    or effective.get('case_cache_environment') != activation['case_cache_environment']):
                raise ValueError('Effective activated cache environment differs from this case')
            validate_owned(plan, helpers)
            if origin is not None:
                verified = read(data/'run/evaluation_origin_verified.json')
                if verified.get('verified_before_activation') is not True or verified.get('origin_id') != origin['origin_id']:
                    raise ValueError('Same-host OFF prefix was not verified before activation')
                origin_id = verified['origin_id']
            else:
                origin_id = make_origin(out, case, plan, helpers)
            seal = dict(schema='oran.environment-validation-run.v4', completed=True, case=case,
                worker_id=plan['worker_id'], actual_system_hostname=socket.gethostname(), stage=plan['stage'],
                study_id=plan['study_id'], plan_sha256=plan['plan_sha256'], source_sha256=plan['source_sha256'],
                source_archive_sha256=plan['source_archive_sha256'], runner_sha256=plan['runner_sha256'],
                frozen_models=plan['frozen_models'], origin_id=origin_id, wall_s=state['wall_s'],
                finished_unix_s=time.time(), training_performed=False,
                post_deployment_kpis_used_for_model_training=False, scheduling_only_change=False,
                geometry_change=GEOMETRY_CHANGE, base_config_sha256=case['base_config_sha256'],
                require_on_admission=True,
                environment_adoption_uses_validation_results=True, models_unchanged=True,
                files_sha256={str(p.relative_to(job)):sha(p) for p in job.rglob('*') if p.is_file()})
            atomic(old_seal, seal)
            publish(phase='complete', origin_id=origin_id)
            return dict(id=case['id'], phase='complete', wall_s=state['wall_s'])
    except ManagerStopped as error:
        publish(phase='stopped', error=str(error), wall_s=time.time()-state['started_unix_s'])
        return dict(id=case['id'], phase='stopped', error=str(error))
    except Exception as error:
        publish(phase='failed', error=type(error).__name__+': '+str(error), wall_s=time.time()-state['started_unix_s'])
        return dict(id=case['id'], phase='failed', error=type(error).__name__+': '+str(error))
    finally:
        if child is not None:
            if child.poll() is None:
                registry.stop(child)
            if child.poll() is not None:
                registry.finish(child)


def schedule_cases(cases, *, max_parallel, execute, admission, stop_reason, stop_event,
                   stop_children, publish, poll=.1):
    """OFF barrier, bounded singleton admission, and drain-on-failure policy."""
    queue = [c for c in cases if not c['app_ids']] + [c for c in cases if c['app_ids']]
    running = {}
    result = dict(done=[], failed=[], stopped=[], queued=[], termination=None)
    executor = ThreadPoolExecutor(max_workers=max_parallel)
    no_new = False
    last_reason = None
    try:
        while queue or running:
            reason = stop_reason()
            if reason or stop_event.is_set():
                result['termination'] = reason or 'manager_stop_requested'
                no_new = True
                stop_event.set()
                stop_children()
            for future, case in list(running.items()):
                if not future.done():
                    continue
                del running[future]
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = dict(id=case['id'], phase='failed', error=type(error).__name__+': '+str(error))
                if outcome['phase'] == 'complete':
                    result['done'].append(case['id'])
                elif outcome['phase'] == 'stopped':
                    result['stopped'].append(outcome)
                    no_new = True
                else:
                    result['failed'].append(outcome)
                    no_new = True  # Already-running siblings finish normally.
            while queue and not no_new and len(running) < max_parallel:
                case = queue[0]
                # Every OFF runs alone. No ON starts until all pending/running
                # OFF references have completed and written their origin proof.
                if running and (not case['app_ids'] or any(not c['app_ids'] for c in running.values())):
                    break
                gate = admission(case, len(running))
                last_reason = gate.get('reason')
                if gate.get('terminal'):
                    no_new = True
                    result['termination'] = last_reason
                    break
                if not gate.get('allowed') or stop_event.is_set():
                    break
                queue.pop(0)
                running[executor.submit(execute, case)] = case
                if not case['app_ids']:
                    break
            result['queued'] = [c['id'] for c in queue]
            publish(dict(result, running=list(running.values()), admission_reason=last_reason,
                draining_after_failure=bool(result['failed']) and bool(running)))
            if no_new and not running:
                break
            if queue or running:
                time.sleep(poll)
        return result
    except BaseException:
        stop_event.set()
        stop_children()
        raise
    finally:
        try:
            if stop_event.is_set():
                stop_children()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def run_worker(plan_path, pin, *, retry_failed=False):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(read(plan_path), pin, plan_path=plan_path)
    paths = study_paths(plan)
    if Path(__file__).resolve() != paths['runner_path'] or sha(__file__) != plan['runner_sha256']:
        raise ValueError('Run the exact externally pinned worker-v4 module')
    helpers = _runtime(plan['source_root'])  # Import once, only on manager thread.
    validate_owned(plan, helpers)
    with (SCOPE/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = paths['output_root']
        out.mkdir(parents=True, exist_ok=True)
        for name in ('jobs', 'checkpoints', 'partial', 'origins', 'locks', 'case_status'):
            (out/name).mkdir(exist_ok=True)
        snapshot = out/'plan.json'
        if snapshot.exists() and read(snapshot) != plan:
            raise ValueError('Retain the earlier plan; choose a new study identity')
        if not snapshot.exists():
            atomic(snapshot, plan)
        status_path = out/'study_status.json'
        old = read(status_path) if status_path.exists() else {}
        status = dict(schema='oran.environment-worker-status.v4', study_id=plan['study_id'], stage=plan['stage'],
            worker_id=plan['worker_id'], actual_system_hostname=socket.gethostname(), worker_pid=os.getpid(),
            plan_sha256=pin, source_sha256=plan['source_sha256'], source_archive_sha256=plan['source_archive_sha256'],
            runner_sha256=plan['runner_sha256'], max_parallel=plan.get('max_parallel', 2), done=[],
            failed=old.get('failed', []), stopped=[], running=[], queued=len(plan['cases']), completed=False,
            phase='initializing', models_unchanged=True, raw_data_retained=True, training_performed=False,
            scheduling_only_change=False, geometry_change=GEOMETRY_CHANGE, require_on_admission=True,
            base_config_pins={c['base_config']:c['base_config_sha256'] for c in plan['cases']},
            physical_parallel_parity_verified=False,
            post_deployment_kpis_used_for_model_training=False, environment_adoption_uses_validation_results=True)
        def write_status(**values):
            status.update(values, updated_unix_s=time.time(), free=helpers.resources(SCOPE))
            atomic(status_path, status)
        if status['failed'] and not retry_failed:
            write_status(phase='retained_failure_requires_review', completed=True)
            return status
        if retry_failed:
            atomic(out/('retained-failures-'+str(time.time_ns())+'.json'), status['failed'])
            status['failed'] = []
        cached = [c['id'] for c in plan['cases'] if completed(out, c, plan)]
        todo = [c for c in plan['cases'] if c['id'] not in cached]
        stop_event = Event()
        registry = OwnedChildren(stop_event)
        progress, progress_lock = {}, RLock()
        received_signals = []
        previous = {n:signal.getsignal(n) for n in (signal.SIGINT, signal.SIGTERM)}
        def received(number, frame):
            received_signals.append(number)
            stop_event.set()
        for number in previous:
            signal.signal(number, received)
        deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
        def reason():
            if received_signals:
                return 'signal_'+str(received_signals[-1])
            if (SCOPE/'STOP').exists() or (out/'STOP').exists():
                return 'stop_file'
            if time.time() >= deadline-plan.get('hard_stop_margin_s', 180):
                return 'deadline'
            return None
        def admit(case, active_count):
            estimate = case.get('estimated_wall_s', max(180, case['slots']*.12))
            if time.time()+estimate+plan.get('analysis_reserve_s', 0) > deadline:
                return dict(allowed=False, terminal=True, reason='budget_no_new_admission')
            lease = ui_admission(plan)
            if not (out/'ADMISSION_ALLOWED').is_file():
                return dict(allowed=False, reason='waiting_for_stage_admission')
            on_gate = on_admission(plan, case, out)
            if not on_gate['allowed']:
                return on_gate
            if not lease['allowed']:
                return dict(allowed=False, reason='waiting_for_user_ui')
            if registry.external(helpers.external_simulations):
                return dict(allowed=False, reason='waiting_for_external_simulation')
            if not resource_admission(helpers.resources(SCOPE)):
                return dict(allowed=False, reason='waiting_for_storage_or_memory')
            return dict(allowed=True)
        def execute(case):
            return execute_case(plan, case, out, helpers, registry, stop_event, progress, progress_lock)
        def update(schedule):
            with progress_lock:
                running = [deepcopy(progress.get(c['id'], c)) for c in schedule['running']]
            phase = ('draining_after_failure' if schedule['draining_after_failure'] else 'running' if running
                     else schedule['admission_reason'] or 'initializing')
            write_status(done=cached+schedule['done'], failed=schedule['failed'], stopped=schedule['stopped'],
                running=running, queued=len(schedule['queued']), phase=phase)
        try:
            outcome = schedule_cases(todo, max_parallel=plan.get('max_parallel', 2), execute=execute,
                admission=admit, stop_reason=reason, stop_event=stop_event, stop_children=registry.stop_all,
                publish=update, poll=plan.get('poll_seconds', 2))
            phase = ('failed_retained' if outcome['failed'] else 'stopped_retained' if outcome['stopped']
                     else outcome['termination'] or 'complete')
            write_status(phase=phase, completed=True, done=cached+outcome['done'], failed=outcome['failed'],
                stopped=outcome['stopped'], running=[], queued=len(outcome['queued']))
            return status
        except BaseException:
            stop_event.set()
            registry.stop_all()
            raise
        finally:
            try:
                registry.stop_all()
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    run_worker(args.worker_plan, args.plan_sha256, retry_failed=args.retry_failed)


if __name__ == '__main__':
    main()
