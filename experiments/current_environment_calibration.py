"""Fixed, preregistered P–K interventions over an unchanged simulator snapshot.

The probe is an external experimental instrument. It substitutes only the
timeline adapter, uses normal RIC contracts/admission/delay, never reads KPIs,
and never invokes a trained controller. Both instrument and source are pinned.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = Path('/dev/shm/oran_smo_selected_20261123/v1/source')
SCOPE = ROOT/'outputs/pk_calibration_20261123_v2'
FIELDS = frozenset(('ue_offload_fraction', 'ue_drx_configs', 'scheduler_mode',
                    'ue_scheduler_weights', 'cell_sleep_mode'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def probe(root, argv):
    sys.path.insert(0, str(root))
    from ric.near_rt.xapp_base import XAppBase
    from ric.xapp_contract import XAppContract, validate_action
    from models.action import ConfigAction, ControlDecision
    import experiments.fixed_control_timeline as adapter

    class FixedParameterProbe(XAppBase):
        def __init__(self, path, interval=5):
            super().__init__('fixed_control_timeline', 100, list(FIELDS), contract=XAppContract((), FIELDS))
            self.spec = read(path)
            self.sent = False
            self.inputs = []
            if set(self.spec) != {'switch_slot', 'vector', 'plan_sha256', 'arm'}:
                raise ValueError('Unknown fixed experimental stimulus')
            if set(self.spec['vector']) != FIELDS:
                raise ValueError('A complete fixed parameter vector is required')

        def run(self, observation, policy_snapshot):
            slot = observation.slot_id
            if self.sent or slot < self.spec['switch_slot']:
                return ControlDecision()
            if slot != self.spec['switch_slot']:
                raise ValueError('Missed preregistered RIC boundary')
            self.sent = True
            vector = self.spec['vector']
            actions = []
            for cell in observation.cell_observations:
                members = {u.ue_id for u in observation.ue_observations if u.serving_cell_id == cell.cell_id}
                values = {}
                for field, value in vector.items():
                    if field in ('scheduler_mode', 'cell_sleep_mode'):
                        values[field] = value[cell.cell_id]
                    else:
                        values[field] = {u: deepcopy(v) for u, v in value.items() if u in members}
                actions.append(ConfigAction(source_name=self.name, target_cell_id=cell.cell_id,
                    effective_from_slot=slot, effective_until_slot=slot+39, **values))
            errors = [e for a in actions for e in validate_action(a, self.contract, observation)]
            if errors:
                raise ValueError('Fixed probe contract failure: '+str(errors))
            self.inputs.append(dict(slot=slot, stimulus=deepcopy(self.spec), kpis_used=False))
            return ControlDecision(config_actions=actions)

    adapter.FixedControlTimeline = FixedParameterProbe
    sys.argv = [str(root/'experiments/run_rt_sys.py'), *argv]
    runpy.run_path(str(root/'experiments/run_rt_sys.py'), run_name='__main__')


def resources():
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return dict(available_memory_bytes=int(mem['MemAvailable'].split()[0])*1024,
        free_ram_bytes=os.statvfs(SCOPE).f_bavail*os.statvfs(SCOPE).f_frsize,
        free_system_bytes=os.statvfs('/').f_bavail*os.statvfs('/').f_frsize)


def environment(case_id):
    env = dict(os.environ)
    cache = SCOPE/'cache'/case_id
    names = {'TMPDIR':'tmp', 'TMP':'tmp', 'TEMP':'tmp', 'XDG_CACHE_HOME':'xdg-cache',
        'XDG_CONFIG_HOME':'xdg-config', 'XDG_DATA_HOME':'xdg-data', 'TORCH_HOME':'torch',
        'TORCH_EXTENSIONS_DIR':'extensions', 'TORCHINDUCTOR_CACHE_DIR':'inductor',
        'TRITON_CACHE_DIR':'triton', 'NUMBA_CACHE_DIR':'numba', 'MESA_SHADER_CACHE_DIR':'mesa',
        'HF_HOME':'huggingface', 'PIP_CACHE_DIR':'pip', 'UV_CACHE_DIR':'uv',
        'PYTHONPYCACHEPREFIX':'pycache', 'JAX_COMPILATION_CACHE_DIR':'jax'}
    for key, name in names.items():
        path = cache/name
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    # activate-gpu.sh intentionally uses these exact owned RAM paths.
    for key, name in {'CUDA_CACHE_PATH':'cuda-orin', 'CUPY_CACHE_DIR':'cupy',
                      'DRJIT_CACHE_DIR':'drjit', 'MPLCONFIGDIR':'matplotlib'}.items():
        env[key] = str(ROOT/'.cache'/name)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def worker(plan_path):
    plan_path = Path(plan_path).resolve()
    if plan_path.parent != SCOPE:
        raise ValueError('Use the dedicated RAM campaign scope')
    plan = read(plan_path)
    pin = plan.pop('plan_sha256')
    if digest(plan) != pin or sha(__file__) != plan['driver_sha256']:
        raise ValueError('Preregistered calibration or driver changed')
    sys.path.insert(0, str(ROOT))
    from smo.runtime_recipe import source_pins
    if source_pins(ROOT) != plan['source_pins']:
        raise ValueError('Network dynamics differ from the current UI environment')
    if read(ROOT/'.smo-owned-worker.json')['sha256'] != plan['source_archive_sha256']:
        raise ValueError('Sealed source archive differs')
    for name, item in plan['profiles'].items():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Current environment profile changed: '+name)
    for name, pin_model in plan['frozen_models'].items():
        if sha(ROOT/name) != pin_model:
            raise ValueError('Frozen model changed')
    lock = (SCOPE/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = SCOPE/'status.json'
    if status_path.exists():
        raise ValueError('Do not overwrite an earlier calibration worker')
    state = dict(worker_id=plan['worker_id'], pid=os.getpid(), plan_sha256=pin,
        phase='initializing', completed=False, done=[], failed=[], running=[],
        queued=len(plan['cases']), kpis_used_for_action_selection=False)
    mutex = threading.Lock()
    stopped = threading.Event()
    children = {}

    def publish():
        atomic(status_path, {**deepcopy(state), 'updated_unix_s':time.time(), 'free':resources()})

    def execute(case):
        job = SCOPE/'jobs'/case['name']
        if job.exists():
            raise ValueError('Calibration cannot overwrite raw evidence')
        while True:
            free = resources()
            if stopped.is_set() or (SCOPE/'STOP').exists():
                raise RuntimeError('Campaign stopped before admission')
            if free['available_memory_bytes'] > 14*1024**3 and free['free_ram_bytes'] > 4*1024**3 and free['free_system_bytes'] > 256*1024**2:
                break
            time.sleep(3)
        job.mkdir(parents=True)
        spec = dict(switch_slot=plan['switch_slot'], vector=case['vector'], plan_sha256=pin, arm=case['arm'])
        atomic(job/'stimulus.json', spec)
        atomic(job/'case.json', case)
        run = job/'run'
        args = ['--device','cuda','--rt-device','cpu','--scene','munich','--cells','3','--ues','6',
            '--algebra-backend','torch','--nr-execution','batched','--recording-mode','experiment',
            '--diagnostic-window-slots','40','--model','fast','--fast-profile','system',
            '--slots',str(plan['total_slots']),'--seed',str(case['seed']),
            '--traffic-load',str(plan['traffic_load']),'--ue-services','drx-mec','--xapp-pool','chongyu',
            '--record-compression','gzip','--control-timeline',str(job/'stimulus.json'),
            '--no-html','--channel-cache-root',str(SCOPE/'channels'/str(case['seed'])),
            '--live-state',str(run/'live.json'),'--output',str(run)]
        for name, flag in [('base_config','--config'),('ue_services_config','--ue-services-config'),
            ('traffic_profile','--traffic-profile'),('ap_services_profile','--ap-services-profile')]:
            args += [flag, plan['profiles'][name]['path']]
        command = ['bash','-c','source "$1"; shift; exec python "$@"','calibration',
            str(ROOT/'scripts/orin/activate-gpu.sh'),str(Path(__file__).resolve()),'--probe',str(ROOT),*args]
        env = environment(case['name'])
        atomic(job/'command.json', dict(argv=command, environment={k:v for k,v in env.items() if str(SCOPE) in v or str(ROOT/'.cache') in v},
            driver_sha256=plan['driver_sha256'], source_pins=plan['source_pins']))
        started = time.time()
        progress = dict(name=case['name'], arm=case['arm'], seed=case['seed'], slot=0, started_unix_s=started)
        with (job/'simulation.log').open('wb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            with mutex:
                children[case['name']] = child
                state['running'].append(progress)
                state['queued'] -= 1
                state['phase'] = 'running'
                publish()
            while child.poll() is None:
                if stopped.is_set() or (SCOPE/'STOP').exists():
                    os.killpg(child.pid, signal.SIGINT)
                try:
                    progress['slot'] = read(run/'live.json').get('slot', 0)
                except (OSError, ValueError):
                    pass
                with mutex:
                    publish()
                time.sleep(2)
        try:
            result = read(run/'run_status.json')
            if child.returncode != 0 or not result.get('completed'):
                raise ValueError('Physical calibration did not complete: '+str(child.returncode))
            atomic(job/'seal.json', dict(completed=True, case=case, plan_sha256=pin,
                driver_sha256=plan['driver_sha256'], source_pins=plan['source_pins'],
                source_archive_sha256=plan['source_archive_sha256'], wall_s=time.time()-started,
                files_sha256={str(f.relative_to(job)):sha(f) for f in job.rglob('*') if f.is_file()},
                learned_xapps_invoked=False, combination_evaluation_data_used=False))
            with mutex:
                state['done'].append(case['name'])
        except Exception as exc:
            stopped.set()
            with mutex:
                state['failed'].append(dict(name=case['name'], error=str(exc)))
            raise
        finally:
            with mutex:
                children.pop(case['name'], None)
                state['running'].remove(progress)
                publish()

    publish()
    # References precede their interventions. Each paired seed stays on one host.
    try:
        for role in ('reference', 'intervention'):
            with ThreadPoolExecutor(max_workers=plan['max_parallel']) as pool:
                futures = [pool.submit(execute, c) for c in plan['cases'] if c['role'] == role]
                for future in as_completed(futures):
                    future.result()
        state.update(phase='complete', completed=True)
    except BaseException as exc:
        stopped.set()
        state.update(phase='failed_retained', completed=True, error=str(exc))
    finally:
        publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--worker-plan', type=Path)
    own, rest = parser.parse_known_args()
    if own.probe:
        probe(own.probe.resolve(), rest)
    elif own.worker_plan and not rest:
        worker(own.worker_plan)
    else:
        raise SystemExit('Choose --probe or --worker-plan')
