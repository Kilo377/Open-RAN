#!/usr/bin/env python3
"""Coordinate isolated pressure tests; preserve raw runs and yield to UI work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_rcp_perturbation import atomic

SCOPE = '/dev/shm/oran_environment_20261122'
POOL = Path('/Users/kilobao/Desktop/ORAN/SMO/evaluation-v2-20261121/workers-v2b/pool-config.json')
OLD_ARCHIVE = '/dev/shm/oran_quantification_20261122/source-v2.tgz'
RUNNER_SOURCE = ROOT/'experiments/environment_validation_worker.py'
SELECTED_WORKERS = None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ssh(worker, code, payload, *, timeout=90):
    # Large versioned runners/profiles exceed some SSH mux command limits.
    # Transport the structured payload through stdin, never a shell argument.
    bootstrap = 'import sys;sys.argv=[sys.argv[0],sys.stdin.buffer.read().decode()]\n'+code
    command = shlex.join(['python3', '-c', bootstrap])
    result = subprocess.run(['ssh', '-S', worker['socket'], '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=10', worker['host'], command],
        input=json.dumps(payload).encode(), capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(worker['id'] + ': ' + result.stderr.decode(errors='replace')[-2500:])
    return json.loads(result.stdout)


def plans(directory):
    result = []
    pool = {w['id']: w for w in json.loads(POOL.read_text())['workers']}
    for path in sorted(directory.glob('agx*.json')):
        plan = json.loads(path.read_text())
        if SELECTED_WORKERS is not None and plan['worker_id'] not in SELECTED_WORKERS:
            continue
        runner_module().validate_plan(plan, plan['plan_sha256'])
        worker = pool[plan['worker_id']]
        result.append((worker, plan))
    expected = {'agx1', 'agx2'} if SELECTED_WORKERS is None else SELECTED_WORKERS
    if len(result) != len(expected) or {p['worker_id'] for _, p in result} != expected:
        raise ValueError('Every explicitly assigned worker needs exactly one plan')
    return result


def prepare(directory, artifact, protocol):
    runner = RUNNER_SOURCE.read_bytes()
    profiles = {p.name: p.read_text() for p in (protocol/'profiles').glob('*.json')}
    code = r'''
import hashlib,json,os,shutil,sys,tarfile
from pathlib import Path
c=json.loads(sys.argv[1]);p=c['plan'];scope=Path(c['scope']);root=Path(p['source_root'])
if scope!=Path('/dev/shm/oran_environment_20261122') or root!=scope/'source':raise ValueError('Owned scope required')
archive=scope/'source-v2.tgz';old=Path(c['old_archive'])
if hashlib.sha256(old.read_bytes()).hexdigest()!=p['source_archive_sha256']:raise ValueError('Reusable archive pin mismatch')
scope.mkdir(exist_ok=True)
if archive.exists():
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=p['source_archive_sha256']:raise ValueError('Owned archive mismatch')
else:shutil.copyfile(old,archive)
if root.exists():
 marker=json.loads((root/'.smo-owned-worker.json').read_text())
 if marker['sha256']!=p['source_archive_sha256']:raise ValueError('Existing source copy differs')
else:
 root.mkdir()
 with tarfile.open(archive,'r:gz')as a:
  for m in a.getmembers():
   if Path(m.name).is_absolute()or'..'in Path(m.name).parts:raise ValueError('Unsafe source archive path')
  a.extractall(root)
 venv=Path(c['python']).parents[1]
 if not(venv/'bin/python').is_file():raise ValueError('Existing read-only venv missing')
 (root/'.venv-orin').symlink_to(venv,target_is_directory=True)
 (root/'.smo-owned-worker.json').write_text(json.dumps(dict(root=str(root),venv=str(venv),sha256=p['source_archive_sha256'])))
sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import source_hash,sha,atomic
if source_hash(root)!=p['source_sha256']:raise ValueError('Immutable physical source mismatch')
if any(sha(root/f)!=pin for f,pin in p['frozen_models'].items()):raise ValueError('Frozen model mismatch')
runner=Path(p.get('runner_path',scope/'runner/worker.py'))
if runner.parent.resolve()!=scope/'runner' or runner.suffix!='.py':raise ValueError('Use an owned versioned audit runner')
runner.parent.mkdir(exist_ok=True);raw=c['runner'].encode()
if hashlib.sha256(raw).hexdigest()!=p['runner_sha256']:raise ValueError('New audit runner pin mismatch')
if runner.exists():
 if runner.read_bytes()!=raw:raise ValueError('Do not replace a previously pinned runner')
else:runner.write_bytes(raw)
profiles=scope/'protocol/profiles';profiles.mkdir(parents=True,exist_ok=True)
for name,text in c['profiles'].items():
 if Path(name).name!=name:raise ValueError('Unsafe profile path')
 target=profiles/name
 if target.exists():
  if target.read_text()!=text:raise ValueError('Do not replace pinned profile')
 else:target.write_text(text)
out=Path(p.get('output_root',root/'outputs/environment_validation'/p['stage']));out.mkdir(parents=True,exist_ok=True)
saved=out/'plan.json'
if saved.exists():
 if json.loads(saved.read_text())!=p:raise ValueError('Do not overwrite a stage plan')
else:atomic(saved,p)
active=Path(p.get('active_plan_path',scope/'plans'/(p['stage']+'.json')))
if not active.resolve().is_relative_to(scope/'plans') or active.suffix!='.json':raise ValueError('Use an external owned active plan')
if active.exists():
 if json.loads(active.read_text())!=p:raise ValueError('Do not overwrite active stage plan')
else:atomic(active,p)
print(json.dumps(dict(prepared=True,output=str(out),source_sha256=source_hash(root),model_count=len(p['frozen_models']),runner_sha256=sha(runner))))
'''
    for worker, plan in plans(directory):
        if plan['runner_sha256'] != digest(runner):
            raise ValueError('Plan must pin the reviewed runner bytes')
        reply = ssh(worker, code, dict(plan=plan, scope=SCOPE, old_archive=OLD_ARCHIVE,
            python=worker['python'], runner=runner.decode(), profiles=profiles), timeout=180)
        atomic(artifact/'deployment'/worker['id']/'worker-config.json', {**worker,
            'project_root': plan['source_root'], 'output_subdir': plan['stage'],
            'plan_sha256': plan['plan_sha256'], **reply})
        print(json.dumps(dict(worker=worker['id'], **reply)), flush=True)


def runner_module():
    spec = importlib.util.spec_from_file_location('environment_worker', RUNNER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def heartbeat(directory, ui_url):
    with urllib.request.urlopen(ui_url.rstrip('/')+'/api/deployment', timeout=10) as response:
        snapshot = runner_module().ui_snapshot(json.load(response))
    code = r'''
import json,os,sys
from pathlib import Path
c=json.loads(sys.argv[1]);p=Path(c['scope'])/'ui-admission.json';tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
tmp.write_text(json.dumps(c['snapshot']));tmp.replace(p)
print(json.dumps(dict(worker=c['id'],blocked=c['snapshot'].get('blocked_worker_ids'),sampled=c['snapshot'].get('sampled_unix_s'))))
'''
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(lambda wp: ssh(wp[0], code, dict(scope=SCOPE,
            id=wp[0]['id'], snapshot=snapshot)), plans(directory)))


def launch(directory):
    code = r'''
import json,os,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]);p=c['plan'];scope=Path(c['scope']);root=Path(p['source_root']);out=Path(p.get('output_root',root/'outputs/environment_validation'/p['stage']));runner=Path(p.get('runner_path',scope/'runner/worker.py'))
sys.dont_write_bytecode=True;sys.path.insert(0,p['source_root'])
from experiments.quantification_study import isolated_environment
status=out/'study_status.json';s=json.loads(status.read_text())if status.exists()else{}
pid=s.get('worker_pid')
if isinstance(pid,int):
 try:
  cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
  if str(runner).encode()in cmd:print(json.dumps(dict(pid=pid,existing=True)));sys.exit(0)
 except FileNotFoundError:pass
if s.get('completed'):
 print(json.dumps(dict(completed=True,existing=True)));sys.exit(0)
(out/'ADMISSION_ALLOWED').touch()
env=isolated_environment(Path(p['source_root']))
for key in('tmp','.cache'):(Path(p['source_root'])/key).mkdir(exist_ok=True)
with(out/'worker.log').open('ab')as log:
 process=subprocess.Popen([c['python'],'-u',str(runner),'--worker-plan',str(p.get('active_plan_path',scope/'plans'/(p['stage']+'.json'))),'--plan-sha256',p['plan_sha256']],
  cwd=p['source_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps(dict(pid=process.pid,existing=False)))
'''
    for worker, plan in plans(directory):
        print(json.dumps(dict(worker=worker['id'], **ssh(worker, code,
            dict(plan=plan, scope=SCOPE, python=worker['python'])))), flush=True)


def mirror(worker, plan, artifact):
    dest = artifact/'workers'/worker['id']
    dest.mkdir(parents=True, exist_ok=True)
    shell = shlex.join(['ssh', '-S', worker['socket'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10'])
    remote = worker['host']+':'+str(plan.get('output_root',Path(plan['source_root'])/'outputs/environment_validation'/plan['stage']))+'/'
    reply = subprocess.run(['rsync', '-a', '--partial', '--exclude', '*.tmp',
        '-e', shell, remote, str(dest)+'/'], capture_output=True, text=True, timeout=180)
    if reply.returncode not in (0, 24):
        raise RuntimeError(reply.stderr[-1500:])
    verified_path = dest/'received-checkpoints.json'
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    state_path = dest/'study_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    final_verification = state.get('completed') is True
    for path in sorted((dest/'checkpoints').glob('*.json')):
        seal_pin = digest(path.read_bytes())
        if not final_verification and verified.get(path.stem, {}).get('seal_sha256') == seal_pin:
            continue
        seal = json.loads(path.read_text())
        expected_case = next((c for c in plan['cases'] if c['id'] == path.stem), None)
        if (seal.get('completed') is not True or not seal.get('files_sha256')
                or not expected_case or seal.get('case') != expected_case
                or seal.get('plan_sha256') != plan['plan_sha256']
                or seal.get('source_sha256') != plan['source_sha256']
                or seal.get('runner_sha256') != plan['runner_sha256']):
            raise ValueError('Mirrored checkpoint identity differs from the pinned plan')
        folder = dest/'jobs'/path.stem
        def bad_files():
            result = []
            for name, pin in seal['files_sha256'].items():
                target = (folder/name).resolve()
                if not target.is_relative_to(folder.resolve()):
                    raise ValueError('Unsafe checkpoint relative path')
                if not target.is_file() or digest(target.read_bytes()) != pin:
                    result.append(name)
            return result
        if bad_files():
            subprocess.run(['rsync', '-ac', '--partial', '-e', shell,
                remote+'jobs/'+path.stem+'/', str(folder)+'/'], capture_output=True,
                timeout=180, check=True)
        bad = bad_files()
        if bad:
            raise ValueError('Mirrored checksum mismatch: '+str(bad))
        verified[path.stem] = dict(seal_sha256=seal_pin, files=len(seal['files_sha256']),
            verified_unix_s=time.time(), case=seal['case'], wall_s=seal.get('wall_s'))
    atomic(verified_path, verified)
    return dict(worker_id=worker['id'], phase=state.get('phase'),
        done=len(state.get('done', [])), verified=len(verified), failed=state.get('failed', []),
        running=state.get('running', []), queued=state.get('queued'),
        completed=state.get('completed'), free=state.get('free'),
        expected_cases=len(plan['cases']),
        all_checkpoints_verified=set(verified) == {c['id'] for c in plan['cases']})


def monitor(directory, artifact, ui_url, once=False):
    artifact.mkdir(parents=True, exist_ok=True)
    lock = (artifact/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    last = None
    while True:
        # Failing to reach the UI deliberately lets the admission lease expire.
        error = None
        try:
            heartbeat(directory, ui_url)
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(mirror, w, p, artifact) for w, p in plans(directory)]
            results = []
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as exc:
                    results.append(dict(error=repr(exc)))
        status = dict(workers=results, ui_heartbeat_error=error, updated_unix_s=time.time())
        atomic(artifact/'coordinator-status.json', status)
        signature = [(r.get('worker_id'), r.get('phase'), r.get('done'), r.get('verified'),
            [(c.get('name'), c.get('slot')) for c in r.get('running', [])]) for r in results]
        if signature != last:
            print(json.dumps(status), flush=True)
            last = signature
        terminal = len(results) == len(plans(directory)) and all(r.get('completed') for r in results)
        success = terminal and all(not r.get('failed') and r.get('all_checkpoints_verified') for r in results)
        if success:
            atomic(artifact/'coordinator-completion.json', dict(**status,
                completed=True, successful=True, raw_mirror_verified=True))
        if terminal and any(r.get('failed') or r.get('phase') != 'complete' for r in results):
            atomic(artifact/'coordinator-completion.json', dict(**status,
                completed=True, successful=False, raw_mirror_verified=False))
            raise RuntimeError('An isolated worker failed; partial data and logs were retained')
        if once or success:
            break
        time.sleep(25)


def main():
    global RUNNER_SOURCE, SELECTED_WORKERS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans-dir', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--ui-url', default='http://127.0.0.1:53505')
    p.add_argument('--runner-source', type=Path, default=RUNNER_SOURCE,
        help='Versioned external audit wrapper; the physical simulator source stays immutable.')
    p.add_argument('--worker', choices=['agx1', 'agx2'], action='append',
        help='Explicit subset for a host-local parity study; default requires both hosts.')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--launch', action='store_true')
    p.add_argument('--monitor', action='store_true')
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    RUNNER_SOURCE = a.runner_source.resolve()
    SELECTED_WORKERS = set(a.worker) if a.worker else None
    if a.prepare:
        prepare(a.plans_dir, a.artifact, a.protocol)
    if a.launch:
        heartbeat(a.plans_dir, a.ui_url)
        launch(a.plans_dir)
    if a.monitor or a.once:
        monitor(a.plans_dir, a.artifact, a.ui_url, once=a.once)


if __name__ == '__main__':
    main()
