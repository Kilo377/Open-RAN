#!/usr/bin/env python3
"""Prepare/coordinate the separately pinned indexed source; no implicit ON/source adoption."""
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
RUNNER_SOURCE = ROOT/'experiments/environment_validation_worker_v5.py'
SOURCE_ARCHIVE = Path('/Users/kilobao/Desktop/ORAN/SMO/environment-validation-20261122/performance-prototype/source-indexed-v1.tgz')
BASELINE_ARCHIVE = Path('/Users/kilobao/Desktop/ORAN/SMO/quantification-study-20261122/versions/v2/source.tgz')
SOURCE_EQUIVALENCE = SOURCE_ARCHIVE.parent/'protocol/source-equivalence-candidate-v1.json'
SOURCE_APPROVAL = SOURCE_ARCHIVE.parent/'protocol/source-approval-parity-v2.json'
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


def archive_members(path):
    import tarfile
    members={}
    with tarfile.open(path,'r:gz')as archive:
        for item in archive.getmembers():
            relative=Path(item.name)
            if relative.is_absolute()or'..'in relative.parts:
                raise ValueError('Unsafe source archive path')
            if item.isdir():continue
            if not item.isfile()or item.name in members:
                raise ValueError('Only unique safe regular source members are accepted')
            members[item.name]=archive.extractfile(item).read()
    return members


def validate_source_inputs():
    runner=runner_module()
    for path,pin in[(SOURCE_ARCHIVE,runner.ARCHIVE_SHA256),(BASELINE_ARCHIVE,runner.BASELINE_ARCHIVE_SHA256),
                     (SOURCE_EQUIVALENCE,runner.SOURCE_EQUIVALENCE_SHA256),(SOURCE_APPROVAL,runner.SOURCE_TRIAL_APPROVAL_SHA256)]:
        if digest(path.read_bytes())!=pin:raise ValueError('Local source/proof pin differs: '+str(path))
    proof=json.loads(SOURCE_EQUIVALENCE.read_text());approval=json.loads(SOURCE_APPROVAL.read_text())
    original,candidate=archive_members(BASELINE_ARCHIVE),archive_members(SOURCE_ARCHIVE)
    oldpins={k:digest(v)for k,v in original.items()};newpins={k:digest(v)for k,v in candidate.items()}
    changed={k for k in oldpins.keys()|newpins.keys()if oldpins.get(k)!=newpins.get(k)}
    if changed!=runner.INDEXED_MEMBERS or changed!=set(proof['changed_regular_members']):
        raise ValueError('Only the declared three bookkeeping source members may differ')
    for path,row in proof['changed_regular_members'].items():
        if row['baseline_sha256']!=oldpins.get(path)or row['sha256']!=newpins.get(path):
            raise ValueError('Declared source diff does not match actual archives: '+path)
    for path,pin in proof['frozen_models'].items():
        if oldpins.get(path)!=pin or newpins.get(path)!=pin:raise ValueError('Frozen trained model changed')
    for key in('oracle_proof','exact_patch'):
        reference=proof[key]
        if digest(Path(reference['path']).read_bytes())!=reference['sha256']:
            raise ValueError('Pinned prototype oracle/patch evidence changed')
    if (proof['source_sha256']!=runner.SOURCE_SHA256 or proof['source_archive_sha256']!=runner.ARCHIVE_SHA256
            or proof['all_other_regular_members_equal']is not True or proof['oracle_verified']is not True
            or approval['approved']is not True or approval['candidate_proof']['sha256']!=runner.SOURCE_EQUIVALENCE_SHA256
            or approval['physical_parity_verified']is not False):
        raise ValueError('The source contract is parity-only and must remain truthful')
    return dict(source_sha256=runner.SOURCE_SHA256,source_archive_sha256=runner.ARCHIVE_SHA256,
        baseline_regular_members=len(original),candidate_regular_members=len(candidate),changed_members=sorted(changed),
        physical_parity_claimed=False)


def prepare(directory, artifact, protocol):
    contract=validate_source_inputs();runner=RUNNER_SOURCE.read_bytes()
    profiles={p.name:p.read_text()for p in(protocol/'profiles').glob('*.json')}
    # All prepare writes stay in the existing dedicated RAM scope/new physical root.
    stage_code=r"""
import hashlib,json,sys
from pathlib import Path
c=json.loads(sys.argv[1]);scope=Path(c['scope']);p=c['plan']
if scope!=Path('/dev/shm/oran_environment_20261122')or Path(p['source_root'])!=scope/'source-indexed-v1':raise ValueError('Separate indexed root required')
scope.mkdir(exist_ok=True);uploads=scope/'.uploads';uploads.mkdir(exist_ok=True)
archive=Path(p['source_archive_path']);expected=scope/'source-indexed-v1.tgz'
if archive!=expected:raise ValueError('Indexed archive must not replace old CBE archive')
if archive.exists():
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=p['source_archive_sha256']:raise ValueError('Do not replace a pinned indexed archive')
 print(json.dumps(dict(upload_required=False,staging=str(uploads/(p['source_archive_sha256']+'.tgz')))))
else:print(json.dumps(dict(upload_required=True,staging=str(uploads/(p['source_archive_sha256']+'.tgz')))))
"""
    code=r"""
import hashlib,json,os,shutil,sys,tarfile,time
from pathlib import Path
c=json.loads(sys.argv[1]);p=c['plan'];scope=Path(c['scope']);root=Path(p['source_root']);archive=Path(p['source_archive_path'])
if scope!=Path('/dev/shm/oran_environment_20261122')or root!=scope/'source-indexed-v1'or archive!=scope/'source-indexed-v1.tgz':raise ValueError('Owned separate indexed source required')
if not archive.exists():
 staged=Path(c['staging'])
 if staged.parent!=scope/'.uploads'or hashlib.sha256(staged.read_bytes()).hexdigest()!=p['source_archive_sha256']:raise ValueError('Uploaded candidate differs')
 staged.replace(archive)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=p['source_archive_sha256']:raise ValueError('Candidate archive pin mismatch')
if root.exists():
 marker=json.loads((root/'.smo-owned-worker.json').read_text())
 if marker['root']!=str(root)or marker['sha256']!=p['source_archive_sha256']:raise ValueError('Existing indexed source copy differs')
else:
 partial=scope/('.source-indexed-extract-'+str(time.time_ns()));partial.mkdir()
 with tarfile.open(archive,'r:gz')as a:
  names=set()
  for member in a.getmembers():
   name=Path(member.name)
   if name.is_absolute()or'..'in name.parts or not member.isfile()or member.name in names:raise ValueError('Unsafe source archive member')
   names.add(member.name)
  a.extractall(partial)
 venv=Path(c['python']).parents[1]
 if not(venv/'bin/python').is_file():raise ValueError('Reuse the existing read-only environment')
 (partial/'.venv-orin').symlink_to(venv,target_is_directory=True)
 (partial/'.smo-owned-worker.json').write_text(json.dumps(dict(root=str(root),venv=str(venv),sha256=p['source_archive_sha256'])))
 partial.replace(root)
sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import source_hash,sha,atomic
if source_hash(root)!=p['source_sha256']:raise ValueError('Actual indexed source fingerprint differs')
if any(sha(root/name)!=pin for name,pin in p['frozen_models'].items()):raise ValueError('Frozen model changed')
def install(path,raw):
 path=Path(path)
 if not path.resolve().is_relative_to(scope)or path.resolve()==scope:raise ValueError('Owned input path required')
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=raw:raise ValueError('Do not overwrite pinned input: '+str(path))
 else:path.write_bytes(raw)
runner=Path(p['runner_path'])
if runner!=scope/'runner/worker-v5.py'or hashlib.sha256(c['runner'].encode()).hexdigest()!=p['runner_sha256']:raise ValueError('External v5 runner mismatch')
install(runner,c['runner'].encode())
install(p['source_equivalence_path'],c['equivalence'].encode());install(p['source_trial_approval_path'],c['approval'].encode())
for name,text in c['profiles'].items():
 if Path(name).name!=name:raise ValueError('Unsafe profile name')
 install(scope/'protocol/profiles'/name,text.encode())
import importlib.util
spec=importlib.util.spec_from_file_location('indexed_worker_preflight',runner);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.validate_plan(p,p['plan_sha256']);module.validate_owned(p,module._runtime(root))
out=Path(p['output_root']);out.mkdir(parents=True,exist_ok=True)
install(out/'plan.json',(json.dumps(p,indent=2)+'\n').encode())
install(p['active_plan_path'],(json.dumps(p,indent=2)+'\n').encode())
print(json.dumps(dict(prepared=True,output=str(out),source_sha256=source_hash(root),source_archive_sha256=sha(archive),model_count=len(p['frozen_models']),runner_sha256=sha(runner),source_equivalence_sha256=sha(p['source_equivalence_path']),physical_parity_claimed=False)))
"""
    for worker,plan in plans(directory):
        if plan['runner_sha256']!=digest(runner):raise ValueError('Plan must pin the final reviewed v5 runner')
        if plan['source_sha256']!=contract['source_sha256']or plan['source_archive_sha256']!=contract['source_archive_sha256']:
            raise ValueError('The plan must use the audited actual indexed source pins')
        staging=ssh(worker,stage_code,dict(plan=plan,scope=SCOPE))
        if staging['upload_required']:
            subprocess.run(['scp','-o','ControlPath='+worker['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=10',
                str(SOURCE_ARCHIVE),worker['host']+':'+staging['staging']],check=True,capture_output=True,timeout=180)
        reply=ssh(worker,code,dict(plan=plan,scope=SCOPE,staging=staging['staging'],python=worker['python'],runner=runner.decode(),
            profiles=profiles,equivalence=SOURCE_EQUIVALENCE.read_text(),approval=SOURCE_APPROVAL.read_text()),timeout=240)
        atomic(artifact/'deployment'/worker['id']/'worker-config.json',{**worker,'project_root':plan['source_root'],
            'output_subdir':plan['stage'],'plan_sha256':plan['plan_sha256'],**reply})
        print(json.dumps(dict(worker=worker['id'],**reply)),flush=True)


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
    global RUNNER_SOURCE, SELECTED_WORKERS, SOURCE_ARCHIVE, BASELINE_ARCHIVE, SOURCE_EQUIVALENCE, SOURCE_APPROVAL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans-dir', type=Path)
    p.add_argument('--artifact', type=Path)
    p.add_argument('--protocol', type=Path)
    p.add_argument('--ui-url', default='http://127.0.0.1:53505')
    p.add_argument('--runner-source', type=Path, default=RUNNER_SOURCE,
        help='Versioned external audit wrapper; the physical simulator source stays immutable.')
    p.add_argument('--worker', choices=['agx1', 'agx2'], action='append',
        help='Explicit subset for a host-local parity study; default requires both hosts.')
    p.add_argument('--source-archive',type=Path,default=SOURCE_ARCHIVE)
    p.add_argument('--baseline-archive',type=Path,default=BASELINE_ARCHIVE)
    p.add_argument('--source-equivalence',type=Path,default=SOURCE_EQUIVALENCE)
    p.add_argument('--source-approval',type=Path,default=SOURCE_APPROVAL)
    p.add_argument('--preflight-only',action='store_true',help='Validate local true source/proof pins; no SSH or simulation')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--launch', action='store_true')
    p.add_argument('--monitor', action='store_true')
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    RUNNER_SOURCE = a.runner_source.resolve()
    SOURCE_ARCHIVE=a.source_archive.resolve();BASELINE_ARCHIVE=a.baseline_archive.resolve()
    SOURCE_EQUIVALENCE=a.source_equivalence.resolve();SOURCE_APPROVAL=a.source_approval.resolve()
    if a.preflight_only:
        print(json.dumps(validate_source_inputs(),sort_keys=True));return
    if any(x is None for x in(a.plans_dir,a.artifact,a.protocol)):
        p.error('--plans-dir, --artifact and --protocol are required except for --preflight-only')
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
