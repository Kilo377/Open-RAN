#!/usr/bin/env python3
"""Adopt the full native run, then train five real warm-start burst episodes.

Only process boundaries change: 10 s native + five 4 s burst runs = 30 s.
The frozen evaluator uses the reusable native membership audit and a pinned plan.
Incomplete burst runs are retained and never silently replayed or rolled back.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.run_urllc_qos_study import PROFILE, audit_case, command_for
from experiments import run_urllc_qos_study as study
from experiments.run_xapp_long_validation import atomic_json, load, sha256, source_fingerprint

PINNED_SOURCE = '9d77a27b3765559ab81184dac443e1dd01e45391f651cedcd9e591ae7e1fc372'
MIN_FREE = 1024**3
RUNTIME_FLOOR = 256*1024**2
from experiments.urllc_control_audit import AUDIT_VERSION, classify_stale_rejections

ORCHESTRATOR_VERSION = 'native-membership-audit-v2'


def require_membership_rejection_audit():
    """Require the reusable native auditor; never monkeypatch record readers."""
    from experiments import run_urllc_qos_study as study
    if getattr(study, 'CONTROLLER_AUDIT_VERSION', None) != AUDIT_VERSION:
        raise ValueError('Native reusable membership rejection audit is required')


def episodes():
    result = []
    for name, scenario, seed, seconds, balanced in [('native', 'native', 179, 10., 40)] + [
            ('burst_'+str(i), 'burst', 180+i, 4., 8) for i in range(5)]:
        relative = 'training/'+name
        result.append(dict(name=name, scenario=scenario, seed=seed, seconds=seconds,
            balanced_start_windows=balanced, case_dir=relative,
            before_model=None if name == 'native' else relative+'.model-before.json',
            after_model=relative+'.model-after.json', audit=relative+'.audit.json'))
    return result


def manifest(plan):
    return dict(version=2, source_sha256=PINNED_SOURCE,
        orchestrator_sha256=sha256(Path(__file__)), orchestrator_version=ORCHESTRATOR_VERSION, auditor_version=AUDIT_VERSION, profile_sha256=plan['profile_sha256'],
        profiles_sha256=plan['profiles_sha256'],
        total_seconds=30., native_seconds=10., burst_seconds=20.,
        evaluation_seeds=plan['evaluation_seeds'],
        evaluation_scenarios=plan['evaluation_scenarios'], eval_seconds=plan['eval_seconds'], episodes=episodes(),
        scope='Six real online episodes; exact saved model passed between runs; no result filtering',
        labels='Outcome windows including initial/carryover state, not independent packet SLA probabilities')


def exact_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.tmp.'+str(os.getpid()))
    with temporary.open('wb') as stream:
        stream.write(source.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, target)
    if sha256(source) != sha256(target):
        raise ValueError('Exact model copy mismatch')


def pid_running(pid, expected_folder):
    """Reject PID reuse/unrelated processes; a Linux zombie has already exited."""
    path = Path('/proc')/str(pid)
    if not path.exists():
        return False
    try:
        stat = (path/'stat').read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z':
            return False
        command = (path/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    values = [p.decode(errors='replace') for p in command if p]
    if not any(v.endswith('experiments/run_rt_sys.py') for v in values) or '--output' not in values:
        raise ValueError('Native PID belongs to an unrelated process')
    if Path(values[values.index('--output')+1]).resolve() != expected_folder.resolve():
        raise ValueError('Native PID output differs from the adopted case')
    return True


class Status:
    def __init__(self, output):
        self.output = output
        self.values = dict(stage='starting', parent_pid=os.getpid(), source_sha256=PINNED_SOURCE)
        self.lock = threading.Lock(); self.stop = threading.Event()
        self.main_owned = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def write(self, **values):
        with self.lock:
            self.values.update(values)
            payload = dict(self.values, heartbeat_unix=time.time(),
                free_disk_bytes=shutil.disk_usage(self.output).free)
            atomic_json(self.output/'partitioned-status.json', payload)
            if self.main_owned:
                atomic_json(self.output/'status.json', payload)

    def _loop(self):
        while not self.stop.wait(5):
            self.write()

    def __enter__(self):
        self.write(); self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join()


def verify_source(plan):
    if plan['source_sha256'] != PINNED_SOURCE or source_fingerprint(ROOT) != PINNED_SOURCE:
        raise ValueError('Pinned source changed; existing data retained')
    profiles = {name: sha256(path) for name, path in study.PROFILES.items() if path is not None}
    if sha256(PROFILE) != plan['profile_sha256'] or profiles != plan['profiles_sha256']:
        raise ValueError('Pinned traffic profiles changed')


def check_recorded_inputs(folder, episode, plan):
    cfg = load(folder/'config_snapshot.json')
    run_status = load(folder/'run_status.json')
    if not run_status.get('completed'):
        raise ValueError('Incomplete episode retained; will not replay '+str(folder))
    actual_s = cfg['manifest']['total_slots'] * cfg['manifest']['slot_duration_ms']/1000
    if (cfg['source_sha256'] != plan['source_sha256'] or actual_s != episode['seconds']
            or cfg['manifest']['rng_seed'] != episode['seed']
            or cfg['configs']['sim']['rng_seed'] != episode['seed']):
        raise ValueError('Episode source/duration mismatch')
    apps = cfg['configs']['ric'].get('xapps', [])
    if (len(apps) != 1 or apps[0].get('parameters', {}).get('balanced_start_windows') != episode['balanced_start_windows']
            or apps[0]['parameters'].get('seed') != episode['seed']):
        raise ValueError('Episode warm-start exploration differs from declared partition')


def require_learner_seed(model):
    # CLI/config seeds vary with the physical episode. A warm-start load must
    # retain the original learner's RNG identity from the exact JSON artifact.
    if load(model).get('seed') != 179:
        raise ValueError('Warm-start learner seed changed from the native artifact')


def launch(command, folder, status):
    if shutil.disk_usage(folder.parent).free < MIN_FREE:
        raise RuntimeError('One GiB disk reserve reached before launch; completed data retained')
    with folder.with_name(folder.name+'.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status.write(stage='running_partition', case=str(folder.relative_to(status.output)),
                command=command, child_pid=process.pid, started_unix=time.time())
            while process.poll() is None:
                if shutil.disk_usage(folder.parent).free < RUNTIME_FLOOR:
                    raise RuntimeError('Runtime disk floor reached; partial episode retained')
                time.sleep(5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
        if process.returncode:
            raise RuntimeError('Episode failed; retained log '+str(folder.with_name(folder.name+'.log')))


def episode_command(output, episode, model):
    result = command_for(output, episode['case_dir'], episode['scenario'], 'train',
        episode['seconds'], episode['seed'], model)
    # argparse takes the last occurrence. This overrides the original helper's
    # fresh-training default without editing the fingerprinted experiment code.
    return result + ['--urllc-xapp-balanced-start-windows', str(episode['balanced_start_windows'])]


def save_audit(output, episode, model, previous, plan, status):
    folder = output/episode['case_dir']; after = output/episode['after_model']
    status.write(stage='auditing_partition', case=episode['case_dir'], child_pid=None)
    check_recorded_inputs(folder, episode, plan)
    require_learner_seed(model)
    audit = audit_case(folder, 'train', model, previous=previous)
    exact_copy(model, after)
    before = output/episode['before_model'] if episode['before_model'] else None
    audit.update(name=episode['case_dir'], scenario=episode['scenario'], arm='train', seed=episode['seed'],
        balanced_start_windows=episode['balanced_start_windows'], episode_model_sha256=sha256(after),
        before_model_sha256=sha256(before) if before else None,
        after_model=str(after), before_model=str(before) if before else None)
    atomic_json(output/episode['audit'], audit)
    return audit


def verify_completed(output, episode, previous, plan, status, previous_sha256=None):
    path = output/episode['audit']; saved = load(path)
    after = output/episode['after_model']
    before = output/episode['before_model'] if episode['before_model'] else None
    if (saved.get('source_sha256') != PINNED_SOURCE or sha256(after) != saved.get('episode_model_sha256')
            or (before is not None and (sha256(before) != saved.get('before_model_sha256')
                or (previous_sha256 is not None and sha256(before) != previous_sha256)
                or load(before) != previous))):
        raise ValueError('Completed episode model lineage/hash mismatch')
    status.write(stage='verifying_completed_partition', case=episode['case_dir'], child_pid=None)
    check_recorded_inputs(output/episode['case_dir'], episode, plan)
    require_learner_seed(after)
    actual = audit_case(output/episode['case_dir'], 'train', after, previous=previous)
    if actual['online_training_audit'] != saved['online_training_audit']:
        raise ValueError('Completed episode raw audit differs from saved lineage')
    return saved


def train(args, plan, status):
    output = args.output; model = output/'models/urllc-qos-rf.json'
    declared = manifest(plan); manifest_path = output/'training-partitions.json'
    if manifest_path.exists():
        if not args.resume or load(manifest_path) != declared:
            raise ValueError('Existing partition manifest requires identical --resume inputs/script')
    else:
        atomic_json(manifest_path, declared)
    if (output/'training/burst').exists():
        raise ValueError('Unpartitioned burst directory exists; preserve it and resolve explicitly')
    results = []; previous = None; expected_latest = None; seen_incomplete = False
    for episode in declared['episodes']:
        verify_source(plan)
        folder = output/episode['case_dir']; after = output/episode['after_model']
        audit_path = output/episode['audit']
        before = output/episode['before_model'] if episode['before_model'] else None
        folder.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            if not args.resume or seen_incomplete:
                raise ValueError('Completed episodes must form an immutable prefix')
            saved = verify_completed(output, episode, previous, plan, status, expected_latest)
            results.append(saved); previous = load(after); expected_latest = sha256(after)
            continue
        seen_incomplete = True
        # No later directory can already exist when this episode is incomplete.
        later = declared['episodes'][len(results)+1:]
        if any((output/e['case_dir']).exists() or (output/e['audit']).exists() for e in later):
            raise ValueError('Episode lineage has a gap; no weights are rolled back')
        if episode['name'] == 'native' and folder.exists():
            if not args.resume:
                raise ValueError('Adoption of existing native run requires --resume')
            while True:
                alive = pid_running(args.native_pid, folder) if args.native_pid else False
                run_status = load(folder/'run_status.json') if (folder/'run_status.json').exists() else {}
                status.write(stage='waiting_native', case=episode['case_dir'], child_pid=args.native_pid if alive else None,
                    native_last_completed_slot=run_status.get('last_completed_slot'))
                if not alive:
                    if not run_status.get('completed'):
                        raise ValueError('Adopted native process exited without complete raw recording/HTML')
                    break
                if shutil.disk_usage(output).free < RUNTIME_FLOOR:
                    # The adopted process is not ours to stop; report without killing it.
                    raise RuntimeError('Disk floor reached while adopting native; child left untouched')
                time.sleep(5)
        elif folder.exists():
            # Recover a fully completed run without executing it twice. A partial
            # run is refused even when a more recent mutable model exists.
            if not args.resume or not (folder/'run_status.json').exists() or not load(folder/'run_status.json').get('completed'):
                raise ValueError('Incomplete burst retained; do not replay '+str(folder))
            if (before is None or not before.exists() or load(before) != previous
                    or sha256(before) != expected_latest):
                raise ValueError('Completed unaudited episode lacks exact before-model lineage')
        else:
            if expected_latest is not None and sha256(model) != expected_latest:
                raise ValueError('Latest mutable model differs from the previous exact checkpoint')
            if episode['name'] == 'native' and model.exists():
                raise ValueError('Fresh native training requires an empty model path')
            if before:
                exact_copy(model, before)
            launch(episode_command(output, episode, model), folder, status)
        saved = save_audit(output, episode, model, previous, plan, status)
        results.append(saved); previous = load(after); expected_latest = sha256(after)
    if sha256(model) != expected_latest:
        raise ValueError('Final mutable model differs from the verified partition chain')
    atomic_json(output/'training-partition-results.json', results)
    payload = load(model)
    atomic_json(output/'training-model-stamp.json', dict(sha256=sha256(model), samples=len(payload['x']),
        trained_samples=payload['trained_samples'], fit_count=payload['fit_count'],
        actual_training_seconds=sum(r['simulated_s'] for r in results),
        actual_native_seconds=sum(r['simulated_s'] for r in results if r['scenario']=='native'),
        actual_burst_seconds=sum(r['simulated_s'] for r in results if r['scenario']=='burst'),
        training_partitions_sha256=sha256(manifest_path), episode_model_sha256=[r['episode_model_sha256'] for r in results]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-pid', type=int)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[56])
    parser.add_argument('--eval-seconds', type=float,
        help='Per-case duration; inherits the shared study default (3.2 s)')
    parser.add_argument('--eval-scenarios', choices=study.SCENARIOS, nargs='+',
        help='Inherits the shared study default: burst, native, alternating')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true', help='Run the frozen evaluator using the reusable native membership auditor')
    args = parser.parse_args(argv); args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT/'outputs') or args.output == ROOT/'outputs':
        raise ValueError('Require a dedicated project outputs directory')
    if (not args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds)
            or any(s < 0 or s > 2**32-1 or s in range(179,185) for s in args.eval_seeds)
            or (args.native_pid is not None and args.native_pid <= 1)):
        raise ValueError('Require disjoint evaluation seeds and a valid native PID')
    effective = study.parse_args(study_arguments(args))
    args.eval_seconds = effective.eval_seconds
    args.eval_scenarios = effective.eval_scenarios
    return args


def evaluation_options(args):
    """One option list is reused by the plan and both evaluation handoffs."""
    result = ['--eval-seeds', *map(str, args.eval_seeds)]
    if args.eval_seconds is not None:
        result += ['--eval-seconds', str(args.eval_seconds)]
    if args.eval_scenarios is not None:
        result += ['--eval-scenarios', *args.eval_scenarios]
    return result


def study_arguments(args):
    # Physical training partitions are unchanged; their plan must nevertheless
    # be generated by the same parser and schema as the frozen evaluator.
    result = ['--output', str(args.output), '--phase', 'evaluate',
        '--train-seed', '179', '--train-seconds', '30', '--train-native-seconds', '10',
        '--pilot-seconds', '.8', *evaluation_options(args)]
    if args.resume:
        result.append('--resume')
    return result


def expected_plan_for(args):
    return study.plan_for(study.parse_args(study_arguments(args)))


def main(argv=None):
    args = parse_args(argv); output = args.output; output.mkdir(parents=True, exist_ok=True)
    expected_plan = expected_plan_for(args)
    verify_source(expected_plan)
    require_membership_rejection_audit()
    if args.evaluate_only:
        if not args.resume or load(output/'plan.json') != expected_plan:
            raise ValueError('Original frozen evaluation requires the identical existing study plan')
        sys.argv=[str(ROOT/'experiments/run_urllc_qos_study.py'), *study_arguments(args)]
        return study.main()
    with (output/'.partitioned.lock').open('a') as partition_lock:
        fcntl.flock(partition_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'plan.json').exists():
            if not args.resume or load(output/'plan.json') != expected_plan:
                raise ValueError('Original study plan changed; no replacement is performed')
        else:
            atomic_json(output/'plan.json', expected_plan)
        with Status(output) as status:
            try:
                # Exclude every normal study supervisor while training. The
                # adopted native child has its own session and does not hold it.
                with (output/'.study.lock').open('a') as study_lock:
                    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    train(args, expected_plan, status)
                verify_source(expected_plan)
                command = [sys.executable, str(Path(__file__)), '--output', str(output),
                    '--evaluate-only', '--resume', *evaluation_options(args)]
                status.write(stage='starting_original_evaluation', command=command, child_pid=None)
                status.main_owned = False
                with (output/'partitioned-evaluation.log').open('a') as stream:
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    status.write(stage='original_evaluation_running', child_pid=proc.pid)
                    result = proc.wait()
                if result:
                    raise RuntimeError('Original frozen evaluator failed; see partitioned-evaluation.log')
                status.write(stage='complete', child_pid=None, model_sha256=sha256(output/'models/urllc-qos-rf.json'))
            except BaseException as error:
                status.write(stage='failed', error_type=type(error).__name__, error=str(error), child_pid=None)
                raise


if __name__ == '__main__':
    main()
