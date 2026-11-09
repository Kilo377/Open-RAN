"""Small online-training pilot followed by frozen, matched six-UE evaluation.

Run inside the accepted AGX GPU environment. This driver never invents model
labels: the live xApp records its executed actions and delayed outcomes. It
does not change the default network configuration or optimize simulator KPIs.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('ue_0', 'ue_4')
POPULATION = {'ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8'}


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Missing/nonfinite metric: ' + name)
    return float(value)


def mature_cohorts(report, duration_s):
    """Return only arrival cohorts whose complete deadline horizon is visible."""
    result = {}
    for cohort in report.get('ue_task_cohorts') or []:
        start = finite(cohort['start_s'], 'cohort start')
        end = finite(cohort['end_s'], 'cohort end')
        mature = finite(cohort['mature_after_s'], 'cohort maturity')
        if start < 0 or end <= start or mature < end:
            raise ValueError('Invalid cohort interval')
        if mature > duration_s + 1e-9:
            continue
        key = (round(start, 9), round(end, 9))
        if key in result:
            raise ValueError('Repeated task cohort')
        for name in ('generated', 'completed', 'on_time_completed', 'failed', 'pending'):
            count = cohort[name]
            if type(count) is not int or count < 0:
                raise ValueError('Invalid cohort count: ' + name)
        if cohort['generated'] != cohort['completed'] + cohort['failed'] + cohort['pending']:
            raise ValueError('Task cohort conservation failed')
        if cohort['on_time_completed'] > cohort['completed']:
            raise ValueError('Invalid on-time task count')
        result[key] = cohort
    return result


def aggregate_cohorts(cohorts):
    counts = {name: sum(c[name] for c in cohorts) for name in
              ('generated', 'completed', 'on_time_completed', 'failed', 'pending')}
    generated = counts['generated']
    latency = sum(finite(c.get('completion_latency_ms_sum', 0), 'cohort latency') for c in cohorts)
    return dict(counts, on_time_rate=counts['on_time_completed'] / generated if generated else None,
                completed_task_mean_latency_ms=latency / counts['completed'] if counts['completed'] else None)


def observed_mature_cohorts(folder, slot_ms, duration_s):
    """Accumulate bounded public snapshots; a final snapshot alone loses history."""
    folder = Path(folder)
    paths = [p for p in [folder / 'rt_sys_trace.jsonl.gz', folder / 'rt_sys_trace.jsonl'] if p.is_file()]
    if len(paths) != 1:
        raise ValueError('One unambiguous recorded trace is required for cohort evaluation')
    result = {uid: {} for uid in TARGETS}
    opener = gzip.open if paths[0].suffix == '.gz' else open
    last_slot = 0
    with opener(paths[0], 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            slot = row['slot_id']
            if slot <= last_slot:
                raise ValueError('Repeated/nonmonotonic cohort trace slot')
            last_slot = slot
            observed_s = min(duration_s, slot * slot_ms / 1000)
            for uid in TARGETS:
                report = row.get('ue_services', {}).get(uid, {})
                result[uid].update(mature_cohorts(report, observed_s))
    return result


def compare_pair(off_folder, frozen_folder, thresholds, task_start_s=.1):
    """Evaluate a predeclared cohort/energy contract without training leakage."""
    off_folder, frozen_folder = Path(off_folder), Path(frozen_folder)
    off_config = load(off_folder / 'config_snapshot.json')
    frozen_config = load(frozen_folder / 'config_snapshot.json')
    if off_config['manifest'] != frozen_config['manifest']:
        raise ValueError('OFF/frozen deployment, movement, radio or seed differs')
    if off_config['configs']['ue'] != frozen_config['configs']['ue']:
        raise ValueError('OFF/frozen business traffic differs')
    manifest = off_config['manifest']
    if {u['ue_id'] for u in manifest['ues']} != POPULATION or len(manifest['cells']) != 3:
        raise ValueError('The pilot requires the accepted six-UE/three-cell network')
    duration = manifest['total_slots'] * manifest['slot_duration_ms'] / 1000
    a = load(off_folder / 'final_summary.json')
    b = load(frozen_folder / 'final_summary.json')
    off = a['ue_services']['per_ue']
    frozen = b['ue_services']['per_ue']
    if set(off) != POPULATION or set(frozen) != POPULATION:
        raise ValueError('Service population differs')
    energy = lambda reports, ids: sum(finite(reports[u]['ue_energy_total_j'], 'UE energy') for u in ids)
    delivered = lambda summary, ids: sum(summary['packet_metrics'][u]['delivered_bytes'] for u in ids)
    target_off, target_on = energy(off, TARGETS), energy(frozen, TARGETS)
    if target_off <= 0:
        raise ValueError('Nonpositive reference energy')
    per_ue, common_intervals = {}, None
    off_cohorts = observed_mature_cohorts(off_folder, manifest['slot_duration_ms'], duration)
    frozen_cohorts = observed_mature_cohorts(frozen_folder, manifest['slot_duration_ms'], duration)
    for uid in TARGETS:
        ca = {key: value for key, value in off_cohorts[uid].items() if key[0] >= task_start_s - 1e-9}
        cb = {key: value for key, value in frozen_cohorts[uid].items() if key[0] >= task_start_s - 1e-9}
        if not ca or set(ca) != set(cb):
            raise ValueError('Missing/mismatched mature task cohorts: ' + uid)
        if common_intervals is None:
            common_intervals = set(ca)
        elif common_intervals != set(ca):
            raise ValueError('Target UEs have different cohort coverage')
        keys = sorted(ca)
        ma, mb = aggregate_cohorts([ca[k] for k in keys]), aggregate_cohorts([cb[k] for k in keys])
        if ma['generated'] <= 0 or ma['generated'] != mb['generated']:
            raise ValueError('Missing/changed generated task demand: ' + uid)
        da, db = delivered(a, [uid]), delivered(b, [uid])
        # Zero reference goodput must not be interpreted as a throughput gain.
        ratio = db / da if da else (1. if db == 0 else None)
        task_ok = mb['on_time_rate'] >= ma['on_time_rate'] - thresholds['max_task_rate_drop'] - 1e-12
        goodput_ok = da == 0 or db >= da * (1 - thresholds['max_delivered_fraction_drop']) - 1e-9
        per_ue[uid] = dict(off=ma, frozen=mb, task_rate_change=mb['on_time_rate'] - ma['on_time_rate'],
                          off_energy_j=off[uid]['ue_energy_total_j'], frozen_energy_j=frozen[uid]['ue_energy_total_j'],
                          off_delivered_bytes=da, frozen_delivered_bytes=db,
                          delivered_ratio=ratio, task_guard_passed=task_ok, goodput_guard_passed=goodput_ok)
    saving = (target_off - target_on) / target_off
    network_off, network_on = energy(off, POPULATION), energy(frozen, POPULATION)
    target_da, target_db = delivered(a, TARGETS), delivered(b, TARGETS)
    guards = all(v['task_guard_passed'] and v['goodput_guard_passed'] for v in per_ue.values())
    return dict(seed=manifest['rng_seed'], duration_s=duration,
                matched_radio_movement_business=True,
                task_scoring_start_s=task_start_s,
                energy_goodput_scope='Entire run, including startup; task cohorts exclude the first 100 ms.',
                mature_cohort_intervals_s=[list(k) for k in sorted(common_intervals)],
                per_ue=per_ue, target_off_energy_j=target_off, target_frozen_energy_j=target_on,
                target_energy_saving_fraction=saving,
                network_off_ue_energy_j=network_off, network_frozen_ue_energy_j=network_on,
                network_ue_energy_saving_fraction=(network_off-network_on)/network_off if network_off else None,
                target_off_goodput_mbps=target_da * 8 / duration / 1e6,
                target_frozen_goodput_mbps=target_db * 8 / duration / 1e6,
                network_off_goodput_mbps=delivered(a, POPULATION) * 8 / duration / 1e6,
                network_frozen_goodput_mbps=delivered(b, POPULATION) * 8 / duration / 1e6,
                task_and_goodput_guards_passed=guards,
                accepted=saving >= thresholds['min_target_energy_saving_fraction'] and guards)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, help='Artifact under project outputs; default OUTPUT/model.json')
    p.add_argument('--train-seconds', type=float, default=2.8)
    p.add_argument('--eval-seconds', type=float, default=1.)
    p.add_argument('--train-seed', type=int, default=11)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[22, 33])
    p.add_argument('--window-ms', type=float, default=100.)
    p.add_argument('--explore-windows', type=int, default=9)
    p.add_argument('--skip-training', action='store_true', help='Evaluate an existing frozen model')
    p.add_argument('--resume-online', action='store_true', help='Continue online learning from an existing artifact')
    p.add_argument('--train-only', action='store_true', help='Collect online samples and stop before held-out evaluation')
    p.add_argument('--fixed-reference', action='store_true', help='After first RF pair passes, compare a separate constant-offload reference at seed 22')
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()
    model = (args.model or output / 'model.json').resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise ValueError('Choose a fresh study directory under project outputs')
    if not model.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Model must stay under project outputs')
    if args.skip_training and args.train_only:
        raise ValueError('Choose training or existing-model evaluation')
    if args.skip_training and args.resume_online:
        raise ValueError('--skip-training and --resume-online are mutually exclusive')
    if args.skip_training and not model.is_file():
        raise ValueError('--skip-training requires an existing model')
    if args.resume_online and not model.is_file():
        raise ValueError('--resume-online requires an existing model')
    if not args.skip_training and not args.resume_online and model.exists():
        raise ValueError('Online pilot requires a fresh model path')
    for value in [args.train_seconds, args.eval_seconds, args.window_ms]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite durations required')
    if args.explore_windows < 1 or not args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError('Positive exploration budget and unique evaluation seeds required')
    if args.train_seed in args.eval_seeds:
        raise ValueError('Evaluation seeds must be held out from training')
    output.mkdir(parents=True)
    model.parent.mkdir(parents=True, exist_ok=True)
    thresholds = dict(min_target_energy_saving_fraction=.05, max_task_rate_drop=.05,
                      max_delivered_fraction_drop=.05)
    receipt = dict(completed=False, accepted=None, status='initializing',
                   ue_ids=sorted(POPULATION), target_ue_ids=list(TARGETS),
                   train_seed=args.train_seed, evaluation_seeds=args.eval_seeds,
                   train_seconds=args.train_seconds, evaluation_seconds=args.eval_seconds,
                   window_ms=args.window_ms, exploration_windows=args.explore_windows,
                   resumed_online=args.resume_online, fixed_reference_requested=args.fixed_reference,
                   model=str(model), thresholds=thresholds, cases=[], comparisons=[],
                   cost_scope='Sequential runs share geometry cache per seed; wall-time differences are not a controlled inference-overhead benchmark.',
                   metric_scope='Modelled UE radio/CPU energy and delivered payload cover the entire run, including startup and MEC result bytes. Task compliance uses matched mature arrival cohorts starting at 100 ms, including pending/failed tasks.')
    if args.resume_online:
        receipt['initial_model_sha256'] = sha256(model)

    def write():
        temp = output / 'acceptance.tmp'
        temp.write_text(json.dumps(receipt, indent=2, allow_nan=False))
        temp.replace(output / 'acceptance.json')

    def run_case(name, mode, seed, seconds, extra=()):
        receipt.update(status='running', running=name)
        write()
        folder = output / name
        command = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
                   '--duration-s', str(seconds), '--seed', str(seed),
                   '--ue-services', 'drx-mec', '--record-compression', 'gzip',
                   '--energy-xapp-mode', mode, '--energy-xapp-window-ms', str(args.window_ms),
                   '--energy-xapp-explore-windows', str(args.explore_windows),
                   '--channel-cache-root', str(ROOT / 'channel_cache' / output.name / f'seed_{seed}'),
                   '--output', str(folder), *extra]
        if mode != 'off':
            command.extend(['--energy-xapp-model', str(model)])
        begin = time.perf_counter()
        with (output / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed with code {result.returncode}; inspect its log')
        status = load(folder / 'run_status.json')
        performance = load(folder / 'performance.json')
        if not status.get('completed') or not status.get('replay', {}).get('audits_passed'):
            raise ValueError('Simulation or HTML audit incomplete: ' + name)
        if mode=='online':
            trained=load(folder/'final_summary.json').get('ue_energy_xapp',{})
            if trained.get('trained_samples',0)<12 or trained.get('fit_count',0)<1:
                raise ValueError('Online run did not produce a genuinely fitted RF')
        receipt['cases'].append(dict(name=name, mode=mode, seed=seed, output=str(folder),
            simulated_s=seconds, source_sha256=status['source_sha256'],
            simulation_wall_s=performance['simulation_s'], internal_wall_s=status['elapsed_s'],
            external_wall_s=time.perf_counter()-begin, actual_decodes=performance['actual_decodes'],
            html_audits_passed=True, html=status['replay']['html']))
        write()
        return folder

    write()
    try:
        if not args.skip_training:
            run_case('online_training', 'online', args.train_seed, args.train_seconds)
        if not model.is_file():
            raise ValueError('Training did not produce its model artifact')
        receipt['frozen_model_sha256'] = sha256(model)
        if args.train_only:
            receipt.update(completed=True, status='training_completed_not_evaluated')
        else:
            for i, seed in enumerate(args.eval_seeds):
                # Alternate pair order so the second held-out seed does not
                # always put the learning policy after the native baseline.
                folders = {}
                order = ['off', 'frozen'] if i % 2 == 0 else ['frozen', 'off']
                for mode in order:
                    folders[mode] = run_case(f'eval_seed{seed}_{mode}', mode, seed, args.eval_seconds)
                    if sha256(model) != receipt['frozen_model_sha256']:
                        raise ValueError('Evaluation modified the frozen training artifact')
                comparison = compare_pair(folders['off'], folders['frozen'], thresholds)
                receipt['comparisons'].append(comparison)
                if not comparison['accepted']:
                    receipt.update(completed=True, accepted=False, status='needs_refinement')
                    break
                write()
                if i == 0 and args.fixed_reference:
                    # This is an engineering reference, not learned RF output.
                    # It changes offload only; all baseline DRX values remain.
                    timeline = output / 'fixed_reference_timeline.json'
                    timeline.write_text(json.dumps([dict(slot=5, config=dict(
                        apply_scope='cluster', ue_offload_fraction={uid: 1. for uid in TARGETS}))], indent=2))
                    if seed == 22:
                        reference_off = folders['off']
                    else:
                        reference_off = run_case('fixed_reference_seed22_off', 'off', 22, args.eval_seconds)
                    reference = run_case('fixed_reference_seed22_offload1', 'off', 22, args.eval_seconds,
                                         ['--control-timeline', str(timeline)])
                    if sha256(model) != receipt['frozen_model_sha256']:
                        raise ValueError('Fixed reference modified the frozen training artifact')
                    ref = compare_pair(reference_off, reference, thresholds)
                    ref['reference_meets_same_contract'] = ref.pop('accepted')
                    ref.update(algorithm='constant_offload_1_with_baseline_drx', learned_model=False,
                               control_timeline=str(timeline))
                    receipt['fixed_reference'] = ref
                    write()
            else:
                receipt.update(completed=True, accepted=True, status='accepted_on_held_out_seeds')
        sources = {c['source_sha256'] for c in receipt['cases']}
        if len(sources) != 1:
            raise ValueError('Executed source changed during the study')
        receipt.pop('running', None)
        write()
    except BaseException as error:
        receipt.update(completed=False, accepted=False, status='failed',
                       failure=dict(type=type(error).__name__, message=str(error)))
        write()
        raise
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
