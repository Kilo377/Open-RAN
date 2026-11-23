"""Bounded online MEC-budget QGBDT study with recorded random mitigation."""
import argparse
from collections import Counter
import hashlib
import json
import math
import random
from pathlib import Path
import subprocess
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_latency_energy_study import aggregate_offloaded_cohorts, jsonl
from experiments.run_ue_energy_study import (POPULATION, TARGETS, aggregate_cohorts,
                                           finite, load, observed_mature_cohorts)

ENERGY = 'ue_energy_rf'
BUDGET = 'mec_offload_budget_qgbdt'
UID = 'ue_4'
ARMS = ('off', 'x01', 'budget', 'both_random')
ENERGY_SHA256 = '9baf497d3d84e01c0aaa609ec050f75fcbd24af33a39f8bd5fedf5e16ac15e2b'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_stamp(path):
    p = load(path)
    return dict(sha256=digest(path), format=p['format'], version=p['version'],
                samples=len(p['x']), trained_samples=p['trained_samples'],
                fit_count=p['fit_count'], prediction_batches=p.get('prediction_batches', 0),
                fit_parameters=p.get('fit_parameters'), feature_names=p['feature_names'])


def training_readiness(path):
    if not Path(path).is_file():
        return dict(ready=False, status='rules_only', reason='No measured model artifact', samples=0)
    p = load(path)
    stamp = model_stamp(path)
    if p.get('format') != 'chongyu.online-qgbdt':
        raise ValueError('Unexpected budget model format')
    index = p['feature_names'].index('candidate_offload_fraction')
    counts = Counter(format(row[index], 'g') for row in p['x'])
    ready = (stamp['samples'] >= 12 and stamp['trained_samples'] >= 12 and
             counts['0'] >= 6 and counts['0.5'] >= 6 and
             stamp['fit_count'] > 0 and bool(p['forest']))
    return dict(ready=bool(ready), status='qgbdt' if ready else 'rules_only',
                reason='Actual measured training coverage and fitted forest' if ready else
                       'Requires 12 actual rows, six per feasible action and a successful fit',
                action_samples=dict(counts), **stamp)


def training_evidence(folder, model_path, previous=None):
    """Audit new labels against actual executed, mature public cohorts."""
    rows = list(jsonl(folder, 'budget_qgbdt_events'))
    samples = [r for r in rows if r.get('kind') == 'measured_sample']
    confirmations = [r for r in rows if r.get('kind') == 'executed_policy_confirmed']
    config = load(Path(folder) / 'config_snapshot.json')
    slot_ms = config['manifest']['slot_duration_ms']
    profile = config['configs']['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue'][UID]
    for sample in samples:
        c = sample['cohort']
        aggregate_cohorts([c])
        aggregate_offloaded_cohorts([c])
        deadline = profile['deadline_ms']
        if (c['generated'] <= 0 or
                c['generated'] != c['completed'] + c['failed'] + c['pending'] or
                c['mature_after_s'] > sample['produced_slot'] * slot_ms / 1000 + 1e-9 or
                c['mature_after_s'] < c['end_s'] + deadline / 1000 - 1e-9):
            raise ValueError('Online budget label lacks its complete mature demand horizon')
        if not any(r.get('cohort') == c['index'] and r.get('p') == sample['p'] and
                   r.get('owner') == BUDGET and r['produced_slot'] <= sample['produced_slot']
                   for r in confirmations):
            raise ValueError('Online budget label lacks executed UID ownership confirmation')
        penalty = (c['completion_latency_ms_sum'] + deadline * (c['failed'] + c['pending'])) / c['generated']
        utility = c['on_time_completed'] / c['generated'] - .25 * penalty / deadline
        if not math.isclose(utility, sample['utility'], abs_tol=1e-12, rel_tol=0):
            raise ValueError('Online budget label differs from measured utility')
    previous = previous or dict(x=[], y=[])
    if Path(model_path).is_file():
        model = load(model_path)
        n = len(previous['x'])
        if model['x'][:n] != previous['x'] or model['y'][:n] != previous['y']:
            raise ValueError('Continuation changed existing actual training rows')
        persisted_features = [[struct.unpack('f', struct.pack('f', v))[0] for v in r['features']]
                              for r in samples]
        if (model['x'][n:] != persisted_features or
                model['y'][n:] != [r['utility'] for r in samples]):
            raise ValueError('Saved new rows do not match actual online labels')
    elif samples:
        raise ValueError('Measured training rows were not persisted')
    return dict(new_measured_rows=len(samples), confirmations=len(confirmations),
                event_counts=dict(Counter(r.get('kind') for r in rows)),
                actual_mature_labels_verified=True,
                not_evaluable_reasons=[r.get('reason') for r in rows if r.get('kind') == 'not_evaluable'])


def cohort_demand(cohorts, task_profile, budget_gflops):
    """Expected demand uses executed generation policy, not random route counts."""
    rate = 1000. / finite(task_profile['interval_ms'], 'task interval')
    work = finite(task_profile['compute_gflop'], 'task work')
    if rate <= 0 or work <= 0:
        raise ValueError('Positive task interval/work required')
    rows = []
    for c in cohorts:
        p = finite(c.get('mean_offload_fraction'), 'executed cohort mean offload fraction')
        if not 0 <= p <= 1:
            raise ValueError('Executed offload fraction outside [0,1]')
        route = aggregate_offloaded_cohorts([c])
        expected = rate * work * p
        rows.append(dict(start_s=c['start_s'], end_s=c['end_s'], generated=c['generated'],
            mean_executed_offload_fraction=p, expected_mec_gflops=expected,
            expected_budget_violation=expected > budget_gflops + 1e-12,
            realized_offloaded_generated=route['generated'],
            realized_mec_work_gflop=route['generated'] * work))
    generated = sum(r['generated'] for r in rows)
    duration = sum(r['end_s'] - r['start_s'] for r in rows)
    if not generated or duration <= 0:
        raise ValueError('No scored task demand')
    mean_p = sum(r['mean_executed_offload_fraction'] * r['generated'] for r in rows) / generated
    return dict(cohorts=rows, mean_executed_offload_fraction=mean_p,
                expected_mec_gflops=rate * work * mean_p,
                expected_budget_gflops=budget_gflops,
                expected_budget_violating_cohorts=sum(r['expected_budget_violation'] for r in rows),
                realized_offloaded_generated=sum(r['realized_offloaded_generated'] for r in rows),
                realized_mec_gflops=sum(r['realized_mec_work_gflop'] for r in rows) / duration,
                scope='Expected=actual cohort mean executed p × configured task rate × work; realized routing counts reported separately')


def controller_evidence(folder, summary, kind, stamp, mode, require_execution=True):
    ident = ENERGY if kind == 'energy' else BUDGET
    key = 'ue_energy_xapp' if kind == 'energy' else 'budget_xapp'
    learning = summary.get(key, {})
    descriptor = next((a for a in summary.get('xapp_library', {}).get('instances', [])
                       if a.get('implementation_id') == ident), {})
    events = list(jsonl(folder, 'energy_rf_events' if kind == 'energy' else 'budget_qgbdt_events'))
    confirmed = [r for r in events if r.get('kind') == 'executed_policy_confirmed' and
                 r.get('ue_id', UID if kind == 'budget' else None) == UID]
    expected_fields = {'ue_offload_fraction'} if kind == 'budget' else {'ue_offload_fraction', 'ue_drx_configs'}
    field_surface = set(descriptor.get('writable_actions', []))
    unchanged = (stamp is None or all(learning.get(k) == stamp[k] for k in
                                     ('samples', 'trained_samples', 'fit_count')))
    prediction_delta = (learning.get('prediction_batches', 0) - stamp.get('prediction_batches', 0)
                        if stamp is not None else 0)
    used = (learning.get('rf_selected_actions', 0) > 0 if kind == 'energy' else
            mode == 'rules' or prediction_delta > 0)
    effects, rejections = 0, 0
    for receipt in jsonl(folder, 'control_receipts'):
        provenances = receipt.get('provenance', [])
        if isinstance(provenances, dict):
            provenances = [provenances]
        owns = any(p.get('implementation_id') == ident for p in provenances)
        owns = owns or any(r.get('winner') == ident and r.get('ue_id') == UID
                           for r in receipt.get('mitigation_records', []))
        if not owns:
            continue
        if receipt.get('status') == 'rejected':
            rejections += 1
        if receipt.get('status') == 'applied':
            requested = {}
            for f in receipt.get('fields', []):
                if f.get('field') == 'ue_offload_fraction' and isinstance(f.get('value'), dict):
                    requested.update(f['value'])
            for value in receipt.get('after', {}).values():
                actual = value.get('ue_service_policies', {}).get(UID, {}).get('offload_fraction')
                if UID in requested and type(actual) in (float, int) and math.isclose(actual, requested[UID], abs_tol=1e-12):
                    effects += 1
    executed = bool(confirmed) and effects > 0
    valid = (bool(descriptor.get('active_cell_ids')) and expected_fields <= field_surface and
             used and unchanged and (executed or not require_execution))
    return dict(implementation_id=ident, mode=mode, declared_rcps=sorted(field_surface),
                confirmed_ue4_policies=len(confirmed), runtime_effect_receipts=effects,
                rejected_receipts=rejections, prediction_batches_this_run=prediction_delta,
                genuine_model_inference=used and mode != 'rules', rules_ablation=mode == 'rules',
                execution_required=require_execution, executed_and_confirmed=executed,
                frozen_sample_fit_counts_unchanged=unchanged, learning=learning,
                event_counts=dict(Counter(r.get('kind') for r in events)), valid_control_evidence=bool(valid))


def arbitration_evidence(folder):
    selections = {}
    conflict_keys, errors = set(), []
    draws = {}
    for row in jsonl(folder, 'rule_decisions'):
        if row.get('reason') not in ('direct_conflict_random_selection', 'direct_conflict_request_selection') or row.get('ue_id') != UID:
            continue
        if row.get('field') != 'ue_offload_fraction':
            errors.append('Unexpected random field')
            continue
        candidates = row.get('candidates', [])
        if not candidates and row.get('winner') == 'rule_withdrawal':
            continue
        sources = [c.get('source') for c in candidates]
        if not sources or len(set(sources)) != len(sources) or not set(sources) <= {ENERGY, BUDGET}:
            errors.append('Unexpected or duplicated UE4 candidate source')
            continue
        winner = row.get('winner')
        values = {c['source']: c['requested_value'] for c in candidates}
        conflict = len(set(values.values())) > 1
        draw = row.get('draw')
        uniform = (set(sources) == {ENERGY, BUDGET} and row.get('probability') == .5 and
                   type(draw) in (float, int) and 0 <= draw < 1 and
                   winner == sorted(sources)[min(int(draw * len(sources)), len(sources) - 1)]) if conflict else (
                   row.get('probability') == 1 and draw is None)
        if winner not in values or row.get('selected_value') != values[winner] or not uniform:
            errors.append('Invalid uniform random selection')
        key = (row.get('cell_id'), row.get('window_index'), row.get('generation'))
        if conflict:
            conflict_keys.add(key)
            index = row.get('draw_index')
            if (type(index) is not int or index < 1 or type(row.get('mitigation_seed')) is not int or
                    row.get('stream_key') != 'ue_offload_fraction:' + UID):
                errors.append('Missing exact seeded RNG stream evidence')
            elif index in draws and draws[index] != (row['mitigation_seed'], draw):
                errors.append('Conflicting repeated RNG draw')
            else:
                draws[index] = (row['mitigation_seed'], draw)
        signature = (winner, row.get('selected_value'), row.get('draw'))
        if key in selections and selections[key]['signature'] != signature:
            errors.append('Selection changed within one held epoch')
        selections[key] = dict(signature=signature, row=row)
    if draws:
        seed = next(iter(draws.values()))[0]
        rng = random.Random(f'{seed}:direct-conflict-mitigation:ue_offload_fraction:{UID}')
        if sorted(draws) != list(range(1, len(draws) + 1)):
            errors.append('Nonconsecutive RNG draw indices')
        for index in sorted(draws):
            actual_seed, draw = draws[index]
            if actual_seed != seed or draw != rng.random():
                errors.append('Recorded draw disagrees with independent seeded RNG replay')
    executed, rejected, winners = 0, 0, Counter()
    for receipt in jsonl(folder, 'control_receipts'):
        for record in receipt.get('mitigation_records', []):
            if record.get('ue_id') != UID or record.get('winner') not in (ENERGY, BUDGET):
                continue
            if receipt.get('status') == 'rejected':
                rejected += 1
                continue
            if receipt.get('status') != 'applied' or record.get('status') != 'applied':
                continue
            key = (record.get('cell_id'), record.get('window_index'), record.get('generation'))
            chosen = selections.get(key)
            if chosen is None or chosen['signature'][:2] != (record.get('winner'), record.get('selected_value')):
                errors.append('Executed choice lacks its exact arbitration receipt')
                continue
            actual = receipt.get('after', {}).get(record['cell_id'], {}).get('ue_service_policies', {}).get(UID, {}).get('offload_fraction')
            if actual != record.get('selected_value'):
                errors.append('Executed UE4 policy disagrees with selected value')
                continue
            executed += 1
            winners[record['winner']] += 1
    return dict(unique_random_epochs=len(conflict_keys), all_selection_epochs=len(selections),
                executed_matching_receipts=executed, rejected_receipts=rejected,
                runtime_winner_receipts=dict(winners), errors=errors[:20],
                expected_probability=.5, field='ue_offload_fraction', ue_id=UID,
                both_contenders_executed=set(winners) == {ENERGY, BUDGET},
                seeded_rng_replay_passed=bool(draws) and not any('RNG' in error for error in errors),
                valid=bool(conflict_keys and executed and not errors),
                note='Repeated held-state actuation receipts are not independent random draws')


def evaluate_study(folders, energy_stamp, budget_stamp, budget_mode, budget_gflops=.75):
    if set(folders) != set(ARMS):
        raise ValueError('OFF/X01/BUDGET/BOTH_RANDOM required')
    folders = {k: Path(v) for k, v in folders.items()}
    configs = {k: load(v / 'config_snapshot.json') for k, v in folders.items()}
    summaries = {k: load(v / 'final_summary.json') for k, v in folders.items()}
    base = configs['off']
    manifest = base['manifest']
    if {u['ue_id'] for u in manifest['ues']} != POPULATION or len(manifest['cells']) != 3:
        raise ValueError('Native six-UE/three-cell manifest required')
    services = base['configs']['phy']['rt_sys']['ue_services']
    profile = services['mec']['tasks_by_ue'][UID]
    for arm, cfg in configs.items():
        if (cfg['manifest'] != manifest or cfg['configs']['ue'] != base['configs']['ue'] or
                cfg['configs']['phy']['rt_sys']['ue_services'] != services):
            raise ValueError('Radio/movement/demand/service mismatch: ' + arm)
        expected = 'uniform_random' if arm == 'both_random' else 'priority'
        if cfg['configs']['ric'].get('direct_conflict_mitigation', 'priority') != expected:
            raise ValueError('Wrong mitigation mode for ' + arm)
    duration = manifest['total_slots'] * manifest['slot_duration_ms'] / 1000
    measured = {k: observed_mature_cohorts(v, manifest['slot_duration_ms'], duration) for k, v in folders.items()}
    metrics, intervals, demand = {}, None, {}
    for arm in ARMS:
        summary = summaries[arm]
        users = {}
        for uid in sorted(POPULATION):
            packet = summary['packet_metrics'][uid]
            user = dict(ue_energy_j=finite(summary['ue_services']['per_ue'][uid]['ue_energy_total_j'], 'energy'),
                        delivered_bytes=packet['delivered_bytes'],
                        goodput_mbps=packet['delivered_bytes'] * 8 / duration / 1e6)
            if uid in TARGETS:
                cohorts = {k: v for k, v in measured[arm][uid].items() if k[0] >= .1 - 1e-9}
                if not cohorts:
                    raise ValueError('Missing mature cohorts')
                if intervals is None:
                    intervals = set(cohorts)
                elif intervals != set(cohorts):
                    raise ValueError('Mature creation intervals differ')
                rows = [cohorts[k] for k in sorted(cohorts)]
                user['tasks'] = aggregate_cohorts(rows)
                user['offloaded_tasks'] = aggregate_offloaded_cohorts(rows)
                key = uid
                if arm == 'off':
                    demand[key] = [c['generated'] for c in rows]
                elif demand[key] != [c['generated'] for c in rows]:
                    raise ValueError('Exogenous task demand differs')
                user['task_qos_passed'] = (user['tasks']['on_time_rate'] is not None and
                    user['tasks']['on_time_rate'] >= .95 - 1e-12 and
                    user['tasks']['completed_task_mean_latency_ms'] is not None and
                    user['tasks']['completed_task_mean_latency_ms'] <= 30 + 1e-12)
                if uid == UID:
                    user['mec_demand'] = cohort_demand(rows, profile, budget_gflops)
            users[uid] = user
        delivery = {u: users[u]['delivered_bytes'] >= .95 * metrics['off']['users'][u]['delivered_bytes'] - 1e-9
                    for u in POPULATION} if arm != 'off' else {u: True for u in POPULATION}
        metrics[arm] = dict(users=users, per_ue_delivery_guard=delivery,
            all_delivery_guards_passed=all(delivery.values()),
            all_task_qos_guards_passed=all(users[u]['task_qos_passed'] for u in TARGETS),
            network_goodput_mbps=sum(u['goodput_mbps'] for u in users.values()))
    arbitration = arbitration_evidence(folders['both_random'])
    evidence = {
        'x01': {'energy': controller_evidence(folders['x01'], summaries['x01'], 'energy', energy_stamp, 'frozen')},
        'budget': {'budget': controller_evidence(folders['budget'], summaries['budget'], 'budget', budget_stamp, budget_mode)},
        'both_random': {kind: controller_evidence(folders['both_random'], summaries['both_random'], kind,
                            energy_stamp if kind == 'energy' else budget_stamp,
                            'frozen' if kind == 'energy' else budget_mode,
                            require_execution=bool(arbitration['runtime_winner_receipts'].get(ENERGY if kind == 'energy' else BUDGET)))
                        for kind in ('energy', 'budget')}}
    native_energy = metrics['off']['users'][UID]['ue_energy_j']
    if native_energy <= 0:
        raise ValueError('Positive native UE4 energy required')
    saving = 1 - metrics['x01']['users'][UID]['ue_energy_j'] / native_energy
    demand_metric = lambda arm: metrics[arm]['users'][UID]['mec_demand']
    budget_feasible = demand_metric('budget')['expected_mec_gflops'] <= budget_gflops + 1e-12
    energy_effective = saving >= .05 - 1e-12
    qos = all(metrics[a]['all_task_qos_guards_passed'] and metrics[a]['all_delivery_guards_passed']
              for a in ('x01', 'budget'))
    direct_surface = (set(evidence['both_random']['energy']['declared_rcps']) &
                      set(evidence['both_random']['budget']['declared_rcps']))
    valid = (all(e['valid_control_evidence'] for case in evidence.values() for e in case.values()) and
             arbitration['valid'] and 'ue_offload_fraction' in direct_surface)
    structural_m1 = arbitration['valid'] and 'ue_offload_fraction' in direct_surface
    isolated_valid = all(e['valid_control_evidence'] for arm in ('x01', 'budget') for e in evidence[arm].values())
    intent_case = (isolated_valid and energy_effective and budget_feasible and qos and
                   demand_metric('x01')['expected_mec_gflops'] > budget_gflops + 1e-12)
    outcome = dict(energy_ue4_at_most_95pct_native=metrics['both_random']['users'][UID]['ue_energy_j'] <= .95 * native_energy + 1e-12,
                   aggregate_expected_mec_budget_met=demand_metric('both_random')['expected_mec_gflops'] <= budget_gflops + 1e-12,
                   task_qos_met=metrics['both_random']['all_task_qos_guards_passed'],
                   delivery_met=metrics['both_random']['all_delivery_guards_passed'])
    return dict(seed=manifest['rng_seed'], duration_s=duration, budget_mode=budget_mode,
                calibration_not_independent_validation=True, task_scoring_start_s=.1,
                mature_creation_intervals_s=[list(k) for k in sorted(intervals)],
                task_profile={k: profile[k] for k in ('interval_ms', 'compute_gflop', 'deadline_ms')},
                thresholds=dict(ue4_energy_ratio=.95, mec_budget_gflops=budget_gflops,
                                all_task_on_time_rate=.95, all_task_mean_latency_ms=30,
                                per_ue_delivery_ratio=.95),
                arms=metrics, controller_evidence=evidence, arbitration=arbitration,
                actual_rcp_intersection=sorted(direct_surface), valid_control_evidence=valid,
                x01_ue4_energy_saving_fraction=saving, x01_energy_intent_achieved=energy_effective,
                isolated_budget_feasible=budget_feasible, isolated_qos_guards_passed=qos,
                individual_intent_case_evidence_passed=bool(intent_case),
                structural_m1_observed=bool(structural_m1), m1_observed=bool(structural_m1),
                random_mitigation_outcome=dict(outcome, all_intents_and_guards_met=all(outcome.values()),
                    every_cohort_within_budget_diagnostic=demand_metric('both_random')['expected_budget_violating_cohorts'] == 0),
                classification='m1_conflict_observed' if structural_m1 else
                    'insufficient_control_evidence' if not valid else 'no_m1_conflict_observed',
                interpretation='M1 observation and random mitigation outcome are separate. No post-random harm is required, no learned-model claim for rules mode, no significance or exhaustive infeasibility claim.')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--energy-model', type=Path, default=ROOT / 'outputs/models/ue_energy_rf/model.json')
    p.add_argument('--budget-model', type=Path)
    p.add_argument('--train-seconds', type=float, default=4.)
    p.add_argument('--max-train-seconds', type=float, default=6.)
    p.add_argument('--train-seed', type=int, default=137)
    p.add_argument('--eval-seed', type=int, default=55)
    p.add_argument('--eval-seconds', type=float, default=1.)
    p.add_argument('--budget-gflops', type=float, default=.75)
    p.add_argument('--mitigation-seed', type=int, default=1701)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def command_for(args, output, energy_model, budget_model, name, seed, seconds, energy_mode, budget_mode):
    command = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
        '--duration-s', str(seconds), '--seed', str(seed), '--xapp-pool', 'chongyu',
        '--ue-services', 'drx-mec', '--record-compression', 'gzip',
        '--energy-xapp-mode', energy_mode, '--latency-xapp-mode', 'off',
        '--no-energy-xapp-deploy-drx',  # Historical study keeps its original action constraint.
        '--budget-xapp-mode', budget_mode, '--budget-xapp-gflops', str(args.budget_gflops),
        '--budget-xapp-window-ms', '100', '--direct-conflict-mitigation',
        'uniform_random' if name.endswith('both_random') else 'priority',
        '--mitigation-seed', str(args.mitigation_seed), '--mitigation-window-ms', '100',
        '--channel-cache-root', str(ROOT / 'channel_cache' / output.name / ('seed_' + str(seed))),
        '--output', str(output / name)]
    if energy_mode != 'off':
        command += ['--energy-xapp-model', str(energy_model)]
    if budget_mode != 'off':
        command += ['--budget-xapp-model', str(budget_model)]
    return command


def main():
    args = parse_args()
    output = args.output.resolve()
    energy_model = args.energy_model.resolve()
    budget_model = (args.budget_model or output / 'budget_qgbdt_model.json').resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Choose a fresh output directory under outputs')
    if any(not p.is_relative_to(ROOT / 'outputs') for p in (energy_model, budget_model)) or energy_model == budget_model:
        raise ValueError('Use distinct model paths under outputs')
    if budget_model.exists():
        raise ValueError('Choose a fresh budget model path; historical artifacts stay immutable')
    if (not all(math.isfinite(v) and v > 0 for v in (args.train_seconds, args.max_train_seconds,
            args.eval_seconds, args.budget_gflops)) or not args.train_seconds <= args.max_train_seconds <= 6):
        raise ValueError('Positive finite settings and no more than six seconds total training required')
    if args.train_seed in (args.eval_seed, args.eval_seed - 1) or args.mitigation_seed < 0:
        raise ValueError('Distinct training/continuation/evaluation seeds and nonnegative mitigation seed required')
    if not math.isclose(args.budget_gflops, .75, abs_tol=1e-12, rel_tol=0):
        raise ValueError('This approved two-action study fixes the MEC budget at .75 GFLOP/s')
    if args.dry_run:
        plan = [command_for(args, output, energy_model, budget_model, 'budget_online_training', args.train_seed, args.train_seconds, 'off', 'online')]
        remaining = args.max_train_seconds - args.train_seconds
        if remaining:
            plan.append(command_for(args, output, energy_model, budget_model, 'budget_online_extension', args.train_seed + 1, remaining, 'off', 'online'))
        plan += [command_for(args, output, energy_model, budget_model, 'seed' + str(args.eval_seed) + '_' + arm,
            args.eval_seed, args.eval_seconds, 'frozen' if arm in ('x01', 'both_random') else 'off',
            'frozen' if arm in ('budget', 'both_random') else 'off') for arm in ARMS]
        print(json.dumps(dict(dry_run=True, commands=plan,
            optional_extension='Only if first training has insufficient actual coverage; max six simulated seconds total',
            insufficient_coverage='Use explicitly labelled rules ablation; never fabricate a fit',
            expected_energy_sha256=ENERGY_SHA256), indent=2))
        return
    if not energy_model.is_file() or digest(energy_model) != ENERGY_SHA256:
        raise ValueError('The accepted unchanged X01 artifact (9baf...) is required')
    output.mkdir(parents=True)
    budget_model.parent.mkdir(parents=True, exist_ok=True)
    energy_stamp = model_stamp(energy_model)
    receipt = dict(completed=False, status='initializing', cases=[], training_readiness=[], training_evidence=[],
        protocol='MEC budget QGBDT isolated online training; OFF/X01/BUDGET/BOTH_RANDOM',
        train_seed=args.train_seed, optional_continuation_seed=args.train_seed + 1,
        eval_seed=args.eval_seed, calibration_not_independent_validation=True,
        max_training_simulated_s=args.max_train_seconds, training_simulated_s=0.,
        energy_model=str(energy_model), budget_model=str(budget_model), energy_model_before=energy_stamp,
        budget_gflops=args.budget_gflops, mitigation_seed=args.mitigation_seed,
        mitigation_window_ms=100, expected_training_coverage=dict(min_rows=12, min_rows_per_action=6),
        scope='Native six moving UEs; actual RF X01 unchanged; measured QGBDT or explicit rules-only ablation. Expected MEC demand and realized routing are separate. No tuning loop.')

    def write():
        temporary = output / 'acceptance.tmp'
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False))
        temporary.replace(output / 'acceptance.json')

    def run(name, seed, seconds, energy_mode, budget_mode):
        receipt.update(status='running', running=name)
        write()
        command = command_for(args, output, energy_model, budget_model, name, seed, seconds, energy_mode, budget_mode)
        began = time.perf_counter()
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(name + ' failed; inspect its log')
        folder = output / name
        status, perf = load(folder / 'run_status.json'), load(folder / 'performance.json')
        if not status.get('completed') or not status.get('replay', {}).get('audits_passed'):
            raise ValueError('Simulation/HTML audit incomplete: ' + name)
        if digest(energy_model) != energy_stamp['sha256']:
            raise ValueError('Study changed the accepted X01 artifact')
        receipt['cases'].append(dict(name=name, seed=seed, simulated_s=seconds,
            energy_mode=energy_mode, budget_mode=budget_mode, output=str(folder), command=command,
            source_sha256=status['source_sha256'], simulation_wall_s=perf['simulation_s'],
            external_wall_s=time.perf_counter() - began, internal_wall_s=status['elapsed_s'],
            html=status['replay']['html'], html_audits_passed=True))
        if len({c['source_sha256'] for c in receipt['cases']}) != 1:
            raise ValueError('Source changed during the study')
        write()
        return folder

    write()
    try:
        trained = run('budget_online_training', args.train_seed, args.train_seconds, 'off', 'online')
        receipt['training_evidence'].append(training_evidence(trained, budget_model))
        receipt['training_simulated_s'] += args.train_seconds
        readiness = training_readiness(budget_model)
        receipt['training_readiness'].append(readiness)
        remaining = args.max_train_seconds - args.train_seconds
        if not readiness['ready'] and remaining > 0:
            initial = load(budget_model) if budget_model.is_file() else None
            trained = run('budget_online_extension', args.train_seed + 1, remaining, 'off', 'online')
            receipt['training_evidence'].append(training_evidence(trained, budget_model, initial))
            receipt['training_simulated_s'] += remaining
            if initial:
                updated = load(budget_model)
                n = len(initial['x'])
                if updated['x'][:n] != initial['x'] or updated['y'][:n] != initial['y']:
                    raise ValueError('Continuation changed existing actual training rows')
            readiness = training_readiness(budget_model)
            receipt['training_readiness'].append(readiness)
        budget_mode = 'frozen' if readiness['ready'] else 'rules'
        stamp = model_stamp(budget_model) if budget_model.is_file() else None
        receipt.update(budget_evaluation_mode=budget_mode, budget_model_before_evaluation=stamp,
                       learned_model_claim=bool(readiness['ready']))
        write()
        folders = {}
        for arm in ARMS:
            folders[arm] = run('seed' + str(args.eval_seed) + '_' + arm, args.eval_seed,
                args.eval_seconds, 'frozen' if arm in ('x01', 'both_random') else 'off',
                budget_mode if arm in ('budget', 'both_random') else 'off')
            if stamp and digest(budget_model) != stamp['sha256']:
                raise ValueError('Evaluation modified the frozen budget artifact')
        evaluation = evaluate_study(folders, energy_stamp, stamp, budget_mode, args.budget_gflops)
        receipt.update(completed=True, evaluation=evaluation,
            status='rules_only' if budget_mode == 'rules' else
                   'completed_m1_observed' if evaluation['m1_observed'] else
                   'completed_no_m1_observed' if evaluation['valid_control_evidence'] else 'insufficient_evidence')
        receipt.pop('running', None)
        write()
    except BaseException as error:
        receipt.update(completed=False, status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        write()
        raise
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
