"""Budgeted online X12 training and frozen OFF/X01/X12/BOTH factorial pilot."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_ue_energy_study import (POPULATION, TARGETS, aggregate_cohorts,
                                           finite, load, observed_mature_cohorts)

ARMS = ('off', 'x01', 'x12', 'both')
APP_IDS = {'x01': 'ue_energy_rf', 'x12': 'latency_scheduler_rf'}
SUMMARY_KEYS = {'x01': 'ue_energy_xapp', 'x12': 'latency_xapp'}
OBJECTIVE_REVISION = 'offloaded_ar_task_e2e_v2'


def objective_scope():
    return dict(revision=OBJECTIVE_REVISION,
                x12_primary='Completed offloaded AR-task end-to-end mean latency (paper Eq. 14)',
                primary_improvement_fraction=.05,
                population='Actually offloaded tasks in matched mature creation cohorts; routed counts may differ by arm',
                all_task_guards='Per-target on-time-rate loss <=.05 and mean-delay increase <=max(1 ms,5%) versus OFF',
                previous_results='Earlier overall-task-delay pilots remain failed under their original objective; no retrospective relabelling')


def aggregate_offloaded_cohorts(cohorts):
    """Use measured route counters, never infer a route from a policy fraction."""
    names = ('generated', 'completed', 'on_time_completed', 'failed', 'pending')
    counts = dict.fromkeys(names, 0)
    latency = 0.
    for cohort in cohorts:
        route = {}
        for name in names:
            value = cohort.get('offloaded_' + name)
            if type(value) is not int or value < 0 or value > cohort[name]:
                raise ValueError('Missing/invalid measured offloaded cohort count: ' + name)
            route[name] = value
        if (route['generated'] != route['completed'] + route['failed'] + route['pending'] or
                route['on_time_completed'] > route['completed']):
            raise ValueError('Offloaded cohort conservation failed')
        value = finite(cohort.get('offloaded_completion_latency_ms_sum'), 'offloaded cohort latency')
        all_latency = finite(cohort.get('completion_latency_ms_sum'), 'all-task cohort latency')
        if value < 0 or (not route['completed'] and value != 0) or value > all_latency + 1e-7:
            raise ValueError('Invalid measured offloaded cohort latency')
        latency += value
        for name in names:
            counts[name] += route[name]
    generated = counts['generated']
    return dict(counts,
                on_time_rate=counts['on_time_completed'] / generated if generated else None,
                completion_rate=counts['completed'] / generated if generated else None,
                completed_task_mean_latency_ms=latency / counts['completed'] if counts['completed'] else None)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(folder, stem):
    folder = Path(folder)
    paths = [p for p in (folder / (stem + '.jsonl'), folder / (stem + '.jsonl.gz')) if p.is_file()]
    if len(paths) != 1:
        raise ValueError('One recorded ' + stem + ' stream required')
    opener = gzip.open if paths[0].suffix == '.gz' else open
    with opener(paths[0], 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def artifact_stamp(path):
    payload = load(path)
    return dict(sha256=sha256(path), samples=len(payload['x']),
                trained_samples=payload['trained_samples'], fit_count=payload['fit_count'],
                prediction_batches=payload.get('prediction_batches', 0),
                artifact_version=payload.get('version'),
                fit_parameters=payload.get('fit_parameters',
                    {'max_features': 1.} if payload['trained_samples'] else None),
                feature_names=payload['feature_names'])


def resumed_training_evidence(before, after, initial_payload, updated_payload):
    """A resumed online run must add real history and perform a new fit."""
    if (after['samples'] <= before['samples'] or
            after['trained_samples'] <= before['trained_samples'] or
            after['fit_count'] <= before['fit_count'] or
            after['sha256'] == before['sha256']):
        raise ValueError('Resumed online training must increase samples, trained_samples and fit_count')
    if (updated_payload['feature_names'] != initial_payload['feature_names'] or
            updated_payload['x'][:before['samples']] != initial_payload['x'] or
            updated_payload['y'][:before['samples']] != initial_payload['y']):
        raise ValueError('Resumed training changed the original measured dataset prefix')
    return dict(verified=True, initial_rows_retained=before['samples'],
                new_measured_rows=after['samples'] - before['samples'],
                trained_samples_increase=after['trained_samples'] - before['trained_samples'],
                additional_fits=after['fit_count'] - before['fit_count'],
                before=before, after=after)


def app_evidence(folder, summary, kind, stamp):
    """Check declared RCPs, RF use, ownership confirmation and real actuation."""
    ident = APP_IDS[kind]
    descriptors = summary.get('xapp_library', {}).get('instances', [])
    descriptor = next((a for a in descriptors if a.get('implementation_id') == ident), None)
    learning = summary.get(SUMMARY_KEYS[kind], {})
    fields = set(descriptor.get('writable_actions', [])) if descriptor else set()
    log_name = 'energy_rf_events' if kind == 'x01' else 'latency_rf_events'
    events = Counter(row.get('kind') for row in jsonl(folder, log_name))
    applied, rejected, executed_fields, after_effect = 0, 0, set(), 0
    native_effects, non_native_effects, verified_uids = 0, 0, set()
    scheduler_modes, verified_weights = Counter(), {uid: set() for uid in TARGETS}
    for receipt in jsonl(folder, 'control_receipts'):
        provenance = receipt.get('provenance', [])
        if isinstance(provenance, dict):
            provenance = [provenance]
        if not any(p.get('implementation_id') == ident for p in provenance):
            continue
        if receipt.get('status') == 'rejected':
            rejected += 1
        if receipt.get('status') != 'applied':
            continue
        applied += 1
        executed_fields.update(f['field'] for f in receipt.get('fields', []) if f['field'] in fields)
        after = receipt.get('after', {}).values()
        if kind == 'x12':
            requested = {}
            for field in receipt.get('fields', []):
                if field.get('field') == 'ue_scheduler_weights' and isinstance(field.get('value'), dict):
                    requested.update(field['value'])
            requested = {uid: value for uid, value in requested.items() if uid in TARGETS}
            actual = {}
            modes = []
            for value in after:
                mode = value.get('scheduler_mode')
                if mode not in ('round_robin', 'latency_aware'):
                    continue
                weights = value.get('ue_scheduler_weights', {})
                if not isinstance(weights, dict):
                    continue
                for uid in requested:
                    if uid in weights:
                        actual[uid] = weights[uid]
                        modes.append(mode)
            matches = bool(requested) and all(
                type(weight) in (int, float) and weight in (.5, 1., 2.) and
                type(actual.get(uid)) in (int, float) and
                math.isclose(actual[uid], weight, abs_tol=1e-12, rel_tol=0)
                for uid, weight in requested.items())
            if matches:
                after_effect += 1
                native = all(mode == 'round_robin' for mode in modes) and all(weight == 1 for weight in requested.values())
                native_effects += native
                non_native_effects += not native
                verified_uids.update(requested)
                scheduler_modes.update(modes)
                for uid, weight in requested.items():
                    verified_weights[uid].add(float(weight))
        else:
            after_effect += any(bool(a.get('ue_service_policies')) for a in after)
    unchanged = (learning.get('samples') == stamp['samples'] and
                 learning.get('trained_samples') == stamp['trained_samples'] and
                 learning.get('fit_count') == stamp['fit_count'])
    fit_parameters_unchanged = (kind != 'x12' or
                                learning.get('artifact_fit_parameters') == stamp.get('fit_parameters'))
    prediction_delta = learning.get('prediction_batches', 0) - stamp.get('prediction_batches', 0)
    rf_used = (prediction_delta > 0 if kind == 'x12' else learning.get('rf_selected_actions', 0) > 0)
    confirmed = events['executed_policy_confirmed'] > 0
    active = bool(descriptor and descriptor.get('active_cell_ids'))
    target_weights_verified = kind != 'x12' or set(TARGETS) <= verified_uids
    valid = active and bool(fields) and fields <= executed_fields and applied > 0 and after_effect > 0 and target_weights_verified and confirmed and rf_used and unchanged and fit_parameters_unchanged and rejected == 0
    return dict(implementation_id=ident, declared_rcps=sorted(fields),
                active_cells=descriptor.get('active_cell_ids', []) if descriptor else [],
                learning=learning, event_counts=dict(events), applied_receipts=applied,
                rejected_receipts=rejected, executed_rcps=sorted(executed_fields),
                runtime_effect_receipts=after_effect, executed_policy_confirmed=confirmed,
                native_abstention_only=kind == 'x12' and native_effects > 0 and non_native_effects == 0,
                native_abstention_receipts=native_effects, non_native_scheduler_receipts=non_native_effects,
                target_weights_verified=target_weights_verified,
                verified_scheduler_modes=dict(scheduler_modes),
                verified_target_weights={uid: sorted(values) for uid, values in verified_weights.items()} if kind == 'x12' else {},
                rf_prediction_batches_this_run=prediction_delta if kind == 'x12' else None,
                rf_selected_real_actions=learning.get('rf_selected_actions', 0) > 0,
                rf_selected_action_count=learning.get('rf_selected_actions', 0), genuine_rf_used=rf_used,
                frozen_sample_fit_counts_unchanged=unchanged,
                frozen_fit_parameters_unchanged=fit_parameters_unchanged,
                valid_control_evidence=bool(valid))


def latency_harm(reference, candidate):
    """Operational threshold, not a statistical significance test."""
    if reference is None or candidate is None:
        return dict(delta_ms=None, threshold_ms=None, harmful=False, comparable=False)
    threshold = max(1., .05 * reference)
    delta = candidate - reference
    return dict(delta_ms=delta, threshold_ms=threshold,
                harmful=delta > threshold + 1e-12, comparable=True)


def interaction(off, x01, x12, both):
    return both - x01 - x12 + off


def standalone_effectiveness(metrics, evidence):
    """Do not call an already ineffective isolated controller a successful intent."""
    off = metrics['off']
    result = {}
    for arm in ('x01', 'x12'):
        users = metrics[arm]['users']
        delivery = {uid: users[uid]['delivered_bytes'] >= .95 * off['users'][uid]['delivered_bytes'] - 1e-9
                    for uid in POPULATION}
        task = {uid: users[uid]['tasks']['on_time_rate'] >= off['users'][uid]['tasks']['on_time_rate'] - .05 - 1e-12
                for uid in TARGETS}
        all_task_delay = {uid: latency_harm(off['users'][uid]['tasks']['completed_task_mean_latency_ms'],
                                          users[uid]['tasks']['completed_task_mean_latency_ms'])
                          for uid in TARGETS}
        delay_guard = {uid: value['comparable'] and not value['harmful']
                       for uid, value in all_task_delay.items()}
        deadlines = {}
        for uid in POPULATION - set(TARGETS):
            a, b = off['users'][uid]['packet_deadline_fraction'], users[uid]['packet_deadline_fraction']
            deadlines[uid] = a is None or b is None or b <= a + .05 + 1e-12
        guard = all(delivery.values()) and all(task.values()) and all(delay_guard.values()) and all(deadlines.values())
        if arm == 'x01':
            baseline, candidate = off['target_energy_j'], metrics[arm]['target_energy_j']
            improvement = (baseline - candidate) / baseline if baseline > 0 else None
            non_native = True
        else:
            baseline = off['target_offloaded_task_completed_mean_latency_ms']
            candidate = metrics[arm]['target_offloaded_task_completed_mean_latency_ms']
            improvement = ((baseline - candidate) / baseline
                           if baseline is not None and baseline > 0 and candidate is not None else None)
            non_native = not evidence['x12']['x12']['native_abstention_only']
        result[arm] = dict(improvement_fraction=improvement,
            objective='pooled_target_UE_energy' if arm == 'x01' else 'pooled_target_completed_offloaded_AR_task_mean_E2E_latency',
            required_improvement_fraction=.05, per_ue_delivery_guard=delivery,
            target_task_on_time_guard=task, target_all_task_mean_delay_guard=delay_guard,
            target_all_task_mean_delay_comparison=all_task_delay,
            non_target_packet_deadline_guard=deadlines,
            qos_guards_passed=guard, non_native_control_used=non_native,
            individually_effective=bool(improvement is not None and improvement >= .05 - 1e-12 and guard and non_native))
    result['both_individually_effective'] = all(result[arm]['individually_effective'] for arm in ('x01', 'x12'))
    return result


def evaluate_factorial(folders, stamps, task_start_s=.1):
    """Read one matched seed's four arms; never manufacture a conflict label."""
    if set(folders) != set(ARMS):
        raise ValueError('All four factorial arms are required')
    folders = {arm: Path(path) for arm, path in folders.items()}
    configs = {arm: load(path / 'config_snapshot.json') for arm, path in folders.items()}
    summaries = {arm: load(path / 'final_summary.json') for arm, path in folders.items()}
    base = configs['off']
    manifest = base['manifest']
    if {u['ue_id'] for u in manifest['ues']} != POPULATION or len(manifest['cells']) != 3:
        raise ValueError('Accepted six-UE/three-cell network required')
    for arm in ARMS:
        if configs[arm]['manifest'] != manifest or configs[arm]['configs']['ue'] != base['configs']['ue']:
            raise ValueError('Factorial radio/movement/seed/business mismatch: ' + arm)
    duration = manifest['total_slots'] * manifest['slot_duration_ms'] / 1000
    cohorts = {arm: observed_mature_cohorts(path, manifest['slot_duration_ms'], duration)
               for arm, path in folders.items()}
    metrics, intervals = {}, None
    for arm in ARMS:
        summary = summaries[arm]
        services = summary['ue_services']['per_ue']
        if set(services) != POPULATION:
            raise ValueError('Factorial service population differs')
        users = {}
        for uid in sorted(POPULATION):
            packet = summary['packet_metrics'][uid]
            admitted = packet.get('admitted_packets', 0)
            user = dict(ue_energy_j=finite(services[uid]['ue_energy_total_j'], 'UE energy'),
                        delivered_bytes=packet['delivered_bytes'],
                        goodput_mbps=packet['delivered_bytes'] * 8 / duration / 1e6,
                        packet_mean_latency_ms=packet.get('mean_completion_latency_ms'),
                        packet_deadline_misses=packet.get('deadline_missed_packets', 0),
                        packet_deadline_fraction=packet.get('deadline_missed_packets', 0) / admitted if admitted else None,
                        packet_overflow_bytes=packet.get('overflow_bytes', 0),
                        packet_pending_bytes=packet.get('pending_bytes', 0))
            if uid in TARGETS:
                available = {k: v for k, v in cohorts[arm][uid].items() if k[0] >= task_start_s - 1e-9}
                if not available:
                    raise ValueError('Missing mature task cohorts: ' + arm + '/' + uid)
                if intervals is None:
                    intervals = set(available)
                elif set(available) != intervals:
                    raise ValueError('Mature task intervals differ across arms or UEs')
                scored = [available[k] for k in sorted(available)]
                user['tasks'] = aggregate_cohorts(scored)
                user['offloaded_tasks'] = aggregate_offloaded_cohorts(scored)
            users[uid] = user
        task_counts = {uid: users[uid]['tasks']['generated'] for uid in TARGETS}
        if any(count <= 0 for count in task_counts.values()):
            raise ValueError('No generated task demand in the scored interval')
        if arm != 'off':
            for uid in TARGETS:
                if task_counts[uid] != metrics['off']['users'][uid]['tasks']['generated']:
                    raise ValueError('Generated task demand differs between arms')
        target_completed = sum(users[u]['tasks']['completed'] for u in TARGETS)
        target_generated = sum(task_counts.values())
        latency_sum = sum((users[u]['tasks']['completed_task_mean_latency_ms'] or 0) *
                          users[u]['tasks']['completed'] for u in TARGETS)
        offloaded_completed = sum(users[u]['offloaded_tasks']['completed'] for u in TARGETS)
        offloaded_generated = sum(users[u]['offloaded_tasks']['generated'] for u in TARGETS)
        offloaded_latency_sum = sum((users[u]['offloaded_tasks']['completed_task_mean_latency_ms'] or 0) *
                                    users[u]['offloaded_tasks']['completed'] for u in TARGETS)
        metrics[arm] = dict(users=users,
            target_energy_j=sum(users[u]['ue_energy_j'] for u in TARGETS),
            network_ue_energy_j=sum(u['ue_energy_j'] for u in users.values()),
            target_task_on_time_rate=sum(users[u]['tasks']['on_time_completed'] for u in TARGETS) / target_generated,
            target_task_completed_mean_latency_ms=latency_sum / target_completed if target_completed else None,
            target_offloaded_task_generated=offloaded_generated,
            target_offloaded_task_completed=offloaded_completed,
            target_offloaded_task_on_time_rate=(sum(users[u]['offloaded_tasks']['on_time_completed'] for u in TARGETS) /
                                                offloaded_generated if offloaded_generated else None),
            target_offloaded_task_completed_mean_latency_ms=(offloaded_latency_sum / offloaded_completed
                                                             if offloaded_completed else None),
            network_goodput_mbps=sum(u['goodput_mbps'] for u in users.values()))
    evidence = {}
    for arm, enabled in [('x01', ['x01']), ('x12', ['x12']), ('both', ['x01', 'x12'])]:
        evidence[arm] = {kind: app_evidence(folders[arm], summaries[arm], kind, stamps[kind]) for kind in enabled}
    energy_rcps = set(evidence['both']['x01']['declared_rcps'])
    latency_rcps = set(evidence['both']['x12']['declared_rcps'])
    overlap = energy_rcps & latency_rcps
    valid = all(item['valid_control_evidence'] for case in evidence.values() for item in case.values())
    direct_surface = dict(x01=sorted(energy_rcps), x12=sorted(latency_rcps),
                          intersection=sorted(overlap), disjoint=bool(energy_rcps and latency_rcps and not overlap))
    individual = standalone_effectiveness(metrics, evidence)
    e = {arm: metrics[arm]['target_energy_j'] for arm in ARMS}
    l = {arm: metrics[arm]['target_task_completed_mean_latency_ms'] for arm in ARMS}
    ol = {arm: metrics[arm]['target_offloaded_task_completed_mean_latency_ms'] for arm in ARMS}
    energy_increase = e['both'] / e['x01'] - 1 if e['x01'] else None
    task_comparison, all_task_guards = {}, {}
    for uid in TARGETS:
        t12 = metrics['x12']['users'][uid]['offloaded_tasks']
        tb = metrics['both']['users'][uid]['offloaded_tasks']
        harm = latency_harm(t12['completed_task_mean_latency_ms'], tb['completed_task_mean_latency_ms'])
        harm.update(scope='Actually offloaded AR tasks; conditional routed subpopulations may differ',
                    reference_generated=t12['generated'], candidate_generated=tb['generated'],
                    reference_on_time_rate=t12['on_time_rate'], candidate_on_time_rate=tb['on_time_rate'])
        task_comparison[uid] = harm
        a = metrics['x12']['users'][uid]['tasks']
        b = metrics['both']['users'][uid]['tasks']
        all_harm = latency_harm(a['completed_task_mean_latency_ms'], b['completed_task_mean_latency_ms'])
        all_harm.update(scope='All tasks; global guard separate from the routed primary objective',
                        on_time_rate_change=b['on_time_rate'] - a['on_time_rate'],
                        task_reliability_harm=b['on_time_rate'] < a['on_time_rate'] - .05 - 1e-12)
        all_task_guards[uid] = all_harm
    energy_harm = energy_increase is not None and energy_increase > .02 + 1e-12
    offloaded_harm = any(v['harmful'] for v in task_comparison.values())
    all_guard_harm = any(v['harmful'] or v['task_reliability_harm'] for v in all_task_guards.values())
    task_harm = offloaded_harm or all_guard_harm
    raw_harm = energy_harm or task_harm
    classification = ('insufficient_control_evidence' if not valid else
                      'direct_rcp_overlap_invalidates_indirect_case' if not direct_surface['disjoint'] else
                      'controllers_not_individually_effective' if not individual['both_individually_effective'] else
                      'indirect_operational_conflict_observed' if raw_harm else 'no_detected_conflict')
    non_target = {}
    for uid in sorted(POPULATION - set(TARGETS)):
        a, b = metrics['off']['users'][uid], metrics['both']['users'][uid]
        delta = None if a['packet_deadline_fraction'] is None or b['packet_deadline_fraction'] is None else b['packet_deadline_fraction'] - a['packet_deadline_fraction']
        non_target[uid] = dict(goodput_ratio=b['delivered_bytes'] / a['delivered_bytes'] if a['delivered_bytes'] else None,
            goodput_harm=bool(a['delivered_bytes'] and b['delivered_bytes'] < .95 * a['delivered_bytes']),
            packet_mean_latency=latency_harm(a['packet_mean_latency_ms'], b['packet_mean_latency_ms']),
            packet_deadline_fraction_change=delta, deadline_harm=delta is not None and delta > .05 + 1e-12)
    return dict(seed=manifest['rng_seed'], duration_s=duration,
                objective_scope=objective_scope(),
                matched_radio_movement_business=True, task_scoring_start_s=task_start_s,
                mature_cohort_intervals_s=[list(k) for k in sorted(intervals)], arms=metrics,
                controller_evidence=evidence, controlled_parameter_surface=direct_surface,
                standalone_effectiveness=individual,
                baseline_scheduler_family='round_robin',
                x12_scheduler_family='Native round_robin/weight1 abstention, or latency_aware with explicit target-UE weights after delayed control',
                operational_thresholds=dict(energy_increase_fraction=.02, latency_increase_ms_floor=1.,
                                            latency_increase_fraction=.05, on_time_rate_drop=.05,
                                            primary_latency_scope='Completed offloaded AR-task E2E',
                                            guard_latency_scope='All completed tasks'),
                both_vs_x01_energy_increase_fraction=energy_increase,
                both_vs_x12_task_comparison=task_comparison,
                both_vs_x12_all_task_guards=all_task_guards,
                operational_offloaded_latency_harm=offloaded_harm,
                operational_all_task_guard_harm=all_guard_harm,
                operational_energy_harm=energy_harm, operational_task_harm=task_harm,
                operational_tradeoff_flag=raw_harm,
                task_deadline_non_success_counts={arm: {uid: metrics[arm]['users'][uid]['tasks']['generated'] -
                    metrics[arm]['users'][uid]['tasks']['on_time_completed'] for uid in TARGETS} for arm in ARMS},
                factorial_interaction=dict(definition='D = BOTH - X01 - X12 + OFF',
                    target_energy_j=interaction(*(e[a] for a in ARMS)),
                    target_completed_task_mean_latency_ms=interaction(*(l[a] for a in ARMS)) if all(v is not None for v in l.values()) else None,
                    target_offloaded_task_completed_mean_latency_ms=interaction(*(ol[a] for a in ARMS)) if all(v is not None for v in ol.values()) else None,
                    target_on_time_rate=interaction(*(metrics[a]['target_task_on_time_rate'] for a in ARMS)),
                    latency_scopes='Both all-task and conditional offloaded-AR-task mean E2E are reported separately',
                    harmful_direction='Positive energy/delay D; negative on-time-rate D. These descriptive interactions alone do not establish conflict.'),
                non_target_off_vs_both=non_target, classification=classification,
                interpretation='Operational paired tradeoff thresholds, not statistical significance or proof of objective incompatibility. No conflict is forced.')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--energy-model', type=Path, default=ROOT / 'outputs/models/ue_energy_rf/model.json')
    p.add_argument('--latency-model', type=Path)
    p.add_argument('--train-seconds', type=float, default=4.)
    p.add_argument('--eval-seconds', type=float, default=1.)
    p.add_argument('--train-seed', type=int, default=99)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[55])
    p.add_argument('--window-ms', type=float, default=100.)
    p.add_argument('--explore-windows', type=int, default=8)
    p.add_argument('--traffic-load', type=float, default=1., help='Explicit fixed demand multiplier for all training/evaluation arms')
    training = p.add_mutually_exclusive_group()
    training.add_argument('--skip-training', action='store_true')
    training.add_argument('--resume-online', action='store_true',
                          help='Resume an existing copied X12 artifact using new measured online data')
    p.add_argument('--resume-parent-model', type=Path,
                   help='Optional immutable original artifact whose copy is resumed')
    p.add_argument('--training-cv-report', type=Path,
                   help='Existing training-only CV JSON report, recorded by path and hash before training')
    p.add_argument('--train-only', action='store_true')
    return p.parse_args(argv)


def main():
    args = parse_args()
    output = args.output.resolve()
    energy_model = args.energy_model.resolve()
    latency_model = (args.latency_model or output / 'latency_rf_model.json').resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise ValueError('Choose a fresh study directory under outputs')
    for path in (energy_model, latency_model):
        if not path.is_relative_to(ROOT / 'outputs'):
            raise ValueError('Model artifacts must remain under outputs')
    if args.skip_training and args.train_only:
        raise ValueError('Choose training or existing-model evaluation')
    if (args.skip_training or args.resume_online) and not latency_model.is_file():
        raise ValueError('--skip-training/--resume-online requires an existing X12 model')
    if not args.skip_training and not args.resume_online and latency_model.exists():
        raise ValueError('Online pilot requires a fresh X12 model path')
    if latency_model == energy_model:
        raise ValueError('X01 and X12 model paths must differ')
    initial_payload = load(latency_model) if args.resume_online else None
    initial_stamp = artifact_stamp(latency_model) if args.resume_online else None
    if initial_stamp and (not initial_stamp['trained_samples'] or not initial_stamp['fit_count']):
        raise ValueError('--resume-online requires a previously trained X12 model')
    parent_path = args.resume_parent_model.resolve() if args.resume_parent_model else None
    parent_stamp = None
    if parent_path:
        if (not args.resume_online or parent_path == latency_model or
                not parent_path.is_relative_to(ROOT / 'outputs') or not parent_path.is_file()):
            raise ValueError('Resume parent must be a separate existing artifact under outputs')
        parent_stamp = artifact_stamp(parent_path)
        if parent_stamp['sha256'] != initial_stamp['sha256']:
            raise ValueError('Resumed copy does not match the immutable parent artifact')
    cv_report = None
    if args.training_cv_report:
        cv_path = args.training_cv_report.resolve()
        if not cv_path.is_relative_to(ROOT / 'outputs') or not cv_path.is_file():
            raise ValueError('Training CV report must be an existing JSON file under outputs')
        load(cv_path)
        cv_report = dict(path=str(cv_path), sha256=sha256(cv_path),
                         scope='Training-only CV supplied before this run; no validation outcomes used for selection')
    if not args.train_only and not energy_model.is_file():
        raise ValueError('Factorial evaluation requires the accepted X01 model')
    for value in (args.train_seconds, args.eval_seconds, args.window_ms, args.traffic_load):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite duration/load required')
    if args.window_ms != 100 or args.explore_windows < 1 or not args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds) or args.train_seed in args.eval_seeds:
        raise ValueError('Use 100 ms cohorts, a positive exploration budget and distinct held-out X12 evaluation seeds')
    output.mkdir(parents=True)
    latency_model.parent.mkdir(parents=True, exist_ok=True)
    energy_stamp = artifact_stamp(energy_model) if energy_model.is_file() else None
    receipt = dict(completed=False, status='initializing', cases=[], factorials=[],
                   objective_scope=objective_scope(),
                   train_seed=args.train_seed, evaluation_seeds=args.eval_seeds,
                   evaluation_role='Seed 55 is refinement validation informed by the failed first pilot; use a predeclared new seed such as 77 only after validation passes.',
                   seed_roles={str(seed): 'refinement_validation' if seed == 55 else
                               'independent_confirmation' if seed == 77 else 'additional_evaluation'
                               for seed in args.eval_seeds},
                   validation_passed_by_seed={},
                   train_seconds=args.train_seconds, evaluation_seconds=args.eval_seconds,
                   traffic_load=args.traffic_load, explicit_stress=args.traffic_load != 1,
                   x01_model=str(energy_model), x12_model=str(latency_model),
                   training_lineage=dict(mode='resume_online' if args.resume_online else
                                         'skipped' if args.skip_training else 'fresh_online',
                       initial_copied_artifact=initial_stamp,
                       immutable_parent=dict(path=str(parent_path), stamp=parent_stamp) if parent_path else None,
                       training_cv_report=cv_report),
                   source_frozen_required=True,
                   scope='Native six-UE network; modelled UE power, abstract shared UL, actual NR/SYS DL. Offloaded AR-task E2E primary; all-task guards. Conditional routed populations may differ. Short paired factorial study; no forced conflict or significance claim.')

    def write():
        temporary = output / 'acceptance.tmp'
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False))
        temporary.replace(output / 'acceptance.json')

    def run(name, seed, seconds, energy_mode, latency_mode):
        receipt.update(status='running', running=name)
        write()
        folder = output / name
        command = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
                   '--duration-s', str(seconds), '--seed', str(seed), '--traffic-load', str(args.traffic_load),
                   '--ue-services', 'drx-mec', '--record-compression', 'gzip',
                   '--energy-xapp-mode', energy_mode, '--latency-xapp-mode', latency_mode,
                   '--latency-xapp-window-ms', str(args.window_ms),
                   '--latency-xapp-explore-windows', str(args.explore_windows),
                   '--channel-cache-root', str(ROOT / 'channel_cache' / output.name / f'seed_{seed}'),
                   '--output', str(folder)]
        if energy_mode != 'off':
            command.extend(['--energy-xapp-model', str(energy_model)])
        if latency_mode != 'off':
            command.extend(['--latency-xapp-model', str(latency_model)])
        start = time.perf_counter()
        with (output / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed with code {result.returncode}; inspect its log')
        status, perf = load(folder / 'run_status.json'), load(folder / 'performance.json')
        if not status.get('completed') or not status.get('replay', {}).get('audits_passed'):
            raise ValueError('Simulation/HTML audit incomplete: ' + name)
        receipt['cases'].append(dict(name=name, seed=seed, energy_mode=energy_mode, latency_mode=latency_mode,
            output=str(folder), simulated_s=seconds, command=command,
            source_sha256=status['source_sha256'], simulation_wall_s=perf['simulation_s'],
            internal_wall_s=status['elapsed_s'], external_wall_s=time.perf_counter()-start,
            html=status['replay']['html'], html_audits_passed=True))
        if energy_stamp and sha256(energy_model) != energy_stamp['sha256']:
            raise ValueError('X12 study modified the accepted X01 model')
        if parent_stamp and sha256(parent_path) != parent_stamp['sha256']:
            raise ValueError('Online study modified the immutable parent X12 model')
        if cv_report and sha256(cv_report['path']) != cv_report['sha256']:
            raise ValueError('Training CV report changed during the study')
        write()
        return folder

    write()
    try:
        if not args.skip_training:
            run('x12_online_training', args.train_seed, args.train_seconds, 'off', 'online')
        if not latency_model.is_file():
            raise ValueError('X12 training did not produce a model')
        latency_stamp = artifact_stamp(latency_model)
        if args.resume_online:
            update = resumed_training_evidence(initial_stamp, latency_stamp,
                                                initial_payload, load(latency_model))
            events = Counter(row.get('kind') for row in jsonl(output / 'x12_online_training', 'latency_rf_events'))
            if events['measured_sample'] != update['new_measured_rows']:
                raise ValueError('Resumed dataset growth does not match newly measured online event rows')
            training = load(output / 'x12_online_training' / 'final_summary.json')['latency_xapp']
            if (training.get('artifact_fit_parameters') != latency_stamp['fit_parameters'] or
                    training.get('configured_fit_parameters') != latency_stamp['fit_parameters']):
                raise ValueError('Resumed fitted parameters do not match the actual online configuration')
            update['new_online_measured_events'] = events['measured_sample']
            receipt['training_lineage']['online_update'] = update
        receipt['model_stamps'] = dict(x01=energy_stamp, x12=latency_stamp)
        if args.train_only:
            receipt.update(completed=True, status='training_completed_not_evaluated')
        else:
            for repeat, seed in enumerate(args.eval_seeds):
                folders = {}
                # Alternate order between complete seeds; cache reuse is for cost,
                # not a claim that later-arm wall time is xApp inference savings.
                order = ARMS if repeat % 2 == 0 else tuple(reversed(ARMS))
                for arm in order:
                    folders[arm] = run(f'seed{seed}_{arm}', seed, args.eval_seconds,
                                       'frozen' if arm in ('x01', 'both') else 'off',
                                       'frozen' if arm in ('x12', 'both') else 'off')
                    if sha256(latency_model) != latency_stamp['sha256']:
                        raise ValueError('Evaluation modified the frozen X12 model')
                result = evaluate_factorial(folders, dict(x01=energy_stamp, x12=latency_stamp))
                receipt['factorials'].append(result)
                validated = (result['standalone_effectiveness']['both_individually_effective'] and
                             result['controlled_parameter_surface']['disjoint'] and
                             all(item['valid_control_evidence'] for case in result['controller_evidence'].values()
                                 for item in case.values()))
                receipt['validation_passed_by_seed'][str(seed)] = validated
                write()
                if not validated:
                    receipt['additional_seeds_skipped'] = args.eval_seeds[repeat + 1:]
                    break
            receipt.update(completed=True, status='factorial_completed' if all(receipt['validation_passed_by_seed'].values()) else 'needs_refinement',
                           classifications={str(f['seed']): f['classification'] for f in receipt['factorials']})
        if len({c['source_sha256'] for c in receipt['cases']}) != 1:
            raise ValueError('Source/config changed during the study')
        receipt.pop('running', None)
        write()
    except BaseException as error:
        receipt.update(completed=False, status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        write()
        raise
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
