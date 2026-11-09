"""Bounded, matched X08 / UE6-URLLC scheduler request conflict experiment.

The throughput model is immutable. The new RF receives only measured online
windows. Four frozen evaluations report objective gains and QoS costs separately;
neither random arbitration nor a non-additive KPI proves an indirect conflict.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_latency_energy_study import jsonl
from experiments.run_ue_energy_study import load, POPULATION
from experiments.run_throughput_mlp_study import (
    actual_packet_metrics, arrival_audit, collection_eval_compatibility,
    model_stamp as throughput_stamp, readiness as throughput_readiness, sha256)

X08 = 'throughput_scheduler_mlp'
X12 = 'latency_urllc_rf'
ARMS = ('off', 'x08', 'urllc', 'both_random')
FIELDS = {'scheduler_mode', 'ue_scheduler_weights'}
LOGS = {X08: 'throughput_mlp_events', X12: 'urllc_rf_events'}
SUMMARIES = {X08: 'throughput_xapp', X12: 'urllc_xapp'}
RF_TARGETS = ['ue_6_unique_failure_events_per_s', 'ue_6_completion_latency_ms_per_s',
              'ue_6_completed_packets_per_s', 'network_goodput_mbps', 'ue_2_unique_failure_events_per_s']
RF_FEATURES = ['network_queue_kb', 'network_reported_goodput_mbps', 'network_offered_mbps',
    'ue_6_queue_kb', 'ue_6_reference_sinr_db', 'ue_6_reported_goodput_mbps',
    'ue_2_queue_kb', 'backlogged_ue_fraction'] + ['action_' + str(i) for i in range(4)]
X08_FEATURES = ['network_queue_kb', 'network_reported_goodput_mbps', 'network_offered_mbps',
    'minimum_reference_sinr_db', 'mean_reference_sinr_db', 'urllc_max_hol_budget_ratio',
    'backlogged_ue_fraction', 'urllc_cumulative_failure_events'] + ['action_' + str(i) for i in range(4)]
X08_TARGETS = ['network_goodput_mbps'] + [u + '_goodput_mbps' for u in sorted(POPULATION)] + [u + '_failure_event_sum_per_s' for u in ('ue_2', 'ue_6')]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _f32(values):
    return [struct.unpack('f', struct.pack('f', v))[0] for v in values]


def rf_stamp(path):
    p = load(path)
    return dict(sha256=sha256(path), format=p['format'], version=p['version'],
        samples=len(p['x']), trained_samples=p['trained_samples'], fit_count=p['fit_count'],
        prediction_batches=p.get('prediction_batches', 0), feature_names=p['feature_names'],
        fit_parameters=p.get('fit_parameters'))


def rf_readiness(path):
    if not Path(path).is_file():
        return dict(ready=False, reason='No actual RF model artifact')
    p = load(path)
    from ric.near_rt.pools.chongyu.online_packet_rf import OnlinePacketRF
    # Strict JSON loading/inference validation only: no fit or sklearn import.
    OnlinePacketRF.load(path, RF_FEATURES)
    names = p['feature_names']
    indices = [names.index('action_' + str(i)) for i in range(4)]
    fitted = p['trained_samples']
    counts = Counter()
    for x in p['x'][:fitted]:
        action = [x[i] for i in indices]
        if any(v not in (0., 1.) for v in action) or sum(action) != 1:
            raise ValueError('RF training action must be an actual one-hot candidate')
        counts[action.index(1.)] += 1
    ready = (p.get('format') == 'chongyu.online-packet-rf' and p.get('target_names') == RF_TARGETS and
             0 <= fitted <= len(p['x']) <= 48 and len(p['x']) == len(p['y']) and
             fitted >= 24 and all(counts[i] >= 6 for i in range(4)) and
             p['fit_count'] > 0 and bool(p.get('forest')))
    return dict(ready=bool(ready), samples=len(p['x']), trained_samples=fitted,
        fitted_action_counts={str(i): counts[i] for i in range(4)}, fit_count=p['fit_count'],
        minimum_actual_samples=24, minimum_actual_samples_per_action=6, maximum_actual_rows=48)


def completed_case(folder):
    folder = Path(folder)
    status, config = load(folder / 'run_status.json'), load(folder / 'config_snapshot.json')
    if not status.get('completed') or not status.get('replay', {}).get('audits_passed'):
        raise ValueError('Completed simulation and audited HTML required: ' + str(folder))
    html = Path(status['replay'].get('html', 'network-replay.html'))
    if not html.is_absolute():
        html = folder / html
    if not html.is_file():
        raise ValueError('Audited HTML is missing')
    source = status.get('source_sha256')
    if not source or config.get('source_sha256') != source:
        raise ValueError('Snapshot/run source fingerprints disagree')
    raw = {str(p): sha256(p) for p in folder.iterdir() if p.name in
           ('packet_events.jsonl', 'packet_events.jsonl.gz', 'rt_sys_trace.jsonl', 'rt_sys_trace.jsonl.gz',
            'urllc_rf_events.jsonl', 'throughput_mlp_events.jsonl', 'rule_decisions.jsonl',
            'rule_decisions.jsonl.gz', 'control_receipts.jsonl', 'control_receipts.jsonl.gz')}
    return dict(source_sha256=source, html=str(html), html_sha256=sha256(html), raw_file_sha256=raw,
                config_sha256=sha256(folder / 'config_snapshot.json'),
                summary_sha256=sha256(folder / 'final_summary.json'), html_audits_passed=True)


def _snapshot_verified(snapshot, owners, membership, source):
    if set(membership) != POPULATION or set(membership.values()) - set(snapshot):
        return False
    covered = set()
    for cid, cell in snapshot.items():
        expected = {u for u, serving in membership.items() if serving == cid}
        actual = cell.get('ue_scheduler_weights', {})
        owner = owners.get(cid, {})
        if cell.get('scheduler_mode') not in ('round_robin', 'latency_aware') or owner.get('scheduler_mode') != source:
            return False
        if expected and (owner.get('ue_scheduler_weights') != source or set(actual) != expected):
            return False
        if any(not _finite(v) or v not in (1., 1.5, 2.) for v in actual.values()):
            return False
        covered.update(actual)
    return covered == POPULATION


def _confirmation_actuated(confirmation, receipts):
    """A selected owner is insufficient without an earlier applied ACK."""
    produced = confirmation['produced_slot']
    for cid, snapshot in confirmation['executed_snapshot'].items():
        fields = ['scheduler_mode'] + (['ue_scheduler_weights'] if snapshot['ue_scheduler_weights'] else [])
        for field in fields:
            candidates = []
            for index, receipt in enumerate(receipts):
                applied = receipt.get('apply_slot', receipt.get('slot_id'))
                requested = _requested(receipt)
                if (receipt.get('status') == 'applied' and type(applied) is int and applied <= produced and
                        cid in receipt.get('after', {}) and field in requested):
                    candidates.append((applied, index, receipt, requested))
            if not candidates:
                return False
            _, _, receipt, requested = max(candidates, key=lambda c: c[:2])
            provenance = receipt.get('provenance', [])
            if isinstance(provenance, dict):
                provenance = [provenance]
            if (not any(p.get('implementation_id') == X12 for p in provenance) or
                    receipt['after'][cid].get(field) != snapshot[field]):
                return False
            if field == 'scheduler_mode' and requested[field] != snapshot[field]:
                return False
            if field == 'ue_scheduler_weights' and any(requested[field].get(u) != v for u, v in snapshot[field].items()):
                return False
    return True


def training_evidence(folder, model, previous=None, payload=None):
    """Audit every actual window, including outcomes observed after the row cap."""
    folder = Path(folder)
    cfg, summary = load(folder / 'config_snapshot.json'), load(folder / 'final_summary.json')
    apps = cfg['configs']['ric'].get('xapps', [])
    if len(apps) != 1 or apps[0].get('implementation_id') != X12 or apps[0]['parameters'].get('mode') != 'online':
        raise ValueError('Training must isolate the genuine online URLLC controller')
    p, before = payload or load(model), previous or {'x': [], 'y': []}
    if p['feature_names'] != RF_FEATURES or p['target_names'] != RF_TARGETS:
        raise ValueError('RF measured schema must be the explicit packet-outcome model')
    if p['x'][:len(before['x'])] != before['x'] or p['y'][:len(before['y'])] != before['y']:
        raise ValueError('Online extension changed historical measured rows')
    events = list(jsonl(folder, LOGS[X12]))
    receipts = list(jsonl(folder, 'control_receipts'))
    confirmations, samples = [], []
    last_end = -1
    slot_ms, stop = cfg['manifest']['slot_duration_ms'], cfg['manifest']['total_slots']
    for row in events:
        if row.get('kind') == 'executed_policy_confirmed':
            confirmations.append(row)
        if row.get('kind') != 'measured_sample':
            continue
        start, end, duration = row['produced_start_slot'], row['produced_end_slot'], row['duration_s']
        if not (type(start) is int and type(end) is int and start >= last_end and end <= stop and
                end > start and duration >= .1 - 1e-12 and math.isclose(duration, (end - start) * slot_ms / 1000, abs_tol=1e-12)):
            raise ValueError('Invalid actual RF measurement horizon')
        eligible = [c for c in confirmations if c['produced_slot'] <= start]
        confirmation = eligible[-1] if eligible else None
        if (confirmation is None or confirmation.get('action') != row['action'] or confirmation.get('executed_snapshot') != row['snapshot'] or
                (start - confirmation['produced_slot']) * slot_ms < 20 - 1e-9 or
                not _snapshot_verified(row['snapshot'], confirmation.get('owners', {}), row['membership'], X12) or
                not _confirmation_actuated(confirmation, receipts)):
            raise ValueError('RF label lacks complete actual ownership/ACK and 20 ms wash-in')
        # Exact target arithmetic is checked below against the packet deltas.
        _verify_training_label(row)
        samples.append(row)
        last_end = end
    _verify_raw_packet_windows(folder, samples, cfg)
    room = 48 - len(before['x'])
    added = samples[:room]
    if any(r.get('retained_for_training') is not (i < room) for i, r in enumerate(samples)):
        raise ValueError('Measured row retention flags disagree with the actual fixed cap')
    if p['x'][len(before['x']):] != [_f32(r['features']) for r in added] or p['y'][len(before['y']):] != [r['targets'] for r in added]:
        raise ValueError('Persisted RF rows differ from actual measured feature/label prefix')
    learned = summary[SUMMARIES[X12]]
    if learned.get('samples') != len(p['x']) or learned.get('trained_samples') != p['trained_samples'] or learned.get('fit_count') != p['fit_count']:
        raise ValueError('Actual RF summary and saved fitted-prefix counts disagree')
    fits = [r for r in events if r.get('kind') == 'rf_fit']
    if p['fit_count'] != before.get('fit_count', 0) + len(fits) or any(
            r.get('fit_count') != before.get('fit_count', 0) + i + 1 for i, r in enumerate(fits)):
        raise ValueError('RF fitted counters lack genuine online fit events')
    if fits and fits[-1].get('samples') != p['trained_samples']:
        raise ValueError('Last actual RF fit differs from the saved fitted prefix')
    return dict(new_actual_rows=len(added), observed_measured_windows=len(samples),
        measured_windows_not_added_due_to_row_cap=len(samples) - len(added),
        actual_packet_deltas_and_ack_verified=True, actual_applied_ack_before_produced_confirmation_verified=True,
        old_measured_prefix_unchanged=True,
        event_counts=dict(Counter(e.get('kind') for e in events)))


def _verify_raw_packet_windows(folder, samples, config):
    """Compare public window deltas with immutable, packet-ID event records."""
    from bisect import bisect_right
    records = defaultdict(list)
    first_failure = {}
    budgets = {u['ue_id']: u['latency_budget_ms'] for u in config['configs']['ue']['ues']}
    if any(budgets[u] != 10 for u in ('ue_2', 'ue_6')):
        raise ValueError('Original UE2/UE6 10 ms deadlines must be preserved')
    for event in jsonl(folder, 'packet_events'):
        uid, slot, kind = event['ue_id'], event['slot_id'], event['kind']
        if kind == 'arrival':
            records[('offered_bits_by_ue', uid)].append((slot, event['bytes'] * 8))
            if uid in ('ue_2', 'ue_6'):
                name = 'admitted_packets_by_ue' if event['admitted'] else 'overflow_packets_by_ue'
                records[(name, uid)].append((slot, 1))
        if kind == 'delivery':
            records[('delivered_bits_by_ue', uid)].append((slot, event['bytes'] * 8))
            if uid in ('ue_2', 'ue_6') and event['completed']:
                latency = event['latency_ms']
                records[('completed_packets_by_ue', uid)].append((slot, 1))
                records[('completion_latency_ms_by_ue', uid)].append((slot, latency))
                name = 'late_completed_packets_by_ue' if latency > budgets[uid] else 'on_time_completed_packets_by_ue'
                records[(name, uid)].append((slot, 1))
                if latency > budgets[uid]:
                    key = (uid, event['packet_id'])
                    first_failure[key] = min(first_failure.get(key, slot), slot)
        if kind == 'deadline_miss' and uid in ('ue_2', 'ue_6'):
            key = (uid, event['packet_id'])
            first_failure[key] = min(first_failure.get(key, slot), slot)
    for (uid, _), slot in first_failure.items():
        records[('deadline_failed_packets_by_ue', uid)].append((slot, 1))
    prefix = {}
    for key, values in records.items():
        ordered = sorted(values)
        slots, sums = [], [0]
        for slot, value in ordered:
            slots.append(slot)
            sums.append(sums[-1] + value)
        prefix[key] = (slots, sums)
    names = ('delivered_bits_by_ue', 'offered_bits_by_ue', 'deadline_failed_packets_by_ue',
             'admitted_packets_by_ue', 'overflow_packets_by_ue', 'completed_packets_by_ue',
             'on_time_completed_packets_by_ue', 'late_completed_packets_by_ue', 'completion_latency_ms_by_ue')
    for sample in samples:
        start, end = sample['produced_start_slot'], sample['produced_end_slot']
        for name in names:
            for uid, measured in sample[name].items():
                slots, sums = prefix.get((name, uid), ([], [0]))
                actual = sums[bisect_right(slots, end)] - sums[bisect_right(slots, start)]
                if not math.isclose(measured, actual, abs_tol=1e-7, rel_tol=1e-12):
                    raise ValueError('Public RF counter delta differs from raw packet events: ' + name + '/' + uid)


def _verify_training_label(row):
    action = row['action']
    if type(action) is not int or not 0 <= action <= 3 or row['features'][-4:] != [float(action == i) for i in range(4)]:
        raise ValueError('RF actual action and feature encoding disagree')
    if row.get('target_order') != RF_TARGETS or row.get('original_latency_budgets_ms') != {'ue_2': 10, 'ue_6': 10}:
        raise ValueError('RF target order or original packet deadlines changed')
    for cell in row['snapshot'].values():
        expected_mode = 'round_robin' if action == 0 else 'latency_aware'
        if cell['scheduler_mode'] != expected_mode or any(
                weight != ((1.5 if action == 2 else 2. if action == 3 else 1.) if uid == 'ue_6' else 1.)
                for uid, weight in cell['ue_scheduler_weights'].items()):
            raise ValueError('RF action differs from its held actual UE6 priority map')
    duration = row['duration_s']
    for name in ('delivered_bits_by_ue', 'offered_bits_by_ue'):
        values = row[name]
        if set(values) != POPULATION or any(type(v) is not int or v < 0 for v in values.values()):
            raise ValueError('Incomplete actual per-UE bit counters')
    for name in ('deadline_failed_packets_by_ue', 'admitted_packets_by_ue', 'overflow_packets_by_ue',
                 'completed_packets_by_ue', 'on_time_completed_packets_by_ue', 'late_completed_packets_by_ue'):
        values = row[name]
        if set(values) != {'ue_2', 'ue_6'} or any(type(v) is not int or v < 0 for v in values.values()):
            raise ValueError('Incomplete actual URLLC packet counter deltas')
    latency = row['completion_latency_ms_by_ue']
    if set(latency) != {'ue_2', 'ue_6'} or any(not _finite(v) or v < 0 for v in latency.values()):
        raise ValueError('Invalid actual completion-latency sum')
    for uid in ('ue_2', 'ue_6'):
        completed = row['completed_packets_by_ue'][uid]
        if completed != row['on_time_completed_packets_by_ue'][uid] + row['late_completed_packets_by_ue'][uid]:
            raise ValueError('Actual completed packet accounting failed')
        if completed == 0 and latency[uid] != 0:
            raise ValueError('Nonzero completion latency without any actual completion')
        mean = row['completion_mean_ms_by_ue'][uid]
        if mean != (latency[uid] / completed if completed else None):
            raise ValueError('Actual completion mean must be undefined without completions')
    failed = lambda uid: row['deadline_failed_packets_by_ue'][uid] + row['overflow_packets_by_ue'][uid]
    expected = [failed('ue_6') / duration, latency['ue_6'] / duration,
                row['completed_packets_by_ue']['ue_6'] / duration,
                sum(row['delivered_bits_by_ue'].values()) / duration / 1e6, failed('ue_2') / duration]
    if len(row['targets']) != 5 or any(not _finite(a) or not math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-12)
                                        for a, b in zip(row['targets'], expected)):
        raise ValueError('RF target labels disagree with actual packet-delta arithmetic')


def _requested(receipt):
    return {f['field']: f['value'] for f in receipt.get('fields', []) if f.get('field') in FIELDS}


def trained_lineage(model, x08_stamp):
    """Re-audit the completed train phase without modifying its files or model."""
    model = Path(model)
    path = model.parent / 'acceptance.json'
    receipt = load(path)
    stamp = rf_stamp(model)
    if (not receipt.get('completed') or receipt.get('status') != 'trained_no_evaluation' or
            receipt.get('urllc_model_before_evaluation') != stamp or
            receipt.get('throughput_model_before', {}).get('sha256') != x08_stamp['sha256']):
        raise ValueError('Evaluation requires the immutable completed train-phase receipt and original X08 artifact')
    cases = receipt.get('cases', [])
    if not 1 <= len(cases) <= 2 or any(c.get('arm') != 'train' for c in cases):
        raise ValueError('Only the bounded isolated online training episodes are reusable')
    p, previous, audited, protected = load(model), None, [], {str(path): sha256(path), str(model): sha256(model)}
    for case in cases:
        folder = Path(case['output'])
        audit = completed_case(folder)
        if audit != case.get('audit'):
            raise ValueError('Immutable collector status/HTML/raw-recording hashes changed')
        learning = load(folder / 'final_summary.json')[SUMMARIES[X12]]
        count = learning['samples']
        prefix = dict(p, x=p['x'][:count], y=p['y'][:count],
                      trained_samples=learning['trained_samples'], fit_count=learning['fit_count'])
        proof = training_evidence(folder, model, previous, payload=prefix)
        audited.append(dict(collection_folder=str(folder), audit=audit, actual_training_evidence=proof))
        for name in ('run_status.json', 'config_snapshot.json', 'final_summary.json'):
            protected[str(folder / name)] = sha256(folder / name)
        protected.update(audit['raw_file_sha256'])
        protected[audit['html']] = audit['html_sha256']
        previous = prefix
    if len(previous['x']) != len(p['x']) or not rf_readiness(model)['ready']:
        raise ValueError('Completed collectors do not cover the entire genuinely fitted RF artifact')
    return dict(training_receipt=str(path), collection_folder=audited[0]['collection_folder'],
        collector_source_sha256s=sorted({a['audit']['source_sha256'] for a in audited}),
        measured_online_episodes=audited, original_training_simulated_s=receipt['training_simulated_s'],
        protected_original_inputs=protected, actual_model_sha256=stamp['sha256'],
        source_scope='Original priority-only collector source is explicitly separate from the new scheduler-arbitration evaluation source. No retraining or row replacement; paired evaluation arms must share one source.')


def protect_lineage(lineage):
    for path, digest in lineage.get('protected_original_inputs', {}).items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise ValueError('Historical training artifact or recording changed: ' + path)


def _executed_match(receipt, requested):
    if set(requested) != FIELDS or requested['scheduler_mode'] not in ('round_robin', 'latency_aware'):
        return False, set()
    actual = {}
    cells = receipt.get('after', {})
    for cell in cells.values():
        if cell.get('scheduler_mode') != requested['scheduler_mode']:
            return False, set()
        actual.update(cell.get('ue_scheduler_weights', {}))
    weights = requested['ue_scheduler_weights']
    return (bool(cells) and bool(weights) and all(actual.get(u) == v for u, v in weights.items())), set(weights)


def selected_action(app, predictions):
    """Reproduce the declared frozen selection rule, without invoking the app."""
    if app == X08:
        index = max(range(4), key=lambda i: (predictions[i][0], -i))
        if index == 0 or predictions[index][0] < 1.01 * max(0., predictions[0][0]):
            index = 0
        return index
    tolerance = .001 / .1
    feasible = [i for i in range(4) if predictions[i][4] <= predictions[0][4] + tolerance]
    lowest_failure = min(predictions[i][0] for i in feasible)
    candidates = [i for i in feasible if predictions[i][0] <= lowest_failure + tolerance]
    means = [p[1] / p[2] if p[2] > 0 else None for p in predictions]
    finite = [means[i] for i in candidates if means[i] is not None]
    if finite:
        best = min(finite)
        candidates = [i for i in candidates if means[i] is not None and means[i] <= best * 1.01 + 1e-9]
    index = max(candidates, key=lambda i: (predictions[i][3], -i))
    if 0 in candidates and predictions[index][3] < 1.01 * predictions[0][3]:
        index = 0
    return index


def prediction_evidence(events, app, stamp, model_path):
    errors, replayed, choices = [], 0, Counter()
    path = Path(model_path) if model_path else None
    if path is not None and not path.is_absolute():
        path = ROOT / path
    try:
        if path is None or not path.is_file() or sha256(path) != stamp['sha256']:
            raise ValueError('Configured frozen inference artifact is missing or differs from its stamp')
        if app == X08:
            from ric.near_rt.pools.chongyu.online_throughput_mlp import OnlineThroughputMLP
            learner = OnlineThroughputMLP.load(path, X08_FEATURES, X08_TARGETS)
            if not learner.optimizer_verified:
                raise ValueError('Unverified MLP optimizer')
        else:
            from ric.near_rt.pools.chongyu.online_packet_rf import OnlinePacketRF
            learner = OnlinePacketRF.load(path, RF_FEATURES)
        for event in events:
            if event.get('kind') != 'decision':
                continue
            features, action = event.get('features'), event.get('action')
            if (not isinstance(features, list) or len(features) != 12 or any(not _finite(v) for v in features) or
                    type(action) is not int or not 0 <= action < 4 or
                    features[-4:] != [float(action == i) for i in range(4)] or
                    event.get('status') != ('mlp' if app == X08 else 'rf')):
                raise ValueError('Frozen decision lacks actual inference context/action encoding')
            predictions = learner.predict([features[:-4] + [float(i == j) for j in range(4)] for i in range(4)])
            audit = event.get('prediction_guard_audit' if app == X08 else 'prediction_audit') or {}
            recorded = audit.get('raw_predictions')
            if (not isinstance(recorded, list) or len(recorded) != 4 or
                    any(not isinstance(a, list) or len(a) != len(b) or any(not _finite(x) or not math.isclose(x, y, abs_tol=1e-9, rel_tol=1e-10)
                        for x, y in zip(a, b)) for a, b in zip(recorded, predictions))):
                raise ValueError('Recorded forecasts differ from strict frozen-artifact replay')
            expected = selected_action(app, predictions)
            chosen = event.get('prediction')
            if (action != expected or not isinstance(chosen, list) or len(chosen) != len(predictions[action]) or
                    any(not _finite(x) or not math.isclose(x, y, abs_tol=1e-9, rel_tol=1e-10) for x, y in zip(chosen, predictions[action]))):
                raise ValueError('Actual frozen action does not follow its model forecasts and declared selection rule')
            if app == X08 and (event.get('policy') != 'throughput' or event.get('constraints_active') is not False or audit.get('rejected_actions') != {}):
                raise ValueError('X08 replay must be explicitly unconstrained throughput policy')
            if app == X12 and audit.get('selected_action') != action:
                raise ValueError('RF selection audit differs from its actual decision')
            replayed += 1
            choices[str(action)] += 1
        if sha256(path) != stamp['sha256']:
            raise ValueError('Offline inference replay changed the immutable model bytes')
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        errors.append(str(error))
    return dict(valid=bool(replayed and not errors), replayed_frozen_decisions=replayed,
        selected_action_counts=dict(choices), errors=errors,
        scope='Strict JSON models replay actual logged feature contexts, all four forecasts and declared frozen selection. Predict counters alone are insufficient; replay never fits or saves.')


def controller_evidence(folder, summary, app, stamp, combined=False, model_path=None):
    learning = summary.get(SUMMARIES[app], {})
    descriptor = next((a for a in summary.get('xapp_library', {}).get('instances', [])
                       if a.get('implementation_id') == app), {})
    events = list(jsonl(folder, LOGS[app]))
    decisions = sum(e.get('kind') == 'decision' for e in events)
    confirmations = sum(e.get('kind') == 'executed_policy_confirmed' for e in events)
    verified, effects = set(), 0
    for receipt in jsonl(folder, 'control_receipts'):
        provenance = receipt.get('provenance', [])
        if isinstance(provenance, dict):
            provenance = [provenance]
        if receipt.get('status') != 'applied' or not any(p.get('implementation_id') == app for p in provenance):
            continue
        matches, uids = _executed_match(receipt, _requested(receipt))
        if matches:
            effects += 1
            verified.update(uids)
    frozen = (learning.get('mode') == 'frozen' and learning.get('model_sha256') == stamp['sha256'] and
              all(learning.get(k) == stamp[k] for k in ('samples', 'trained_samples', 'fit_count')) and
              learning.get('fit_parameters') == stamp.get('fit_parameters'))
    if app == X08:
        frozen = (frozen and learning.get('policy') == 'throughput' and learning.get('constraints_active') is False and
                  learning.get('optimizer_verified') is True and stamp.get('optimizer_verified') is True and
                  learning.get('optimizer_metadata') == stamp.get('optimizer_metadata'))
    predictions = learning.get('prediction_batches', 0) - stamp.get('prediction_batches', 0)
    inference = prediction_evidence(events, app, stamp, model_path)
    # A finite coin sequence need not execute every contender. In BOTH the
    # arbitration audit proves selected requests; losing apps may lack ACKs.
    runtime = decisions > 0 if combined else confirmations > 0 and effects > 0 and verified == POPULATION
    valid = (frozen and predictions > 0 and inference['valid'] and inference['replayed_frozen_decisions'] == predictions and
             runtime and bool(descriptor.get('active_cell_ids')) and
             FIELDS <= set(descriptor.get('writable_actions', [])))
    return dict(valid=bool(valid), frozen_model_unchanged=bool(frozen),
        genuine_prediction_batches=predictions, decisions=decisions, confirmations=confirmations,
        matching_runtime_receipts=effects, verified_uids=sorted(verified), inference_replay=inference, learning=learning)


def arbitration_evidence(folder, mitigation_seed):
    config = load(Path(folder) / 'config_snapshot.json')
    window_slots = 100 / config['manifest']['slot_duration_ms']
    if not math.isclose(window_slots, round(window_slots), abs_tol=1e-9):
        raise ValueError('100 ms mitigation must align to slots')
    window_slots = round(window_slots)
    selections, draws = {}, defaultdict(dict)
    errors, conflicts = [], set()
    for row in jsonl(folder, 'rule_decisions'):
        if row.get('field') != 'scheduler_bundle' or row.get('reason') not in ('direct_conflict_random_selection', 'direct_conflict_request_selection'):
            continue
        candidates = row.get('candidates', [])
        if not candidates and row.get('winner') == 'rule_withdrawal':
            continue
        sources = [c.get('source') for c in candidates]
        values = {c.get('source'): c.get('requested_value') for c in candidates}
        cid, index, generation = row.get('cell_id'), row.get('window_index'), row.get('generation')
        key = (cid, index, generation)
        if not sources or sources != sorted(set(sources)) or not set(sources) <= {X08, X12}:
            errors.append('Unexpected or duplicated scheduler contender')
            continue
        if any(not isinstance(v, dict) or set(v) != FIELDS or
               v['scheduler_mode'] not in ('round_robin', 'latency_aware') or
               not isinstance(v['ue_scheduler_weights'], dict) or
               not set(v['ue_scheduler_weights']) <= POPULATION or
               any(not _finite(w) or w not in (1., 1.5, 2.) for w in v['ue_scheduler_weights'].values())
               for v in values.values()):
            errors.append('Invalid whole scheduler candidate bundle')
            continue
        if len({tuple(sorted(v['ue_scheduler_weights'])) for v in values.values()}) != 1:
            errors.append('Candidate bundles cover different served UE sets')
        conflict = len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1
        winner, draw = row.get('winner'), row.get('draw')
        valid_draw = (_finite(draw) and 0 <= draw < 1 and row.get('probability') == 1 / len(sources) and
                      winner == sources[min(int(draw * len(sources)), len(sources) - 1)]) if conflict else (
                      draw is None and row.get('probability') == 1 and winner in values)
        if winner not in values or row.get('selected_value') != values.get(winner) or not valid_draw:
            errors.append('Invalid whole-request uniform selection')
        if (type(index) is not int or type(generation) is not int or generation < 1 or
                index != row.get('slot_id', -1) // window_slots or
                row.get('window_start_slot') != index * window_slots or
                row.get('window_end_slot_exclusive') != (index + 1) * window_slots or
                row.get('mitigation_seed') != mitigation_seed):
            errors.append('Invalid independent 100 ms selection window')
        if conflict:
            conflicts.add(key)
            draw_index = row.get('draw_index')
            if set(sources) != {X08, X12} or row.get('stream_key') != 'scheduler_bundle:' + str(cid) or type(draw_index) is not int or draw_index < 1:
                errors.append('Missing exact independently seeded scheduler RNG evidence')
            elif draw_index in draws[cid] and draws[cid][draw_index] != draw:
                errors.append('Conflicting repeated scheduler RNG draw')
            else:
                draws[cid][draw_index] = draw
        signature = (winner, row.get('selected_value'), draw, candidates)
        if key in selections and selections[key]['signature'] != signature:
            errors.append('Whole scheduler choice changed inside a held epoch')
        selections[key] = dict(signature=signature, row=row)
    for cid, stream in draws.items():
        rng = random.Random(f'{mitigation_seed}:direct-conflict-mitigation:scheduler_bundle:{cid}')
        if sorted(stream) != list(range(1, len(stream) + 1)):
            errors.append('Nonconsecutive scheduler RNG draw indices')
        for index in sorted(stream):
            if stream[index] != rng.random():
                errors.append('Scheduler RNG draw differs from independent seeded replay')
    executed, rejected, executed_conflicts = 0, 0, set()
    winners, rejection_reasons = Counter(), Counter()
    executed_epochs = set()
    for receipt in jsonl(folder, 'control_receipts'):
        for record in receipt.get('mitigation_records', []):
            if record.get('field') != 'scheduler_bundle' or record.get('winner') not in (X08, X12):
                continue
            key = (record.get('cell_id'), record.get('window_index'), record.get('generation'))
            chosen = selections.get(key)
            if receipt.get('status') == 'rejected' or record.get('status') == 'rejected':
                rejected += 1
                rejection_reasons[record.get('rejection_reason') or 'unspecified'] += 1
                continue
            if receipt.get('status') != 'applied' or record.get('status') != 'applied':
                continue
            if chosen is None or (record.get('winner'), record.get('selected_value')) != chosen['signature'][:2]:
                errors.append('Executed bundle lacks its exact selection receipt')
                continue
            selected = chosen['row']
            provenance = receipt.get('provenance', [])
            if isinstance(provenance, dict):
                provenance = [provenance]
            app_sources = {p.get('implementation_id') for p in provenance} & {X08, X12}
            applied = receipt.get('apply_slot', receipt.get('slot_id'))
            if (app_sources != {record['winner']} or record.get('candidates') != selected.get('candidates') or
                    record.get('draw') != selected.get('draw') or record.get('request_id') != selected.get('request_id') or
                    type(applied) is not int or not record['window_start_slot'] <= applied < record['window_end_slot_exclusive']):
                errors.append('Executed bundle provenance, raw requests or actuation window disagree with selection')
                continue
            actual = receipt.get('after', {}).get(record['cell_id'], {})
            bundle = {f: actual.get(f) for f in FIELDS}
            if bundle != record.get('selected_value'):
                errors.append('Executed scheduler differs from selected complete bundle')
                continue
            tuning = record.get('framework_tuning', {})
            if not isinstance(tuning, dict) or any(f != 'latency_bias' or v != 1. or actual.get(f) != v for f, v in tuning.items()):
                errors.append('Framework scheduler tuning is not transparently executed')
                continue
            executed += 1
            executed_epochs.add(key)
            winners[record['winner']] += 1
            if key in conflicts:
                executed_conflicts.add(key)
    occupied = {k for k in conflicts if selections[k]['row']['selected_value']['ue_scheduler_weights']}
    ue6 = {k for k in conflicts if 'ue_6' in selections[k]['row']['selected_value']['ue_scheduler_weights']}
    return dict(valid=bool(selections and executed and not errors),
        unequal_conflict_epochs=len(conflicts), all_selection_epochs=len(selections),
        occupied_unequal_conflict_epochs=len(occupied), ue6_serving_cell_conflict_epochs=len(ue6),
        executed_selection_epochs=len(executed_epochs), executed_conflict_epochs=len(executed_conflicts),
        executed_occupied_conflict_epochs=len(executed_conflicts & occupied),
        executed_ue6_serving_cell_conflict_epochs=len(executed_conflicts & ue6),
        matching_runtime_receipts=executed, rejected_receipts=rejected,
        rejection_reasons=dict(rejection_reasons), winner_runtime_receipts=dict(winners),
        winner_conflict_epochs=dict(Counter(selections[k]['row']['winner'] for k in conflicts)),
        winner_executed_epochs=dict(Counter(selections[k]['row']['winner'] for k in executed_epochs)),
        both_contenders_executed=set(winners) == {X08, X12}, seeded_rng_replay_passed=bool(draws) and not any('RNG' in e for e in errors),
        errors=errors[:30], selection_epochs=[v['row'] for _, v in sorted(selections.items())],
        scope='Per-cell whole scheduler bundles; repeated held-state receipts are not independent draws. A finite valid random sequence need not select both sources.')


def evaluate_study(folders, x08_stamp, urllc_stamp, mitigation_seed=1701,
                   urllc_failure_max=.01, protection_improvement_pp=20., goodput_harm_fraction=.02, deadline_harm_pp=2.):
    if set(folders) != set(ARMS):
        raise ValueError('All four matched scheduler cases are required')
    folders = {a: Path(p) for a, p in folders.items()}
    audits = {a: completed_case(p) for a, p in folders.items()}
    if len({a['source_sha256'] for a in audits.values()}) != 1:
        raise ValueError('Evaluation source changed between arms')
    configs = {a: load(p / 'config_snapshot.json') for a, p in folders.items()}
    summaries = {a: load(p / 'final_summary.json') for a, p in folders.items()}
    base = configs['off']
    if any(k not in base['configs'] for k in ('ue', 'phy', 'sim', 'cell', 'beamforming')):
        raise ValueError('Complete radio/PHY/traffic configuration snapshot required')
    expected_apps = {'off': set(), 'x08': {X08}, 'urllc': {X12}, 'both_random': {X08, X12}}
    arms, evidence = {}, {}
    for arm in ARMS:
        cfg, summary = configs[arm], summaries[arm]
        if cfg['manifest'] != base['manifest'] or any(cfg['configs'].get(k) != base['configs'].get(k)
                for k in ('ue', 'phy', 'sim', 'cell', 'beamforming')):
            raise ValueError('Matched traffic/radio/mobility/PHY configuration mismatch: ' + arm)
        if {u['ue_id'] for u in cfg['manifest']['ues']} != POPULATION or len(cfg['manifest']['cells']) != 3:
            raise ValueError('Original six-UE/three-cell population required')
        ric = cfg['configs']['ric']
        apps = ric.get('xapps', [])
        if len(apps) != len(expected_apps[arm]) or {a.get('implementation_id') for a in apps} != expected_apps[arm]:
            raise ValueError('Unexpected active scheduler app set: ' + arm)
        for app in apps:
            if app['parameters'].get('mode') != 'frozen':
                raise ValueError('Every evaluation controller must be frozen')
            if app['implementation_id'] == X08 and app['parameters'].get('policy') != 'throughput':
                raise ValueError('X08 must use explicitly labelled throughput research policy')
        if arm == 'both_random' and (ric.get('direct_conflict_mitigation') != 'uniform_random' or
                ric.get('mitigation_seed') != mitigation_seed or ric.get('mitigation_window_ms') != 100):
            raise ValueError('BOTH must use explicit independently seeded 100 ms random mitigation')
        users = actual_packet_metrics(folders[arm], cfg, summary)
        arrivals = arrival_audit(folders[arm], users)
        if arrivals['trace_slots'] != cfg['manifest']['total_slots']:
            raise ValueError('Trace horizon does not cover the entire run')
        arms[arm] = dict(users=users, network_goodput_mbps=sum(u['goodput_mbps'] for u in users.values()),
            arrival_audit=arrivals, byte_conservation_verified=True)
        model_paths = {a['implementation_id']: a['parameters'].get('model_path') for a in apps}
        evidence[arm] = {app: controller_evidence(folders[arm], summary, app,
            x08_stamp if app == X08 else urllc_stamp, combined=arm == 'both_random', model_path=model_paths[app]) for app in expected_apps[arm]}
    native = arms['off']
    if native['network_goodput_mbps'] <= 0:
        raise ValueError('Positive native delivered goodput required')
    for arm in ARMS[1:]:
        if arms[arm]['arrival_audit']['exogenous_slot_inputs_sha256'] != native['arrival_audit']['exogenous_slot_inputs_sha256']:
            raise ValueError('Actual exogenous arrivals differ between matched cases')
        for uid in ('ue_2', 'ue_6'):
            if arms[arm]['users'][uid]['mature_offered_inputs_sha256'] != native['users'][uid]['mature_offered_inputs_sha256']:
                raise ValueError('URLLC mature packet-ID cohort mismatch')
    guards = {}
    for arm in ARMS[1:]:
        users = arms[arm]['users']
        delivery = {u: users[u]['delivered_bytes'] >= .95 * native['users'][u]['delivered_bytes'] - 1e-9 for u in POPULATION}
        url = {}
        for u in ('ue_2', 'ue_6'):
            b, c = native['users'][u]['mature_failure_fraction'], users[u]['mature_failure_fraction']
            delta = None if b is None or c is None else c - b
            url[u] = dict(delta_fraction=delta, passed=delta is not None and delta <= .05 + 1e-12)
        guards[arm] = dict(per_ue_delivery=delivery, original_10ms_unique_packet_failures=url,
            passed=all(delivery.values()) and all(v['passed'] for v in url.values()))
    arbitration = arbitration_evidence(folders['both_random'], mitigation_seed)
    valid = all(v['valid'] for arm in evidence.values() for v in arm.values()) and arbitration['valid']
    gain = arms['x08']['network_goodput_mbps'] / native['network_goodput_mbps'] - 1
    a, b = native['users']['ue_6']['mature_failure_fraction'], arms['urllc']['users']['ue_6']['mature_failure_fraction']
    improvement_pp = None if a is None or b is None else (a - b) * 100
    x08_failure = arms['x08']['users']['ue_6']['mature_failure_fraction']
    protection_pp = None if x08_failure is None or b is None else (x08_failure - b) * 100
    individual = dict(x08=bool(evidence['x08'][X08]['valid'] and gain >= .05 - 1e-12),
        urllc=bool(evidence['urllc'][X12]['valid'] and b is not None and b <= urllc_failure_max + 1e-12))
    useful_protection = protection_pp is not None and protection_pp >= protection_improvement_pp - 1e-12
    both_failure = arms['both_random']['users']['ue_6']['mature_failure_fraction']
    failure_harm = None if b is None or both_failure is None else (both_failure - b) * 100
    goodput_delta = arms['both_random']['network_goodput_mbps'] / arms['x08']['network_goodput_mbps'] - 1
    target_harm = dict(both_goodput_change_vs_x08_fraction=goodput_delta,
        both_ue6_failure_change_vs_urllc_pp=failure_harm,
        x08_target_harmed=goodput_delta < -goodput_harm_fraction - 1e-12,
        urllc_target_harmed=failure_harm is not None and failure_harm > deadline_harm_pp + 1e-12)
    structural = arbitration['valid'] and arbitration['executed_occupied_conflict_epochs'] > 0
    classification = ('invalid_evidence' if not valid else 'no_observed_direct_conflict' if not structural else
        'direct_conflict_observed_standalone_target_not_met' if not all(individual.values()) else
        'direct_conflict_observed_with_target_harm' if any(target_harm[k] for k in ('x08_target_harmed', 'urllc_target_harmed')) else
        'direct_conflict_observed_no_detected_target_harm')
    return dict(classification=classification, valid_evidence=bool(valid),
        structural_direct_conflict_observed=bool(structural), standalone_targets_met=individual,
        ue6_serving_cell_direct_conflict_observed=bool(arbitration['valid'] and arbitration['executed_ue6_serving_cell_conflict_epochs'] > 0),
        urllc_protection_vs_x08_pp=protection_pp, useful_urllc_protection_vs_x08=bool(useful_protection),
        intended_case_evidence_met=bool(valid and all(individual.values()) and useful_protection),
        network_seed=base['manifest']['rng_seed'], mitigation_seed=mitigation_seed,
        calibration_not_independent_validation=base['manifest']['rng_seed'] == 55,
        duration_s=base['manifest']['total_slots'] * base['manifest']['slot_duration_ms'] / 1000,
        throughput_policy='throughput', target_ue='ue_6', original_packet_deadline_ms=10,
        thresholds=dict(x08_goodput_gain_fraction=.05, urllc_mature_failure_max=urllc_failure_max,
            urllc_protection_vs_x08_min_pp=protection_improvement_pp,
            both_goodput_harm_fraction=goodput_harm_fraction, both_ue6_failure_harm_pp=deadline_harm_pp,
            diagnostic_min_delivery_ratio=.95, diagnostic_max_failure_degradation_pp=5.),
        x08_goodput_gain_vs_off_fraction=gain, urllc_ue6_failure_improvement_vs_off_pp=improvement_pp,
        target_harm=target_harm, offline_qos_guards=guards, arms=arms,
        controller_evidence=evidence, arbitration=arbitration, run_audits=audits,
        interpretation='Unequal overlapping requests prove a direct control conflict only after execution audit. Native UE6 failures may already be zero: X12 target satisfaction is preservation, not improvement over that floor. Protection versus X08 and post-arbitration target harm are separate finite-run observations. Completed-packet quantiles exclude pending/drop outcomes. No statistical significance, indirect-conflict or ML-superiority claim.')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('train', 'evaluate', 'all'), default='all')
    p.add_argument('--throughput-model', type=Path, default=ROOT / 'outputs/x08_load2x_recovery_20261105/throughput_mlp_model.json')
    p.add_argument('--urllc-model', type=Path)
    p.add_argument('--train-seconds', type=float, default=6.)
    p.add_argument('--max-train-seconds', type=float, default=10.)
    p.add_argument('--train-seed', type=int, default=179)
    p.add_argument('--eval-seed', type=int, default=55)
    p.add_argument('--eval-seconds', type=float, default=1.)
    p.add_argument('--eval-workers', type=int, choices=(1, 2, 3), default=1,
                   help='Run OFF first to warm cache, then up to three independent frozen evaluations')
    p.add_argument('--mitigation-seed', type=int, default=1701)
    p.add_argument('--urllc-failure-max', type=float, default=.01)
    p.add_argument('--protection-improvement-pp', type=float, default=20.)
    p.add_argument('--goodput-harm-fraction', type=float, default=.02)
    p.add_argument('--deadline-harm-pp', type=float, default=2.)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def command_for(args, output, x08_model, urllc_model, name, seed, seconds, arm):
    return ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
        '--duration-s', str(seconds), '--seed', str(seed), '--traffic-load', '2',
        '--xapp-pool', 'chongyu', '--ue-services', 'drx-mec', '--record-compression', 'gzip',
        '--energy-xapp-mode', 'off', '--latency-xapp-mode', 'off', '--budget-xapp-mode', 'off',
        '--throughput-xapp-mode', 'frozen' if arm in ('x08', 'both_random') else 'off',
        '--throughput-xapp-policy', 'throughput', '--throughput-xapp-model', str(x08_model),
        '--throughput-xapp-window-ms', '100',
        '--urllc-xapp-mode', 'online' if arm == 'train' else 'frozen' if arm in ('urllc', 'both_random') else 'off',
        '--urllc-xapp-model', str(urllc_model), '--urllc-xapp-window-ms', '100',
        '--direct-conflict-mitigation', 'uniform_random' if arm == 'both_random' else 'priority',
        '--mitigation-seed', str(args.mitigation_seed), '--mitigation-window-ms', '100',
        '--channel-cache-root', str(ROOT / 'channel_cache' / output.name / ('seed_' + str(seed))),
        '--output', str(output / name)]


def run_evaluation_cases(cases, runner, workers=1):
    """Warm OFF serially; await every submitted frozen case before returning."""
    if workers not in (1, 2, 3) or [c[3] for c in cases] != list(ARMS):
        raise ValueError('One ordered OFF/X08/URLLC/BOTH evaluation and 1-3 workers required')
    results = {'off': runner(cases[0])}
    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(runner, case): case[3] for case in cases[1:]}
        for future in as_completed(futures):
            arm = futures[future]
            try:
                results[arm] = future.result()
            except BaseException as error:
                failures[arm] = error
    if failures:
        arms = [arm for arm in ARMS if arm in failures]
        raise RuntimeError('Frozen evaluation failed after awaiting all submitted cases: ' + ', '.join(arms)) from failures[arms[0]]
    return {arm: results[arm] for arm in ARMS}


def main():
    args = parse_args()
    output = args.output.resolve()
    x08_model = args.throughput_model.resolve()
    urllc_model = (args.urllc_model or output / 'urllc_rf_model.json').resolve()
    if output.exists() or any(not p.is_relative_to(ROOT / 'outputs') for p in (output, x08_model, urllc_model)):
        raise ValueError('Use a fresh output and model paths under project outputs')
    if args.phase != 'evaluate' and (urllc_model.exists() or not urllc_model.is_relative_to(output)):
        raise ValueError('Online RF requires a fresh model inside the new output; historical artifacts are immutable')
    if (not all(_finite(v) and v > 0 for v in (args.train_seconds, args.max_train_seconds, args.eval_seconds)) or
            not args.train_seconds <= args.max_train_seconds <= 10 or
            not all(_finite(v) and v >= 0 for v in (args.urllc_failure_max, args.protection_improvement_pp, args.goodput_harm_fraction, args.deadline_harm_pp))):
        raise ValueError('Invalid finite budget or predeclared thresholds')
    if args.eval_seed in (args.train_seed, args.train_seed + 1) or args.mitigation_seed in (args.eval_seed, args.train_seed, args.train_seed + 1):
        raise ValueError('Training, evaluation and mitigation RNG seeds must be distinct')
    plan = []
    if args.phase != 'evaluate':
        plan.append(('online_training', args.train_seed, args.train_seconds, 'train'))
        if args.max_train_seconds > args.train_seconds:
            plan.append(('online_coverage_extension', args.train_seed + 1, args.max_train_seconds - args.train_seconds, 'train'))
    if args.phase != 'train':
        plan += [('seed' + str(args.eval_seed) + '_' + a, args.eval_seed, args.eval_seconds, a) for a in ARMS]
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, phase=args.phase, matched_traffic_load=2,
            eval_workers=args.eval_workers, evaluation_order='OFF serial warmup, then independent frozen cases',
            commands=[command_for(args, output, x08_model, urllc_model, *c) for c in plan],
            extension_condition='Only insufficient real fitted-prefix action coverage; never weak KPI performance',
            automatic_longer_or_additional_seed_runs=False), indent=2))
        return
    if not throughput_readiness(x08_model)['ready']:
        raise ValueError('Immutable X08 artifact must contain its genuinely trained verified optimizer')
    original_x08 = throughput_stamp(x08_model)
    lineage = None
    if args.phase == 'evaluate':
        if not rf_readiness(urllc_model)['ready']:
            raise ValueError('Frozen evaluation requires a genuinely trained URLLC artifact')
        lineage = trained_lineage(urllc_model, original_x08)
    output.mkdir(parents=True)
    urllc_model.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(completed=False, status='initializing', phase=args.phase,
        protocol='Matched 2x-load native six-UE OFF/X08/UE6-URLLC-RF/BOTH atomic random scheduler study',
        throughput_model=str(x08_model), throughput_model_before=original_x08,
        urllc_model=str(urllc_model), throughput_policy='throughput', matched_traffic_load=2,
        train_seed=args.train_seed, eval_seed=args.eval_seed, mitigation_seed=args.mitigation_seed,
        eval_workers=args.eval_workers, evaluation_order='OFF serial warmup, then independent frozen cases',
        calibration_not_independent_validation=args.eval_seed == 55, cases=[], training_evidence=[],
        training_simulated_s=0., automatic_longer_or_additional_seed_runs=False,
        thresholds=dict(urllc_failure_max=args.urllc_failure_max, protection_improvement_pp=args.protection_improvement_pp,
            goodput_harm_fraction=args.goodput_harm_fraction, deadline_harm_pp=args.deadline_harm_pp))
    if lineage:
        receipt['training_lineage'] = lineage
    lock = threading.RLock()
    running = set()
    order = {case[0]: i for i, case in enumerate(plan)}
    frozen_stamp = None

    def write():
        with lock:
            atomic_json(output / 'acceptance.json', receipt)

    def protect_x08():
        if sha256(x08_model) != original_x08['sha256']:
            raise ValueError('Immutable historical X08 artifact changed')
        if lineage:
            protect_lineage(lineage)
        if frozen_stamp and sha256(urllc_model) != frozen_stamp['sha256']:
            raise ValueError('Frozen evaluation changed the measured RF model')

    def run(case):
        name, seed, seconds, arm = case
        with lock:
            running.add(name)
            active = sorted(running, key=order.get)
            receipt.update(status='awaiting_remaining_cases_after_failure' if receipt.get('case_failures') else 'running',
                           running=active[0], running_cases=active)
            write()
        try:
            protect_x08()
            command = command_for(args, output, x08_model, urllc_model, *case)
            began = time.perf_counter()
            with (output / (name + '.log')).open('w') as log:
                child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode:
                raise RuntimeError('Simulation failed: ' + name)
            folder = output / name
            audit = completed_case(folder)
            performance = load(folder / 'performance.json')
            protect_x08()
            with lock:
                receipt['cases'].append(dict(name=name, arm=arm, seed=seed, simulated_s=seconds,
                    command=command, output=str(folder), audit=audit,
                    simulation_wall_s=performance['simulation_s'], external_wall_s=time.perf_counter() - began))
                receipt['cases'].sort(key=lambda c: order[c['name']])
                if len({c['audit']['source_sha256'] for c in receipt['cases']}) != 1:
                    raise ValueError('Source changed during the study')
                if lineage and arm == 'off':
                    lineage['compatibility_with_evaluation'] = collection_eval_compatibility(lineage, load(folder / 'config_snapshot.json'))
                    lineage['evaluation_source_sha256'] = audit['source_sha256']
                    lineage['collector_and_evaluation_sources_differ'] = audit['source_sha256'] not in lineage['collector_source_sha256s']
                write()
            return folder
        except BaseException as error:
            with lock:
                receipt.setdefault('case_failures', []).append(dict(name=name, type=type(error).__name__, message=str(error)))
                receipt['case_failures'].sort(key=lambda c: order[c['name']])
                receipt.update(completed=False, status='awaiting_remaining_cases_after_failure')
                write()
            raise
        finally:
            with lock:
                running.discard(name)
                receipt['running_cases'] = sorted(running, key=order.get)
                if receipt['running_cases']:
                    receipt['running'] = receipt['running_cases'][0]
                else:
                    receipt.pop('running', None)
                write()

    write()
    try:
        if args.phase != 'evaluate':
            previous = None
            for case in [c for c in plan if c[3] == 'train']:
                folder = run(case)
                receipt['training_simulated_s'] += case[2]
                receipt['training_evidence'].append(training_evidence(folder, urllc_model, previous))
                receipt['training_readiness'] = rf_readiness(urllc_model)
                if receipt['training_readiness']['ready']:
                    break
                previous = load(urllc_model)
            if not receipt['training_readiness']['ready']:
                receipt.update(completed=True, status='not_trained', learned_model_claim=False)
                receipt.pop('running', None)
                write()
                return
        stamp = rf_stamp(urllc_model)
        receipt.update(urllc_model_before_evaluation=stamp, learned_model_claim=True)
        if args.phase == 'train':
            receipt.update(completed=True, status='trained_no_evaluation')
        else:
            frozen_stamp = stamp
            folders = run_evaluation_cases([c for c in plan if c[3] != 'train'], run, args.eval_workers)
            evaluation = evaluate_study(folders, original_x08, stamp, args.mitigation_seed,
                args.urllc_failure_max, args.protection_improvement_pp, args.goodput_harm_fraction, args.deadline_harm_pp)
            receipt.update(completed=True, status=evaluation['classification'], evaluation=evaluation)
            atomic_json(output / 'comparison-data.json', evaluation)
        receipt.pop('running', None)
        protect_x08()
        write()
    except BaseException as error:
        receipt.update(completed=False, status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        write()
        raise
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
