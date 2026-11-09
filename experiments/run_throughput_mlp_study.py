"""Finite native-network X08 study: RR, neutral family, fixed and learned MLP."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_latency_energy_study import jsonl
from experiments.run_ue_energy_study import finite, load, POPULATION

APP = 'throughput_scheduler_mlp'
ARMS = ('off', 'neutral', 'fixed', 'mlp')
URLLC = ('ue_2', 'ue_6')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, q):
    """Linear sample quantile of completed-packet observations only."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def actual_packet_metrics(folder, config, summary):
    profiles = {u['ue_id']: u for u in config['configs']['ue']['ues']}
    if set(profiles) != POPULATION or set(summary['packet_metrics']) != POPULATION:
        raise ValueError('Native six-UE packet population required')
    final_slot = config['manifest']['total_slots']
    slot_ms = config['manifest']['slot_duration_ms']
    duration = final_slot * slot_ms / 1000
    users = {uid: dict(arrivals={}, completed={}, deadline_miss=set(), delivered_bytes=0)
             for uid in profiles}
    for event in jsonl(folder, 'packet_events'):
        uid = event['ue_id']
        state = users[uid]
        pid = event.get('packet_id')
        if event['kind'] in ('arrival', 'initial'):
            if pid in state['arrivals']:
                raise ValueError('Repeated packet arrival identifier')
            state['arrivals'][pid] = event
        elif event['kind'] == 'delivery':
            state['delivered_bytes'] += event['bytes']
            if event['completed']:
                if pid in state['completed']:
                    raise ValueError('Repeated completed packet identifier')
                state['completed'][pid] = finite(event['latency_ms'], 'packet completion latency')
        elif event['kind'] == 'deadline_miss':
            state['deadline_miss'].add(pid)
    result = {}
    for uid, state in users.items():
        packet = summary['packet_metrics'][uid]
        admitted = {pid for pid, e in state['arrivals'].items() if e['admitted']}
        overflow = set(state['arrivals']) - admitted
        completed = set(state['completed'])
        pending = admitted - completed
        budget = finite(profiles[uid]['latency_budget_ms'], 'original packet deadline')
        if uid in URLLC and budget != 10:
            raise ValueError('Original URLLC 10 ms packet budget must be preserved')
        if not completed <= admitted or not state['deadline_miss'] <= admitted:
            raise ValueError('Packet events reference unadmitted packets')
        overdue_pending = {pid for pid in pending if
            (final_slot - state['arrivals'][pid]['slot_id']) * slot_ms > budget}
        late_completed = {pid for pid, latency in state['completed'].items() if latency > budget}
        deadline_failed = state['deadline_miss'] | overdue_pending | late_completed
        failed_or_overflow = deadline_failed | overflow
        mature = {pid for pid, e in state['arrivals'].items() if
                  (final_slot - e['slot_id']) * slot_ms > budget}
        mature_failed = mature & failed_or_overflow
        mature_inputs = sorted((pid, state['arrivals'][pid]['slot_id'], state['arrivals'][pid]['bytes']) for pid in mature)
        latencies = list(state['completed'].values())
        offered_bytes = sum(e['bytes'] for e in state['arrivals'].values())
        if (len(admitted) != packet['admitted_packets'] or len(completed) != packet['completed_packets'] or
                len(overflow) != packet['overflow_packets'] or len(pending) != packet['pending_packets'] or
                len(state['deadline_miss']) != packet['deadline_missed_packets'] or
                state['delivered_bytes'] != packet['delivered_bytes'] or
                offered_bytes != packet['initial_bytes'] + packet['offered_bytes'] or
                not math.isclose(sum(latencies), packet['completion_latency_ms_sum'], abs_tol=1e-7)):
            raise ValueError('Packet event/final-counter audit failed: ' + uid)
        if offered_bytes != packet['delivered_bytes'] + packet['pending_bytes'] + packet['overflow_bytes']:
            raise ValueError('Packet byte conservation failed: ' + uid)
        result[uid] = dict(delivered_bytes=packet['delivered_bytes'], goodput_mbps=packet['delivered_bytes'] * 8 / duration / 1e6,
            initial_bytes=packet['initial_bytes'], offered_bytes=packet['offered_bytes'],
            pending_bytes=packet['pending_bytes'], overflow_bytes=packet['overflow_bytes'],
            admitted_packets=len(admitted), offered_packets=len(state['arrivals']),
            completed_packets=len(completed), pending_packets=len(pending), overflow_packets=len(overflow),
            overdue_pending_packets=len(overdue_pending), pending_within_deadline_packets=len(pending - overdue_pending),
            late_completed_packets=len(late_completed), logged_deadline_misses=len(state['deadline_miss']),
            unique_deadline_failed_packets=len(deadline_failed), deadline_or_overflow_failed_packets=len(failed_or_overflow),
            deadline_failure_fraction_among_admitted=len(deadline_failed) / len(admitted) if admitted else None,
            deadline_or_overflow_failure_fraction=len(failed_or_overflow) / len(state['arrivals']) if state['arrivals'] else None,
            mature_offered_packets=len(mature), mature_failed_packets=len(mature_failed),
            mature_failure_fraction=len(mature_failed) / len(mature) if mature else None,
            censored_offered_packets=len(state['arrivals']) - len(mature),
            mature_offered_inputs_sha256=hashlib.sha256(json.dumps(mature_inputs, separators=(',', ':')).encode()).hexdigest(),
            original_packet_deadline_ms=budget,
            completed_packet_mean_latency_ms=sum(latencies) / len(latencies) if latencies else None,
            completed_packet_p50_latency_ms=quantile(latencies, .5), completed_packet_p95_latency_ms=quantile(latencies, .95))
    return result


def arrival_audit(folder, users):
    fingerprint = hashlib.sha256()
    external, returned = Counter(), Counter()
    slots = 0
    for row in jsonl(folder, 'rt_sys_trace'):
        if row['slot_id'] != slots + 1:
            raise ValueError('Exogenous-demand audit requires every ordered trace slot')
        slots += 1
        values = row.get('traffic_arrivals', {})
        if set(values) != POPULATION:
            raise ValueError('Missing recorded exogenous/MEC arrival components')
        for uid in sorted(POPULATION):
            exog, mec = values[uid]['exogenous_bytes'], values[uid]['mec_result_bytes']
            if type(exog) is not int or type(mec) is not int or min(exog, mec) < 0:
                raise ValueError('Invalid actual traffic component')
            external[uid] += exog
            returned[uid] += mec
        fingerprint.update(json.dumps([row['slot_id']] + [values[u]['exogenous_bytes'] for u in sorted(POPULATION)], separators=(',', ':')).encode())
    for uid in POPULATION:
        if external[uid] + returned[uid] != users[uid]['offered_bytes']:
            raise ValueError('Recorded arrival components disagree with offered packet bytes: ' + uid)
    return dict(exogenous_slot_inputs_sha256=fingerprint.hexdigest(), trace_slots=slots,
                exogenous_bytes=dict(external), endogenous_mec_result_bytes=dict(returned),
                scope='Actual per-slot generator bytes and actual MEC-result releases are separate; no universal physical capacity bound is inferred')


def model_stamp(path):
    p = load(path)
    optimizer = p.get('optimizer_metadata') or {'gradient_convention': 'legacy_unrecorded'}
    return dict(sha256=sha256(path), format=p['format'], version=p['version'],
                samples=len(p['x']), trained_samples=p['trained_samples'], fit_count=p['fit_count'],
                prediction_batches=p.get('prediction_batches', 0), fit_parameters=p.get('fit_parameters'),
                feature_names=p['feature_names'], fit_diagnostics=p.get('fit_diagnostics'),
                optimizer_metadata=optimizer, optimizer_verified=optimizer_metadata_verified(optimizer))


def optimizer_metadata_verified(metadata):
    if not isinstance(metadata, dict):
        return False
    convention = metadata.get('gradient_convention')
    if metadata.get('objective') != 'half_mean_standardized_squared_error_plus_L2':
        return False
    if convention not in ('native_mean_output_consistent', 'project_corrected_mean_output'):
        return False
    tolerance = metadata.get('relative_tolerance')
    error = metadata.get('consistency_max_relative_error')
    if (type(tolerance) not in (int, float) or tolerance != 1e-5 or
            type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= tolerance):
        return False
    digest = metadata.get('source_sha256')
    if (not isinstance(metadata.get('sklearn_version'), str) or not metadata['sklearn_version'] or
            not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        return False
    checks = metadata.get('checks', [])
    required = {(1, 0), (1, 1), (9, 0), (9, 1)}
    seen = set()
    chosen_errors = []
    key = 'corrected_relative_error' if convention == 'project_corrected_mean_output' else 'native_relative_error'
    for check in checks:
        chosen = check.get(key)
        if type(chosen) not in (int, float) or not math.isfinite(chosen) or not 0 <= chosen <= tolerance:
            return False
        seen.add((check.get('outputs'), check.get('alpha')))
        chosen_errors.append(chosen)
    return seen == required and len(checks) == 4 and abs(max(chosen_errors) - error) <= 1e-15


def measured_data_sha256(payload):
    value = {k: payload[k] for k in ('seed', 'feature_names', 'target_names', 'x', 'y')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def assert_collection_unchanged(lineage):
    for path, expected in lineage['protected_original_inputs'].items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError('Original measured collection/artifact changed: ' + path)


def _normalized_collection_inputs(value):
    allowed = {'rng_seed', 'total_slots', 'cache_root', 'cache_dir', 'cache_path', 'channel_cache_root', 'cache_key'}
    if isinstance(value, dict):
        return {k: _normalized_collection_inputs(v) for k, v in value.items() if k not in allowed}
    if isinstance(value, list):
        return [_normalized_collection_inputs(v) for v in value]
    return value


def _trajectory_position(trajectory, t):
    times, positions = trajectory['times_s'], trajectory['positions_m']
    if len(times) != len(positions) or not times or t < times[0] - 1e-9 or t > times[-1] + 1e-9:
        raise ValueError('Invalid shared trajectory horizon')
    if t >= times[-1]:
        return positions[-1]
    index = next(i for i in range(len(times) - 1) if times[i] <= t <= times[i + 1])
    span = times[index + 1] - times[index]
    if span <= 0:
        raise ValueError('Nonmonotonic collection trajectory')
    fraction = (t - times[index]) / span
    return [a + (b - a) * fraction for a, b in zip(positions[index], positions[index + 1])]


def _constant_trajectory_speed(trajectory):
    speeds = []
    for start, end, a, b in zip(trajectory['times_s'], trajectory['times_s'][1:],
                                trajectory['positions_m'], trajectory['positions_m'][1:]):
        if end <= start:
            raise ValueError('Nonmonotonic collection trajectory')
        speeds.append(math.sqrt(sum((y - x) ** 2 for x, y in zip(a, b))) / (end - start))
    if not speeds or any(not math.isclose(v, speeds[0], rel_tol=0, abs_tol=1e-8) for v in speeds):
        return None
    return speeds[0]


def collection_eval_compatibility(lineage, evaluation_config):
    collector = load(Path(lineage['collection_folder']) / 'config_snapshot.json')
    for section in ('ue', 'cell', 'beamforming'):
        if collector['configs'][section] != evaluation_config['configs'][section]:
            raise ValueError('Measured collector/evaluation business or radio mismatch: ' + section)
    for section in ('sim', 'phy'):
        if _normalized_collection_inputs(collector['configs'][section]) != _normalized_collection_inputs(evaluation_config['configs'][section]):
            raise ValueError('Measured collector/evaluation input mismatch beyond seed/duration/cache: ' + section)
    a, b = collector['manifest'], evaluation_config['manifest']
    if _normalized_collection_inputs({k: v for k, v in a.items() if k != 'ues'}) != _normalized_collection_inputs({k: v for k, v in b.items() if k != 'ues'}):
        raise ValueError('Measured collector/evaluation deployment or radio manifest mismatch')
    users_a, users_b = ({u['ue_id']: u for u in manifest['ues']} for manifest in (a, b))
    if set(users_a) != set(users_b):
        raise ValueError('Measured collector/evaluation UE population differs')
    same_realization = True
    speeds_verified = True
    for uid, user in users_a.items():
        other = users_b[uid]
        if {k: v for k, v in user.items() if k != 'trajectory'} != {k: v for k, v in other.items() if k != 'trajectory'}:
            raise ValueError('Measured collector/evaluation UE deployment differs')
        ta, tb = user['trajectory'], other['trajectory']
        speed_a, speed_b = _constant_trajectory_speed(ta), _constant_trajectory_speed(tb)
        if speed_a is not None and speed_b is not None:
            if not math.isclose(speed_a, speed_b, rel_tol=0, abs_tol=1e-8):
                raise ValueError('Measured collector/evaluation constant-speed mobility model differs')
        else:
            speeds_verified = False
        end = min(ta['times_s'][-1], tb['times_s'][-1])
        times = {0., end} | {t for t in ta['times_s'] + tb['times_s'] if 0 <= t <= end}
        for t in times:
            pa, pb = _trajectory_position(ta, t), _trajectory_position(tb, t)
            if len(pa) != len(pb) or any(not math.isclose(x, y, rel_tol=0, abs_tol=1e-8) for x, y in zip(pa, pb)):
                same_realization = False
    return dict(verified=True, exact_sections=['ue', 'cell', 'beamforming'],
                seed_duration_and_cache_identifiers_only=True,
                constant_speed_mobility_parameters_verified=speeds_verified,
                shared_trajectory_realization_equal_diagnostic=same_realization,
                scope='Configured deployment/route/business/radio inputs match; training seed and duration may differ. Constant-speed model parameters are checked, while realized trajectory equality is diagnostic, not a seed-invariance requirement.')


def throughput_schema():
    from ric.near_rt.pools.chongyu.throughput_scheduler_mlp import FEATURES, TARGETS
    return list(FEATURES), list(TARGETS)


def new_measured_learner(features, targets, seed):
    from ric.near_rt.pools.chongyu.online_throughput_mlp import OnlineThroughputMLP
    return OnlineThroughputMLP(features, targets, seed)


def current_source_sha256():
    from phy.rt_sys.contracts import source_fingerprint
    return source_fingerprint(ROOT)


def audit_measured_collection(folder, old_model, traffic_load):
    folder, old_model = Path(folder), Path(old_model)
    status_path, config_path = folder / 'run_status.json', folder / 'config_snapshot.json'
    status, config = load(status_path), load(config_path)
    if (not status.get('completed') or status.get('stage') not in (None, 'completed') or
            not status.get('replay', {}).get('audits_passed')):
        raise ValueError('Measured collector must be completed with successful HTML audits')
    html = Path(status['replay'].get('html', str(folder / 'network-replay.html')))
    if not html.is_absolute():
        html = folder / html
    if not html.is_file():
        raise ValueError('Measured collector audited HTML artifact is missing')
    if config.get('source_sha256') != status.get('source_sha256'):
        raise ValueError('Collector source provenance disagrees between status and snapshot')
    apps = config['configs']['ric'].get('xapps', [])
    if (len(apps) != 1 or apps[0].get('implementation_id') != APP or
            apps[0].get('parameters', {}).get('mode') != 'online'):
        raise ValueError('Measured collector must be an isolated genuine online X08 run')
    configured_load = config['manifest'].get('traffic_load')
    parent_receipt_path = folder.parent / 'acceptance.json'
    parent_receipt = load(parent_receipt_path) if parent_receipt_path.is_file() else None
    if parent_receipt and Path(parent_receipt.get('model', '')).resolve() == old_model.resolve():
        configured_load = parent_receipt.get('traffic_load', configured_load)
    # Match the actual manifest/config later in evaluate_study as well; some
    # existing manifests record only the concrete scaled profiles, not a knob.
    if configured_load is not None and configured_load != traffic_load:
        raise ValueError('Measured collector and requested evaluation traffic load differ')
    paths = [folder / ('throughput_mlp_events.jsonl' + suffix) for suffix in ('', '.gz')]
    logs = [p for p in paths if p.is_file()]
    if len(logs) != 1 or not old_model.is_file():
        raise ValueError('One recorded event stream and the original measured model are required')
    summary_path = folder / 'final_summary.json'
    if not summary_path.is_file():
        raise ValueError('Completed collector final summary is missing')
    protected = {str(p.resolve()): sha256(p) for p in (old_model, status_path, config_path, summary_path, html, logs[0])}
    payload = load(old_model)
    proof = training_evidence(folder, old_model)
    features, targets = throughput_schema()
    if (payload.get('format') != 'chongyu.online-throughput-mlp' or
            payload.get('feature_names') != features or payload.get('target_names') != targets):
        raise ValueError('Measured rows do not match the unchanged X08 feature/target schema')
    counts = action_counts(payload['x'], features)
    if not 24 <= len(payload['x']) <= 40 or any(counts.get(str(i), 0) < 6 for i in range(4)):
        raise ValueError('Measured collection requires at least 24 actual rows and six per action')
    lineage = dict(collection_folder=str(folder.resolve()), original_model=str(old_model.resolve()),
        collection_source_sha256=status['source_sha256'], original_artifact_sha256=sha256(old_model),
        original_artifact_version=payload['version'], original_optimizer_metadata=payload.get('optimizer_metadata'),
        original_optimizer_status=(payload.get('optimizer_metadata') or {}).get('gradient_convention', 'legacy_unrecorded'),
        measured_data_sha256=measured_data_sha256(payload), measured_rows=len(payload['x']),
        action_samples=counts, original_model_seed=payload['seed'],
        collection_traffic_load=configured_load,
        original_profiles_sha256=hashlib.sha256(json.dumps(config['configs'].get('ue'), sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        original_collection_simulated_s=config['manifest']['total_slots'] * config['manifest']['slot_duration_ms'] / 1000,
        actual_collection_evidence=proof, protected_original_inputs=protected,
        original_parent_receipt=dict(path=str(parent_receipt_path.resolve()), sha256=sha256(parent_receipt_path)) if parent_receipt else None,
        collector_was_not_current_source_run=True,
        reason='One fixed-hyperparameter refit after a verified multioutput loss/gradient consistency correction; no new physics or synthetic rows')
    assert_collection_unchanged(lineage)
    return lineage, payload


def refit_measured_collection(folder, old_model, new_model, traffic_load):
    if Path(new_model).exists() or Path(new_model).resolve() == Path(old_model).resolve():
        raise ValueError('Corrected fit requires a fresh artifact, never overwriting the original')
    lineage, payload = audit_measured_collection(folder, old_model, traffic_load)
    fitting_source = current_source_sha256()
    started = time.perf_counter()
    learner = new_measured_learner(payload['feature_names'], payload['target_names'], payload['seed'])
    for features, targets in zip(payload['x'], payload['y']):
        learner.add(features, targets)
    if (not learner.fit() or learner.fit_count != 1 or learner.trained_samples != len(payload['x']) or
            not getattr(learner, 'optimizer_verified', False)):
        raise ValueError('One actual coherent-gradient fit on all measured rows is required')
    learner.save(new_model)
    fitted = load(new_model)
    if measured_data_sha256(fitted) != lineage['measured_data_sha256']:
        raise ValueError('Corrected fitting changed the original actual measured data')
    stamp = model_stamp(new_model)
    if not stamp['optimizer_verified']:
        raise ValueError('Corrected artifact lacks verified coherent-gradient metadata')
    assert_collection_unchanged(lineage)
    lineage.update(fitting_source_sha256=fitting_source, fitting_wall_s=time.perf_counter() - started,
                   new_artifact=stamp, corrected_fit_count=1, new_physics_training_s=0.)
    return lineage


def prepare_measured_recovery(episodes, old_model, new_model, traffic_load):
    """Retain all 40 original rows and audit both completed source episodes."""
    if len(episodes) != 2 or not Path(old_model).is_file() or Path(new_model).exists():
        raise ValueError('Recovery requires two completed episodes, an immutable model and a fresh model copy')
    original_hash = sha256(old_model)
    payload = load(old_model)
    features, targets = throughput_schema()
    if (payload.get('format') != 'chongyu.online-throughput-mlp' or
            payload.get('feature_names') != features or payload.get('target_names') != targets or
            len(payload['x']) != 40 or len(payload['y']) != 40 or payload['trained_samples'] or payload['fit_count']):
        raise ValueError('Technical recovery must retain the exact original 40 real untrained rows')
    protected = {str(Path(old_model).resolve()): original_hash}
    proofs, previous_rows, total_physics = [], 0, 0.
    for folder in map(Path, episodes):
        status_path, config_path, summary_path = (folder / n for n in
            ('run_status.json', 'config_snapshot.json', 'final_summary.json'))
        status, config, summary = load(status_path), load(config_path), load(summary_path)
        if not status.get('completed') or not status.get('replay', {}).get('audits_passed'):
            raise ValueError('Every original recovery episode must be completed and HTML-audited')
        if config.get('source_sha256') != status.get('source_sha256'):
            raise ValueError('Recovery episode source fingerprints disagree')
        apps = config['configs']['ric'].get('xapps', [])
        if len(apps) != 1 or apps[0].get('implementation_id') != APP or apps[0].get('parameters', {}).get('mode') != 'online':
            raise ValueError('Recovery history must consist of isolated actual online X08 episodes')
        learning = summary['throughput_xapp']
        if learning.get('maximum_measured_rows') != 40:
            raise ValueError('Original episodes must declare their actual historical row cap 40')
        after = learning['samples']
        if type(after) is not int or not previous_rows <= after <= 40:
            raise ValueError('Nonmonotonic actual recovery dataset history')
        html = Path(status['replay'].get('html', str(folder / 'network-replay.html')))
        if not html.is_absolute():html = folder / html
        logs = [folder / ('throughput_mlp_events.jsonl' + s) for s in ('', '.gz')]
        logs = [p for p in logs if p.is_file()]
        if not html.is_file() or len(logs) != 1:
            raise ValueError('Original recovery episode HTML/events are missing')
        for p in (status_path, config_path, summary_path, html, logs[0]):
            protected[str(p.resolve())] = sha256(p)
        previous = dict(x=payload['x'][:previous_rows], y=payload['y'][:previous_rows])
        recorded_prefix = dict(x=payload['x'][:after], y=payload['y'][:after])
        evidence = training_evidence(folder, old_model, previous, row_cap=40, persisted_payload=recorded_prefix)
        if after - previous_rows != evidence['new_actual_rows']:
            raise ValueError('Original episode sample counts disagree with its measured event history')
        physics = config['manifest']['total_slots'] * config['manifest']['slot_duration_ms'] / 1000
        total_physics += physics
        proofs.append(dict(folder=str(folder.resolve()), source_sha256=status['source_sha256'],
            actual_recorded_row_cap=40, before_samples=previous_rows, after_samples=after,
            simulated_s=physics, actual_collection_evidence=evidence))
        parent_path = folder.parent / 'acceptance.json'
        if parent_path.is_file():
            parent = load(parent_path)
            if Path(parent.get('model', '')).resolve() == Path(old_model).resolve() and parent.get('traffic_load', traffic_load) != traffic_load:
                raise ValueError('Recovery may not silently change the original traffic multiplier')
        if proofs[:-1]:
            collection_eval_compatibility({'collection_folder': proofs[0]['folder']}, config)
        previous_rows = after
    if previous_rows != 40 or total_physics > 6 + 1e-9:
        raise ValueError('Recovery history must end at all 40 actual rows within the original six-second budget')
    lineage = dict(collection_folder=proofs[0]['folder'], episodes=proofs,
        original_model=str(Path(old_model).resolve()), original_artifact_sha256=original_hash,
        original_measured_rows=40, original_action_samples=action_counts(payload['x'], features),
        original_model_seed=payload['seed'], measured_data_sha256=measured_data_sha256(payload),
        original_training_simulated_s=total_physics, protected_original_inputs=protected,
        original_recorded_row_cap=40, recovery_recorded_row_cap=48,
        reason='Technical online exploration bootstrap recovery before any high-load performance evaluation; no neural hyperparameter/PHY/QoS change',
        frozen_policy_scope='Post-coverage/frozen selection remains unchanged; historical native results retain their original source provenance')
    assert_collection_unchanged(lineage)
    Path(new_model).parent.mkdir(parents=True, exist_ok=True)
    Path(new_model).write_bytes(Path(old_model).read_bytes())
    if sha256(new_model) != original_hash:
        raise ValueError('Recovery model copy changed the original bytes')
    assert_collection_unchanged(lineage)
    return lineage


def action_counts(rows, feature_names):
    indices = [feature_names.index('action_' + str(i)) for i in range(4)]
    result = Counter()
    for row in rows:
        onehot = [row[i] for i in indices]
        if any(v not in (0, 1) for v in onehot) or sum(onehot) != 1:
            raise ValueError('Invalid recorded action one-hot encoding')
        result[str(onehot.index(1))] += 1
    return dict(result)


def readiness(path):
    if not Path(path).is_file():
        return dict(ready=False, reason='No measured artifact exists', samples=0)
    model = load(path)
    stamp = model_stamp(path)
    if stamp['samples'] > 48:
        raise ValueError('MLP measured training row cap exceeded')
    # Training action identity is stored in the artifact and cross-checked with
    # the actual event stream; only the fitted prefix can authorize deployment.
    if model['format'] != 'chongyu.online-throughput-mlp':
        raise ValueError('Expected genuine throughput MLP artifact')
    all_counts = action_counts(model['x'], model['feature_names'])
    fitted_counts = action_counts(model['x'][:stamp['trained_samples']], model['feature_names'])
    covered = all(fitted_counts.get(str(i), 0) >= 6 for i in range(4))
    ready = (stamp['trained_samples'] >= 24 and stamp['fit_count'] > 0 and covered and
             bool(model['coefs']) and bool(model['x_scaler']) and bool(model['y_scaler']) and stamp['optimizer_verified'])
    return dict(ready=bool(ready), action_samples=dict(all_counts), trained_action_samples=dict(fitted_counts),
                reason='At least 24 real fitted rows with six per action' if ready else
                       'Insufficient measured fitted action coverage; no frozen deployment claim', **stamp)


def training_evidence(folder, model_path, previous=None, *, row_cap=None, persisted_payload=None):
    events = list(jsonl(folder, 'throughput_mlp_events'))
    samples = [r for r in events if r.get('kind') == 'measured_sample']
    confirmations = [r for r in events if r.get('kind') == 'executed_policy_confirmed']
    config = load(Path(folder) / 'config_snapshot.json')
    if row_cap is None:
        summary_path = Path(folder) / 'final_summary.json'
        summary = load(summary_path) if summary_path.is_file() else {}
        row_cap = summary.get('throughput_xapp', {}).get('maximum_measured_rows', 40)
    if row_cap not in (40, 48):
        raise ValueError('Unexpected recorded MLP row cap')
    slot_s = config['manifest']['slot_duration_ms'] / 1000
    ids = sorted(POPULATION)
    last_end = -1
    for row in samples:
        start, end = row['produced_start_slot'], row['produced_end_slot']
        duration = finite(row['duration_s'], 'measured window duration')
        if (start < last_end or start >= end or end > config['manifest']['total_slots'] or
                duration < .1 - 1e-12 or not math.isclose(duration, (end - start) * slot_s, abs_tol=1e-12)):
            raise ValueError('Invalid or overlapping measured throughput windows')
        last_end = end
        prior = [r for r in confirmations if r['produced_slot'] <= start]
        confirmed = max(prior, key=lambda r: r['produced_slot']) if prior else None
        if (not confirmed or confirmed['action'] != row['action'] or
                confirmed['executed_snapshot'] != row['snapshot'] or
                (start - confirmed['produced_slot']) * slot_s < .02 - 1e-12):
            raise ValueError('Throughput sample lacks real policy confirmation and 20 ms wash-in')
        snapshot_uids = {u for c in row['snapshot'].values() for u in c['ue_scheduler_weights']}
        if snapshot_uids != POPULATION or set(row['membership']) != POPULATION:
            raise ValueError('Throughput training must observe the actual whole six-UE network')
        for cid, values in row['snapshot'].items():
            fields = ['scheduler_mode']
            if values['ue_scheduler_weights']:
                fields.append('ue_scheduler_weights')
            if any(confirmed['owners'].get(cid, {}).get(field) != APP for field in fields):
                raise ValueError('Throughput sample lacks actual executed field ownership')
        delivered, offered = row['delivered_bits_by_ue'], row['offered_bits_by_ue']
        if set(delivered) != POPULATION or set(offered) != POPULATION:
            raise ValueError('Missing actual UE delivery/offered counters')
        if any(finite(v, 'measured bit delta') < 0 for values in (delivered, offered) for v in values.values()):
            raise ValueError('Negative measured bit delta')
        rates = []
        for uid in URLLC:
            counters = [row[name][uid] for name in ('deadline_misses_by_ue', 'admitted_packets_by_ue',
                'overflow_packets_by_ue', 'completed_packets_by_ue', 'late_completed_packets_by_ue')]
            if any(type(v) is not int or v < 0 for v in counters):
                raise ValueError('Invalid real packet-counter delta')
            rates.append((counters[0] + counters[2] + counters[4]) / duration)
        goodputs = [delivered[uid] / duration / 1e6 for uid in ids]
        expected = [sum(goodputs)] + goodputs + rates
        if len(row['targets']) != 9 or any(not math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-12) for a, b in zip(row['targets'], expected)):
            raise ValueError('MLP labels do not equal actual delivered Mbps and conservative packet event rates')
    previous = previous or dict(x=[], y=[])
    previous_rows = len(previous['x'])
    if previous_rows > row_cap:
        raise ValueError('Previous MLP dataset exceeds the fixed row cap')
    added_samples = samples[:row_cap - previous_rows]
    if persisted_payload is not None or Path(model_path).is_file():
        model = persisted_payload if persisted_payload is not None else load(model_path)
        n = previous_rows
        if model['x'][:n] != previous['x'] or model['y'][:n] != previous['y']:
            raise ValueError('Continuation altered existing actual MLP data')
        if model['x'][n:] != [r['features'] for r in added_samples] or model['y'][n:] != [r['targets'] for r in added_samples]:
            raise ValueError('Saved MLP rows differ from new actual online event rows')
    elif samples:
        raise ValueError('Actual online MLP rows were not persisted')
    return dict(new_actual_rows=len(added_samples), observed_measured_windows=len(samples),
                measured_windows_not_added_due_to_row_cap=len(samples) - len(added_samples),
                fixed_training_row_cap=row_cap,
                actual_confirmation_washin_and_label_arithmetic_verified=True,
                event_counts=dict(Counter(r.get('kind') for r in events)),
                not_evaluable_reasons=[r.get('reason') for r in events if r.get('kind') == 'not_evaluable'],
                label_scope='First seven outputs are actual Mbps; last two are conservative failure-event sums/s, not unique packet failure probabilities')


def controller_evidence(folder, summary, stamp, mode):
    learning = summary.get('throughput_xapp', {})
    descriptor = next((a for a in summary.get('xapp_library', {}).get('instances', [])
                       if a.get('implementation_id') == APP), {})
    events = list(jsonl(folder, 'throughput_mlp_events'))
    confirmations = [r for r in events if r.get('kind') == 'executed_policy_confirmed']
    effects, rejections, verified = 0, 0, set()
    modes, weights = Counter(), {uid: set() for uid in POPULATION}
    for receipt in jsonl(folder, 'control_receipts'):
        provenance = receipt.get('provenance', [])
        if isinstance(provenance, dict):
            provenance = [provenance]
        if not any(p.get('implementation_id') == APP for p in provenance):
            continue
        if receipt.get('status') == 'rejected':
            rejections += 1
        if receipt.get('status') != 'applied':
            continue
        requested_weights = {}
        requested_mode = None
        for f in receipt.get('fields', []):
            if f.get('field') == 'ue_scheduler_weights':
                requested_weights.update(f['value'])
            if f.get('field') == 'scheduler_mode':
                requested_mode = f['value']
        actual = {}
        actual_modes = []
        for cell in receipt.get('after', {}).values():
            actual.update(cell.get('ue_scheduler_weights', {}))
            actual_modes.append(cell.get('scheduler_mode'))
        matches = (bool(requested_weights) and requested_mode in ('round_robin', 'latency_aware') and
            all(m == requested_mode for m in actual_modes) and
            all(uid in actual and type(weight) in (int, float) and weight in (1., 1.5, 2.) and
                math.isclose(actual[uid], weight, abs_tol=1e-12, rel_tol=0)
                for uid, weight in requested_weights.items()))
        if matches:
            effects += 1
            verified.update(requested_weights)
            modes.update(actual_modes)
            for uid, weight in requested_weights.items():
                weights[uid].add(weight)
    frozen = (mode != 'frozen' or stamp is not None and
              all(learning.get(k) == stamp[k] for k in ('samples', 'trained_samples', 'fit_count')) and
              learning.get('fit_parameters') == stamp['fit_parameters'] and
              learning.get('model_sha256') == stamp['sha256'] and stamp.get('optimizer_verified') is True and
              learning.get('optimizer_verified') is True and
              learning.get('optimizer_metadata') == stamp.get('optimizer_metadata'))
    prediction_delta = learning.get('prediction_batches', 0) - (stamp or {}).get('prediction_batches', 0)
    inferred = mode == 'frozen' and prediction_delta > 0
    valid = (bool(descriptor.get('active_cell_ids')) and
             {'scheduler_mode', 'ue_scheduler_weights'} <= set(descriptor.get('writable_actions', [])) and
             bool(confirmations) and effects > 0 and verified == POPULATION and frozen and
             (mode == 'fixed' or inferred) and rejections == 0)
    return dict(implementation_id=APP, mode=mode, valid_control_evidence=bool(valid),
                executed_policy_confirmations=len(confirmations), runtime_effect_receipts=effects,
                rejected_receipts=rejections, verified_uids=sorted(verified),
                verified_scheduler_modes=dict(modes), verified_weights={u: sorted(w) for u, w in weights.items()},
                genuine_mlp_inference=inferred, prediction_batches_this_run=prediction_delta if mode == 'frozen' else None,
                frozen_sample_fit_metadata_unchanged=bool(frozen), learning=learning,
                actually_loaded_model_sha256=learning.get('model_sha256'),
                event_counts=dict(Counter(r.get('kind') for r in events)),
                scope='Actual requested mode and per-UID weights must match executed runtime maps; fixed ablations make no ML claim')


def policy_evidence(config, summary, policy, mode):
    if policy not in ('guarded', 'throughput'):
        raise ValueError('Explicit guarded/throughput evaluation policy required')
    apps = config['configs']['ric'].get('xapps', [])
    app = next((a for a in apps if a.get('implementation_id') == APP), {})
    configured = app.get('parameters', {}).get('policy')
    learning = summary.get('throughput_xapp', {})
    actual = learning.get('policy')
    expected_constraints = mode != 'fixed' and policy == 'guarded'
    if configured != policy or actual != policy:
        raise ValueError('Configured/actual throughput policy mismatch')
    if learning.get('constraints_active') is not expected_constraints:
        raise ValueError('Actual throughput policy constraints mismatch')
    return dict(requested_policy=policy, configured_policy=configured, actual_policy=actual,
                constraints_active=expected_constraints)


def evaluate_study(folders, stamp, policy=None):
    if set(folders) != set(ARMS):
        raise ValueError('OFF/neutral/fixed/MLP arms are all required')
    folders = {k: Path(v) for k, v in folders.items()}
    configs = {k: load(v / 'config_snapshot.json') for k, v in folders.items()}
    summaries = {k: load(v / 'final_summary.json') for k, v in folders.items()}
    if policy is None:
        apps = configs['mlp']['configs']['ric'].get('xapps', [])
        policy = next((a.get('parameters', {}).get('policy') for a in apps
                       if a.get('implementation_id') == APP), None)
    policies = {}
    base = configs['off']
    manifest = base['manifest']
    if {u['ue_id'] for u in manifest['ues']} != POPULATION or len(manifest['cells']) != 3:
        raise ValueError('Native six moving UEs and three cells required')
    sections = ('ue', 'phy', 'sim', 'cell', 'beamforming')
    if any(section not in base['configs'] for section in sections):
        raise ValueError('Complete physical and simulation configuration snapshot required')
    for arm, cfg in configs.items():
        if cfg['manifest'] != manifest or any(cfg['configs'].get(section) != base['configs'][section] for section in sections):
            raise ValueError('Radio/PHY/mobility/traffic/service mismatch: ' + arm)
        apps = cfg['configs']['ric'].get('xapps', [])
        if (arm == 'off' and apps or arm != 'off' and
                (len(apps) != 1 or apps[0].get('implementation_id') != APP)):
            raise ValueError('Only the isolated throughput controller may be active')
        if arm in ('neutral', 'fixed'):
            expected = 1 if arm == 'neutral' else 3
            if apps[0]['parameters'].get('mode') != 'fixed' or apps[0]['parameters'].get('fixed_action') != expected:
                raise ValueError('Wrong fixed scheduler-family ablation: ' + arm)
        if arm == 'mlp' and apps[0]['parameters'].get('mode') != 'frozen':
            raise ValueError('MLP evaluation must be frozen')
        if arm != 'off':
            policies[arm] = policy_evidence(cfg, summaries[arm], policy, 'frozen' if arm == 'mlp' else 'fixed')
    arms = {}
    for arm in ARMS:
        users = actual_packet_metrics(folders[arm], configs[arm], summaries[arm])
        delivered = sum(u['delivered_bytes'] for u in users.values())
        offered = sum(u['initial_bytes'] + u['offered_bytes'] for u in users.values())
        arms[arm] = dict(users=users, network_goodput_mbps=sum(u['goodput_mbps'] for u in users.values()),
            network_delivered_bytes=delivered, network_offered_including_initial_bytes=offered,
            recorded_offered_payload_ceiling_gain_fraction=offered / delivered - 1 if delivered else None,
            network_pending_bytes=sum(u['pending_bytes'] for u in users.values()),
            network_overflow_bytes=sum(u['overflow_bytes'] for u in users.values()))
        arms[arm]['arrival_audit'] = arrival_audit(folders[arm], users)
    native = arms['off']
    if any(arms[a]['arrival_audit']['exogenous_slot_inputs_sha256'] != native['arrival_audit']['exogenous_slot_inputs_sha256']
           for a in ('neutral', 'fixed', 'mlp')):
        raise ValueError('Actual exogenous slot-by-slot arrivals differ between matched arms')
    if native['network_goodput_mbps'] <= 0:
        raise ValueError('Positive native goodput required for relative evaluation')
    guards, evidence = {}, {}
    for arm in ('neutral', 'fixed', 'mlp'):
        users = arms[arm]['users']
        delivery, url = {}, {}
        for uid, user in users.items():
            reference = native['users'][uid]
            ratio = user['delivered_bytes'] / reference['delivered_bytes'] if reference['delivered_bytes'] else None
            user['goodput_ratio_vs_off'] = ratio
            user['offered_bytes_match_off'] = user['initial_bytes'] + user['offered_bytes'] == reference['initial_bytes'] + reference['offered_bytes']
            delivery[uid] = user['delivered_bytes'] >= .95 * reference['delivered_bytes'] - 1e-9
            if uid in URLLC:
                if reference['mature_offered_inputs_sha256'] != user['mature_offered_inputs_sha256']:
                    raise ValueError('URLLC exogenous mature packet inputs differ: ' + uid)
                a, b = reference['mature_failure_fraction'], user['mature_failure_fraction']
                delta = None if a is None or b is None else b - a
                url[uid] = dict(original_budget_ms=10., reference_failure_fraction=a,
                    candidate_failure_fraction=b, delta_fraction=delta,
                    passed=delta is not None and delta <= .05 + 1e-12,
                    denominator='Offered packet IDs with a fully observed original 10 ms deadline horizon; unique late, overdue-pending and overflow failures')
        guards[arm] = dict(per_ue_delivery=delivery, urllc_deadline_failure=url,
                           passed=all(delivery.values()) and all(v['passed'] for v in url.values()))
        evidence[arm] = controller_evidence(folders[arm], summaries[arm], stamp if arm == 'mlp' else None,
                                             'frozen' if arm == 'mlp' else 'fixed')
        evidence[arm]['policy_evidence'] = policies[arm]
    gain = {arm: arms[arm]['network_goodput_mbps'] / native['network_goodput_mbps'] - 1
            for arm in ('neutral', 'fixed', 'mlp')}
    ml_vs_fixed = arms['mlp']['network_goodput_mbps'] / arms['fixed']['network_goodput_mbps'] - 1 if arms['fixed']['network_goodput_mbps'] else None
    ml_vs_neutral = arms['mlp']['network_goodput_mbps'] / arms['neutral']['network_goodput_mbps'] - 1 if arms['neutral']['network_goodput_mbps'] else None
    valid = all(v['valid_control_evidence'] for v in evidence.values())
    target_met = valid and gain['mlp'] >= .05 - 1e-12
    qos_passed = guards['mlp']['passed']
    passes = target_met and qos_passed
    limited = native['recorded_offered_payload_ceiling_gain_fraction'] is not None and native['recorded_offered_payload_ceiling_gain_fraction'] < .05
    if policy == 'guarded':
        classification = ('insufficient_control_evidence' if not valid else 'qos_guard_failed' if not qos_passed else
            'controller_effective' if passes else 'below_target_headroom_limited' if limited else 'below_target_experimental')
    else:
        classification = ('insufficient_control_evidence' if not valid else
            'throughput_target_met_qos_preserved' if target_met and qos_passed else
            'throughput_target_met_with_qos_impact' if target_met else
            'throughput_below_target_with_qos_impact' if not qos_passed else
            'throughput_below_target_headroom_limited' if limited else 'throughput_below_target_experimental')
    return dict(seed=manifest['rng_seed'], calibration_not_independent_validation=True,
        throughput_xapp_policy=policy, policy_evidence=policies,
        offline_qos_guard_role='acceptance_requirement' if policy == 'guarded' else 'reported_research_outcome',
        duration_s=manifest['total_slots'] * manifest['slot_duration_ms'] / 1000,
        scope='Whole-run actual DL delivered bytes with matched per-study traffic/PHY; completed-packet quantiles exclude pending/drop outcomes and are not a standalone SLA test',
        arms=arms, guards=guards, controller_evidence=evidence,
        objective=dict(network_goodput_gain_required=.05, per_ue_delivery_ratio_min=.95,
                       urllc_deadline_failure_degradation_max=.05, urllc_packet_deadlines_ms=10),
        network_goodput_gain_vs_rr=gain, mlp_gain_vs_fixed=ml_vs_fixed, mlp_gain_vs_neutral=ml_vs_neutral,
        positive_ml_advantage_vs_fixed=ml_vs_fixed is not None and ml_vs_fixed > 1e-12,
        native_recorded_offered_headroom_below_target=limited,
        headroom_scope='Bound relative to payload actually offered in the native recorded run; MEC-returned bytes are endogenous and may shift in timing. This is not a universal PHY capacity limit.',
        valid_control_evidence=valid, objective_target_met=bool(target_met),
        offline_qos_guard_passed=bool(qos_passed), controller_effective=bool(passes),
        controller_effective_scope='Combined >=5% actual goodput gain, valid control evidence and all offline QoS guards; throughput policy reports its objective separately',
        classification=classification,
        interpretation='RR-relative controller gain, scheduler-family gain and ML-versus-fixed gain are separate. A fitted model is not proof of improvement; no traffic or hyperparameter tuning follows a weak result.')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, help='Fresh MLP artifact, or existing frozen artifact with --skip-training')
    p.add_argument('--train-seconds', type=float, default=6.)
    p.add_argument('--max-train-seconds', type=float, default=10.)
    p.add_argument('--train-seed', type=int, default=173)
    p.add_argument('--eval-seed', type=int, default=55)
    p.add_argument('--eval-seconds', type=float, default=1.)
    p.add_argument('--traffic-load', type=float, default=1., help='Explicit matched demand multiplier; nonnative demand is labelled separately')
    p.add_argument('--throughput-xapp-policy', choices=('guarded', 'throughput'), default='throughput',
                   help='Throughput research objective by default; guarded additionally requires offline QoS acceptance')
    training = p.add_mutually_exclusive_group()
    training.add_argument('--skip-training', action='store_true')
    training.add_argument('--measured-training', type=Path,
                          help='Completed original online collector to audit and reuse for one corrected fit; no new training simulation')
    training.add_argument('--resume-measured-model', type=Path,
                          help='Immutable 40-row untrained model for one bounded actual online bootstrap recovery')
    p.add_argument('--measured-model', type=Path, help='Immutable original JSON containing that collector\'s actual x/y rows')
    p.add_argument('--resume-measured-episode', type=Path, action='append', default=[],
                   help='Original completed online episode; supply both chronological episodes')
    p.add_argument('--resume-seconds', type=float, default=2.)
    p.add_argument('--resume-seed', type=int, default=175)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def command_for(args, output, model, name, seed, seconds, mode, fixed_action=3):
    return ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
        '--duration-s', str(seconds), '--seed', str(seed), '--traffic-load', str(args.traffic_load),
        '--xapp-pool', 'chongyu', '--ue-services', 'drx-mec', '--record-compression', 'gzip',
        '--energy-xapp-mode', 'off', '--latency-xapp-mode', 'off', '--budget-xapp-mode', 'off',
        '--throughput-xapp-mode', mode, '--throughput-xapp-model', str(model),
        '--throughput-xapp-policy', args.throughput_xapp_policy,
        '--throughput-xapp-window-ms', '100', '--throughput-xapp-fixed-action', str(fixed_action),
        '--direct-conflict-mitigation', 'priority',
        '--channel-cache-root', str(ROOT / 'channel_cache' / output.name / ('seed_' + str(seed))),
        '--output', str(output / name)]


def main():
    args = parse_args()
    output = args.output.resolve()
    model = (args.model or output / 'throughput_mlp_model.json').resolve()
    measured_folder = args.measured_training.resolve() if args.measured_training else None
    measured_model = args.measured_model.resolve() if args.measured_model else None
    resume_model = args.resume_measured_model.resolve() if args.resume_measured_model else None
    resume_episodes = [p.resolve() for p in args.resume_measured_episode]
    if output.exists() or not output.is_relative_to(ROOT / 'outputs') or not model.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Choose a fresh study output and a model path under outputs')
    if args.skip_training and not args.dry_run and not model.is_file():
        raise ValueError('Frozen evaluation requires an existing trained MLP artifact')
    if not args.skip_training and model.exists():
        raise ValueError('Online study requires a fresh model; historical artifacts remain immutable')
    if bool(measured_folder) != bool(measured_model):
        raise ValueError('--measured-training and --measured-model must be supplied together')
    if measured_folder:
        if (not measured_folder.is_relative_to(ROOT / 'outputs') or not measured_model.is_relative_to(ROOT / 'outputs') or
                not model.is_relative_to(output) or measured_model == model):
            raise ValueError('Reuse immutable measured inputs under outputs and save the fresh corrected model inside the new study output')
    if resume_model:
        if (len(resume_episodes) != 2 or not model.is_relative_to(output) or model == resume_model or
                any(not p.is_relative_to(ROOT / 'outputs') for p in [resume_model] + resume_episodes) or
                not math.isfinite(args.resume_seconds) or not 0 < args.resume_seconds <= 2):
            raise ValueError('Recovery requires two original episodes, a separate model copy inside the fresh output, and one recovery of at most two seconds')
        if args.eval_seed == args.resume_seed:
            raise ValueError('Recovery and evaluation seeds must differ')
    elif resume_episodes:
        raise ValueError('--resume-measured-episode requires --resume-measured-model')
    if (not all(math.isfinite(v) and v > 0 for v in (args.train_seconds, args.max_train_seconds, args.eval_seconds, args.traffic_load)) or
            not args.train_seconds <= args.max_train_seconds <= 10):
        raise ValueError('Positive finite settings and no more than ten simulated seconds of training required')
    if args.eval_seed in (args.train_seed, args.train_seed + 1):
        raise ValueError('Training/continuation and evaluation network seeds must differ')
    modes = dict(off=('off', 3), neutral=('fixed', 1), fixed=('fixed', 3), mlp=('frozen', 3))
    if args.dry_run:
        commands = []
        if resume_model:
            commands.append(command_for(args, output, model, 'online_recovery', args.resume_seed, args.resume_seconds, 'online'))
        elif not args.skip_training and not measured_folder:
            commands.append(command_for(args, output, model, 'online_training', args.train_seed, args.train_seconds, 'online'))
            remaining = args.max_train_seconds - args.train_seconds
            if remaining:
                commands.append(command_for(args, output, model, 'online_coverage_extension', args.train_seed + 1, remaining, 'online'))
        commands += [command_for(args, output, model, 'seed' + str(args.eval_seed) + '_' + arm,
                        args.eval_seed, args.eval_seconds, *modes[arm]) for arm in ARMS]
        print(json.dumps(dict(dry_run=True, commands=commands,
            optional_extension='Only insufficient actual fitted action coverage permits the one bounded extension; never extend for weak performance',
            evaluation_guard='Stop as not_trained if actual fitted coverage is still insufficient',
            native_traffic=args.traffic_load == 1, traffic_load=args.traffic_load,
            throughput_xapp_policy=args.throughput_xapp_policy,
            measured_collection_refit=dict(folder=str(measured_folder), model=str(measured_model),
                fit_count=1, new_training_physics_s=0., required='Original completed/audited run, exact real row/label provenance and verified new optimizer') if measured_folder else None,
            measured_online_recovery=dict(original_model=str(resume_model), episodes=[str(p) for p in resume_episodes],
                new_physics_s=args.resume_seconds, original_rows_retained=40, new_row_cap=48,
                no_further_extension=True) if resume_model else None,
            independent_seed_211_scheduled=False), indent=2))
        return
    output.mkdir(parents=True)
    model.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(completed=False, status='initializing', cases=[], training_evidence=[], training_readiness=[],
        protocol='X08 genuine online MLP; matched native RR / neutral weighted family / fixed weighted / frozen MLP',
        train_seed=args.train_seed, continuation_seed=args.train_seed + 1, eval_seed=args.eval_seed,
        recovery_seed=args.resume_seed if resume_model else None,
        predeclared_recovery_simulated_s=args.resume_seconds if resume_model else None,
        calibration_not_independent_validation=True, model=str(model), training_simulated_s=0.,
        training_mode='measured_online_recovery' if resume_model else 'measured_collection_refit' if measured_folder else 'skipped' if args.skip_training else 'fresh_online_collection',
        max_training_simulated_s=args.max_train_seconds, traffic_load=args.traffic_load,
        throughput_xapp_policy=args.throughput_xapp_policy,
        offline_qos_guard_role='acceptance_requirement' if args.throughput_xapp_policy == 'guarded' else 'reported_research_outcome',
        explicit_contention_profile=args.traffic_load != 1, independent_seed_211_scheduled=False,
        thresholds=dict(network_goodput_gain=.05, all_ue_delivery_ratio=.95, urllc_failure_increase=.05),
        scope='Original six moving UEs, PHY and services; only explicit matched traffic-load multiplier may differ between separate studies. No performance/hyperparameter tuning loop.')

    def write():
        temporary = output / 'acceptance.tmp'
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False))
        temporary.replace(output / 'acceptance.json')

    def run(name, seed, seconds, mode, fixed_action=3):
        lineage = receipt.get('measured_collection_reuse') or receipt.get('measured_episode_recovery')
        if lineage:
            assert_collection_unchanged(lineage)
        receipt.update(status='running', running=name)
        write()
        command = command_for(args, output, model, name, seed, seconds, mode, fixed_action)
        began = time.perf_counter()
        with (output / (name + '.log')).open('w') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(name + ' failed; inspect its immutable log')
        folder = output / name
        status, performance = load(folder / 'run_status.json'), load(folder / 'performance.json')
        if not status.get('completed') or not status.get('replay', {}).get('audits_passed'):
            raise ValueError('Simulation/HTML audit incomplete: ' + name)
        actual_policy = (policy_evidence(load(folder / 'config_snapshot.json'), load(folder / 'final_summary.json'),
                                        args.throughput_xapp_policy, mode) if mode != 'off' else None)
        receipt['cases'].append(dict(name=name, seed=seed, simulated_s=seconds, mode=mode,
            throughput_xapp_policy=args.throughput_xapp_policy, actual_policy_evidence=actual_policy,
            fixed_action=fixed_action if mode == 'fixed' else None, output=str(folder), command=command,
            source_sha256=status['source_sha256'], simulation_wall_s=performance['simulation_s'],
            internal_wall_s=status['elapsed_s'], external_wall_s=time.perf_counter() - began,
            html=status['replay']['html'], html_audits_passed=True))
        if len({r['source_sha256'] for r in receipt['cases']}) != 1:
            raise ValueError('Source changed during the study')
        if lineage and lineage.get('fitting_source_sha256') and status['source_sha256'] != lineage['fitting_source_sha256']:
            raise ValueError('Corrected fitting and evaluation source fingerprints differ')
        if lineage:
            assert_collection_unchanged(lineage)
            if name.endswith('_off'):
                lineage['compatibility_with_evaluation'] = collection_eval_compatibility(lineage, load(folder / 'config_snapshot.json'))
        write()
        return folder

    write()
    try:
        if resume_model:
            lineage = prepare_measured_recovery(resume_episodes, resume_model, model, args.traffic_load)
            receipt['measured_episode_recovery'] = lineage
            initial = load(model)
            folder = run('online_recovery', args.resume_seed, args.resume_seconds, 'online')
            receipt['training_simulated_s'] += args.resume_seconds
            receipt['training_evidence'].append(training_evidence(folder, model, initial))
            updated = load(model)
            if updated['seed'] != initial['seed']:
                raise ValueError('Recovery changed the original learner seed')
            coverage = readiness(model)
            receipt['training_readiness'].append(coverage)
            lineage.update(recovery_simulated_s=args.resume_seconds,
                total_training_simulated_s_including_history=lineage['original_training_simulated_s'] + args.resume_seconds,
                recovery_source_sha256=load(folder / 'run_status.json')['source_sha256'],
                recovered_model_sha256=sha256(model), original_40_row_prefix_retained=True,
                new_measured_rows=len(updated['x']) - 40, new_verified_fit=coverage['ready'])
            if lineage['total_training_simulated_s_including_history'] > 10 + 1e-9:
                raise ValueError('Technical recovery exceeded the original ten-second overall training budget')
            assert_collection_unchanged(lineage)
        elif measured_folder:
            receipt.update(status='auditing_and_refitting_measured_collection', running='measured_collection_refit')
            write()
            lineage = refit_measured_collection(measured_folder, measured_model, model, args.traffic_load)
            receipt['measured_collection_reuse'] = lineage
            coverage = readiness(model)
            receipt['training_readiness'].append(coverage)
        elif not args.skip_training:
            folder = run('online_training', args.train_seed, args.train_seconds, 'online')
            receipt['training_simulated_s'] += args.train_seconds
            receipt['training_evidence'].append(training_evidence(folder, model))
            coverage = readiness(model)
            receipt['training_readiness'].append(coverage)
            remaining = args.max_train_seconds - args.train_seconds
            if not coverage['ready'] and remaining > 0:
                initial = load(model) if model.is_file() else None
                folder = run('online_coverage_extension', args.train_seed + 1, remaining, 'online')
                receipt['training_simulated_s'] += remaining
                receipt['training_evidence'].append(training_evidence(folder, model, initial))
                coverage = readiness(model)
                receipt['training_readiness'].append(coverage)
        else:
            coverage = readiness(model)
            receipt['training_readiness'].append(coverage)
        if not coverage['ready']:
            receipt.update(completed=True, status='not_trained', learned_model_claim=False,
                           stop_reason='Bounded actual fitted coverage is insufficient; no frozen evaluation or fake ML claim')
            receipt.pop('running', None)
            write()
            print(json.dumps(receipt, indent=2, allow_nan=False))
            return
        stamp = model_stamp(model)
        receipt.update(model_before_evaluation=stamp, learned_model_claim=True)
        write()
        folders = {}
        for arm in ARMS:
            folders[arm] = run('seed' + str(args.eval_seed) + '_' + arm, args.eval_seed,
                                args.eval_seconds, *modes[arm])
            if sha256(model) != stamp['sha256']:
                raise ValueError('Evaluation changed the frozen MLP artifact')
        evaluation = evaluate_study(folders, stamp, args.throughput_xapp_policy)
        receipt.update(completed=True, status=evaluation['classification'], evaluation=evaluation)
        receipt.pop('running', None)
        write()
    except BaseException as error:
        receipt.update(completed=False, status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        write()
        raise
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
