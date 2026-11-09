"""Bounded X02 AP-energy / X08 throughput study with real availability control."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_ue_energy_study import load, POPULATION
from experiments.run_throughput_mlp_study import (
    actual_packet_metrics, arrival_audit, model_stamp as mlp_stamp,
    readiness as mlp_readiness, quantile, sha256, collection_eval_compatibility)
from experiments.run_scheduler_conflict_study import (
    X08, atomic_json, completed_case, controller_evidence as scheduler_evidence,
    jsonl)

APP = 'ap_energy_rf'
ARMS = ('off', 'x02', 'x08', 'both')
FEATURES = ['network_queue_kb', 'network_reported_goodput_mbps', 'network_offered_mbps',
    'maximum_cell_queue_kb', 'maximum_cell_offered_mbps', 'minimum_occupied_cell_offered_mbps',
    'backlogged_ue_fraction', 'empty_cell_fraction'] + ['action_' + str(i) for i in range(4)]
TARGETS = ['network_ap_power_w', 'network_goodput_mbps', 'ue_2_unique_failure_events_per_s', 'ue_6_unique_failure_events_per_s']
EMPTY_CONTROL_AMENDMENT = ('Declared before the first RF training sample: retain the failed v2 strict no-harm receipt; '
    'permit structurally valid empty-cell control with reference/CQI coupling, without adding numerical tolerance or tuning QoS thresholds. '
    'Removing an empty AP changes the full-band reference-measurement interference hypothesis and AMC, not its actual scheduled data RF. '
    'Strict packet-KPI no-harm remains a separate unmodified diagnostic, not an accepted claim.')


def close(a, b):
    return type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def packet_statistics(folder, config, summary):
    users = actual_packet_metrics(folder, config, summary)
    latency = {u: [] for u in users}
    for event in jsonl(folder, 'packet_events'):
        if event['kind'] == 'delivery' and event['completed']:
            latency[event['ue_id']].append(event['latency_ms'])
    profiles = {u['ue_id']: u for u in config['configs']['ue']['ues']}
    def group(ids):
        delays = [d for uid in ids for d in latency[uid]]
        n = sum(users[u]['mature_offered_packets'] for u in ids)
        failed = sum(users[u]['mature_failed_packets'] for u in ids)
        offered = sum(users[u]['initial_bytes'] + users[u]['offered_bytes'] for u in ids)
        delivered = sum(users[u]['delivered_bytes'] for u in ids)
        ratios = [users[u]['delivered_bytes'] / (users[u]['initial_bytes'] + users[u]['offered_bytes'])
                  for u in ids if users[u]['initial_bytes'] + users[u]['offered_bytes']]
        return dict(ue_ids=ids, deadlines_ms=sorted({profiles[u]['latency_budget_ms'] for u in ids}),
            goodput_mbps=sum(users[u]['goodput_mbps'] for u in ids),
            mature_packets=n, failed_mature_packets=failed, failure_fraction=failed / n if n else None,
            completed_packets=len(delays), completed_mean_ms=sum(delays) / len(delays) if delays else None,
            completed_p50_ms=quantile(delays, .5), completed_p95_ms=quantile(delays, .95),
            completed_p99_ms=quantile(delays, .99), offered_bytes=offered, delivered_bytes=delivered,
            delivery_ratio=delivered / offered if offered else None,
            pending_bytes=sum(users[u]['pending_bytes'] for u in ids),
            overflow_bytes=sum(users[u]['overflow_bytes'] for u in ids),
            jain_delivery_satisfaction=sum(ratios) ** 2 / (len(ratios) * sum(x * x for x in ratios)) if ratios and any(ratios) else None)
    classes = {name: group(sorted(u for u in users if profiles[u]['slice_id'].lower() == name))
               for name in sorted({profiles[u]['slice_id'].lower() for u in users})}
    return dict(users=users, network=group(sorted(users)), traffic_classes=classes,
        arrival_audit=arrival_audit(folder, users),
        definitions='Failure fractions pool unique mature packet counts using each UE original deadline; latency quantiles pool actual completed samples and exclude pending/drop outcomes. Network is not an all-10ms SLA.')


def ap_energy_audit(folder, config, summary):
    """Independently reconstruct each recorded modeled energy component."""
    service = config['configs']['phy']['rt_sys']['ap_services']
    if service.get('enabled') is not True or summary['ap_services']['config'] != service:
        raise ValueError('The same explicit enabled AP model is required')
    power, dt = service['power'], config['manifest']['slot_duration_ms'] / 1000
    totals, states, wakes, last_sleep = {}, {}, {}, {}
    inactive_slots, occupied_sleep, count, emitted_rf = 0, 0, 0, 0.
    for row in jsonl(folder, 'rt_sys_trace'):
        count += 1
        if row['slot_id'] != count:
            raise ValueError('AP energy audit requires every ordered recorded slot')
        cells = row.get('ap_services', {}).get('cells', {})
        if set(cells) != set(summary['ap_services']['cells']):
            raise ValueError('Actual per-slot AP ledger is missing cells')
        for cid, cell in cells.items():
            state, policy = cell['ap_state'], cell['ap_policy']
            emitted_rf += cell['emitted_rf_power_w'] * dt
            if state not in ('active', 'sleeping', 'waking') or cell['ap_available'] is not (state == 'active'):
                raise ValueError('AP availability and actual runtime state disagree')
            totals.setdefault(cid, Counter()); states.setdefault(cid, Counter())
            components = dict.fromkeys(('static', 'load', 'pa', 'sleep', 'waking', 'transition'), 0.)
            if state == 'active':
                components.update(static=power['static_w'] * dt,
                    load=power['load_w'] * cell['ap_load_fraction'] * dt,
                    pa=cell['emitted_rf_power_w'] / power['pa_efficiency'] * dt)
            elif state == 'sleeping':
                if policy not in power['sleep_w']:
                    raise ValueError('Sleeping AP lacks an explicit sleep-power profile')
                last_sleep[cid] = policy
                components['sleep'] = power['sleep_w'][policy] * dt
            else:
                components['waking'] = power['waking_w'] * dt
            new_wakes = cell['ap_wake_transitions_total'] - wakes.get(cid, 0)
            if type(new_wakes) is not int or new_wakes < 0 or (new_wakes and cid not in last_sleep):
                raise ValueError('Invalid actual AP wake transition counter')
            if new_wakes:
                components['transition'] = new_wakes * power['wake_transition_j'][last_sleep[cid]]
            wakes[cid] = cell['ap_wake_transitions_total']
            recorded = cell['slot_energy_components_j']
            if set(recorded) != set(components) or any(not close(recorded[k], v) for k, v in components.items()):
                raise ValueError('AP slot energy components disagree with the declared power model')
            energy = sum(components.values())
            if not close(cell['slot_energy_j'], energy) or not close(cell['ap_power_w'] * dt, energy):
                raise ValueError('AP slot power/energy arithmetic failed')
            totals[cid].update(components); states[cid][state] += 1
            if (not close(cell['ap_energy_total_j'], sum(totals[cid].values())) or
                    any(not close(cell['ap_energy_components_j'].get(k, 0), v) for k, v in totals[cid].items()) or
                    cell['ap_available_slots_total'] != states[cid]['active'] or
                    cell['ap_sleep_slots_total'] != states[cid]['sleeping'] or cell['ap_waking_slots_total'] != states[cid]['waking'] or
                    not close(cell['ap_available_fraction'], states[cid]['active'] / count)):
                raise ValueError('AP cumulative energy/availability conservation failed')
            if state != 'active':
                inactive_slots += 1
                radio = row['cells'][cid]
                if cell['emitted_rf_power_w'] != 0 or radio['average_used_w'] != 0 or any(
                        u.get('scheduled') for u in row['ues'] if u['cell_id'] == cid):
                    raise ValueError('Unavailable AP emitted data or retained an actual grant')
                occupied_sleep += bool(any(u['cell_id'] == cid for u in row['ues']))
    final = summary['ap_services']
    if count != config['manifest']['total_slots'] or final['accounted_slots'] != count:
        raise ValueError('AP energy horizon differs from the actual completed simulation')
    energy = sum(sum(v.values()) for v in totals.values())
    if not close(final['network_ap_energy_total_j'], energy) or any(
            not close(final['cells'][cid]['ap_energy_total_j'], sum(values.values())) for cid, values in totals.items()):
        raise ValueError('Final AP energy summary disagrees with the reconstructed ledger')
    return dict(network_ap_energy_j=energy, emitted_rf_energy_j=emitted_rf, components_j={c: dict(v) for c, v in totals.items()},
        states_slots={c: dict(v) for c, v in states.items()}, wake_transitions=wakes,
        unavailable_cell_slots=inactive_slots, occupied_unavailable_cell_slots=occupied_sleep,
        actual_slot_energy_reconstruction_verified=True,
        transition_audit_scope='Approved single AP owner with at most one policy change per cell per slot; hidden multiple in-slot sleep-depth changes are outside this study',
        scope='Modeled AP static/load/PA/sleep/waking/transition energy; distinct from emitted RF joules and host-device power')


def case_data(folder):
    folder = Path(folder)
    config, summary = load(folder / 'config_snapshot.json'), load(folder / 'final_summary.json')
    result = packet_statistics(folder, config, summary)
    result['ap_energy'] = ap_energy_audit(folder, config, summary)
    result['audit'] = completed_case(folder)
    for name in ('ap_rf_events.jsonl', 'ap_rf_events.jsonl.gz'):
        if (folder / name).is_file():
            result['audit']['raw_file_sha256'][str(folder / name)] = sha256(folder / name)
    return result


def model_stamp(path):
    p = load(path)
    return dict(sha256=sha256(path), samples=len(p['x']), trained_samples=p['trained_samples'],
        fit_count=p['fit_count'], prediction_batches=p['prediction_batches'], fit_parameters=p['fit_parameters'])


def readiness(path):
    if not Path(path).is_file():
        return dict(ready=False, reason='No measured model artifact')
    from ric.near_rt.pools.chongyu.online_ap_rf import OnlineApRF
    OnlineApRF.load(path, FEATURES)
    p = load(path)
    counts = Counter(x[-4:].index(1.) for x in p['x'][:p['trained_samples']])
    return dict(ready=bool(p['forest'] and p['trained_samples'] >= 24 and all(counts[i] >= 6 for i in range(4))),
        samples=len(p['x']), trained_samples=p['trained_samples'], fitted_action_counts={str(i): counts[i] for i in range(4)})


def _action_map(membership, cells, action):
    occupied = set(membership.values())
    return {c: 'active' if action == 0 else 'deep_sleep' if c not in occupied else
            'micro_sleep' if action == 2 else 'light_sleep' if action == 3 else 'active' for c in cells}


def training_evidence(folder, model, previous=None, payload=None):
    """Exactly one actual global label per completed, settled cycle window."""
    folder = Path(folder)
    cfg, summary = load(folder / 'config_snapshot.json'), load(folder / 'final_summary.json')
    ap_energy_audit(folder, cfg, summary)
    apps = cfg['configs']['ric']['xapps']
    if len(apps) != 1 or apps[0]['implementation_id'] != APP or apps[0]['parameters']['mode'] != 'online':
        raise ValueError('RF collection must isolate online AP control')
    p, before = payload or load(model), previous or dict(x=[], y=[], fit_count=0)
    if p['feature_names'] != FEATURES or p['target_names'] != TARGETS or p['x'][:len(before['x'])] != before['x'] or p['y'][:len(before['y'])] != before['y']:
        raise ValueError('Actual AP feature/target schema or immutable measured prefix changed')
    traces = {r['slot_id']: r for r in jsonl(folder, 'rt_sys_trace')}
    packets = list(jsonl(folder, 'packet_events'))
    first_failure = {}
    for e in packets:
        if e['ue_id'] in ('ue_2', 'ue_6') and (e['kind'] == 'deadline_miss' or
                e['kind'] == 'delivery' and e['completed'] and e['latency_ms'] > 10):
            key = (e['ue_id'], e['packet_id']); first_failure[key] = min(first_failure.get(key, e['slot_id']), e['slot_id'])
    confirmations, samples, fits = [], [], []
    dt, last_end = cfg['manifest']['slot_duration_ms'] / 1000, -1
    receipts = list(jsonl(folder, 'control_receipts'))
    for r in jsonl(folder, 'ap_rf_events'):
        if r['kind'] == 'executed_policy_confirmed': confirmations.append(r)
        if r['kind'] == 'rf_fit': fits.append(r)
        if r['kind'] != 'measured_sample': continue
        start, end, duration, action = r['produced_start_slot'], r['produced_end_slot'], r['duration_s'], r['action']
        if start < last_end or end not in traces or start not in traces or end <= start or duration < .1 - 1e-9 or not close(duration, (end - start) * dt):
            raise ValueError('Invalid actual AP outcome window')
        last_end = end
        if (type(action) is not int or not 0 <= action < 4 or r['features'][-4:] != [float(action == i) for i in range(4)] or r['target_order'] != TARGETS or
                r.get('original_latency_budgets_ms') != {'ue_2': 10, 'ue_6': 10}):
            raise ValueError('Actual AP arm/feature/target encoding mismatch')
        prior = [c for c in confirmations if c['produced_slot'] <= start]
        c = prior[-1] if prior else None
        begin, finish = traces[start]['ap_services']['cells'], traces[end]['ap_services']['cells']
        cells = set(begin)
        if (c is None or c['action'] != action or c['executed_snapshot'] != r['snapshot'] or
                set(r['membership']) != POPULATION or r['snapshot'] != _action_map(r['membership'], cells, action)):
            raise ValueError('AP label lacks its actual whole-network action confirmation')
        for cid in cells:
            applied, period = c['actual_applied_slots'][cid], r['cycle_period_slots_by_cell'][cid]
            ack = [a for a in receipts if a.get('status') == 'applied' and a.get('apply_slot', -1) <= c['produced_slot'] and
                a.get('after', {}).get(cid, {}).get('ap_policy_owner') == APP and
                a['after'][cid].get('ap_policy') == r['snapshot'][cid] and
                a['after'][cid].get('ap_policy_generation') == r['actual_policy_generations'][cid]]
            if (not ack or c['actual_owners'][cid] != APP or type(applied) is not int or applied > c['produced_slot'] or start - applied < period or
                    applied != begin[cid]['ap_policy_applied_slot'] or period != begin[cid]['ap_cycle_period_slots'] or
                    r['start_cycle_phase_by_cell'][cid] != begin[cid]['ap_cycle_phase_slot'] or
                    r['end_cycle_phase_by_cell'][cid] != finish[cid]['ap_cycle_phase_slot'] or
                    (end - start) % period or
                    r['start_cycle_phase_by_cell'][cid] != r['end_cycle_phase_by_cell'][cid] or
                    r['snapshot'][cid] == 'active' and begin[cid]['ap_state'] != 'active'):
                raise ValueError('AP sample lacks real applied ACK, settled initial wake and complete aligned cycles')
            for slot in range(start, end + 1):
                actual = traces[slot]['ap_services']['cells'][cid]
                if actual['ap_policy'] != r['snapshot'][cid] or actual['ap_policy_owner'] != APP or actual['ap_policy_generation'] != r['actual_policy_generations'][cid]:
                    raise ValueError('AP actual policy/owner/generation changed inside measured window')
            for field, key in (('ap_energy_j_by_cell', 'ap_energy_total_j'), ('ap_available_slots_by_cell', 'ap_available_slots_total'),
                               ('ap_sleep_slots_by_cell', 'ap_sleep_slots_total'), ('ap_waking_slots_by_cell', 'ap_waking_slots_total')):
                if not close(r[field][cid], finish[cid][key] - begin[cid][key]):
                    raise ValueError('AP label counters differ from actual slot-ledger deltas')
            if sum(r[f][cid] for f in ('ap_available_slots_by_cell', 'ap_sleep_slots_by_cell', 'ap_waking_slots_by_cell')) != end - start:
                raise ValueError('AP availability window does not conserve slots')
        for name, ids in (('delivered_bits_by_ue', POPULATION), ('offered_bits_by_ue', POPULATION),
                          ('deadline_failed_packets_by_ue', ('ue_2', 'ue_6')), ('overflow_packets_by_ue', ('ue_2', 'ue_6'))):
            if set(r[name]) != set(ids): raise ValueError('Missing actual packet counters')
            for uid in ids:
                if name == 'deadline_failed_packets_by_ue':
                    value = sum(start < slot <= end for (u, _), slot in first_failure.items() if u == uid)
                else:
                    events = [e for e in packets if e['ue_id'] == uid and start < e['slot_id'] <= end]
                    if name == 'delivered_bits_by_ue': value = sum(e['bytes'] * 8 for e in events if e['kind'] == 'delivery')
                    elif name == 'offered_bits_by_ue': value = sum(e['bytes'] * 8 for e in events if e['kind'] == 'arrival')
                    else: value = sum(e['kind'] == 'arrival' and not e['admitted'] for e in events)
                if r[name][uid] != value: raise ValueError('AP label packet deltas differ from actual packet events')
        if any({u['ue_id']: u['cell_id'] for u in traces[slot]['ues']} != r['membership'] for slot in range(start, end + 1)):
            raise ValueError('Actual serving membership changed inside the measured AP window')
        targets = [sum(r['ap_energy_j_by_cell'].values()) / duration, sum(r['delivered_bits_by_ue'].values()) / duration / 1e6] + [
            (r['deadline_failed_packets_by_ue'][u] + r['overflow_packets_by_ue'][u]) / duration for u in ('ue_2', 'ue_6')]
        if len(r['targets']) != 4 or any(not close(x, y) for x, y in zip(r['targets'], targets)):
            raise ValueError('AP RF labels disagree with measured energy/packet arithmetic')
        samples.append(r)
    room = 48 - len(before['x']); added = samples[:room]
    if (p['x'][len(before['x']):] != [[struct.unpack('f', struct.pack('f', x))[0] for x in r['features']] for r in added] or
            p['y'][len(before['y']):] != [r['targets'] for r in added] or
            any(r['retained_for_training'] is not (i < room) for i, r in enumerate(samples)) or
            p['fit_count'] != before.get('fit_count', 0) + len(fits) or fits and fits[-1]['samples'] != p['trained_samples']):
        raise ValueError('Persisted AP RF history or genuine online fit evidence disagrees')
    if any(summary['ap_xapp'][k] != p[k] for k in ('trained_samples', 'fit_count')) or summary['ap_xapp']['samples'] != len(p['x']):
        raise ValueError('Actual AP summary and saved fitted-prefix counts disagree')
    return dict(new_actual_global_rows=len(added), measured_windows=len(samples),
        raw_energy_packet_labels_verified=True, complete_cycle_actual_ack_verified=True,
        retained_after_cap=len(samples) - len(added), historical_prefix_unchanged=True)


def ap_control_evidence(folder, config, summary, stamp=None):
    events = list(jsonl(folder, 'ap_rf_events'))
    app = next(a for a in config['configs']['ric']['xapps'] if a['implementation_id'] == APP)
    mode = app['parameters']['mode']
    learner, predictions, replayed, errors = None, 0, 0, []
    learning = summary['ap_xapp']
    if mode == 'frozen':
        from ric.near_rt.pools.chongyu.online_ap_rf import OnlineApRF
        path = Path(app['parameters']['model_path'])
        if not path.is_absolute(): path = ROOT / path
        if stamp is None or sha256(path) != stamp['sha256']: raise ValueError('Frozen AP model bytes changed')
        learner = OnlineApRF.load(path, FEATURES)
        frozen = (learning['model_sha256'] == stamp['sha256'] and all(learning[k] == stamp[k] for k in ('samples', 'trained_samples', 'fit_count', 'fit_parameters')))
        predictions = learning['prediction_batches'] - stamp['prediction_batches']
        for event in events:
            if event['kind'] != 'decision': continue
            features, action = event.get('features'), event.get('action')
            if not isinstance(features, list) or len(features) != 12 or type(action) is not int or not 0 <= action < 4 or features[-4:] != [float(action == i) for i in range(4)]:
                errors.append('Missing actual frozen inference context'); continue
            values = learner.predict([features[:-4] + [float(i == j) for j in range(4)] for i in range(4)])
            lowest = min(v[0] for v in values); tolerance = max(1e-9, abs(lowest) * 1e-6)
            finalists = [i for i in range(4) if values[i][0] <= lowest + tolerance]
            expected = min(finalists, key=lambda i: (values[i][2] + values[i][3], -values[i][1], i))
            if values[0][0] - values[expected][0] < .01 * values[0][0]: expected = 0
            audit = event.get('prediction_audit') or {}
            recorded = audit.get('raw_predictions', [])
            if (event.get('status') != 'rf' or expected != action or audit.get('selected_action') != action or
                    audit.get('constraints_active') is not False or len(recorded) != 4 or
                    any(len(a) != 4 or any(not close(x, y) for x, y in zip(a, b)) for a, b in zip(recorded, values)) or
                    len(event.get('prediction') or []) != 4 or any(not close(x, y) for x, y in zip(event['prediction'], values[action]))):
                errors.append('AP forecasts/selection differ from strict frozen-model replay')
            replayed += 1
        if sha256(path) != stamp['sha256']: raise ValueError('Offline AP inference replay changed the artifact')
        frozen = frozen and predictions == replayed and replayed > 0 and not errors
    else:
        frozen = mode == 'fixed' and app['parameters']['fixed_action'] == 1
    applied_cells, receipts = set(), 0
    for r in jsonl(folder, 'control_receipts'):
        provenance = r.get('provenance', [])
        if isinstance(provenance, dict): provenance = [provenance]
        requested = next((f['value'] for f in r.get('fields', []) if f['field'] == 'cell_sleep_mode'), None)
        if r.get('status') != 'applied' or not any(p.get('implementation_id') == APP for p in provenance): continue
        for cid, actual in r.get('after', {}).items():
            if requested is not None and actual.get('cell_sleep_mode') == requested and actual.get('ap_policy') == requested and actual.get('ap_policy_owner') == APP:
                applied_cells.add(cid); receipts += 1
    descriptors = summary['xapp_library']['instances']
    descriptor = next((d for d in descriptors if d['implementation_id'] == APP), {})
    confirmations = sum(e['kind'] == 'executed_policy_confirmed' for e in events)
    return dict(valid=bool(frozen and confirmations and applied_cells == set(summary['ap_services']['cells']) and
            set(descriptor.get('writable_actions', [])) == {'cell_sleep_mode'}),
        actual_applied_cells=sorted(applied_cells), actual_receipts=receipts,
        genuine_replayed_inferences=replayed, prediction_batches_delta=predictions,
        frozen_metadata_and_forecasts_unchanged=bool(frozen), errors=errors, objective='energy_first_QoS_observed_not_constrained')


def matched_inputs(folders):
    configs = {a: load(Path(p) / 'config_snapshot.json') for a, p in folders.items()}
    base = configs['off']
    for a, c in configs.items():
        if c['manifest'] != base['manifest'] or any(c['configs'][k] != base['configs'][k] for k in ('ue', 'phy', 'sim', 'cell', 'beamforming')):
            raise ValueError('AP experiment must preserve matched traffic/radio/PHY/profile/initial state: ' + a)
    return configs


def legacy_parity(folder, reference):
    current, old = load(Path(folder) / 'final_summary.json'), load(Path(reference) / 'final_summary.json')
    keys = ('packet_metrics', 'ue_services', 'tb_outcomes')
    # Host elapsed instrumentation is not a simulated network outcome.
    # Keep every other model state/counter/configuration byte in the comparison.
    ignored = ('ue_services.module_wall_s',)
    def outcomes(summary, key):
        value = deepcopy(summary[key])
        if key == 'ue_services': value.pop('module_wall_s', None)
        return value
    equal = {k: k in current and k in old and outcomes(current, k) == outcomes(old, k) for k in keys}
    if not all(equal.values()): raise ValueError('Enabled all-active AP case differs from its immutable historical reference')
    return dict(verified=True, exact_summary_sections=equal, reference=str(reference),
        reference_summary_sha256=sha256(Path(reference) / 'final_summary.json'),
        excluded_host_timing_fields=list(ignored),
        scope='Exact packet, UE-service and TB outcomes excluding host elapsed timing; the new modeled AP-energy field is additional')


def negative_control(folders):
    configs = matched_inputs(folders)
    data = {a: case_data(p) for a, p in folders.items()}
    base, candidate = data['off'], data['empty_only']
    if base['audit']['source_sha256'] != candidate['audit']['source_sha256']:
        raise ValueError('Negative control source must remain identical')
    if base['arrival_audit']['exogenous_slot_inputs_sha256'] != candidate['arrival_audit']['exogenous_slot_inputs_sha256']:
        raise ValueError('Negative-control exogenous inputs differ')
    no_harm = all(candidate['users'][u]['delivered_bytes'] >= base['users'][u]['delivered_bytes'] and
        candidate['users'][u]['mature_failed_packets'] <= base['users'][u]['mature_failed_packets'] and
        candidate['users'][u]['pending_bytes'] <= base['users'][u]['pending_bytes'] and
        candidate['users'][u]['overflow_bytes'] <= base['users'][u]['overflow_bytes'] and
        (base['users'][u]['completed_packet_p95_latency_ms'] is None or
         candidate['users'][u]['completed_packet_p95_latency_ms'] is not None and candidate['users'][u]['completed_packet_p95_latency_ms'] <= base['users'][u]['completed_packet_p95_latency_ms'] + 1e-9)
        for u in POPULATION)
    control = ap_control_evidence(folders['empty_only'], configs['empty_only'], load(Path(folders['empty_only']) / 'final_summary.json'))
    saving = 1 - candidate['ap_energy']['network_ap_energy_j'] / base['ap_energy']['network_ap_energy_j']
    passed = control['valid'] and saving > 1e-12 and no_harm and candidate['ap_energy']['occupied_unavailable_cell_slots'] == 0
    structural = control['valid'] and saving > 1e-12 and candidate['ap_energy']['occupied_unavailable_cell_slots'] == 0
    deltas = {}
    fields = ('delivered_bytes', 'goodput_mbps', 'pending_bytes', 'overflow_bytes', 'admitted_packets',
        'completed_packets', 'pending_packets', 'overflow_packets', 'mature_offered_packets', 'mature_failed_packets',
        'unique_deadline_failed_packets', 'mature_failure_fraction', 'completed_packet_mean_latency_ms',
        'completed_packet_p50_latency_ms', 'completed_packet_p95_latency_ms')
    for uid in sorted(POPULATION):
        a, b = base['users'][uid], candidate['users'][uid]
        deltas[uid] = {k: b[k] - a[k] if a.get(k) is not None and b.get(k) is not None else None for k in fields}
    return dict(passed=bool(passed), modeled_ap_energy_saving_fraction=saving, no_observed_packet_kpi_harm=no_harm,
        structural_validation_passed=bool(structural), per_ue_qos_delta_vs_off=deltas,
        duration_s=configs['off']['manifest']['total_slots'] * configs['off']['manifest']['slot_duration_ms'] / 1000,
        network_seed=configs['off']['manifest']['rng_seed'],
        fixed_empty_only_control=control, cases=data,
        scope='Analytic fixed empty-cell control, not learned RF; passed retains strict packet-KPI no-harm. Structural acceptance is a separate explicit protocol option.')


def empty_control_acceptance(result, allow_reference_coupling=False):
    strict = result.get('passed') is True
    return dict(mode='reference_cqi_coupled_structural' if allow_reference_coupling else 'strict_no_harm',
        accepted=bool(strict or allow_reference_coupling and result.get('structural_validation_passed') is True),
        strict_negative_passed=strict, structural_validation_passed=result.get('structural_validation_passed'),
        protocol_amendment=EMPTY_CONTROL_AMENDMENT if allow_reference_coupling else None)


def revalidate_empty_control_acceptance(receipt, actual):
    recorded = receipt.get('negative_control') or {}
    if recorded.get('passed') is not actual.get('passed') or recorded.get('no_observed_packet_kpi_harm') is not actual.get('no_observed_packet_kpi_harm'):
        raise ValueError('Original strict empty-control result differs from actual recordings')
    stored = receipt.get('empty_control_acceptance')
    if stored is None and recorded.get('passed') is True:
        # Original strict no-harm studies remain valid without new metadata.
        return empty_control_acceptance(actual)
    explicit = receipt.get('allow_empty_reference_coupling') is True
    expected = empty_control_acceptance(actual, explicit)
    if stored != expected or not expected['accepted']:
        raise ValueError('Original training lacks explicit accepted/revalidated empty-reference protocol; failed v2 is not retroactively accepted')
    return expected


def evaluate_study(folders, x08_stamp, ap_stamp):
    configs = matched_inputs(folders)
    data = {a: case_data(p) for a, p in folders.items()}
    if set(folders) != set(ARMS) or len({d['audit']['source_sha256'] for d in data.values()}) != 1:
        raise ValueError('All four cases require one frozen evaluation source')
    expected = {'off': set(), 'x02': {APP}, 'x08': {X08}, 'both': {APP, X08}}
    if data['off']['ap_energy']['unavailable_cell_slots'] or data['off']['network']['goodput_mbps'] <= 0:
        raise ValueError('Enabled all-active positive-goodput OFF baseline required')
    evidence = {}
    for a in ARMS:
        cfg, summary = configs[a], load(Path(folders[a]) / 'final_summary.json')
        apps = cfg['configs']['ric']['xapps']
        if {v['implementation_id'] for v in apps} != expected[a] or len(apps) != len(expected[a]) or cfg['configs']['ric'].get('direct_conflict_mitigation', 'priority') != 'priority':
            raise ValueError('Only disjoint AP and scheduler controls, with no random mitigation, are permitted')
        evidence[a] = {}
        for app in apps:
            if app['parameters']['mode'] != 'frozen': raise ValueError('Every evaluated learner must be frozen')
            if app['implementation_id'] == APP: evidence[a][APP] = ap_control_evidence(folders[a], cfg, summary, ap_stamp)
            else:
                if app['parameters'].get('policy') != 'throughput': raise ValueError('X08 research policy must be explicit')
                evidence[a][X08] = scheduler_evidence(Path(folders[a]), summary, X08, x08_stamp, model_path=app['parameters']['model_path'])
        if data[a]['arrival_audit']['exogenous_slot_inputs_sha256'] != data['off']['arrival_audit']['exogenous_slot_inputs_sha256']:
            raise ValueError('Actual exogenous traffic differs between paired cases')
        for uid in ('ue_2', 'ue_6'):
            if data[a]['users'][uid]['mature_offered_inputs_sha256'] != data['off']['users'][uid]['mature_offered_inputs_sha256']:
                raise ValueError('Mature URLLC input cohorts differ')
    if any(r.get('reason') == 'direct_conflict_random_selection' for r in jsonl(folders['both'], 'rule_decisions')):
        raise ValueError('Random direct-conflict selection cannot establish the requested indirect case')
    descriptors = load(Path(folders['both']) / 'final_summary.json')['xapp_library']['instances']
    contracts = {d['implementation_id']: set(d['writable_actions']) for d in descriptors if d['implementation_id'] in (APP, X08)}
    disjoint = set(contracts) == {APP, X08} and not (contracts[APP] & contracts[X08])
    energy = {a: d['ap_energy']['network_ap_energy_j'] for a, d in data.items()}
    rate = {a: d['network']['goodput_mbps'] for a, d in data.items()}
    individually = dict(x02=1 - energy['x02'] / energy['off'] >= .05 - 1e-12,
        x08=rate['x08'] / rate['off'] - 1 >= .05 - 1e-12)
    harm = dict(x02_ap_energy_increase_vs_alone_fraction=energy['both'] / energy['x02'] - 1,
                x08_goodput_loss_vs_alone_fraction=1 - rate['both'] / rate['x08'])
    qos = {}
    for a in ARMS[1:]:
        delivery = {u: data[a]['users'][u]['delivered_bytes'] >= .95 * data['off']['users'][u]['delivered_bytes'] for u in POPULATION}
        b, c = data['off']['traffic_classes']['urllc']['failure_fraction'], data[a]['traffic_classes']['urllc']['failure_fraction']
        delta = None if b is None or c is None else c - b
        qos[a] = dict(per_ue_delivery_at_least_95pct_off=delivery, pooled_urllc_failure_delta=delta,
            passed=all(delivery.values()) and delta is not None and delta <= .05 + 1e-12)
    valid = disjoint and all(v.get('valid', v.get('valid_control_evidence', False)) for cases in evidence.values() for v in cases.values())
    target_harm = any(v > .02 + 1e-12 for v in harm.values())
    return dict(valid_evidence=bool(valid), rcp_intersection=sorted(contracts.get(APP, set()) & contracts.get(X08, set())),
        both_disjoint_controls_actuated=bool(valid), independently_effective=individually,
        observed_pattern_consistent_with_m2=bool(valid and all(individually.values()) and target_harm),
        classification='invalid_evidence' if not valid else 'controllers_not_individually_effective' if not all(individually.values()) else
            'disjoint_controls_with_target_harm' if target_harm else 'no_detected_m2_conflict',
        target_harm=harm, factorial_interaction=dict(ap_energy_j=energy['both'] - energy['x02'] - energy['x08'] + energy['off'],
            network_goodput_mbps=rate['both'] - rate['x02'] - rate['x08'] + rate['off']),
        cases=data, controller_evidence=evidence, offline_qos_diagnostics=qos,
        thresholds=dict(individual_objective_gain_fraction=.05, post_combination_target_harm_fraction=.02),
        queue_harq_audit_scope='All offered/initial packet bytes equal delivered+pending+overflow; AP unavailability cannot create grants or RF emission. A separate internal queue-versus-HARQ-reservation identity is not inferred from these records.',
        calibration_not_independent_validation=configs['off']['manifest']['rng_seed'] == 55,
        scope='Matched finite-run, energy-first X02 and throughput-first X08. QoS is fully observed, not guaranteed. Count-pooled network/classes use each original packet deadline. Non-additivity alone does not prove conflict or statistical significance.')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('negative', 'all', 'evaluate'), default='all')
    p.add_argument('--ap-services-profile', type=Path, default=ROOT / 'configs/rt_sys/ap_services_reference.json')
    p.add_argument('--throughput-model', type=Path, default=ROOT / 'outputs/x08_load2x_recovery_20261105/throughput_mlp_model.json')
    p.add_argument('--ap-model', type=Path)
    p.add_argument('--reference-off', type=Path, default=ROOT / 'outputs/scheduler_conflict_case_20261105/seed55_off')
    p.add_argument('--reference-x08', type=Path, default=ROOT / 'outputs/scheduler_conflict_case_20261105/seed55_x08')
    p.add_argument('--train-seconds', type=float, default=4.)
    p.add_argument('--max-train-seconds', type=float, default=6.)
    p.add_argument('--train-seed', type=int, default=181)
    p.add_argument('--eval-seed', type=int, default=55)
    p.add_argument('--eval-seconds', type=float, default=1.)
    p.add_argument('--eval-workers', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--allow-empty-reference-coupling', action='store_true',
        help='Explicit structural empty-cell acceptance; preserves the strict no-harm diagnostic and all UE QoS changes')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def command_for(args, output, model, name, seed, seconds, arm):
    return ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast', '--duration-s', str(seconds),
        '--seed', str(seed), '--traffic-load', '2', '--xapp-pool', 'chongyu', '--ue-services', 'drx-mec',
        '--record-compression', 'gzip', '--ap-services-profile', str(args.ap_services_profile.resolve()),
        '--energy-xapp-mode', 'off', '--latency-xapp-mode', 'off', '--urllc-xapp-mode', 'off', '--budget-xapp-mode', 'off',
        '--throughput-xapp-mode', 'frozen' if arm in ('x08', 'both') else 'off',
        '--throughput-xapp-policy', 'throughput', '--throughput-xapp-model', str(args.throughput_model.resolve()),
        '--ap-xapp-mode', 'fixed' if arm == 'empty_only' else 'online' if arm == 'train' else 'frozen' if arm in ('x02', 'both') else 'off',
        '--ap-xapp-model', str(model), '--ap-xapp-window-ms', '100', '--ap-xapp-fixed-action', '1',
        '--direct-conflict-mitigation', 'priority',
        '--channel-cache-root', str(ROOT / 'channel_cache' / output.name / ('seed_' + str(seed))), '--output', str(output / name)]


def training_lineage(model, x08):
    model = Path(model); receipt_path = model.parent / 'acceptance.json'; r = load(receipt_path)
    if r.get('ap_model_before_evaluation') != model_stamp(model) or r.get('throughput_model_before', {}).get('sha256') != x08['sha256']:
        raise ValueError('Frozen reuse requires the original measured training/model')
    empty_folders = {c['arm']: Path(c['output']) for c in r['cases'] if c['arm'] in ('off', 'empty_only')}
    if set(empty_folders) != {'off', 'empty_only'}: raise ValueError('Original actual empty-control cases are required')
    actual_empty = negative_control(empty_folders)
    acceptance = revalidate_empty_control_acceptance(r, actual_empty)
    p, previous, proof, protected = load(model), None, [], {str(model): sha256(model), str(receipt_path): sha256(receipt_path)}
    for c in [c for c in r['cases'] if c['arm'] in ('off', 'empty_only')]:
        audit = actual_empty['cases'][c['arm']]['audit']
        if audit != c['audit']: raise ValueError('Original empty-control recording/HTML hashes changed')
        protected.update(audit['raw_file_sha256']); protected[audit['html']] = audit['html_sha256']
        for name in ('config_snapshot.json', 'final_summary.json', 'run_status.json'):
            path = Path(c['output']) / name; protected[str(path)] = sha256(path)
    for c in [c for c in r['cases'] if c['arm'] == 'train']:
        folder = Path(c['output']); audit = case_data(folder)['audit']
        if audit != c['audit']: raise ValueError('Original training recording/HTML hashes changed')
        summary = load(folder / 'final_summary.json')['ap_xapp']; n = summary['samples']
        prefix = dict(p, x=p['x'][:n], y=p['y'][:n], trained_samples=summary['trained_samples'], fit_count=summary['fit_count'])
        proof.append(training_evidence(folder, model, previous, prefix)); previous = prefix
        protected.update(audit['raw_file_sha256'])
        protected[audit['html']] = audit['html_sha256']
        for name in ('config_snapshot.json', 'final_summary.json', 'run_status.json'): protected[str(folder / name)] = sha256(folder / name)
    if not proof or len(previous['x']) != len(p['x']) or not readiness(model)['ready']: raise ValueError('Incomplete genuine AP training history')
    return dict(training_receipt=str(receipt_path), collection_folder=next(c['output'] for c in r['cases'] if c['arm'] == 'train'),
        negative_control=actual_empty, empty_control_acceptance=acceptance,
        original_ap_services_profile=r['ap_services_profile'], actual_training_proof=proof, protected_original_inputs=protected,
        collector_source_sha256s=sorted({c['audit']['source_sha256'] for c in r['cases'] if c['arm'] == 'train'}),
        source_scope='Collector source is recorded separately; all new frozen comparison cases share one source. Original models/data remain immutable.')


def main():
    args = parse_args(); output = args.output.resolve()
    model = (args.ap_model or output / 'ap_rf_model.json').resolve(); x08 = args.throughput_model.resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'outputs') or not model.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Fresh study/model paths under project outputs are required')
    if args.phase != 'evaluate' and (model.exists() or not model.is_relative_to(output)):
        raise ValueError('Training requires a fresh model inside the new output')
    if (not 0 < args.train_seconds <= args.max_train_seconds <= 6 or not 0 < args.eval_seconds or
            not all(math.isfinite(v) for v in (args.train_seconds, args.max_train_seconds, args.eval_seconds)) or
            args.eval_seed in (args.train_seed, args.train_seed + 1)):
        raise ValueError('Finite 4+2-second maximum-six budget and distinct train/evaluation seeds required')
    cases = [('seed' + str(args.eval_seed) + '_off', args.eval_seed, args.eval_seconds, 'off')]
    if args.phase != 'evaluate': cases.append(('seed' + str(args.eval_seed) + '_empty_only', args.eval_seed, args.eval_seconds, 'empty_only'))
    if args.phase == 'all':
        cases.append(('online_training', args.train_seed, args.train_seconds, 'train'))
        if args.max_train_seconds > args.train_seconds:
            cases.append(('online_coverage_extension', args.train_seed + 1, args.max_train_seconds - args.train_seconds, 'train'))
    if args.phase != 'negative': cases += [('seed' + str(args.eval_seed) + '_' + a, args.eval_seed, args.eval_seconds, a) for a in ARMS[1:]]
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, commands=[command_for(args, output, model, *c) for c in cases],
            eval_workers=args.eval_workers, shared_off_baseline=True, maximum_training_physics_s=args.max_train_seconds,
            allow_empty_reference_coupling=args.allow_empty_reference_coupling,
            extension_condition='Insufficient genuine global fitted action coverage only; never weak KPI performance',
            no_automatic_extra_seeds_or_longer_trials=True), indent=2)); return
    if not mlp_readiness(x08)['ready']: raise ValueError('Original verified X08 model required')
    for reference in (args.reference_off, args.reference_x08):
        if not (reference / 'final_summary.json').is_file(): raise ValueError('Historical all-active parity references are required before running')
    profile = load(args.ap_services_profile)
    if profile['ap_services'].get('enabled') is not True: raise ValueError('Explicit enabled AP profile required')
    before_x08, frozen, lineage = mlp_stamp(x08), None, None
    if args.phase == 'evaluate':
        lineage = training_lineage(model, before_x08)
        if lineage['original_ap_services_profile'] != profile: raise ValueError('Frozen AP model requires its original measured power/cycle profile')
    output.mkdir(parents=True); model.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(completed=False, status='initializing', phase=args.phase, cases=[], training_evidence=[],
        throughput_model_before=before_x08, ap_model=str(model), ap_services_profile=profile,
        profile_sha256=sha256(args.ap_services_profile), eval_workers=args.eval_workers,
        historical_reference_sha256={str(p / 'final_summary.json'): sha256(p / 'final_summary.json') for p in (args.reference_off, args.reference_x08)},
        train_seed=args.train_seed, eval_seed=args.eval_seed, training_simulated_s=0.,
        maximum_training_simulated_s=args.max_train_seconds, training_lineage=lineage,
        allow_empty_reference_coupling=args.allow_empty_reference_coupling,
        empty_control_protocol_amendment=EMPTY_CONTROL_AMENDMENT if args.allow_empty_reference_coupling else None,
        energy_scope='Modeled AP device energy, distinct from emitted RF and host hardware measurements',
        no_automatic_extra_seeds_or_longer_trials=True)
    if lineage:
        receipt['negative_control'] = lineage['negative_control']
        receipt['empty_control_acceptance'] = lineage['empty_control_acceptance']
        receipt['empty_control_protocol_origin'] = lineage['training_receipt']
    lock = threading.RLock(); active = set(); order = {c[0]: i for i, c in enumerate(cases)}
    def write():
        with lock: atomic_json(output / 'acceptance.json', receipt)
    def protect():
        if sha256(x08) != before_x08['sha256'] or sha256(args.ap_services_profile) != receipt['profile_sha256']:
            raise ValueError('Original X08 model or declared AP profile changed')
        if frozen and sha256(model) != frozen['sha256']: raise ValueError('Frozen AP artifact changed')
        for path, digest in (lineage or {}).get('protected_original_inputs', {}).items():
            if sha256(path) != digest: raise ValueError('Immutable AP training history changed')
        for path, digest in receipt['historical_reference_sha256'].items():
            if sha256(path) != digest: raise ValueError('Historical all-active parity reference changed')
    def run(case):
        name, seed, seconds, arm = case
        with lock:
            active.add(name); receipt.update(status='awaiting_remaining_cases_after_failure' if receipt.get('case_failures') else 'running', running_cases=sorted(active, key=order.get)); write()
        try:
            protect(); began = time.perf_counter(); command = command_for(args, output, model, *case)
            with (output / (name + '.log')).open('w') as log: child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode: raise RuntimeError('Actual case failed: ' + name)
            folder = output / name; data = case_data(folder); protect()
            with lock:
                if lineage and arm == 'off':
                    lineage['compatibility_with_evaluation'] = collection_eval_compatibility(lineage, load(folder / 'config_snapshot.json'))
                    lineage['evaluation_source_sha256'] = data['audit']['source_sha256']
                    lineage['collector_and_evaluation_sources_differ'] = data['audit']['source_sha256'] not in lineage['collector_source_sha256s']
                receipt['cases'].append(dict(name=name, arm=arm, seed=seed, simulated_s=seconds, output=str(folder),
                    audit=data['audit'], command=command, external_wall_s=time.perf_counter() - began,
                    simulation_wall_s=load(folder / 'performance.json')['simulation_s']))
                receipt['cases'].sort(key=lambda c: order[c['name']])
                if len({c['audit']['source_sha256'] for c in receipt['cases']}) != 1: raise ValueError('Study source changed')
                write()
            return folder
        except BaseException as error:
            with lock:
                receipt.setdefault('case_failures', []).append(dict(name=name, type=type(error).__name__, message=str(error)))
                receipt['case_failures'].sort(key=lambda c: order[c['name']]); receipt['status'] = 'awaiting_remaining_cases_after_failure'; write()
            raise
        finally:
            with lock: active.discard(name); receipt['running_cases'] = sorted(active, key=order.get); write()
    write()
    try:
        off = run(cases[0]); receipt['off_historical_parity'] = legacy_parity(off, args.reference_off)
        if args.phase != 'evaluate':
            empty = run(cases[1]); receipt['negative_control'] = negative_control(dict(off=off, empty_only=empty)); write()
            receipt['empty_control_acceptance'] = empty_control_acceptance(receipt['negative_control'], args.allow_empty_reference_coupling); write()
            if not receipt['empty_control_acceptance']['accepted']:
                receipt.update(completed=True, status='negative_control_failed'); write(); return
        if args.phase == 'negative':
            receipt.update(completed=True, status='negative_control_passed' if receipt['negative_control']['passed'] else 'empty_control_structurally_accepted'); write(); return
        if args.phase == 'all':
            previous = None
            for case in [c for c in cases if c[3] == 'train']:
                folder = run(case); receipt['training_simulated_s'] += case[2]
                receipt['training_evidence'].append(training_evidence(folder, model, previous))
                receipt['training_readiness'] = readiness(model); write()
                if receipt['training_readiness']['ready']: break
                previous = load(model)
            if not receipt['training_readiness']['ready']:
                receipt.update(completed=True, status='not_trained', learned_model_claim=False); write(); return
        frozen = model_stamp(model); receipt.update(ap_model_before_evaluation=frozen, learned_model_claim=True); write()
        folders, errors = {'off': off}, []
        with ThreadPoolExecutor(max_workers=args.eval_workers) as pool:
            futures = {pool.submit(run, c): c[3] for c in cases if c[3] in ARMS[1:]}
            for future in as_completed(futures):
                try: folders[futures[future]] = future.result()
                except BaseException as error: errors.append(error)
        if errors: raise RuntimeError('Frozen comparison failed; all submitted cases have been awaited') from errors[0]
        receipt['x08_historical_parity'] = legacy_parity(folders['x08'], args.reference_x08)
        evaluation = evaluate_study({a: folders[a] for a in ARMS}, before_x08, frozen)
        receipt.update(completed=True, status=evaluation['classification'], evaluation=evaluation)
        atomic_json(output / 'comparison-data.json', evaluation); protect(); write()
    except BaseException as error:
        receipt.update(completed=False, status='failed', failure=dict(type=type(error).__name__, message=str(error))); write(); raise
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__': main()
