"""Read saved acceptance data; never run simulation or modify input artifacts."""
import argparse
import ast
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    with path.open() as stream:
        yield from csv.DictReader(stream)


def number(row, key):
    return float(row[key])


def summarize_case(case, audit):
    config = json.loads((case / 'config_snapshot.json').read_text())
    final = json.loads((case / 'final_summary.json').read_text())
    evaluation = json.loads((case / 'evaluation_summary.json').read_text())
    status = json.loads((case / 'run_status.json').read_text())
    assert status['completed'] and status['spark_verified'] and audit['passed']
    configs = {u['ue_id']: u for u in config['configs']['ue']['ues']}
    snapshots = {u['ue_id']: u for u in final['final_per_ue_snapshot']}
    arrivals = defaultdict(int)
    slot_arrivals = {}
    slot_values = defaultdict(list)
    for row in rows(case / 'slot_metrics.csv'):
        slot = int(row['slot_id'])
        values = ast.literal_eval(row['true_traffic_arrival_bytes_by_ue'])
        assert sum(values.values()) == int(row['true_traffic_arrival_bytes_total'])
        slot_arrivals[slot] = values
        for uid, value in values.items():
            arrivals[uid] += value
        for key in ('observable_bler', 'observable_sinr_db', 'scheduler_demand_hol_delay_ms_mean',
                    'true_queue_bytes', 'true_harq_flush_count_total'):
            slot_values[key].append(number(row, key))

    counts = Counter()
    per_ue = {uid: Counter() for uid in configs}
    previous_queue = {uid: u['queue_bytes'] for uid, u in configs.items()}
    windows = defaultdict(Counter)
    positive_sinr = []
    for row in rows(case / 'ue_metrics.csv'):
        slot, uid = int(row['slot_id']), row['ue_id']
        delivered, before = int(row['delivered_bytes']), int(row['queue_bytes'])
        ue = per_ue[uid]
        # Arrivals are reported before the configured queue cap; infer clipping
        # from consecutive pre-PHY queues, then verify the final queue separately.
        clipped = previous_queue[uid] + slot_arrivals[slot][uid] - before
        assert clipped >= 0, (case.name, slot, uid, clipped)
        ue['inferred_queue_cap_discard_bytes'] += clipped
        ue['delivered_bytes'] += delivered
        ue['end_queue_bytes'] = before - delivered
        previous_queue[uid] = before - delivered
        window = 'first_100_slots' if slot <= 100 else 'last_200_slots' if slot > 800 else 'middle_700_slots'
        windows[window]['delivered_bytes'] += delivered
        if row['scheduled'] != 'True':
            continue
        counts['scheduled_grants'] += 1
        ue['scheduled_grants'] += 1
        counts['scheduled_data_re_streams'] += int(row['data_re']) * int(row['layers'])
        counts['physical_capacity_bytes'] += int(row['physical_tb_capacity_bytes'])
        counts['padding_bytes'] += int(row['padding_bytes'])
        ue['physical_capacity_bytes'] += int(row['physical_tb_capacity_bytes'])
        ue['scheduled_data_re_streams'] += int(row['data_re']) * int(row['layers'])
        counts['scheduled_bler_sum'] += number(row, 'bler')
        source = row['ack_source']
        counts[source] += 1
        if source == 'actual_nr_crc_and_bit_check':
            failed = row['reference_tb_ack'] == 'False'
            counts['reference_crc_failures'] += int(failed)
            counts['probe_errors'] += int(row['bler_probe_errors'])
            counts['probe_samples'] += int(row['bler_probe_samples'])
            counts['unsupported_projection_grants'] += int(bool(ast.literal_eval(row['unsupported_nr_mcs'])))
            counts['post_selection_bler_above_target'] += int(number(row, 'bler') > .1)
            positive_sinr.append(number(row, 'raw_sinr_db'))
            ue['reference_attempts'] += 1
            ue['reference_crc_failures'] += int(failed)
        counts['capacity_guard_grants'] += int(row['mac_capacity_guard'] == 'True')
        if row['ack'] == 'False':
            counts['mac_nacks_including_zero_tb'] += 1
            if source == 'no_path':
                reason = 'no_path'
            elif row['reference_tb_ack'] == 'False':
                reason = 'reference_crc_failure'
            elif row['mac_capacity_guard'] == 'True':
                reason = 'capacity_guard_after_reference_success'
            elif int(row['tb_bytes']) == 0:
                reason = 'zero_payload_after_reference_success'
            else:
                reason = 'other'
            counts['mac_nack_' + reason] += 1
            ue['mac_nack_' + reason] += 1
            windows[window]['mac_nack_' + reason] += 1
        counts['scheduled_zero_tb_grants'] += int(int(row['tb_bytes']) == 0)

    assert sum(u['delivered_bytes'] for u in per_ue.values()) == evaluation['delivered_bytes'] == audit['totals']['delivered_bytes']
    assert counts['actual_nr_crc_and_bit_check'] == audit['totals']['nr_reference_attempts']
    assert counts['reference_crc_failures'] == audit['totals']['nr_crc_nacks']
    slices = defaultdict(Counter)
    for uid, ue in per_ue.items():
        initial = configs[uid]['queue_bytes']
        end = int(snapshots[uid]['true_state']['queue_bytes'])
        assert ue['end_queue_bytes'] == end
        assert initial + arrivals[uid] == ue['delivered_bytes'] + end + ue['inferred_queue_cap_discard_bytes']
        ue.update(arrival_bytes=arrivals[uid], initial_queue_bytes=initial,
                  harq_flushes=int(snapshots[uid]['true_state']['harq_flush_count']))
        ue['delivery_fraction_of_offered_bytes'] = ue['delivered_bytes'] / (initial + arrivals[uid])
        ue['slice_id'] = configs[uid]['slice_id']
        bucket = slices[ue['slice_id']]
        for key in ('arrival_bytes', 'initial_queue_bytes', 'delivered_bytes', 'end_queue_bytes',
                    'inferred_queue_cap_discard_bytes', 'harq_flushes', 'scheduled_grants'):
            bucket[key] += ue[key]
    for bucket in slices.values():
        bucket['delivery_fraction_of_offered_bytes'] = bucket['delivered_bytes'] / (bucket['arrival_bytes'] + bucket['initial_queue_bytes'])

    cells = defaultdict(Counter)
    for row in rows(case / 'cell_metrics.csv'):
        cell = cells[row['cell_id']]
        cell['samples'] += 1
        cell['occupied_re'] += int(row['occupied_re'])
        cell['grid_re'] += int(row['grid_re'])
        cell['budget_w_sum'] += number(row, 'budget_w')
        cell['used_w_sum'] += number(row, 'average_used_w')
    for cell in cells.values():
        cell['fft_grid_occupancy_fraction'] = cell['occupied_re'] / cell['grid_re']
        cell['mean_used_power_w'] = cell['used_w_sum'] / cell['samples']
    for name, slots in [('first_100_slots', 100), ('middle_700_slots', 700), ('last_200_slots', 200)]:
        windows[name]['throughput_mbps'] = windows[name]['delivered_bytes'] * 8 / (slots * config['manifest']['slot_duration_ms'] * 1000)
    assert len(slot_arrivals) == config['manifest']['total_slots'] == 1000
    counts['reference_crc_failure_fraction'] = counts['reference_crc_failures'] / counts['actual_nr_crc_and_bit_check']
    counts['independent_probe_failure_fraction'] = counts['probe_errors'] / counts['probe_samples']
    counts['mean_bler_over_scheduled_grants'] = counts['scheduled_bler_sum'] / counts['scheduled_grants']
    return dict(case=case.name, scene=config['manifest']['scene'], ues=len(configs), seed=config['manifest']['rng_seed'],
                throughput_mbps=evaluation['throughput_mbps'], completed_handovers=evaluation['completed_handovers'],
                source_sha256=status['source_sha256'], counts=dict(counts), slices=dict(slices), per_ue=per_ue,
                cells=dict(cells), windows=dict(windows), positive_path_raw_sinr_db_median=statistics.median(positive_sinr),
                mean_observable_bler=statistics.mean(slot_values['observable_bler']),
                mean_observable_sinr_db=statistics.mean(slot_values['observable_sinr_db']),
                mean_scheduler_hol_delay_ms=statistics.mean(slot_values['scheduler_demand_hol_delay_ms_mean']),
                final_snapshot_observable_minus_truth_sinr_db={uid: float(u['observable_state']['sinr_db']) - float(u['true_state']['sinr_db'])
                     for uid, u in snapshots.items() if u['observable_state']['is_scheduled']},
                final_harq_flushes=int(final['final_true_metrics']['harq_flush_count_total']),
                safety_violation_rate=final['experimental_metrics']['safety_violation_rate'],
                safe_baseline_fallback_rate=final['experimental_metrics']['safe_baseline_fallback_rate'],
                mu_resource_groups=audit['totals']['mu_resource_groups'], elapsed_s=status['elapsed_s'],
                input_sha256={f: hashlib.sha256((case / f).read_bytes()).hexdigest() for f in
                              ('config_snapshot.json', 'evaluation_summary.json', 'final_summary.json', 'ue_metrics.csv', 'cell_metrics.csv', 'slot_metrics.csv', 'run_status.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/rt_sys_targeted_matrix_nr'))
    parser.add_argument('--audit', type=Path, default=Path('outputs/rt_sys_targeted_repairs/matrix_audit.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/rt_sys_resume_20261013/analysis'))
    args = parser.parse_args()
    assert not args.output.resolve().is_relative_to(args.root.resolve())
    audit = json.loads(args.audit.read_text())
    assert audit['completed'] and not audit['errors'] and len(audit['cases']) == 12
    cases = [summarize_case(args.root / c['case'], c) for c in audit['cases']]
    groups = []
    for scene in ('simple_street_canyon', 'munich'):
        for count in (12, 24):
            members = [c for c in cases if c['scene'] == scene and c['ues'] == count]
            assert {c['seed'] for c in members} == {1, 2, 3}
            values = [c['throughput_mbps'] for c in members]
            slices = {}
            for name in ('embb', 'urllc', 'voice', 'mmtc'):
                available = [c['slices'][name] for c in members if name in c['slices']]
                if available:
                    sums = {k: sum(b[k] for b in available) for k in ('arrival_bytes', 'initial_queue_bytes', 'delivered_bytes', 'end_queue_bytes', 'inferred_queue_cap_discard_bytes')}
                    sums['delivery_fraction_of_offered_bytes'] = sums['delivered_bytes'] / (sums['arrival_bytes'] + sums['initial_queue_bytes'])
                    slices[name] = sums
            groups.append(dict(scene=scene, ues=count, seeds=3, mean_throughput_mbps=statistics.mean(values),
                               sample_sd_mbps=statistics.stdev(values), min_mbps=min(values), max_mbps=max(values),
                               mean_positive_path_raw_sinr_db_median=statistics.mean(c['positive_path_raw_sinr_db_median'] for c in members),
                               mean_last_200_slots_throughput_mbps=statistics.mean(c['windows']['last_200_slots']['throughput_mbps'] for c in members),
                               mean_fft_grid_occupancy_fraction=statistics.mean(v['fft_grid_occupancy_fraction'] for c in members for v in c['cells'].values()),
                               mean_final_queue_bytes=sum(u['end_queue_bytes'] for c in members for u in c['per_ue'].values()) / 3,
                               slices=slices))
    totals = Counter()
    for c in cases:
        for key in ('scheduled_grants', 'actual_nr_crc_and_bit_check', 'reference_crc_failures', 'no_path',
                    'capacity_guard_grants', 'scheduled_zero_tb_grants', 'mac_nacks_including_zero_tb',
                    'mac_nack_reference_crc_failure', 'mac_nack_no_path', 'mac_nack_capacity_guard_after_reference_success',
                    'mac_nack_zero_payload_after_reference_success', 'mac_nack_other', 'post_selection_bler_above_target',
                    'probe_errors', 'probe_samples', 'unsupported_projection_grants'):
            totals[key] += c['counts'].get(key, 0)
        totals['delivered_bytes'] += sum(u['delivered_bytes'] for u in c['per_ue'].values())
        totals['inferred_queue_cap_discard_bytes'] += sum(u['inferred_queue_cap_discard_bytes'] for u in c['per_ue'].values())
        totals['final_harq_flushes'] += c['final_harq_flushes']
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(completed=True, scope='read-only saved integration matrix; no new long simulations',
                  checks={'audit_totals_match': True, 'per_ue_arrival_delivery_queue_cap_ledger_exact': True,
                          'final_per_ue_queue_match': True},
                  definitions={'inferred_queue_cap_discard_bytes': 'Offered initial queue + uncapped arrivals - delivery - final queue; checked every UE/slot. Inferred byte ledger, not packet-drop timestamps.',
                               'reference_crc_failure_fraction': 'Only actual NR reference executions, excludes no-path and not-scheduled placeholder reference_tb_ack=False.',
                               'occupancy': 'Occupied RE divided by full FFT grid; includes guard subcarriers in denominator.',
                               'throughput_spread': 'Three seeds: sample SD/range, not a generalization confidence interval.'},
                  totals=dict(totals), groups=groups, cases=cases)
    (args.output / 'matrix_analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    # Reproduce the identified counter semantics in an isolated state fixture.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from types import SimpleNamespace
    from ran.scheduler import SimpleScheduler
    from state.harq_state import HarqState, HarqProcessState
    harq = HarqState(processes=[HarqProcessState(process_id=0, pending_bytes=0, timeout_slot=5, retx_deadline_slot=5)])
    SimpleScheduler.__new__(SimpleScheduler)._prune_expired_harq_processes(SimpleNamespace(slot_id=6), SimpleNamespace(harq=harq))
    diagnosis = dict(empty_expired_process_flush_reproduced=harq.harq_flush_count == 1,
                     pending_bytes=0, flush_count=harq.harq_flush_count, flush_reason=harq.last_flush_reason,
                     scope='Isolated in-memory fixture; no matrix or production state changed.')
    assert diagnosis['empty_expired_process_flush_reproduced']
    (args.output / 'empty_harq_flush_diagnosis.json').write_text(json.dumps(diagnosis, indent=2))
    print(json.dumps({'checks': result['checks'], 'totals': result['totals'], 'groups': groups}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
