"""Finite scheduler conflict evidence uses actual packets and executed bundles."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch

from experiments import run_scheduler_conflict_study as study
from ric.near_rt.pools.chongyu.online_packet_rf import FIT_PARAMETERS


def dump(path, value):
    Path(path).write_text(json.dumps(value))


def write_lines(path, values):
    Path(path).write_text(''.join(json.dumps(v) + '\n' for v in values))


def rf_payload(rows=24, fitted=True):
    leaf = dict(children_left=[-1], children_right=[-1], feature=[-2], threshold=[-2.], value=[[0.] * 5])
    x = [[0.] * 8 + [float(i % 4 == j) for j in range(4)] for i in range(rows)]
    return dict(format='chongyu.online-packet-rf', version=1, feature_names=study.RF_FEATURES,
        target_names=study.RF_TARGETS, seed=179, x=x, y=[[0.] * 5 for _ in x],
        forest=[deepcopy(leaf) for _ in range(32)] if fitted else [],
        trained_samples=rows if fitted else 0, fit_count=1 if fitted else 0,
        prediction_batches=0, fit_parameters=FIT_PARAMETERS if fitted else None, fit_s=0., predict_s=0.)


def install_arbitration(folder, agrees=False):
    weights = {u: 1. for u in study.POPULATION}
    requests = {study.X08: dict(scheduler_mode='latency_aware', ue_scheduler_weights=dict(weights, ue_8=2.)),
                study.X12: dict(scheduler_mode='latency_aware', ue_scheduler_weights=dict(weights, ue_6=2.))}
    if agrees:
        requests[study.X12] = deepcopy(requests[study.X08])
    sources = sorted(requests)
    draw = None if agrees else random.Random('1701:direct-conflict-mitigation:scheduler_bundle:cell_0').random()
    winner = sources[0] if agrees else sources[int(draw * 2)]
    row = dict(reason='direct_conflict_request_selection' if agrees else 'direct_conflict_random_selection',
        field='scheduler_bundle', cell_id='cell_0', slot_id=5, window_index=0,
        window_start_slot=0, window_end_slot_exclusive=200, generation=1, mitigation_seed=1701,
        candidates=[dict(source=s, requested_value=requests[s], request_id=s + ':1') for s in sources],
        winner=winner, selected_value=requests[winner], draw=draw, probability=1. if agrees else .5,
        stream_key='scheduler_bundle:cell_0', draw_index=None if agrees else 1)
    record = dict(row, status='applied', reason='direct_conflict_action_executed')
    receipt = dict(status='applied', apply_slot=10, provenance=[dict(implementation_id=winner)],
        fields=[dict(field=f, value=v) for f, v in requests[winner].items()],
        after={'cell_0': requests[winner]}, mitigation_records=[record])
    write_lines(folder / 'rule_decisions.jsonl', [row])
    write_lines(folder / 'control_receipts.jsonl', [receipt])
    return row, receipt


class TestSchedulerConflictTraining(unittest.TestCase):
    def folder(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def fixture(self):
        folder = self.folder()
        membership = {u: 'cell_0' for u in study.POPULATION}
        snapshot = {'cell_0': dict(scheduler_mode='round_robin', ue_scheduler_weights={u: 1. for u in study.POPULATION}),
                    'cell_1': dict(scheduler_mode='round_robin', ue_scheduler_weights={}),
                    'cell_2': dict(scheduler_mode='round_robin', ue_scheduler_weights={})}
        owners = {c: dict(scheduler_mode=study.X12) for c in snapshot}
        owners['cell_0']['ue_scheduler_weights'] = study.X12
        counters = {u: 1 for u in ('ue_2', 'ue_6')}
        zero = {u: 0 for u in counters}
        sample = dict(kind='measured_sample', action=0, features=[.1] + [0.] * 7 + [1., 0., 0., 0.],
            targets=[0., 10., 10., .048, 0.], produced_start_slot=60, produced_end_slot=260, duration_s=.1,
            snapshot=snapshot, membership=membership, delivered_bits_by_ue={u: 800 for u in study.POPULATION},
            offered_bits_by_ue={u: 800 for u in study.POPULATION}, deadline_failed_packets_by_ue=zero,
            overflow_packets_by_ue=zero, completed_packets_by_ue=counters, admitted_packets_by_ue=counters,
            on_time_completed_packets_by_ue=counters, late_completed_packets_by_ue=zero,
            completion_latency_ms_by_ue={u: 1. for u in counters}, completion_mean_ms_by_ue={u: 1. for u in counters},
            original_latency_budgets_ms={u: 10 for u in counters}, target_order=study.RF_TARGETS, retained_for_training=True)
        confirmation = dict(kind='executed_policy_confirmed', action=0, produced_slot=20,
                            executed_snapshot=snapshot, owners=owners)
        events = [confirmation, sample]
        write_lines(folder / 'urllc_rf_events.jsonl', events)
        write_lines(folder / 'control_receipts.jsonl', [dict(status='applied', apply_slot=10,
            provenance=[dict(implementation_id=study.X12)], after=snapshot,
            fields=[dict(field='scheduler_mode', value='round_robin'),
                    dict(field='ue_scheduler_weights', value=snapshot['cell_0']['ue_scheduler_weights'])])])
        packets = []
        for uid in study.POPULATION:
            packets += [dict(kind='arrival', ue_id=uid, packet_id=uid + ':1', slot_id=257, bytes=100, admitted=True),
                        dict(kind='delivery', ue_id=uid, packet_id=uid + ':1', slot_id=259, bytes=100, completed=True, latency_ms=1.)]
        write_lines(folder / 'packet_events.jsonl', packets)
        config = dict(manifest=dict(slot_duration_ms=.5, total_slots=300),
            configs=dict(ue=dict(ues=[dict(ue_id=u, latency_budget_ms=10 if u in counters else 100) for u in study.POPULATION]),
            ric=dict(xapps=[dict(implementation_id=study.X12, parameters=dict(mode='online'))])))
        dump(folder / 'config_snapshot.json', config)
        dump(folder / 'final_summary.json', dict(urllc_xapp=dict(samples=1, trained_samples=0, fit_count=0)))
        model = folder / 'model.json'
        payload = rf_payload(rows=1, fitted=False)
        payload['x'], payload['y'] = [study._f32(sample['features'])], [sample['targets']]
        dump(model, payload)
        return folder, model, events, packets

    def test_actual_packet_labels_and_idle_cell_ack_are_valid(self):
        folder, model, _, _ = self.fixture()
        result = study.training_evidence(folder, model)
        self.assertEqual(result['new_actual_rows'], 1)
        self.assertTrue(result['actual_packet_deltas_and_ack_verified'])

    def test_target_arithmetic_and_packet_event_changes_are_rejected(self):
        for change in ('target', 'raw'):
            with self.subTest(change=change):
                folder, model, events, packets = self.fixture()
                if change == 'target':
                    events[-1]['targets'][0] = 1.
                    write_lines(folder / 'urllc_rf_events.jsonl', events)
                else:
                    packets[-1]['bytes'] = 99
                    write_lines(folder / 'packet_events.jsonl', packets)
                with self.assertRaisesRegex(ValueError, 'arithmetic|raw packet'):
                    study.training_evidence(folder, model)

    def test_occupied_cell_requires_real_weight_ownership(self):
        folder, model, events, _ = self.fixture()
        events[0]['owners']['cell_0'].pop('ue_scheduler_weights')
        write_lines(folder / 'urllc_rf_events.jsonl', events)
        with self.assertRaisesRegex(ValueError, 'ownership/ACK'):
            study.training_evidence(folder, model)

    def test_confirmation_cannot_claim_an_ack_from_the_future(self):
        folder, model, _, _ = self.fixture()
        path = folder / 'control_receipts.jsonl'; receipt = json.loads(path.read_text())
        receipt['apply_slot'] = 21; write_lines(path, [receipt])
        with self.assertRaisesRegex(ValueError, 'ownership/ACK'):
            study.training_evidence(folder, model)

    def test_zero_completions_do_not_receive_a_made_up_mean(self):
        _, _, events, _ = self.fixture()
        row = events[-1]
        row['completed_packets_by_ue'] = {u: 0 for u in ('ue_2', 'ue_6')}
        row['on_time_completed_packets_by_ue'] = {u: 0 for u in ('ue_2', 'ue_6')}
        row['completion_latency_ms_by_ue'] = {u: 0. for u in ('ue_2', 'ue_6')}
        row['completion_mean_ms_by_ue'] = {u: None for u in ('ue_2', 'ue_6')}
        row['targets'] = [0., 0., 0., .048, 0.]
        study._verify_training_label(row)
        row['completion_mean_ms_by_ue']['ue_6'] = 0.
        with self.assertRaisesRegex(ValueError, 'undefined'):
            study._verify_training_label(row)

    def test_real_float32_feature_prefix_and_old_rows_are_immutable(self):
        folder, model, _, _ = self.fixture()
        p = json.loads(model.read_text())
        p['x'][0][0] = .1
        dump(model, p)
        with self.assertRaisesRegex(ValueError, 'Persisted'):
            study.training_evidence(folder, model)
        with self.assertRaisesRegex(ValueError, 'historical'):
            study.training_evidence(folder, model, previous=dict(x=[[2.] * 12], y=[[0.] * 5]))

    def test_readiness_requires_fitted_six_per_actual_action(self):
        folder = self.folder(); model = folder / 'model.json'
        payload = rf_payload(); dump(model, payload)
        self.assertTrue(study.rf_readiness(model)['ready'])
        payload['x'][-1][-4:] = [1., 0., 0., 0.]; dump(model, payload)
        self.assertFalse(study.rf_readiness(model)['ready'])
        payload = rf_payload(rows=1, fitted=False); dump(model, payload)
        self.assertFalse(study.rf_readiness(model)['ready'])

    def test_training_dry_run_is_isolated_and_never_starts_an_evaluation(self):
        root = self.folder().resolve()
        args = study.parse_args(['--output', str(root / 'outputs/new'), '--phase', 'train',
            '--throughput-model', str(root / 'outputs/x08/model.json'), '--dry-run'])
        stream = io.StringIO()
        with patch.object(study, 'ROOT', root), patch.object(study, 'parse_args', return_value=args), \
                patch.object(study.subprocess, 'run') as jobs, redirect_stdout(stream):
            study.main()
        plan = json.loads(stream.getvalue())
        self.assertEqual(len(plan['commands']), 2)
        for command in plan['commands']:
            self.assertEqual(command[command.index('--throughput-xapp-mode') + 1], 'off')
            self.assertEqual(command[command.index('--urllc-xapp-mode') + 1], 'online')
            self.assertEqual(command[command.index('--direct-conflict-mitigation') + 1], 'priority')
            self.assertEqual(command[command.index('--traffic-load') + 1], '2')
        self.assertFalse(plan['automatic_longer_or_additional_seed_runs'])
        jobs.assert_not_called()


class TestSchedulerConflictEvaluation(unittest.TestCase):
    def fixture(self, harm=True, agrees=False):
        from tests import test_throughput_mlp_study as prior
        helper = prior.TestThroughputMLPStudy()
        self.addCleanup(helper.doCleanups)
        # Classification fixtures isolate packet/actuation evidence; strict
        # numerical model replay is covered independently below.
        replay = patch.object(study, 'prediction_evidence', side_effect=lambda events, app, stamp, path:
            dict(valid=True, replayed_frozen_decisions=2 if app == study.X08 else 12, errors=[]))
        replay.start(); self.addCleanup(replay.stop)
        old, x08_stamp = helper.fixture(policy='throughput')
        folders = {a: old[b] for a, b in dict(off='off', x08='mlp', urllc='neutral', both_random='fixed').items()}
        rf_stamp = dict(sha256='urllc-frozen-bytes', samples=24, trained_samples=24,
                        fit_count=1, prediction_batches=0, fit_parameters=FIT_PARAMETERS)
        for arm, folder in folders.items():
            config = json.loads((folder / 'config_snapshot.json').read_text())
            config['source_sha256'] = 'e' * 64
            config['configs']['ric'].update(direct_conflict_mitigation='uniform_random' if arm == 'both_random' else 'priority',
                mitigation_seed=1701, mitigation_window_ms=100)
            config['configs']['ric']['xapps'] = [dict(implementation_id=app,
                parameters=dict(mode='frozen', policy='throughput') if app == study.X08 else dict(mode='frozen'))
                for app in ({'off': [], 'x08': [study.X08], 'urllc': [study.X12], 'both_random': [study.X08, study.X12]}[arm])]
            dump(folder / 'config_snapshot.json', config)
            dump(folder / 'run_status.json', dict(completed=True, source_sha256='e' * 64,
                replay=dict(audits_passed=True, html='network-replay.html')))
            (folder / 'network-replay.html').write_text('<html>Unit fixture</html>')
            summary = json.loads((folder / 'final_summary.json').read_text())
            summary.pop('throughput_xapp', None)
            instances = []
            for app in ({'off': [], 'x08': [study.X08], 'urllc': [study.X12], 'both_random': [study.X08, study.X12]}[arm]):
                instances.append(dict(implementation_id=app, active_cell_ids=['cell_0'], writable_actions=sorted(study.FIELDS)))
                learning = dict(x08_stamp if app == study.X08 else rf_stamp, mode='frozen',
                    prediction_batches=12, model_sha256=(x08_stamp if app == study.X08 else rf_stamp)['sha256'])
                if app == study.X08:
                    learning.update(policy='throughput', constraints_active=False)
                summary[study.SUMMARIES[app]] = learning
                write_lines(folder / (study.LOGS[app] + '.jsonl'), [dict(kind='decision'), dict(kind='executed_policy_confirmed')])
            summary['xapp_library']['instances'] = instances
            if arm == 'urllc':
                receipts = [json.loads(r) for r in (folder / 'control_receipts.jsonl').read_text().splitlines()]
                for receipt in receipts:
                    receipt['provenance'] = [dict(implementation_id=study.X12)]
                write_lines(folder / 'control_receipts.jsonl', receipts)
            if arm in ('x08', 'both_random') and harm:
                import gzip
                with gzip.open(folder / 'packet_events.jsonl.gz', 'rt') as stream:
                    packets = [json.loads(r) for r in stream]
                for event in packets:
                    if event['kind'] == 'delivery' and event['packet_id'] == 'ue_6:1':
                        event.update(slot_id=41, latency_ms=20.)
                packets.append(dict(kind='deadline_miss', ue_id='ue_6', packet_id='ue_6:1', slot_id=25))
                with gzip.open(folder / 'packet_events.jsonl.gz', 'wt') as stream:
                    stream.write(''.join(json.dumps(r) + '\n' for r in packets))
                summary['packet_metrics']['ue_6']['completion_latency_ms_sum'] += 19.
                summary['packet_metrics']['ue_6']['deadline_missed_packets'] = 1
            dump(folder / 'final_summary.json', summary)
        install_arbitration(folders['both_random'], agrees=agrees)
        return folders, x08_stamp, rf_stamp

    def test_native_zero_failure_is_preserved_not_claimed_as_improvement(self):
        result = study.evaluate_study(*self.fixture())
        self.assertTrue(result['valid_evidence'])
        self.assertTrue(result['structural_direct_conflict_observed'])
        self.assertEqual(result['urllc_ue6_failure_improvement_vs_off_pp'], 0.)
        self.assertTrue(result['standalone_targets_met']['urllc'])
        self.assertTrue(result['useful_urllc_protection_vs_x08'])
        self.assertEqual(result['urllc_protection_vs_x08_pp'], 100.)
        self.assertEqual(result['classification'], 'direct_conflict_observed_with_target_harm')
        self.assertTrue(result['target_harm']['urllc_target_harmed'])
        self.assertFalse(result['offline_qos_guards']['both_random']['passed'])
        self.assertEqual(result['arms']['both_random']['users']['ue_6']['unique_deadline_failed_packets'], 1)

    def test_finite_random_sequence_need_not_execute_both_contenders(self):
        folders, _, _ = self.fixture()
        evidence = study.arbitration_evidence(folders['both_random'], 1701)
        self.assertTrue(evidence['valid'])
        self.assertFalse(evidence['both_contenders_executed'])
        self.assertEqual(evidence['unequal_conflict_epochs'], 1)
        self.assertTrue(evidence['seeded_rng_replay_passed'])

    def test_equal_whole_requests_are_not_a_direct_conflict(self):
        result = study.evaluate_study(*self.fixture(harm=False, agrees=True))
        self.assertTrue(result['valid_evidence'])
        self.assertFalse(result['structural_direct_conflict_observed'])
        self.assertEqual(result['classification'], 'no_observed_direct_conflict')
        self.assertFalse(result['useful_urllc_protection_vs_x08'])

    def test_blended_execution_cannot_pass_as_a_selected_bundle(self):
        folders, _, _ = self.fixture()
        path = folders['both_random'] / 'control_receipts.jsonl'
        receipt = json.loads(path.read_text())
        receipt['after']['cell_0']['ue_scheduler_weights']['ue_1'] = 1.5
        write_lines(path, [receipt])
        self.assertFalse(study.arbitration_evidence(folders['both_random'], 1701)['valid'])

    def test_random_seed_and_canonical_draw_are_audited(self):
        folders, _, _ = self.fixture()
        evidence = study.arbitration_evidence(folders['both_random'], 1702)
        self.assertFalse(evidence['valid'])
        self.assertTrue(any('RNG' in e or 'window' in e for e in evidence['errors']))

    def test_different_phy_source_or_loaded_model_is_rejected(self):
        folders, x08, rf = self.fixture()
        path = folders['urllc'] / 'config_snapshot.json'
        config = json.loads(path.read_text()); config['configs']['phy']['changed'] = True; dump(path, config)
        with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
            study.evaluate_study(folders, x08, rf)
        config['configs']['phy'].pop('changed'); dump(path, config)
        path = folders['urllc'] / 'final_summary.json'; summary = json.loads(path.read_text())
        summary['urllc_xapp']['model_sha256'] = 'not-frozen-model'; dump(path, summary)
        self.assertFalse(study.evaluate_study(folders, x08, rf)['valid_evidence'])

    def test_packet_trace_totals_must_conserve_actual_bytes(self):
        folders, x08, rf = self.fixture()
        path = folders['off'] / 'final_summary.json'; summary = json.loads(path.read_text())
        summary['packet_metrics']['ue_0']['pending_bytes'] += 1; dump(path, summary)
        with self.assertRaisesRegex(ValueError, 'conservation'):
            study.evaluate_study(folders, x08, rf)


class TestSeparateTrainingLineage(unittest.TestCase):
    def fixture(self):
        helper = TestSchedulerConflictTraining()
        self.addCleanup(helper.doCleanups)
        folder, model, initial, packets = helper.fixture()
        events, raw, rows, labels, receipts = [], [], [], [], []
        for i in range(24):
            confirm, row = deepcopy(initial)
            action, shift = i % 4, i * 300
            mode = 'round_robin' if action == 0 else 'latency_aware'
            for snapshot in (confirm['executed_snapshot'], row['snapshot']):
                for cell in snapshot.values():
                    cell['scheduler_mode'] = mode
                    for uid in cell['ue_scheduler_weights']:
                        cell['ue_scheduler_weights'][uid] = (1.5 if action == 2 else 2. if action == 3 else 1.) if uid == 'ue_6' else 1.
            confirm.update(action=action, produced_slot=20 + shift)
            row.update(action=action, produced_start_slot=60 + shift, produced_end_slot=260 + shift,
                       features=[.1] + [0.] * 7 + [float(action == j) for j in range(4)])
            events += [confirm, row]
            receipts.append(dict(status='applied', apply_slot=10 + shift,
                provenance=[dict(implementation_id=study.X12)], after=deepcopy(row['snapshot']),
                fields=[dict(field='scheduler_mode', value=mode),
                        dict(field='ue_scheduler_weights', value=deepcopy(row['snapshot']['cell_0']['ue_scheduler_weights']))]))
            rows.append(study._f32(row['features'])); labels.append(row['targets'])
            for packet in deepcopy(packets):
                packet['slot_id'] += shift
                packet['packet_id'] += ':' + str(i)
                raw.append(packet)
        events.append(dict(kind='rf_fit', samples=24, fit_count=1, fit_parameters=FIT_PARAMETERS))
        write_lines(folder / 'urllc_rf_events.jsonl', events)
        write_lines(folder / 'control_receipts.jsonl', receipts)
        write_lines(folder / 'packet_events.jsonl', raw)
        config = json.loads((folder / 'config_snapshot.json').read_text())
        config.update(source_sha256='c' * 64); config['manifest']['total_slots'] = 7500
        dump(folder / 'config_snapshot.json', config)
        dump(folder / 'final_summary.json', dict(urllc_xapp=dict(samples=24, trained_samples=24, fit_count=1)))
        payload = rf_payload(); payload.update(x=rows, y=labels); dump(model, payload)
        dump(folder / 'run_status.json', dict(completed=True, source_sha256='c' * 64,
            replay=dict(audits_passed=True, html='network-replay.html')))
        (folder / 'network-replay.html').write_text('<html>Original collector fixture</html>')
        x08 = dict(sha256='immutable-existing-x08')
        receipt = dict(completed=True, status='trained_no_evaluation', urllc_model_before_evaluation=study.rf_stamp(model),
            throughput_model_before=x08, training_simulated_s=3.75,
            cases=[dict(arm='train', output=str(folder), audit=study.completed_case(folder))])
        dump(folder / 'acceptance.json', receipt)
        return folder, model, x08

    def test_completed_real_training_can_be_linked_without_identical_evaluation_source(self):
        folder, model, x08 = self.fixture()
        original = {p: p.read_bytes() for p in folder.iterdir() if p.is_file()}
        lineage = study.trained_lineage(model, x08)
        self.assertEqual(lineage['collector_source_sha256s'], ['c' * 64])
        self.assertIn('explicitly separate', lineage['source_scope'])
        self.assertEqual(lineage['measured_online_episodes'][0]['actual_training_evidence']['new_actual_rows'], 24)
        study.protect_lineage(lineage)
        self.assertEqual(original, {p: p.read_bytes() for p in folder.iterdir() if p.is_file()})

    def test_altered_original_model_or_incomplete_collector_cannot_be_reused(self):
        for change in ('model', 'status', 'recording', 'x08'):
            with self.subTest(change=change):
                folder, model, x08 = self.fixture()
                if change == 'model':
                    payload = json.loads(model.read_text()); payload['y'][0][0] = 1.; dump(model, payload)
                elif change == 'status':
                    status = json.loads((folder / 'run_status.json').read_text()); status['completed'] = False
                    dump(folder / 'run_status.json', status)
                elif change == 'recording':
                    (folder / 'packet_events.jsonl').write_text((folder / 'packet_events.jsonl').read_text() + '\n')
                else:
                    x08 = dict(sha256='different-x08-model')
                with self.assertRaises(ValueError):
                    study.trained_lineage(model, x08)

    def test_protected_parent_artifacts_remain_immutable_after_link(self):
        folder, model, x08 = self.fixture()
        lineage = study.trained_lineage(model, x08)
        (folder / 'network-replay.html').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Historical'):
            study.protect_lineage(lineage)


class TestFrozenForecastReplay(unittest.TestCase):
    def folder(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def rf_fixture(self):
        folder = self.folder(); model = folder / 'rf.json'
        payload = rf_payload()
        tree = dict(children_left=[1, -1, -1], children_right=[2, -1, -1], feature=[10, -2, -2],
            threshold=[.5, -2., -2.], value=[[0.] * 5, [10., 100., 10., 1., 0.], [0., 100., 10., 1., 0.]])
        payload['forest'] = [deepcopy(tree) for _ in range(32)]; dump(model, payload)
        from ric.near_rt.pools.chongyu.online_packet_rf import OnlinePacketRF
        learner = OnlinePacketRF.load(model, study.RF_FEATURES)
        predictions = learner.predict([[0.] * 8 + [float(i == j) for j in range(4)] for i in range(4)])
        event = dict(kind='decision', status='rf', action=2, features=[0.] * 8 + [0., 0., 1., 0.],
            prediction=predictions[2], prediction_audit=dict(raw_predictions=predictions, selected_action=2))
        return model, study.rf_stamp(model), event

    def test_rf_forecasts_and_nonbaseline_choice_replay_from_frozen_bytes(self):
        model, stamp, event = self.rf_fixture(); before = model.read_bytes()
        replay = study.prediction_evidence([event], study.X12, stamp, model)
        self.assertTrue(replay['valid'], replay)
        self.assertEqual(replay['selected_action_counts'], {'2': 1})
        self.assertEqual(before, model.read_bytes())

    def test_forecasts_or_fixed_action_cannot_masquerade_as_rf_selection(self):
        for corruption in ('forecasts', 'choice'):
            with self.subTest(corruption=corruption):
                model, stamp, event = self.rf_fixture()
                if corruption == 'forecasts':
                    event['prediction_audit']['raw_predictions'][0][0] += 1.
                else:
                    event.update(action=3, features=[0.] * 8 + [0., 0., 0., 1.],
                                 prediction=event['prediction_audit']['raw_predictions'][3])
                    event['prediction_audit']['selected_action'] = 3
                self.assertFalse(study.prediction_evidence([event], study.X12, stamp, model)['valid'])

    def test_throughput_mlp_argmax_and_one_percent_margin_replay(self):
        from ric.near_rt.pools.chongyu.online_throughput_mlp import OnlineThroughputMLP, parameters
        from tests.test_throughput_mlp_study import optimizer_proof
        model = self.folder() / 'mlp.json'
        learner = OnlineThroughputMLP(study.X08_FEATURES, study.X08_TARGETS, 173)
        for i in range(24):
            learner.add([0.] * 8 + [float(i % 4 == j) for j in range(4)], [0.] * 9)
        learner.coefs = [[[0.] * 8 for _ in range(12)], [[0.] * 9 for _ in range(8)]]
        learner.coefs[0][11][0] = 1.; learner.coefs[1][0][0] = 10.
        learner.intercepts = [[0.] * 8, [10.] + [1.] * 6 + [0.] * 2]
        learner.x_scaler = dict(mean=[0.] * 12, scale=[1.] * 12)
        learner.y_scaler = dict(mean=[0.] * 9, scale=[1.] * 9)
        learner.fit_parameters = parameters(173); learner.optimizer_metadata = optimizer_proof()
        learner.fit_diagnostics = dict(n_iter=1, loss=0., converged=True)
        learner.trained_samples = 24; learner.fit_count = 1
        learner.save(model)  # Test-only coefficients; no optimizer/fit is run.
        before = model.read_bytes()
        predictions = learner.predict([[0.] * 8 + [float(i == j) for j in range(4)] for i in range(4)])
        event = dict(kind='decision', status='mlp', action=3, features=[0.] * 8 + [0., 0., 0., 1.],
            prediction=predictions[3], policy='throughput', constraints_active=False,
            prediction_guard_audit=dict(raw_predictions=predictions, rejected_actions={}))
        result = study.prediction_evidence([event], study.X08, study.throughput_stamp(model), model)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['selected_action_counts'], {'3': 1})
        self.assertEqual(before, model.read_bytes())
        self.assertEqual(study.selected_action(study.X08, [[10.] * 9, [10.05] * 9, [9.] * 9, [9.] * 9]), 0)

    def test_rf_peer_guard_and_missing_completions_follow_the_declared_rule(self):
        predictions = [[10., 10., 1., 10., 0.], [0., 1., 1., 20., 1.],
                       [0., 0., 0., 20., 0.], [0., 5., 1., 10., 0.]]
        # Action1 is excluded by UE2; action2 has no completed mean, so action3
        # has the best observed-completion mean among lowest-failure finalists.
        self.assertEqual(study.selected_action(study.X12, predictions), 3)


class TestParallelSchedulerEvaluation(unittest.TestCase):
    def cases(self):
        return [('seed55_' + arm, 55, 1., arm) for arm in study.ARMS]

    def test_off_finishes_first_worker_limit_and_stable_case_order(self):
        for workers in (1, 2, 3):
            with self.subTest(workers=workers):
                lock, off_done = threading.Lock(), threading.Event()
                gate = threading.Barrier(workers)
                state = dict(active=0, maximum=0, started=0)
                def runner(case):
                    arm = case[3]
                    if arm == 'off':
                        off_done.set()
                        return 'off-result'
                    self.assertTrue(off_done.is_set())
                    with lock:
                        state['active'] += 1; state['started'] += 1
                        state['maximum'] = max(state['maximum'], state['active'])
                        first_group = state['started'] <= workers
                    if first_group:
                        gate.wait(timeout=2)
                    with lock:
                        state['active'] -= 1
                    return arm + '-result'
                result = study.run_evaluation_cases(self.cases(), runner, workers)
                self.assertEqual(list(result), list(study.ARMS))
                self.assertEqual(state['maximum'], workers)
                self.assertEqual(state['active'], 0)

    def test_failure_waits_for_every_submitted_case_and_off_failure_starts_none(self):
        done, lock = set(), threading.Lock()
        gate = threading.Barrier(3)
        def runner(case):
            arm = case[3]
            if arm == 'off':
                return 'off'
            gate.wait(timeout=2)
            with lock:
                done.add(arm)
            if arm == 'x08':
                raise ValueError('intentional fake failure')
            return arm
        with self.assertRaisesRegex(RuntimeError, 'awaiting all submitted cases'):
            study.run_evaluation_cases(self.cases(), runner, 3)
        self.assertEqual(done, {'x08', 'urllc', 'both_random'})
        called = []
        def off_failure(case):
            called.append(case[3]); raise ValueError('failed warmup')
        with self.assertRaisesRegex(ValueError, 'warmup'):
            study.run_evaluation_cases(self.cases(), off_failure, 3)
        self.assertEqual(called, ['off'])

    def test_main_parallel_receipts_are_atomic_sorted_and_fail_closed(self):
        for failing in (False, True):
            with self.subTest(failing=failing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); models = root / 'outputs/models'; models.mkdir(parents=True)
                x08, rf = models / 'x08.json', models / 'rf.json'
                dump(x08, {}); dump(rf, {})
                output = root / 'outputs/new'
                args = study.parse_args(['--output', str(output), '--phase', 'evaluate', '--eval-workers', '3',
                    '--throughput-model', str(x08), '--urllc-model', str(rf)])
                gate = threading.Barrier(3); off_done = threading.Event(); snapshots = []
                original_atomic = study.atomic_json
                def atomic(path, value):
                    snapshot = deepcopy(value)
                    original_atomic(path, value)
                    self.assertEqual(json.loads(Path(path).read_text()), snapshot)
                    snapshots.append(snapshot)
                def child(command, **kwargs):
                    folder = Path(command[command.index('--output') + 1]); folder.mkdir()
                    arm = next(a for a in study.ARMS if folder.name == 'seed55_' + a)
                    if arm == 'off':
                        off_done.set()
                    else:
                        self.assertTrue(off_done.is_set()); gate.wait(timeout=2)
                    dump(folder / 'performance.json', dict(simulation_s=.01))
                    return type('FakeChild', (), dict(returncode=int(failing and arm == 'x08')))()
                with patch.object(study, 'ROOT', root), patch.object(study, 'parse_args', return_value=args), \
                        patch.object(study, 'throughput_readiness', return_value=dict(ready=True)), \
                        patch.object(study, 'rf_readiness', return_value=dict(ready=True)), \
                        patch.object(study, 'throughput_stamp', return_value=dict(sha256=study.sha256(x08))), \
                        patch.object(study, 'rf_stamp', return_value=dict(sha256=study.sha256(rf))), \
                        patch.object(study, 'trained_lineage', return_value={}), \
                        patch.object(study, 'completed_case', return_value=dict(source_sha256='fixed-source')), \
                        patch.object(study, 'evaluate_study', return_value=dict(classification='fake_valid')) as evaluate, \
                        patch.object(study.subprocess, 'run', side_effect=child), patch.object(study, 'atomic_json', side_effect=atomic), \
                        redirect_stdout(io.StringIO()):
                    if failing:
                        with self.assertRaisesRegex(RuntimeError, 'awaiting all submitted cases'):
                            study.main()
                    else:
                        study.main()
                receipt = json.loads((output / 'acceptance.json').read_text())
                self.assertEqual(receipt['running_cases'], [])
                self.assertTrue(any(len(s.get('running_cases', [])) == 3 for s in snapshots))
                self.assertEqual([c['arm'] for c in receipt['cases']],
                                 ['off', 'urllc', 'both_random'] if failing else list(study.ARMS))
                self.assertEqual(receipt['completed'], not failing)
                self.assertEqual(receipt['status'], 'failed' if failing else 'fake_valid')
                self.assertEqual(evaluate.call_count, 0 if failing else 1)
                self.assertFalse((output / 'acceptance.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
