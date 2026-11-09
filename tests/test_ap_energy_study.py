"""AP study audits actual power ledgers, global samples and pooled packets."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import run_ap_energy_study as s
from phy.rt_sys.ap_services import ApServices
from ric.near_rt.pools.chongyu.online_ap_rf import OnlineApRF, FIT_PARAMETERS


def dump(path, value): Path(path).write_text(json.dumps(value))
def lines(path, rows): Path(path).write_text(''.join(json.dumps(r) + '\n' for r in rows))


def profile():
    return dict(enabled=True, power=dict(static_w=10., load_w=2., pa_efficiency=.5, waking_w=8.,
        sleep_w=dict(micro_sleep=1., light_sleep=.5, deep_sleep=.1),
        wake_transition_j=dict(micro_sleep=0., light_sleep=.02, deep_sleep=.1)),
        profiles=dict(micro_sleep=dict(period_ms=2., sleep_ms=.5, wakeup_ms=0.),
            light_sleep=dict(period_ms=4., sleep_ms=1., wakeup_ms=1.), deep_sleep=dict(wakeup_ms=1.)))


def payload(rows=1, fitted=False):
    leaf = dict(children_left=[-1], children_right=[-1], feature=[-2], threshold=[-2.], value=[[35., .048, 0., 0.]])
    return dict(format='chongyu.online-ap-rf', version=1, feature_names=s.FEATURES, target_names=s.TARGETS,
        seed=181, x=[[0.] * 8 + [float(i % 4 == j) for j in range(4)] for i in range(rows)],
        y=[[35., .048, 0., 0.] for _ in range(rows)], forest=[deepcopy(leaf) for _ in range(32)] if fitted else [],
        trained_samples=rows if fitted else 0, fit_count=1 if fitted else 0, prediction_batches=0,
        fit_parameters=FIT_PARAMETERS if fitted else None, fit_s=0., predict_s=0.)


class TestApEnergyStudy(unittest.TestCase):
    def folder(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        return Path(temp.name).resolve()

    def fixture(self, action=0, mode='online', total=300):
        folder = self.folder()
        cells = {c: SimpleNamespace(cell=SimpleNamespace(cell_sleep_mode='active')) for c in ('cell_0', 'cell_1', 'cell_2')}
        state = SimpleNamespace(managed_cells=cells, slot_id=0)
        ap = ApServices(state, profile(), .5)
        membership = {u: 'cell_0' if u in ('ue_0', 'ue_1', 'ue_2') else 'cell_2' for u in s.POPULATION}
        snapshot = s._action_map(membership, set(cells), action)
        trace, receipt = [], None
        for slot in range(1, total + 1):
            state.slot_id = slot; ap.before_scheduling()
            if slot == 10:
                for cid, policy in snapshot.items(): ap.set_policy(cid, policy, slot, owner=s.APP, request_id=cid + ':1')
                receipt = dict(status='applied', apply_slot=slot, provenance=[dict(implementation_id=s.APP)],
                    after={c: dict(ap.policy_receipt(c), cell_sleep_mode=snapshot[c]) for c in cells},
                    fields=[dict(field='cell_sleep_mode', value='active')])
            radio = {c: dict(average_used_w=1. if c != 'cell_1' and r.cell.ap_available else 0.,
                occupied_re=1 if c != 'cell_1' and r.cell.ap_available else 0, active_grid_re=4) for c, r in cells.items()}
            row = dict(slot_id=slot, cells=radio, ues=[dict(ue_id=u, cell_id=membership[u], scheduled=False) for u in sorted(s.POPULATION)],
                traffic_arrivals={u: dict(exogenous_bytes=100 if slot == 201 else 0, mec_result_bytes=0) for u in s.POPULATION})
            ap.after_phy(row); trace.append(deepcopy(row))
        lines(folder / 'rt_sys_trace.jsonl', trace)
        profiles = [dict(ue_id=u, slice_id='urllc' if u in ('ue_2', 'ue_6') else 'embb',
            latency_budget_ms=10 if u in ('ue_2', 'ue_6') else 100) for u in sorted(s.POPULATION)]
        config = dict(source_sha256='a' * 64,
            manifest=dict(ues=[dict(ue_id=u) for u in sorted(s.POPULATION)], cells=[dict(cell_id=c) for c in cells],
                total_slots=total, slot_duration_ms=.5, rng_seed=55),
            configs=dict(ue=dict(ues=profiles), phy=dict(rt_sys=dict(ap_services=profile())), sim={}, cell={}, beamforming={},
                ric=dict(direct_conflict_mitigation='priority', xapps=[dict(implementation_id=s.APP,
                    parameters=dict(mode=mode, fixed_action=1, model_path=str(folder / 'model.json')))])))
        packets = []
        for u in s.POPULATION:
            packets += [dict(kind='arrival', ue_id=u, packet_id=u + ':1', slot_id=201, bytes=100, admitted=True),
                        dict(kind='delivery', ue_id=u, packet_id=u + ':1', slot_id=205, bytes=100, completed=True, latency_ms=2.)]
        lines(folder / 'packet_events.jsonl', packets)
        packet_metrics = {u: dict(initial_bytes=0, offered_bytes=100, delivered_bytes=100,
            overflow_bytes=0, overflow_packets=0, admitted_packets=1, completed_packets=1,
            pending_bytes=0, pending_packets=0, completion_latency_ms_sum=2., deadline_missed_packets=0) for u in s.POPULATION}
        summary = dict(ap_services=ap.summary(), ap_xapp=dict(samples=1, trained_samples=0, fit_count=0, mode=mode),
            packet_metrics=packet_metrics, xapp_library=dict(instances=[dict(implementation_id=s.APP,
                writable_actions=['cell_sleep_mode'], active_cell_ids=list(cells))]), ue_services={}, tb_outcomes={})
        dump(folder / 'config_snapshot.json', config); dump(folder / 'final_summary.json', summary)
        dump(folder / 'run_status.json', dict(completed=True, source_sha256='a' * 64, replay=dict(audits_passed=True, html='network-replay.html')))
        (folder / 'network-replay.html').write_text('<html>Unit fixture</html>')
        # Per-cell actual requested AP policies; never forge a full-map scalar.
        receipts = []
        for cid in cells:
            r = deepcopy(receipt); r['after'] = {cid: receipt['after'][cid]}
            r['fields'] = [dict(field='cell_sleep_mode', value=snapshot[cid])]; receipts.append(r)
        lines(folder / 'control_receipts.jsonl', receipts); lines(folder / 'rule_decisions.jsonl', [])
        begin, end = trace[59]['ap_services']['cells'], trace[259]['ap_services']['cells']
        counter_maps = {'ap_energy_j_by_cell': 'ap_energy_total_j', 'ap_available_slots_by_cell': 'ap_available_slots_total',
            'ap_sleep_slots_by_cell': 'ap_sleep_slots_total', 'ap_waking_slots_by_cell': 'ap_waking_slots_total'}
        sample = dict(kind='measured_sample', action=action, features=[.1] + [0.] * 7 + [float(action == i) for i in range(4)],
            targets=[sum(end[c]['ap_energy_total_j'] - begin[c]['ap_energy_total_j'] for c in cells) / .1, .048, 0., 0.],
            original_latency_budgets_ms={'ue_2': 10, 'ue_6': 10},
            target_order=s.TARGETS, produced_start_slot=60, produced_end_slot=260, duration_s=.1,
            snapshot=snapshot, membership=membership, actual_policy_generations={c: begin[c]['ap_policy_generation'] for c in cells},
            cycle_period_slots_by_cell={c: 8 for c in cells}, start_cycle_phase_by_cell={c: 3 for c in cells},
            end_cycle_phase_by_cell={c: 3 for c in cells}, delivered_bits_by_ue={u: 800 for u in s.POPULATION},
            offered_bits_by_ue={u: 800 for u in s.POPULATION}, deadline_failed_packets_by_ue={u: 0 for u in ('ue_2', 'ue_6')},
            overflow_packets_by_ue={u: 0 for u in ('ue_2', 'ue_6')}, retained_for_training=True)
        for name, key in counter_maps.items(): sample[name] = {c: end[c][key] - begin[c][key] for c in cells}
        confirmation = dict(kind='executed_policy_confirmed', action=action, produced_slot=20, executed_snapshot=snapshot,
            actual_owners={c: s.APP for c in cells}, actual_applied_slots={c: 10 for c in cells}, generations=sample['actual_policy_generations'])
        events = [confirmation, sample]; lines(folder / 'ap_rf_events.jsonl', events)
        model = folder / 'model.json'; p = payload()
        p['x'] = [[struct.unpack('f', struct.pack('f', v))[0] for v in sample['features']]]; p['y'] = [sample['targets']]; dump(model, p)
        return folder, model, config, summary, trace, events

    def test_actual_energy_and_wake_components_reconstruct(self):
        for action in (0, 1, 2, 3):
            with self.subTest(action=action):
                folder, _, config, summary, _, _ = self.fixture(action=action)
                result = s.ap_energy_audit(folder, config, summary)
                self.assertTrue(result['actual_slot_energy_reconstruction_verified'])
                self.assertGreater(result['network_ap_energy_j'], result['emitted_rf_energy_j'])
                if action == 3: self.assertGreater(result['components_j']['cell_0']['transition'], 0.)

    def test_unavailable_grant_or_changed_component_fails(self):
        for change in ('grant', 'energy'):
            folder, _, config, summary, trace, _ = self.fixture(action=1)
            if change == 'grant': trace[15]['cells']['cell_1']['average_used_w'] = 1.
            else: trace[15]['ap_services']['cells']['cell_1']['slot_energy_components_j']['sleep'] += .01
            lines(folder / 'rt_sys_trace.jsonl', trace)
            with self.assertRaises(ValueError): s.ap_energy_audit(folder, config, summary)

    def test_one_global_measured_sample_is_real_not_triplicated_by_cells(self):
        folder, model, _, _, _, events = self.fixture()
        self.assertEqual(events[-1]['start_cycle_phase_by_cell']['cell_0'], 3)
        result = s.training_evidence(folder, model)
        self.assertEqual(result['new_actual_global_rows'], 1)
        self.assertTrue(result['raw_energy_packet_labels_verified'])

    def test_empty_only_negative_control_saves_energy_without_kpi_harm(self):
        baseline = self.fixture(action=0, mode='fixed')[0]
        empty = self.fixture(action=1, mode='fixed')[0]
        result = s.negative_control(dict(off=baseline, empty_only=empty))
        self.assertTrue(result['passed'], result)
        self.assertGreater(result['modeled_ap_energy_saving_fraction'], 0.)
        self.assertEqual(result['cases']['empty_only']['ap_energy']['occupied_unavailable_cell_slots'], 0)
        occupied_sleep = self.fixture(action=2, mode='fixed')[0]
        bad = s.negative_control(dict(off=baseline, empty_only=occupied_sleep))
        self.assertFalse(bad['passed'])
        self.assertFalse(bad['structural_validation_passed'])
        self.assertFalse(s.empty_control_acceptance(bad, True)['accepted'])

    def test_explicit_reference_coupling_keeps_strict_counterexample_failed(self):
        baseline = self.fixture(action=0, mode='fixed')[0]
        empty = self.fixture(action=1, mode='fixed')[0]
        events = [json.loads(v) for v in (empty / 'packet_events.jsonl').read_text().splitlines()]
        for e in events:
            if e['kind'] == 'delivery' and e['ue_id'] == 'ue_0': e.update(bytes=99, completed=False)
        lines(empty / 'packet_events.jsonl', events)
        summary = json.loads((empty / 'final_summary.json').read_text())
        summary['packet_metrics']['ue_0'].update(delivered_bytes=99, completed_packets=0,
            pending_bytes=1, pending_packets=1, completion_latency_ms_sum=0.)
        dump(empty / 'final_summary.json', summary)
        result = s.negative_control(dict(off=baseline, empty_only=empty))
        self.assertFalse(result['passed']); self.assertFalse(result['no_observed_packet_kpi_harm'])
        self.assertTrue(result['structural_validation_passed'])
        self.assertEqual(result['per_ue_qos_delta_vs_off']['ue_0']['delivered_bytes'], -1)
        self.assertEqual(result['per_ue_qos_delta_vs_off']['ue_0']['pending_bytes'], 1)
        self.assertFalse(s.empty_control_acceptance(result)['accepted'])
        amended = s.empty_control_acceptance(result, True)
        self.assertTrue(amended['accepted']); self.assertFalse(amended['strict_negative_passed'])
        self.assertEqual(amended['mode'], 'reference_cqi_coupled_structural')
        r = dict(negative_control=result, allow_empty_reference_coupling=True, empty_control_acceptance=amended)
        self.assertEqual(s.revalidate_empty_control_acceptance(r, result), amended)
        with self.assertRaisesRegex(ValueError, 'not retroactively'):
            s.revalidate_empty_control_acceptance(dict(negative_control=result), result)
        bad = dict(result, structural_validation_passed=False)
        with self.assertRaises(ValueError): s.revalidate_empty_control_acceptance(r, bad)

    def test_old_strict_receipts_remain_accepted_without_amendment(self):
        strict = dict(passed=True, no_observed_packet_kpi_harm=True, structural_validation_passed=True)
        self.assertEqual(s.revalidate_empty_control_acceptance(dict(negative_control=strict), strict)['mode'], 'strict_no_harm')
        self.assertFalse(s.parse_args(['--output', 'outputs/new']).allow_empty_reference_coupling)

    def test_historical_all_active_parity_ignores_only_new_ap_energy_and_host_timing(self):
        folder, _, _, summary, _, _ = self.fixture()
        summary['ue_services']['module_wall_s'] = .1
        dump(folder / 'final_summary.json', summary)
        reference = self.folder(); old = deepcopy(summary); old.pop('ap_services')
        old['ue_services']['module_wall_s'] = .2
        dump(reference / 'final_summary.json', old)
        proof = s.legacy_parity(folder, reference)
        self.assertTrue(proof['verified'])
        self.assertEqual(proof['excluded_host_timing_fields'], ['ue_services.module_wall_s'])
        old['ue_services']['actual_state'] = 1; dump(reference / 'final_summary.json', old)
        with self.assertRaisesRegex(ValueError, 'historical'): s.legacy_parity(folder, reference)
        old['ue_services'].pop('actual_state')
        old['packet_metrics']['ue_0']['delivered_bytes'] -= 1; dump(reference / 'final_summary.json', old)
        with self.assertRaisesRegex(ValueError, 'historical'): s.legacy_parity(folder, reference)

    def test_actual_frozen_rf_forecasts_select_energy_without_a_false_qos_guarantee(self):
        folder, model, config, summary, _, events = self.fixture(action=3, mode='frozen')
        p = payload(24, True)
        tree = dict(children_left=[1, -1, -1], children_right=[2, -1, -1], feature=[11, -2, -2], threshold=[.5, -2., -2.],
            value=[[0.] * 4, [35., .048, 0., 0.], [25., .02, 100., 200.]])
        p['forest'] = [deepcopy(tree) for _ in range(32)]; dump(model, p)
        learner = OnlineApRF.load(model, s.FEATURES)
        forecasts = learner.predict([[0.] * 8 + [float(i == j) for j in range(4)] for i in range(4)])
        decision = dict(kind='decision', status='rf', action=3, features=[0.] * 8 + [0., 0., 0., 1.],
            prediction=forecasts[3], prediction_audit=dict(raw_predictions=forecasts, selected_action=3, constraints_active=False))
        lines(folder / 'ap_rf_events.jsonl', [events[0], decision])
        stamp = s.model_stamp(model)
        summary['ap_xapp'].update(samples=24, trained_samples=24, fit_count=1, fit_parameters=FIT_PARAMETERS,
            prediction_batches=1, model_sha256=stamp['sha256'])
        before = model.read_bytes()
        result = s.ap_control_evidence(folder, config, summary, stamp)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['genuine_replayed_inferences'], 1)
        self.assertEqual(model.read_bytes(), before)
        decision['prediction_audit']['raw_predictions'][0][0] += 1.
        lines(folder / 'ap_rf_events.jsonl', [events[0], decision])
        self.assertFalse(s.ap_control_evidence(folder, config, summary, stamp)['valid'])

    def test_disjoint_controls_do_not_force_an_m2_claim_when_x08_has_no_gain(self):
        folders = {}
        for arm in s.ARMS:
            folder, _, config, summary, _, _ = self.fixture(action=1 if arm in ('x02', 'both') else 0, mode='frozen')
            apps = ([] if arm == 'off' else [s.APP] if arm == 'x02' else [s.X08] if arm == 'x08' else [s.APP, s.X08])
            config['configs']['ric']['xapps'] = [dict(implementation_id=a, parameters=dict(mode='frozen', policy='throughput', model_path='unused')) for a in apps]
            summary['xapp_library']['instances'] = [dict(implementation_id=a,
                writable_actions=['cell_sleep_mode'] if a == s.APP else ['scheduler_mode', 'ue_scheduler_weights']) for a in apps]
            dump(folder / 'config_snapshot.json', config); dump(folder / 'final_summary.json', summary); folders[arm] = folder
        with patch.object(s, 'ap_control_evidence', return_value=dict(valid=True)), \
                patch.object(s, 'scheduler_evidence', return_value=dict(valid_control_evidence=True)):
            result = s.evaluate_study(folders, {}, {})
        self.assertEqual(result['rcp_intersection'], [])
        self.assertTrue(result['valid_evidence'])
        self.assertFalse(result['independently_effective']['x08'])
        self.assertFalse(result['observed_pattern_consistent_with_m2'])
        self.assertEqual(result['classification'], 'controllers_not_individually_effective')

    def test_future_ack_and_fabricated_energy_label_are_rejected(self):
        for change in ('ack', 'label'):
            folder, model, _, _, _, events = self.fixture()
            if change == 'ack': events[0]['actual_applied_slots']['cell_0'] = 100
            else: events[-1]['ap_energy_j_by_cell']['cell_0'] += 1.
            lines(folder / 'ap_rf_events.jsonl', events)
            with self.assertRaises(ValueError): s.training_evidence(folder, model)

    def test_readiness_uses_real_fitted_prefix_coverage(self):
        folder = self.folder(); model = folder / 'model.json'; p = payload(24, True); dump(model, p)
        self.assertTrue(s.readiness(model)['ready'])
        p['x'][-1][-4:] = [1., 0., 0., 0.]; dump(model, p)
        self.assertFalse(s.readiness(model)['ready'])

    def test_completed_latency_and_failure_are_count_pooled(self):
        folder, _, config, summary, trace, _ = self.fixture()
        packets = []; metrics = {}
        for u in s.POPULATION:
            n = 3 if u == 'ue_6' else 1; latencies = []
            for i in range(n):
                birth = 20 + i * 20; late = u == 'ue_2'; delay = 20. if late else 1.
                packets += [dict(kind='arrival', ue_id=u, packet_id=f'{u}:{i}', slot_id=birth, bytes=100, admitted=True),
                    dict(kind='delivery', ue_id=u, packet_id=f'{u}:{i}', slot_id=birth + int(delay * 2), bytes=100, completed=True, latency_ms=delay)]
                if late: packets.append(dict(kind='deadline_miss', ue_id=u, packet_id=f'{u}:{i}', slot_id=birth + 21))
                latencies.append(delay)
            metrics[u] = dict(initial_bytes=0, offered_bytes=n * 100, delivered_bytes=n * 100, overflow_bytes=0,
                overflow_packets=0, admitted_packets=n, completed_packets=n, pending_bytes=0, pending_packets=0,
                completion_latency_ms_sum=sum(latencies), deadline_missed_packets=int(u == 'ue_2'))
        lines(folder / 'packet_events.jsonl', packets); summary['packet_metrics'] = metrics
        for row in trace:
            for u in s.POPULATION:
                row['traffic_arrivals'][u]['exogenous_bytes'] = sum(e['bytes'] for e in packets if e['kind'] == 'arrival' and e['slot_id'] == row['slot_id'] and e['ue_id'] == u)
        lines(folder / 'rt_sys_trace.jsonl', trace)
        result = s.packet_statistics(folder, config, summary)
        self.assertEqual(result['traffic_classes']['urllc']['failure_fraction'], .25)
        self.assertAlmostEqual(result['traffic_classes']['urllc']['completed_p95_ms'], 17.15)
        self.assertEqual(result['network']['failure_fraction'], 1 / 8)

    def test_dry_run_all_keeps_enabled_profile_and_disjoint_priority_controls(self):
        root = self.folder(); args = s.parse_args(['--output', str(root / 'outputs/new'), '--eval-workers', '3', '--dry-run'])
        stream = io.StringIO()
        with patch.object(s, 'ROOT', root), patch.object(s, 'parse_args', return_value=args), patch.object(s.subprocess, 'run') as jobs, redirect_stdout(stream): s.main()
        plan = json.loads(stream.getvalue()); self.assertEqual(len(plan['commands']), 7)
        self.assertTrue(plan['shared_off_baseline']); self.assertEqual(plan['maximum_training_physics_s'], 6.)
        self.assertTrue(all('--ap-services-profile' in c and c[c.index('--traffic-load') + 1] == '2' and
            c[c.index('--direct-conflict-mitigation') + 1] == 'priority' for c in plan['commands']))
        self.assertEqual([c[c.index('--ap-xapp-mode') + 1] for c in plan['commands']], ['off', 'fixed', 'online', 'online', 'frozen', 'off', 'frozen'])
        jobs.assert_not_called()

    def test_parallel_frozen_cases_wait_after_off_and_fail_closed(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                root = self.folder(); models = root / 'outputs/models'; models.mkdir(parents=True)
                nn, rf = models / 'nn.json', models / 'rf.json'; dump(nn, {}); dump(rf, {})
                power = root / 'profile.json'; dump(power, dict(ap_services=profile()))
                refs = []
                for name in ('reference_off', 'reference_x08'):
                    ref = root / name; ref.mkdir(); dump(ref / 'final_summary.json', {}); refs.append(ref)
                output = root / 'outputs/new'
                args = s.parse_args(['--phase', 'evaluate', '--output', str(output), '--ap-model', str(rf),
                    '--throughput-model', str(nn), '--ap-services-profile', str(power), '--eval-workers', '3',
                    '--reference-off', str(refs[0]), '--reference-x08', str(refs[1])])
                off_done, gate, finished = threading.Event(), threading.Barrier(3), set()
                def runner(command, **kwargs):
                    folder = Path(command[command.index('--output') + 1]); folder.mkdir()
                    arm = next(a for a in s.ARMS if folder.name == 'seed55_' + a)
                    if arm == 'off': off_done.set()
                    else:
                        self.assertTrue(off_done.is_set()); gate.wait(timeout=2)
                    dump(folder / 'performance.json', dict(simulation_s=.01)); dump(folder / 'config_snapshot.json', {}); finished.add(arm)
                    return SimpleNamespace(returncode=int(fail and arm == 'x08'))
                with patch.object(s, 'ROOT', root), patch.object(s, 'parse_args', return_value=args), \
                        patch.object(s, 'mlp_readiness', return_value=dict(ready=True)), \
                        patch.object(s, 'mlp_stamp', return_value=dict(sha256=s.sha256(nn))), \
                        patch.object(s, 'model_stamp', return_value=dict(sha256=s.sha256(rf))), \
                        patch.object(s, 'training_lineage', return_value=dict(original_ap_services_profile=dict(ap_services=profile()), collector_source_sha256s=['same'],
                            negative_control=dict(passed=True), empty_control_acceptance=dict(accepted=True), training_receipt='original-receipt')), \
                        patch.object(s, 'collection_eval_compatibility', return_value=dict(verified=True)), \
                        patch.object(s, 'legacy_parity', return_value=dict(verified=True)), \
                        patch.object(s, 'case_data', return_value=dict(audit=dict(source_sha256='same'))), \
                        patch.object(s, 'evaluate_study', return_value=dict(classification='unit_complete')) as evaluate, \
                        patch.object(s.subprocess, 'run', side_effect=runner), redirect_stdout(io.StringIO()):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'all submitted cases'): s.main()
                    else: s.main()
                self.assertEqual(finished, set(s.ARMS))
                receipt = json.loads((output / 'acceptance.json').read_text())
                self.assertEqual(receipt['completed'], not fail)
                self.assertEqual(receipt['running_cases'], [])
                self.assertEqual([c['arm'] for c in receipt['cases']], ['off', 'x02', 'both'] if fail else list(s.ARMS))
                self.assertEqual(evaluate.call_count, 0 if fail else 1)


if __name__ == '__main__': unittest.main()
