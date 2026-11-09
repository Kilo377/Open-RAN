"""Factorial labels must follow paired effects and executed controls."""
import gzip
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_latency_energy_study import (APP_IDS, ARMS, POPULATION,
                                                aggregate_offloaded_cohorts,
                                                artifact_stamp, evaluate_factorial,
                                                interaction, main, parse_args,
                                                resumed_training_evidence)


class TestLatencyEnergyStudy(unittest.TestCase):
    def fixture(self, conflict=False, overlap=False, wrong_samples=False, native_only=False, bad_weight=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        folders = {arm: root / arm for arm in ARMS}
        stamps = {kind: dict(samples=18, trained_samples=18, fit_count=4) for kind in APP_IDS}
        powers = dict(off=1., x01=.8, x12=1.05, both=.9 if conflict else .79)
        latency = dict(off=25., x01=20., x12=15., both=18. if conflict else 14.)
        for arm, folder in folders.items():
            folder.mkdir()
            manifest = dict(ues=[dict(ue_id=u) for u in sorted(POPULATION)],
                            cells=[{}, {}, {}], rng_seed=55, total_slots=2000, slot_duration_ms=.5)
            config = dict(manifest=manifest, configs=dict(ue=dict(ues=sorted(POPULATION))))
            (folder / 'config_snapshot.json').write_text(json.dumps(config))
            enabled = [] if arm == 'off' else ['x01', 'x12'] if arm == 'both' else [arm]
            summary = dict(ue_services=dict(per_ue={u: dict(ue_energy_total_j=powers[arm]) for u in POPULATION}),
                           packet_metrics={u: dict(delivered_bytes=10000, admitted_packets=100,
                            mean_completion_latency_ms=1., deadline_missed_packets=0,
                            overflow_bytes=0, pending_bytes=0) for u in POPULATION},
                           xapp_library=dict(instances=[]))
            receipts = []
            for kind in enabled:
                fields = (['ue_drx_configs', 'ue_offload_fraction'] if kind == 'x01' else
                          ['scheduler_mode', 'ue_scheduler_weights'] + (['ue_offload_fraction'] if overlap else []))
                summary['xapp_library']['instances'].append(dict(implementation_id=APP_IDS[kind],
                    writable_actions=fields, active_cell_ids=['cell_0', 'cell_2']))
                learning = dict(mode='frozen', samples=19 if wrong_samples and kind == 'x12' else 18,
                                trained_samples=18, fit_count=4, rf_selected_actions=0 if native_only and kind == 'x12' else 2,
                                prediction_batches=2)
                summary['ue_energy_xapp' if kind == 'x01' else 'latency_xapp'] = learning
                log = folder / ('energy_rf_events.jsonl' if kind == 'x01' else 'latency_rf_events.jsonl')
                log.write_text(json.dumps(dict(kind='executed_policy_confirmed')) + '\n')
                receipts.append(dict(status='applied', provenance=[dict(implementation_id=APP_IDS[kind])],
                    fields=[dict(field=f, value={'ue_0': 1., 'ue_4': 1.} if f == 'ue_scheduler_weights' else None) for f in fields], after={'cell_0': dict(
                        scheduler_mode='round_robin' if native_only and kind == 'x12' else 'latency_aware',
                        ue_scheduler_weights={'ue_0': .25 if bad_weight and kind == 'x12' else 1., 'ue_4': 1.},
                        ue_service_policies={'ue_0': dict(offload_fraction=1.)})}))
            (folder / 'final_summary.json').write_text(json.dumps(summary))
            (folder / 'control_receipts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in receipts))
            c = dict(start_s=.1, end_s=.2, mature_after_s=.3, generated=2,
                     completed=2, on_time_completed=2, failed=0, pending=0,
                     completion_latency_ms_sum=2 * latency[arm],
                     offloaded_generated=1, offloaded_completed=1,
                     offloaded_on_time_completed=1, offloaded_failed=0,
                     offloaded_pending=0, offloaded_completion_latency_ms_sum=latency[arm])
            with gzip.open(folder / 'rt_sys_trace.jsonl.gz', 'wt') as stream:
                stream.write(json.dumps(dict(slot_id=600, ue_services={u: dict(ue_task_cohorts=[c]) for u in ('ue_0', 'ue_4')})) + '\n')
        return folders, stamps

    def change_cohorts(self, folder, **values):
        path = folder / 'rt_sys_trace.jsonl.gz'
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        for row in rows:
            for user in row['ue_services'].values():
                user['ue_task_cohorts'][0].update(values)
        with gzip.open(path, 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')

    def test_synergy_is_not_forced_to_be_a_conflict(self):
        folders, stamps = self.fixture()
        result = evaluate_factorial(folders, stamps)
        self.assertEqual(result['classification'], 'no_detected_conflict')
        self.assertTrue(result['controlled_parameter_surface']['disjoint'])
        # D can be positive when both is better than either isolated policy.
        self.assertGreater(result['factorial_interaction']['target_completed_task_mean_latency_ms'], 0)

    def test_harmful_disjoint_coupling_is_reported(self):
        folders, stamps = self.fixture(conflict=True)
        result = evaluate_factorial(folders, stamps)
        self.assertEqual(result['classification'], 'indirect_operational_conflict_observed')
        self.assertTrue(result['operational_energy_harm'])
        self.assertTrue(result['operational_task_harm'])
        self.assertAlmostEqual(result['factorial_interaction']['target_energy_j'], .1)
        self.assertTrue(result['standalone_effectiveness']['both_individually_effective'])

    def test_overlap_cannot_be_called_indirect(self):
        folders, stamps = self.fixture(conflict=True, overlap=True)
        result = evaluate_factorial(folders, stamps)
        self.assertEqual(result['classification'], 'direct_rcp_overlap_invalidates_indirect_case')

    def test_frozen_training_changes_invalidate_control_evidence(self):
        folders, stamps = self.fixture(conflict=True, wrong_samples=True)
        result = evaluate_factorial(folders, stamps)
        self.assertEqual(result['classification'], 'insufficient_control_evidence')

    def test_demand_mismatch_is_rejected(self):
        folders, stamps = self.fixture()
        path = folders['both'] / 'config_snapshot.json'
        config = json.loads(path.read_text())
        config['configs']['ue']['extra'] = 'changed demand'
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            evaluate_factorial(folders, stamps)

    def test_non_target_harm_is_visible(self):
        folders, stamps = self.fixture()
        path = folders['both'] / 'final_summary.json'
        summary = json.loads(path.read_text())
        summary['packet_metrics']['ue_2']['delivered_bytes'] = 9000
        summary['packet_metrics']['ue_2']['deadline_missed_packets'] = 6
        path.write_text(json.dumps(summary))
        result = evaluate_factorial(folders, stamps)
        self.assertTrue(result['non_target_off_vs_both']['ue_2']['goodput_harm'])
        self.assertTrue(result['non_target_off_vs_both']['ue_2']['deadline_harm'])

    def test_ineffective_x12_is_not_promoted_to_a_successful_conflict(self):
        folders, stamps = self.fixture(conflict=True)
        self.change_cohorts(folders['x12'], offloaded_completion_latency_ms_sum=26.)
        result = evaluate_factorial(folders, stamps)
        self.assertEqual(result['classification'], 'controllers_not_individually_effective')
        self.assertFalse(result['standalone_effectiveness']['x12']['individually_effective'])
        self.assertTrue(result['operational_tradeoff_flag'])

    def test_offloaded_primary_is_explicit_and_all_task_mean_remains_visible(self):
        folders, stamps = self.fixture()
        self.change_cohorts(folders['x12'], completion_latency_ms_sum=50.)
        result = evaluate_factorial(folders, stamps)
        self.assertEqual(result['objective_scope']['revision'], 'offloaded_ar_task_e2e_v2')
        self.assertEqual(result['arms']['x12']['target_task_completed_mean_latency_ms'], 25.)
        self.assertEqual(result['arms']['x12']['target_offloaded_task_completed_mean_latency_ms'], 15.)
        self.assertTrue(result['standalone_effectiveness']['x12']['individually_effective'])
        self.assertEqual(result['classification'], 'no_detected_conflict')
        self.assertIn('target_offloaded_task_completed_mean_latency_ms', result['factorial_interaction'])

    def test_offloaded_gain_cannot_hide_all_task_delay_harm(self):
        folders, stamps = self.fixture()
        self.change_cohorts(folders['x12'], completion_latency_ms_sum=54.)
        result = evaluate_factorial(folders, stamps)
        self.assertFalse(result['standalone_effectiveness']['x12']['target_all_task_mean_delay_guard']['ue_0'])
        self.assertEqual(result['classification'], 'controllers_not_individually_effective')

    def test_routed_counts_can_differ_while_exogenous_task_demand_matches(self):
        folders, stamps = self.fixture()
        for arm in ('x01', 'both'):
            self.change_cohorts(folders[arm], offloaded_generated=2, offloaded_completed=2,
                                offloaded_on_time_completed=2,
                                offloaded_completion_latency_ms_sum=40. if arm == 'x01' else 28.)
        result = evaluate_factorial(folders, stamps)
        self.assertEqual(result['arms']['off']['target_offloaded_task_generated'], 2)
        self.assertEqual(result['arms']['both']['target_offloaded_task_generated'], 4)
        self.assertEqual(result['arms']['both']['users']['ue_0']['tasks']['generated'], 2)

    def test_missing_historical_route_counters_are_rejected(self):
        folders, stamps = self.fixture()
        self.change_cohorts(folders['off'], offloaded_generated=None)
        with self.assertRaisesRegex(ValueError, 'measured offloaded'):
            evaluate_factorial(folders, stamps)

    def test_nonconserving_measured_route_counters_are_rejected(self):
        folders, stamps = self.fixture()
        self.change_cohorts(folders['x12'], offloaded_generated=2)
        with self.assertRaisesRegex(ValueError, 'conservation'):
            evaluate_factorial(folders, stamps)

    def test_all_generated_task_counts_must_still_match(self):
        folders, stamps = self.fixture()
        self.change_cohorts(folders['both'], generated=3, completed=3, on_time_completed=3)
        with self.assertRaisesRegex(ValueError, 'Generated task demand differs'):
            evaluate_factorial(folders, stamps)

    def test_offloaded_pending_tasks_stay_in_denominator(self):
        cohort = dict(generated=3, completed=2, on_time_completed=2, failed=0, pending=1,
                      completion_latency_ms_sum=24.,
                      offloaded_generated=2, offloaded_completed=1, offloaded_on_time_completed=1,
                      offloaded_failed=0, offloaded_pending=1, offloaded_completion_latency_ms_sum=12.)
        result = aggregate_offloaded_cohorts([cohort])
        self.assertEqual(result['on_time_rate'], .5)
        self.assertEqual(result['completion_rate'], .5)
        self.assertEqual(result['completed_task_mean_latency_ms'], 12.)

    def test_no_routed_outcomes_are_not_replaced_with_all_task_labels(self):
        folders, stamps = self.fixture()
        self.change_cohorts(folders['x12'], offloaded_generated=0, offloaded_completed=0,
                            offloaded_on_time_completed=0, offloaded_completion_latency_ms_sum=0.)
        result = evaluate_factorial(folders, stamps)
        self.assertIsNone(result['arms']['x12']['target_offloaded_task_completed_mean_latency_ms'])
        self.assertEqual(result['classification'], 'controllers_not_individually_effective')

    def test_route_latency_cannot_exceed_total_measured_latency(self):
        folders, stamps = self.fixture()
        self.change_cohorts(folders['x12'], offloaded_completion_latency_ms_sum=31.)
        with self.assertRaisesRegex(ValueError, 'measured offloaded cohort latency'):
            evaluate_factorial(folders, stamps)

    def test_primary_and_all_task_comparisons_are_separate(self):
        folders, stamps = self.fixture()
        self.change_cohorts(folders['both'], completion_latency_ms_sum=40.)
        result = evaluate_factorial(folders, stamps)
        self.assertFalse(result['both_vs_x12_task_comparison']['ue_0']['harmful'])
        self.assertTrue(result['both_vs_x12_all_task_guards']['ue_0']['harmful'])
        self.assertTrue(result['operational_all_task_guard_harm'])
        self.assertFalse(result['operational_offloaded_latency_harm'])

    def test_native_abstention_is_valid_control_but_not_an_effect(self):
        folders, stamps = self.fixture(native_only=True)
        result = evaluate_factorial(folders, stamps)
        self.assertTrue(result['controller_evidence']['x12']['x12']['valid_control_evidence'])
        self.assertTrue(result['controller_evidence']['x12']['x12']['native_abstention_only'])
        self.assertFalse(result['standalone_effectiveness']['x12']['individually_effective'])
        self.assertEqual(result['classification'], 'controllers_not_individually_effective')

    def test_historical_predictions_do_not_prove_new_frozen_inference(self):
        folders, stamps = self.fixture(native_only=True)
        stamps['x12']['prediction_batches'] = 2
        result = evaluate_factorial(folders, stamps)
        self.assertFalse(result['controller_evidence']['x12']['x12']['genuine_rf_used'])
        self.assertEqual(result['classification'], 'insufficient_control_evidence')

    def test_incorrect_actual_weight_invalidates_execution_evidence(self):
        folders, stamps = self.fixture(bad_weight=True)
        result = evaluate_factorial(folders, stamps)
        self.assertEqual(result['classification'], 'insufficient_control_evidence')

    def test_x12_standalone_non_target_delivery_loss_fails_effectiveness(self):
        folders, stamps = self.fixture()
        path = folders['x12'] / 'final_summary.json'
        summary = json.loads(path.read_text())
        summary['packet_metrics']['ue_2']['delivered_bytes'] = 9000
        path.write_text(json.dumps(summary))
        result = evaluate_factorial(folders, stamps)
        self.assertFalse(result['standalone_effectiveness']['x12']['per_ue_delivery_guard']['ue_2'])
        self.assertEqual(result['classification'], 'controllers_not_individually_effective')

    def test_interaction_direction(self):
        self.assertEqual(interaction(10, 8, 9, 10), 3)

    def test_resume_growth_requires_more_rows_trained_rows_and_fits(self):
        before = dict(samples=2, trained_samples=2, fit_count=1, sha256='before')
        after = dict(samples=3, trained_samples=3, fit_count=2, sha256='after')
        initial = dict(feature_names=['a'], x=[[1], [2]], y=[[3], [4]])
        updated = dict(feature_names=['a'], x=[[1], [2], [5]], y=[[3], [4], [6]])
        evidence = resumed_training_evidence(before, after, initial, updated)
        self.assertEqual(evidence['new_measured_rows'], 1)
        self.assertEqual(evidence['initial_rows_retained'], 2)
        for counter in ('samples', 'trained_samples', 'fit_count', 'sha256'):
            with self.subTest(counter=counter):
                unchanged = dict(after, **{counter: before[counter]})
                with self.assertRaisesRegex(ValueError, 'must increase'):
                    resumed_training_evidence(before, unchanged, initial, updated)

    def test_resume_cannot_replace_original_measured_rows(self):
        before = dict(samples=1, trained_samples=1, fit_count=1, sha256='before')
        after = dict(samples=2, trained_samples=2, fit_count=2, sha256='after')
        initial = dict(feature_names=['a'], x=[[1]], y=[[3]])
        updated = dict(feature_names=['a'], x=[[99], [5]], y=[[3], [6]])
        with self.assertRaisesRegex(ValueError, 'dataset prefix'):
            resumed_training_evidence(before, after, initial, updated)

    def test_artifact_stamp_distinguishes_actual_v1_and_v2_fit_parameters(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'model.json'
        payload = dict(version=1, x=[[1]], trained_samples=1, fit_count=1,
                       feature_names=['a'])
        path.write_text(json.dumps(payload))
        self.assertEqual(artifact_stamp(path)['fit_parameters'], {'max_features': 1.})
        payload.update(version=2, fit_parameters={'max_features': .5})
        path.write_text(json.dumps(payload))
        self.assertEqual(artifact_stamp(path)['fit_parameters'], {'max_features': .5})

    def test_resume_and_skip_training_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--output', 'outputs/unused', '--resume-online', '--skip-training'])

    def test_resume_requires_existing_model_before_any_job(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        args = parse_args(['--output', str(root / 'outputs/study'), '--resume-online',
                           '--latency-model', str(root / 'outputs/missing.json'),
                           '--energy-model', str(root / 'outputs/energy.json')])
        with patch('experiments.run_latency_energy_study.ROOT', root), \
                patch('experiments.run_latency_energy_study.parse_args', return_value=args):
            with self.assertRaisesRegex(ValueError, 'existing X12 model'):
                main()
        self.assertFalse((root / 'outputs/study').exists())

    def test_frozen_actual_fit_metadata_must_match_artifact(self):
        folders, stamps = self.fixture()
        stamps['x12']['fit_parameters'] = {'max_features': .5}
        result = evaluate_factorial(folders, stamps)
        self.assertFalse(result['controller_evidence']['x12']['x12']['frozen_fit_parameters_unchanged'])
        self.assertEqual(result['classification'], 'insufficient_control_evidence')


if __name__ == '__main__':
    unittest.main()
