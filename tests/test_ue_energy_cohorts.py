"""Bounded, generation-cohort task evidence for delayed online controllers."""
from copy import deepcopy
import unittest

from models.action import ConfigAction
from tests.test_ue_services import fixture, step


def config(**profile):
    return dict(task_cohort_ms=100, mec=dict(enabled=True, local_gflops=1.,
                tasks_by_ue={'ue_0': dict(interval_ms=20, compute_gflop=.001,
                                         deadline_ms=100, offload_fraction=0., **profile)}))


class TestUeEnergyCohorts(unittest.TestCase):
    def test_generation_membership_and_conservation(self):
        _, model = fixture(config())
        for slot in range(1, 201):
            step(model, slot)
        rows = model.reports['ue_0']['ue_task_cohorts']
        self.assertEqual([r['index'] for r in rows], [0, 1])
        for row in rows:
            self.assertEqual(row['generated'], 5)
            self.assertEqual(row['completed'], 5)
            self.assertEqual(row['generated'], row['completed'] + row['failed'] + row['pending'])
            self.assertEqual(row['on_time_completed'], 5)
            self.assertAlmostEqual(row['completion_latency_ms_sum'], 5.)
            self.assertEqual(row['offload_fraction'], 0.)

    def test_maturity_uses_cohort_end_plus_deadline(self):
        _, model = fixture(config())
        for slot in range(1, 101):
            step(model, slot)
        row = model.reports['ue_0']['ue_task_cohorts'][0]
        self.assertAlmostEqual(row['start_s'], 0.)
        self.assertAlmostEqual(row['end_s'], .1)
        self.assertAlmostEqual(row['mature_after_s'], .2)
        self.assertLess(model.slot * model.dt, row['mature_after_s'])
        for slot in range(101, 201):
            step(model, slot)
        self.assertGreaterEqual(model.slot * model.dt + 1e-12, row['mature_after_s'])

    def test_late_completion_counts_deadline_only_once(self):
        cfg = config()
        cfg['mec']['tasks_by_ue']['ue_0'].update(interval_ms=1000, compute_gflop=.01, deadline_ms=5)
        _, model = fixture(cfg)
        for slot in range(1, 11):
            step(model, slot)
        row = model.reports['ue_0']['ue_task_cohorts'][0]
        self.assertEqual(row['deadline_misses'], 1)
        self.assertEqual(row['completed'], 1)
        self.assertEqual(row['on_time_completed'], 0)
        self.assertEqual(row['pending'], 0)

    def test_result_ack_isolation_and_duplicate_ack(self):
        cfg = config()
        cfg['mec']['tasks_by_ue'] = {f'ue_{i}': dict(interval_ms=1000, compute_gflop=.001,
                                                  offload_fraction=1.) for i in range(2)}
        state, model = fixture(cfg, count=2)
        for slot in range(1, 21):
            step(model, slot)
        first, second = state.ues
        self.assertEqual(model.reports['ue_0']['ue_task_cohorts'][0]['pending'], 1)
        first.packet_ledger.reserve('result', [('lc_0', 200)])
        first.packet_ledger.acknowledge('result', 20)
        first.packet_ledger.acknowledge('result', 21)
        model._reports()
        a = model.reports['ue_0']['ue_task_cohorts'][0]
        b = model.reports['ue_1']['ue_task_cohorts'][0]
        self.assertEqual((a['generated'], a['completed'], a['pending']), (1, 1, 0))
        self.assertEqual((b['generated'], b['completed'], b['pending']), (1, 0, 1))
        self.assertEqual(second.packet_ledger.stats['delivered_bytes'], 0)

    def test_mixed_policy_is_identified_at_generation(self):
        cfg = config()
        cfg['mec']['tasks_by_ue']['ue_0'].update(interval_ms=50)
        _, model = fixture(cfg)
        step(model, 1)
        model.apply_action(ConfigAction(ue_offload_fraction={'ue_0': 1.},
                                       ue_drx_configs={'ue_0': {'inactivity_ms': 4.}}))
        for slot in range(2, 52):
            step(model, slot)
        row = model.reports['ue_0']['ue_task_cohorts'][0]
        self.assertTrue(row['mixed_policy'])
        self.assertIsNone(row['offload_fraction'])
        self.assertIsNone(row['drx_config'])
        self.assertEqual((row['offload_fraction_min'], row['offload_fraction_max']), (0., 1.))

    def test_telemetry_and_late_old_results_stay_bounded(self):
        cfg = config()
        cfg['mec'].update(local_gflops=.001, max_queued_tasks=1)
        cfg['mec']['tasks_by_ue']['ue_0'].update(interval_ms=1, compute_gflop=.1)
        _, model = fixture(cfg)
        for slot in range(1, 1801):
            step(model, slot)
        self.assertEqual(len(model.cohorts['ue_0']), 16)
        rows = model.reports['ue_0']['ue_task_cohorts']
        self.assertEqual(len(rows), 8)
        self.assertEqual([r['index'] for r in rows], list(range(10, 18)))
        for row in rows:
            self.assertEqual(row['generated'], row['completed'] + row['failed'] + row['pending'])
        model._serve(model.local['ue_0'], 1., lambda task: model._finish(task, 'completed', now=2.))
        model._reports()
        self.assertNotIn(0, model.cohorts['ue_0'])
        self.assertEqual(len(model.cohorts['ue_0']), 16)
        self.assertEqual(model.stats['ue_0']['completed'], 1)

    def test_default_off_and_non_task_ue_report_unavailable(self):
        _, off = fixture()
        self.assertIsNone(off.reports['ue_0']['ue_task_cohorts'])
        _, model = fixture(config(), count=2)
        step(model, 1)
        self.assertIsNone(model.reports['ue_1']['ue_task_cohorts'])

    def test_cohorts_do_not_change_rng_energy_or_task_outcomes(self):
        cfg = config()
        cfg['mec']['tasks_by_ue']['ue_0']['offload_fraction'] = .5
        a_cfg = deepcopy(cfg)
        a_cfg.pop('task_cohort_ms')
        _, a = fixture(a_cfg)
        _, b = fixture(cfg)
        for slot in range(1, 301):
            step(a, slot, deliver=True)
            step(b, slot, deliver=True)
        self.assertEqual(a.stats, b.stats)
        self.assertEqual(a.energy, b.energy)
        self.assertEqual(a.users['ue_0'].packet_ledger.stats, b.users['ue_0'].packet_ledger.stats)

    def test_fractional_time_boundaries_use_integer_slots(self):
        cfg = config()
        cfg['mec']['tasks_by_ue']['ue_0']['interval_ms'] = 50
        _, model = fixture(cfg, dt=.5)
        for slot in range(1, 802):
            step(model, slot)
        row = next(r for r in model.reports['ue_0']['ue_task_cohorts'] if r['index'] == 3)
        self.assertEqual(row['generated'], 2)

    def test_invalid_cohort_timing_rejected(self):
        for value in [-1, True, .25, float('nan')]:
            with self.assertRaises(ValueError):
                fixture({'task_cohort_ms': value})


if __name__ == '__main__':
    unittest.main()
