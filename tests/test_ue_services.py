"""Physical accounting and causal task/DRX tests without a PHY dependency."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from phy.rt_sys.ue_services import UeServices, POWER_DEFAULTS
from traffic.packet_ledger import PacketLedger
from models.action import ConfigAction


def fixture(config=None, count=1, dt=1.):
    ues = []
    for i in range(count):
        lc = NS(logical_channel_id='lc_0', queue_bytes=0, max_queue_bytes=10000, packet_delay_budget_ms=100.)
        u = NS(ue_id=f'ue_{i}', logical_channels=[lc], queue_bytes=0, harq=NS(processes=[]))
        u.packet_ledger = PacketLedger(u, dt)
        ues.append(u)
    state = NS(ues=ues, managed_cells={'cell_0': NS()}, connections={u.ue_id: NS(serving_cell_id='cell_0', connection_state='connected', handover_pending_target_cell_id=None) for u in ues})
    model = UeServices(state, config or {}, dt, seed=1)
    return state, model


def step(model, slot, deliver=False):
    model.begin_slot(slot)
    model.before_scheduling()
    rows = []
    for uid, u in model.users.items():
        amount = sum(c.queue_bytes for c in u.logical_channels) if deliver and u.ue_radio_available else 0
        if amount:
            u.packet_ledger.deliver(u.logical_channels[0], amount, slot)
            lc=u.logical_channels[0]
            lc.queue_bytes=sum(p['remaining_bytes'] for p in u.packet_ledger.queues[lc.logical_channel_id])
        rows.append(dict(ue_id=uid, scheduled=bool(amount), allocated_re=100 if amount else 0,
                         ack_source='sys' if amount else 'not_scheduled', is_retx=False,
                         resource_allocation={'symbols': list(range(14))}))
    trace = dict(ues=rows)
    model.after_phy(trace)
    return trace


class TestUeServices(unittest.TestCase):
    def test_energy_integrates_watts_and_time(self):
        _, m = fixture()
        for s in range(1, 1001):
            step(m, s)
        self.assertAlmostEqual(m.reports['ue_0']['ue_energy_total_j'], .3)
        e = m.energy['ue_0']
        self.assertAlmostEqual(e['total_j'], e['radio_j'] + e['compute_j'] + e['transition_j'])

    def test_idle_drx_reduces_energy_and_blocks_scheduling(self):
        state, m = fixture({'drx': dict(enabled=True, cycle_ms=20, on_duration_ms=2, inactivity_ms=0, wakeup_ms=1)})
        u = state.ues[0]
        u.packet_ledger.add(u.logical_channels[0], 200, 1)
        active = []
        for s in range(1, 101):
            step(m, s)
            active.append(u.ue_radio_available)
        self.assertFalse(active[5])
        self.assertTrue(active[20])
        self.assertEqual(u.packet_ledger.stats['pending_bytes'], 200)
        self.assertLess(m.reports['ue_0']['ue_energy_total_j'], .03)
        self.assertGreater(m.energy['ue_0']['transition_j'], 0)

    def test_pending_harq_and_uplink_keep_radio_awake(self):
        state, m = fixture({'drx': dict(enabled=True, cycle_ms=20, on_duration_ms=2, wakeup_ms=0)})
        state.ues[0].harq.processes = [NS(is_busy=True, is_waiting_ack=True, pending_bytes=10)]
        for s in range(1, 25):
            step(m, s)
            self.assertTrue(state.ues[0].ue_radio_available)

    def test_local_compute_time_and_energy(self):
        _, m = fixture({'mec': dict(enabled=True, local_gflops=1., tasks_by_ue={'ue_0': dict(interval_ms=100, compute_gflop=.01, offload_fraction=0.)})})
        for s in range(1, 11):
            step(m, s)
        self.assertEqual(m.stats['ue_0']['completed'], 1)
        self.assertAlmostEqual(m.stats['ue_0']['completion_latency_ms_sum'], 10.)
        self.assertAlmostEqual(m.energy['ue_0']['compute_j'], (.05 + .8) * .01)

    def test_edge_result_requires_its_own_actual_radio_delivery(self):
        state, m = fixture({'mec': dict(enabled=True, tasks_by_ue={'ue_0': dict(interval_ms=100, compute_gflop=.01, offload_fraction=1.)})})
        for s in range(1, 20):
            step(m, s)
        self.assertEqual(m.stats['ue_0']['completed'], 0)
        u = state.ues[0]
        self.assertEqual(u.packet_ledger.stats['pending_bytes'], 200)
        u.packet_ledger.reserve('result', [('lc_0', 200)])
        u.packet_ledger.acknowledge('result', 20)
        self.assertEqual(m.stats['ue_0']['completed'], 1)
        # Duplicate ACK cannot complete the task or count bytes twice.
        u.packet_ledger.acknowledge('result', 21)
        self.assertEqual(m.stats['ue_0']['completed'], 1)
        self.assertEqual(u.packet_ledger.stats['delivered_bytes'], 200)

    def test_shared_upload_budget_and_bounded_overload(self):
        profiles={f'ue_{i}': dict(interval_ms=1, input_bytes=1000, offload_fraction=1.) for i in range(2)}
        _, m = fixture({'mec': dict(enabled=True, max_queued_tasks=2, uplink_bps_per_cell=8000, tasks_by_ue=profiles)}, count=2)
        for s in range(1, 11):
            step(m, s)
        self.assertLessEqual(m.server_stats['cell_0']['uploaded_bits'], 80.)
        self.assertLessEqual(len(m.tasks), 4)
        self.assertGreater(sum(v['failed'] for v in m.stats.values()), 0)
        for uid, stats in m.stats.items():
            self.assertEqual(stats['generated'], stats['completed'] + stats['failed'] + sum(t['uid']==uid for t in m.tasks.values()))

    def test_deadline_failure_includes_pending_tasks(self):
        _, m = fixture({'mec': dict(enabled=True, local_gflops=.01, tasks_by_ue={'ue_0': dict(interval_ms=100, compute_gflop=.1, deadline_ms=5, offload_fraction=0.)})})
        for s in range(1, 11):
            step(m, s)
        self.assertEqual(m.stats['ue_0']['deadline_misses'], 1)
        self.assertEqual(m.reports['ue_0']['ue_task_metrics']['success_ratio'], 0)

    def test_invalid_actions_are_atomic_and_cell_scoped(self):
        _, m = fixture()
        original = deepcopy(m.drx)
        for action in [ConfigAction(ue_drx_configs={'ue_0': {'cycle_ms': .25}}),
                       ConfigAction(ue_drx_configs={'unknown': {'enabled': True}}),
                       ConfigAction(ue_offload_fraction={'ue_0': .8})]:
            with self.assertRaises(ValueError):
                m.validate_action(action, ['cell_0'])
        with self.assertRaises(ValueError):
            m.validate_action(ConfigAction(ue_drx_configs={'ue_0': {'enabled': True}}), ['other'])
        self.assertEqual(m.drx, original)
        action = ConfigAction(ue_drx_configs={'ue_0': {'enabled': True}})
        m.validate_action(action, ['cell_0'])
        m.apply_action(action)
        self.assertTrue(m.drx['ue_0']['enabled'])

    def test_result_overflow_does_not_count_task_success(self):
        state, m = fixture({'mec': dict(enabled=True, tasks_by_ue={'ue_0': dict(interval_ms=100, offload_fraction=1.)})})
        state.ues[0].logical_channels[0].max_queue_bytes=0
        for s in range(1, 20):
            step(m, s)
        self.assertEqual(m.stats['ue_0']['result_overflow'], 1)
        self.assertEqual(m.stats['ue_0']['completed'], 0)

    def test_upload_no_path_is_not_success_and_other_users_can_progress(self):
        profiles={f'ue_{i}': dict(interval_ms=100, offload_fraction=1.) for i in range(2)}
        state, m = fixture({'mec': dict(enabled=True, tasks_by_ue=profiles)}, count=2)
        state.paper=NS(raw={'ue_0': {'cell_0': {'no_path': True}}, 'ue_1': {'cell_0': {'no_path': False}}})
        for s in range(1, 20):
            step(m, s, deliver=True)
        self.assertEqual(m.stats['ue_0']['completed'], 0)
        self.assertEqual(m.stats['ue_1']['completed'], 1)
        self.assertEqual(m.server_stats['cell_0']['uploaded_bits'], 8000)

    def test_upload_follows_handover_without_double_service(self):
        state, m = fixture({'mec': dict(enabled=True, uplink_bps_per_cell=8000, tasks_by_ue={'ue_0': dict(interval_ms=100, offload_fraction=1.)})})
        state.managed_cells['cell_1']=NS()
        m.upload['cell_1']=__import__('collections').deque()
        m.edge['cell_1']=__import__('collections').deque()
        m.server_stats['cell_1']=__import__('collections').Counter()
        step(m, 1)
        state.connections['ue_0'].serving_cell_id='cell_1'
        step(m, 2)
        self.assertEqual(m.server_stats['cell_0']['uploaded_bits'], 8)
        self.assertEqual(m.server_stats['cell_1']['uploaded_bits'], 8)
        self.assertEqual(len(m.upload['cell_0']), 0)


if __name__ == '__main__':
    unittest.main()
