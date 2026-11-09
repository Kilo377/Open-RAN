"""Measured route-specific task accounting without changing service physics."""
from copy import deepcopy
import hashlib
import json
import random

import pytest

from phy.rt_sys.ue_services import OFFLOADED_COUNTER_DEFAULTS
from tests.test_ue_services import fixture, step


def config(fraction=.5, **mec):
    return {'task_cohort_ms': 100, 'mec': dict(enabled=True, local_gflops=2.,
        tasks_by_ue={'ue_0': dict(interval_ms=50, compute_gflop=.05,
                                 offload_fraction=fraction)}, **mec)}


def assert_route_conservation(row):
    assert row['offloaded_generated'] == (
        row['offloaded_completed'] + row['offloaded_failed'] + row['offloaded_pending'])
    assert row['offloaded_on_time_completed'] <= row['offloaded_completed']
    assert row['offloaded_generated'] <= row['generated']


def test_route_latency_uses_real_completion_events_and_excludes_local_cpu_tasks():
    _, model = fixture(config())
    routes, completions = {}, []
    for slot in range(1, 501):
        step(model, slot, deliver=True)
        routes.update({tid: t['offloaded'] for tid, t in model.tasks.items()})
        completions.extend(deepcopy(e) for e in model.events if e['kind'] == 'task_completed')
    offloaded = [e['latency_ms'] for e in completions if routes[e['task_id']]]
    local = [e['latency_ms'] for e in completions if not routes[e['task_id']]]
    assert offloaded and local
    assert local == pytest.approx([25.] * len(local))
    metrics = model.reports['ue_0']['ue_task_metrics']
    assert metrics['offloaded_completed'] == len(offloaded)
    assert metrics['offloaded_generated'] == metrics['offloaded']
    assert metrics['offloaded_completion_latency_ms_sum'] == pytest.approx(sum(offloaded))
    assert metrics['offloaded_mean_completion_latency_ms'] == pytest.approx(sum(offloaded) / len(offloaded))
    assert metrics['completion_latency_ms_sum'] == pytest.approx(sum(offloaded) + sum(local))
    assert metrics['offloaded_mean_completion_latency_ms'] != pytest.approx(metrics['mean_completion_latency_ms'])
    assert metrics['offloaded_success_ratio'] == 1.
    assert_route_conservation(metrics)
    for row in model.reports['ue_0']['ue_task_cohorts']:
        assert_route_conservation(row)


def test_all_local_tasks_have_no_offloaded_latency_observation():
    _, model = fixture(config(0.))
    for slot in range(1, 101):
        step(model, slot, deliver=True)
    metrics = model.reports['ue_0']['ue_task_metrics']
    assert metrics['completed'] == 2
    assert metrics['mean_completion_latency_ms'] == pytest.approx(25.)
    assert metrics['offloaded_mean_completion_latency_ms'] is None
    assert metrics['offloaded_success_ratio'] is None
    assert all(metrics[key] == 0 for key in OFFLOADED_COUNTER_DEFAULTS)
    for row in model.reports['ue_0']['ue_task_cohorts']:
        assert all(row[key] == 0 for key in OFFLOADED_COUNTER_DEFAULTS)


def test_offloaded_pending_spans_upload_backhaul_compute_and_actual_result_ack():
    cfg = config(1.)
    cfg['mec']['tasks_by_ue']['ue_0']['interval_ms'] = 1000
    state, model = fixture(cfg)
    stages = set()
    for slot in range(1, 21):
        step(model, slot)
        stages.update(t['stage'] for t in model.tasks.values())
        metrics = model.reports['ue_0']['ue_task_metrics']
        assert metrics['offloaded_generated'] == metrics['offloaded_pending'] == 1
        assert metrics['offloaded_completed'] == 0
        assert metrics['offloaded_mean_completion_latency_ms'] is None
        assert_route_conservation(metrics)
    assert stages == {'upload', 'backhaul', 'compute', 'downlink'}
    before = deepcopy(model.reports['ue_0']['ue_task_cohorts'][0])
    ledger = state.ues[0].packet_ledger
    ledger.reserve('result', [('lc_0', 200)])
    ledger.acknowledge('result', 20)
    ledger.acknowledge('result', 21)
    model._reports()
    metrics = model.reports['ue_0']['ue_task_metrics']
    row = model.reports['ue_0']['ue_task_cohorts'][0]
    assert before['offloaded_pending'] == 1
    assert row['offloaded_pending'] == 0 and row['offloaded_completed'] == 1
    assert metrics['offloaded_completed'] == 1
    assert metrics['offloaded_completion_latency_ms_sum'] == pytest.approx(20.)
    assert metrics['offloaded_success_ratio'] == 1.
    assert_route_conservation(row)


def test_pre_route_admission_failure_stays_in_all_task_guard_without_rng_draw():
    cfg = config(1., max_queued_tasks=1, uplink_bps_per_cell=8000)
    cfg['mec']['tasks_by_ue']['ue_0']['interval_ms'] = 1
    _, model = fixture(cfg)
    step(model, 1)
    step(model, 2)
    metrics = model.reports['ue_0']['ue_task_metrics']
    assert (metrics['generated'], metrics['failed'], metrics['pending']) == (2, 1, 1)
    assert (metrics['offloaded_generated'], metrics['offloaded_failed'], metrics['offloaded_pending']) == (1, 0, 1)
    assert metrics['success_ratio'] == 0.
    expected = random.Random('1:ue-services:ue_0')
    expected.random()
    assert model.rng['ue_0'].getstate() == expected.getstate()
    assert_route_conservation(metrics)
    assert_route_conservation(model.reports['ue_0']['ue_task_cohorts'][0])


def test_route_assigned_upload_queue_failure_is_an_offloaded_failure():
    cfg = config(1., max_queued_tasks=1, uplink_bps_per_cell=8000)
    cfg['mec']['tasks_by_ue']['ue_1'] = deepcopy(cfg['mec']['tasks_by_ue']['ue_0'])
    _, model = fixture(cfg, count=2)
    step(model, 1)
    metrics = model.reports['ue_1']['ue_task_metrics']
    assert (metrics['generated'], metrics['failed']) == (1, 1)
    assert (metrics['offloaded_generated'], metrics['offloaded_failed'], metrics['offloaded_pending']) == (1, 1, 0)
    assert metrics['offloaded_success_ratio'] == 0.
    assert metrics['offloaded_mean_completion_latency_ms'] is None
    assert_route_conservation(metrics)


def test_result_overflow_is_an_offloaded_failure_not_a_completed_latency():
    cfg = config(1.)
    cfg['mec']['tasks_by_ue']['ue_0']['interval_ms'] = 1000
    state, model = fixture(cfg)
    state.ues[0].logical_channels[0].max_queue_bytes = 100
    for slot in range(1, 21):
        step(model, slot)
    metrics = model.reports['ue_0']['ue_task_metrics']
    assert metrics['result_overflow'] == 1
    assert (metrics['offloaded_generated'], metrics['offloaded_failed'], metrics['offloaded_pending']) == (1, 1, 0)
    assert metrics['offloaded_completed'] == 0
    assert metrics['offloaded_mean_completion_latency_ms'] is None
    assert_route_conservation(model.reports['ue_0']['ue_task_cohorts'][0])


def test_late_offloaded_ack_uses_actual_latency_and_counts_unsuccessful_route():
    cfg = config(1.)
    cfg['mec']['tasks_by_ue']['ue_0'].update(interval_ms=1000, deadline_ms=5)
    state, model = fixture(cfg)
    for slot in range(1, 21):
        step(model, slot)
    state.ues[0].packet_ledger.reserve('result', [('lc_0', 200)])
    state.ues[0].packet_ledger.acknowledge('result', 20)
    model._reports()
    metrics = model.reports['ue_0']['ue_task_metrics']
    assert metrics['offloaded_completed'] == 1 and metrics['offloaded_on_time_completed'] == 0
    assert metrics['offloaded_mean_completion_latency_ms'] == pytest.approx(20.)
    assert metrics['offloaded_success_ratio'] == 0.
    assert metrics['deadline_misses'] == 1


def test_route_telemetry_preserves_pre_change_physics_queues_rng_and_legacy_counters():
    # Reverified against released source6a37d9f8 (before stage diagnostics) and
    # the current class under the same packet ledger. Both give this identical
    # snapshot; the older d6d6... expectation was already stale in that release.
    # Excludes only the six route counters and task route tag. Stage diagnostics
    # live in a separate observer, so no physical fields are excluded for them.
    cfg = {'task_cohort_ms': 100, 'mec': {'enabled': True, 'max_queued_tasks': 3,
        'uplink_bps_per_cell': 80000, 'local_gflops': 1., 'server_gflops': .1,
        'tasks_by_ue': {f'ue_{i}': dict(interval_ms=20, input_bytes=1000,
            result_bytes=200, compute_gflop=.005, deadline_ms=40,
            offload_fraction=(.4, .7)[i]) for i in range(2)}}}
    state, model = fixture(cfg, count=2)
    for slot in range(1, 501):
        state.connections['ue_0'].connection_state = 'connected' if not 80 <= slot < 130 else 'handover_interruption'
        step(model, slot, deliver=slot % 3 == 0)
    payload = dict(
        stats={uid: {k: v for k, v in row.items() if k not in OFFLOADED_COUNTER_DEFAULTS}
               for uid, row in model.stats.items()}, energy=model.energy,
        server_stats=model.server_stats,
        ledgers={uid: u.packet_ledger.stats for uid, u in model.users.items()},
        rng={uid: r.getstate() for uid, r in model.rng.items()},
        tasks={tid: {k: v for k, v in task.items() if k != 'offloaded'}
               for tid, task in model.tasks.items()}, future=model.future,
        local={k: list(v) for k, v in model.local.items()},
        upload={k: list(v) for k, v in model.upload.items()},
        edge={k: list(v) for k, v in model.edge.items()})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == '9937eeb742772032fb8bed6ee7d230c8ff8245a80554740aa949ca0d874e4fd0'
    for uid, metrics in model.stats.items():
        assert_route_conservation(metrics)
        assert metrics['offloaded_generated'] == metrics['offloaded']
        for row in model.reports[uid]['ue_task_cohorts']:
            assert_route_conservation(row)
