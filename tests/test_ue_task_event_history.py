"""Long-horizon offline birth/route evidence without changing simulation decisions."""
from copy import deepcopy
import random

import pytest

from experiments.task_event_metrics import TaskEventCollector
from tests.test_ue_services import fixture, step


def config(**extra):
    return dict(task_cohort_ms=100, mec=dict(enabled=True, max_queued_tasks=1,
        uplink_bps_per_cell=8000, tasks_by_ue={'ue_0': dict(interval_ms=100,
        compute_gflop=.01, offload_fraction=1., deadline_ms=100, **extra)}))


def test_old_offloaded_completion_reconstructs_after_public_and_internal_cohort_eviction():
    state, model = fixture(config())
    collector = TaskEventCollector(2.1, start_s=0., target_uids=('ue_0',))
    for slot in range(1, 2001):
        for event in step(model, slot).get('ue_service_events', []): collector.observe(event)
    assert 0 not in model.cohorts['ue_0']
    assert all(r['index'] != 0 for r in model.reports['ue_0']['ue_task_cohorts'])
    user = state.ues[0]
    assert user.packet_ledger.stats['pending_bytes'] == 200
    user.packet_ledger.reserve('late-result', [('lc_0', 200)])
    model.events.clear()
    user.packet_ledger.acknowledge('late-result', 2000)
    for event in model.events: collector.observe(event)
    row = collector.summary()['per_ue']['ue_0']['cohorts'][0]
    assert row['index'] == 0 and row['offloaded_completed'] == 1
    assert row['offloaded_pending'] == 0 and row['offloaded_on_time_completed'] == 0
    assert row['offloaded_completion_latency_ms_sum'] == pytest.approx(2000.)
    assert model.stats['ue_0']['offloaded_completed'] == 1


def test_preroute_rejection_is_nullable_and_does_not_draw_extra_random_numbers():
    _, model = fixture(config())
    reference = random.Random(); reference.setstate(model.rng['ue_0'].getstate())
    reference.random()  # Only the first admitted task reaches original routing.
    events = []
    for slot in range(1, 202): events.extend(deepcopy(step(model, slot).get('ue_service_events', [])))
    assert model.rng['ue_0'].getstate() == reference.getstate()
    births = [e for e in events if e['kind'] == 'task_generated']
    assert [e['offloaded'] for e in births] == [True, None, None]
    rejects = [e for e in events if e['kind'] == 'task_queue_overflow']
    assert len(rejects) == 2 and all(e['offloaded'] is None for e in rejects)
    collector = TaskEventCollector(.3, start_s=0., target_uids=('ue_0',))
    for event in events: collector.observe(event)
    all_tasks = collector.summary()['per_ue']['ue_0']['whole_run']
    assert all_tasks['generated'] == 3 and all_tasks['failed'] == 2
    assert all_tasks['offloaded_generated'] == 1 and all_tasks['offloaded_failed'] == 0


def test_mature_selection_excludes_startup_and_censored_births_without_fake_zero_mean():
    _, model = fixture(config())
    collector = TaskEventCollector(.2, start_s=.1, target_uids=('ue_0', 'ue_4'))
    for slot in range(1, 201):
        for event in step(model, slot).get('ue_service_events', []): collector.observe(event)
    data = collector.summary()['per_ue']
    assert data['ue_0']['cohorts'] == []
    assert data['ue_4']['cohorts'] == []
    assert data['ue_0']['whole_run']['pending'] == 1


def test_shared_queue_rejection_after_routing_preserves_true_route_on_both_events():
    cfg = config()
    cfg['mec']['tasks_by_ue']['ue_1'] = dict(cfg['mec']['tasks_by_ue']['ue_0'])
    _, model = fixture(cfg, count=2)
    trace = step(model, 1)
    routed = [e for e in trace['ue_service_events'] if e['ue_id'] == 'ue_1']
    assert [e['kind'] for e in routed] == ['task_generated', 'task_queue_overflow']
    assert all(e['offloaded'] is True for e in routed)
    collector = TaskEventCollector(.3, start_s=0., target_uids=('ue_0', 'ue_1'))
    for event in trace['ue_service_events']: collector.observe(event)
    row = collector.summary()['per_ue']['ue_1']['whole_run']
    assert row['offloaded_generated'] == row['offloaded_failed'] == 1
    assert row['unassigned'] == 0


def test_missing_route_duplicate_and_inconsistent_terminal_records_fail_closed():
    good = dict(kind='task_generated', task_id=1, ue_id='ue_0', time_s=.001,
        created_s=.001, deadline_s=.1, cohort_index=0, offloaded=True)
    collector = TaskEventCollector(1., start_s=0., target_uids=('ue_0',))
    bad = dict(good); bad.pop('offloaded')
    with pytest.raises(ValueError, match='provenance'): collector.observe(bad)
    collector.observe(good)
    with pytest.raises(ValueError, match='Repeated'): collector.observe(good)
    terminal = dict(good, kind='task_completed', time_s=.9, latency_ms=899.)
    bad = dict(terminal, offloaded=False)
    with pytest.raises(ValueError, match='immutable'): collector.observe(bad)
    collector.observe(terminal)
    with pytest.raises(ValueError, match='Repeated'): collector.observe(terminal)


def test_event_schema_only_adds_provenance_without_changing_legacy_values():
    _, model = fixture(config())
    trace = step(model, 1)
    event = trace['ue_service_events'][0]
    assert {k:v for k,v in event.items() if k not in ('created_s','deadline_s','offloaded','cohort_index')} == dict(
        kind='task_generated', task_id=1, ue_id='ue_0', time_s=0.)
    assert event['created_s'] == 0. and event['deadline_s'] == .1
    assert event['cohort_index'] == 0 and event['offloaded'] is True


def test_exact_generation_endpoint_belongs_to_new_cohort_with_float_tolerance():
    collector = TaskEventCollector(.5, start_s=0., target_uids=('ue_0',))
    event = dict(kind='task_generated', task_id=1, ue_id='ue_0', time_s=.1,
        created_s=.1, deadline_s=.1, cohort_index=0, offloaded=False)
    with pytest.raises(ValueError, match='cohort mismatch'): collector.observe(event)
    event['cohort_index'] = 1
    collector.observe(event)
    collector.observe(dict(event, task_id=2, created_s=.3-.2, time_s=.3-.2))
    rows = collector.summary()['per_ue']['ue_0']['cohorts']
    assert len(rows) == 1 and rows[0]['index'] == 1 and rows[0]['generated'] == 2
