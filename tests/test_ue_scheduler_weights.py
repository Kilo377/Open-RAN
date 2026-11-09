"""UID scheduler weights: real consumer, atomic controls and HO-safe leases."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from models.action import ConfigAction
from models.policy import PolicyUpdate
from phy.rt_sys.control import audit_action, runtime_control_values
from phy.rt_sys.paper_controls import DEFAULTS, apply_values, validate_transaction
from ran.local_executor import LocalExecutor
from ran.scheduler import SimpleScheduler
from ric.controller_manager import ControllerManager
from ric.interfaces.e2_like import E2LikeInterface
from ric.near_rt.pools.chongyu.latency_scheduler_rf import LatencySchedulerRF
from ric.near_rt.pools.chongyu.ue_energy_rf import ACTIONS as ENERGY_ACTIONS
from ric.near_rt.rule_based_v1.leases import RuleLeases
from ric.near_rt.rule_based_v1.telemetry import executed_configs
from ric.xapp_contract import XAppContract, validate_action
from state.ue_state import UeState


def user(uid):
    return UeState(uid, 1000, 0., 10., 10., .1, 0, 'embb', 'video', 100., 1)


def state():
    cells = {}
    for cid in ('cell_0', 'cell_1'):
        cell = NS(cell_id=cid, tx_power_dbm=30., power_cap_dbm=None,
            nominal_tx_power_dbm=30., cell_sleep_mode='active',
            current_codebook_id=0, current_beam_profile_id=0,
            steering_hysteresis_db=3., **deepcopy(DEFAULTS))
        scheduler = NS(packing_strategy='frequency_first', scheduler_mode='round_robin',
            scheduler_policy_hint='balanced', latency_bias=None, slice_weight_adjustment={})
        cells[cid] = NS(cell=cell, scheduler=scheduler, resource_grid=NS(n_prbs_total=8))
    return NS(slot_id=1, ues=[user('ue_0'), user('ue_1')], managed_cells=cells,
        connections={'ue_0': NS(serving_cell_id='cell_0'),
                     'ue_1': NS(serving_cell_id='cell_1')},
        cell=cells['cell_0'].cell, scheduler=cells['cell_0'].scheduler,
        paper=NS(adapter=NS(config={'tx_cols': 2}), reset_triggers=lambda cid: None))


def transaction(s, action):
    """Use the simulator's validation-before-execution transaction ordering."""
    executor = LocalExecutor()
    targets = executor.resolve_target_cell_ids(s, action)
    reason = validate_transaction(s, action, targets)
    if reason is None:
        applied = executor.apply_action(s, action)
        apply_values(s, action, applied)
    return reason


def observed(s, *, cohorts=(), energy=1., bits=100000):
    configs = executed_configs(s)
    users = [NS(ue_id=u.ue_id, serving_cell_id=s.connections[u.ue_id].serving_cell_id,
        queue_bytes=1000, hol_delay_ms=2., throughput_mbps=1., offered_rate_bps=1e6,
        reference_sinr_db=10., ue_energy_total_j=energy,
        delivered_bits_total=bits, offered_bits_total=bits, ue_active_fraction=.8,
        ue_task_metrics={'pending': 0, 'offloaded_pending': 0} if u.ue_id == 'ue_0' else None,
        ue_task_cohorts=deepcopy(cohorts) if u.ue_id == 'ue_0' else None,
        ue_service_policy=deepcopy(ENERGY_ACTIONS[1]),
        kpi_metadata={'ue_energy_total_j': {'produced_at_slot': s.slot_id, 'valid': True}})
        for u in s.ues]
    return NS(slot_id=s.slot_id, cell_id='cell_0', paper_interfaces_enabled=True,
        ue_observations=users, total_queue_bytes=2000,
        cell_observations=[NS(cell_id=cid, total_queue_bytes=1000, executed_config=cfg,
            kpi_metadata={'executed_config': {'produced_at_slot': s.slot_id, 'valid': True}})
            for cid, cfg in configs.items()],
        paper_report={'values': {'cell': {cid: {'executed_config': cfg}
                                          for cid, cfg in configs.items()}}})


def test_default_weights_and_executed_maps_exist_without_mec():
    s = state()
    assert not hasattr(s, 'ue_services')
    assert all(u.scheduler_weight == 1. for u in s.ues)
    reports = executed_configs(s)
    assert reports['cell_0']['ue_scheduler_weights'] == {'ue_0': 1.}
    assert reports['cell_1']['ue_scheduler_weights'] == {'ue_1': 1.}
    assert runtime_control_values(s, ['cell_0'])['cell_0']['ue_scheduler_weights'] == {'ue_0': 1.}
    reports['cell_0']['ue_scheduler_weights']['ue_0'] = 2.
    assert s.ues[0].scheduler_weight == 1.


def test_new_weight_preserves_legacy_positional_harq_argument():
    from state.harq_state import HarqState
    harq = HarqState()
    ue = UeState('legacy', 0, 0., 10., 10., .1, 0, 'embb', 'video', 100., 1, harq)
    assert ue.harq is harq and ue.scheduler_weight == 1.


def test_weight_control_has_scheduler_consumer_and_public_contract():
    s = state()
    action = ConfigAction(target_cell_id='cell_0', ue_scheduler_weights={'ue_0': 2.})
    rows = audit_action(action)
    assert [(r['field'], r['route']) for r in rows] == [('ue_scheduler_weights', 'scheduler')]
    contract = XAppContract((), frozenset({'ue_scheduler_weights'}), cell_ids=('cell_0',))
    assert contract.available
    assert not validate_action(action, contract, observed(s))
    assert transaction(s, action) is None
    assert s.ues[0].scheduler_weight == 2.
    assert not hasattr(s.cell, 'ue_scheduler_weights')
    assert s.ues[1].scheduler_weight == 1.


@pytest.mark.parametrize('weights', [
    {'ue_0': True}, {'ue_0': '2'}, {'ue_0': None}, {'ue_0': float('nan')},
    {'ue_0': float('inf')}, {'ue_0': .49}, {'ue_0': 2.01},
    {'ue_0': 2., 'missing': 1.}, {'ue_0': 2., 'ue_1': 1.}, [1.],
])
def test_invalid_map_rejects_the_entire_power_and_weight_transaction(weights):
    s = state()
    action = ConfigAction(target_cell_id='cell_0', tx_power_dbm=20.,
                          ue_scheduler_weights=weights)
    assert transaction(s, action) is not None
    assert s.cell.tx_power_dbm == 30.
    assert [u.scheduler_weight for u in s.ues] == [1., 1.]
    contract = XAppContract((), frozenset({'ue_scheduler_weights', 'tx_power_dbm'}))
    assert validate_action(action, contract, observed(s))


def test_cluster_map_accepts_only_known_ues_and_scalar_boundary_weights():
    s = state()
    action = ConfigAction(apply_scope='cluster', ue_scheduler_weights={'ue_0': .5, 'ue_1': 2})
    assert transaction(s, action) is None
    assert [u.scheduler_weight for u in s.ues] == [.5, 2.]
    assert all(not hasattr(r.cell, 'ue_scheduler_weights') for r in s.managed_cells.values())


def test_weight_transaction_requires_opt_in_paper_backend():
    s = state()
    s.paper = None
    assert transaction(s, ConfigAction(ue_scheduler_weights={'ue_0': 2.})) == 'paper_interfaces_disabled'
    assert s.ues[0].scheduler_weight == 1.


def test_real_scheduler_weight_changes_target_score_without_inventing_capacity():
    s = state()
    scheduler = SimpleScheduler()
    target, other = s.ues
    base = scheduler._score_for_mode(s, target, 'latency_aware')
    other_base = scheduler._score_for_mode(s, other, 'latency_aware')
    rr_base = scheduler._score_for_mode(s, target, 'round_robin', round_robin_rank=0)
    assert transaction(s, ConfigAction(target_cell_id='cell_0', ue_scheduler_weights={'ue_0': 2.})) is None
    assert scheduler._score_for_mode(s, target, 'latency_aware') == pytest.approx(2 * base)
    assert scheduler._score_for_mode(s, other, 'latency_aware') == other_base
    assert scheduler._score_for_mode(s, target, 'round_robin', round_robin_rank=0) == rr_base
    assert scheduler.total_prbs_per_slot == 48 and scheduler.total_symbols_per_slot == 14


def lease_manager():
    app = NS(name='x12', rule_based_v1=True, library_instance=False,
             allowed_cells=('cell_0', 'cell_1'), controlled_fields=['ue_scheduler_weights'], priority=20)
    m = NS(e2_interfaces=[NS(xapp=app)], current_action_by_scope={},
           enabled={'x12'}, instance_selection=NS(mode='exclusive'))
    m._is_xapp_enabled = lambda name: name in m.enabled
    return m


def reconcile(leases, manager, s, actions=()):
    winners, scopes, dropped = {}, {}, []
    leases.reconcile(manager, observed(s), winners, scopes, list(actions), dropped)
    return winners, dropped


def test_uid_weight_follows_handover_and_off_restores_at_new_serving_cell():
    s = state()
    leases, manager = RuleLeases(), lease_manager()
    action = ConfigAction(source_name='x12', target_cell_id='cell_0',
                          ue_scheduler_weights={'ue_0': 2.})
    winners, _ = reconcile(leases, manager, s, [action])
    assert winners['cell:cell_0']['ue_scheduler_weights'][1] == {'ue_0': 2.}
    manager.current_action_by_scope['cell:cell_0'] = deepcopy(action)
    assert transaction(s, action) is None
    s.connections['ue_0'].serving_cell_id = 'cell_1'
    assert executed_configs(s)['cell_1']['ue_scheduler_weights'] == {'ue_0': 2., 'ue_1': 1.}
    assert transaction(s, action) == 'ue_scheduler_weight_target_outside_cell_scope'
    held, dropped = reconcile(leases, manager, s)
    assert not any(fields for fields in held.values())
    assert manager.current_action_by_scope['cell:cell_0'].ue_scheduler_weights is None
    assert any(d.reason == 'ue_service_request_moved_cell' for d in dropped)
    manager.enabled.clear()
    restored, _ = reconcile(leases, manager, s)
    assert restored['cell:cell_1']['ue_scheduler_weights'][1] == {'ue_0': 1.}
    assert transaction(s, ConfigAction(target_cell_id='cell_1', ue_scheduler_weights=
                                      restored['cell:cell_1']['ue_scheduler_weights'][1])) is None
    assert [u.scheduler_weight for u in s.ues] == [1., 1.]
    reconcile(leases, manager, s)
    assert not leases.ue_service_owned


def test_off_restore_does_not_change_another_ues_weight():
    s = state()
    s.connections['ue_1'].serving_cell_id = 'cell_0'
    s.ues[1].scheduler_weight = .75
    leases, manager = RuleLeases(), lease_manager()
    action = ConfigAction(source_name='x12', target_cell_id='cell_0', ue_scheduler_weights={'ue_0': 2.})
    reconcile(leases, manager, s, [action])
    transaction(s, action)
    manager.enabled.clear()
    restored, _ = reconcile(leases, manager, s)
    values = restored['cell:cell_0']['ue_scheduler_weights'][1]
    assert values == {'ue_0': 1.}
    transaction(s, ConfigAction(target_cell_id='cell_0', ue_scheduler_weights=values))
    assert s.ues[1].scheduler_weight == .75


def test_x12_manager_leases_execution_consumer_and_mature_task_label_chain():
    s = state()
    app = LatencySchedulerRF(['cell_0'])
    app.initial_bindings = {'cell_0': {app.category_id: app.name}}
    app.windows['cell_0'] = 1  # The actual second exploration arm is weight 2.
    manager = ControllerManager([], [E2LikeInterface(app)], 1, 100000)
    manager._initial_rapp_run_completed = True
    manager.current_action_by_scope['cell:cell_0'] = ConfigAction(
        target_cell_id='cell_0', scheduler_policy_hint='balanced')
    scheduler = SimpleScheduler()
    before = scheduler._score_for_mode(s, s.ues[0], 'latency_aware')

    def tick(slot, **values):
        s.slot_id = slot
        result = manager.update(slot, observed(s, **values))
        for action in result.config_actions:
            assert transaction(s, action) is None
        return result

    first = tick(10)
    received = next(a for a in first.trace.received_config_actions if a.source_name == app.name)
    assert received.scheduler_policy_hint is received.latency_bias is None
    assert received.ue_scheduler_weights == {'ue_0': 2.}
    assert s.scheduler.scheduler_mode == 'latency_aware'
    assert s.ues[0].scheduler_weight == 2.
    assert scheduler._score_for_mode(s, s.ues[0], 'latency_aware') == pytest.approx(2 * before)
    assert s.ues[1].scheduler_weight == 1.
    assert manager.rule_leases.owned['cell_0', 'ue_scheduler_weights'] == app.name
    tick(20)
    assert app.pending['cell_0']['confirmed']
    tick(200)
    # Explicit unit-observation outcome; this fixture does not claim NR execution.
    cohort = dict(index=1, generated=2, completed=2, on_time_completed=2,
        failed=0, pending=0, completion_latency_ms_sum=30., mature_after_s=.3,
        offloaded_generated=2, offloaded_completed=2, offloaded_on_time_completed=2,
        offloaded_failed=0, offloaded_pending=0, offloaded_completion_latency_ms_sum=30.,
        mixed_policy=False, drx_config=deepcopy(ENERGY_ACTIONS[1]['drx']), offload_fraction=.5)
    tick(600, cohorts=[cohort], energy=1.2, bits=300000)
    assert app.learner.sample_count == 1
    sample = next(e for e in app.events if e['kind'] == 'measured_sample')
    assert sample['action'] == 3
    assert sample['targets'] == pytest.approx([1., 1., 0., 15.])
    manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}), slot_id=601)
    tick(620)
    assert s.scheduler.scheduler_mode == 'round_robin'
    assert s.ues[0].scheduler_weight == 1.
