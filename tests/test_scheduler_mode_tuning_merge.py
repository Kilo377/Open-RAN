"""Merged scheduler tuning must follow the family the executor actually runs."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from models.action import ConfigAction, ControlDecision
from models.policy import PolicyUpdate
from ran.local_executor import LocalExecutor
from ric.controller_manager import ControllerManager
from ric.interfaces.e2_like import E2LikeInterface
from ric.near_rt.pools.base import PoolXApp
from ric.near_rt.pools.chongyu.ue_energy_rf import ACTIONS


def runtime(mode='round_robin', bias=None):
    scheduler = NS(scheduler_mode=mode, scheduler_policy_hint={
        'round_robin': 'balanced', 'latency_aware': 'latency_bias',
        'slice_aware': 'slice_bias'}[mode], latency_bias=bias,
        slice_weight_adjustment={})
    cell = NS(cell_id='cell_0')
    return NS(slot_id=1, cell=cell, scheduler=scheduler)


@pytest.mark.parametrize('mode,hint,expected_mode,expected_bias,expected_weights', [
    ('latency_aware', 'balanced', 'latency_aware', 2., {}),
    ('latency_aware', 'slice_bias', 'latency_aware', 2., {}),
    ('slice_aware', 'balanced', 'slice_aware', None, {'urllc': 1.5}),
    ('slice_aware', 'latency_bias', 'slice_aware', None, {'urllc': 1.5}),
    ('round_robin', 'latency_bias', 'round_robin', None, {}),
    ('round_robin', 'slice_bias', 'round_robin', None, {}),
    (None, 'latency_bias', 'latency_aware', 2., {}),
    (None, 'slice_bias', 'slice_aware', None, {'urllc': 1.5}),
    (None, 'balanced', 'round_robin', None, {}),
    (' ', ' LATENCY_BIAS ', 'latency_aware', 2., {}),
])
def test_merge_and_executor_agree_on_explicit_mode_precedence(
        mode, hint, expected_mode, expected_bias, expected_weights):
    manager = ControllerManager([], [], 1, 1000)
    winners = {'cell:cell_0': {
        key: (20, value, 'controller') for key, value in {
            'scheduler_mode': mode, 'scheduler_policy_hint': hint,
            'latency_bias': 2., 'slice_weight_adjustment': {'urllc': 1.5}
        }.items() if value is not None}}
    actions = manager._apply_field_winners(winners, {'cell:cell_0': ('cell', 'cell_0')})
    action = next(a for a in actions if a.target_cell_id == 'cell_0')
    assert action.latency_bias == expected_bias
    assert action.slice_weight_adjustment == expected_weights
    state = runtime()
    LocalExecutor().apply_action(state, action)
    assert state.scheduler.scheduler_mode == expected_mode
    assert state.scheduler.latency_bias == expected_bias
    assert state.scheduler.slice_weight_adjustment == expected_weights


def report(state, slot, *, cohorts=(), energy=1., bits=100000):
    cfg = deepcopy(vars(state.scheduler))
    user = NS(ue_id='ue_0', serving_cell_id='cell_0', queue_bytes=1000,
        hol_delay_ms=2., throughput_mbps=1., offered_rate_bps=1e6,
        reference_sinr_db=10., ue_energy_total_j=energy,
        delivered_bits_total=bits, offered_bits_total=bits,
        ue_active_fraction=.8, ue_task_metrics={'pending': 0},
        ue_task_cohorts=deepcopy(cohorts), ue_service_policy=deepcopy(ACTIONS[1]),
        kpi_metadata={'ue_energy_total_j': {'produced_at_slot': slot, 'valid': True}})
    cell = NS(cell_id='cell_0', total_queue_bytes=1000, executed_config=cfg,
        kpi_metadata={'executed_config': {'produced_at_slot': slot, 'valid': True}})
    return NS(slot_id=slot, ue_observations=[user], cell_observations=[cell],
        total_queue_bytes=1000,
        paper_report={'values': {'cell': {'cell_0': {'executed_config': cfg}}}})


class ModeBiasFixture(PoolXApp):
    """Test-only controller for the original explicit-mode/bias regression."""
    implementation_id = 'test_mode_bias'
    category_id = 'test_scheduler'
    writable_actions = ('scheduler_mode', 'latency_bias')

    def run(self, observation, policy_snapshot):
        return ControlDecision(config_actions=[ConfigAction(source_name=self.name,
            target_cell_id='cell_0', scheduler_mode='latency_aware', latency_bias=2.)])


def setup_mode_bias(state):
    app = ModeBiasFixture(['cell_0'])
    app.initial_bindings = {'cell_0': {app.category_id: app.name}}
    manager = ControllerManager([], [E2LikeInterface(app)], 1, 100000)
    manager._initial_rapp_run_completed = True
    # A previous default/withdrawal action can leave this harmless old hint.
    manager.current_action_by_scope['cell:cell_0'] = ConfigAction(
        target_cell_id='cell_0', scheduler_policy_hint='balanced')
    return manager, app


def tick(manager, state, slot, **values):
    state.slot_id = slot
    decision = manager.update(slot, report(state, slot, **values))
    for action in decision.config_actions:
        LocalExecutor().apply_action(state, action)
    return decision


def test_mode_bias_request_survives_merge_execution_and_lease_ownership():
    state = runtime()
    manager, app = setup_mode_bias(state)
    first = tick(manager, state, 10)
    received = next(a for a in first.trace.received_config_actions if a.source_name == app.name)
    assert received.scheduler_policy_hint is None  # No undeclared hint workaround.
    assert state.scheduler.scheduler_mode == 'latency_aware'
    assert state.scheduler.latency_bias == 2.
    assert all(manager.rule_leases.owned['cell_0', field] == app.name
               for field in ('scheduler_mode', 'latency_bias'))
    tick(manager, state, 20)
    assert state.scheduler.latency_bias == 2.


@pytest.mark.parametrize('native_mode,native_bias', [('round_robin', None), ('latency_aware', .75)])
def test_mode_bias_withdrawal_restores_the_actual_native_scheduler_family(native_mode, native_bias):
    state = runtime(native_mode, native_bias)
    manager, app = setup_mode_bias(state)
    tick(manager, state, 10)
    assert state.scheduler.latency_bias == 2.
    manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}), slot_id=11)
    tick(manager, state, 20)
    assert state.scheduler.scheduler_mode == native_mode
    assert state.scheduler.latency_bias == native_bias
    assert not any(owner == app.name for owner in manager.rule_leases.owned.values())


def test_normalization_is_per_cell_and_does_not_change_native_default():
    manager = ControllerManager([], [], 1, 1000)
    native = manager.current_action_by_scope['cell:__default__']
    native.scheduler_policy_hint = 'balanced'
    manager._apply_field_winners({'cell:cell_1': {
        'scheduler_mode': (20, 'latency_aware', 'x12'),
        'latency_bias': (20, 2., 'x12')}}, {'cell:cell_1': ('cell', 'cell_1')})
    assert native.scheduler_mode is None
    assert native.scheduler_policy_hint == 'balanced'
    assert native.latency_bias is None
    assert manager.current_action_by_scope['cell:cell_1'].latency_bias == 2.
