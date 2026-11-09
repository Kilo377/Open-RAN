"""QoS deadline actions change real MAC/contract allocation, not packet outcomes."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from models.action import ConfigAction
from models.scheduling import UeSchedulingDemand
from phy.rt_sys.paper_controls import DEFAULTS, apply_values, validate_transaction
from phy.rt_sys.scheduler import RtSysScheduler
from phy.rt_sys.tx_contract import TbTxContract, data_re
from ran.local_executor import LocalExecutor
from ran.scheduler import SimpleScheduler
from ric.near_rt.rule_based_v1.leases import RuleLeases
from ric.xapp_contract import XAppContract, validate_action
from state.scheduler_state import SchedulerState
from state.ue_state import UeState


def user(uid, *, priority=1, budget=100., queue=1000):
    return UeState(uid, queue, 2., 10., 10., .1, 0, 'urllc' if budget <= 20 else 'embb',
                   'cbr' if budget <= 20 else 'video', budget, priority)


def runtime(users, mode='qos_deadline'):
    cell = NS(cell_id='cell_0', bandwidth_hz=10e6, tx_power_dbm=30., power_cap_dbm=None,
              nominal_tx_power_dbm=30., cell_sleep_mode='active', current_codebook_id=0,
              current_beam_profile_id=0, steering_hysteresis_db=3., **deepcopy(DEFAULTS))
    scheduler = SchedulerState(scheduler_mode=mode)
    managed = NS(cell=cell, scheduler=scheduler, resource_grid=NS(n_prbs_total=24))
    return NS(slot_id=10, slot_duration_ms=.5, ues=users, cell=cell, scheduler=scheduler,
              managed_cells={'cell_0': managed},
              connections={u.ue_id: NS(serving_cell_id='cell_0') for u in users},
              control=NS(active_policy_constraints=[]),
              paper=NS(adapter=NS(config={'tx_cols': 2}), reset_triggers=lambda cid: None))


def test_deadline_urgency_grows_with_consumed_budget_not_raw_queue():
    target = user('u', priority=3, budget=10., queue=137)
    embb = user('e', queue=240000)
    state = runtime([target, embb])
    scheduler = SimpleScheduler(total_prbs_per_slot=24)
    values = [scheduler._qos_deadline_score(state, target,
              UeSchedulingDemand('u', hol_delay_ms=hol, priority=3)) for hol in (0., 5., 9.)]
    assert values[1] == pytest.approx(2 * values[0])
    assert values[2] == pytest.approx(10 * values[0])
    baseline = scheduler._qos_deadline_score(state, embb)
    embb.queue_bytes = 10**12
    assert scheduler._qos_deadline_score(state, embb) == baseline
    assert values[0] > baseline
    # Non-critical eMBB HOL does not trigger the URLLC urgency factor.
    assert scheduler._qos_deadline_score(state, embb, UeSchedulingDemand('e', hol_delay_ms=99., priority=1)) == baseline


def test_overdue_weight_is_finite_and_weight4_remains_opt_in():
    target = user('u', priority=3, budget=10.)
    state = runtime([target])
    scheduler = SimpleScheduler()
    demand = UeSchedulingDemand('u', hol_delay_ms=10000., priority=3)
    before = scheduler._score_for_mode(state, target, 'qos_deadline', demand=demand)
    target.scheduler_weight = 4.
    assert scheduler._score_for_mode(state, target, 'qos_deadline', demand=demand) == 4 * before
    assert np.isfinite(before) and before > 0
    assert scheduler._score_for_mode(state, target, 'round_robin', round_robin_rank=0) == 3.
    with pytest.raises(ValueError, match='Invalid UE scheduler weight'):
        scheduler._score_for_mode(state, target, 'latency_aware', demand=demand)


def test_real_mac_selection_uses_new_score_and_keeps_legacy_rr_turn():
    target = user('u', priority=3, budget=10., queue=137)
    embb = user('e', queue=240000)
    scheduler = SimpleScheduler(total_prbs_per_slot=24, max_ues_per_slot=1)
    state = runtime([embb, target])
    state.paper = None
    assert scheduler.run(state) == ['u']
    trace = state.scheduler.last_stage_trace
    assert trace.selection_reasons_by_ue['u'] == 'selected_for_qos_deadline_priority'
    assert state.scheduler.grants[0].priority_score > 0
    assert hasattr(state.scheduler, 'qos_average_rate_mbps')
    # Identical eligible order under untouched native mode still chooses eMBB.
    state = runtime([user('e', queue=240000), user('u', priority=3, budget=10., queue=137)], mode='round_robin')
    state.paper = None
    assert scheduler.run(state) == ['e']
    assert not hasattr(state.scheduler, 'qos_average_rate_mbps')


def test_actual_delivered_rate_ema_updates_once_per_slot():
    target = user('u')
    state = runtime([target])
    scheduler = SimpleScheduler()
    scheduler._refresh_qos_rate_history(state)
    initial = state.scheduler.qos_average_rate_mbps['u']
    target.throughput_mbps = 10.
    scheduler._refresh_qos_rate_history(state)
    assert state.scheduler.qos_average_rate_mbps['u'] == initial
    state.slot_id += 1
    scheduler._refresh_qos_rate_history(state)
    assert state.scheduler.qos_average_rate_mbps['u'] == pytest.approx(.975 * initial + .025 * 10.)


def test_no_hidden_byte_weight_is_added_during_real_grant_building():
    target = user('u', priority=3, budget=10., queue=137)
    embb = user('e', queue=240000)
    state = runtime([embb, target])
    state.paper = None
    scheduler = SimpleScheduler(total_prbs_per_slot=24, max_ues_per_slot=2)
    assert set(scheduler.run(state)) == {'u', 'e'}
    grants = {g.ue_id: g for g in state.scheduler.grants}
    assert grants['u'].priority_score > grants['e'].priority_score
    before = grants['e'].priority_score
    state = runtime([user('e', queue=24000000), user('u', priority=3, budget=10., queue=137)])
    state.paper = None
    scheduler.run(state)
    assert next(g for g in state.scheduler.grants if g.ue_id == 'e').priority_score == before


def packing(monkeypatch):
    import phy.rt_sys.tx_contract as contracts
    monkeypatch.setattr(contracts, 'nr_budget', lambda table, mcs, count: (2, 2 * count, 8 * max(1, count // 8)))
    monkeypatch.setattr(contracts, 'codec_format_supported', lambda *args: True)
    monkeypatch.setattr(SimpleScheduler, '_apply_spatial_multiplexing', lambda *args: None)
    scheduler = RtSysScheduler(total_prbs_per_slot=24, total_symbols_per_slot=14,
        max_ues_per_slot=8, min_prbs_per_grant=2, min_symbols_per_grant=2)
    scheduler._contracts = scheduler._paper = True
    state = NS(slot_id=10, cell=NS(ue_prb_limits={}, prb_power_w=None),
               scheduler=NS(scheduler_mode='qos_deadline', packing_strategy='frequency_first'))
    return scheduler, state


def grant(uid, score, group=None, retx=False):
    return NS(ue_id=uid, is_retx=retx, priority_score=score, spatial_group_id=group,
        spatial_group_size=2 if group is not None else 1, target_mcs=5,
        mcs_table_id='qam64', dmrs_pattern_id='dmrs_type1_default', constraint_hits=[],
        harq_process_id=0, allocated_prbs=1, allocated_symbols=14, start_prb=0,
        start_symbol=0, grant_bytes_budget=0)


def radio_user(uid):
    return NS(ue_id=uid, rank=1, ri=1, harq=NS(processes=[], events=[]))


def assert_no_overlap(grants):
    occupied = np.zeros((14, 24), dtype=bool)
    groups = set()
    for g in grants:
        key = ('mu', g.spatial_group_id) if g.spatial_group_id is not None else ('su', g.ue_id)
        if key in groups:
            continue
        groups.add(key)
        assert 0 <= g.start_prb < 24 and 0 <= g.start_symbol < 14
        assert g.start_prb + g.allocated_prbs <= 24 and g.start_symbol + g.allocated_symbols <= 14
        rectangle = occupied[g.start_symbol:g.start_symbol + g.allocated_symbols,
                             g.start_prb:g.start_prb + g.allocated_prbs]
        assert not rectangle.any()
        rectangle[:] = True
    return occupied


def test_qos_priority_reaches_immutable_tb_resource_consumer(monkeypatch):
    scheduler, state = packing(monkeypatch)
    allocations = []
    for score in (1., 4.):
        grants = [grant('url', score), grant('embb', 1.)]
        scheduler._apply_spatial_multiplexing(state, grants, {g.ue_id: radio_user(g.ue_id) for g in grants})
        allocations.append([g.allocated_prbs for g in grants])
        assert assert_no_overlap(grants).all()
        assert all(g.grant_bytes_budget > 0 for g in grants)
    assert allocations[0] == [12, 12]
    assert allocations[1][0] > allocations[0][0]
    assert allocations[1][1] < allocations[0][1]


def test_mu_member_count_does_not_double_group_urgency(monkeypatch):
    scheduler, state = packing(monkeypatch)
    grants = [grant('a', 1., group=7), grant('b', 1., group=7), grant('url', 1.)]
    scheduler._apply_spatial_multiplexing(state, grants, {g.ue_id: radio_user(g.ue_id) for g in grants})
    assert [g.allocated_prbs for g in grants] == [12, 12, 12]
    assert grants[0].start_prb == grants[1].start_prb
    assert assert_no_overlap(grants).all()


def test_qos_keeps_original_retransmission_contract_and_power_holes(monkeypatch):
    scheduler, state = packing(monkeypatch)
    contract = TbTxContract('retx', 'cell_0', 5, 'qam64', 1, 4, 3,
        'dmrs_type1_default', data_re(4, 3, 'dmrs_type1_default'), 240, 128, 12, (('lc_0', 12),))
    original = deepcopy(contract)
    users = {uid: radio_user(uid) for uid in ('r', 'u', 'e')}
    users['r'].harq.processes = [NS(process_id=0, tx_contract=contract, retx_deadline_slot=15, last_tx_slot=8)]
    state.cell.prb_power_w = [1.] * 24
    state.cell.prb_power_w[12] = 0.
    grants = [grant('u', 4.), grant('r', .1, retx=True), grant('e', 1.)]
    scheduler._apply_spatial_multiplexing(state, grants, users)
    retx = next(g for g in grants if g.is_retx)
    assert contract == original and retx.tx_contract is contract
    assert (retx.allocated_prbs, retx.allocated_symbols, retx.requested_bytes) == (4, 3, 12)
    assert not assert_no_overlap(grants)[:, 12].any()


def public_observation(state):
    return NS(cell_id='cell_0', paper_interfaces_enabled=True,
        ue_observations=[NS(ue_id=u.ue_id, serving_cell_id='cell_0') for u in state.ues],
        cell_observations=[NS(cell_id='cell_0', executed_config={
            'scheduler_mode': state.scheduler.scheduler_mode,
            'ue_scheduler_weights': {u.ue_id: u.scheduler_weight for u in state.ues}})])


@pytest.mark.parametrize('mode,weight,valid', [
    ('qos_deadline', 4., True), ('qos_deadline', 4.01, False),
    ('qos_deadline', float('nan'), False), ('qos_deadline', True, False),
    ('latency_aware', 2.01, False), ('round_robin', 4., False),
])
def test_mode_aware_weight_range_in_public_and_transaction_validation(mode, weight, valid):
    state = runtime([user('u')], mode='round_robin')
    action = ConfigAction(target_cell_id='cell_0', scheduler_mode=mode, ue_scheduler_weights={'u': weight})
    contract = XAppContract((), frozenset({'scheduler_mode', 'ue_scheduler_weights'}), cell_ids=('cell_0',))
    assert (not validate_action(action, contract, public_observation(state))) == valid
    assert (validate_transaction(state, action, ['cell_0']) is None) == valid


def test_control_executes_high_weight_then_restores_full_native_bundle():
    state = runtime([user('u')], mode='round_robin')
    executor = LocalExecutor()
    for mode, weight in [('qos_deadline', 4.), ('round_robin', 1.)]:
        action = ConfigAction(target_cell_id='cell_0', scheduler_mode=mode, ue_scheduler_weights={'u': weight})
        assert validate_transaction(state, action, ['cell_0']) is None
        applied = executor.apply_action(state, action)
        apply_values(state, action, applied)
        assert state.scheduler.scheduler_mode == mode
        assert state.ues[0].scheduler_weight == weight
        assert state.scheduler.latency_bias is None
        assert not state.scheduler.slice_weight_adjustment


@pytest.mark.parametrize('requested', [dict(scheduler_mode='latency_aware'), dict(scheduler_policy_hint='latency_bias')])
def test_mode_only_exit_requires_reset_of_high_qos_weights(requested):
    state = runtime([user('u')])
    state.ues[0].scheduler_weight = 4.
    action = ConfigAction(target_cell_id='cell_0', **requested)
    contract = XAppContract((), frozenset({'scheduler_mode', 'scheduler_policy_hint', 'ue_scheduler_weights'}), cell_ids=('cell_0',))
    assert validate_transaction(state, action, ['cell_0']) == 'legacy_scheduler_requires_weight_reset'
    assert validate_action(action, contract, public_observation(state))
    action.ue_scheduler_weights = {'u': 1.}
    assert validate_transaction(state, action, ['cell_0']) is None
    assert not validate_action(action, contract, public_observation(state))


def test_random_conflict_bundle_accepts_new_atomic_mode_and_validates_range():
    action = ConfigAction(scheduler_mode='qos_deadline', ue_scheduler_weights={'u': 4., 'e': 1.})
    value = RuleLeases._scheduler_value(action, {'u', 'e'})
    assert value == {'scheduler_mode': 'qos_deadline', 'ue_scheduler_weights': {'u': 4., 'e': 1.}}
    action.scheduler_mode = 'latency_aware'
    with pytest.raises(ValueError, match='bundle weights'):
        RuleLeases._scheduler_value(action, {'u', 'e'})
