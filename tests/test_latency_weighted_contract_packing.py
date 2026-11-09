"""Priority reaches real contract rectangles without altering native RR packing.

Codec math and MU grouping are isolated fixtures; production budget/shape
functions are still called by the repacker. Existing NR integration tests
exercise those functions with Sionna itself.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from phy.rt_sys.scheduler import RtSysScheduler
from phy.rt_sys.tx_contract import TbTxContract, data_re
from ran.scheduler import SimpleScheduler


def setup(monkeypatch, mode='latency_aware', strategy='frequency_first', prbs=24, symbols=14):
    import phy.rt_sys.tx_contract as contracts
    monkeypatch.setattr(contracts, 'nr_budget', lambda table, mcs, count: (2, 2 * count, 8 * max(1, count // 8)))
    monkeypatch.setattr(contracts, 'codec_format_supported', lambda *args: True)
    # Spatial grouping is independently tested. Here group identities are
    # explicit so these tests isolate the resource consumer after grouping.
    monkeypatch.setattr(SimpleScheduler, '_apply_spatial_multiplexing', lambda *args: None)
    scheduler = RtSysScheduler(total_prbs_per_slot=prbs, total_symbols_per_slot=symbols,
        max_ues_per_slot=8, min_prbs_per_grant=2, min_symbols_per_grant=2)
    scheduler._contracts = True
    scheduler._paper = True
    state = NS(slot_id=5, cell=NS(ue_prb_limits={}, prb_power_w=None, ret_enabled=False),
               scheduler=NS(scheduler_mode=mode, packing_strategy=strategy))
    return scheduler, state


def user(uid, **values):
    return NS(ue_id=uid, rank=1, ri=1, num_rx_ant=1,
              harq=NS(processes=[], pending_retx_bytes=0, events=[]), **values)


def grant(uid, score=1., group=None, retx=False):
    return NS(ue_id=uid, is_retx=retx, priority_score=score,
              spatial_group_id=group, spatial_group_size=2 if group is not None else 1,
              target_mcs=5, mcs_table_id='qam64', dmrs_pattern_id='dmrs_type1_default',
              constraint_hits=[], harq_process_id=0, allocated_prbs=1, allocated_symbols=14,
              start_prb=0, start_symbol=0, grant_bytes_budget=0)


def unique_occupancy(grants, prbs=24, symbols=14):
    occupied = np.zeros((symbols, prbs), dtype=bool)
    groups = set()
    for g in grants:
        key = ('mu', g.spatial_group_id) if g.spatial_group_id is not None else ('su', g.ue_id)
        if key in groups:
            continue
        groups.add(key)
        assert 0 <= g.start_prb < prbs and 0 <= g.start_symbol < symbols
        assert g.start_prb + g.allocated_prbs <= prbs
        assert g.start_symbol + g.allocated_symbols <= symbols
        rectangle = occupied[g.start_symbol:g.start_symbol + g.allocated_symbols,
                             g.start_prb:g.start_prb + g.allocated_prbs]
        assert not rectangle.any()
        rectangle[:] = True
    return occupied


def test_real_latency_bias_changes_final_contract_resources(monkeypatch):
    scheduler, state = setup(monkeypatch)
    ue0 = user('ue_0', queue_bytes=21000, latency_budget_ms=100., priority=1, sinr_db=10.)
    ue2 = user('ue_2', queue_bytes=500, latency_budget_ms=10., priority=3, sinr_db=10.)
    allocations = []
    for bias in [.5, 2.]:
        state.scheduler.latency_bias = bias
        state.scheduler.priority_boost_by_ue = {}
        grants = [grant(u.ue_id, scheduler._latency_score(state, u)) for u in [ue0, ue2]]
        scheduler._apply_spatial_multiplexing(state, grants, {u.ue_id: u for u in [ue0, ue2]})
        allocations.append([g.allocated_prbs for g in grants])
        assert sum(g.allocated_prbs for g in grants) == 24
        assert unique_occupancy(grants).all()
    assert allocations[0][0] > allocations[1][0]
    assert allocations[0][1] < allocations[1][1]


def test_target_ue_weight_changes_real_priority_and_resources_but_not_rr(monkeypatch):
    scheduler, state = setup(monkeypatch)
    target = user('ue_0', queue_bytes=1000, latency_budget_ms=100., priority=1, sinr_db=10.)
    other = user('ue_2', queue_bytes=1000, latency_budget_ms=100., priority=1, sinr_db=10.)
    state.scheduler.latency_bias = None
    state.scheduler.priority_boost_by_ue = {}
    baseline_other = scheduler._latency_score(state, other)
    allocations = []
    for weight in [.5, 2.]:
        target.scheduler_weight = weight
        assert scheduler._latency_score(state, other) == baseline_other
        assert scheduler._latency_score(state, target) == baseline_other * weight
        grants = [grant(u.ue_id, scheduler._latency_score(state, u)) for u in [target, other]]
        scheduler._apply_spatial_multiplexing(state, grants, {u.ue_id: u for u in [target, other]})
        allocations.append([g.allocated_prbs for g in grants])
        assert unique_occupancy(grants).all()
    assert allocations == [[8, 16], [16, 8]]
    state.scheduler.scheduler_mode = 'round_robin'
    for weight in [.5, 2.]:
        target.scheduler_weight = weight
        scores = [scheduler._score_for_mode(state, u, 'round_robin', round_robin_rank=i)
                  for i, u in enumerate([target, other])]
        assert scores == [9., 8.]
        grants = [grant(u.ue_id, score) for u, score in zip([target, other], scores)]
        scheduler._apply_spatial_multiplexing(state, grants, {u.ue_id: u for u in [target, other]})
        assert [g.allocated_prbs for g in grants] == [12, 12]


def test_round_robin_keeps_exact_historical_equal_greedy_shapes(monkeypatch):
    scheduler, state = setup(monkeypatch, mode='round_robin', prbs=25)
    expected = [(0, 8), (8, 8), (16, 9)]
    for scores in [[1., 1., 1.], [10000., 1., 0.], [1., 0., 10000.]]:
        grants = [grant(f'u{i}', score) for i, score in enumerate(scores)]
        scheduler._apply_spatial_multiplexing(state, grants, {g.ue_id: user(g.ue_id) for g in grants})
        assert [(g.start_prb, g.allocated_prbs) for g in grants] == expected
        assert all(g.start_symbol == 0 and g.allocated_symbols == 14 for g in grants)
        assert unique_occupancy(grants, prbs=25).all()


def test_minimum_room_prevents_high_weight_starving_other_fit_groups(monkeypatch):
    scheduler, state = setup(monkeypatch)
    grants = [grant('a', 1e9), grant('b', 1.), grant('c', 1.)]
    scheduler._apply_spatial_multiplexing(state, grants, {g.ue_id: user(g.ue_id) for g in grants})
    assert [g.allocated_prbs for g in grants] == [22, 1, 1]
    assert unique_occupancy(grants).all()


def test_mu_members_share_one_weighted_rectangle(monkeypatch):
    scheduler, state = setup(monkeypatch)
    grants = [grant('a', 3., group=7), grant('b', 6., group=7), grant('c', 1.)]
    scheduler._apply_spatial_multiplexing(state, grants, {g.ue_id: user(g.ue_id) for g in grants})
    a, b, c = grants
    assert (a.start_prb, a.allocated_prbs, a.start_symbol, a.allocated_symbols) == (b.start_prb, b.allocated_prbs, b.start_symbol, b.allocated_symbols)
    assert a.allocated_prbs > c.allocated_prbs
    assert unique_occupancy(grants).all()


def test_retx_shape_is_immutable_and_reserved_before_weighted_new_grants(monkeypatch):
    scheduler, state = setup(monkeypatch)
    contract = TbTxContract('retx', 'cell_0', 5, 'qam64', 1, 4, 3,
        'dmrs_type1_default', data_re(4, 3, 'dmrs_type1_default'), 240, 128, 12, (('lc_0', 12),))
    original = deepcopy(contract)
    users = {uid: user(uid) for uid in ['r', 'a', 'b']}
    users['r'].harq.processes = [NS(process_id=0, tx_contract=contract, retx_deadline_slot=9, last_tx_slot=2)]
    grants = [grant('a', 9.), grant('r', .1, retx=True), grant('b', 1.)]
    scheduler._apply_spatial_multiplexing(state, grants, users)
    retx = next(g for g in grants if g.is_retx)
    assert contract == original and retx.tx_contract is contract
    assert (retx.allocated_prbs, retx.allocated_symbols, retx.target_mcs, retx.mcs_table_id,
            retx.requested_bytes, retx.grant_bytes_budget) == (4, 3, 5, 'qam64', 12, 16)
    assert all(g.start_prb >= 4 for g in grants if not g.is_retx)
    assert next(g for g in grants if g.ue_id == 'a').allocated_prbs > next(g for g in grants if g.ue_id == 'b').allocated_prbs
    unique_occupancy(grants)


def test_power_holes_fragment_runs_without_overlap_or_spanning(monkeypatch):
    scheduler, state = setup(monkeypatch)
    state.cell.prb_power_w = [1.] * 24
    state.cell.prb_power_w[4] = state.cell.prb_power_w[11] = 0.
    grants = [grant('a', 9.), grant('b', 2.), grant('c', 1.)]
    scheduler._apply_spatial_multiplexing(state, grants, {g.ue_id: user(g.ue_id) for g in grants})
    used = unique_occupancy(grants)
    assert not used[:, 4].any() and not used[:, 11].any()
    assert used.sum() <= 22 * 14


def test_ue_caps_and_codec_shape_limits_are_still_enforced(monkeypatch):
    import phy.rt_sys.tx_contract as contracts
    scheduler, state = setup(monkeypatch)
    state.cell.ue_prb_limits = {'a': 3}
    monkeypatch.setattr(contracts, 'codec_format_supported', lambda table, mcs, count: count <= data_re(8, 14, 'dmrs_type1_default'))
    grants = [grant('a', 9.), grant('b', 1.), grant('c', 1.)]
    scheduler._apply_spatial_multiplexing(state, grants, {g.ue_id: user(g.ue_id) for g in grants})
    assert grants[0].allocated_prbs <= 3
    assert all(g.allocated_prbs <= 8 for g in grants)
    assert any('nr_codec_resource_shape_limit' in g.constraint_hits for g in grants)
    unique_occupancy(grants)


def test_time_first_weighting_stays_inside_symbol_budget(monkeypatch):
    scheduler, state = setup(monkeypatch, strategy='time_first')
    grants = [grant('a', 9.), grant('b', 1.)]
    scheduler._apply_spatial_multiplexing(state, grants, {g.ue_id: user(g.ue_id) for g in grants})
    assert grants[0].allocated_symbols > grants[1].allocated_symbols
    assert sum(g.allocated_symbols for g in grants) == 14
    assert all(g.allocated_prbs == 24 for g in grants)
    assert unique_occupancy(grants).all()
