from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.rt_link_statistics import LinkStatistics, LATENCY_SCOPE
import test_rt_sys as fixtures
from test_rt_sys import channel, phy, ue


def test_pooled_nacks_cannot_hide_a_broken_first_transmission_link():
    stats = LinkStatistics()
    rows = [dict(ue_id='bad', actual_decode=True, ack=False, is_retx=False, resource_mode='mu')]
    original = deepcopy(rows)
    stats.observe(1, rows)
    assert rows == original
    stats.observe(2, [dict(ue_id='bad', actual_decode=True, ack=True, is_retx=True, resource_mode='su')])
    for slot in range(3, 22):
        stats.observe(slot, [dict(ue_id='good', actual_decode=True, ack=True, resource_mode='su')])
    result = stats.snapshot()
    assert result['network']['actual_decode_attempts'] == 21
    assert result['network']['first_tx_bler'] == pytest.approx(1 / 20)
    assert result['per_ue']['bad']['first_tx_bler'] == 1
    assert result['per_ue']['bad']['retx_error_rate'] == 0
    assert result['per_ue']['bad']['by_resource_mode']['mu']['first_tx_attempts'] == 1
    assert result['per_ue']['bad']['by_resource_mode']['su']['retx_attempts'] == 1


def test_non_transmissions_duplicates_missing_modes_and_empty_denominators():
    stats = LinkStatistics()
    assert stats.user('absent')['first_tx_bler'] is None
    assert stats.user('absent')['retx_error_rate'] is None
    rows = [dict(ue_id='u', actual_decode=False, ack=False, scheduled=True),
            dict(ue_id='u', actual_decode=True, ack=False)]
    stats.observe(3, rows)
    stats.observe(3, rows)
    assert stats.snapshot()['network']['actual_decode_attempts'] == 1
    assert set(stats.user('u')['by_resource_mode']) == {'unknown'}
    with pytest.raises(ValueError, match='increasing'):
        stats.observe(2, rows)
    assert 'same-slot completion is zero' in LATENCY_SCOPE


def test_mode_uses_executed_resource_overlap_not_scheduled_user_count():
    adapter = fixtures.PhysicsTests().adapter()
    h = np.array([[[[1e-5, 0]]], [[[.3e-5, 1e-5]]]])
    snap = channel(h, ('u0', 'u1'), ('c0',))
    # Two UEs in the same cell, on orthogonal PRBs, are both SU.
    adapter.run_cluster_slot([phy('c0', [ue('u0'), ue('u1', start=1)])], snap)
    assert all(r['resource_mode'] == 'su' and r['mu_data_re'] == 0 for r in adapter.trace['ues'])
    # On overlapping PRBs they are MU, even without a scheduler group label.
    adapter.run_cluster_slot([phy('c0', [ue('u0'), ue('u1')])], snap)
    for row in adapter.trace['ues']:
        assert row['resource_mode'] == 'mu'
        assert row['mu_data_re'] == row['data_re'] > 0
        assert row['su_data_re'] == 0
        assert row['intra_cell_interference_dbm'] > -270


def test_partial_overlap_and_idle_are_not_mislabeled():
    adapter = fixtures.PhysicsTests().adapter()
    snap = channel(np.ones((2, 1, 1, 2)) * 1e-5, ('u0', 'u1'), ('c0',))
    wide = ue('u0')
    wide.allocated_prbs = 2
    wide.allocated_subcarrier_indices = list(range(24))
    adapter.run_cluster_slot([phy('c0', [wide, ue('u1')])], snap)
    row = adapter.trace['ues'][0]
    assert row['resource_mode'] == 'mixed'
    assert row['mu_data_re'] > 0 and row['su_data_re'] > 0
    assert row['mu_data_re'] + row['su_data_re'] == row['data_re']
    adapter.run_cluster_slot([phy('c0', [ue('u0', False), ue('u1')])], snap)
    assert adapter.trace['ues'][0]['resource_mode'] == 'idle'
    assert adapter.trace['ues'][0]['intra_cell_interference_dbm'] is None


def test_mu_switch_does_not_change_ue_rank_or_radio_initial_conditions(tmp_path):
    from phy.rt_sys.scenario import build_case
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / 'configs/rt_sys/default.json').read_text())
    variants = []
    for enabled in (True, False):
        base['simulation']['scheduler_enable_spatial_multiplexing'] = enabled
        path = tmp_path / f'{enabled}.json'
        path.write_text(json.dumps(base))
        variants.append(build_case('munich', 3, 12, total_slots=400, config_path=path))
    assert variants[0][0] == variants[1][0]
    assert variants[0][1]['ue'] == variants[1][1]['ue']
    assert all(u['rank'] == u['ri'] == 1 for u in variants[1][1]['ue']['ues'])
    base['ue_initial_rank'] = 3
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match='antenna limits'):
        build_case(config_path=path)
