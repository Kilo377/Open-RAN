"""True AP counter labels, consumer-ACK attribution and standalone RF schema."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from ric.near_rt.pools import build_pool_xapp, pool_catalog
import ric.near_rt.pools.chongyu.ap_energy_rf as module
from ric.near_rt.pools.chongyu.ap_energy_rf import (
    ApEnergyRF, USERS, URLLC, FEATURES, TARGETS, PACKET_COUNTERS, AP_COUNTERS, AP_PROOF, action_map)
from ric.near_rt.pools.chongyu.online_ap_rf import OnlineApRF, FIT_PARAMETERS

CELLS = ['cell_0', 'cell_1', 'cell_2']


def app(parameters=None):
    c = ApEnergyRF(CELLS, parameters=parameters)
    c.select(CELLS, {cid: 1 for cid in CELLS})
    return c


def obs(slot=1, snapshot=None, owner=None, bits=100000, packet_delta=None,
        energy_delta=0., generation=1, applied=10, empty=True, availability=None):
    users = []
    for i, uid in enumerate(USERS):
        cid = ('cell_0' if uid in ('ue_0', 'ue_6', 'ue_8') else 'cell_2') if empty else CELLS[i//2]
        counters = dict(zip(PACKET_COUNTERS, [0, 0, 100, 90]))
        for key, value in (packet_delta or {}).get(uid, {}).items(): counters[key] += value
        metadata = {key: dict(produced_at_slot=slot, valid=True) for key in
                    ('delivered_bits_total', 'offered_bits_total')+PACKET_COUNTERS}
        users.append(NS(ue_id=uid, serving_cell_id=cid, queue_bytes=1000,
            throughput_mbps=1., offered_rate_bps=1e6, delivered_bits_total=bits, offered_bits_total=bits,
            slice_id='urllc' if uid in URLLC else 'embb', latency_budget_ms=10. if uid in URLLC else 100.,
            kpi_metadata=metadata, **counters))
    snapshot = snapshot or action_map(users, CELLS, 0)
    cells = []
    for cid in CELLS:
        metadata = {key: dict(produced_at_slot=slot, valid=True) for key in ('executed_config',)+AP_COUNTERS+AP_PROOF}
        if owner is None:
            metadata['ap_policy_owner']['valid'] = metadata['ap_policy_applied_slot']['valid'] = False
        slots = (availability or {}).get(cid, (slot, 0, 0))
        cells.append(NS(cell_id=cid, executed_config=dict(cell_sleep_mode=snapshot[cid], scheduler_mode='round_robin'),
            ap_policy=snapshot[cid], ap_policy_owner=owner, ap_policy_applied_slot=applied if owner else None,
            ap_policy_generation=generation if owner else 0, ap_cycle_period_slots=200,
            ap_cycle_phase_slot=max(0, slot-1)%200, ap_energy_total_j=1.+energy_delta,
            ap_available_slots_total=slots[0], ap_sleep_slots_total=slots[1], ap_waking_slots_total=slots[2],
            kpi_metadata=metadata))
    return NS(slot_id=slot, ue_observations=users, cell_observations=cells)


def start(c):
    decision = c.run(obs(), NS())
    assert c.pending is not None # Native owner=None cannot block bootstrap.
    snapshot = deepcopy(c.pending['snapshot'])
    c.run(obs(10, snapshot, c.name), NS())
    assert c.pending['confirmed_slot'] == 10
    c.run(obs(209, snapshot, c.name), NS())
    assert 'start_slot' not in c.pending
    c.run(obs(210, snapshot, c.name), NS())
    assert c.pending['start_slot'] == 210
    return snapshot, decision


def rows(model, n=24):
    for i in range(n):
        x = [float(i%3)]*8+[float(j == i%4) for j in range(4)]
        model.add(x, [200.-20*(i%4), 50.+i/100, float(i%4), float(i%2)])


def artifact(tmp_path, monkeypatch):
    pytest.importorskip('sklearn')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    model = OnlineApRF(FEATURES)
    rows(model)
    assert model.fit()
    path = tmp_path/'outputs/ap.json'
    model.save(path)
    return path


def test_catalog_writable_scope_and_four_network_policies():
    assert pool_catalog('chongyu')['ap_energy_rf']['category_id'] == 'ap_energy_saving'
    c = build_pool_xapp(dict(implementation_id='ap_energy_rf', cell_ids=CELLS,
        parameters={'mode': 'fixed', 'fixed_action': 3}), 'chongyu')
    c.select(CELLS, {cid: 1 for cid in CELLS})
    decision = c.run(obs(), NS())
    assert isinstance(c, ApEnergyRF) and len(decision.config_actions) == 3
    assert c.pending['snapshot'] == {'cell_0': 'light_sleep', 'cell_1': 'deep_sleep', 'cell_2': 'light_sleep'}
    for action in decision.config_actions:
        assert action._library_provenance == c.identity(action.target_cell_id)
        assert action.scheduler_mode is action.ue_scheduler_weights is action.ue_offload_fraction is None
    assert set(c.controlled_fields) == {'cell_sleep_mode'}
    users = obs().ue_observations
    assert action_map(users, CELLS, 0) == {cid: 'active' for cid in CELLS}
    assert action_map(users, CELLS, 1) == {'cell_0': 'active', 'cell_1': 'deep_sleep', 'cell_2': 'active'}
    assert action_map(users, CELLS, 2) == {'cell_0': 'micro_sleep', 'cell_1': 'deep_sleep', 'cell_2': 'micro_sleep'}


def test_no_selected_owner_or_pre_ack_report_confirms_execution():
    c = app()
    c.run(obs(), NS())
    snapshot = deepcopy(c.pending['snapshot'])
    c.run(obs(10, snapshot, owner='selected_other'), NS())
    assert not c.pending['confirmed']
    c.run(obs(20, snapshot, owner=c.name, applied=21), NS())
    assert not c.pending['confirmed']
    c.run(obs(30, snapshot, owner=c.name, applied=21), NS())
    assert c.pending['confirmed_slot'] == 30


def test_true_whole_ap_joules_and_all_ue_bits_one_global_row():
    c = app()
    snapshot, _ = start(c)
    c.run(obs(409, snapshot, c.name, bits=200000, energy_delta=2.), NS())
    assert c.learner.sample_count == 0
    packet = {'ue_2': {'deadline_failed_packets_total': 1, 'overflow_packets_total': 2},
              'ue_6': {'deadline_failed_packets_total': 4}}
    c.run(obs(410, snapshot, c.name, bits=200000, energy_delta=2., packet_delta=packet), NS())
    assert c.learner.sample_count == 1
    sample = next(e for e in c.events if e['kind'] == 'measured_sample')
    assert sample['targets'] == pytest.approx([60., 6., 30., 40.])
    assert sample['ap_energy_j_by_cell'] == {cid: 2. for cid in CELLS}
    assert sample['produced_start_slot'] == 210 and sample['produced_end_slot'] == 410
    assert sample['duration_s'] == .1
    assert sample['start_cycle_phase_by_cell'] == sample['end_cycle_phase_by_cell']
    assert set(sample['delivered_bits_by_ue']) == set(USERS)
    assert c.pending['action'] == 1 # Bad outcomes never remove rows/starve exploration.


def test_unchanged_executed_owner_generation_reuses_original_ack():
    c = app({'mode': 'fixed', 'fixed_action': 0})
    snapshot, _ = start(c)
    c.run(obs(410, snapshot, c.name), NS())
    assert c.pending['requested_slot'] == 410
    c.run(obs(420, snapshot, c.name, applied=10), NS())
    assert c.pending['confirmed'] and c.pending['start_slot'] == 420


@pytest.mark.parametrize('change', ['owner', 'mode', 'generation', 'other_scheduler', 'membership', 'scope',
    'reset', 'spliced_ap', 'spliced_packet', 'profile', 'availability'])
def test_unexecuted_confounded_or_invalid_ap_windows_do_not_train(change):
    c = app()
    snapshot, _ = start(c)
    report = obs(410, snapshot, c.name, bits=200000, energy_delta=2.)
    if change == 'owner': report.cell_observations[0].ap_policy_owner = 'other'
    elif change == 'mode': report.cell_observations[0].ap_policy = 'light_sleep'
    elif change == 'generation': report.cell_observations[0].ap_policy_generation += 1
    elif change == 'other_scheduler': report.cell_observations[0].executed_config['scheduler_mode'] = 'latency_aware'
    elif change == 'membership': report.ue_observations[0].serving_cell_id = 'cell_2'
    elif change == 'scope': c.select(['cell_0'], {'cell_0': 1})
    elif change == 'reset': report.cell_observations[0].ap_energy_total_j = 0.
    elif change == 'spliced_ap': report.cell_observations[0].kpi_metadata['ap_energy_total_j']['produced_at_slot'] = 409
    elif change == 'spliced_packet': report.ue_observations[4].kpi_metadata['deadline_failed_packets_total']['produced_at_slot'] = 409
    elif change == 'profile': report.ue_observations[4].latency_budget_ms = 20.
    else: report.cell_observations[0].ap_available_slots_total -= 1
    c.run(report, NS())
    assert c.learner.sample_count == 0 and not any(e['kind'] == 'measured_sample' for e in c.events)


def test_partial_window_phase_mismatch_waits_without_fabricating_cycle():
    c = app()
    snapshot, _ = start(c)
    report = obs(410, snapshot, c.name)
    report.cell_observations[0].ap_cycle_phase_slot = 10
    c.run(report, NS())
    assert c.learner.sample_count == 0 and c.pending['start_slot'] == 210


def test_untrained_rows_resume_real_coverage_without_rewriting_model(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path/'outputs/partial.json'
    c = app({'model_path': str(path)})
    snapshot, _ = start(c)
    c.run(obs(410, snapshot, c.name), NS())
    original = path.read_bytes()
    resumed = app({'model_path': str(path)})
    resumed.run(obs(), NS())
    assert resumed.action_samples() == {'0': 1, '1': 0, '2': 0, '3': 0}
    assert resumed.pending['action'] == 1 and not resumed.learner.trained
    assert path.read_bytes() == original


def test_bounded_rf_actual_fit_matches_sklearn_and_safe_json(tmp_path):
    sklearn = pytest.importorskip('sklearn.ensemble')
    model = OnlineApRF(FEATURES)
    rows(model)
    expected = sklearn.RandomForestRegressor(random_state=181, **FIT_PARAMETERS).fit(model._x, model._y)
    assert model.fit() and model.trained_samples == 24 and model.fit_count == 1
    candidates = model._x + [[.3]*8+[1., 0., 0., 0.]]
    actual = model.predict(candidates)
    for a,b in zip(actual, expected.predict(candidates).tolist()): assert a == pytest.approx(b, abs=1e-12)
    path = tmp_path/'model.json'
    model.save(path)
    restored = OnlineApRF.load(path, FEATURES)
    assert restored.predict(candidates) == actual and restored._x == model._x
    assert restored.fit_parameters == FIT_PARAMETERS
    assert restored.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_actual_inference_never_changes_rows_fits_or_bytes(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    original = path.read_bytes()
    c = app({'mode': 'frozen', 'model_path': str(path)})
    def forbidden(*args, **kwargs): raise AssertionError('Frozen AP model changed')
    for method in ('add', 'fit', 'save'): monkeypatch.setattr(c.learner, method, forbidden)
    snapshot, _ = start(c)
    c.run(obs(410, snapshot, c.name), NS())
    assert c.learner.sample_count == c.learner.trained_samples == 24
    assert c.learner.fit_count == 1 and c.learner.prediction_batches == 2
    assert c.summary()['rf_selected_actions'] == 2 and path.read_bytes() == original
    assert c.last_decisions[0]['features'] and c.last_decisions[0]['prediction_audit']['raw_predictions']


def test_energy_first_does_not_hide_qos_cost_or_force_sleep(monkeypatch):
    c = app()
    rows(c.learner)
    c.learner._forest = [{}] # Unit-only stub; strict real fit tested separately.
    predictions = [[200., 50., 0., 0.], [160., 50., 0., 0.], [140., 49., 1., 2.], [120., 45., 10., 20.]]
    monkeypatch.setattr(c.learner, 'predict', lambda x: deepcopy(predictions))
    c.run(obs(), NS())
    assert c.pending['action'] == 3
    assert c.last_decisions[0]['constraints_active'] is False
    assert c.last_decisions[0]['prediction'][2:] == [10., 20.]
    predictions[:] = [[200., 50., 0., 0.]]*4
    c.pending = None
    c.run(obs(), NS())
    assert c.pending['action'] == 0 and c.status == 'rf'


@pytest.mark.parametrize('mutation', ['format', 'features', 'targets', 'tree', 'params', 'rows', 'nan'])
def test_safe_ap_schema_rejects_tampered_artifacts(tmp_path, monkeypatch, mutation):
    path = artifact(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    if mutation == 'format': payload['format'] = 'chongyu.online-packet-rf'
    elif mutation == 'features': payload['feature_names'][0] = 'future_energy'
    elif mutation == 'targets': payload['target_names'][0] = 'rf_energy_w'
    elif mutation == 'tree': payload['forest'][0]['children_left'][0] = 0
    elif mutation == 'params': payload['fit_parameters']['max_depth'] = True
    elif mutation == 'rows': payload['x'].append(payload['x'][0])
    else: payload['y'][0][0] = float('nan')
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): OnlineApRF.load(path, FEATURES)


@pytest.mark.parametrize('parameters', [{'mode': 'bad'}, {'window_ms': 10}, {'seed': True}, {'fixed_action': 4}, {'target_cell_id': 'cell_0'}])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError): app(parameters)
