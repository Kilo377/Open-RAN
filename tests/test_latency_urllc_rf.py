"""Independent packet RF schema, measured windows, executed ownership and arms."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from ric.near_rt.pools import build_pool_xapp, pool_catalog
import ric.near_rt.pools.chongyu.latency_urllc_rf as module
from ric.near_rt.pools.chongyu.latency_urllc_rf import (
    LatencyUrllcRF, USERS, URLLC, FEATURES, TARGETS, PACKET_COUNTERS, action_map)
from ric.near_rt.pools.chongyu.online_packet_rf import OnlinePacketRF, FIT_PARAMETERS

CELLS = ['cell_0', 'cell_1', 'cell_2']


def app(parameters=None):
    c = LatencyUrllcRF(CELLS, parameters=parameters)
    c.select(CELLS, {cid: 1 for cid in CELLS})
    return c


def obs(slot=1, snapshot=None, owner=None, bits=100000, packet_delta=None, empty=False):
    users = []
    for i, uid in enumerate(USERS):
        cid = CELLS[i//2]
        if empty:
            cid = 'cell_0' if uid in ('ue_0', 'ue_6', 'ue_8') else 'cell_2'
        counters = dict(zip(PACKET_COUNTERS, [0, 0, 90, 180., 100, 90, 0]))
        for key, value in (packet_delta or {}).get(uid, {}).items():
            counters[key] += value
        metadata = {key: dict(produced_at_slot=slot, valid=True) for key in
                    ('delivered_bits_total', 'offered_bits_total')+PACKET_COUNTERS}
        users.append(NS(ue_id=uid, serving_cell_id=cid, queue_bytes=1000,
            reference_sinr_db=10.+i, throughput_mbps=1., offered_rate_bps=1e6,
            delivered_bits_total=bits, offered_bits_total=bits,
            slice_id='urllc' if uid in URLLC else 'embb', latency_budget_ms=10. if uid in URLLC else 100.,
            kpi_metadata=metadata, **counters))
    snapshot = deepcopy(snapshot) if snapshot is not None else action_map(users, 0)
    for cid in CELLS:
        snapshot.setdefault(cid, dict(scheduler_mode='round_robin', ue_scheduler_weights={}))
    cells = [NS(cell_id=cid, executed_config={**snapshot[cid], 'latency_bias': None},
        kpi_metadata={'executed_config': dict(produced_at_slot=slot, valid=True)}) for cid in CELLS]
    owners = {cid: dict(scheduler_mode=owner, ue_scheduler_weights=owner) for cid in CELLS}
    if empty:
        owners['cell_1'].pop('ue_scheduler_weights')
    return NS(slot_id=slot, ue_observations=users, cell_observations=cells, control_ownership=owners)


def start(c, empty=False):
    decision = c.run(obs(empty=empty), NS())
    snapshot = deepcopy(c.pending['snapshot'])
    c.run(obs(10, snapshot, c.name, empty=empty), NS())
    assert c.pending['confirmed_slot'] == 10
    c.run(obs(50, snapshot, c.name, empty=empty), NS())
    assert c.pending['start_slot'] == 50
    return snapshot, decision


def sample_rows(m, n=24):
    for i in range(n):
        x = [float(i%3)]*(len(FEATURES)-4)+[float(j == i%4) for j in range(4)]
        m.add(x, [float(i%4), 20.+i, 10., 6.+i/100, 0.])


def artifact(tmp_path, monkeypatch):
    pytest.importorskip('sklearn')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    model = OnlinePacketRF(FEATURES)
    sample_rows(model)
    assert model.fit()
    path = tmp_path/'outputs/urllc.json'
    model.save(path)
    return path


def test_catalog_separate_mec_variant_and_complete_global_actions():
    catalog = pool_catalog('chongyu')
    assert 'latency_scheduler_rf' in catalog and 'latency_urllc_rf' in catalog
    c = build_pool_xapp(dict(implementation_id='latency_urllc_rf', cell_ids=CELLS,
        parameters={'mode': 'fixed', 'fixed_action': 3}), 'chongyu')
    assert isinstance(c, LatencyUrllcRF) and c.category_id == 'latency_scheduler'
    c.select(CELLS, {cid: 1 for cid in CELLS})
    decision = c.run(obs(empty=True), NS())
    assert len(decision.config_actions) == 3
    maps = {a.target_cell_id: a.ue_scheduler_weights for a in decision.config_actions}
    assert maps['cell_1'] == {}
    merged = {uid: w for m in maps.values() for uid, w in m.items()}
    assert set(merged) == set(USERS) and merged['ue_6'] == 2.
    assert all(w == 1. for uid,w in merged.items() if uid != 'ue_6')
    for action in decision.config_actions:
        assert action.scheduler_mode == 'latency_aware'
        assert action._library_provenance == c.identity(action.target_cell_id)
        assert action.ue_offload_fraction is action.ue_drx_configs is None


def test_one_true_global_packet_sample_after_common_ack_washout():
    c = app()
    snapshot, _ = start(c)
    c.run(obs(249, snapshot, c.name, bits=200000), NS())
    assert c.learner.sample_count == 0
    packet = {'ue_6': dict(deadline_failed_packets_total=2, overflow_packets_total=1,
        completed_packets_total=10, completion_latency_ms_total=35.,
        on_time_completed_packets_total=8, late_completed_packets_total=2, admitted_packets_total=9),
        'ue_2': dict(deadline_failed_packets_total=1)}
    c.run(obs(250, snapshot, c.name, bits=200000, packet_delta=packet), NS())
    assert c.learner.sample_count == 1
    event = next(e for e in c.events if e['kind'] == 'measured_sample')
    assert event['targets'] == pytest.approx([30., 350., 100., 6., 10.])
    assert event['completion_mean_ms_by_ue']['ue_6'] == 3.5
    assert event['completion_mean_ms_by_ue']['ue_2'] is None
    assert event['duration_s'] == .1 and event['produced_start_slot'] == 50
    assert event['produced_end_slot'] == 250 and event['target_order'] == TARGETS
    assert set(event['delivered_bits_by_ue']) == set(USERS)
    assert event['original_latency_budgets_ms'] == {'ue_2': 10., 'ue_6': 10.}
    assert c.pending['action'] == 1


def test_no_completion_bad_window_retained_without_invented_mean():
    c = app()
    snapshot, _ = start(c)
    packet = {'ue_6': dict(deadline_failed_packets_total=5, overflow_packets_total=2)}
    c.run(obs(250, snapshot, c.name, packet_delta=packet), NS())
    assert c.learner.sample_count == 1 and c.learner._y[0] == [70., 0., 0., 0., 0.]
    event = next(e for e in c.events if e['kind'] == 'measured_sample')
    assert event['completion_mean_ms_by_ue']['ue_6'] is None
    assert c.pending['action'] == 1 # No guard-induced starvation after a bad RR window.


def test_empty_cell_requires_mode_only_occupied_cells_require_weights():
    c = app()
    snapshot, _ = start(c, empty=True)
    c.run(obs(250, snapshot, c.name, bits=200000, empty=True), NS())
    assert c.learner.sample_count == 1
    c = app()
    snapshot, _ = start(c, empty=True)
    report = obs(250, snapshot, c.name, empty=True)
    report.control_ownership['cell_0'].pop('ue_scheduler_weights')
    c.run(report, NS())
    assert c.learner.sample_count == 0 and c.invalidated == 1


@pytest.mark.parametrize('change', ['owner', 'mode', 'weight', 'scope', 'membership', 'spliced', 'counter_reset', 'profile', 'missing_ue', 'latency_bias'])
def test_unexecuted_confounded_or_invalid_windows_do_not_train(change):
    c = app({'mode': 'fixed', 'fixed_action': 3})
    snapshot, _ = start(c)
    report = obs(250, snapshot, c.name, bits=200000)
    if change == 'owner': report.control_ownership['cell_0']['scheduler_mode'] = 'x08'
    elif change == 'mode': report.cell_observations[0].executed_config['scheduler_mode'] = 'round_robin'
    elif change == 'weight': report.cell_observations[2].executed_config['ue_scheduler_weights']['ue_6'] = 1.
    elif change == 'scope': c.select(['cell_0'], {'cell_0': 1})
    elif change == 'membership': report.ue_observations[0].serving_cell_id = 'cell_2'
    elif change == 'spliced': report.ue_observations[4].kpi_metadata['completion_latency_ms_total']['produced_at_slot'] = 249
    elif change == 'counter_reset': report.ue_observations[4].completion_latency_ms_total = 0.
    elif change == 'profile': report.ue_observations[4].latency_budget_ms = 11.
    elif change == 'latency_bias': report.cell_observations[0].executed_config['latency_bias'] = 2.
    else: report.ue_observations.pop()
    c.run(report, NS())
    assert not any(e['kind'] == 'measured_sample' for e in c.events)
    assert c.learner.sample_count == 0


def test_unconfirmed_requests_timeout_and_retry_no_false_sample():
    c = app()
    c.run(obs(), NS())
    c.run(obs(402), NS())
    assert c.invalidated == 1 and c.pending['requested_slot'] == 402
    assert c.pending['action'] == 0 and c.learner.sample_count == 0


def test_untrained_rows_resume_balanced_real_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path/'outputs/partial.json'
    c = app({'model_path': str(path)})
    snapshot, _ = start(c)
    c.run(obs(250, snapshot, c.name), NS())
    original = path.read_bytes()
    resumed = app({'model_path': str(path)})
    assert resumed.action_samples() == {'0': 1, '1': 0, '2': 0, '3': 0}
    resumed.run(obs(), NS())
    assert resumed.pending['action'] == 1 and resumed.learner.fit_count == 0
    assert path.read_bytes() == original


def test_rf_real_fit_matches_sklearn_and_strict_json_roundtrip(tmp_path):
    sklearn = pytest.importorskip('sklearn.ensemble')
    model = OnlinePacketRF(FEATURES)
    sample_rows(model)
    expected = sklearn.RandomForestRegressor(random_state=179, **FIT_PARAMETERS).fit(model._x, model._y)
    assert model.fit() and model.trained_samples == 24 and model.fit_count == 1
    rows = model._x + [[.2]*8+[0., 1., 0., 0.]]
    actual = model.predict(rows)
    for a, b in zip(actual, expected.predict(rows).tolist()):
        assert a == pytest.approx(b, abs=1e-12)
    path = tmp_path/'model.json'
    model.save(path)
    reloaded = OnlinePacketRF.load(path, FEATURES)
    assert reloaded.predict(rows) == actual and reloaded._x == model._x
    assert reloaded.fit_parameters == FIT_PARAMETERS
    assert reloaded.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_true_inference_rr_choice_no_add_fit_save(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    original = path.read_bytes()
    c = app({'mode': 'frozen', 'model_path': str(path)})
    def forbidden(*args, **kwargs): raise AssertionError('Frozen mode changed a model')
    monkeypatch.setattr(c.learner, 'add', forbidden)
    monkeypatch.setattr(c.learner, 'fit', forbidden)
    monkeypatch.setattr(c.learner, 'save', forbidden)
    snapshot, _ = start(c)
    c.run(obs(250, snapshot, c.name, bits=200000), NS())
    assert c.learner.sample_count == c.learner.trained_samples == 24
    assert c.learner.fit_count == 1 and c.learner.prediction_batches == 2
    assert c.summary()['rf_selected_actions'] == 2 and path.read_bytes() == original
    assert c.status == 'rf' and c.last_decisions[0]['prediction_audit']['raw_predictions']


def test_selection_failures_then_actual_completion_mean_then_network(monkeypatch):
    c = app()
    sample_rows(c.learner)
    c.learner._forest = [{}] # Unit-only stub; real tree inference tested separately.
    predictions = [[0., 200., 100., 10., 0.], [0., 190., 100., 11., 0.],
                   [50., 100., 100., 15., 0.], [0., 150., 100., 12., 0.]]
    monkeypatch.setattr(c.learner, 'predict', lambda x: deepcopy(predictions))
    c.run(obs(), NS())
    assert c.pending['action'] == 3
    predictions[3][4] = 1. # Peer URLLC gets worse than the same-context RR.
    c.pending = None
    c.run(obs(), NS())
    assert c.pending['action'] == 1
    predictions[1] = predictions[0][:]
    c.pending = None
    c.run(obs(), NS())
    assert c.pending['action'] == 0 and c.status == 'rf'


@pytest.mark.parametrize('mutation', ['format', 'features', 'targets', 'tree', 'params', 'rows', 'nonfinite'])
def test_safe_artifact_rejects_schema_tree_and_parameter_tampering(tmp_path, monkeypatch, mutation):
    path = artifact(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    if mutation == 'format': payload['format'] = 'chongyu.online-energy-rf'
    elif mutation == 'features': payload['feature_names'][0] = 'future_outcome'
    elif mutation == 'targets': payload['target_names'][0] = 'task_late_rate'
    elif mutation == 'tree': payload['forest'][0]['children_left'][0] = 0
    elif mutation == 'params': payload['fit_parameters']['max_depth'] = True
    elif mutation == 'rows': payload['x'].append(payload['x'][0])
    else: payload['y'][0][0] = float('nan')
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): OnlinePacketRF.load(path, FEATURES)


def test_frozen_requires_coverage_and_new_packet_schema(tmp_path, monkeypatch):
    with pytest.raises(ValueError): app({'mode': 'frozen'})
    path = artifact(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    payload['x'][0][-4:] = [0., 1., 0., 0.]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='six per arm'): app({'mode': 'frozen', 'model_path': str(path)})


@pytest.mark.parametrize('parameters', [{'mode': 'bad'}, {'window_ms': 10}, {'seed': True}, {'fixed_action': 4}, {'unknown': 1}])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError): app(parameters)
