"""Versioned QoS RF: real public outcomes, both URLLC UEs, strict checkpoints."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from test_latency_urllc_rf import CELLS, obs as legacy_obs
import ric.near_rt.pools.chongyu.latency_urllc_rf as legacy
from ric.near_rt.pools.chongyu.latency_urllc_rf import LatencyUrllcRF, URLLC
from ric.near_rt.pools.chongyu.latency_urllc_qos_rf import (
    LatencyUrllcQosRF, FEATURES, action_map, REFIT_ROWS)
from ric.near_rt.pools.chongyu.online_urllc_qos_rf import (
    OnlineUrllcQosRF, FIT_PARAMETERS, TARGETS)


def app(parameters=None):
    c = LatencyUrllcRF(CELLS, parameters=dict(policy_version='qos_v2', **(parameters or {})))
    c.select(CELLS, {cid: 1 for cid in CELLS})
    return c


def obs(slot=1, snapshot=None, owner=None, **kwargs):
    result = legacy_obs(slot, snapshot, owner, **kwargs)
    for u in result.ue_observations:
        u.hol_delay_ms = 2.
        u.retx_fraction = None
        u.kpi_metadata['hol_delay_ms'] = dict(produced_at_slot=slot, valid=True)
    return result


def start(c):
    c.run(obs(), NS())
    snapshot = deepcopy(c.pending['snapshot'])
    c.run(obs(10, snapshot, c.name), NS())
    c.run(obs(50, snapshot, c.name), NS())
    return snapshot


def rows(m, n=40):
    for i in range(n):
        arm = i % 4
        x = [float(i % 3)]*(len(FEATURES)-4) + [float(arm == j) for j in range(4)]
        m.add(x, [float(arm), 2000.+10*i, 200., 45.+i/100, float(arm)/3, 2*float(arm)/3])


def artifact(tmp_path, monkeypatch):
    pytest.importorskip('sklearn')
    monkeypatch.setattr(legacy, 'ROOT', tmp_path)
    m = OnlineUrllcQosRF(FEATURES)
    rows(m)
    assert m.fit()
    path = tmp_path/'outputs/qos.json'
    m.save(path)
    return path


def test_factory_legacy_default_and_opt_in_dual_ue_actions():
    assert type(LatencyUrllcRF(CELLS)) is LatencyUrllcRF
    c = app({'mode': 'fixed', 'fixed_action': 3})
    assert type(c) is LatencyUrllcQosRF
    result = c.run(obs(empty=True), NS())
    assert len(result.config_actions) == 3
    assert all(a.scheduler_mode == 'qos_deadline' for a in result.config_actions)
    weights = {uid: w for a in result.config_actions for uid,w in a.ue_scheduler_weights.items()}
    assert all(weights[uid] == 4. for uid in URLLC)
    assert all(w == 1. for uid,w in weights.items() if uid not in URLLC)
    assert c.last_decisions[0]['target_ue_ids'] == list(URLLC)
    assert c.summary()['policy_version'] == 'qos_v2'


def test_features_use_only_fresh_public_hol_normalized_demand_and_retx_missing_flag():
    c = app()
    report = obs()
    features = c._features(report.ue_observations, 2)
    assert len(features) == len(FEATURES)
    assert features[FEATURES.index('ue_6_hol_over_pdb')] == .2
    assert features[FEATURES.index('ue_6_queue_over_deadline_arrivals')] == .8
    assert features[FEATURES.index('ue_6_retx_report_available')] == 0.
    report.ue_observations[4].hol_delay_ms = None
    assert not c.run(report, NS()).config_actions
    assert 'HOL' in c.status_reason


def test_measured_labels_pool_both_ues_and_disclose_window_semantics():
    c = app()
    snapshot = start(c)
    delta = {
        'ue_2': dict(deadline_failed_packets_total=3, overflow_packets_total=1,
            completed_packets_total=10, completion_latency_ms_total=20.,
            on_time_completed_packets_total=7, late_completed_packets_total=3),
        'ue_6': dict(deadline_failed_packets_total=2, overflow_packets_total=1,
            completed_packets_total=20, completion_latency_ms_total=100.,
            on_time_completed_packets_total=18, late_completed_packets_total=2)}
    c.run(obs(250, snapshot, c.name, bits=200000, packet_delta=delta), NS())
    assert c.learner.sample_count == 1
    assert c.learner._y[0] == pytest.approx([70., 1200., 300., 6., 40., 30.])
    measured = next(e for e in c.events if e['kind'] == 'measured_sample')
    assert measured['target_order'] == TARGETS
    assert measured['label_scope'] == 'outcome_window_including_carryover_not_arrival_cohort_probability'
    assert c.pending['action'] == 1


def test_bad_execution_ownership_never_produces_training_label():
    c = app({'mode': 'fixed', 'fixed_action': 3})
    snapshot = start(c)
    report = obs(250, snapshot, c.name)
    report.control_ownership['cell_0']['ue_scheduler_weights'] = 'other'
    c.run(report, NS())
    assert c.invalidated == 1 and c.learner.sample_count == 0
    assert not any(e['kind'] == 'measured_sample' for e in c.events)


def test_selection_aggregate_failures_each_ue_guard_and_rr_ties(monkeypatch):
    c = app()
    rows(c.learner)
    c.mode = 'frozen'
    c.learner._forest = [{}]  # Ranking unit test, real trees tested separately.
    predictions = [[20., 200., 100., 50., 10., 10.],
        [10., 200., 100., 48., 5., 5.],
        [10., 100., 100., 47., 5., 5.],
        [0., 50., 100., 46., 0., 0.]]
    monkeypatch.setattr(c.learner, 'predict', lambda _: deepcopy(predictions))
    assert c._choose(obs().ue_observations)[0] == 3
    predictions[3][4] = 11.  # Aggregate gain must not hide UE2 degradation.
    assert c._choose(obs().ue_observations)[0] == 2
    for i in range(1,4): predictions[i] = predictions[0][:]
    assert c._choose(obs().ue_observations)[0] == 0
    assert c.last_prediction_audit['predicted_worst_ue_failure_events_per_s'] == [10.]*4


def test_true_fit_normalization_matches_sklearn_and_json_roundtrip(tmp_path):
    sklearn = pytest.importorskip('sklearn.ensemble')
    m = OnlineUrllcQosRF(FEATURES)
    rows(m)
    assert m.fit()
    normalized = [[v/s for v,s in zip(row,m.target_scales)] for row in m._y]
    reference = sklearn.RandomForestRegressor(random_state=179, **FIT_PARAMETERS).fit(m._x, normalized)
    predicted = m.predict(m._x)
    for actual, expected in zip(predicted, reference.predict(m._x).tolist()):
        assert actual == pytest.approx([v*s for v,s in zip(expected,m.target_scales)], abs=1e-10)
    path = tmp_path/'model.json'; m.save(path)
    loaded = OnlineUrllcQosRF.load(path, FEATURES)
    assert loaded.predict(m._x) == predicted
    assert loaded.trained_samples == 40 and len(loaded._forest) == 64
    assert loaded.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_online_fit_checkpoints_are_immutable_and_refit_every_20_rows(tmp_path, monkeypatch):
    pytest.importorskip('sklearn')
    monkeypatch.setattr(legacy, 'ROOT', tmp_path)
    path = tmp_path/'outputs/model.json'
    c = app({'model_path': str(path)})
    # Test mechanics with representative measured-row API; no physics claim.
    for i in range(60):
        arm = i % 4
        c._retain_sample({'features': [float(i%3)]*(len(FEATURES)-4)+[float(arm == j) for j in range(4)]},
            [float(arm), 100., 100., 50., float(arm)/2, float(arm)/2])
    assert c.learner.fit_count == 2 and REFIT_ROWS == 20
    checkpoints = sorted((path.parent/'model_checkpoints').glob('*.json'))
    assert len(checkpoints) == 2
    assert [json.loads(p.read_text())['trained_samples'] for p in checkpoints] == [40, 60]
    first = checkpoints[0].read_bytes()
    c._retain_sample({'features': c.learner._x[-1]}, c.learner._y[-1])
    assert checkpoints[0].read_bytes() == first and c.learner.trained_samples == 60


def test_frozen_executes_real_predictions_without_writing_or_training(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    original = path.read_bytes()
    c = app({'mode': 'frozen', 'model_path': str(path)})
    def forbidden(*args, **kwargs): raise AssertionError('Frozen model mutated')
    monkeypatch.setattr(c.learner, 'add', forbidden)
    monkeypatch.setattr(c.learner, 'fit', forbidden)
    monkeypatch.setattr(c.learner, 'save', forbidden)
    snapshot = start(c)
    c.run(obs(250, snapshot, c.name, bits=200000), NS())
    assert c.learner.prediction_batches == 2
    assert c.rf_selected_actions == 2 and c.learner.trained_samples == 40
    assert path.read_bytes() == original


def test_resumed_model_balances_new_run_rows_without_resetting_old_data(tmp_path, monkeypatch):
    path = artifact(tmp_path, monkeypatch)
    c = app({'model_path': str(path), 'balanced_start_windows': 8})
    original_x, original_y = deepcopy(c.learner._x), deepcopy(c.learner._y)
    previous_fit_count = c.learner.fit_count
    for i in range(8):
        arm, prediction, status, reason = c._choose(obs().ue_observations)
        assert arm == i % 4 and prediction is None and status == 'rules_only'
        assert reason == 'balanced_new_context_measured_qos_exploration'
        # The ordinary retain path only receives an executed, measured window.
        c._retain_sample({'features': c._features(obs().ue_observations, arm)},
            [float(arm), 100., 100., 40., float(arm)/2, float(arm)/2])
    assert c.learner._x[:40] == original_x and c.learner._y[:40] == original_y
    assert c.learner.fit_count == previous_fit_count and c.learner.trained_samples == 40
    assert c.summary()['run_start_samples'] == 40
    assert c.summary()['run_balanced_action_samples'] == {str(i): 2 for i in range(4)}
    assert c.summary()['run_balanced_start_complete']
    events = [e for e in c.events if e['kind'] == 'run_balanced_sample']
    assert len(events) == 8 and events[-1]['complete']
    assert c._choose(obs().ue_observations)[3] != 'balanced_new_context_measured_qos_exploration'


@pytest.mark.parametrize('value', [True, -4, 1, 81, 84, 4.0])
def test_balanced_start_rejects_invalid_window_counts(value):
    with pytest.raises(ValueError, match='balanced_start_windows'):
        app({'balanced_start_windows': value})


def test_balanced_start_requires_online_and_capacity(monkeypatch):
    with pytest.raises(ValueError, match='online'):
        app({'mode': 'fixed', 'balanced_start_windows': 4})
    monkeypatch.setattr('ric.near_rt.pools.chongyu.latency_urllc_qos_rf.MAX_ROWS', 39)
    with pytest.raises(ValueError, match='capacity'):
        app({'balanced_start_windows': 40})


@pytest.mark.parametrize('mutation', ['format', 'target_scales', 'tree', 'trees', 'features', 'params'])
def test_reject_legacy_or_tampered_artifacts(tmp_path, monkeypatch, mutation):
    path = artifact(tmp_path, monkeypatch)
    p = json.loads(path.read_text())
    if mutation == 'format': p['format'] = 'chongyu.online-packet-rf'
    elif mutation == 'target_scales': p['target_scales'][0] = 0
    elif mutation == 'tree': p['forest'][0]['children_left'][0] = 0
    elif mutation == 'trees': p['forest'].pop()
    elif mutation == 'features': p['feature_names'][0] = 'private_channel_truth'
    else: p['fit_parameters']['n_estimators'] = 32
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError): OnlineUrllcQosRF.load(path, FEATURES)
