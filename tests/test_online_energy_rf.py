import builtins
import json
import math

import pytest

from ric.near_rt.pools.chongyu.online_rf import OnlineEnergyRF


def data(model, n=24):
    for i in range(n):
        model.add([i / n, float(i % 3)], [1 - i / n, i / n, .1 * (i % 3), 10 + i])
    return model


def trained(max_features=1.0):
    pytest.importorskip('sklearn')
    model = data(OnlineEnergyRF(['load', 'action'], trees=8))
    assert (model.fit() if max_features == 1.0 else model.fit(max_features=max_features))
    return model


def test_collects_measured_rows_without_implicit_fit_or_dependency(tmp_path, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith('sklearn'):
            raise AssertionError('Premature sklearn import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    model = data(OnlineEnergyRF(['load', 'action']), 4)
    assert model.sample_count == 4 and not model.trained
    assert model.predict([[.2, 1.]]) is None
    assert model.fit() is False
    path = tmp_path / 'untrained.json'
    model.save(path)
    loaded = OnlineEnergyRF.load(path, ['load', 'action'])
    assert loaded.sample_count == 4 and loaded.predict([[.2, 1.]]) is None


@pytest.mark.parametrize('features,targets', [([0.], [1, 2, 3, 4]),
    ([0., math.nan], [1, 2, 3, 4]), ([0., 1.], [1, 2, math.inf, 4]),
    ([0., 1.], [1, 2, 3]), ([True, 1.], [1, 2, 3, 4]),
    ([1e300, 1.], [1, 2, 3, 4])])
def test_invalid_sample_is_atomic(features, targets):
    model = OnlineEnergyRF(['a', 'b'])
    with pytest.raises(ValueError):
        model.add(features, targets)
    assert model.sample_count == 0


def test_safe_tree_inference_observes_sklearn_float32_threshold_semantics(tmp_path):
    model = OnlineEnergyRF(['load'], min_samples=2, trees=1)
    model.add([0.], [0., 0., 0., 0.])
    model.add([1.], [1., 1., 1., 1.])
    path = tmp_path / 'fixture.json'
    model.save(path)
    payload = json.loads(path.read_text())
    # Explicit hand-built inference fixture; this is not a trained artifact.
    payload.update(trained_samples=2, fit_count=1, fit_parameters={'max_features':1.0}, forest=[dict(
        children_left=[1, -1, -1], children_right=[2, -1, -1],
        feature=[0, -2, -2], threshold=[.5, -2., -2.],
        value=[[.5] * 4, [0.] * 4, [1.] * 4])])
    path.write_text(json.dumps(payload))
    loaded = OnlineEnergyRF.load(path, ['load'])
    assert loaded.predict([[.50000001], [.500001]]) == [[0.] * 4, [1.] * 4]


@pytest.mark.parametrize('max_features', [1.0, .5])
def test_real_rf_predictions_match_sklearn_and_are_reproducible(max_features):
    pytest.importorskip('sklearn')
    from sklearn.ensemble import RandomForestRegressor
    model = trained(max_features)
    reference = RandomForestRegressor(n_estimators=8, max_depth=6,
        min_samples_leaf=2, n_jobs=1, random_state=1, max_features=max_features).fit(model._x, model._y)
    candidates = [[.123456789, 0.], [.7, 1.], [.999999999, 2.]]
    actual = model.predict(candidates)
    expected = reference.predict(candidates).tolist()
    for a, b in zip(actual, expected):
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12)
    other = data(OnlineEnergyRF(['load', 'action'], trees=8))
    other.fit(max_features=max_features)
    assert model.fit_parameters == {'max_features': max_features}
    assert other.predict(candidates) == actual
    assert model.trained_samples == 24 and model.fit_count == 1


def test_json_round_trip_frozen_inference_and_resumable_training(tmp_path, monkeypatch):
    model = trained()
    candidates = [[.2, 0.], [.8, 2.]]
    expected = model.predict(candidates)
    path = tmp_path / 'model.json'
    model.save(path)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith('sklearn'):
            raise AssertionError('Frozen inference imported sklearn')
        return original(name, *args, **kwargs)
    with monkeypatch.context() as scope:
        scope.setattr(builtins, '__import__', guarded)
        loaded = OnlineEnergyRF.load(path, ['load', 'action'])
        assert loaded.predict(candidates) == expected
    assert loaded.fit_count == 1 and loaded.sample_count == 24
    loaded.add([.5, 1.], [.2, .8, 0., 8.])
    assert loaded.trained_samples == 24
    loaded.fit()
    assert loaded.trained_samples == 25 and loaded.fit_count == 2


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(version=True),
    lambda p: p.update(feature_names=['wrong', 'names']),
    lambda p: p.update(trained_samples=1000),
    lambda p: p['forest'][0]['children_left'].__setitem__(0, 0),
    lambda p: p['forest'][0]['children_right'].__setitem__(0, -1),
    lambda p: p['forest'][0]['feature'].__setitem__(0, 99),
    lambda p: p['forest'][0]['threshold'].__setitem__(0, math.nan),
    lambda p: p['forest'][0]['value'].__setitem__(0, [1, 2, 3]),
    lambda p: p['forest'][0]['children_left'].append(-1),
    lambda p: p['x'].append([1., 2.]),
    lambda p: p.update(fit_s=-1),
])
def test_tampered_artifacts_are_rejected(tmp_path, mutation):
    model = trained()
    path = tmp_path / 'model.json'
    model.save(path)
    payload = json.loads(path.read_text())
    mutation(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        OnlineEnergyRF.load(path, ['load', 'action'])


def test_duplicate_keys_and_wrong_expected_feature_order_rejected(tmp_path):
    model = OnlineEnergyRF(['load', 'action'])
    path = tmp_path / 'model.json'
    model.save(path)
    with pytest.raises(ValueError):
        OnlineEnergyRF.load(path, ['action', 'load'])
    path.write_text('{"format":"a","format":"b"}')
    with pytest.raises(ValueError):
        OnlineEnergyRF.load(path, ['load', 'action'])


def inference_fixture(tmp_path, max_features=1.):
    """Explicit JSON-only inference fixture; never used as a deployment model."""
    model = OnlineEnergyRF(['load'], min_samples=2, trees=1)
    model.add([0.], [0., 0., 0., 10.]); model.add([1.], [1., 1., 0., 20.])
    path = tmp_path / 'fixture.json'
    model.save(path)
    payload = json.loads(path.read_text())
    payload.update(trained_samples=2, fit_count=1,
        fit_parameters={'max_features': max_features}, forest=[dict(
        children_left=[-1], children_right=[-1], feature=[-2], threshold=[-2.],
        value=[[.5, .5, 0., 15.]])])
    path.write_text(json.dumps(payload))
    return path, payload


@pytest.mark.parametrize('value', [True, False, 0., -.1, 1.1, math.nan, math.inf, '0.5', None])
def test_fit_rejects_invalid_fraction_before_import_or_minimum_sample_check(value):
    model = OnlineEnergyRF(['load'])
    with pytest.raises(ValueError, match='max_features'):
        model.fit(max_features=value)
    assert model.fit_parameters is None and model.fit_count == 0


def test_valid_unfitted_parameter_does_not_claim_a_fit(tmp_path):
    model = OnlineEnergyRF(['load'])
    assert model.fit(max_features=.5) is False
    assert model.fit_parameters is None
    path = tmp_path / 'untrained.json'; model.save(path)
    payload = json.loads(path.read_text())
    assert payload['version'] == 2 and payload['fit_parameters'] is None
    assert OnlineEnergyRF.load(path, ['load']).fit_parameters is None
    payload['fit_parameters'] = {'max_features': .5}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Untrained'):
        OnlineEnergyRF.load(path, ['load'])


def test_v1_loading_keeps_inference_default_fit_metadata_and_original_bytes(tmp_path):
    path, payload = inference_fixture(tmp_path)
    v2 = OnlineEnergyRF.load(path, ['load'])
    payload['version'] = 1; payload.pop('fit_parameters')
    legacy = tmp_path / 'legacy.json'; legacy.write_text(json.dumps(payload))
    original = legacy.read_bytes()
    loaded = OnlineEnergyRF.load(legacy, ['load'])
    assert loaded.predict([[.2], [.9]]) == v2.predict([[.2], [.9]])
    assert loaded.fit_parameters == {'max_features': 1.}
    assert legacy.read_bytes() == original
    migrated = tmp_path / 'migrated.json'; loaded.save(migrated)
    assert json.loads(migrated.read_text())['fit_parameters'] == {'max_features': 1.}


def test_v2_round_trip_retains_actual_fit_parameter_without_inference_dependency(tmp_path, monkeypatch):
    path, _ = inference_fixture(tmp_path, .5)
    model = OnlineEnergyRF.load(path, ['load'])
    model.add([.3], [.1, .9, 0., 10.])
    assert model.fit_parameters == {'max_features': .5} and model.trained_samples == 2
    saved = tmp_path / 'saved.json'; model.save(saved)
    actual = OnlineEnergyRF.load(saved, ['load'])
    assert actual.fit_parameters == {'max_features': .5}
    assert actual.predict([[.4]]) == [[.5, .5, 0., 15.]]


@pytest.mark.parametrize('parameters', [None, {}, {'max_features': .5, 'unknown': 1},
    {'max_features': True}, {'max_features': 0}, {'max_features': 2},
    {'max_features': math.nan}, {'max_features': math.inf}, {'max_features': '0.5'}])
def test_v2_fit_parameter_tampering_is_rejected(tmp_path, parameters):
    path, payload = inference_fixture(tmp_path)
    payload['fit_parameters'] = parameters
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        OnlineEnergyRF.load(path, ['load'])


def test_v1_rejects_new_fit_parameter_field_in_legacy_schema(tmp_path):
    path, payload = inference_fixture(tmp_path)
    payload['version'] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='schema'):
        OnlineEnergyRF.load(path, ['load'])
