"""Measured whole-AP-energy RF with plain JSON, dependency-free inference.

This schema is deliberately separate from the MEC, UE-energy and packet RF artifacts. Input
rows follow sklearn's float32 tree convention; outcomes retain double precision.
Only fit() imports sklearn. No pickle or executable artifact content is accepted.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import uuid

from .online_rf import _integer, _names, _number, _row, _unique_object

FORMAT = 'chongyu.online-ap-rf'
VERSION = 1
MIN_ROWS = 24
MAX_ROWS = 512
TARGETS = ['network_ap_power_w', 'network_goodput_mbps',
           'ue_2_unique_failure_events_per_s', 'ue_6_unique_failure_events_per_s']
FIT_PARAMETERS = dict(n_estimators=32, max_depth=6, min_samples_leaf=2,
                      max_features=1.0, n_jobs=1, bootstrap=True)


def _tree(tree, width):
    keys = {'children_left', 'children_right', 'feature', 'threshold', 'value'}
    if not isinstance(tree, dict) or set(tree) != keys:
        raise ValueError('Invalid AP RF tree schema')
    if any(not isinstance(tree[k], list) for k in keys):
        raise ValueError('Invalid AP RF tree arrays')
    size = len(tree['feature'])
    if not 1 <= size <= 127 or any(len(tree[k]) != size for k in keys):
        raise ValueError('Invalid AP RF tree lengths')
    parents = [0] * size
    depths = [0] * size
    for i in range(size):
        left, right = tree['children_left'][i], tree['children_right'][i]
        _integer(left, 'left child', -1, size - 1)
        _integer(right, 'right child', -1, size - 1)
        _number(tree['threshold'][i], 'threshold')
        values = _row(tree['value'][i], len(TARGETS), 'tree value')
        if any(v < 0 for v in values):
            raise ValueError('Negative AP RF outcome')
        if left == right == -1:
            if type(tree['feature'][i]) is not int or tree['feature'][i] != -2:
                raise ValueError('Invalid AP RF leaf')
        else:
            _integer(tree['feature'][i], 'feature', 0, width - 1)
            if left <= i or right <= i or left == right:
                raise ValueError('Invalid AP RF children')
            for child in (left, right):
                parents[child] += 1
                depths[child] = depths[i] + 1
                if depths[child] > 6:
                    raise ValueError('AP RF tree exceeds depth')
    if parents[0] or any(n != 1 for n in parents[1:]):
        raise ValueError('Unreachable or shared AP RF node')
    return tree


class OnlineApRF:
    """Fixed small RF fitted exclusively on explicitly added measured rows."""
    def __init__(self, feature_names, seed=181):
        self.feature_names = _names(feature_names)
        self.seed = _integer(seed, 'seed', 0, 2**32 - 1)
        self._x, self._y, self._forest = [], [], []
        self.trained_samples = self.fit_count = self.prediction_batches = 0
        self.fit_s = self.predict_s = 0.
        self.fit_parameters = None
        self.artifact_sha256 = None

    @property
    def sample_count(self):
        return len(self._x)

    @property
    def trained(self):
        return bool(self._forest)

    def add(self, features, targets):
        if self.sample_count >= MAX_ROWS:
            raise ValueError('AP RF measured row cap reached')
        x = _row(features, len(self.feature_names), 'features', float32=True)
        y = _row(targets, len(TARGETS), 'targets')
        if any(v < 0 for v in y):
            raise ValueError('Negative AP outcome')
        self._x.append(x)
        self._y.append(y)

    def fit(self):
        if self.sample_count < MIN_ROWS:
            return False
        from sklearn.ensemble import RandomForestRegressor
        started = time.perf_counter()
        model = RandomForestRegressor(random_state=self.seed, **FIT_PARAMETERS)
        model.fit(self._x, self._y)
        forest = []
        for estimator in model.estimators_:
            t = estimator.tree_
            forest.append(_tree(dict(children_left=t.children_left.tolist(),
                children_right=t.children_right.tolist(), feature=t.feature.tolist(),
                threshold=t.threshold.tolist(),
                value=[[float(v[0]) for v in node] for node in t.value.tolist()]), len(self.feature_names)))
        self._forest = forest
        self.fit_parameters = dict(FIT_PARAMETERS)
        self.trained_samples = self.sample_count
        self.fit_count += 1
        self.fit_s += time.perf_counter() - started
        return True

    def predict(self, candidates):
        if not isinstance(candidates, (list, tuple)):
            raise ValueError('AP RF candidates must be a sequence')
        rows = [_row(r, len(self.feature_names), 'candidate', float32=True) for r in candidates]
        if not self.trained:
            return None
        started = time.perf_counter()
        result = []
        for row in rows:
            values = [0.] * len(TARGETS)
            for tree in self._forest:
                node = 0
                while tree['children_left'][node] != -1:
                    node = (tree['children_left'][node] if row[tree['feature'][node]] <= tree['threshold'][node]
                            else tree['children_right'][node])
                values = [a+b for a, b in zip(values, tree['value'][node])]
            result.append([v / len(self._forest) for v in values])
        self.predict_s += time.perf_counter() - started
        self.prediction_batches += 1
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format=FORMAT, version=VERSION, feature_names=self.feature_names,
            target_names=TARGETS, seed=self.seed, x=self._x, y=self._y, forest=self._forest,
            trained_samples=self.trained_samples, fit_count=self.fit_count,
            prediction_batches=self.prediction_batches, fit_parameters=self.fit_parameters,
            fit_s=self.fit_s, predict_s=self.predict_s)
        raw = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
        temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            temporary.write_bytes(raw)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.artifact_sha256 = hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls, path, expected_features):
        path = Path(path)
        if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Invalid AP RF artifact size')
        raw = path.read_bytes()
        p = json.loads(raw, object_pairs_hook=_unique_object)
        keys = {'format', 'version', 'feature_names', 'target_names', 'seed', 'x', 'y', 'forest',
                'trained_samples', 'fit_count', 'prediction_batches', 'fit_parameters', 'fit_s', 'predict_s'}
        if not isinstance(p, dict) or set(p) != keys or p['format'] != FORMAT or type(p['version']) is not int or p['version'] != VERSION:
            raise ValueError('Invalid AP RF artifact schema')
        if p['feature_names'] != _names(expected_features) or p['target_names'] != TARGETS:
            raise ValueError('AP RF feature or outcome schema mismatch')
        model = cls(expected_features, p['seed'])
        if not isinstance(p['x'], list) or not isinstance(p['y'], list) or len(p['x']) != len(p['y']) or len(p['x']) > MAX_ROWS:
            raise ValueError('Invalid AP RF measured rows')
        for x, y in zip(p['x'], p['y']):
            model.add(x, y)
        model.trained_samples = _integer(p['trained_samples'], 'trained_samples', 0, model.sample_count)
        model.fit_count = _integer(p['fit_count'], 'fit_count', 0, MAX_ROWS)
        model.prediction_batches = _integer(p['prediction_batches'], 'prediction_batches', 0, 10**9)
        model.fit_s, model.predict_s = (_number(p[k], k) for k in ('fit_s', 'predict_s'))
        if min(model.fit_s, model.predict_s) < 0:
            raise ValueError('Negative AP RF timing')
        if not isinstance(p['forest'], list):
            raise ValueError('Invalid AP RF forest')
        if p['forest']:
            params = p['fit_parameters']
            if not isinstance(params, dict) or set(params) != set(FIT_PARAMETERS) or any(type(params[k]) is not type(v) or params[k] != v for k,v in FIT_PARAMETERS.items()):
                raise ValueError('Unknown AP RF fit parameters')
            if len(p['forest']) != 32 or model.trained_samples < MIN_ROWS or not model.fit_count:
                raise ValueError('Invalid fitted AP RF metadata')
            model._forest = [_tree(t, len(model.feature_names)) for t in p['forest']]
            model.fit_parameters = dict(params)
        elif model.trained_samples or model.fit_count or p['fit_parameters'] is not None or model.prediction_batches:
            raise ValueError('Untrained AP RF claims fitted state')
        model.artifact_sha256 = hashlib.sha256(raw).hexdigest()
        return model
