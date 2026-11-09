"""Small batched online RF surrogate with safe, dependency-free inference.

Samples must be measured outcomes supplied by the controller. ``add`` never
creates labels or fits implicitly. Scikit-learn is imported only by ``fit``;
deployed inference and JSON loading require only Python's standard library.
"""
from pathlib import Path
import json
import math
import os
import struct
import time
import uuid


TARGET_NAMES = ('energy_w', 'normalized_goodput', 'task_late_rate',
                'task_mean_latency_ms')
FORMAT = 'chongyu.online-energy-rf'
VERSION = 2
MAX_TREES = 256
MAX_FEATURES = 512
MAX_ROWS = 100000
MAX_FILE_BYTES = 64 * 1024 * 1024


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Invalid ' + name)
    return value


def _number(value, name):
    if type(value) not in (int, float):
        raise ValueError('Invalid ' + name)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('Invalid ' + name) from error
    if not math.isfinite(result):
        raise ValueError('Invalid ' + name)
    return result


def _max_features(value):
    value = _number(value, 'max_features')
    if not 0 < value <= 1:
        raise ValueError('Invalid max_features')
    return value


def _names(names):
    if not isinstance(names, (list, tuple)) or not 1 <= len(names) <= MAX_FEATURES:
        raise ValueError('Invalid feature names')
    if any(not isinstance(x, str) or not x.strip() for x in names) or len(set(names)) != len(names):
        raise ValueError('Feature names must be unique nonempty strings')
    return list(names)


def _row(row, width, name, *, float32=False):
    if not isinstance(row, (list, tuple)) or len(row) != width:
        raise ValueError('Invalid ' + name + ' dimensions')
    result = [_number(value, name) for value in row]
    if float32:
        # sklearn's tree predictor converts its inputs to float32 even though
        # thresholds and leaf values are stored as double precision.
        try:
            result = [struct.unpack('f', struct.pack('f', v))[0] for v in result]
        except (OverflowError, struct.error) as error:
            raise ValueError('Feature exceeds float32 range') from error
        if not all(math.isfinite(v) for v in result):
            raise ValueError('Feature exceeds float32 range')
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate artifact key: ' + key)
        result[key] = value
    return result


def _validate_tree(tree, feature_count):
    keys = {'children_left', 'children_right', 'feature', 'threshold', 'value'}
    if not isinstance(tree, dict) or set(tree) != keys:
        raise ValueError('Invalid tree schema')
    if any(not isinstance(tree[k], list) for k in keys):
        raise ValueError('Invalid tree arrays')
    size = len(tree['feature'])
    if not 1 <= size <= 127 or any(len(tree[k]) != size for k in keys):
        raise ValueError('Invalid tree array lengths')
    parents = [0] * size
    for i in range(size):
        left, right = tree['children_left'][i], tree['children_right'][i]
        feature = tree['feature'][i]
        _integer(left, 'left child', -1, size - 1)
        _integer(right, 'right child', -1, size - 1)
        _number(tree['threshold'][i], 'threshold')
        _row(tree['value'][i], len(TARGET_NAMES), 'leaf value')
        if left == right == -1:
            if feature != -2:
                raise ValueError('Invalid leaf feature')
        else:
            _integer(feature, 'tree feature', 0, feature_count - 1)
            if left <= i or right <= i or left == right:
                raise ValueError('Tree contains invalid children or a cycle')
            parents[left] += 1
            parents[right] += 1
    if parents[0] != 0 or any(p != 1 for p in parents[1:]):
        raise ValueError('Tree contains unreachable or shared nodes')
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > 6:
            raise ValueError('Tree exceeds configured depth')
        if tree['children_left'][node] != -1:
            stack.extend([(tree['children_left'][node], depth + 1),
                          (tree['children_right'][node], depth + 1)])
    return tree


class OnlineEnergyRF:
    """Measured-data RF, refitted in batches rather than after every sample.

    The four outputs are mean UE power, goodput relative to offered traffic,
    task late/miss rate, and completed-task mean latency. The caller determines
    the observation window and whether an outcome is eligible for training.
    """
    def __init__(self, feature_names, min_samples=12, trees=32, seed=1):
        self.feature_names = _names(feature_names)
        self.min_samples = _integer(min_samples, 'min_samples', 2, MAX_ROWS)
        self.trees = _integer(trees, 'tree count', 1, MAX_TREES)
        self.seed = _integer(seed, 'seed', 0, 2**32 - 1)
        self._x, self._y, self._forest = [], [], []
        self.trained_samples = 0
        self.fit_s = self.predict_s = 0.
        self.fit_count = self.prediction_batches = 0
        self.fit_parameters = None

    @property
    def sample_count(self):
        return len(self._x)

    @property
    def trained(self):
        return bool(self._forest)

    def add(self, features, targets):
        if self.sample_count >= MAX_ROWS:
            raise ValueError('Online sample limit reached')
        x = _row(features, len(self.feature_names), 'features', float32=True)
        y = _row(targets, len(TARGET_NAMES), 'targets')
        # Validate both rows before mutating the training dataset.
        self._x.append(x)
        self._y.append(y)

    def fit(self, *, max_features=1.0):
        max_features = _max_features(max_features)
        if self.sample_count < self.min_samples:
            return False
        from sklearn.ensemble import RandomForestRegressor
        started = time.perf_counter()
        model = RandomForestRegressor(n_estimators=self.trees, max_depth=6,
                                      min_samples_leaf=2, n_jobs=1,
                                      random_state=self.seed, max_features=max_features)
        model.fit(self._x, self._y)
        forest = []
        for estimator in model.estimators_:
            t = estimator.tree_
            tree = dict(children_left=t.children_left.tolist(),
                        children_right=t.children_right.tolist(),
                        feature=t.feature.tolist(), threshold=t.threshold.tolist(),
                        value=[[float(output[0]) for output in node]
                               for node in t.value.tolist()])
            forest.append(_validate_tree(tree, len(self.feature_names)))
        self._forest = forest
        self.fit_parameters = {'max_features': max_features}
        self.trained_samples = self.sample_count
        self.fit_count += 1
        self.fit_s += time.perf_counter() - started
        return True

    def predict(self, candidates):
        if not isinstance(candidates, (list, tuple)):
            raise ValueError('Candidates must be a sequence')
        rows = [_row(row, len(self.feature_names), 'candidate', float32=True)
                for row in candidates]
        if not self.trained:
            return None
        started = time.perf_counter()
        results = []
        for row in rows:
            prediction = [0.] * len(TARGET_NAMES)
            for tree in self._forest:
                node = 0
                while tree['children_left'][node] != -1:
                    node = (tree['children_left'][node] if
                            row[tree['feature'][node]] <= tree['threshold'][node]
                            else tree['children_right'][node])
                for i, value in enumerate(tree['value'][node]):
                    prediction[i] += value
            results.append([v / len(self._forest) for v in prediction])
        self.predict_s += time.perf_counter() - started
        self.prediction_batches += 1
        return results

    def save(self, path):
        """Write atomic plain JSON; no Python object deserialization is used."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format=FORMAT, version=VERSION,
                       feature_names=self.feature_names, target_names=list(TARGET_NAMES),
                       min_samples=self.min_samples, trees=self.trees, seed=self.seed,
                       x=self._x, y=self._y, forest=self._forest,
                       trained_samples=self.trained_samples,
                       fit_s=self.fit_s, predict_s=self.predict_s,
                       fit_count=self.fit_count, prediction_batches=self.prediction_batches,
                       fit_parameters=self.fit_parameters)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, expected_features):
        path = Path(path)
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('RF artifact exceeds size limit')
        payload = json.loads(path.read_text(), object_pairs_hook=_unique_object)
        required = {'format', 'version', 'feature_names', 'target_names',
                    'min_samples', 'trees', 'seed', 'x', 'y', 'forest',
                    'trained_samples', 'fit_s', 'predict_s', 'fit_count',
                    'prediction_batches'}
        if not isinstance(payload, dict):
            raise ValueError('Invalid RF artifact schema')
        version = payload.get('version')
        if type(version) is not int or version not in (1, VERSION):
            raise ValueError('Unsupported RF artifact format/version')
        if version == 2:
            required.add('fit_parameters')
        if set(payload) != required:
            raise ValueError('Invalid RF artifact schema')
        if payload['format'] != FORMAT:
            raise ValueError('Unsupported RF artifact format/version')
        if payload['feature_names'] != _names(expected_features) or payload['target_names'] != list(TARGET_NAMES):
            raise ValueError('RF artifact feature/target schema mismatch')
        result = cls(payload['feature_names'], payload['min_samples'],
                     payload['trees'], payload['seed'])
        if not isinstance(payload['x'], list) or not isinstance(payload['y'], list) or len(payload['x']) != len(payload['y']) or len(payload['x']) > MAX_ROWS:
            raise ValueError('Invalid persisted training rows')
        for x, y in zip(payload['x'], payload['y']):
            result.add(x, y)
        forest = payload['forest']
        if not isinstance(forest, list) or len(forest) not in (0, result.trees):
            raise ValueError('Invalid forest size')
        result._forest = [_validate_tree(t, len(result.feature_names)) for t in forest]
        result.trained_samples = _integer(payload['trained_samples'], 'trained sample count', 0, result.sample_count)
        if (result.trained and result.trained_samples < result.min_samples) or (not result.trained and result.trained_samples != 0):
            raise ValueError('Inconsistent trained sample count')
        result.fit_count = _integer(payload['fit_count'], 'fit count', 0, MAX_ROWS)
        result.prediction_batches = _integer(payload['prediction_batches'], 'prediction batch count', 0, 2**63 - 1)
        if bool(result.fit_count) != result.trained:
            raise ValueError('Inconsistent fit count')
        if version == 1:
            # V1 used sklearn's max_features=1.0 default for every actual fit.
            result.fit_parameters = {'max_features': 1.0} if result.trained else None
        else:
            parameters = payload['fit_parameters']
            if result.trained:
                if not isinstance(parameters, dict) or set(parameters) != {'max_features'}:
                    raise ValueError('Invalid RF fit parameter schema')
                result.fit_parameters = {'max_features': _max_features(parameters['max_features'])}
            elif parameters is not None:
                raise ValueError('Untrained forest cannot have fitted parameters')
        for key in ('fit_s', 'predict_s'):
            value = _number(payload[key], key)
            if value < 0:
                raise ValueError('Negative timing counter')
            setattr(result, key, value)
        return result
