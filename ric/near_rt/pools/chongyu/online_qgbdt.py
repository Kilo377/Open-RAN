"""Measured-data quantile boosting with safe standard-library JSON inference."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .online_rf import _integer, _names, _number, _row, _unique_object

FORMAT = 'chongyu.online-qgbdt'
VERSION = 1
MAX_ROWS = 24
LABEL = 'all_task_on_time_ratio_minus_0.25_deadline_penalized_mean_ratio'


def parameters(seed):
    return dict(loss='quantile', alpha=.2, n_estimators=32, learning_rate=.05,
                max_depth=2, min_samples_leaf=3, random_state=seed)


def validate_tree(tree, width):
    keys = {'children_left', 'children_right', 'feature', 'threshold', 'value'}
    if not isinstance(tree, dict) or set(tree) != keys or any(not isinstance(tree[k], list) for k in keys):
        raise ValueError('Invalid QGBDT tree schema')
    size = len(tree['feature'])
    if not 1 <= size <= 7 or any(len(tree[k]) != size for k in keys):
        raise ValueError('Invalid QGBDT tree lengths')
    parents = [0] * size
    for i in range(size):
        left, right = tree['children_left'][i], tree['children_right'][i]
        _integer(left, 'left child', -1, size-1); _integer(right, 'right child', -1, size-1)
        _number(tree['threshold'][i], 'threshold'); _number(tree['value'][i], 'leaf value')
        if left == right == -1:
            _integer(tree['feature'][i], 'leaf feature', -2, -2)
        else:
            _integer(tree['feature'][i], 'tree feature', 0, width-1)
            if left <= i or right <= i or left == right: raise ValueError('Invalid QGBDT tree children')
            parents[left] += 1; parents[right] += 1
    if parents[0] or any(n != 1 for n in parents[1:]): raise ValueError('Unreachable/shared QGBDT tree nodes')
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > 2: raise ValueError('QGBDT tree exceeds depth')
        if tree['children_left'][node] != -1:
            stack.extend([(tree['children_left'][node], depth+1), (tree['children_right'][node], depth+1)])
    return tree


class OnlineQGBDT:
    """Fixed small quantile model; samples and scalar labels are never invented."""
    def __init__(self, feature_names, seed=137):
        self.feature_names = _names(feature_names)
        self.seed = _integer(seed, 'seed', 0, 2**32-1)
        self._x, self._y, self._forest = [], [], []
        self.initial_prediction = self.fit_parameters = None
        self.trained_samples = self.fit_count = self.prediction_batches = 0
        self.fit_s = self.predict_s = 0.
        self.artifact_sha256 = None

    @property
    def sample_count(self): return len(self._x)

    @property
    def trained(self): return bool(self._forest)

    def add(self, features, utility):
        if self.sample_count >= MAX_ROWS: raise ValueError('QGBDT measured-row cap reached')
        x = _row(features, len(self.feature_names), 'features', float32=True)
        y = _number(utility, 'measured utility')
        self._x.append(x); self._y.append(y)

    def fit(self):
        if self.sample_count < 12: return False
        from sklearn.ensemble import GradientBoostingRegressor
        start = time.perf_counter()
        model = GradientBoostingRegressor(**parameters(self.seed)).fit(self._x, self._y)
        forest = []
        for stage in model.estimators_:
            tree = stage[0].tree_
            forest.append(validate_tree(dict(children_left=tree.children_left.tolist(),
                children_right=tree.children_right.tolist(), feature=tree.feature.tolist(),
                threshold=tree.threshold.tolist(), value=[float(v[0][0]) for v in tree.value.tolist()]), len(self.feature_names)))
        self.initial_prediction = float(model.init_.constant_[0][0])
        self._forest = forest
        self.fit_parameters = parameters(self.seed)
        self.trained_samples = self.sample_count
        self.fit_count += 1
        self.fit_s += time.perf_counter() - start
        return True

    def predict(self, candidates):
        if not isinstance(candidates, (list, tuple)): raise ValueError('Invalid QGBDT candidates')
        rows = [_row(r, len(self.feature_names), 'candidate', float32=True) for r in candidates]
        if not self.trained: return None
        start = time.perf_counter(); result = []
        for row in rows:
            value = self.initial_prediction
            for tree in self._forest:
                node = 0
                while tree['children_left'][node] != -1:
                    node = tree['children_left'][node] if row[tree['feature'][node]] <= tree['threshold'][node] else tree['children_right'][node]
                value += self.fit_parameters['learning_rate'] * tree['value'][node]
            result.append(value)
        self.prediction_batches += 1
        self.predict_s += time.perf_counter()-start
        return result

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format=FORMAT, version=VERSION, feature_names=self.feature_names,
            label=LABEL, seed=self.seed, x=self._x, y=self._y, forest=self._forest,
            initial_prediction=self.initial_prediction, fit_parameters=self.fit_parameters,
            trained_samples=self.trained_samples, fit_count=self.fit_count,
            prediction_batches=self.prediction_batches, fit_s=self.fit_s, predict_s=self.predict_s)
        raw = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
        temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            temporary.write_bytes(raw); os.replace(temporary, path)
        finally: temporary.unlink(missing_ok=True)
        self.artifact_sha256 = hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls, path, expected_features):
        path = Path(path)
        if not path.is_file() or path.stat().st_size > 4*1024*1024: raise ValueError('Invalid QGBDT artifact size')
        raw = path.read_bytes(); p = json.loads(raw, object_pairs_hook=_unique_object)
        keys = {'format','version','feature_names','label','seed','x','y','forest',
                'initial_prediction','fit_parameters','trained_samples','fit_count',
                'prediction_batches','fit_s','predict_s'}
        if not isinstance(p, dict) or set(p) != keys: raise ValueError('Invalid QGBDT artifact schema')
        if p['format'] != FORMAT or type(p['version']) is not int or p['version'] != VERSION: raise ValueError('Invalid QGBDT artifact version')
        if p['feature_names'] != _names(expected_features) or p['label'] != LABEL: raise ValueError('QGBDT feature/label schema mismatch')
        result = cls(p['feature_names'], p['seed'])
        if not isinstance(p['x'], list) or not isinstance(p['y'], list) or len(p['x']) != len(p['y']) or len(p['x']) > MAX_ROWS: raise ValueError('Invalid QGBDT measured rows')
        for x, y in zip(p['x'], p['y']): result.add(x, y)
        if not isinstance(p['forest'], list) or len(p['forest']) not in (0,32): raise ValueError('Invalid QGBDT forest size')
        result._forest = [validate_tree(t,len(result.feature_names)) for t in p['forest']]
        result.trained_samples = _integer(p['trained_samples'],'trained samples',0,result.sample_count)
        result.fit_count = _integer(p['fit_count'],'fit count',0,MAX_ROWS)
        if result.trained:
            if result.trained_samples < 12 or not result.fit_count: raise ValueError('Inconsistent QGBDT fitted counts')
            fit = p['fit_parameters']; expected = parameters(result.seed)
            if not isinstance(fit,dict) or set(fit) != set(expected): raise ValueError('Invalid QGBDT fit schema')
            for key, value in expected.items():
                if type(fit[key]) is not type(value) or fit[key] != value: raise ValueError('Invalid QGBDT fitted parameter '+key)
            result.fit_parameters = dict(fit)
            result.initial_prediction = _number(p['initial_prediction'],'initial prediction')
        elif result.trained_samples or result.fit_count or p['fit_parameters'] is not None or p['initial_prediction'] is not None:
            raise ValueError('Untrained QGBDT cannot claim fitted metadata')
        result.prediction_batches = _integer(p['prediction_batches'],'prediction batches',0,2**63-1)
        for key in ('fit_s','predict_s'):
            value = _number(p[key],key)
            if value < 0: raise ValueError('Negative QGBDT timing')
            setattr(result,key,value)
        result.artifact_sha256 = hashlib.sha256(raw).hexdigest()
        return result
