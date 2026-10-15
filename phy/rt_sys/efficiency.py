"""Exact resource grouping and non-synchronizing inclusive wall-clock timing."""
from collections import defaultdict
from functools import wraps
from time import perf_counter
import numpy as np


def resource_patterns(entries, ns, nf):
    """Preserve row-major coordinates and first-occurrence pattern order exactly."""
    if not entries:
        return {}
    masks = np.stack([e['mask'] & (np.asarray(e['power']) > 0) for e in entries], axis=-1)
    unique, first, inverse = np.unique(masks.reshape(-1, len(entries)), axis=0,
                                      return_index=True, return_inverse=True)
    result = {}
    for index in np.argsort(first):
        owners = tuple(np.flatnonzero(unique[index]).tolist())
        if owners:
            flat = np.flatnonzero(inverse == index)
            result[owners] = list(zip((flat // nf).tolist(), (flat % nf).tolist()))
    return result


class StageTimings:
    """Inclusive host wall times; nested stages must never be summed as exclusive."""
    def __init__(self):
        self.stages = defaultdict(lambda: dict(calls=0, wall_s=0.))

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        @wraps(original)
        def measured(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                item = self.stages[label]
                item['calls'] += 1
                item['wall_s'] += perf_counter() - start
        setattr(obj, name, measured)

    def report(self):
        return dict(stages=dict(self.stages), clock='inclusive_host_wall_no_cuda_fences',
                    interpretation='Nested and asynchronous stages are not additive GPU execution times.')
