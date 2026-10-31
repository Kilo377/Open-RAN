"""Exact resource grouping and non-synchronizing inclusive wall-clock timing."""
from collections import defaultdict
from functools import wraps
from time import perf_counter
import numpy as np


def configure_torch_threads(config):
    """Scope CPU parallelism to this process, independently of Dr.Jit's pool."""
    intra=config.get('torch_num_threads')
    inter=config.get('torch_num_interop_threads')
    for name,value in [('torch_num_threads',intra),('torch_num_interop_threads',inter)]:
        if value is not None and (type(value) is not int or value<1):
            raise ValueError(f'{name} must be a positive integer')
    if intra is None and inter is None:return None
    import torch
    if intra is not None:torch.set_num_threads(intra)
    if inter is not None and torch.get_num_interop_threads()!=inter:
        torch.set_num_interop_threads(inter)
    return dict(intra_op=torch.get_num_threads(),inter_op=torch.get_num_interop_threads())


def diagnostic_power_sums(algebra, inter_parts, intra_parts, length):
    """One device transfer, retaining the scalar path's float64 addition order."""
    inter=np.zeros(length);intra=np.zeros(length)
    parts=inter_parts+intra_parts
    if parts:
        host=algebra.numpy(algebra.cat([p.reshape(1,-1) for p in parts],axis=0))
        for row in host[:len(inter_parts)]:inter+=row
        for row in host[len(inter_parts):]:intra+=row
    return inter,intra


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
