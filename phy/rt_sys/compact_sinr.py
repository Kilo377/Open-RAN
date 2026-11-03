"""Lossless multiplicities for the system model's repeated RE SINRs.

The inverse map retains data-RE/layer row-major order for actual NR fallback.
This changes storage/reduction only, not channel sampling or resource budgets.
"""
import numpy as np


class CompactSinr:
    def __init__(self, values, weights, inverse):
        self.values = np.asarray(values, dtype=float)
        self.weights = weights
        self.inverse = inverse
        self.count = len(inverse)
        self.no_path = not np.any(self.values[self.weights > 0] > 0)
        if not self.no_path:
            self.values = np.maximum(self.values, 1e-30)

    def __len__(self):
        return self.count

    def mean(self):
        return float(np.dot(self.values, self.weights) / max(1, self.count))

    def expand(self):
        return self.values[self.inverse]

    def validate(self):
        used = self.values[self.weights > 0]
        if not self.count or not np.isfinite(used).all() or np.any(used <= 0):
            raise ValueError('Invalid compact SINR')
        return self


def entry_layout(entries, patterns, ns, nf):
    """Cache exact DMRS-aware multiplicities and fallback ordering per layout."""
    from phy.rt_sys.tx_contract import data_mask
    result = []
    for i, e in enumerate(entries):
        layers = e['layers']
        data = data_mask(e['mask'], e['ue'].dmrs_pattern_id)
        positions = np.flatnonzero(np.repeat(data[..., None], layers, axis=-1))
        inverse = np.zeros(len(positions), dtype=np.intp)
        parts = {}
        start = 1  # zero-power allocated REs still become erasures, not unused REs
        for pattern, fulls, fullf, s, f, mapping in patterns:
            if i not in pattern:
                continue
            selected = data[fulls, fullf]
            for k in range(layers):
                flat = (fulls[selected] * nf + fullf[selected]) * layers + k
                inverse[np.searchsorted(positions, flat)] = start + mapping[selected]
                parts[pattern, k] = slice(start, start + len(s))
                start += len(s)
        weights = np.bincount(inverse, minlength=start)
        result.append((data, inverse, weights, parts, start))
    return result
