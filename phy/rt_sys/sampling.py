"""Representative channel coordinates; allocation and power masks stay intact."""
from collections import defaultdict
import numpy as np


def sample_coordinates(coordinates, samples, offset):
    """Map each RE to its own PRB/bin within an exact stream pattern.

    Choose a real coordinate nearest the frequency-bin centre and median
    symbol. Expand resulting SINRs before applying the original data/DMRS mask.
    """
    if samples not in (1, 2, 4, 12):
        raise ValueError('Sampling must divide the 12 subcarriers of a PRB')
    groups = defaultdict(list)
    for i, (symbol, frequency) in enumerate(coordinates):
        relative = frequency - offset
        groups[(relative // 12, min(samples - 1, (relative % 12) * samples // 12))].append(i)
    selected = []
    inverse = np.empty(len(coordinates), dtype=np.int64)
    for (prb, bin_), indices in groups.items():
        center_f = offset + 12 * prb + (bin_ + .5) * 12 / samples - .5
        center_s = float(np.median([coordinates[i][0] for i in indices]))
        best = min(indices, key=lambda i: (abs(coordinates[i][1] - center_f),
                                          abs(coordinates[i][0] - center_s), i))
        inverse[indices] = len(selected)
        selected.append(coordinates[best])
    return selected, inverse
