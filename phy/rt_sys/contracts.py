"""Explicit channel contract, with absolute (not unit-normalized) CFR."""
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def keyed_uniform(seed, slot, ue_id, purpose='tb_ack'):
    key = digest([int(seed), int(slot), str(ue_id), purpose])
    return (int(key[:16], 16) + .5) / 2**64


@dataclass
class ChannelSnapshot:
    slot_id: int
    ue_ids: tuple
    cell_ids: tuple
    # [UE, cell, rx antenna, tx antenna, symbol, subcarrier]
    cfr: np.ndarray
    positions: np.ndarray
    cache_key: str
    metadata: dict = field(default_factory=dict)

    def validate(self):
        if len(set(self.ue_ids)) != len(self.ue_ids) or len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError('Duplicate channel IDs')
        if self.cfr.ndim != 6 or self.cfr.shape[:2] != (len(self.ue_ids), len(self.cell_ids)):
            raise ValueError('CFR must be [UE,cell,RX,TX,symbol,subcarrier]')
        if self.positions.shape != (len(self.ue_ids), 3) or not np.isfinite(self.cfr).all():
            raise ValueError('Invalid channel snapshot')
        if self.metadata.get('normalized', False):
            raise ValueError('Unit-normalized propagation channels are forbidden')
        return self

    def link(self, ue_id, cell_id):
        return self.cfr[self.ue_ids.index(ue_id), self.cell_ids.index(cell_id)]
