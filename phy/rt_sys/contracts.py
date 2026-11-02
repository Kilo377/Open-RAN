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


def has_tb_feedback(row):
    """A transmitted TB outcome, including fast sampling; no-path is not a trial."""
    if 'feedback_available' in row:return bool(row['feedback_available'])
    return bool(row.get('actual_decode') or row.get('ack_source') in (
        'fast_sys_bernoulli','official_sys_default_bernoulli','validated_bler_bank_bernoulli'))


def source_fingerprint(root):
    """Record executed copy content, including uncommitted source/config edits."""
    result=hashlib.sha256()
    for name in ('core','phy','ran','ric','state','models','traffic','experiments','configs'):
        for path in sorted((root/name).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.json','.yaml','.yml'):
                result.update(str(path.relative_to(root)).encode());result.update(path.read_bytes())
    return result.hexdigest()


@dataclass
class ChannelSnapshot:
    slot_id: int
    ue_ids: tuple
    cell_ids: tuple
    # [UE, cell, rx antenna, tx antenna, symbol, subcarrier]
    cfr: np.ndarray | None
    positions: np.ndarray
    cache_key: str
    metadata: dict = field(default_factory=dict)
    # Fast system runs may reconstruct only the coordinates actually consumed.
    path_sampler: object = None

    @property
    def grid_shape(self):
        return self.path_sampler.shape if self.path_sampler is not None else self.cfr.shape

    def samples(self, ui, ci, symbols, frequencies):
        """Paired coordinates, returned as [RX, TX, coordinate]."""
        if self.path_sampler is not None:
            return self.path_sampler.paired(ui, ci, symbols, frequencies)
        return self.cfr[ui,ci][:,:,symbols,frequencies]

    def samples_for_links(self,links,symbols,frequencies):
        if self.path_sampler is not None:
            return self.path_sampler.paired_links(links,symbols,frequencies)
        return {link:self.samples(*link,symbols,frequencies) for link in links}

    def grid(self, ui, ci, symbols, frequencies):
        """Cartesian symbol/frequency subset; physical grid indices are retained."""
        if self.path_sampler is not None:
            return self.path_sampler.cartesian(ui, ci, symbols, frequencies)
        s=np.asarray(symbols);f=np.asarray(frequencies)
        if len(s) and len(f) and np.all(np.diff(s)==1) and np.all(np.diff(f)==1):
            return self.cfr[ui,ci][:,:,s[0]:s[-1]+1,f[0]:f[-1]+1]
        return np.take(np.take(self.cfr[ui,ci],symbols,axis=-2),frequencies,axis=-1)
    # Execution caches stay outside the serialized host contract.
    @property
    def device_cfr(self):return getattr(self,'_device_cfr',None)

    @device_cfr.setter
    def device_cfr(self,value):self._device_cfr=value

    @property
    def device_backend(self):return getattr(self,'_device_backend',None)

    @device_backend.setter
    def device_backend(self,value):self._device_backend=value

    def validate(self):
        if len(set(self.ue_ids)) != len(self.ue_ids) or len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError('Duplicate channel IDs')
        shape=self.grid_shape
        if len(shape) != 6 or shape[:2] != (len(self.ue_ids), len(self.cell_ids)):
            raise ValueError('CFR must be [UE,cell,RX,TX,symbol,subcarrier]')
        if self.positions.shape != (len(self.ue_ids), 3) or not np.isfinite(self.positions).all():
            raise ValueError('Invalid channel snapshot')
        if self.path_sampler is None and not np.isfinite(self.cfr).all():
            raise ValueError('Invalid channel snapshot')
        if self.path_sampler is not None and self.cfr is not None:
            raise ValueError('Choose dense CFR or on-demand paths, not both')
        if self.metadata.get('normalized', False):
            raise ValueError('Unit-normalized propagation channels are forbidden')
        if self.device_cfr is not None and self.device_cfr.shape!=shape:
            raise ValueError('Retained device CFR shape differs from host CFR')
        return self

    def link(self, ue_id, cell_id):
        if self.path_sampler is None:
            return self.cfr[self.ue_ids.index(ue_id),self.cell_ids.index(cell_id)]
        return self.grid(self.ue_ids.index(ue_id),self.cell_ids.index(cell_id),
                         np.arange(self.grid_shape[-2]),np.arange(self.grid_shape[-1]))
