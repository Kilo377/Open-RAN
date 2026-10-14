"""Immutable MAC/PHY budget. Positions may move; coded shape may not.

Independent Python adaptation of OCUDU's original-TBS retransmission rule,
reference b78d09a77ab060cda752f32483e551f1e9132f66. Our Chase receiver
also requires identical RE ordering, layer count and pilot layout.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


def data_mask(mask, pattern):
    data = mask.copy()
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows):
        data[rows[0], ::2] = False
        if pattern and 'dense' in pattern:
            data[rows[-1], ::2] = False
    return data


def data_re(prbs, symbols, pattern):
    return int(prbs) * (12 * int(symbols) - 6 * (min(int(symbols), 2) if pattern and 'dense' in pattern else 1))


@lru_cache(maxsize=4096)
def nr_budget(table, mcs, count):
    # Same Sionna SYS NR calculator as the actual codec; CPU scalar planning
    # avoids synchronizing the GPU for every scheduler grant.
    from sionna.phy.nr.utils import MCSDecoderNR, TransportBlockNR
    q, rate = MCSDecoderNR(device='cpu')(mcs, {'qam64':1, 'qam256':2, 'qam64_low_se':3}[table], 1)
    bits, _, _ = TransportBlockNR(device='cpu').transport_block_size(q, rate, q * count)
    return int(q), int(q) * count, int(bits)


@dataclass(frozen=True)
class TbTxContract:
    tb_id: str
    cell_id: str
    mcs: int
    table: str
    layers: int
    prbs: int
    symbols: int
    pilot_pattern: str | None
    data_re: int
    coded_bits: int
    nr_tbs_bits: int
    payload_bytes: int
    logical_channels: tuple[tuple[str, int], ...]

    def validate(self, ue, cell_id):
        actual = (str(cell_id), ue.target_mcs, ue.mcs_table_id, int(ue.layer_count or 1),
                  ue.allocated_prbs, ue.allocated_symbols, ue.dmrs_pattern_id,
                  ue.requested_bytes, ue.grant_bytes_budget)
        expected = (self.cell_id, self.mcs, self.table, self.layers, self.prbs,
                    self.symbols, self.pilot_pattern, self.payload_bytes, self.nr_tbs_bits // 8)
        if actual != expected:
            raise ValueError(f'TB contract conflict before transmission: {self.tb_id}: {actual} != {expected}')
        if self.payload_bytes <= 0 or self.payload_bytes * 8 > self.nr_tbs_bits:
            raise ValueError('Invalid reserved TB payload')


@lru_cache(maxsize=4096)
def codec_format_supported(table, mcs, count):
    """Structural codec check only; no channel samples, encoding or decoding.

    Sionna's mother encoder rejects BG1 repetition. The existing RV0 low-rate
    extension supports one TB <=3824 bits. Reduce new-TB resources before
    reservation instead of discovering that boundary after RF transmission.
    """
    from sionna.phy.nr.utils import MCSDecoderNR, calculate_tb_size
    from sionna.phy.fec.ldpc import LDPC5GEncoder
    q,rate=MCSDecoderNR(device='cpu')(mcs,{'qam64':1,'qam256':2,'qam64_low_se':3}[table],1)
    _,coded,bits=nr_budget(table,mcs,count)
    cfg=calculate_tb_size(target_tb_size=bits,num_coded_bits=coded,
        target_coderate=rate,modulation_order=q,verbose=False,device='cpu')
    cb,n=int(cfg[1]),int(np.asarray(cfg[5]).max())
    effective=cb/n
    if effective>.95:return False
    try:
        if effective<.2:raise ValueError('low rate')
        LDPC5GEncoder._sel_basegraph(None,cb,effective,None)
        return True
    except ValueError:
        return bits<=3824 and int(cfg[2])==1
