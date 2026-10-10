"""Sionna 2.2 SYS bridge. No empirical BLER fallback."""
import math
import numpy as np
from phy.rt_sys.contracts import keyed_uniform

TABLES = {'qam64': 1, 'qam256': 2, 'qam64_low_se': 3}


class SionnaAbstraction:
    def __init__(self, device='cuda', target_bler=.1):
        import torch
        from importlib.metadata import version
        if version('sionna') != '2.2.0':
            raise RuntimeError('RT/SYS backend requires sionna==2.2.0; use its dedicated environment')
        from sionna.sys import PHYAbstraction
        from sionna.phy.nr.utils import MCSDecoderNR, TransportBlockNR
        self.torch, self.device, self.target_bler = torch, device, target_bler
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Sionna SYS GPU unavailable; CPU must be explicitly requested')
        self.phy = PHYAbstraction(device=device)
        self.decoder = MCSDecoderNR(device=device)
        self.transport = TransportBlockNR(device=device)

    def evaluate(self, sinr, table, mcs, seed, slot, ue_id, fixed=False):
        t = self.torch
        if table not in ('qam64','qam256'):
            raise ValueError(f'SYS default EESM supports qam64/qam256 only, requested {table}')
        table_index = TABLES[table]
        # Resource mask is represented by zero SINRs. Data REs of disconnected
        # links are handled as deterministic NACKs by the adapter.
        tensor = t.as_tensor(sinr[None, :, :, None, :], dtype=t.float32, device=self.device)
        count = int(np.count_nonzero(sinr > 0))
        supported = sorted(int(k) for k in self.phy.bler_table['category'][1]['index'][table_index]['MCS'])
        eligible = [k for k in supported if k <= int(mcs)]
        candidates = list(reversed(eligible)) if eligible else [supported[0]]
        for candidate in candidates:
            index = t.tensor([[candidate]], dtype=t.int32, device=self.device)
            with t.no_grad():
                _, _, eff, tbler, cbler = self.phy(index, sinr=tensor,
                    mcs_table_index=table_index, mcs_category=1)
                qm, rate = self.decoder(index, table_index, 1)
                capacity, _, _ = self.transport.transport_block_size(qm, rate, qm*count)
            error = float(tbler.item())
            if fixed or (math.isfinite(error) and (error <= self.target_bler or candidate == candidates[-1])):
                break
        if not math.isfinite(error):
            raise ValueError(f'Uncalibrated SYS BLER table entry: {table}/{candidate}')
        if not 0 <= error <= 1:
            raise ValueError(f'Invalid SYS TBLER {error}')
        return {'mcs':candidate, 'qm':int(qm.item()), 'rate':float(rate.item()),
                'capacity_bytes':int(capacity.item())//8,
                'sinr_eff':float(eff.item()), 'tbler':error, 'cbler':float(cbler.item()),
                'ack':keyed_uniform(seed,slot,ue_id) >= error,
                'mcs_projection':None if candidate == mcs else f'calibrated_table_projection:{mcs}->{candidate}'}
