"""RT reference-signal measurements in watts per active subcarrier.

Unit-norm reference weights use the configured beam/codebook. Interference is
a full-band reference-transmission hypothesis; data SINR comes separately from
the executed grants. No empirical pathloss, antenna gain or fast fading added.
"""
import math
import numpy as np
from phy.rt_sys.numerics import beam_subspace


def reference_measurements(snapshot, state, noise_figure_db=7):
    if getattr(state,'paper',None):return state.paper.serving_references()
    result = {}
    for uid in snapshot.ue_ids:
        serving = state.connections[uid].serving_cell_id
        ue = next(u for u in state.ues if u.ue_id == uid)
        powers = {}
        beam_gains = {}
        for cid in snapshot.cell_ids:
            runtime = state.managed_cells[cid]
            cell = runtime.cell
            nsc = runtime.resource_grid.n_prbs_total * 12
            channel = snapshot.link(uid, cid)
            offset = (channel.shape[-1] - nsc) // 2
            basis = beam_subspace(channel.shape[1], cell.current_codebook_id,
                                  cell.current_beam_profile_id, ue.serving_beam_id)
            vector = basis.sum(axis=1) / math.sqrt(basis.shape[1])
            signal = np.einsum('rtsf,t->rsf', channel[..., offset:offset+nsc], vector)
            gain = float(np.mean(np.sum(abs(signal)**2, axis=0)))
            isotropic_gain = float(np.mean(np.sum(abs(channel[..., offset:offset+nsc])**2, axis=0)))
            beam_gains[cid] = 10*math.log10(max(gain/max(isotropic_gain, 1e-30), 1e-30))
            powers[cid] = (10**((cell.tx_power_dbm-30)/10) * gain / nsc
                           if cell.cell_sleep_mode != 'deep_sleep' else 0.)
        cell = state.managed_cells[serving].cell
        spacing = snapshot.metadata.get('subcarrier_spacing_hz', 30000)
        noise = 10**((-174 + 10*math.log10(spacing) + noise_figure_db - 30)/10)
        signal = powers[serving]
        interference = sum(p for cid, p in powers.items() if cid != serving)
        result[uid] = dict(rsrp_dbm=10*math.log10(max(signal, 1e-30))+30,
                           reference_beam_gain_db=beam_gains[serving],
                           reference_sinr_db=10*math.log10(max(signal/(noise+interference), 1e-30)),
                           serving_cell_id=serving, no_path=signal == 0.,
                           reference_interference_w=interference, noise_per_subcarrier_w=noise)
    return result
