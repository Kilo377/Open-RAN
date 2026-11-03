"""Physical primitives for the opt-in paper interfaces; no link probes."""
import math
import numpy as np
from phy.rt_sys.numerics import beam_subspace


def ret_basis(rows, cols, tilt_deg, codebook=0, beam=0):
    """Local z points up; Sionna orders rows within each column, downward in local z.

    A fixed analog vertical vector restricts digital RZF to horizontal ports.
    Tilting a full-rank digital basis would cancel the intended RET response.
    """
    if not 0 <= tilt_deg <= 15 or codebook not in (0, 1):
        raise ValueError('Unsupported RET angle/horizontal codebook')
    z=(rows-1)/2-np.arange(rows)
    vertical=np.exp(2j*np.pi*.5*z*np.sin(np.deg2rad(tilt_deg)))/np.sqrt(rows)
    horizontal=np.exp(2j*np.pi*np.arange(cols)[:,None]*np.arange(cols)[None,:]/cols)/np.sqrt(cols)
    if codebook==1:horizontal=horizontal[:,[beam%cols]]
    return np.kron(horizontal,vertical[:,None])


def basis(snapshot, cell, beam=0, reference=False, cache_static=False):
    rows=int(snapshot.metadata.get('tx_rows',1));cols=int(snapshot.metadata.get('tx_cols',snapshot.grid_shape[3]))
    if getattr(cell,'ret_enabled',False):
        if rows*cols!=snapshot.grid_shape[3]:raise ValueError('RET array layout missing/mismatched')
        return ret_basis(rows,cols,cell.ret_tilt_deg,cell.ret_codebook_id,beam)
    basis_for=beam_subspace
    if cache_static:
        from phy.rt_sys.numerics import cached_beam_subspace
        basis_for=cached_beam_subspace
    return basis_for(snapshot.grid_shape[3],cell.current_codebook_id,cell.current_beam_profile_id,beam)


def prb_powers(cell, nprb):
    vector=getattr(cell,'prb_power_w',None)
    if vector is None:return np.full(nprb,10**((float(cell.tx_power_dbm)-30)/10)/nprb)
    return np.asarray(vector,float)


def quantify_cqi(sinr_db, thresholds=None):
    bins=np.asarray(list(range(-7,22,2)) if thresholds is None else thresholds,float)
    if bins.shape!=(15,) or not np.isfinite(bins).all() or not np.all(np.diff(bins)>0):
        raise ValueError('CQI requires 15 finite increasing thresholds')
    if sinr_db is None or not np.isfinite(sinr_db):return None
    return int(np.searchsorted(bins,sinr_db,side='right'))


def probe_gains(snapshot,ui,ci,cell,nprb,beam_reports,beam_id,cache_static=False):
    ns,nf=snapshot.grid_shape[-2:];offset=(nf-nprb*12)//2
    sampler=snapshot.path_sampler;nt=snapshot.grid_shape[3]
    h=None if sampler is not None else snapshot.grid(ui,ci,np.arange(ns),np.arange(offset,offset+nprb*12))
    def power(vectors):
        if sampler is not None:return sampler.mean_reference_power(ui,ci,vectors,nprb,offset).mean(axis=0)
        signal=np.einsum('rtsf,tk->rksf',h,vectors)
        return np.mean(np.sum(abs(signal)**2,axis=(0,1))/vectors.shape[1],axis=0).reshape(nprb,12).mean(axis=1)
    probes=[]
    for beam in range(int(snapshot.metadata.get('tx_cols',nt)) if getattr(cell,'ret_enabled',False) else nt):
        b=basis(snapshot,cell,beam,True,cache_static=cache_static);v=b[:,beam%b.shape[1]]
        gain=power(v[:,None])
        probes.append(gain)
    best=int(np.argmax([g.mean() for g in probes]));gain=probes[best];alternatives=[]
    if beam_reports:
        for cb in (0,1,2):
            for profile in (0,1):
                if cache_static:
                    from phy.rt_sys.numerics import cached_beam_subspace
                    b=cached_beam_subspace(nt,cb,profile,beam_id)
                else:b=beam_subspace(nt,cb,profile,beam_id)
                g=power(b)
                alternatives.append((cb,profile,g))
    return gain,best,alternatives


class ReferenceGainCache:
    def __init__(self,update_ms,slot_ms,cache_static=False):
        slots=update_ms/slot_ms
        if not np.isfinite(slots) or slots<1 or not np.isclose(slots,round(slots)):
            raise ValueError('Reference gain interval must align to whole slots')
        self.slots=int(round(slots));self.epoch=None;self.cache={};self.hits=0;self.builds=0
        self.cache_static=cache_static
    def get(self,snapshot,state,ui,ci,cell,nprb,beam_reports):
        epoch=(snapshot.cache_key,snapshot.slot_id//self.slots,snapshot.metadata.get('geometry_block_slot'))
        if epoch!=self.epoch:self.cache.clear();self.epoch=epoch
        beam_id=state.connections[snapshot.ue_ids[ui]].serving_beam_id
        key=(ui,ci,nprb,cell.current_codebook_id,cell.current_beam_profile_id,
            getattr(cell,'ret_enabled',False),getattr(cell,'ret_tilt_deg',0),getattr(cell,'ret_codebook_id',0),
            snapshot.metadata.get('tx_rows'),snapshot.metadata.get('tx_cols'),beam_reports,beam_id if beam_reports else 0)
        if key not in self.cache:
            self.cache[key]=probe_gains(snapshot,ui,ci,cell,nprb,beam_reports,beam_id,cache_static=self.cache_static);self.builds+=1
        else:self.hits+=1
        return self.cache[key]


def references(snapshot,state,noise_figure_db=7):
    """Common UE/candidate, wideband/subband reference-transmission hypothesis.

    This is a configured full-band reference hypothesis, not data interference
    or an always-on waveform added to the emitted-RF energy ledger.
    """
    result={};spacing=snapshot.metadata.get('subcarrier_spacing_hz',30000)
    noise=10**((-174+10*math.log10(spacing)+noise_figure_db-30)/10)
    for ui,uid in enumerate(snapshot.ue_ids):
        powers={};beams={};alternatives={}
        for ci,cid in enumerate(snapshot.cell_ids):
            runtime=state.managed_cells[cid];cell=runtime.cell;nprb=runtime.resource_grid.n_prbs_total
            beam_reports=state.paper.config.get('beam_reports',False)
            cache=getattr(state.paper,'gain_cache',None)
            gain,best,alternative=(cache.get(snapshot,state,ui,ci,cell,nprb,beam_reports) if cache is not None else
                probe_gains(snapshot,ui,ci,cell,nprb,beam_reports,state.connections[uid].serving_beam_id))
            p=prb_powers(cell,nprb)*gain/12
            if cell.cell_sleep_mode=='deep_sleep':p=np.zeros(nprb)
            powers[cid]=p;beams[cid]=best
            if beam_reports:
                alternatives[cid]=[(cb,profile,prb_powers(cell,nprb)*g/12) for cb,profile,g in alternative]
        result[uid]={}
        for cid,p in powers.items():
            inter=sum((v for c,v in powers.items() if c!=cid),np.zeros_like(p));ratio=p/(inter+noise)
            result[uid][cid]=dict(rsrp_dbm=10*math.log10(max(float(p.mean()),1e-30))+30,
                reference_sinr_db=10*math.log10(max(float(ratio.mean()),1e-30)),
                prb_sinr_db=(10*np.log10(np.maximum(ratio,1e-30))).tolist(),
                reference_interference_w=float(inter.mean()),noise_per_subcarrier_w=noise,
                no_path=not bool(np.any(p>0)),candidate_beam_id=beams[cid])
            if cid in alternatives:
                result[uid][cid]['beam_reference_reports']=[dict(cell_id=cid,codebook_id=cb,beam_profile_id=profile,
                    reference_sinr_db=10*math.log10(max(float((signal/(inter+noise)).mean()),1e-30)),
                    valid=bool(np.any(signal>0)) and state.managed_cells[cid].cell.cell_sleep_mode=='active')
                    for cb,profile,signal in alternatives[cid]]
    return result
