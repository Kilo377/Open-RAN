"""Explicit CSI perturbations and guarded Chase combining assumptions."""
import numpy as np
from phy.rt_sys.contracts import keyed_uniform


def estimate_csi(snapshot,seed,nmse_db):
    if nmse_db is None:return snapshot.cfr
    if not np.isfinite(nmse_db):raise ValueError('Finite CSI NMSE required')
    estimated=snapshot.cfr.copy()
    for ui,uid in enumerate(snapshot.ue_ids):
        for ci,cid in enumerate(snapshot.cell_ids):
            h=snapshot.cfr[ui,ci];gain=float(np.mean(abs(h)**2))
            generator=np.random.default_rng(int(keyed_uniform(seed,snapshot.slot_id,uid,'csi_'+cid)*2**53))
            error=(generator.standard_normal(h.shape,dtype=np.float32)+1j*generator.standard_normal(h.shape,dtype=np.float32))
            estimated[ui,ci]+=error*np.sqrt(gain*10**(nmse_db/10)/2)
    return estimated


class ChaseCombiner:
    def __init__(self,mode='none'):
        if mode not in ('none','chase'):raise ValueError('HARQ mode must be none or chase; IR is not modeled')
        self.mode=mode;self.pending={}

    def combine(self,ue,cell_id,slot,values):
        if self.mode=='none':return values,'disabled'
        key=(ue.ue_id,ue.harq_process_id)
        # Chase repeats the same coded bits; a changed RV/MCS/resource count
        # cannot claim soft-combining gain under this abstraction.
        mask=values>0
        signature=(ue.harq_ndi,ue.target_mcs,ue.mcs_table_id,int(mask.sum()),ue.requested_bytes,
                   ue.harq_redundancy_version,cell_id)
        old=self.pending.get(key);reason='new_tb'
        if ue.is_retx and old is not None:
            reason='reset_incompatible_coded_bits'
            if old['signature']==signature:
                merged=values.copy();merged[mask]+=old['sinr'];values=merged;reason='chase_combined'
        self.pending[key]={'signature':signature,'sinr':values[mask].copy(),'slot':slot}
        # Bound retained stale processes, even if an original HARQ process is dropped.
        self.pending={k:v for k,v in self.pending.items() if slot-v['slot']<=2000}
        return values,reason

    def acknowledge(self,ue):
        self.pending.pop((ue.ue_id,ue.harq_process_id),None)

    def set_payload(self,ue,size):
        old=self.pending.get((ue.ue_id,ue.harq_process_id))
        if old is not None:
            signature=list(old['signature']);signature[4]=size;old['signature']=tuple(signature)
