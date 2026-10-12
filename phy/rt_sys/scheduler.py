"""Existing MAC policy restricted to MCS entries supported by SYS EESM."""
from ran.scheduler import SimpleScheduler
import numpy as np
from phy.rt_sys.numerics import beam_subspace


class RtSysScheduler(SimpleScheduler):
    def _build_decision(self,state,*args,**kwargs):
        decision=super()._build_decision(state,*args,**kwargs)
        if getattr(state,'rt_harq_combining','none')=='chase':
            for grant in decision.grants:grant.harq_redundancy_version=0
        return decision

    def _compute_spatial_compatibility_score(self,state,left_ue,right_ue):
        snapshot=getattr(state,'rt_channel_snapshot',None)
        if snapshot is None:
            raise RuntimeError('RT MU scheduling requires a channel snapshot')
        if self._get_layer_count(state,left_ue)+self._get_layer_count(state,right_ue)>self._get_max_total_spatial_layers(state):
            return 0.
        nt=snapshot.cfr.shape[3]
        dft=beam_subspace(nt,0,0,0); directions=set(); vectors=[]; channels=[]
        for ue in (left_ue,right_ue):
            basis=beam_subspace(nt,state.cell.current_codebook_id,state.cell.current_beam_profile_id,ue.serving_beam_id)
            directions.update(np.argmax(abs(dft.conj().T@basis),axis=0).tolist())
            h=snapshot.link(ue.ue_id,state.cell.cell_id)
            # Sample eight frequencies over the active band, not guard tones.
            nsc=state.resource_grid.n_prbs_total*12; offset=(h.shape[-1]-nsc)//2
            indices=np.linspace(offset,offset+nsc-1,min(8,nsc),dtype=int)
            h=h[:,:,0,indices].transpose(2,0,1)
            channels.append(h)
        basis=dft[:,sorted(directions)]
        for h in channels:
            projected=h@basis
            covariance=np.einsum('frt,frv->tv',projected.conj(),projected)
            _,v=np.linalg.eigh(covariance)
            if float(np.trace(covariance).real)<1e-25:return 0.
            vectors.append(v[:,-1])
        if len(directions)<self._get_layer_count(state,left_ue)+self._get_layer_count(state,right_ue):return 0.
        correlation=float(abs(np.vdot(*vectors))**2)
        return 1-correlation if correlation<.8 else 0.

    def _derive_target_mcs(self, ue, **kwargs):
        target = super()._derive_target_mcs(ue, **kwargs)
        cap = super()._derive_mcs_cap(ue, kwargs['is_retx'], target)
        # The legacy MAC can emit target > cap for retransmissions or after
        # slice-aware boosts. Reconcile before constructing the grant/TB.
        # SYS table 1 starts at MCS 3; table 2 is selected only at high MCS.
        return max(3, min(target, cap if cap is not None else 27))

    def _derive_mcs_table_id(self, *args, **kwargs):
        selected = super()._derive_mcs_table_id(*args, **kwargs)
        # Default SYS EESM beta parameters cover NR tables 1 and 2 only.
        # URLLC/retransmission grants use table 1; never fabricate table-3 beta.
        return 'qam64' if selected == 'qam64_low_se' else selected
