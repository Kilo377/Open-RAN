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
        if getattr(ue, 'rt_feedback_mode', None) == 'nr_tb_reference':
            process = self._get_active_harq_process(ue) if kwargs['is_retx'] else None
            if process is not None and process.mcs_index is not None:
                return process.mcs_index
            # Actual post-LMMSE NR probes choose within this ceiling. A stale
            # empirical SINR guess must not exclude executable low-rate entries.
            return 16 if ue.latency_budget_ms <= 20 else 27
        target = super()._derive_target_mcs(ue, **kwargs)
        cap = super()._derive_mcs_cap(ue, kwargs['is_retx'], target)
        # The legacy MAC can emit target > cap for retransmissions or after
        # slice-aware boosts. Reconcile before constructing the grant/TB.
        # SYS table 1 starts at MCS 3; table 2 is selected only at high MCS.
        return max(3, min(target, cap if cap is not None else 27))

    def _derive_mcs_table_id(self, *args, **kwargs):
        ue = args[0] if args else kwargs['ue']
        if getattr(ue, 'rt_feedback_mode', None) == 'nr_tb_reference':
            process = self._get_active_harq_process(ue) if kwargs['is_retx'] else None
            if process is not None and process.mcs_table_id is not None:
                return process.mcs_table_id
            return 'qam64_low_se' if ue.latency_budget_ms <= 20 else 'qam64'
        selected = super()._derive_mcs_table_id(*args, **kwargs)
        # Default SYS EESM beta parameters cover NR tables 1 and 2 only.
        # URLLC/retransmission grants use table 1; never fabricate table-3 beta.
        return 'qam64' if selected == 'qam64_low_se' else selected

    def _derive_mcs_cap(self, ue, is_retx, target_mcs):
        if getattr(ue, 'rt_feedback_mode', None) == 'nr_tb_reference':
            return int(target_mcs)
        return super()._derive_mcs_cap(ue, is_retx, target_mcs)

    def _get_effective_link_quality_db(self, ue):
        if getattr(ue, 'rt_feedback_mode', None) == 'nr_tb_reference':
            return float(ue.post_eq_sinr_db if getattr(ue, 'last_scheduled_slot', None) is not None else ue.sinr_db)
        return super()._get_effective_link_quality_db(ue)
