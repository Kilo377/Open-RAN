"""Existing MAC priorities with RT-compatible resource packing and MCS."""
from ran.scheduler import SimpleScheduler
import numpy as np
from phy.rt_sys.numerics import beam_subspace


class RtSysScheduler(SimpleScheduler):
    def run(self,state):
        self._rt_packing=self._get_packing_strategy(state)
        return super().run(state)

    def _max_grants_allowed(self,eligible_count):
        strategy=getattr(self,'_rt_packing',self.packing_strategy)
        limit=(self.total_symbols_per_slot//self.min_symbols_per_grant if strategy=='time_first'
               else self.total_prbs_per_slot//self.min_prbs_per_grant)
        return max(0,min(eligible_count,self.max_ues_per_slot,limit))

    def _apply_spatial_multiplexing(self,state,grants,ue_map):
        super()._apply_spatial_multiplexing(state,grants,ue_map)
        self._repack_rt_groups(state,grants,ue_map)

    def _repack_rt_groups(self,state,grants,ue_map):
        # Partition one axis only. Splitting both PRBs AND symbols before FDM
        # left most of the Cartesian grid unused. Reclaim partner rectangles
        # after MU grouping and give each group the full orthogonal axis.
        groups={}
        for grant in grants:
            key=('mu',grant.spatial_group_id) if grant.spatial_group_id is not None else ('su',grant.ue_id)
            groups.setdefault(key,[]).append(grant)
        if not groups:return
        strategy=self._get_packing_strategy(state)
        minimum=self.min_symbols_per_grant if strategy=='time_first' else self.min_prbs_per_grant
        total=self.total_symbols_per_slot if strategy=='time_first' else self.total_prbs_per_slot
        if len(groups)*minimum>total:raise ValueError('RT grant groups exceed packed resource budget')
        weights=[sum(max(0.,g.priority_score) for g in members) for members in groups.values()]
        remaining=total-len(groups)*minimum
        extras=(super()._allocate_extra_symbols(weights,remaining) if strategy=='time_first'
                else super()._allocate_extra_prbs(weights,remaining)) or [0]*len(groups)
        offset=0
        for members,extra in zip(groups.values(),extras):
            size=minimum+extra
            for grant in members:
                grant.allocated_prbs=self.total_prbs_per_slot if strategy=='time_first' else size
                grant.allocated_symbols=size if strategy=='time_first' else self.total_symbols_per_slot
                grant.start_prb=0 if strategy=='time_first' else offset
                grant.start_symbol=offset if strategy=='time_first' else 0
                grant.resource_block_indices=list(range(grant.start_prb,grant.start_prb+grant.allocated_prbs))
                grant.ofdm_symbol_indices=list(range(grant.start_symbol,grant.start_symbol+grant.allocated_symbols))
                grant.grant_bytes_budget=self._estimate_grant_bytes_budget(allocated_prbs=grant.allocated_prbs,
                    allocated_symbols=grant.allocated_symbols,target_mcs=int(grant.target_mcs or 0),
                    requested_bytes=grant.requested_bytes,mcs_table_id=grant.mcs_table_id,
                    tbs_selection_method=grant.tbs_selection_method,dmrs_pattern_id=grant.dmrs_pattern_id,
                    num_layers=self._get_layer_count(state,ue_map[grant.ue_id]))
            offset+=size
        assert offset==total

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
        if getattr(ue, 'rt_feedback_mode', None) in ('nr_tb_reference','nr_cached_reference','nr_bler_bank'):
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
        if getattr(ue, 'rt_feedback_mode', None) in ('nr_tb_reference','nr_cached_reference','nr_bler_bank'):
            process = self._get_active_harq_process(ue) if kwargs['is_retx'] else None
            if process is not None and process.mcs_table_id is not None:
                return process.mcs_table_id
            return 'qam64_low_se' if ue.latency_budget_ms <= 20 else 'qam64'
        selected = super()._derive_mcs_table_id(*args, **kwargs)
        # Default SYS EESM beta parameters cover NR tables 1 and 2 only.
        # URLLC/retransmission grants use table 1; never fabricate table-3 beta.
        return 'qam64' if selected == 'qam64_low_se' else selected

    def _derive_mcs_cap(self, ue, is_retx, target_mcs):
        if getattr(ue, 'rt_feedback_mode', None) in ('nr_tb_reference','nr_cached_reference','nr_bler_bank'):
            return int(target_mcs)
        return super()._derive_mcs_cap(ue, is_retx, target_mcs)

    def _get_effective_link_quality_db(self, ue):
        if getattr(ue, 'rt_feedback_mode', None) in ('nr_tb_reference','nr_cached_reference','nr_bler_bank'):
            return float(ue.post_eq_sinr_db if getattr(ue, 'last_scheduled_slot', None) is not None else ue.sinr_db)
        return super()._get_effective_link_quality_db(ue)
