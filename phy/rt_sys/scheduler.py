"""Existing MAC priorities with RT-compatible resource packing and MCS."""
from ran.scheduler import SimpleScheduler
import numpy as np
from phy.rt_sys.numerics import beam_subspace


class RtSysScheduler(SimpleScheduler):
    _paper=None
    def run(self,state):
        self._rt_state=state
        self._paper=getattr(state,'paper',None)
        if self._paper:
            cap=self._paper.layer_caps[state.cell.cell_id]
            state.scheduler.max_total_spatial_layers=min(cap,self._paper.adapter.current_snapshot.metadata['tx_cols'] if state.cell.ret_codebook_id==0 else 1) if state.cell.ret_enabled else cap
        self._contracts=bool(getattr(state,'rt_tx_contracts',False))
        self._rt_packing=self._get_packing_strategy(state)
        if self._contracts:
            from phy.rt_sys.olla import OllaState
            for ue in state.ues:
                if not hasattr(ue,'du_olla'):ue.du_olla=OllaState()
                retained=[]
                for process in ue.harq.processes:
                    c=process.tx_contract
                    if c is not None and c.cell_id!=str(state.cell.cell_id):
                        self._record_harq_process_flush(ue.harq,reason='handover',slot_id=state.slot_id,process=process)
                        ue.packet_ledger.release(c.tb_id)
                    else:retained.append(process)
                ue.harq.processes=retained
                ue.harq.pending_retx_bytes=sum(p.pending_bytes for p in retained)
                ue.harq.pending_retx_logical_channels=[lc for p in retained for lc in p.pending_logical_channels]
                ue.du_olla.consume(state.slot_id,state.cell.cell_id)
        result=super().run(state)
        if self._contracts:
            for ue in state.ues:
                active={p.tb_id for p in ue.harq.processes}
                for tb in list(ue.packet_ledger.reservations):
                    if tb not in active:ue.packet_ledger.release(tb)
        return result

    def _is_ue_connected(self,state,ue):
        if getattr(state,'paper',None):
            cell=state.cell
            blocked=(getattr(cell,'ue_prb_limits',{}).get(ue.ue_id,1)==0 or
                (getattr(cell,'prb_power_w',None) is not None and not any(cell.prb_power_w)))
            if blocked:
                for p in ue.harq.processes:
                    if p.tx_contract is not None and getattr(p,'last_paper_wait_slot',None)!=state.slot_id:
                        p.last_paper_wait_slot=state.slot_id
                        ue.harq.retx_deferred_resource=getattr(ue.harq,'retx_deferred_resource',0)+1
                        ue.harq.events.append(dict(kind='retx_deferred_resource',slot_id=state.slot_id,tb_id=p.tb_id,reason='zero_power_or_budget'))
                return False
        return super()._is_ue_connected(state,ue)

    def _get_layer_count(self,state,ue):
        layers=super()._get_layer_count(state,ue)
        if getattr(state,'paper',None) and getattr(state.cell,'ret_enabled',False):
            layers=min(layers,state.paper.adapter.current_snapshot.metadata['tx_cols'] if state.cell.ret_codebook_id==0 else 1)
        return layers

    def _max_grants_allowed(self,eligible_count):
        strategy=getattr(self,'_rt_packing',self.packing_strategy)
        limit=(self.total_symbols_per_slot//self.min_symbols_per_grant if strategy=='time_first'
               else self.total_prbs_per_slot//self.min_prbs_per_grant)
        return max(0,min(eligible_count,self.max_ues_per_slot,limit))

    def _apply_spatial_multiplexing(self,state,grants,ue_map):
        if not self._contracts:
            super()._apply_spatial_multiplexing(state,grants,ue_map)
            self._repack_rt_groups(state,grants,ue_map)
            return
        from phy.rt_sys.tx_contract import nr_budget, data_re, codec_format_supported
        retx = [g for g in grants if g.is_retx]
        new = [g for g in grants if not g.is_retx]
        occupied = np.zeros((self.total_symbols_per_slot,self.total_prbs_per_slot), bool)
        limits=getattr(state.cell,'ue_prb_limits',{}) if self._paper else {}
        vector=getattr(state.cell,'prb_power_w',None) if self._paper else None
        if vector is not None:occupied[:,np.asarray(vector)<=0]=True
        kept = []
        def process(g):
            return next(p for p in ue_map[g.ue_id].harq.processes if p.process_id==g.harq_process_id)
        retx.sort(key=lambda g: (process(g).retx_deadline_slot or 10**12,
            process(g).last_tx_slot or -1, g.ue_id, g.harq_process_id))
        for g in retx:
            c = process(g).tx_contract
            if c is None:
                raise ValueError('Retransmission lacks original TB contract')
            allowed=limits.get(g.ue_id,self.total_prbs_per_slot)>=c.prbs
            location = next(((s,p) for s in range(occupied.shape[0]-c.symbols+1)
                for p in range(occupied.shape[1]-c.prbs+1)
                if allowed and not occupied[s:s+c.symbols,p:p+c.prbs].any()), None)
            if location is None:
                ue_map[g.ue_id].harq.retx_deferred_resource=getattr(ue_map[g.ue_id].harq,'retx_deferred_resource',0)+1
                ue_map[g.ue_id].harq.events.append(dict(kind='retx_deferred_resource',
                    slot_id=state.slot_id, tb_id=c.tb_id, process_id=g.harq_process_id))
                continue
            s,p = location
            occupied[s:s+c.symbols,p:p+c.prbs] = True
            g.start_symbol,g.start_prb = s,p
            g.allocated_symbols,g.allocated_prbs = c.symbols,c.prbs
            g.target_mcs=g.mcs_cap=c.mcs;g.mcs_table_id=c.table
            g.dmrs_pattern_id=c.pilot_pattern
            g.requested_bytes=c.payload_bytes;g.grant_bytes_budget=c.nr_tbs_bits//8
            g.tx_contract=c;g.spatial_group_id=None;g.spatial_group_size=1
            kept.append(g)
        super()._apply_spatial_multiplexing(state,new,ue_map)
        groups={}
        for g in new:
            key=('mu',g.spatial_group_id) if g.spatial_group_id is not None else ('su',g.ue_id)
            groups.setdefault(key,[]).append(g)
        time_first=self._get_packing_strategy(state)=='time_first'
        # New transmissions use full orthogonal axis. Original retransmission
        # rectangles are reserved first and are never passed to a repacker.
        free=~occupied.any(axis=1 if time_first else 0)
        runs=[];start=None
        for i,value in enumerate(list(free)+[False]):
            if value and start is None:start=i
            if not value and start is not None:runs.append([start,i-start]);start=None
        minimum=self.min_symbols_per_grant if time_first else (1 if self._paper else self.min_prbs_per_grant)
        members_list=list(groups.values())
        for index,members in enumerate(members_list):
            candidates=[r for r in runs if r[1]>=minimum]
            if not candidates:continue
            r=max(candidates,key=lambda r:r[1])
            remaining_groups=len(members_list)-index
            size=max(minimum,r[1]//remaining_groups)
            if self._paper:
                size=min(size,*(limits.get(g.ue_id,self.total_prbs_per_slot) for g in members))
            initial_size=size
            while size>=minimum:
                if all(codec_format_supported(g.mcs_table_id,int(g.target_mcs),
                    data_re(self.total_prbs_per_slot if time_first else size,
                            size if time_first else self.total_symbols_per_slot,g.dmrs_pattern_id)
                    * self._get_layer_count(state,ue_map[g.ue_id])) for g in members):break
                size-=1
            if size<minimum:continue
            for g in members:
                if size<initial_size:g.constraint_hits.append('nr_codec_resource_shape_limit')
                g.start_symbol,g.start_prb=(r[0],0) if time_first else (0,r[0])
                g.allocated_symbols,g.allocated_prbs=(size,self.total_prbs_per_slot) if time_first else (self.total_symbols_per_slot,size)
                count=data_re(g.allocated_prbs,g.allocated_symbols,g.dmrs_pattern_id)*self._get_layer_count(state,ue_map[g.ue_id])
                _,_,bits=nr_budget(g.mcs_table_id,int(g.target_mcs),count)
                g.grant_bytes_budget=bits//8
                kept.append(g)
            r[0]+=size;r[1]-=size
        for g in kept:
            g.resource_block_indices=list(range(g.start_prb,g.start_prb+g.allocated_prbs))
            g.ofdm_symbol_indices=list(range(g.start_symbol,g.start_symbol+g.allocated_symbols))
        grants[:]=kept

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
        from phy.rt_sys.tx_contract import TbTxContract, data_re, nr_budget
        from phy.rt_sys.control import limit_logical_allocations
        decision=super()._build_decision(state,*args,**kwargs)
        if self._contracts:
            ue_map=kwargs['ue_map']
            valid=[]
            for g in decision.grants:
                ue=ue_map[g.ue_id]
                process=next(p for p in ue.harq.processes if p.process_id==g.harq_process_id)
                if not g.is_retx:
                    ledger=ue.packet_ledger
                    allocations=[]
                    for lc in g.logical_channel_allocations:
                        lc.scheduled_bytes=min(lc.scheduled_bytes,ledger.available(lc.logical_channel_id))
                        allocations.append((lc.logical_channel_id,lc.scheduled_bytes))
                    ranges=ledger.reserve(process.tb_id,allocations)
                    counts={}
                    for lc,_,a,b in ranges:counts[lc]=counts.get(lc,0)+b-a
                    payload=sum(counts.values())
                    if not payload:
                        ledger.release(process.tb_id)
                        continue
                    layers=self._get_layer_count(state,ue)
                    count=data_re(g.allocated_prbs,g.allocated_symbols,g.dmrs_pattern_id)*layers
                    _,coded,bits=nr_budget(g.mcs_table_id,int(g.target_mcs),count)
                    g.tx_contract=TbTxContract(process.tb_id,str(state.cell.cell_id),int(g.target_mcs),
                        g.mcs_table_id,layers,g.allocated_prbs,g.allocated_symbols,g.dmrs_pattern_id,
                        count,coded,bits,payload,tuple(counts.items()))
                    process.tx_contract=g.tx_contract
                    g.requested_bytes=payload
                    for lc in g.logical_channel_allocations:
                        lc.scheduled_bytes=counts.get(lc.logical_channel_id,0)
                    g.logical_channel_allocations=limit_logical_allocations(g.logical_channel_allocations,payload)
                g.link_adaptation_reason=getattr(ue,'olla_selection_reason','original_retx_contract') if not g.is_retx else 'original_retx_contract'
                valid.append(g)
            decision.grants=valid
            # Empty provisional HARQ processes must not occupy a process slot.
            for ue in ue_map.values():
                ue.harq.processes[:]=[p for p in ue.harq.processes
                    if p.pending_bytes>0 or any(g.ue_id==ue.ue_id and g.harq_process_id==p.process_id for g in valid)]
                active={p.tb_id for p in ue.harq.processes}
                for tb in list(ue.packet_ledger.reservations):
                    if tb not in active:ue.packet_ledger.release(tb)
        if self._contracts or getattr(state,'rt_harq_combining','none')=='chase':
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
        if getattr(ue, 'rt_feedback_mode', None) == 'nr_direct' and getattr(self,'_contracts',False):
            process = self._get_active_harq_process(ue) if kwargs['is_retx'] else None
            if process is not None and process.mcs_index is not None:
                return process.mcs_index
            from phy.mcs_profiles import select_mcs_index_from_sinr
            measured=next((u for u in getattr(getattr(self._rt_state,'rt_du_observation',None),'ue_observations',[]) if u.ue_id==ue.ue_id),None)
            table='qam64_low_se' if ue.latency_budget_ms<=20 else 'qam64'
            maximum=16 if ue.latency_budget_ms<=20 else 27
            age=self._rt_state.slot_id-(getattr(measured,'measurement_produced_at_slot',0) or 0)
            max_age=max(1,getattr(measured,'sinr_window_slots',1)+getattr(measured,'queue_delay_slots_applied',0)+5)
            ue.olla_selection=dict(slot_id=self._rt_state.slot_id,cell_id=str(self._rt_state.cell.cell_id),
                table=table,min_mcs=0,max_mcs=maximum,measurement_age_slots=age,
                measurement_produced_at_slot=getattr(measured,'measurement_produced_at_slot',None),
                measurement_delivered_at_slot=getattr(measured,'measurement_delivered_at_slot',None),
                reference_sinr_db=getattr(measured,'reference_sinr_db',None),max_age_slots=max_age,
                offset_db=ue.du_olla.offsets.get(table,0.))
            if measured is None or measured.reference_sinr_db is None or measured.measurement_delivered_at_slot is None or measured.serving_cell_id!=str(self._rt_state.cell.cell_id) or age>max_age:
                ue.olla_selection_reason='missing_or_stale_reference_measurement'
                ue.olla_selection['reason']=ue.olla_selection_reason
                return 0
            ue.olla_selection_reason='delivered_reference_sinr_plus_du_olla'
            ue.olla_selection['reason']=ue.olla_selection_reason
            return select_mcs_index_from_sinr(measured.reference_sinr_db+ue.du_olla.offsets.get(table,0.),
                mcs_table_id=table,max_mcs_index=maximum)
        if getattr(ue,'rt_feedback_mode',None)=='nr_direct':
            process=self._get_active_harq_process(ue) if kwargs['is_retx'] else None
            if process is not None and process.mcs_index is not None:return process.mcs_index
            return super()._derive_target_mcs(ue,**kwargs)
        if getattr(ue, 'rt_feedback_mode', None) in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'):
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
        if getattr(ue, 'rt_feedback_mode', None) in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'):
            process = self._get_active_harq_process(ue) if kwargs['is_retx'] else None
            if process is not None and process.mcs_table_id is not None:
                return process.mcs_table_id
            return 'qam64_low_se' if ue.latency_budget_ms <= 20 else 'qam64'
        selected = super()._derive_mcs_table_id(*args, **kwargs)
        # Default SYS EESM beta parameters cover NR tables 1 and 2 only.
        # URLLC/retransmission grants use table 1; never fabricate table-3 beta.
        return 'qam64' if selected == 'qam64_low_se' else selected

    def _derive_mcs_cap(self, ue, is_retx, target_mcs):
        if getattr(ue, 'rt_feedback_mode', None) in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'):
            return int(target_mcs)
        return super()._derive_mcs_cap(ue, is_retx, target_mcs)

    def _get_effective_link_quality_db(self, ue):
        if getattr(ue, 'rt_feedback_mode', None) in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'):
            return float(ue.post_eq_sinr_db if getattr(ue, 'last_scheduled_slot', None) is not None else ue.sinr_db)
        return super()._get_effective_link_quality_db(ue)
