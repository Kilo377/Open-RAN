"""Cluster-wide downlink RT/SYS executor, retaining legacy PhyResult contracts."""
from dataclasses import fields, asdict,replace
import math
import numpy as np
from models.phy_result import PhyResult, PhyUeResult
from phy.mcs_profiles import select_mcs_index_from_sinr
from phy.rt_sys.numerics import Algebra, beam_subspace
from phy.rt_sys.abstraction import SionnaAbstraction
from phy.rt_sys.fidelity import estimate_csi,ChaseCombiner


class RtSysPhyAdapter:
    cluster_execution = True

    def __init__(self, config, *, abstraction=None, channel_provider=None):
        self.config = dict(config)
        self.device = self.config.get('device', 'cuda')
        if abstraction is None and self.device == 'numpy':
            raise ValueError('NumPy device is fixture-only; production requires PyTorch/Sionna')
        self.math = Algebra(self.device)
        mode=self.config.get('feedback_mode','sys_abstraction')
        if mode not in ('sys_abstraction','nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'):raise ValueError('Unknown RT/SYS feedback mode')
        if abstraction is not None:self.abstraction=abstraction
        elif mode in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'):
            if any(self.config.get(k) for k in ('bler_table_files','beta_table_file','calibration_file')):
                raise ValueError('NR reference cannot use a BLER/EESM calibration table')
            from phy.rt_sys.reference import NrReferenceAbstraction,CachedNrReferenceAbstraction,StatisticalBankAbstraction
            from phy.rt_sys.direct import DirectNrAbstraction
            if mode=='nr_bler_bank' and self.config.get('harq_combining','none')!='none':raise ValueError('Statistical BLER bank cannot claim waveform HARQ combining; use nr_cached_reference')
            backend={'nr_direct':DirectNrAbstraction,'nr_cached_reference':CachedNrReferenceAbstraction,'nr_tb_reference':NrReferenceAbstraction,'nr_bler_bank':StatisticalBankAbstraction}[mode]
            self.abstraction=backend(self.device,self.config.get('bler_target',.1),
                {**self.config.get('reference',{}),'harq_combining':self.config.get('harq_combining','none')})
        else:
            if not self.config.get('calibration_file') and not self.config.get('allow_unvalidated_sys',False):
                raise ValueError('Unvalidated SYS BLER/EESM feedback requires explicit diagnostic opt-in; use nr_tb_reference for actual feedback')
            self.abstraction=SionnaAbstraction(self.device, self.config.get('bler_target', .1),self.config.get('bler_table_files'),self.config.get('beta_table_file'),self.config.get('calibration_file'))
        self.channel_provider = channel_provider
        if self.channel_provider is None and self.config.get('manifest_path'):
            from pathlib import Path
            import json
            from phy.rt_sys.channel import RtChannelProvider
            root=Path(__file__).resolve().parents[2]
            cache=Path(self.config.get('cache_root',root/'channel_cache')).resolve()
            if not cache.is_relative_to(root):
                raise ValueError('RT cache must remain inside the independent copy')
            manifest=json.loads(Path(self.config['manifest_path']).read_text())
            self.channel_provider=RtChannelProvider(manifest,cache,device=self.device)
        self.seed = int(self.config.get('rng_seed', 1))
        self.trace = {}
        self.current_snapshot = None
        self.scheduler_snapshot=None
        self.combiner=ChaseCombiner('none' if mode in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank') else self.config.get('harq_combining','none'))

    def run_slot(self, phy_input):
        raise RuntimeError('RT/SYS requires run_cluster_slot with all managed-cell grants')

    def release_inactive_contexts(self,state):
        receiver=getattr(self.abstraction,'harq',None)
        if receiver is not None:
            valid={(u.ue_id,p.process_id): (p.ndi,str(state.connections[u.ue_id].serving_cell_id))
                for u in state.ues for p in u.harq.processes if p.tx_contract is not None}
            for key,context in list(receiver.pending.items()):
                sig=context['signature']
                if key not in valid or valid[key]!=(sig[0],sig[4]):
                    receiver.pending.pop(key)
                    receiver.stats['lifecycle_context_cleanup']+=1

    def bind_state(self, state):
        if self.channel_provider is None:
            raise RuntimeError('No RT channel provider configured')
        self.release_inactive_contexts(state)
        self.current_snapshot = self.channel_provider.snapshot(int(state.slot_id))
        self.current_snapshot.validate()
        self.scheduler_snapshot=replace(self.current_snapshot,
            cfr=estimate_csi(self.current_snapshot,self.seed,self.config.get('csi_error_nmse_db')))
        state.rt_channel_snapshot=self.scheduler_snapshot
        state.rt_harq_combining=self.config.get('harq_combining','none')
        for i, uid in enumerate(self.current_snapshot.ue_ids):
            link = state.links[uid]
            link.x_m, link.y_m, link.z_m = map(float, self.current_snapshot.positions[i])
        return self.current_snapshot

    def _mask(self, ue, ns, nf, offset=0, active_subcarriers=None):
        mask = np.zeros((ns, nf), bool)
        syms = list(ue.allocated_symbol_indices) or list(range(ue.start_symbol, ue.start_symbol+ue.allocated_symbols))
        scs = list(ue.allocated_subcarrier_indices)
        if not scs:
            prbs = list(ue.resource_block_indices) or list(range(ue.start_prb, ue.start_prb+ue.allocated_prbs))
            scs = [p*12+i for p in prbs for i in range(12)]
        if active_subcarriers is not None and scs and (min(scs)<0 or max(scs)>=active_subcarriers):
            raise ValueError(f'Grant exceeds active PRB band for {ue.ue_id}')
        scs = [s+offset for s in scs]
        if not syms or not scs or min(syms)<0 or max(syms)>=ns or min(scs)<0 or max(scs)>=nf:
            raise ValueError(f'Invalid grant grid for {ue.ue_id}')
        mask[np.ix_(syms, scs)] = True
        return mask

    def run_cluster_slot(self, phy_inputs, channel_snapshot):
        snapshot = channel_snapshot.validate()
        if not phy_inputs:
            return {}
        if any(p.slot_id != snapshot.slot_id for p in phy_inputs):
            raise ValueError('Stale channel snapshot')
        if len({p.cell_id for p in phy_inputs}) != len(phy_inputs):
            raise ValueError('Duplicate cell PHY input')
        radio = [p.runtime_context for p in phy_inputs if p.runtime_context is not None]
        for name in ('carrier_frequency_hz','bandwidth_hz','subcarrier_spacing_hz','n_prbs_total','num_ofdm_symbols_per_slot','cyclic_prefix_type'):
            if len({getattr(r,name) for r in radio}) > 1:
                raise ValueError(f'V1 requires common frequency/numerology: {name}')
        ns, nf = snapshot.cfr.shape[-2:]
        nr, nt = snapshot.cfr.shape[2:4]
        if radio and (radio[0].num_ofdm_symbols_per_slot != ns):
            raise ValueError('RT and grant symbol grids differ')
        metadata = snapshot.metadata
        if radio and any(abs(getattr(r,k)-metadata[k]) > 1e-3 for r in radio
                         for k in ('carrier_frequency_hz','subcarrier_spacing_hz','bandwidth_hz') if k in metadata):
            raise ValueError('RT and runtime carrier grids differ')
        a = self.math
        h = a.array(snapshot.cfr)
        estimated=a.array(estimate_csi(snapshot,self.seed,self.config.get('csi_error_nmse_db'))) if self.config.get('csi_error_nmse_db') is not None else h
        noise = 10**((-174+10*math.log10(self.config.get('subcarrier_spacing_hz',30000))+
                      self.config.get('noise_figure_db',7)-30)/10)
        entries = []
        precoder_trace=[]
        all_seen = set()
        cell_trace = {}
        for p in phy_inputs:
            if p.cell_id not in snapshot.cell_ids:
                raise ValueError(f'Unknown cell {p.cell_id}')
            active = []
            for ue in p.ue_inputs:
                if ue.ue_id not in snapshot.ue_ids:
                    raise ValueError(f'Missing RT UE {ue.ue_id}')
                if ue.ue_id in all_seen:
                    raise ValueError(f'UE served by multiple cells: {ue.ue_id}')
                all_seen.add(ue.ue_id)
                if not ue.is_scheduled:
                    continue
                contract=getattr(ue,'tx_contract',None)
                if contract is not None:
                    contract.validate(ue,p.cell_id)
                    receiver=getattr(self.abstraction,'harq',None)
                    if ue.is_retx and (ue.ue_id,int(ue.harq_process_id)) not in receiver.pending:
                        raise ValueError('Retransmission missing physical codeword before transmission')
                layers = int(ue.layer_count or 1)
                if not 1 <= layers <= min(nr,nt):
                    raise ValueError('Requested layer count exceeds antenna capability')
                entry = {'p':p, 'ue':ue, 'ci':snapshot.cell_ids.index(p.cell_id),
                         'ui':snapshot.ue_ids.index(ue.ue_id), 'mask':self._mask(ue,ns,nf,(nf-p.runtime_context.n_prbs_total*12)//2 if p.runtime_context else 0,p.runtime_context.n_prbs_total*12 if p.runtime_context else None),
                         'layers':layers, 'sinr':np.zeros((ns,nf,layers)),
                         'inter':[], 'precoder_norms':[]}
                entries.append(entry); active.append(entry)
            # Uniform energy across all active RE-streams including pilots;
            # average OFDM-symbol power sums to the configured cell budget.
            total = sum(e['mask'].sum()*e['layers'] for e in active)
            budget = 10**((float(p.tx_power_dbm)-30)/10)
            if metadata.get('sleep_modes',{}).get(p.cell_id) == 'deep_sleep':
                budget = 0.0
            power = budget*ns/max(1,total)
            for e in active: e['power'] = power
            cell_trace[p.cell_id] = {'budget_w':budget, 'average_used_w':power*total/ns,
                                    're_streams':int(total),'emitted_re_streams':int(total) if budget>0 else 0,
                                    'occupied_re':int(np.logical_or.reduce([e['mask'] for e in active]).sum()) if active and budget>0 else 0,
                                    'grid_re':int(ns*nf),
                                    'tx_power_dbm':float(p.tx_power_dbm)}
        # Batch all REs with the same transmitter/stream set across symbols.
        # Each RE retains its true channel and the original per-symbol ridge.
        patterns={}
        for symbol in range(ns):
            for f in range(nf):
                pattern=tuple(i for i,e in enumerate(entries) if e['mask'][symbol,f] and e['power']>0)
                if pattern:patterns.setdefault(pattern,[]).append((symbol,f))
        for pattern,coordinates in patterns.items():
            symbols,frequencies=map(list,zip(*coordinates))
            tx = []
            for ci in sorted({entries[i]['ci'] for i in pattern}):
                group = [entries[i] for i in pattern if entries[i]['ci']==ci]
                stream_rows = []
                owners = []
                dft=beam_subspace(nt,0,0,0)
                directions=set()
                for member in group:
                    sub=beam_subspace(nt,member['p'].current_codebook_id,
                        member['p'].current_beam_profile_id,int(member['ue'].serving_beam_id))
                    directions.update(np.argmax(abs(dft.conj().T @ sub),axis=0).tolist())
                basis=dft[:,sorted(directions)]
                for e in group:
                    hu = estimated[e['ui'],ci][:,:,symbols,frequencies].moveaxis(-1,0) if a.torch else np.moveaxis(estimated[e['ui'],ci][:,:,symbols,frequencies],-1,0)
                    u,_,_ = a.svd(hu)
                    effective = a.adj(u)[..., :e['layers'], :] @ hu
                    stream_rows.append(effective)
                    owners.extend((e,k) for k in range(e['layers']))
                if len(owners) > basis.shape[-1]:
                    raise ValueError('Scheduled MU layers exceed physical beam subspace')
                rows = a.cat(stream_rows, -2)
                # Relative ridge keeps physical scale and works at small RT gains.
                row_power=a.numpy(a.sum(abs(rows)**2,-1))
                symbol_indices={symbol:np.flatnonzero(np.asarray(symbols)==symbol) for symbol in set(symbols)}
                scales={symbol:max(float(row_power[indices].mean()),1e-30) for symbol,indices in symbol_indices.items()}
                alpha=a.array(np.asarray([scales[s] for s in symbols])[:,None,None])*self.config.get('rzf_relative_regularization',.01)
                weights = a.rzf(rows,a.array(basis),alpha)
                samples=a.numpy(weights)
                correlations=None
                if len({e['ue'].ue_id for e,k in owners})==2 and len(owners)==2:
                    projected=a.numpy(rows)@basis
                    numerator=abs(np.sum(projected[:,0].conj()*projected[:,1],axis=-1))**2
                    denominator=np.sum(abs(projected[:,0])**2,axis=-1)*np.sum(abs(projected[:,1])**2,axis=-1)
                    correlations=np.clip(numerator/np.maximum(denominator,1e-60),0,1)
                for symbol,indices in sorted(symbol_indices.items()):
                    sample=samples[indices[0]]
                    precoder_trace.append({'cell_id':snapshot.cell_ids[ci], 'symbol':symbol,
                        'representative_subcarrier':frequencies[indices[0]],'subcarrier_count':len(indices),
                        'beam_subspace_dimension':basis.shape[-1],
                        'streams':[{'ue_id':e['ue'].ue_id,'layer':k,'power_per_re_w':e['power']} for e,k in owners],
                        'weights_real':sample.real.tolist(),'weights_imag':sample.imag.tolist(),
                        'mean_pair_channel_correlation':float(correlations[indices].mean()) if correlations is not None else None})
                for j,(e,k) in enumerate(owners):
                    w = weights[...,j:j+1]*math.sqrt(e['power'])
                    e['precoder_norms'].append(float(a.numpy(a.sum(abs(weights[...,j:j+1])**2,-2)).mean()))
                    tx.append((e,k,w))
            for e,k,w in tx:
                received = []
                desired = None
                estimated_desired=None;estimated_received=[]
                inter_power = np.zeros(len(frequencies))
                for other,ok,ow in tx:
                    hu = h[e['ui'],other['ci']][:,:,symbols,frequencies]
                    hu = hu.moveaxis(-1,0) if a.torch else np.moveaxis(hu,-1,0)
                    signal = hu @ ow
                    eh=estimated[e['ui'],other['ci']][:,:,symbols,frequencies]
                    eh=eh.moveaxis(-1,0) if a.torch else np.moveaxis(eh,-1,0)
                    esignal=eh@ow
                    if other is e and ok == k:
                        desired = signal;estimated_desired=esignal
                    else:
                        received.append(signal)
                        estimated_received.append(esignal)
                        if other['ci'] != e['ci']:
                            inter_power += a.numpy(a.sum(abs(signal)**2,(-2,-1)))
                others = a.cat(received) if received else a.array(np.zeros((len(frequencies),nr,0)))
                eothers=a.cat(estimated_received) if estimated_received else a.array(np.zeros((len(frequencies),nr,0)))
                values = a.numpy(a.lmmse(desired,others,noise,estimated_desired,eothers)
                    if self.config.get('csi_error_nmse_db') is not None else a.lmmse(desired,others,noise))
                e['sinr'][symbols,frequencies,k] = values
                e['inter'].extend(inter_power.tolist())
        results = {}
        trace_ues = []
        for p in phy_inputs:
            outputs = []
            for ue in p.ue_inputs:
                e = next((e for e in entries if e['ue'] is ue),None)
                payload = {f.name:getattr(ue,f.name) for f in fields(PhyUeResult)
                           if hasattr(ue,f.name) and f.name not in {'sinr_db','bler','queue_bytes'}}
                result = PhyUeResult(**payload)
                metrics={}
                result.queue_bytes = ue.queue_bytes
                result.phy_mode = 'rt_sys'; result.waveform_backend = 'sionna_rt_sys'
                if e is not None:
                    from phy.rt_sys.tx_contract import data_mask
                    data = data_mask(e['mask'],ue.dmrs_pattern_id)
                    values = e['sinr'].copy(); values[~data,:] = 0
                    result.resource_grid_re_count = int(e['mask'].sum())
                    result.effective_data_re_count = int(data.sum())
                    result.dmrs_overhead_fraction = 1-data.sum()/max(1,e['mask'].sum())
                    no_path=not np.any(values[data,:]>0)
                    # Zero SINR on an allocated stream is an erasure, not an
                    # unscheduled RE. SYS uses zero as its resource-mask value.
                    if not no_path:values[data,:]=np.maximum(values[data,:],1e-30)
                    usable = values[values>0]
                    values,combining_reason=self.combiner.combine(ue,p.cell_id,p.slot_id,values)
                    mean = float(usable.mean()) if not no_path else 0
                    result.sinr_db = result.effective_sinr_db = 10*math.log10(max(mean,1e-30))
                    result.pre_eq_sinr_db = result.post_eq_sinr_db = result.sinr_db
                    result.bler = 1.0
                    if not no_path:
                        maximum = int(ue.mcs_cap if ue.mcs_cap is not None else 27)
                        mcs = int(ue.target_mcs) if ue.target_mcs is not None else int(select_mcs_index_from_sinr(result.sinr_db,mcs_table_id=ue.mcs_table_id,max_mcs_index=maximum))
                        if mcs > maximum: raise ValueError('MCS target exceeds cap')
                        fixed=ue.is_retx if getattr(self.abstraction,'feedback_mode',None) in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank') else ue.target_mcs is not None
                        context={}
                        if getattr(self.abstraction,'supports_harq_context',False):
                            context['harq_context']=dict(process_id=ue.harq_process_id if ue.harq_process_id is not None else 0,ndi=ue.harq_ndi if ue.harq_ndi is not None else 0,is_retx=ue.is_retx,cell_id=p.cell_id)
                        metrics = self.abstraction.evaluate(values, ue.mcs_table_id, mcs,
                            self.seed,p.slot_id,ue.ue_id,fixed=fixed,**context)
                        if metrics['mcs'] > maximum: raise ValueError('MCS cap excludes calibrated SYS entries')
                        result.link_adaptation_reason = ue.link_adaptation_reason or metrics.get('mcs_projection') or ('nr_independent_probe_mapping' if metrics.get('feedback_mode')=='nr_tb_reference' else 'sys_bler_mapping')
                        physical_capacity=metrics['capacity_bytes']
                        if getattr(ue,'tx_contract',None) is not None and physical_capacity != ue.tx_contract.nr_tbs_bits//8:
                            raise RuntimeError('Actual codec differs from planned TB contract')
                        result.mcs_index = metrics['mcs']; result.modulation_order = metrics['qm']
                        result.reference_tb_ack = metrics.get('reference_tb_ack')
                        result.code_rate = metrics['rate']; result.bler = metrics['tbler']
                        result.effective_sinr_db = 10*math.log10(max(metrics['sinr_eff'],1e-30))
                        result.tb_size_bytes = min(ue.queue_bytes,ue.requested_bytes,ue.grant_bytes_budget, metrics['capacity_bytes'])
                        if ue.is_retx and result.tb_size_bytes < ue.requested_bytes:
                            # A pending TB cannot be partially ACKed under the existing HARQ contract.
                            result.ack = False
                        else: result.ack = bool(metrics['ack']) and result.tb_size_bytes>0
                    result.transport_block_size_bytes = result.scheduled_bytes = result.tb_size_bytes
                    result.delivered_bytes = result.tb_size_bytes if result.ack else 0
                    self.combiner.set_payload(ue,result.tb_size_bytes)
                    if result.ack:
                        self.combiner.acknowledge(ue)
                        if getattr(self.abstraction,'supports_harq_context',False):self.abstraction.harq.acknowledge(ue.ue_id,ue.harq_process_id)
                    result.throughput_mbps = result.delivered_bytes*8/((p.runtime_context.slot_duration_ms if p.runtime_context is not None else self.config.get("slot_duration_ms",1))*1000)
                    result.num_layers = e['layers']
                    result.inter_cell_interference_dbm = 10*math.log10(max(float(np.mean(e['inter'])) if e['inter'] else 0,1e-30))+30
                    result.rank = result.ri = e['layers']; result.cqi = max(0,min(15,result.mcs_index//2))
                    result.spatial_mode = 'mu_mimo' if ue.spatial_group_size>1 else 'su_mimo'
                    trace_ues.append({'ue_id':ue.ue_id,'cell_id':p.cell_id,'sinr_db':result.effective_sinr_db,
                        'bler':result.bler,'mcs':result.mcs_index,'allocated_re':result.resource_grid_re_count,
                        'data_re':result.effective_data_re_count,'layers':e['layers'],'ack':result.ack,
                        'tb_bytes':result.tb_size_bytes,'delivered_bytes':result.delivered_bytes,
                        'inter_cell_interference_dbm':result.inter_cell_interference_dbm,
                        'precoder_column_norm_mean':float(np.mean(e['precoder_norms'])) if e['precoder_norms'] else 0,
                        'beam_subspace_codebook':p.current_codebook_id,
                        'scheduled':True,'queue_bytes':ue.queue_bytes,
                        'throughput_mbps':result.throughput_mbps,
                        'mcs_mapping':result.link_adaptation_reason or ''})
                    trace_ues[-1].update(raw_sinr_db=result.sinr_db,harq_combining=combining_reason,
                        physical_tb_capacity_bytes=physical_capacity if not no_path else 0,
                        padding_bytes=max(0,physical_capacity-result.tb_size_bytes) if not no_path else 0)
                else:
                    trace_ues.append({'ue_id':ue.ue_id,'cell_id':p.cell_id,'sinr_db':0.0,
                        'bler':0.0,'mcs':0,'allocated_re':0,'data_re':0,'layers':0,'ack':False,
                        'tb_bytes':0,'delivered_bytes':0,'inter_cell_interference_dbm':-270.0,
                        'precoder_column_norm_mean':0.0,'beam_subspace_codebook':p.current_codebook_id,
                        'scheduled':False,'queue_bytes':ue.queue_bytes,'throughput_mbps':0.0,'mcs_mapping':''})
                    trace_ues[-1].update(raw_sinr_db=0.,harq_combining='not_scheduled',physical_tb_capacity_bytes=0,padding_bytes=0)
                contract=getattr(ue,'tx_contract',None)
                trace_ues[-1].update(tb_id=contract.tb_id if contract else None,is_retx=bool(ue.is_retx),
                    actual_decode=metrics.get('reference_tb_ack') is not None,
                    mcs_table=ue.mcs_table_id,tx_contract=asdict(contract) if contract else None,
                    link_adaptation_reason=ue.link_adaptation_reason)
                outputs.append(result)
                if getattr(self.abstraction,'feedback_mode',None) in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'):
                    defaults={'ack_source':'no_path' if e is not None else 'not_scheduled','bler_probe_errors':0,'bler_probe_samples':0,
                        'bler_wilson_95':[1.,1.] if e is not None else [0.,0.],'amc_probes':[],'unsupported_nr_mcs':[],
                        'bler_probe_seed':None,'ack_seed':None,'reference_tb_errors':None,'reference_tb_ack':False,
                        'effective_sinr_kind':'arithmetic_mean_diagnostic_only'}
                    trace_ues[-1].update({key:metrics.get(key,value) for key,value in defaults.items()})
                    trace_ues[-1]['mac_capacity_guard']=bool(ue.is_retx and metrics.get('ack_source')=='actual_nr_crc_and_bit_check' and result.tb_size_bytes<ue.requested_bytes) if e is not None else False
                    trace_ues[-1]['reference_rv']=0
                    trace_ues[-1]['mac_requested_rv']=ue.harq_redundancy_version
                    trace_ues[-1]['amc_selection_target_met']=metrics.get('amc_selection_target_met')
                    trace_ues[-1]['amc_selection_criterion']=metrics.get('amc_selection_criterion', 'not_evaluated')
                    trace_ues[-1]['coding_backend']=metrics.get('coding_backend', 'not_evaluated')
                    trace_ues[-1]['harq_reference']=metrics.get('harq_reference')
                    trace_ues[-1]['bler_estimate_source']=metrics.get('bler_estimate_source','not_evaluated')
                    trace_ues[-1]['bler_estimate_scope']='Single-attempt BLER; actual LLR combining only in reference Chase mode'
            scheduled = [r for r in outputs if r.is_scheduled]
            results[p.cell_id] = PhyResult(p.slot_id,p.cell_id,outputs,
                float(np.mean([r.sinr_db for r in scheduled])) if scheduled else 0,
                float(np.mean([r.bler for r in scheduled])) if scheduled else 0,
                sum(r.tb_size_bytes for r in outputs))
        self.trace = {'mcs_table_policy':'SYS default EESM tables 1/2; low-SE MAC requests use table 1', 'slot_id':snapshot.slot_id,'cache_key':snapshot.cache_key,'cells':cell_trace,'ues':trace_ues,'precoders':precoder_trace}
        self.trace.update(ideal_csi=self.config.get('csi_error_nmse_db') is None,harq_combining_mode=self.config.get('harq_combining','none'))
        self.trace['phy_calibration_status']='custom_tables_require_validation' if self.config.get('bler_table_files') else 'default_tables_failed_expanded_validation'
        artifact=getattr(self.abstraction,'calibration',None)
        if artifact is not None:
            self.trace['phy_calibration_status']='independently_validated_exact_re_domain'
            self.trace['phy_calibration_digest']=artifact['digest']
            self.trace['phy_calibration_scope']='Exact table/MCS/coded-RE tuples and validation profiles only; not URLLC tails or waveform HARQ'
        if getattr(self.abstraction,'feedback_mode',None) in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'):
            self.trace.update(phy_calibration_status='nr_tb_equivalent_link_reference',default_bler_eesm_used=False,
                feedback_mode=self.abstraction.feedback_mode,reference_decode_trials=self.abstraction.decode_trials+getattr(getattr(self.abstraction,'bank',None),'stats',{}).get('calibration_decode_trials',0),
                phy_calibration_scope='Actual RV0 NR-TB CRC/LDPC/QAM on independent Gaussian post-LMMSE resources; finite-probe or local validated BLER; same-codeword RV0/Chase LLR combining, no IR',
                mcs_table_policy='PDSCH tables 1/2/3; grant target is an AMC ceiling; actual retransmission MCS/table fixed; single-CB low-rate RV0 repetition; unsupported multi-CB repetition is explicit')
        if getattr(self.abstraction,'feedback_mode',None)=='nr_direct':
            self.trace.update(phy_calibration_status='actual_nr_no_probes',
                phy_calibration_scope='Actual NR TB decode at MAC MCS; no target BLER guarantee or probability prediction',
                mcs_table_policy='MAC-selected MCS; SYS NR TBS; RV0/Chase compatibility guards retained')
            for row in self.trace['ues']:
                row['bler_estimate_scope']='Realized block error indicator; controller BLER uses actual ACK/NACK window'
        if getattr(self.abstraction,'feedback_mode',None)=='nr_bler_bank':
            self.trace.update(phy_calibration_status='locally_validated_statistical_bler_bank',
                phy_calibration_scope='Independent actual NR bin-bound validation; sampled TB ACK inside domain; explicit NR on rejected entries; no waveform HARQ',
                default_bler_eesm_used=False)
        return results
