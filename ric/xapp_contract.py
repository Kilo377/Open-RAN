"""Public, measured KPI surface and backend action capabilities.

Diagnostics remain in saved Observation/trace artifacts. Controllers receive a
separate projection without truth, CSI tensors, pseudo-CQI or live references.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from types import SimpleNamespace
import math
from phy.rt_sys.control import PHY, SCHEDULER, ASSOCIATION, ADMIN

SUPPORTED_ACTIONS=frozenset(PHY|SCHEDULER|ASSOCIATION)


@dataclass(frozen=True)
class KpiSpec:
    entity: str
    unit: str
    source: str
    window: str
    sample_count: str | None = None


KPI_REGISTRY={
    'candidate_rsrp_dbm':KpiSpec('ue','dBm','RT_candidate_reference_filtered_noisy_delayed','sinr_window_slots','sinr_effective_sample_count'),
    'queue_bytes':KpiSpec('ue','byte','delivered_queue_report','single_report'),
    'hol_delay_ms':KpiSpec('ue','ms','delivered_packet_hol_report','single_report'),
    'reference_sinr_db':KpiSpec('ue','dB','RT_reference_measurement_filtered_noisy_delayed','sinr_window_slots','sinr_effective_sample_count'),
    'rsrp_dbm':KpiSpec('ue','dBm','RT_reference_measurement_filtered_noisy_delayed','sinr_window_slots','sinr_effective_sample_count'),
    'data_sinr_db':KpiSpec('ue','dB','RT_data_measurement_filtered_noisy_delayed','sinr_window_slots','sinr_effective_sample_count'),
    'throughput_mbps':KpiSpec('ue','Mbit/s','delivered_goodput_report','throughput_window_slots','throughput_window_slots'),
    'retx_fraction':KpiSpec('ue','ratio','delivered_actual_attempt_counter','bler_window_tbs','retx_sample_count'),
    'reference_noise_dbm':KpiSpec('ue','dBm','configured_receiver_noise_report','sinr_window_slots','sinr_effective_sample_count'),
    'first_tx_bler':KpiSpec('ue','ratio','delivered_first_tx_actual_crc_window','bler_window_tbs','first_tx_sample_count'),
    'deadline_miss_rate':KpiSpec('ue','ratio','delivered_packet_deadline_counter','since_start','deadline_sample_count'),
    'bler':KpiSpec('ue','ratio','delivered_actual_crc_window_all_attempts','bler_window_tbs','bler_sample_count'),
    'resource_grid_utilization':KpiSpec('cell','ratio','completed_slot_resource_counter','one_completed_slot'),
    'rf_tx_power_w':KpiSpec('cell','W','completed_slot_RF_energy_counter','one_completed_slot'),
}
# Identity/configuration, counters and contract metadata are deliberately
# distinguished from radio/queue measurements in the public records.
UE_CONFIG='handover_pending_target_cell_id ue_id serving_cell_id serving_beam_id slice_id traffic_type latency_budget_ms priority connection_state handover_count handover_interruption_slots interruption_remaining_slots last_handover_slot'.split()
CELL_CONFIG='cell_id cell_type is_primary carrier_frequency_hz bandwidth_hz subcarrier_spacing_hz tx_power_dbm current_codebook_id current_beam_profile_id scheduler_mode scheduler_policy_hint latency_bias slice_weight_adjustment beam_candidate_preference power_cap_dbm cell_sleep_mode traffic_steering_preference mobility_bias_db handover_cost_db handover_interruption_slots steering_hysteresis_db measurement_event_trigger measurement_event_offset_db resource_grid_total_re resource_grid_allocated_re resource_grid_utilization rf_tx_power_w rf_power_produced_at_slot connected_ue_count scheduled_ue_count active_ue_ids scheduled_ue_ids'.split()
TOP_CONFIG='slot_id cell_id runtime_mode current_codebook_id current_beam_profile_id tx_power_dbm scheduler_mode scheduler_packing_strategy carrier_frequency_hz bandwidth_hz subcarrier_spacing_hz num_ofdm_symbols_per_slot gnb_num_tx_ant resource_grid_total_re resource_grid_allocated_re resource_grid_utilization produced_at_slot delivered_at_slot measurement_produced_at_slot measurement_delivered_at_slot measurement_age_slots measurement_queue_delay_slots measurement_sinr_window_slots measurement_bler_window_tbs measurement_throughput_granularity_slots'.split()
TIMING='retx_sample_count first_tx_sample_count deadline_sample_count measurement_produced_at_slot measurement_delivered_at_slot measurement_age_slots queue_age_slots queue_produced_at_slot sinr_window_slots sinr_effective_sample_count sinr_window_len bler_sample_count bler_window_sample_count bler_window_tbs throughput_window_slots throughput_granularity_slots'.split()


@dataclass(frozen=True)
class XAppContract:
    required_kpis: tuple[str,...]
    actions: frozenset[str]
    entities: tuple[str,...]=('ue','cell','cluster')
    max_age_slots: int | None=None  # None = reporting delay + configured window + xApp interval
    cell_ids: tuple[str,...]=()  # empty means all managed cells

    @property
    def writable_actions(self):return self.actions & SUPPORTED_ACTIONS

    @property
    def available(self):return bool(self.writable_actions) and all(k in KPI_REGISTRY for k in self.required_kpis)


def contract_for(actions,name=None):
    actions=frozenset(actions)
    required={'queue_bytes','throughput_mbps'}
    if actions & PHY:required|={'reference_sinr_db'}
    if actions & SCHEDULER:required|={'hol_delay_ms','resource_grid_utilization'}
    if actions & ASSOCIATION:required|={'reference_sinr_db','candidate_rsrp_dbm'}
    if actions & {'tx_power_dbm','power_cap_dbm','cell_sleep_mode'}:required|={'rf_tx_power_w'}
    if name:
        from ric.near_rt.xapp_variant_catalog import canonical_xapp_category
        category=canonical_xapp_category(name)
        if category in ('latency_scheduler_xapp','urllc_guard_xapp','goal_parameter_tuning_xapp'):
            required|={'bler','retx_fraction','hol_delay_ms'}
        if category in ('ran_slicing_manager_xapp','spectrum_optimizer_xapp'):
            required|={'retx_fraction','hol_delay_ms'}
        if category in ('traffic_steering_xapp','cell_sleep_xapp','spectrum_optimizer_xapp'):
            required|={'rsrp_dbm','reference_sinr_db','reference_noise_dbm','resource_grid_utilization'}
        if category=='mobility_predictor_xapp':required|={'independent_reported_cqi'}
        if category=='massive_mimo_beamformer_xapp':required|={'measured_precoder_match_score'}
    return XAppContract(tuple(sorted(required)),actions)


def is_rt(observation):
    return any(getattr(u,'sinr_kind',None)=='rt_full_band_reference' for u in getattr(observation,'ue_observations',[]))


def _copy(source,names):
    return {name:deepcopy(getattr(source,name)) for name in names if hasattr(source,name)}


def project(observation,contract,interval_slots=5,now_slot=None):
    now=observation.slot_id if now_slot is None else now_slot
    result=SimpleNamespace(**_copy(observation,TOP_CONFIG))
    result.slot_id=now
    result.rt_public_projection=True
    result.ue_observations=[];result.cell_observations=[];result.kpi_records=[]
    errors=[f'unknown_required_kpi:{k}' for k in contract.required_kpis if k not in KPI_REGISTRY]
    for u in observation.ue_observations:
        if contract.cell_ids and u.serving_cell_id not in contract.cell_ids:continue
        v=SimpleNamespace(**_copy(u,UE_CONFIG+TIMING))
        v.sinr_kind=getattr(u,'sinr_kind',None)
        v.kpi_metadata={}
        for name,spec in KPI_REGISTRY.items():
            if spec.entity!='ue':continue
            produced=getattr(u,'queue_produced_at_slot',None) if name in ('queue_bytes','hol_delay_ms') else getattr(u,'measurement_produced_at_slot',None)
            delivered=getattr(u,'measurement_delivered_at_slot',None)
            samples=getattr(u,spec.sample_count,None) if spec.sample_count else 1
            window=(1 if spec.window=='single_report' else 'since_start' if spec.window=='since_start' else getattr(u,spec.window,None))
            value=getattr(u,name,None)
            if name=='candidate_rsrp_dbm':
                candidates=[c for c in getattr(u,'candidate_cell_observations',[])
                    if getattr(c,'measurement_produced_at_slot',None) is not None
                    and getattr(c,'measurement_delivered_at_slot',None) is not None
                    and c.measurement_produced_at_slot<=c.measurement_delivered_at_slot<=now
                    and now-c.measurement_produced_at_slot<=(contract.max_age_slots if contract.max_age_slots is not None else (getattr(c,'measurement_window_slots',1) or 1)+interval_slots+(getattr(observation,'report_delay_slots',0) or 0))]
                value=tuple((c.cell_id,c.rsrp_dbm) for c in candidates) or None
                window=max((c.measurement_window_slots or 1 for c in candidates),default=1)
                samples=len(candidates)
            window_kind=('transport_blocks' if spec.window=='bler_window_tbs' else
                         'cumulative' if spec.window=='since_start' else
                         'reports' if spec.window=='single_report' else 'slots')
            age=None if produced is None else now-produced
            max_age=contract.max_age_slots
            if max_age is None:
                delay=(getattr(observation,'measurement_queue_delay_slots',0) or 0)+(getattr(observation,'report_delay_slots',0) or 0)
                max_age=delay+(window if window_kind=='slots' and isinstance(window,(int,float)) else 1)+interval_slots
            reason=('missing' if value is None or produced is None or delivered is None else
                    'future' if produced>now or delivered>now else
                    'stale' if age>max_age else
                    'no_samples' if samples is None or samples<=0 else None)
            if isinstance(value,float) and not math.isfinite(value):reason='nonfinite'
            valid=reason is None
            setattr(v,name,deepcopy(value) if valid else None)
            record=dict(name=name,entity='ue',entity_id=u.ue_id,unit=spec.unit,source=spec.source,
                window=dict(kind=window_kind,size=None if window_kind=='cumulative' else window),
                window_slots=window if window_kind=='slots' else None,
                window_tbs=window if window_kind=='transport_blocks' else None,
                produced_at_slot=produced,delivered_at_slot=delivered,
                sample_count=samples,valid=valid,reason=reason,age_slots=age)
            v.kpi_metadata[name]=record;result.kpi_records.append(record)
            if name in contract.required_kpis and not valid:errors.append(f'{u.ue_id}:{name}:{reason}')
        # Compatibility aliases have explicit measured sources, never DU truth.
        v.sinr_db=v.reference_sinr_db
        if all(getattr(v,k) is not None for k in ('rsrp_dbm','reference_sinr_db','reference_noise_dbm')):
            v.inter_cell_interference_penalty_db=max(0.,v.rsrp_dbm-v.reference_sinr_db-v.reference_noise_dbm)
        else:v.inter_cell_interference_penalty_db=None
        v.demand_harq_urgency_score=v.retx_fraction
        v.kpi_metadata['inter_cell_interference_penalty_db']=dict(
            source='derived_delivered_RSRP_minus_reference_SINR_minus_receiver_noise',unit='dB',
            window_slots=getattr(u,'sinr_window_slots',None),produced_at_slot=getattr(u,'measurement_produced_at_slot',None),
            delivered_at_slot=getattr(u,'measurement_delivered_at_slot',None),
            sample_count=getattr(u,'sinr_effective_sample_count',0),valid=v.inter_cell_interference_penalty_db is not None)


        v.demand_bytes=v.demand_observed_backlog_bytes=v.queue_bytes
        v.demand_hol_delay_ms=v.hol_delay_ms
        v.post_eq_sinr_db=v.data_sinr_db if v.data_sinr_db is not None else v.reference_sinr_db
        v.candidate_cell_observations=[]
        for candidate in getattr(u,'candidate_cell_observations',[]):
            c=SimpleNamespace(**_copy(candidate,('cell_id','rsrp_dbm','candidate_beam_id','is_serving','is_target',
                'measurement_produced_at_slot','measurement_delivered_at_slot','measurement_window_slots')))
            produced=getattr(c,'measurement_produced_at_slot',None)
            delivered=getattr(c,'measurement_delivered_at_slot',None)
            valid=produced is not None and delivered is not None and produced<=delivered<=now
            age=None if produced is None else now-produced
            limit=contract.max_age_slots if contract.max_age_slots is not None else (getattr(c,'measurement_window_slots',1) or 1)+interval_slots+(getattr(observation,'measurement_queue_delay_slots',0) or 0)
            valid=valid and age<=limit
            if not valid:c.rsrp_dbm=None;c.candidate_beam_id=None
            c.kpi_metadata=dict(source='RT_candidate_reference_filtered_noisy_delayed',unit='dBm',
                produced_at_slot=produced,delivered_at_slot=delivered,sample_count=1 if valid else 0,
                window_slots=getattr(c,'measurement_window_slots',None),valid=valid,age_slots=age)
            c.steering_score_db=c.rsrp_dbm
            v.candidate_cell_observations.append(c)
        result.ue_observations.append(v)
    for cell in observation.cell_observations:
        if contract.cell_ids and cell.cell_id not in contract.cell_ids:continue
        c=SimpleNamespace(**_copy(cell,CELL_CONFIG))
        members=[u for u in result.ue_observations if u.serving_cell_id==cell.cell_id]
        c.total_queue_bytes=sum(u.queue_bytes for u in members) if all(u.queue_bytes is not None for u in members) else None
        values=[u.inter_cell_interference_penalty_db for u in members if u.inter_cell_interference_penalty_db is not None]
        c.inter_cell_interference_penalty_db_mean=sum(values)/len(values) if values else None
        c.kpi_metadata={}
        for name,spec in KPI_REGISTRY.items():
            if spec.entity!='cell':continue
            value=getattr(c,name,None)
            produced=getattr(c,'rf_power_produced_at_slot',None) if name=='rf_tx_power_w' else observation.slot_id
            max_age=contract.max_age_slots if contract.max_age_slots is not None else (getattr(observation,'report_delay_slots',0) or 0)+interval_slots+1
            reason=('missing' if value is None or produced is None else 'future' if produced>now else 'stale' if now-produced>max_age else None)
            record=dict(name=name,entity='cell',entity_id=cell.cell_id,unit=spec.unit,source=spec.source,
                window=dict(kind='slots',size=1),window_tbs=None,window_slots=1,produced_at_slot=produced,delivered_at_slot=now,sample_count=1 if reason is None else 0,
                valid=reason is None,reason=reason,age_slots=now-produced if produced is not None else None)
            c.kpi_metadata[name]=record;result.kpi_records.append(record)
            if name in contract.required_kpis and reason:errors.append(f'{cell.cell_id}:{name}:{reason}')
        result.cell_observations.append(c)
    ues=result.ue_observations
    result.total_queue_bytes=sum(u.queue_bytes for u in ues) if all(u.queue_bytes is not None for u in ues) else None
    result.total_throughput_mbps=sum(u.throughput_mbps for u in ues) if all(u.throughput_mbps is not None for u in ues) else None
    valid_sinr=[u.sinr_db for u in ues if u.sinr_db is not None]
    valid_bler=[u.bler for u in ues if u.bler is not None]
    result.avg_sinr_db=result.avg_sinr_db_all=sum(valid_sinr)/len(valid_sinr) if valid_sinr else None
    result.avg_bler=sum(valid_bler)/len(valid_bler) if valid_bler else None
    result.connected_ue_count=len(ues)
    result.scheduled_ue_ids=deepcopy(getattr(observation,'scheduled_ue_ids',[]))
    result.num_scheduled_ues=len(result.scheduled_ue_ids)
    result.xapp_capabilities={'writable_actions':sorted(contract.writable_actions),
        'unavailable_actions':sorted(contract.actions-SUPPORTED_ACTIONS)}
    if not ues and any(KPI_REGISTRY.get(k,KpiSpec('','','','')).entity=='ue' for k in contract.required_kpis):
        errors.append('no_ue_measurement_entities')
    return result,errors


def validate_action(action,contract,observation):
    problems=[]
    for f in fields(action):
        value=getattr(action,f.name)
        if f.name in ADMIN or value is None or value=={}:continue
        if isinstance(value,float) and not math.isfinite(value):problems.append(f'{f.name}:nonfinite')
        if isinstance(value,dict) and any(isinstance(v,(int,float)) and not math.isfinite(v) for v in value.values()):problems.append(f'{f.name}:nonfinite')
        if f.name not in contract.actions:problems.append(f'{f.name}:undeclared_action')
        elif f.name not in SUPPORTED_ACTIONS:problems.append(f'{f.name}:unsupported_rt_action')
    cells={c.cell_id for c in observation.cell_observations}
    target=action.target_cell_id
    if target is not None and (target not in cells or (contract.cell_ids and target not in contract.cell_ids)):
        problems.append('invalid_target_cell')
    if action.apply_scope not in ('cluster','cell'):
        problems.append('unsupported_scope')
    if contract.cell_ids and action.apply_scope=='cluster' and not cells.issubset(set(contract.cell_ids)):
        problems.append('cluster_scope_exceeds_contract')
    if contract.cell_ids and action.apply_scope=='cell' and target is None and observation.cell_id not in contract.cell_ids:
        problems.append('default_primary_cell_exceeds_contract')
    if action.effective_from_slot is not None and action.effective_until_slot is not None and action.effective_from_slot>action.effective_until_slot:
        problems.append('invalid_effective_window')
    return problems


def capability_record(name,actions):
    c=contract_for(actions,name)
    return dict(xapp_name=name,required_kpis=list(c.required_kpis),entities=list(c.entities),
        writable_actions=sorted(c.writable_actions),unavailable_actions=sorted(c.actions-SUPPORTED_ACTIONS),
        available=c.available,missing_kpi_sources=sorted(k for k in c.required_kpis if k not in KPI_REGISTRY),max_age_slots='configured_report_delay_plus_window_plus_xapp_interval')
