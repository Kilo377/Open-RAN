"""Public executed configuration and bounded event reports, never CSI tensors."""
from copy import deepcopy
from phy.rt_sys.contracts import digest

CONFIG_FIELDS=('tx_power_dbm','power_cap_dbm','nominal_tx_power_dbm','cell_sleep_mode',
 'current_codebook_id','current_beam_profile_id','prb_power_w','power_group_size_prbs',
 'ue_prb_limits','cio_offsets_db','neighbor_cell_ids','time_to_trigger_ms','steering_hysteresis_db',
 'ret_enabled','ret_tilt_deg','ret_codebook_id')


def executed_configs(state):
    result={}
    for cid,r in state.managed_cells.items():
        values={k:deepcopy(getattr(r.cell,k,None)) for k in CONFIG_FIELDS}
        values['codebook_id']=values.pop('current_codebook_id');values['beam_profile_id']=values.pop('current_beam_profile_id')
        values.update(n_prbs=r.resource_grid.n_prbs_total,
            scheduler_mode=r.scheduler.scheduler_mode,scheduler_policy_hint=r.scheduler.scheduler_policy_hint,latency_bias=r.scheduler.latency_bias,
            slice_weight_adjustment=deepcopy(r.scheduler.slice_weight_adjustment))
        result[cid]=values
    return result


def augment(paper,reports):
    cfg=executed_configs(paper.state);slot=paper.state.slot_id
    for cid,values in cfg.items():
        reports['cell'][cid].update(executed_config=values,config_fingerprint=digest(values),config_produced_at_slot=slot)
    for u in paper.state.ues:
        uid=u.ue_id;seq=paper.ue_samples[uid];dt=paper.dt/1000
        reports['ue'][uid].update(
            delivered_bits_window=sum(x[0] for x in seq)*8,
            arrival_bits_window=sum(x[1] for x in seq)*8,
            deadline_misses_window=sum(x[2] for x in seq),
            deadline_packets_window=sum(x[3] for x in seq),
            business_window_s=len(seq)*dt,
            mobility_events=[deepcopy(e) for e in paper.all_events if e['ue_id']==uid and
                (slot-e['slot_id'])*paper.dt<=1000] if paper.events_enabled else None,
            beam_reference_reports=deepcopy(paper.observed[uid][paper.state.connections[uid].serving_cell_id].get('beam_reference_reports')))
