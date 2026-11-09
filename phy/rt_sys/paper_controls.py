"""Opt-in, transaction-validated controls for the paper environment."""
from copy import deepcopy
import math
import numpy as np

FIELDS={'prb_power_w','clear_prb_power','power_group_size_prbs','ue_prb_limits','ue_scheduler_weights',
        'neighbor_cell_ids','cio_offsets_db','ret_enabled','ret_tilt_deg','ret_codebook_id','time_to_trigger_ms'}
COMMANDS={'request_handover','disconnect','reconnect'}
DEFAULTS=dict(prb_power_w=None,power_group_size_prbs=1,ue_prb_limits={},neighbor_cell_ids=None,
              cio_offsets_db={},ret_enabled=False,ret_tilt_deg=0.,ret_codebook_id=0,time_to_trigger_ms=40.)


def proposed_power_budget(cell,action):
    """Mirror LocalExecutor's cap -> sleep -> explicit-power order without mutation.

    The opt-in backend also enforces a persistent cap after sleep transitions.
    Validation must use this resulting budget, rather than the pre-action value.
    """
    tx=float(cell.tx_power_dbm);cap=cell.power_cap_dbm
    mode=str(cell.cell_sleep_mode);nominal=float(getattr(cell,'nominal_tx_power_dbm',tx))
    if action.power_cap_dbm is not None:
        cap=float(action.power_cap_dbm);tx=cap
        if mode=='active' or getattr(cell,'ap_service_enabled',False):nominal=tx
    if action.cell_sleep_mode is not None:
        requested=str(action.cell_sleep_mode).strip().lower()
        if requested:
            if requested not in {'active','micro_sleep','light_sleep','deep_sleep'}:requested='active'
            if mode=='active':nominal=tx
            mode=requested
            if getattr(cell,'ap_service_enabled',False):
                tx=min(nominal,cap) if cap is not None else nominal
            elif mode=='active':
                if cap is not None and cap<=0:cap=None
                tx=min(nominal,cap) if cap is not None else nominal
            elif mode=='micro_sleep':tx=min(nominal,max(12.,nominal-3.))
            elif mode=='light_sleep':tx=min(nominal,max(6.,nominal-8.))
            else:tx=0.
    if action.tx_power_dbm is not None:tx=float(action.tx_power_dbm)
    if cap is not None:tx=min(tx,float(cap))
    return 0. if mode=='deep_sleep' and not getattr(cell,'ap_service_enabled',False) else 10**((tx-30)/10)


def requested(action):
    return {k:deepcopy(getattr(action,k)) for k in FIELDS if getattr(action,k,None) is not None}


def scheduler_action_mode(action):
    """Mirror the executor's explicit-mode-last precedence for validation."""
    explicit=getattr(action,'scheduler_mode',None)
    if explicit is not None and str(explicit).strip():return str(explicit).strip()
    hint=str(getattr(action,'scheduler_policy_hint',None) or '').strip().lower()
    return {'balanced':'round_robin','latency_bias':'latency_aware','slice_bias':'slice_aware'}.get(hint)


def validate_transaction(state,action,targets):
    if not getattr(state,'paper',None):
        from phy.rt_sys.ue_services import ACTION_FIELDS
        return 'paper_interfaces_disabled' if requested(action) or any(getattr(action,k,None) is not None for k in ACTION_FIELDS) else None
    cells=state.managed_cells;uids={u.ue_id for u in state.ues}
    try:
        ap_services=getattr(state,'ap_services',None)
        if ap_services is not None and action.cell_sleep_mode is not None:
            ap_services.validate_policy(action.cell_sleep_mode)
        weights=getattr(action,'ue_scheduler_weights',None)
        if weights is not None:
            if not isinstance(weights,dict):raise ValueError('invalid_ue_scheduler_weights')
            for uid,weight in weights.items():
                if uid not in uids:raise ValueError('unknown_ue_scheduler_weight_target')
                if uid not in state.connections or state.connections[uid].serving_cell_id not in targets:
                    raise ValueError('ue_scheduler_weight_target_outside_cell_scope')
                cid=state.connections[uid].serving_cell_id
                effective_mode=scheduler_action_mode(action) or getattr(cells[cid].scheduler,'scheduler_mode','round_robin')
                max_weight=4. if effective_mode=='qos_deadline' else 2.
                if type(weight) not in (int,float) or not math.isfinite(weight) or not .5<=weight<=max_weight:
                    raise ValueError('invalid_ue_scheduler_weight')
        # A mode-only switch cannot carry the opt-in range into a legacy
        # latency consumer that deliberately retains its historical <=2 bound.
        # Full mode+UID-map restoration replaces the weights atomically.
        if scheduler_action_mode(action) not in (None,'qos_deadline'):
            for ue in state.ues:
                if ue.ue_id in state.connections and state.connections[ue.ue_id].serving_cell_id in targets:
                    effective_weight=(weights or {}).get(ue.ue_id,getattr(ue,'scheduler_weight',1.))
                    if type(effective_weight) not in (int,float) or not math.isfinite(effective_weight) or not .5<=effective_weight<=2.:
                        raise ValueError('legacy_scheduler_requires_weight_reset')
        services=getattr(state,'ue_services',None)
        from phy.rt_sys.ue_services import ACTION_FIELDS
        if any(getattr(action,k,None) is not None for k in ACTION_FIELDS):
            if services is None:raise ValueError('ue_services_disabled')
            services.validate_action(action,targets)
        for cid in targets:
            c=cells[cid].cell;n=cells[cid].resource_grid.n_prbs_total
            values={k:deepcopy(getattr(c,k,v)) for k,v in DEFAULTS.items()}
            values.update({k:v for k,v in requested(action).items() if k!='ue_scheduler_weights'})
            if values.pop('clear_prb_power',False):
                if action.prb_power_w is not None:raise ValueError('clear_and_set_power')
                values['prb_power_w']=None
            group=values['power_group_size_prbs']
            if type(group) is not int or group<1:raise ValueError('invalid_power_group_size')
            if action.prb_power_w is not None:
                a=np.asarray(action.prb_power_w,float)
                if a.shape!=(math.ceil(n/group),):raise ValueError('power_vector_length')
                values['prb_power_w']=[float(a[i//group])/min(group,n-(i//group)*group) for i in range(n)]
            elif action.power_group_size_prbs is not None and getattr(c,'prb_power_w',None) is not None and not action.clear_prb_power:
                raise ValueError('group_change_requires_power_vector')
            if not isinstance(values['ret_enabled'],bool):raise ValueError('invalid_ret_enabled')
            for val in (action.tx_power_dbm,action.power_cap_dbm):
                if val is not None and not np.isfinite(val):raise ValueError('invalid_power_budget')
            if values['ret_enabled']:
                if any(getattr(action,k) is not None for k in ('codebook_id','beam_profile_id','beam_candidate_preference')):raise ValueError('ret_requires_horizontal_codebook_controls')
                ports=state.paper.adapter.config['tx_cols'] if values['ret_codebook_id']==0 else 1
                if any(p.tx_contract is not None and p.tx_contract.layers>ports for u in state.ues if state.connections[u.ue_id].serving_cell_id==cid for p in u.harq.processes):raise ValueError('ret_conflicts_with_active_harq_layers')
            v=values['prb_power_w']
            if v is not None:
                a=np.asarray(v,float)
                if a.shape!=(n,) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('invalid_prb_power')
                budget=proposed_power_budget(c,action)
                if not np.isfinite(budget):raise ValueError('invalid_power_budget')
                if a.sum()>budget+1e-9:raise ValueError('prb_power_exceeds_cell_budget')
            limits=values['ue_prb_limits']
            if not isinstance(limits,dict) or any(u not in uids or type(v) is not int or not 0<=v<=n for u,v in limits.items()):raise ValueError('invalid_ue_prb_limit')
            if (limits or values['prb_power_w'] is not None) and cells[cid].scheduler.packing_strategy!='frequency_first':raise ValueError('prb_limits_require_frequency_first')
            nl=values['neighbor_cell_ids']
            if nl is not None and (not isinstance(nl,list) or len(nl)!=len(set(nl)) or any(x not in cells or x==cid for x in nl)):raise ValueError('invalid_neighbor_list')
            cio=values['cio_offsets_db']
            if not isinstance(cio,dict) or any(x not in cells or x==cid or not np.isfinite(v) for x,v in cio.items()):raise ValueError('invalid_cio_offsets')
            if not np.isfinite(values['time_to_trigger_ms']) or values['time_to_trigger_ms']<0:raise ValueError('invalid_ttt')
            if not np.isfinite(values['ret_tilt_deg']) or not 0<=values['ret_tilt_deg']<=15 or values['ret_codebook_id'] not in (0,1):raise ValueError('invalid_ret')
    except (ValueError,TypeError,OverflowError) as error:return str(error)
    return None


def apply_values(state,action,targets):
    services=getattr(state,'ue_services',None)
    if services is not None:services.apply_action(action)
    # UID policy follows its UE across handover; it is not a cell attribute.
    users={u.ue_id:u for u in state.ues}
    for uid,weight in (getattr(action,'ue_scheduler_weights',None) or {}).items():
        users[uid].scheduler_weight=float(weight)
    for cid in targets:
        c=state.managed_cells[cid].cell;n=state.managed_cells[cid].resource_grid.n_prbs_total
        if c.power_cap_dbm is not None:c.tx_power_dbm=min(c.tx_power_dbm,c.power_cap_dbm)
        changed=False
        for key,value in requested(action).items():
            if key=='ue_scheduler_weights':continue
            old=deepcopy(getattr(c,key,None))
            if key=='clear_prb_power':
                if value:c.prb_power_w=None
            elif key=='prb_power_w':
                group=action.power_group_size_prbs or c.power_group_size_prbs
                c.prb_power_w=[float(value[i//group])/min(group,n-(i//group)*group) for i in range(n)]
            else:setattr(c,key,deepcopy(value))
            changed|=key in {'time_to_trigger_ms','cio_offsets_db','neighbor_cell_ids'} and old!=value
        changed|=bool(getattr(action,'_paper_hysteresis_changed',False))
        if changed:state.paper.reset_triggers(cid)
