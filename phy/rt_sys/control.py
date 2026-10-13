"""Capabilities are explicit: a field must have a runtime consumer."""
from dataclasses import fields
from models.action import ConfigAction

PHY = {'codebook_id','tx_power_dbm','beam_profile_id','beam_candidate_preference','power_cap_dbm','cell_sleep_mode'}
SCHEDULER = {'scheduler_mode','scheduler_policy_hint','latency_bias','slice_weight_adjustment'}
ASSOCIATION = {'traffic_steering_preference','mobility_bias_db','handover_cost_db','handover_interruption_slots','steering_hysteresis_db','measurement_event_trigger','measurement_event_offset_db'}
ADMIN = {'apply_scope','target_cell_id','control_request_id','effective_from_slot','effective_until_slot','source_name','metadata'}


def audit_action(action):
    trace = []
    for f in fields(ConfigAction):
        value = getattr(action, f.name)
        if f.name in ADMIN or value is None or value == {}:
            continue
        route = ('phy' if f.name in PHY else 'scheduler' if f.name in SCHEDULER
                 else 'association' if f.name in ASSOCIATION else 'unsupported')
        trace.append({'field':f.name, 'value':value, 'route':route,
                      'reason':None if route != 'unsupported' else 'No RT/SYS consumer in v1'})
    return trace


def limit_logical_allocations(allocations, budget):
    """Match pending HARQ/LC accounting to the actual transmitted TB payload."""
    from copy import deepcopy
    limited=[]
    remaining=max(0,int(budget))
    for allocation in allocations:
        item=deepcopy(allocation)
        item.scheduled_bytes=min(remaining,max(0,int(item.scheduled_bytes)))
        remaining-=item.scheduled_bytes
        item.prioritized_bytes=min(item.scheduled_bytes,max(0,int(item.prioritized_bytes)))
        item.best_effort_bytes=item.scheduled_bytes-item.prioritized_bytes
        item.used_pbr_credit_bytes=min(item.prioritized_bytes,max(0,int(item.used_pbr_credit_bytes)))
        limited.append(item)
    return limited


def runtime_control_values(state,cell_ids):
    """Record the actual scoped consumers, rather than only the requested fields."""
    result={}
    for cid in cell_ids:
        runtime=state.managed_cells[cid]
        c,s=runtime.cell,runtime.scheduler
        result[cid]={name:getattr(c,name,None) for name in ('tx_power_dbm','current_codebook_id',
            'current_beam_profile_id','power_cap_dbm','cell_sleep_mode','traffic_steering_preference',
            'mobility_bias_db','handover_cost_db','handover_interruption_slots','steering_hysteresis_db',
            'measurement_event_trigger','measurement_event_offset_db')}
        result[cid].update({name:getattr(s,name,None) for name in ('scheduler_mode','scheduler_policy_hint',
            'latency_bias','slice_weight_adjustment')})
    return result
