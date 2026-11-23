"""Small read-only execution snapshot. No PHY, model or learner imports."""
from copy import deepcopy
import math


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def compact_telemetry(state, trace):
    """Called only by the existing live publisher, at its normal publication rate.

    These are executed simulator counters/configurations, not delayed E2
    measurements. The graph labels this provenance explicitly. Large task-cohort
    lists and physical-layer tensors never enter the live graph snapshot.
    """
    services = {}
    for uid, row in trace.get('ue_services', {}).items():
        services[uid] = {k: deepcopy(row[k]) for k in (
            'ue_energy_total_j', 'ue_active_fraction', 'ue_task_metrics', 'ue_service_policy') if k in row}
    ap = {cid: {k: row[k] for k in ('ap_energy_total_j', 'ap_power_w',
        'ap_available_fraction', 'ap_policy') if k in row}
        for cid, row in trace.get('ap_services', {}).get('cells', {}).items()}
    return finite_json(dict(schema='oran.graph-telemetry.v1',
        source='executed_simulator_state', sampled_at_slot=state.slot_id,
        ue_services=services, ap_services=ap,
        ue_scheduler_weights={u.ue_id: getattr(u, 'scheduler_weight', None) for u in state.ues}))
