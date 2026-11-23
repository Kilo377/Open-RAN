"""Population-level presentation, separate from released xApp target contracts.

Metric keys remain stable for exports; the explicit scope version prevents
legacy target-subset measurements from being relabelled as population totals.
"""
from copy import deepcopy

SCOPE = 'ue_population.v1'

PARAMETER_LABELS = {
    'ue_offload_fraction': 'MEC offload probability',
    'ue_drx_configs': 'UE DRX timers',
    'scheduler_mode': 'Downlink scheduling policy',
    'ue_scheduler_weights': 'UE scheduling priority',
    'cell_sleep_mode': 'BS sleep mode',
}

KPI_OVERRIDES = {
    'ue_power_w': ('Total UE power', 'All UEs', 'sum of per-UE modeled radio + CPU energy counter deltas / elapsed seconds over the same observation window; total mean power, not cumulative energy'),
    'ue_energy_j': ('Cumulative UE energy (history)', 'All UEs', 'sum of modeled radio + CPU energy across all UEs; cumulative historical counter, not an instantaneous intent target'),
    'mec_offloaded_latency_ms': ('Mean MEC task delay', 'MEC workload', 'sum of completion latencies / completed offloaded tasks across the workload'),
    'urllc_failure_events_per_s': ('URLLC failure event rate', 'URLLC traffic', 'sum of deadline failures and disjoint overflows / interval; events/s, includes carryover; distinct from packet error percentage'),
    'urllc_failure_fraction': ('URLLC packet error rate', 'URLLC traffic', 'total deadline failures and disjoint overflows / total offered packets over the stated window; canonical fraction displayed as percent; includes carryover, not mature-cohort reliability probability'),
    'urllc_packet_latency_ms': ('Mean URLLC packet delay', 'URLLC traffic', 'total completion latency / total completed URLLC packets in the interval'),
    'mec_expected_gflops': ('Expected MEC compute load', 'MEC workload', 'sum of configured task rate × compute per task × executed offload probability; total offered compute demand, not server utilization or energy'),
    'task_latency_ms': ('Mean task E2E latency', 'MEC workload', 'total completion latency / completed tasks across the workload'),
    'task_success_ratio': ('On-time task completion', 'MEC workload', 'total on-time completions / total generated tasks; pending included'),
    'per_ue_goodput_mbps': ('Mean UE Downlink throughput', 'All UEs', 'network delivered user-data throughput / number of UEs; includes zero-goodput UEs'),
    'packet_latency_ms': ('Mean Downlink packet delay', 'All UEs', 'total completion latency / total completed packets in the interval'),
    'ue_active_fraction': ('Mean UE active-time ratio', 'All UEs', 'mean active-time fraction across all UEs'),
    'mec_realized_gflops': ('Observed MEC compute load', 'MEC workload', 'sum of generated offloaded task counter deltas × compute per task / interval; total offered compute demand, not server utilization or energy'),
}


def graph_view(graph):
    result = deepcopy(graph)
    result['metric_scope'] = SCOPE
    for n in result['nodes']:
        if n['kind'] == 'xapp':
            n['monitoring_scope'] = 'population'
            n['objective_scope'] = 'released_contract'
        elif n['kind'] == 'rcp':
            n['label'] = PARAMETER_LABELS[n['key']]
            n['scope'] = 'UE population' if n['scope'] == 'UE' else 'All cells'
        elif n['kind'] == 'kpi' and n['key'] in KPI_OVERRIDES:
            n['label'], n['scope'], n['aggregation'] = KPI_OVERRIDES[n['key']]
    for e in result['edges']:
        if e['kind'] == 'influences':
            e['note'] = 'Population-level KPI response. A common change across the relevant group is required to identify a population control response.'
    result['notes'].append('Population monitoring aggregates all applicable members. Released xApp targeting and trained policies are unchanged.')
    return result


def task_members(config, services=None):
    profiles = config.get('configs', {}).get('phy', {}).get('rt_sys', {}).get('ue_services', {}).get('mec', {}).get('tasks_by_ue')
    if profiles is not None:
        return list(profiles)
    return [uid for uid, row in (services or {}).items() if row.get('ue_task_metrics') is not None]


def urllc_members(config):
    # Membership follows the configured service class, rather than fixed UE IDs.
    return [u['ue_id'] for u in config.get('configs', {}).get('ue', {}).get('ues', [])
            if u.get('slice_id') == 'urllc']
