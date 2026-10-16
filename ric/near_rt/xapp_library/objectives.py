"""Objective identities describe intent, never ground-truth conflict labels."""
from copy import deepcopy

# category, algorithm, objective, direction, evaluation KPIs, additional required KPIs
OBJECTIVES = [
 ('throughput_power','sum_service_power','sum_goodput','maximize',['throughput_mbps'],[]),
 ('throughput_power','weak_user_power','lower_tail_goodput','maximize',['throughput_mbps'],[]),
 ('rf_energy','rf_efficiency_search','rf_efficiency','maximize',['rf_efficiency_bit_per_j'],['rf_energy_total_j','delivered_bits_total','requested_dl_rate_bps']),
 ('rf_energy','sla_admission_power','ue_sla_count','maximize',['sla_status'],['requested_dl_rate_bps']),
 ('ue_fairness','sum_service_caps','sum_goodput','maximize',['throughput_mbps'],['subband_cqi','subband_prb_groups','executed_prb_slots_total']),
 ('ue_fairness','proportional_utility_caps','log_goodput_utility','maximize',['goodput_utility'],['subband_cqi','subband_prb_groups']),
 ('ue_fairness','starvation_guard_caps','resource_starvation_duration','minimize',['executed_prb_slots_total'],['executed_prb_slots_total','resource_counter_epoch']),
 ('slice_sla','sla_count_weights','slice_sla_count','maximize',['sla_status'],['requested_dl_rate_bps']),
 ('slice_sla','worst_slice_weights','worst_normalized_slice_gap','minimize',['throughput_mbps','requested_dl_rate_bps'],['requested_dl_rate_bps']),
 ('slice_sla','deadline_loss_weights','deadline_losses','minimize',['deadline_misses_window','deadline_packets_window'],['deadline_misses_window','deadline_packets_window']),
 ('traffic_steering','coverage_margin_steering','reference_coverage_margin','maximize',['rsrp_dbm','reference_sinr_db'],['rsrp_dbm','reference_sinr_db']),
 ('traffic_steering','continuity_first_steering','service_interruptions','minimize',['mobility_events'],['rsrp_dbm','reference_sinr_db','mobility_events']),
 ('mobility_robustness','pingpong_guard','pingpong_events','minimize',['mobility_events'],[]),
 ('mobility_robustness','failure_guard','terminal_mobility_failures','minimize',['mobility_events'],[]),
 ('ret_coverage','coverage_fraction_grid','reference_coverage_fraction','maximize',['rsrp_dbm','reference_sinr_db'],[]),
 ('ret_coverage','sum_goodput_grid','sum_goodput','maximize',['delivered_bits_total'],['delivered_bits_total','requested_dl_rate_bps']),
 ('beam_management','sum_goodput_beam','sum_goodput','maximize',['delivered_bits_total'],['delivered_bits_total','requested_dl_rate_bps']),
 ('beam_management','sla_count_beam','ue_sla_count','maximize',['sla_status'],['delivered_bits_total','requested_dl_rate_bps']),
]
EXISTING_METRICS={
 'throughput_power':('maximize',['throughput_mbps','goodput_utility']),
 'rf_energy':('minimize',['rf_energy_j','sla_status']),
 'cell_sleep':('minimize',['rf_energy_j','mobility_events']),
 'load_balancing':('minimize',['resource_grid_utilization']),
 'traffic_steering':('maximize',['reference_sinr_db','throughput_mbps']),
 'mobility_robustness':('minimize',['mobility_events']),
 'ue_fairness':('maximize',['throughput_mbps','executed_prb_slots_total']),
 'slice_sla':('maximize',['sla_status']),
 'latency_guard':('minimize',['hol_delay_ms','deadline_misses_window']),
 'interference_coordination':('maximize',['subband_cqi']),
 'ret_coverage':('maximize',['reference_sinr_db']),
 'beam_management':('maximize',['beam_reference_reports']),
}
OBJECTIVE_ALGORITHMS={r[1] for r in OBJECTIVES}
SEARCH_ALGORITHMS={'rf_efficiency_search','coverage_fraction_grid','sum_goodput_grid','sum_goodput_beam','sla_count_beam'}
OBJECTIVE_DEFAULTS=dict(objective_window_ms=400.,service_floor_bps=1000.,starvation_ms=20.,continuity_bad_ms=100.,
    concentrated_power_fraction=.8,top_group_fraction=.25,weak_user_fraction=.2,pingpong_ms=1000.)


def objective_metadata(ident,algorithm,goal):
    row=next((r for r in OBJECTIVES if r[1]==algorithm),None)
    return dict(objective_id=row[2] if row else 'existing:'+ident.split(':')[0],
        objective_description=goal,objective_direction=row[3] if row else EXISTING_METRICS[ident.split(':')[0]][0],
        evaluation_kpis=deepcopy(row[4]) if row else deepcopy(EXISTING_METRICS[ident.split(':')[0]][1]),
        decision_proxy='delivered-feedback heuristic; CQI scores are rankings, not predicted capacity',
        evaluation_scope='serving cell and its delivered UE cohort',evaluation_window='delivered reporting window; executed search >=400ms',
        protection_conditions=['public KPI validity/age','physical budgets','existing mobility/PHY contracts'],
        evidence_status='requires_version_specific_evidence')
