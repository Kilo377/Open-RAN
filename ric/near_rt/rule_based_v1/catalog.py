"""Versioned functional baselines, not reproductions of trained source policies."""
SOURCES = {
 'srsran':'https://github.com/srsran/oran-sc-ric',
 'oranslice':'https://github.com/wineslab/ORANSlice',
 'nexran':'https://par.nsf.gov/servlets/purl/10320274',
 'orange_sleep':'https://github.com/Orange-OpenSource/ns-O-RAN-flexric/blob/main/docs/xapp-energy-saving.md',
 'orange_rf':'https://github.com/Orange-OpenSource/ns-O-RAN-flexric/blob/main/docs/xapp-rf-reconfiguration.md',
 'multimetric':'https://github.com/alevic44/xapp-multimetric-ho',
 'osc_ts':'https://docs.o-ran-sc.org/projects/o-ran-sc-ric-app-ts/en/latest/user-guide.html',
 'colosseum':'https://github.com/genesys-neu/oran-resource-opt',
 'spectrum':'https://arxiv.org/abs/2601.13769',
 'pacifista':'https://arxiv.org/abs/2405.04395',
 'comix':'2501.14619v1.pdf', 'distillation':'1-s2.0-S138912862500814X-main.pdf',
 'conflict_graph':'2412.14119v2.pdf', 'conflict_ai':'1-s2.0-S1389128626004445-main.pdf',
}
# KPI names are measured inputs, not desired outcomes or imagined predictions.
CATALOG = {
 'throughput_power':(30,20,['subband_cqi','subband_prb_groups','queue_bytes','throughput_mbps'],['prb_power_w','power_group_size_prbs'],[],['comix'],'Demand/fairness-weighted frequency power'),
 'rf_energy':(20,20,['reference_sinr_db','queue_bytes','hol_delay_ms','throughput_mbps'],['tx_power_dbm'],[],['comix','orange_rf'],'RF-only power saving with QoS guards'),
 'cell_sleep':(20,100,['queue_bytes','candidate_rsrp_dbm','candidate_sinr_db','resource_grid_utilization'],['cell_sleep_mode'],['request_handover'],['orange_sleep'],'Migrate then sleep; coverage-safe wake'),
 'load_balancing':(50,20,['candidate_rsrp_dbm','candidate_sinr_db','resource_grid_utilization','queue_bytes'],['cio_offsets_db'],[],['conflict_ai'],'Directed CIO load balancing'),
 'traffic_steering':(80,20,['candidate_rsrp_dbm','candidate_sinr_db','throughput_mbps','hol_delay_ms','resource_grid_utilization'],[],['request_handover'],['osc_ts','multimetric','distillation'],'Measured multi-metric handover'),
 'mobility_robustness':(90,200,['mobility_events','reference_sinr_db'],['time_to_trigger_ms','steering_hysteresis_db'],[],['conflict_ai'],'Event-based hysteresis/TTT feedback'),
 'ue_fairness':(40,20,['queue_bytes','offered_rate_bps','throughput_mbps','resource_allocation'],['ue_prb_limits'],[],['srsran','distillation'],'Demand-limited weighted max-min PRB caps'),
 'slice_sla':(70,20,['queue_bytes','hol_delay_ms','throughput_mbps','offered_rate_bps'],['slice_weight_adjustment','scheduler_mode'],[],['nexran','oranslice'],'Soft slice weight feedback, not hard isolation'),
 'latency_guard':(100,20,['queue_bytes','hol_delay_ms','throughput_mbps'],['scheduler_mode','latency_bias','slice_weight_adjustment'],[],['oranslice','colosseum'],'HOL/deadline-aware DU policy guidance'),
 'interference_coordination':(35,20,['subband_cqi','candidate_rsrp_dbm','queue_bytes','resource_allocation'],['prb_power_w','power_group_size_prbs'],[],['spectrum','comix'],'Measured graph coloring, RF power shaping'),
 'ret_coverage':(60,200,['reference_sinr_db','rsrp_dbm','throughput_mbps','association'],['ret_enabled','ret_codebook_id','ret_tilt_deg'],[],['conflict_ai'],'Live-window RET search with attribution checks'),
 'beam_management':(60,20,['beam_reference_reports','queue_bytes','throughput_mbps'],['codebook_id','beam_profile_id'],[],['pacifista'],'Measured legal-codebook selection'),
}
PROFILES = {
 'power_energy':['throughput_power','rf_energy','interference_coordination'],
 'business_resources':['ue_fairness','slice_sla','latency_guard'],
 'mobility':['load_balancing','traffic_steering','mobility_robustness'],
 'coverage_sleep':['cell_sleep','ret_coverage','beam_management','rf_energy'],
}
DEFAULTS = dict(ewma_ms=100.,power_step_db=.5,sinr_target_db=5.,sinr_deadband_db=1.,
 min_power_dbm=10.,max_power_dbm=30.,cio_step_db=.5,cio_bound_db=6.,load_gap=.2,
 target_max_load=.8,good_rsrp_dbm=-107.,good_sinr_db=-3.,sleep_low_load=.1,
 wake_high_load=.7,sleep_low_ms=200.,sleep_dwell_ms=200.,command_cooldown_ms=100.,
 ts_samples=3,ts_margin=.05,mro_window_ms=1000.,mro_min_attempts=5,
 pingpong_ms=100.,ret_step_deg=1.,ret_gain_db=.1,ret_min_deg=0.,ret_max_deg=15.,
 beam_gain_db=.5,slice_min_weight=.25,slice_max_weight=4.,protected_outer_fraction=.25,
 interference_edge_rsrp_dbm=-105.,minimum_samples=1,max_age_ms=100.)


def configuration(kinds=(),enabled=True):
    return dict(enabled=enabled,monitor=True,applications=list(kinds),parameters={},cell_ids=[])


def export_catalog():
    return {k:dict(priority=s[0],min_action_ms=s[1],required_kpis=s[2],actions=s[3],commands=s[4],
        sources=[SOURCES[x] for x in s[5]],goal=s[6],adaptation='Local deterministic functional baseline; not source algorithm reproduction',
        training_required=False) for k,s in CATALOG.items()}
