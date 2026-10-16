"""Immutable metadata: categories are not algorithm or deployment identities."""
from copy import deepcopy
from ric.near_rt.rule_based_v1.catalog import CATALOG, SOURCES, DEFAULTS, PROFILES

VARIANTS={
 'throughput_power':{'cqi_proportional':'CQI-proportional frequency power','backlog_cqi':'Delivered backlog × CQI power'},
 'rf_energy':{'sinr_pi':'QoS-guarded weakest-reference SINR PI'},
 'cell_sleep':{'empty_only':'Covered empty-cell sleep; no migration'},
 'load_balancing':{'load_pi':'Directed load-gap CIO PI'},
 'traffic_steering':{'a3_rsrp':'Continuous delivered RSRP A3-like trigger','sla_max_sinr':'SLA-guarded strongest reference SINR'},
 'mobility_robustness':{'window_search':'Live completed-event window coordinate search'},
 'ue_fairness':{'equal_share':'Equal PRB caps with rotating remainder','deficit_airtime':'Executed PRB-slot deficit compensation'},
 'slice_sla':{'goodput_pi':'Declared-demand goodput PI weights','service_deficit':'Measured successful-service deficit weights'},
 'latency_guard':{'delay_pi':'Normalized HOL quantile PI bias'},
 'interference_coordination':{'adaptive_sfr':'Fixed-reuse observed-load/CQI soft frequency power'},
 'ret_coverage':{'coverage_grid':'Executed tilt-grid lower-quantile reference search'},
 'beam_management':{'max_min_reference':'Worst-user legal-codebook reference maximization'},
}
from .objectives import OBJECTIVES, OBJECTIVE_DEFAULTS, SEARCH_ALGORITHMS, objective_metadata
for category,algorithm,goal,_,_,_ in OBJECTIVES:
    VARIANTS[category][algorithm]=goal

EXTRA_DEFAULTS=dict(pi_kp=.2,pi_ki=.1,hol_target=.5,delay_quantile=.95,
    ret_window_ms=400.,mro_evaluation_ms=1000.,deficit_horizon_ms=1000.,sfr_cqi_threshold=7.,association_settle_ms=100.)
EXTRA_KPIS={
 'deficit_airtime':('executed_prb_slots_total','resource_counter_epoch'),
 'goodput_pi':('requested_dl_rate_bps',),
 'service_deficit':('requested_dl_rate_bps','delivered_bits_total'),
 'coverage_grid':('rsrp_dbm',),
}


def category_catalog():
    return {k:dict(category_id=k,goal=s[6],writable_actions=list(s[3]),commands=list(s[4]),
        overlaps=sorted(x for x,t in CATALOG.items() if x!=k and set(s[3]) & set(t[3]))) for k,s in CATALOG.items()}


def implementation_catalog():
    result={}
    for category,spec in CATALOG.items():
        for algorithm,description in {'legacy_v1':spec[6],**VARIANTS[category]}.items():
            ident=category+':'+algorithm
            result[ident]=dict(category_id=category,implementation_id=ident,algorithm=algorithm,version='1.0.0',
                description=description,required_kpis=list(dict.fromkeys([*spec[2],*EXTRA_KPIS.get(algorithm,())])),
                writable_actions=list(spec[3]),commands=list(spec[4]),priority=spec[0],min_action_ms=spec[1],
                defaults=deepcopy({**DEFAULTS,**({} if algorithm=='legacy_v1' else EXTRA_DEFAULTS)}),
                sources=[SOURCES[x] for x in spec[5]],
                source_classification='existing_functional_adaptation' if algorithm=='legacy_v1' else 'local_deterministic_baseline',
                acceptance_status='requires_version_specific_evidence',training_required=False)
    for row in OBJECTIVES:
        category,algorithm,_,_,_,extra=row
        spec=result[category+':'+algorithm]
        spec['defaults'].update(OBJECTIVE_DEFAULTS)
        spec['required_kpis']=list(dict.fromkeys(spec['required_kpis']+extra+(['config_revision'] if algorithm in SEARCH_ALGORITHMS else [])))
    for ident,spec in result.items():
        spec.update(objective_metadata(ident,spec['algorithm'],spec['description']))
    return result


IMPLEMENTATIONS=implementation_catalog()
