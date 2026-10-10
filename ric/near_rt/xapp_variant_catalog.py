"""Default variant pools for Near-RT xApps.

This keeps the implementation scalable: each xApp category can expose a set
of deployment-ready profiles without duplicating controller classes.
"""

from __future__ import annotations

from typing import Any


BASE_XAPP_CAPABILITIES: dict[str, dict[str, object]] = {
    "mobility_predictor_xapp": {
        "description": "Predicts mobility, handover risk, and future serving-cell degradation.",
        "kpis": ["handover_risk", "beam_misalignment_risk", "future_cqi", "service_continuity"],
        "tunable_fields": [
            "mobility_bias_db",
            "steering_hysteresis_db",
            "time_to_trigger_ms",
            "mobility_margin_db",
            "measurement_event_offset_db",
        ],
        "tuning_mode": "heuristic_parameter",
    },
    "traffic_steering_xapp": {
        "description": "Steers users across serving/candidate cells to improve load and edge performance.",
        "kpis": ["load_balance", "edge_throughput", "handover_success", "queue_relief"],
        "tunable_fields": [
            "traffic_steering_preference",
            "steering_aggressiveness",
            "load_threshold",
            "cio_bias_db",
            "mobility_bias_db",
            "measurement_event_offset_db",
        ],
        "tuning_mode": "heuristic_parameter",
    },
    "power_saving_xapp": {
        "description": "Caps power and shifts beam preference to reduce energy cost while preserving service.",
        "kpis": ["energy_efficiency", "power_cost", "sla_preservation"],
        "tunable_fields": [
            "power_cap_dbm",
            "beam_candidate_preference",
            "sleep_threshold",
            "wake_up_threshold",
            "dwell_time_slots",
        ],
        "tuning_mode": "heuristic_parameter",
    },
    "spectrum_optimizer_xapp": {
        "description": "Optimizes scheduler policy and slice weights for better spectrum efficiency.",
        "kpis": ["spectral_efficiency", "prb_utilization", "interference_avoidance"],
        "tunable_fields": [
            "scheduler_policy_hint",
            "slice_weight_adjustment",
            "power_offset_db",
            "muting_ratio",
            "coordination_threshold",
        ],
        "tuning_mode": "heuristic_parameter",
    },
    "latency_scheduler_xapp": {
        "description": "Biases scheduling toward low-delay traffic and queue urgency.",
        "kpis": ["latency", "tail_latency", "service_gap", "queue_reduction"],
        "tunable_fields": [
            "scheduler_policy_hint",
            "latency_bias",
            "slice_weight_adjustment",
            "prb_weight",
            "priority_coefficient",
            "delay_threshold_ms",
        ],
        "tuning_mode": "heuristic_parameter",
    },
    "massive_mimo_beamformer_xapp": {
        "description": "Tunes beam preference and capacity-vs-coverage tradeoff for MIMO gain.",
        "kpis": ["sinr", "throughput", "spatial_gain", "coverage"],
        "tunable_fields": [
            "beam_candidate_preference",
            "power_cap_dbm",
            "power_offset_db",
            "coordination_threshold",
        ],
        "tuning_mode": "heuristic_parameter",
    },
    "ran_slicing_manager_xapp": {
        "description": "Adjusts slice resource emphasis to honor SLA priorities.",
        "kpis": ["slice_sla", "fairness", "slice_queue_balance"],
        "tunable_fields": [
            "scheduler_policy_hint",
            "slice_weight_adjustment",
            "prb_weight",
            "priority_coefficient",
        ],
        "tuning_mode": "heuristic_parameter",
    },
    "urllc_guard_xapp": {
        "description": "Acts as a guardrail to protect URLLC reliability and latency.",
        "kpis": ["urllc_reliability", "urllc_latency", "retx_pressure"],
        "tunable_fields": [
            "scheduler_policy_hint",
            "latency_bias",
            "power_cap_dbm",
            "prb_weight",
            "priority_coefficient",
            "delay_threshold_ms",
        ],
        "tuning_mode": "heuristic_parameter",
    },
    "cell_sleep_xapp": {
        "description": "Transitions lightly loaded cells into sleep modes to reduce energy cost.",
        "kpis": ["energy_efficiency", "power_cost", "load_consolidation"],
        "tunable_fields": [
            "cell_sleep_mode",
            "power_cap_dbm",
            "sleep_threshold",
            "wake_up_threshold",
            "dwell_time_slots",
        ],
        "tuning_mode": "heuristic_parameter",
    },
}


def _variant(
    name: str,
    description: str,
    params: dict[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "params": dict(params),
    }


def _dict_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): value for key, value in value.items()}


def _list_object(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return list(value)


DEFAULT_XAPP_VARIANTS: dict[str, list[dict[str, object]]] = {
    "mobility_predictor_xapp": [
        _variant("mobility_predictor_xapp_early_warning", "Early handover-risk detector.", {"risk_threshold": 0.25, "time_to_trigger_ms": 80.0, "mobility_margin_db": 1.5}),
        _variant("mobility_predictor_xapp_balanced", "Balanced mobility forecaster.", {"risk_threshold": 0.35, "time_to_trigger_ms": 120.0, "mobility_margin_db": 2.0}),
        _variant("mobility_predictor_xapp_high_speed", "More sensitive for fast-moving users.", {"risk_threshold": 0.22, "time_to_trigger_ms": 60.0, "mobility_margin_db": 1.2}),
        _variant("mobility_predictor_xapp_cell_edge", "Focus on edge mobility instability.", {"risk_threshold": 0.30, "time_to_trigger_ms": 100.0, "mobility_margin_db": 2.5}),
        _variant("mobility_predictor_xapp_hysteresis_guard", "Bias toward stable serving-cell retention.", {"risk_threshold": 0.40, "time_to_trigger_ms": 180.0, "mobility_margin_db": 3.2}),
        _variant("mobility_predictor_xapp_mmwave_guard", "Prepared for mmWave fragility.", {"risk_threshold": 0.28, "time_to_trigger_ms": 70.0, "mobility_margin_db": 2.8}),
        _variant("mobility_predictor_xapp_conservative", "Only raises strong mobility warnings.", {"risk_threshold": 0.50}),
        _variant("mobility_predictor_xapp_dense_urban", "Urban mobility-sensitive profile.", {"risk_threshold": 0.32}),
        _variant("mobility_predictor_xapp_low_stale_csi", "Acts earlier when mobility risk rises.", {"risk_threshold": 0.27}),
        _variant("mobility_predictor_xapp_handover_push", "Aggressive handover preparation profile.", {"risk_threshold": 0.20}),
    ],
    "traffic_steering_xapp": [
        _variant("traffic_steering_xapp_balanced", "Balanced steering between load and quality.", {"load_threshold": 0.70, "edge_sinr_threshold_db": 5.0, "steering_aggressiveness": 0.55, "cio_bias_db": 1.0}),
        _variant("traffic_steering_xapp_hotspot_relief", "Aggressive overload offload for hotspot cells.", {"load_threshold": 0.55, "edge_sinr_threshold_db": 7.0, "steering_aggressiveness": 0.82, "cio_bias_db": 3.0}),
        _variant("traffic_steering_xapp_edge_rescue", "Favor edge users with better alternatives.", {"load_threshold": 0.68, "edge_sinr_threshold_db": 8.5, "steering_aggressiveness": 0.62, "cio_bias_db": 2.2}),
        _variant("traffic_steering_xapp_mmwave_prefer", "More willing to move users toward mmWave cells.", {"load_threshold": 0.78, "edge_sinr_threshold_db": 4.0}),
        _variant("traffic_steering_xapp_macro_anchor", "More cautious about leaving macro coverage.", {"load_threshold": 0.82, "edge_sinr_threshold_db": 3.5}),
        _variant("traffic_steering_xapp_urlcc_safe", "Conservative steering for sensitive traffic.", {"load_threshold": 0.60, "edge_sinr_threshold_db": 6.5}),
        _variant("traffic_steering_xapp_load_first", "Prioritize load balancing over signal margin.", {"load_threshold": 0.50, "edge_sinr_threshold_db": 3.0}),
        _variant("traffic_steering_xapp_quality_first", "Require stronger signal advantage before steering.", {"load_threshold": 0.74, "edge_sinr_threshold_db": 9.0}),
        _variant("traffic_steering_xapp_dense_cluster", "Dense-cluster steering behavior.", {"load_threshold": 0.58, "edge_sinr_threshold_db": 5.5}),
        _variant("traffic_steering_xapp_sticky", "Sticky steering with fewer moves.", {"load_threshold": 0.88, "edge_sinr_threshold_db": 2.5}),
    ],
    "power_saving_xapp": [
        _variant("power_saving_xapp_micro_trim", "Tiny power trim without quality risk.", {"low_load_threshold": 0.48, "queue_threshold_bytes": 7000, "min_power_cap_dbm": 27.5}),
        _variant("power_saving_xapp_balanced", "Balanced energy saving profile.", {"low_load_threshold": 0.40, "queue_threshold_bytes": 5000, "min_power_cap_dbm": 26.0}),
        _variant("power_saving_xapp_sleepy", "Aggressive power reduction under light load.", {"low_load_threshold": 0.28, "queue_threshold_bytes": 2500, "min_power_cap_dbm": 22.0}),
        _variant("power_saving_xapp_night_mode", "Night-time energy policy.", {"low_load_threshold": 0.22, "queue_threshold_bytes": 2000, "min_power_cap_dbm": 20.0}),
        _variant("power_saving_xapp_urlcc_safe", "Gentle energy saving that preserves reliability.", {"low_load_threshold": 0.52, "queue_threshold_bytes": 9000, "min_power_cap_dbm": 28.0}),
        _variant("power_saving_xapp_mmwave_trim", "Power trim for dense mmWave small cells.", {"low_load_threshold": 0.34, "queue_threshold_bytes": 3500, "min_power_cap_dbm": 23.5}),
        _variant("power_saving_xapp_macro_trim", "Conservative macro energy profile.", {"low_load_threshold": 0.44, "queue_threshold_bytes": 6000, "min_power_cap_dbm": 27.0}),
        _variant("power_saving_xapp_queue_sensitive", "Backs off when queues begin to rise.", {"low_load_threshold": 0.30, "queue_threshold_bytes": 1500, "min_power_cap_dbm": 25.0}),
        _variant("power_saving_xapp_beam_balanced", "Leans toward balanced beams while saving power.", {"low_load_threshold": 0.36, "queue_threshold_bytes": 4500, "min_power_cap_dbm": 24.0}),
        _variant("power_saving_xapp_energy_max", "Maximum energy-saving pressure.", {"low_load_threshold": 0.18, "queue_threshold_bytes": 1200, "min_power_cap_dbm": 18.0}),
    ],
    "spectrum_optimizer_xapp": [
        _variant("spectrum_optimizer_xapp_balanced", "Balanced spectrum-efficiency optimizer.", {"utilization_threshold": 0.65, "interference_threshold_db": 2.5, "coordination_threshold": 2.5, "muting_ratio": 0.08}),
        _variant("spectrum_optimizer_xapp_dense_packing", "More eager to pack resources.", {"utilization_threshold": 0.55, "interference_threshold_db": 3.0, "coordination_threshold": 3.0, "muting_ratio": 0.04}),
        _variant("spectrum_optimizer_xapp_interference_averse", "Avoids aggressive packing under interference.", {"utilization_threshold": 0.72, "interference_threshold_db": 1.5, "coordination_threshold": 1.5, "muting_ratio": 0.18}),
        _variant("spectrum_optimizer_xapp_slice_bias", "More ready to switch into slice-aware mode.", {"utilization_threshold": 0.50, "interference_threshold_db": 2.0}),
        _variant("spectrum_optimizer_xapp_mu_friendly", "Favors higher multiplexing opportunity.", {"utilization_threshold": 0.60, "interference_threshold_db": 3.5}),
        _variant("spectrum_optimizer_xapp_guarded", "Conservative optimizer for stressed cells.", {"utilization_threshold": 0.78, "interference_threshold_db": 1.2}),
        _variant("spectrum_optimizer_xapp_prb_efficiency", "PRB-efficiency focused profile.", {"utilization_threshold": 0.58, "interference_threshold_db": 2.8}),
        _variant("spectrum_optimizer_xapp_latency_mix", "Balances latency and packing.", {"utilization_threshold": 0.62, "interference_threshold_db": 2.2}),
        _variant("spectrum_optimizer_xapp_stable", "Stable spectrum policy with fewer flips.", {"utilization_threshold": 0.75, "interference_threshold_db": 2.7}),
        _variant("spectrum_optimizer_xapp_aggressive", "Aggressive spectrum pressure under congestion.", {"utilization_threshold": 0.45, "interference_threshold_db": 3.8}),
    ],
    "latency_scheduler_xapp": [
        _variant("latency_scheduler_xapp_urllc_fastpath", "Aggressive URLLC latency profile.", {"queue_threshold_bytes": 900, "prb_weight": 1.45, "priority_coefficient": 2.2, "delay_threshold_ms": 10.0}),
        _variant("latency_scheduler_xapp_balanced", "Balanced latency-sensitive scheduling.", {"queue_threshold_bytes": 2500, "prb_weight": 1.0, "priority_coefficient": 1.4, "delay_threshold_ms": 20.0}),
        _variant("latency_scheduler_xapp_tail_guard", "Focus on tail-latency containment.", {"queue_threshold_bytes": 1800, "prb_weight": 1.25, "priority_coefficient": 1.8, "delay_threshold_ms": 15.0}),
        _variant("latency_scheduler_xapp_control_plane", "Prioritize control-plane latency.", {"queue_threshold_bytes": 1200}),
        _variant("latency_scheduler_xapp_backlog_relief", "Trigger sooner under rising backlog.", {"queue_threshold_bytes": 1500}),
        _variant("latency_scheduler_xapp_embb_friendly", "Latency-aware but less aggressive.", {"queue_threshold_bytes": 4200}),
        _variant("latency_scheduler_xapp_harq_rescue", "Favors latency under HARQ stress.", {"queue_threshold_bytes": 1000}),
        _variant("latency_scheduler_xapp_edge_safe", "Edge-user latency protection profile.", {"queue_threshold_bytes": 1600}),
        _variant("latency_scheduler_xapp_service_gap", "Targets service-gap reduction.", {"queue_threshold_bytes": 2200}),
        _variant("latency_scheduler_xapp_extreme", "Highly aggressive low-latency scheduler.", {"queue_threshold_bytes": 700}),
    ],
    "massive_mimo_beamformer_xapp": [
        _variant("massive_mimo_beamformer_xapp_balanced", "Balanced beam/capacity profile.", {"beam_risk_threshold": 0.50, "coverage_sinr_threshold_db": 7.0}),
        _variant("massive_mimo_beamformer_xapp_coverage_boost", "Strong coverage-oriented beam control.", {"beam_risk_threshold": 0.35, "coverage_sinr_threshold_db": 9.0}),
        _variant("massive_mimo_beamformer_xapp_capacity_push", "Aggressive capacity profile.", {"beam_risk_threshold": 0.62, "coverage_sinr_threshold_db": 5.5}),
        _variant("massive_mimo_beamformer_xapp_edge_rescue", "Target cell-edge beam rescue.", {"beam_risk_threshold": 0.30, "coverage_sinr_threshold_db": 10.5}),
        _variant("massive_mimo_beamformer_xapp_mu_dense", "Beam profile for denser MU opportunities.", {"beam_risk_threshold": 0.55, "coverage_sinr_threshold_db": 6.0}),
        _variant("massive_mimo_beamformer_xapp_mmwave_guard", "Conservative mmWave beam protection.", {"beam_risk_threshold": 0.28, "coverage_sinr_threshold_db": 8.5}),
        _variant("massive_mimo_beamformer_xapp_macro_capacity", "Macro-capacity oriented beam profile.", {"beam_risk_threshold": 0.58, "coverage_sinr_threshold_db": 6.5}),
        _variant("massive_mimo_beamformer_xapp_precoder_safe", "Safer precoder choice under uncertainty.", {"beam_risk_threshold": 0.40, "coverage_sinr_threshold_db": 7.8}),
        _variant("massive_mimo_beamformer_xapp_sinr_max", "SINR-maximizing beam choice.", {"beam_risk_threshold": 0.33, "coverage_sinr_threshold_db": 9.5}),
        _variant("massive_mimo_beamformer_xapp_spatial_gain", "Spatial-gain oriented beamformer.", {"beam_risk_threshold": 0.60, "coverage_sinr_threshold_db": 5.8}),
    ],
    "ran_slicing_manager_xapp": [
        _variant("ran_slicing_manager_xapp_balanced", "Balanced slicing manager.", {"slice_dominance_threshold": 0.45}),
        _variant("ran_slicing_manager_xapp_urllc_bias", "URLLC-biased slicing policy.", {"slice_dominance_threshold": 0.35}),
        _variant("ran_slicing_manager_xapp_embb_bias", "eMBB-preserving slicing profile.", {"slice_dominance_threshold": 0.52}),
        _variant("ran_slicing_manager_xapp_mmtc_safe", "Prevents mMTC starvation.", {"slice_dominance_threshold": 0.40}),
        _variant("ran_slicing_manager_xapp_sla_strict", "Stricter SLA-enforcement slicer.", {"slice_dominance_threshold": 0.30}),
        _variant("ran_slicing_manager_xapp_fairness", "Fairness-oriented slicing profile.", {"slice_dominance_threshold": 0.60}),
        _variant("ran_slicing_manager_xapp_congestion", "Acts faster under slice congestion.", {"slice_dominance_threshold": 0.32}),
        _variant("ran_slicing_manager_xapp_macro_mix", "Macro mix slicing profile.", {"slice_dominance_threshold": 0.50}),
        _variant("ran_slicing_manager_xapp_hotspot", "Hotspot-aware slicing emphasis.", {"slice_dominance_threshold": 0.38}),
        _variant("ran_slicing_manager_xapp_guarded", "Conservative slice reweighting.", {"slice_dominance_threshold": 0.68}),
    ],
    "urllc_guard_xapp": [
        _variant("urllc_guard_xapp_strict", "Strict URLLC protection guard.", {"queue_threshold_bytes": 700, "bler_threshold": 0.08, "latency_budget_threshold_ms": 12.0}),
        _variant("urllc_guard_xapp_balanced", "Balanced URLLC guard.", {"queue_threshold_bytes": 1200, "bler_threshold": 0.12, "latency_budget_threshold_ms": 20.0}),
        _variant("urllc_guard_xapp_tail_safe", "Tail-latency protective guard.", {"queue_threshold_bytes": 900, "bler_threshold": 0.10, "latency_budget_threshold_ms": 16.0}),
        _variant("urllc_guard_xapp_edge_safe", "Protect weak-edge URLLC users.", {"queue_threshold_bytes": 1000, "bler_threshold": 0.11, "latency_budget_threshold_ms": 18.0}),
        _variant("urllc_guard_xapp_harq_safe", "Acts sooner under retransmission stress.", {"queue_threshold_bytes": 800, "bler_threshold": 0.09, "latency_budget_threshold_ms": 14.0}),
        _variant("urllc_guard_xapp_conservative", "Guard only severe URLLC risk.", {"queue_threshold_bytes": 1800, "bler_threshold": 0.16, "latency_budget_threshold_ms": 24.0}),
        _variant("urllc_guard_xapp_voice_safe", "Telephony/control protection profile.", {"queue_threshold_bytes": 950, "bler_threshold": 0.10, "latency_budget_threshold_ms": 15.0}),
        _variant("urllc_guard_xapp_mmwave", "URLLC guard for fragile mmWave cells.", {"queue_threshold_bytes": 850, "bler_threshold": 0.09, "latency_budget_threshold_ms": 13.0}),
        _variant("urllc_guard_xapp_cluster_guard", "Cluster-wide URLLC shield.", {"queue_threshold_bytes": 1100, "bler_threshold": 0.11, "latency_budget_threshold_ms": 17.0}),
        _variant("urllc_guard_xapp_emergency", "Emergency URLLC guard rail.", {"queue_threshold_bytes": 600, "bler_threshold": 0.07, "latency_budget_threshold_ms": 10.0}),
    ],
    "cell_sleep_xapp": [
        _variant("cell_sleep_xapp_micro_trim", "Mild micro-sleep preference.", {"micro_sleep_load_threshold": 0.34, "light_sleep_load_threshold": 0.20, "deep_sleep_load_threshold": 0.10, "max_queue_bytes_for_sleep": 5000, "sleep_threshold": 0.34, "wake_up_threshold": 0.46, "dwell_time_slots": 20}),
        _variant("cell_sleep_xapp_balanced", "Balanced cell-sleep policy.", {"micro_sleep_load_threshold": 0.28, "light_sleep_load_threshold": 0.18, "deep_sleep_load_threshold": 0.08, "max_queue_bytes_for_sleep": 4000, "sleep_threshold": 0.28, "wake_up_threshold": 0.42, "dwell_time_slots": 30}),
        _variant("cell_sleep_xapp_aggressive", "Aggressive sleep transition profile.", {"micro_sleep_load_threshold": 0.40, "light_sleep_load_threshold": 0.26, "deep_sleep_load_threshold": 0.14, "max_queue_bytes_for_sleep": 6000, "sleep_threshold": 0.40, "wake_up_threshold": 0.55, "dwell_time_slots": 45}),
        _variant("cell_sleep_xapp_mmwave_hotspot", "Sleep profile for dense small cells.", {"micro_sleep_load_threshold": 0.30, "light_sleep_load_threshold": 0.22, "deep_sleep_load_threshold": 0.12, "max_queue_bytes_for_sleep": 3500}),
        _variant("cell_sleep_xapp_macro_guarded", "Conservative macro sleep policy.", {"micro_sleep_load_threshold": 0.20, "light_sleep_load_threshold": 0.12, "deep_sleep_load_threshold": 0.04, "max_queue_bytes_for_sleep": 2500}),
        _variant("cell_sleep_xapp_night_mode", "Night-time deep sleep preference.", {"micro_sleep_load_threshold": 0.45, "light_sleep_load_threshold": 0.30, "deep_sleep_load_threshold": 0.18, "max_queue_bytes_for_sleep": 7000}),
        _variant("cell_sleep_xapp_urlcc_safe", "Sleep only when URLLC risk is low.", {"micro_sleep_load_threshold": 0.22, "light_sleep_load_threshold": 0.14, "deep_sleep_load_threshold": 0.06, "max_queue_bytes_for_sleep": 1800}),
        _variant("cell_sleep_xapp_queue_strict", "Queue-sensitive sleep controller.", {"micro_sleep_load_threshold": 0.26, "light_sleep_load_threshold": 0.16, "deep_sleep_load_threshold": 0.08, "max_queue_bytes_for_sleep": 1200}),
        _variant("cell_sleep_xapp_cluster_saver", "Cluster-wide energy saver.", {"micro_sleep_load_threshold": 0.36, "light_sleep_load_threshold": 0.24, "deep_sleep_load_threshold": 0.12, "max_queue_bytes_for_sleep": 4500}),
        _variant("cell_sleep_xapp_deep_saver", "Deep sleep maximizing profile.", {"micro_sleep_load_threshold": 0.48, "light_sleep_load_threshold": 0.34, "deep_sleep_load_threshold": 0.20, "max_queue_bytes_for_sleep": 8000}),
    ],
}


def expand_default_variant_xapp_configs(xapp_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded_items: list[dict[str, Any]] = []
    for xapp_item in list(xapp_items or []):
        expanded_items.append(dict(xapp_item))
        if not bool(xapp_item.get("deploy_default_variants", False)):
            continue
        base_type = str(xapp_item.get("type", "") or "")
        base_priority = int(xapp_item.get("priority", 10))
        for index, variant in enumerate(DEFAULT_XAPP_VARIANTS.get(base_type, []), start=1):
            expanded_variant = dict(xapp_item)
            expanded_variant.pop("deploy_default_variants", None)
            expanded_variant.update(_dict_object(variant.get("params", {})))
            expanded_variant["type"] = base_type
            expanded_variant["name"] = str(variant.get("name", f"{base_type}_variant_{index:02d}"))
            expanded_variant["priority"] = int(expanded_variant.get("priority", base_priority)) + index
            expanded_variant["variant_description"] = str(variant.get("description", ""))
            expanded_items.append(expanded_variant)
    return expanded_items


def build_variant_capability_catalog() -> dict[str, dict[str, object]]:
    catalog: dict[str, dict[str, object]] = {}
    for base_type, capability in BASE_XAPP_CAPABILITIES.items():
        catalog[base_type] = dict(capability)
        for variant in DEFAULT_XAPP_VARIANTS.get(base_type, []):
            variant_name = str(variant.get("name", ""))
            if variant_name == "":
                continue
            catalog[variant_name] = {
                "description": f"{capability['description']} {str(variant.get('description', '')).strip()}".strip(),
                "kpis": _list_object(capability.get("kpis", [])),
                "tunable_fields": _list_object(capability.get("tunable_fields", [])),
                "tuning_mode": str(capability.get("tuning_mode", "heuristic_parameter")),
            }
    return catalog


def canonical_xapp_category(xapp_name: str) -> str:
    normalized_name = str(xapp_name or "").strip()
    for base_type in BASE_XAPP_CAPABILITIES.keys():
        if normalized_name == base_type or normalized_name.startswith(f"{base_type}_"):
            return str(base_type)
    return normalized_name
