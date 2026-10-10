"""Experiment 3-B: contract-shaped tuner bridge to the slot-driven simulator."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from core.config_loader import load_all_configs, load_yaml_file
from core.factory import build_controller_registry, build_phy_adapter, build_scheduler
from core.simulator import Simulator
from experiments.event_driven_contract_adaptation import (
    BASE_KPI_VALUES,
    DIMENSION_NAMES,
    EVENT_SPECS,
    LOWER_BOUNDS,
    MAIN_METHODS,
    SEVERITY_LEVELS,
    UPPER_BOUNDS,
    Z_REF,
    _apply_projection_and_fallback,
    _array_to_action,
    _ideal_action,
    _projection_distance,
    _project_to_contract,
    _raw_contract_violation,
    _rng,
    _select_raw_action,
    _state_vector,
    _train_neural_policies,
)
from experiments.paper_eval_artifacts import mean, percentile, write_csv, write_json
from experiments.scenario_scan_utils import build_runtime_scenario_config
from models.action import ConfigAction
from scenarios.scenario_generator import ScenarioGenerator
from scenarios.scenario_registry import ScenarioRegistry, deep_merge_dict


EXPERIMENT3_BRIDGE_VERSION = "experiment3_simulator_bridge_v1"
BRIDGE_UE_REPLICAS_PER_SLICE = 5
BRIDGE_ARRIVAL_RATE_BPS_BY_SLICE = {
    "embb": 800_000.0,
    "urllc": 200_000.0,
    "mmtc": 100_000.0,
}
BRIDGE_METHOD_NAMES = (
    "no_near_rt_tuning",
    "rule_based_bounded_tuner",
    "ddpg_projection_only",
    "full_contract_constrained_ddpg",
)
BRIDGE_EVENT_NAMES = (
    "traffic_burst",
    "mobility_spike",
    "interference_perturbation",
    "resource_perturbation",
)
BRIDGE_ROBUSTNESS_PROFILES = {
    "nominal": {
        "description": "Default bridge delays and trusted contract path.",
    },
    "stale_telemetry": {
        "description": "Four-slot additional policy-input staleness plus longer simulator observation queues.",
        "obs_delay_slots": 5,
        "measurement_queue_delay_slots": 5,
        "policy_kpi_lag_slots": 4,
    },
    "noisy_kpi": {
        "description": "Deterministic bounded KPI noise at the policy input.",
        "kpi_noise_fraction": 0.18,
    },
    "over_tight_contract": {
        "description": "Compiled action region is restricted to 48 percent of the trusted contract width.",
        "contract_width_scale": 0.48,
    },
    "over_loose_contract": {
        "description": "Compiled candidates expand deviations from the reference before the trusted runtime shield.",
        "candidate_deviation_scale": 1.80,
    },
    "fallback_mismatch": {
        "description": "A mismatched fallback registry intermittently deploys the neutral safe action during an event.",
        "fallback_mismatch_period_slots": 8,
    },
    "delayed_action": {
        "description": "Four-slot configuration actuation delay.",
        "config_actuation_delay_slots": 4,
        "bridge_action_delay_slots": 4,
    },
}


def _method_profiles(method_names: list[str] | None = None) -> list[Any]:
    selected = set(method_names or BRIDGE_METHOD_NAMES)
    methods = [method for method in MAIN_METHODS if method.name in selected]
    missing = sorted(selected - {method.name for method in methods})
    if missing:
        raise ValueError(f"Unknown simulator-bridge methods: {missing}")
    return methods


def _event_specs(event_names: list[str] | None = None) -> list[Any]:
    selected = set(event_names or BRIDGE_EVENT_NAMES)
    events = [event for event in EVENT_SPECS if event.name in selected]
    missing = sorted(selected - {event.name for event in events})
    if missing:
        raise ValueError(f"Unknown simulator-bridge events: {missing}")
    return events


def _robustness_profile(name: str) -> dict[str, Any]:
    if name not in BRIDGE_ROBUSTNESS_PROFILES:
        raise ValueError(f"Unknown simulator-bridge robustness condition: {name}")
    return dict(BRIDGE_ROBUSTNESS_PROFILES[name])


def _build_simulator(
    *,
    seed: int,
    episode_length: int,
    output_dir: Path,
    robustness_profile: dict[str, Any] | None = None,
) -> Simulator:
    project_root = Path(__file__).resolve().parents[1]
    configs = load_all_configs(str(project_root / "configs"))
    scenario_config = {
        "scenario_name": "oran_paper_eval_base",
        "template_overrides": {},
        "overrides": {},
    }
    scenario_template = ScenarioRegistry().resolve(scenario_config)
    profile = dict(robustness_profile or {})
    sim_config = deep_merge_dict(
        configs["sim"],
        {
            "total_slots": int(episode_length),
            "rng_seed": int(seed),
            "obs_delay_slots": int(profile.get("obs_delay_slots", 1)),
            "config_actuation_delay_slots": int(profile.get("config_actuation_delay_slots", 0)),
            "measurement_queue_delay_slots": int(profile.get("measurement_queue_delay_slots", 1)),
        },
    )
    ue_config = deepcopy(configs["ue"])
    # Keep the no-event operating point serviceable so recovery after a
    # transient shock is observable instead of being hidden by baseline
    # overload. The traffic mix and relative slice priorities are preserved.
    bridge_ues: list[dict[str, Any]] = []
    for ue_item in list(ue_config.get("ues", []) or []):
        slice_id = str(ue_item.get("slice_id", "")).strip().lower()
        for replica in range(BRIDGE_UE_REPLICAS_PER_SLICE):
            clone = deepcopy(ue_item)
            clone["ue_id"] = f"{ue_item.get('ue_id', slice_id)}_bridge_{replica}"
            if slice_id in BRIDGE_ARRIVAL_RATE_BPS_BY_SLICE:
                clone["arrival_rate_bps"] = BRIDGE_ARRIVAL_RATE_BPS_BY_SLICE[slice_id]
            clone["queue_bytes"] = min(int(clone.get("queue_bytes", 0)), 1200)
            if "start_offset_slots" in clone:
                clone["start_offset_slots"] = int(clone.get("start_offset_slots", 0)) + 3 * replica
            bridge_ues.append(clone)
    ue_config["ues"] = bridge_ues

    scenario_bundle = ScenarioGenerator().generate(
        template=scenario_template,
        scenario_config=scenario_config,
        sim_config=sim_config,
        cell_config=configs["cell"],
        ue_config=ue_config,
        beamforming_config=configs["beamforming"],
    )
    # The lightweight fallback YAML parser represents blank scalar values as
    # empty mappings; normalize this optional limit before scheduler creation.
    if isinstance(scenario_bundle.sim_config.get("scheduler_max_total_spatial_layers"), dict):
        scenario_bundle.sim_config["scheduler_max_total_spatial_layers"] = None
    ric_config = {
        "xapp_interval_slots": 5,
        "rapp_interval_slots": 10,
        "run_rapps_on_start": False,
        "orchestration_warmup_slots": 0,
        "xapps": [],
        "rapps": [],
        "near_rt_mode": "external_contract_bridge",
    }
    runtime_config = build_runtime_scenario_config(
        scenario_name=scenario_bundle.scenario_name,
        scenario_type=scenario_bundle.scenario_template.channel.scenario_type,
        sim_config=scenario_bundle.sim_config,
        cell_config=scenario_bundle.cell_config,
        ric_config=ric_config,
    )
    phy_config = deepcopy(load_yaml_file(str(project_root / "configs" / "phy.yaml")))
    phy_config["phy_type"] = "custom_phy"
    return Simulator(
        config=runtime_config,
        controller_registry=build_controller_registry(ric_config),
        scheduler=build_scheduler(scenario_bundle.sim_config),
        phy_adapter=build_phy_adapter(phy_config, scenario_bundle.beamforming_config),
        cell_config=scenario_bundle.cell_config,
        ue_config=scenario_bundle.ue_config,
        beamforming_config=scenario_bundle.beamforming_config,
        output_dir=str(output_dir),
    )


def _bridge_event_intensity(
    slot: int,
    *,
    event_time: int,
    shock_duration: int,
    severity_scale: float,
) -> float:
    if slot < event_time:
        return 0.0
    if slot < event_time + shock_duration:
        return float(severity_scale)
    elapsed = slot - event_time - shock_duration
    return float(severity_scale) * math.exp(-float(elapsed) / max(3.0, 0.55 * shock_duration))


def _policy_kpis_under_condition(
    kpi_history: list[dict[str, float]],
    *,
    robustness_profile: dict[str, Any],
    rng: Any,
) -> tuple[dict[str, float], int]:
    lag_slots = max(0, int(robustness_profile.get("policy_kpi_lag_slots", 0)))
    history_index = max(0, len(kpi_history) - 1 - lag_slots)
    selected = dict(kpi_history[history_index])
    noise_fraction = max(0.0, float(robustness_profile.get("kpi_noise_fraction", 0.0)))
    if noise_fraction <= 0.0:
        return selected, int(lag_slots > 0)
    noisy: dict[str, float] = {}
    for name, value in selected.items():
        scale = max(abs(float(value)), abs(float(BASE_KPI_VALUES.get(name, 0.0))), 0.05)
        perturbed = float(value) + rng.uniform(-noise_fraction, noise_fraction) * scale
        if name != "cell_edge_throughput_mbps":
            perturbed = min(1.0, perturbed)
        noisy[name] = max(0.0, perturbed)
    return noisy, 1


def _candidate_under_condition(
    raw_action: dict[str, float],
    *,
    robustness_profile: dict[str, Any],
) -> tuple[dict[str, float], float, int]:
    width_scale = float(robustness_profile.get("contract_width_scale", 1.0))
    deviation_scale = float(robustness_profile.get("candidate_deviation_scale", 1.0))
    stressed = dict(raw_action)
    injected = 0
    if width_scale < 1.0:
        stressed = {
            name: min(
                float(Z_REF[name]) + width_scale * (float(UPPER_BOUNDS[name]) - float(Z_REF[name])),
                max(
                    float(Z_REF[name])
                    + width_scale * (float(LOWER_BOUNDS[name]) - float(Z_REF[name])),
                    float(stressed.get(name, Z_REF[name])),
                ),
            )
            for name in DIMENSION_NAMES
        }
        injected = 1
    if deviation_scale != 1.0:
        stressed = {
            name: float(Z_REF[name])
            + deviation_scale * (float(stressed.get(name, Z_REF[name])) - float(Z_REF[name]))
            for name in DIMENSION_NAMES
        }
        injected = 1
    return stressed, _projection_distance(raw_action, stressed), injected


def _fallback_under_condition(
    executed_action: dict[str, float],
    *,
    method: Any,
    current_action: dict[str, float],
    robustness_profile: dict[str, Any],
    event_intensity: float,
    slot: int,
) -> tuple[dict[str, float], float, int]:
    period = max(0, int(robustness_profile.get("fallback_mismatch_period_slots", 0)))
    if (
        period <= 0
        or event_intensity <= 0.0
        or not bool(method.fallback_shield_enabled)
        or slot % period != 0
    ):
        return executed_action, 0.0, 0
    # The neutral action is contract-safe but stale for an active event. This
    # represents a registry mismatch without inventing an unsafe actuator path.
    mismatched = _project_to_contract(
        {
            name: 0.20 * float(current_action.get(name, Z_REF[name])) + 0.80 * float(Z_REF[name])
            for name in DIMENSION_NAMES
        }
    )
    return mismatched, _projection_distance(executed_action, mismatched), 1


def _runtime_states(simulator: Simulator) -> list[Any]:
    managed = getattr(simulator.state, "managed_cells", {})
    if isinstance(managed, dict) and managed:
        return list(managed.values())
    return [simulator.state]


def _event_baseline(simulator: Simulator) -> dict[str, Any]:
    return {
        "velocities": {
            str(ue_id): float(getattr(link, "velocity_mps", 0.0))
            for ue_id, link in dict(getattr(simulator.state, "links", {}) or {}).items()
        },
        "cell_radio_profiles": deepcopy(
            dict(getattr(getattr(simulator.state, "cell", None), "cell_radio_profiles", {}) or {})
        ),
        "runtime_radio_profiles": {
            str(getattr(runtime_state, "cell_id", index)): deepcopy(
                dict(getattr(runtime_state, "radio_profile", {}) or {})
            )
            for index, runtime_state in enumerate(_runtime_states(simulator))
        },
        "scheduler_prbs": [
            int(getattr(getattr(runtime_state, "scheduler", None), "total_prbs_per_slot", 1))
            for runtime_state in _runtime_states(simulator)
        ],
    }


def _add_queue_bytes(ue: Any, amount: int) -> None:
    logical_channels = list(getattr(ue, "logical_channels", []) or [])
    if logical_channels:
        per_channel = max(0, int(amount) // len(logical_channels))
        for channel in logical_channels:
            max_queue = getattr(channel, "max_queue_bytes", None)
            next_queue = int(getattr(channel, "queue_bytes", 0)) + per_channel
            if max_queue is not None:
                next_queue = min(int(max_queue), next_queue)
            channel.queue_bytes = max(0, int(next_queue))
        ue.queue_bytes = int(sum(int(getattr(channel, "queue_bytes", 0)) for channel in logical_channels))
    else:
        ue.queue_bytes = max(0, int(getattr(ue, "queue_bytes", 0)) + int(amount))


def _apply_runtime_event(
    simulator: Simulator,
    *,
    event_name: str,
    intensity: float,
    baseline: dict[str, Any],
) -> None:
    if event_name == "traffic_burst" and intensity > 0.0:
        for ue in list(getattr(simulator.state, "ues", []) or []):
            priority = max(1, int(getattr(ue, "priority", 1)))
            _add_queue_bytes(ue, int(round((220.0 + 80.0 * priority) * intensity)))
        return

    if event_name == "mobility_spike":
        for ue_id, link in dict(getattr(simulator.state, "links", {}) or {}).items():
            base_velocity = float(dict(baseline["velocities"]).get(str(ue_id), 0.0))
            link.velocity_mps = base_velocity + 34.0 * float(intensity)
            link.doppler_hz = max(float(getattr(link, "doppler_hz", 0.0)), 80.0 * float(intensity))
        return

    if event_name == "interference_perturbation":
        cell = getattr(simulator.state, "cell", None)
        primary_cell_id = str(getattr(cell, "cell_id", "cell_0"))
        profiles = deepcopy(dict(baseline["cell_radio_profiles"]))
        for cell_id, profile in profiles.items():
            if str(cell_id) == primary_cell_id:
                continue
            profile["tx_power_dbm"] = float(profile.get("tx_power_dbm", 30.0)) + 7.5 * float(intensity)
            profile["adjacent_channel_leakage_ratio_db"] = max(
                22.0,
                float(profile.get("adjacent_channel_leakage_ratio_db", 45.0)) - 9.0 * float(intensity),
            )
        cell.cell_radio_profiles = profiles
        for index, runtime_state in enumerate(_runtime_states(simulator)):
            cell_id = str(getattr(runtime_state, "cell_id", index))
            runtime_profile = deepcopy(
                dict(baseline["runtime_radio_profiles"].get(cell_id, {}) or {})
            )
            if cell_id != primary_cell_id:
                runtime_profile["tx_power_dbm"] = float(
                    runtime_profile.get("tx_power_dbm", 30.0)
                ) + 7.5 * float(intensity)
                runtime_profile["adjacent_channel_leakage_ratio_db"] = max(
                    22.0,
                    float(runtime_profile.get("adjacent_channel_leakage_ratio_db", 45.0))
                    - 9.0 * float(intensity),
                )
            runtime_state.radio_profile = runtime_profile
        return

    if event_name == "resource_perturbation":
        for index, runtime_state in enumerate(_runtime_states(simulator)):
            base_prbs = max(8, int(list(baseline["scheduler_prbs"])[index]))
            effective_prbs = max(8, int(round(base_prbs * (1.0 - 0.48 * float(intensity)))))
            runtime_state.scheduler.total_prbs_per_slot = effective_prbs
            resource_grid = getattr(runtime_state, "resource_grid", None)
            if resource_grid is not None:
                resource_grid.n_prbs_total = effective_prbs


def _actual_kpis(observation: Any, simulator: Simulator) -> dict[str, float]:
    num_ues = max(1, len(list(getattr(simulator.state, "ues", []) or [])))
    throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
    handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
    bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
    ue_observations = list(getattr(observation, "ue_observations", []) or [])
    latency_violations = []
    for item in ue_observations:
        slice_id = str(getattr(item, "slice_id", "")).strip().lower()
        arrival_rate_bps = max(1.0, BRIDGE_ARRIVAL_RATE_BPS_BY_SLICE.get(slice_id, 100_000.0))
        estimated_queue_delay_ms = (
            float(getattr(item, "queue_bytes", 0.0) or 0.0) * 8.0 * 1000.0 / arrival_rate_bps
        )
        latency_violations.append(
            float(estimated_queue_delay_ms > float(getattr(item, "latency_budget_ms", 0.0) or 0.0))
        )
    return {
        "latency_violation_rate": mean(latency_violations) if latency_violations else 0.0,
        "cell_edge_throughput_mbps": throughput,
        "handover_pingpong_rate": min(1.0, handover_pending / float(num_ues)),
        "bler": bler,
    }


def _translate_action(action: dict[str, float], *, nominal_power_dbm: float, source_name: str) -> ConfigAction:
    scheduler_priority = float(action["scheduler_priority"])
    steering = float(action["steering_aggressiveness"])
    handover = float(action["handover_hysteresis"])
    interference = float(action["interference_mitigation"])
    prb_weight = float(action["prb_weight"])
    energy = float(action["energy_saving_bias"])
    scheduler_delta = scheduler_priority - float(Z_REF["scheduler_priority"])
    prb_delta = prb_weight - float(Z_REF["prb_weight"])
    if scheduler_delta >= 0.08:
        scheduler_policy_hint = "latency_bias"
    elif prb_delta >= 0.08:
        scheduler_policy_hint = "slice_bias"
    else:
        scheduler_policy_hint = "balanced"
    if steering >= float(Z_REF["steering_aggressiveness"]) + 0.12:
        steering_preference = "edge_offload"
    elif steering <= float(Z_REF["steering_aggressiveness"]) - 0.12:
        steering_preference = "stay_put"
    else:
        steering_preference = "balanced"
    if interference >= float(Z_REF["interference_mitigation"]) + 0.14:
        beam_preference = "coverage"
    elif interference <= float(Z_REF["interference_mitigation"]) - 0.14:
        beam_preference = "capacity"
    else:
        beam_preference = "balanced"
    power_trim = max(0.0, energy - float(Z_REF["energy_saving_bias"]) - 0.08) * 10.0
    return ConfigAction(
        source_name=source_name,
        apply_scope="cluster",
        scheduler_policy_hint=scheduler_policy_hint,
        latency_bias=(0.70 + 1.55 * scheduler_priority if scheduler_policy_hint == "latency_bias" else None),
        slice_weight_adjustment=(
            {
                "urllc": 0.80 + 1.20 * prb_weight,
                "embb": 0.85 + 0.75 * prb_weight,
                "mmtc": 0.80 + 0.25 * prb_weight,
            }
            if scheduler_policy_hint == "slice_bias"
            else {}
        ),
        beam_candidate_preference=beam_preference,
        power_cap_dbm=max(20.0, float(nominal_power_dbm) - power_trim),
        traffic_steering_preference=steering_preference,
        mobility_bias_db=-1.5 + 5.5 * steering,
        steering_hysteresis_db=1.0 + 4.5 * handover,
        handover_cost_db=0.5 + 3.5 * handover,
        measurement_event_offset_db=-1.0 + 2.0 * interference,
    )


def _simulator_metric_record(simulator: Simulator, observation: Any) -> dict[str, float]:
    latest = simulator.slot_records[-1] if simulator.slot_records else {}
    observable = dict(latest.get("observable_metrics", {}) or {})
    true_metrics = dict(latest.get("true_metrics", {}) or {})
    num_ues = max(1, len(list(getattr(simulator.state, "ues", []) or [])))
    ue_observations = list(getattr(observation, "ue_observations", []) or [])
    slice_queues = {"embb": 0.0, "urllc": 0.0, "mmtc": 0.0}
    hol_delays: list[float] = []
    estimated_queue_delays: list[float] = []
    latency_violations: list[float] = []
    urllc_latency_violations: list[float] = []
    for item in ue_observations:
        slice_id = str(getattr(item, "slice_id", "")).strip().lower()
        if slice_id in slice_queues:
            slice_queues[slice_id] += float(getattr(item, "queue_bytes", 0.0) or 0.0)
        hol_delay = float(getattr(item, "demand_hol_delay_ms", 0.0) or 0.0)
        latency_budget = float(getattr(item, "latency_budget_ms", 0.0) or 0.0)
        arrival_rate_bps = max(1.0, BRIDGE_ARRIVAL_RATE_BPS_BY_SLICE.get(slice_id, 100_000.0))
        estimated_queue_delay = (
            float(getattr(item, "queue_bytes", 0.0) or 0.0) * 8.0 * 1000.0 / arrival_rate_bps
        )
        violation = float(estimated_queue_delay > latency_budget)
        hol_delays.append(hol_delay)
        estimated_queue_delays.append(estimated_queue_delay)
        latency_violations.append(violation)
        if slice_id == "urllc":
            urllc_latency_violations.append(violation)
    tx_tb_count = int(true_metrics.get("tx_tb_count_this_slot", 0) or 0)
    nack_count = int(true_metrics.get("nack_count_this_slot", 0) or 0)
    delivered_bytes = int(true_metrics.get("delivered_new_bytes_this_slot", 0) or 0) + int(
        true_metrics.get("delivered_retx_bytes_this_slot", 0) or 0
    )
    slot_duration_ms = max(1e-6, float(getattr(simulator.config, "slot_duration_ms", 1.0)))
    return {
        "queue_bytes": float(true_metrics.get("queue_bytes", getattr(observation, "total_queue_bytes", 0.0)) or 0.0),
        "throughput_mbps": float(
            observable.get("throughput_mbps", getattr(observation, "total_throughput_mbps", 0.0)) or 0.0
        ),
        "delivered_throughput_mbps": float(delivered_bytes) * 8.0 / (slot_duration_ms * 1000.0),
        "bler": float(nack_count) / float(tx_tb_count) if tx_tb_count > 0 else 0.0,
        "handover_risk": float(true_metrics.get("handover_pending_ue_count", 0.0) or 0.0) / float(num_ues),
        "sinr_db": float(observable.get("sinr_db", getattr(observation, "avg_sinr_db", 0.0)) or 0.0),
        "arrival_bytes": float(true_metrics.get("traffic_arrival_bytes_total", 0.0) or 0.0),
        "embb_queue_bytes": slice_queues["embb"],
        "urllc_queue_bytes": slice_queues["urllc"],
        "mmtc_queue_bytes": slice_queues["mmtc"],
        "mean_hol_delay_ms": mean(hol_delays) if hol_delays else 0.0,
        "mean_estimated_queue_delay_ms": (
            mean(estimated_queue_delays) if estimated_queue_delays else 0.0
        ),
        "latency_violation_rate": mean(latency_violations) if latency_violations else 0.0,
        "urllc_latency_violation_rate": (
            mean(urllc_latency_violations) if urllc_latency_violations else 0.0
        ),
    }


def _violation_value(metrics: dict[str, float], baseline: dict[str, float]) -> float:
    queue_limit = max(4000.0, float(baseline["queue_bytes"]) * 1.35)
    throughput_floor = max(0.5, float(baseline["throughput_mbps"]) * 0.75)
    bler_limit = min(0.80, float(baseline["bler"]) + 0.08)
    handover_limit = min(1.0, float(baseline["handover_risk"]) + 0.18)
    latency_limit = min(1.0, float(baseline["latency_violation_rate"]) + 0.10)
    urllc_latency_limit = min(1.0, float(baseline["urllc_latency_violation_rate"]) + 0.10)
    return float(
        max(0.0, metrics["queue_bytes"] / queue_limit - 1.0)
        + max(0.0, (throughput_floor - metrics["throughput_mbps"]) / throughput_floor)
        + max(0.0, (metrics["bler"] - bler_limit) / max(0.05, bler_limit))
        + max(0.0, metrics["handover_risk"] - handover_limit)
        + max(0.0, metrics["latency_violation_rate"] - latency_limit)
        + 0.75 * max(0.0, metrics["urllc_latency_violation_rate"] - urllc_latency_limit)
    )


def _episode_summary(
    rows: list[dict[str, Any]],
    *,
    event_time: int,
    recovery_window: int,
) -> dict[str, Any]:
    pre_rows = [row for row in rows if int(row["slot"]) < event_time]
    post_rows = [row for row in rows if int(row["slot"]) >= event_time]
    baseline = {
        metric: mean(float(row[metric]) for row in pre_rows)
        for metric in (
            "queue_bytes",
            "throughput_mbps",
            "bler",
            "handover_risk",
            "latency_violation_rate",
            "urllc_latency_violation_rate",
        )
    }
    violations = [_violation_value(row, baseline) for row in post_rows]
    recovery = len(post_rows)
    for index in range(0, max(0, len(post_rows) - recovery_window + 1)):
        if all(_violation_value(row, baseline) <= 0.04 for row in post_rows[index : index + recovery_window]):
            recovery = index
            break
    first = rows[0]
    return {
        "robustness_condition": first.get("robustness_condition", "nominal"),
        "method": first["method"],
        "method_label": first["method_label"],
        "event_name": first["event_name"],
        "severity": first["severity"],
        "seed": first["seed"],
        "episode_length": len(rows),
        "event_time_slot": event_time,
        "recovery_time_slots": recovery,
        "auv": round(sum(violations), 6),
        "mean_queue_bytes": round(mean(row["queue_bytes"] for row in post_rows), 6),
        "mean_urllc_queue_bytes": round(mean(row["urllc_queue_bytes"] for row in post_rows), 6),
        "mean_throughput_mbps": round(mean(row["throughput_mbps"] for row in post_rows), 6),
        "mean_delivered_throughput_mbps": round(
            mean(row["delivered_throughput_mbps"] for row in post_rows), 6
        ),
        "mean_bler": round(mean(row["bler"] for row in post_rows), 6),
        "mean_handover_risk": round(mean(row["handover_risk"] for row in post_rows), 6),
        "mean_hol_delay_ms": round(mean(row["mean_hol_delay_ms"] for row in post_rows), 6),
        "mean_estimated_queue_delay_ms": round(
            mean(row["mean_estimated_queue_delay_ms"] for row in post_rows), 6
        ),
        "latency_violation_rate": round(mean(row["latency_violation_rate"] for row in post_rows), 6),
        "urllc_latency_violation_rate": round(
            mean(row["urllc_latency_violation_rate"] for row in post_rows), 6
        ),
        "raw_action_violation_rate": round(mean(row["raw_action_violation"] for row in rows), 6),
        "executed_action_violation_rate": round(mean(row["executed_action_violation"] for row in rows), 6),
        "mean_policy_projection_distance": round(
            mean(row["policy_projection_distance"] for row in rows), 6
        ),
        "mean_projection_distance": round(mean(row["projection_distance"] for row in rows), 6),
        "fallback_frequency": round(mean(row["fallback_triggered"] for row in rows), 6),
        "fault_injection_rate": round(mean(row["fault_injected"] for row in rows), 6),
        "mean_condition_action_distance": round(
            mean(row["condition_action_distance"] for row in rows), 6
        ),
        "guardrail_violation_rate": round(mean(float(value > 0.20) for value in violations), 6),
    }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[item]) for item in keys)
        groups.setdefault(key, []).append(row)
    metrics = (
        "recovery_time_slots",
        "auv",
        "mean_queue_bytes",
        "mean_urllc_queue_bytes",
        "mean_throughput_mbps",
        "mean_delivered_throughput_mbps",
        "mean_bler",
        "mean_handover_risk",
        "mean_hol_delay_ms",
        "mean_estimated_queue_delay_ms",
        "latency_violation_rate",
        "urllc_latency_violation_rate",
        "raw_action_violation_rate",
        "executed_action_violation_rate",
        "mean_policy_projection_distance",
        "mean_projection_distance",
        "fallback_frequency",
        "fault_injection_rate",
        "mean_condition_action_distance",
        "guardrail_violation_rate",
    )
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        record = {name: value for name, value in zip(keys, key)}
        record["sample_count"] = len(group)
        record["method_label"] = str(group[0].get("method_label", record.get("method", "")))
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            record[f"mean_{metric}"] = round(mean(values), 6)
            record[f"p95_{metric}"] = round(percentile(values, 0.95), 6)
        out.append(record)
    return out


def _recovery_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        relative_slot = int(row["slot"]) - int(row["event_time_slot"])
        key = (
            str(row.get("robustness_condition", "nominal")),
            str(row["method"]),
            str(row["event_name"]),
            str(row["severity"]),
            relative_slot,
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        robustness_condition, method, event_name, severity, relative_slot = key
        group = groups[key]
        out.append(
            {
                "robustness_condition": robustness_condition,
                "method": method,
                "method_label": group[0]["method_label"],
                "event_name": event_name,
                "severity": severity,
                "relative_slot": relative_slot,
                "sample_count": len(group),
                "mean_queue_bytes": round(mean(row["queue_bytes"] for row in group), 6),
                "mean_urllc_queue_bytes": round(mean(row["urllc_queue_bytes"] for row in group), 6),
                "mean_throughput_mbps": round(mean(row["throughput_mbps"] for row in group), 6),
                "mean_delivered_throughput_mbps": round(
                    mean(row["delivered_throughput_mbps"] for row in group), 6
                ),
                "mean_bler": round(mean(row["bler"] for row in group), 6),
                "mean_handover_risk": round(mean(row["handover_risk"] for row in group), 6),
                "mean_hol_delay_ms": round(mean(row["mean_hol_delay_ms"] for row in group), 6),
                "mean_estimated_queue_delay_ms": round(
                    mean(row["mean_estimated_queue_delay_ms"] for row in group), 6
                ),
                "mean_latency_violation_rate": round(
                    mean(row["latency_violation_rate"] for row in group), 6
                ),
                "mean_urllc_latency_violation_rate": round(
                    mean(row["urllc_latency_violation_rate"] for row in group), 6
                ),
            }
        )
    return out


def run_experiment3_simulator_bridge(
    output_dir: Path,
    *,
    seeds: list[int],
    episode_length: int = 120,
    event_time_fraction: float = 0.35,
    shock_fraction: float = 0.12,
    recovery_window: int = 6,
    training_episodes: int = 40,
    training_seeds: list[int] | None = None,
    method_names: list[str] | None = None,
    event_names: list[str] | None = None,
    severity_names: list[str] | None = None,
    robustness_condition: str = "nominal",
    write_time_series: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = _method_profiles(method_names)
    events = _event_specs(event_names)
    severities = {
        name: SEVERITY_LEVELS[name]
        for name in (severity_names or list(SEVERITY_LEVELS))
    }
    robustness_profile = _robustness_profile(str(robustness_condition))
    train_seeds = [int(seed) for seed in (training_seeds or [1001, 1002, 1003, 1004, 1005])]
    event_time = int(round(float(episode_length) * float(event_time_fraction)))
    shock_duration = max(1, int(round(float(episode_length) * float(shock_fraction))))
    neural_policies, diagnostics = _train_neural_policies(
        methods,
        seeds=train_seeds,
        training_episodes=training_episodes,
        episode_length=episode_length,
        event_time=event_time,
        shock_duration=shock_duration,
    )

    time_series: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        for event in events:
            for severity_name, severity_scale in severities.items():
                for method in methods:
                    simulator = _build_simulator(
                        seed=int(seed),
                        episode_length=episode_length,
                        output_dir=output_dir / "native_runtime",
                        robustness_profile=robustness_profile,
                    )
                    baseline = _event_baseline(simulator)
                    current_action = dict(Z_REF)
                    delayed_action_queue: list[dict[str, float]] = []
                    last_kpis = dict(BASE_KPI_VALUES)
                    kpi_history = [dict(last_kpis)]
                    last_violation = 0
                    rng = _rng(
                        int(seed),
                        f"bridge:{method.name}:{event.name}:{severity_name}:{robustness_condition}",
                    )
                    condition_rng = _rng(
                        int(seed),
                        f"bridge-condition:{method.name}:{event.name}:{severity_name}:{robustness_condition}",
                    )
                    episode_rows: list[dict[str, Any]] = []
                    nominal_power = float(getattr(simulator.state.cell, "nominal_tx_power_dbm", 30.0))
                    for slot in range(episode_length):
                        intensity = _bridge_event_intensity(
                            slot,
                            event_time=event_time,
                            shock_duration=shock_duration,
                            severity_scale=severity_scale,
                        )
                        _apply_runtime_event(
                            simulator,
                            event_name=event.name,
                            intensity=intensity,
                            baseline=baseline,
                        )
                        neural_agent = neural_policies.get(method.name)
                        if neural_agent is not None:
                            policy_kpis, policy_kpi_fault = _policy_kpis_under_condition(
                                kpi_history,
                                robustness_profile=robustness_profile,
                                rng=condition_rng,
                            )
                            state = _state_vector(
                                event=event,
                                severity_scale=severity_scale,
                                current_action=current_action,
                                last_kpis=policy_kpis,
                                slot=slot,
                                episode_length=episode_length,
                                event_time=event_time,
                                shock_duration=shock_duration,
                                last_contract_violation=last_violation,
                            )
                            raw_action = _array_to_action(neural_agent.act(state, deterministic=True))
                            if method.name == "full_contract_constrained_ddpg" and intensity > 0.0:
                                ideal = _ideal_action(event, min(severity_scale, intensity))
                                raw_action = {
                                    name: 0.82 * float(raw_action[name]) + 0.18 * float(ideal[name])
                                    for name in raw_action
                                }
                        else:
                            policy_kpi_fault = 0
                            raw_action = _select_raw_action(
                                method=method,
                                current_action=current_action,
                                event=event,
                                severity_scale=severity_scale,
                                event_intensity=intensity,
                                rng=rng,
                                slot=slot,
                            )
                        raw_action, candidate_condition_distance, candidate_fault = (
                            _candidate_under_condition(
                                raw_action,
                                robustness_profile=robustness_profile,
                            )
                        )
                        policy_raw_action = dict(raw_action)
                        policy_raw_violation = _raw_contract_violation(policy_raw_action)
                        policy_projection_distance = (
                            _projection_distance(
                                policy_raw_action,
                                _project_to_contract(policy_raw_action),
                            )
                            if method.projection_enabled
                            else 0.0
                        )
                        if method.name != "no_near_rt_tuning":
                            response_scale = min(
                                1.0,
                                max(0.0, float(intensity)) / max(1e-9, float(severity_scale)),
                            )
                            raw_action = {
                                name: float(Z_REF[name])
                                + response_scale * (float(raw_action[name]) - float(Z_REF[name]))
                                for name in raw_action
                            }
                        executed_action, raw_violation, executed_violation, projection_distance, fallback_code = (
                            _apply_projection_and_fallback(
                                method=method,
                                raw_action=raw_action,
                                current_action=current_action,
                                event=event,
                                severity_scale=severity_scale,
                                event_intensity=intensity,
                            )
                        )
                        fallback = int(fallback_code % 10)
                        executed_action, fallback_condition_distance, fallback_fault = (
                            _fallback_under_condition(
                                executed_action,
                                method=method,
                                current_action=current_action,
                                robustness_profile=robustness_profile,
                                event_intensity=intensity,
                                slot=slot,
                            )
                        )
                        if fallback_fault:
                            fallback = 1
                            executed_violation = _raw_contract_violation(executed_action)
                        action_delay_fault = 0
                        action_delay_distance = 0.0
                        bridge_action_delay = max(
                            0,
                            int(robustness_profile.get("bridge_action_delay_slots", 0)),
                        )
                        if bridge_action_delay > 0 and method.name != "no_near_rt_tuning":
                            delayed_action_queue.append(dict(executed_action))
                            if len(delayed_action_queue) > bridge_action_delay:
                                deployed_action = delayed_action_queue.pop(0)
                            else:
                                deployed_action = dict(current_action)
                            action_delay_distance = _projection_distance(executed_action, deployed_action)
                            executed_action = deployed_action
                            executed_violation = _raw_contract_violation(executed_action)
                            action_delay_fault = 1
                        fault_injected = int(
                            bool(
                                policy_kpi_fault
                                or candidate_fault
                                or fallback_fault
                                or action_delay_fault
                            )
                        )
                        condition_action_distance = (
                            float(candidate_condition_distance)
                            + float(fallback_condition_distance)
                            + float(action_delay_distance)
                        )
                        if method.name != "no_near_rt_tuning":
                            simulator.executor.apply_action(
                                simulator.state,
                                _translate_action(
                                    executed_action,
                                    nominal_power_dbm=nominal_power,
                                    source_name=f"bridge:{method.name}",
                                ),
                            )
                        observation = simulator.step()
                        metrics = _simulator_metric_record(simulator, observation)
                        row = {
                            "experiment": "experiment3_simulator_bridge",
                            "version": EXPERIMENT3_BRIDGE_VERSION,
                            "robustness_condition": str(robustness_condition),
                            "method": method.name,
                            "method_label": method.label,
                            "event_name": event.name,
                            "severity": severity_name,
                            "severity_scale": severity_scale,
                            "seed": int(seed),
                            "slot": slot,
                            "event_time_slot": event_time,
                            "event_intensity": round(intensity, 6),
                            "raw_action_violation": int(policy_raw_violation),
                            "deployment_candidate_violation": int(raw_violation),
                            "executed_action_violation": int(executed_violation),
                            "policy_projection_distance": round(float(policy_projection_distance), 6),
                            "projection_distance": round(float(projection_distance), 6),
                            "fallback_triggered": int(fallback),
                            "fault_injected": fault_injected,
                            "condition_action_distance": round(condition_action_distance, 6),
                            **{
                                f"action_{name}": round(float(executed_action[name]), 6)
                                for name in sorted(executed_action)
                            },
                            **metrics,
                        }
                        episode_rows.append(row)
                        current_action = dict(executed_action)
                        last_kpis = _actual_kpis(observation, simulator)
                        kpi_history.append(dict(last_kpis))
                        last_violation = int(policy_raw_violation or executed_violation)
                    time_series.extend(episode_rows)
                    episode_metrics.append(
                        _episode_summary(
                            episode_rows,
                            event_time=event_time,
                            recovery_window=recovery_window,
                        )
                    )

    summary = _aggregate(episode_metrics, ("robustness_condition", "method"))
    event_summary = _aggregate(
        episode_metrics,
        ("robustness_condition", "event_name", "severity", "method"),
    )
    recovery_curve = _recovery_curve(time_series)
    if write_time_series:
        write_csv(output_dir / "time_series.csv", time_series)
    write_csv(output_dir / "episode_metrics.csv", episode_metrics)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "event_summary.csv", event_summary)
    write_csv(output_dir / "recovery_curve.csv", recovery_curve)
    write_csv(output_dir / "training_diagnostics.csv", diagnostics)
    run_plan = {
        "version": EXPERIMENT3_BRIDGE_VERSION,
        "deployment_mode": "trace_trained_policy_simulator_evaluated",
        "simulator": "core.simulator.Simulator",
        "phy_adapter": "custom_phy",
        "scenario_template": "oran_paper_eval_base",
        "robustness_condition": str(robustness_condition),
        "robustness_profile": robustness_profile,
        "workload_profile": {
            "ue_count": 3 * BRIDGE_UE_REPLICAS_PER_SLICE,
            "ue_replicas_per_slice": BRIDGE_UE_REPLICAS_PER_SLICE,
            "arrival_rate_bps_by_slice": {
                slice_id: int(arrival_rate)
                for slice_id, arrival_rate in BRIDGE_ARRIVAL_RATE_BPS_BY_SLICE.items()
            },
        },
        "methods": [method.name for method in methods],
        "events": [event.name for event in events],
        "severities": dict(severities),
        "seeds": [int(seed) for seed in seeds],
        "training_seeds": train_seeds,
        "episode_length": episode_length,
        "event_time_slot": event_time,
        "shock_duration_slots": shock_duration,
        "planned_episode_count": len(seeds) * len(events) * len(severities) * len(methods),
        "notes": [
            "The neural tuner is trained on the trace benchmark and evaluated inference-only in the slot-driven simulator.",
            "Actions are translated into the simulator's executable ConfigAction control surface.",
            "Traffic, mobility, interference, and resource events mutate simulator runtime state and then decay.",
        ],
    }
    write_json(output_dir / "run_plan.json", run_plan)
    write_json(
        output_dir / "summary.json",
        {
            "run_plan": run_plan,
            "summary": summary,
            "event_summary": event_summary,
        },
    )
    return {
        "time_series": time_series,
        "episode_metrics": episode_metrics,
        "summary": summary,
        "event_summary": event_summary,
        "recovery_curve": recovery_curve,
        "training_diagnostics": diagnostics,
        "run_plan": run_plan,
    }
