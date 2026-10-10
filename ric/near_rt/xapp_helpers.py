"""Helper utilities shared by Near-RT xApps."""

from __future__ import annotations

from typing import Iterable

from models.action import PolicyConstraint
from models.observation import CellObservation, Observation, UeObservation
from models.policy import PolicySnapshot


def clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    filtered = [float(value) for value in values if value is not None]
    if len(filtered) == 0:
        return float(default)
    return float(sum(filtered) / len(filtered))


def iter_ue_observations(observation: Observation | None) -> list[UeObservation]:
    if observation is None:
        return []
    return list(getattr(observation, "ue_observations", []) or [])


def iter_cell_observations(observation: Observation | None) -> list[CellObservation]:
    if observation is None:
        return []
    cells = list(getattr(observation, "cell_observations", []) or [])
    if len(cells) > 0:
        return cells
    return [
        CellObservation(
            cell_id=str(getattr(observation, "cell_id", "cell_0")),
            scheduler_mode=str(getattr(observation, "scheduler_mode", "round_robin")),
            tx_power_dbm=float(getattr(observation, "tx_power_dbm", 30.0)),
            carrier_frequency_hz=float(getattr(observation, "carrier_frequency_hz", 3.5e9) or 3.5e9),
            bandwidth_hz=float(getattr(observation, "bandwidth_hz", 20e6) or 20e6),
            subcarrier_spacing_hz=float(
                getattr(observation, "subcarrier_spacing_hz", 30e3) or 30e3
            ),
            total_queue_bytes=int(getattr(observation, "total_queue_bytes", 0)),
            connected_ue_count=int(getattr(observation, "connected_ue_count", 0)),
            scheduled_ue_count=int(getattr(observation, "num_scheduled_ues", 0)),
            resource_grid_total_re=int(getattr(observation, "resource_grid_total_re", 0) or 0),
            resource_grid_allocated_re=int(
                getattr(observation, "resource_grid_allocated_re", 0) or 0
            ),
            resource_grid_utilization=float(
                getattr(observation, "resource_grid_utilization", 0.0) or 0.0
            ),
            inter_cell_interference_penalty_db_mean=float(
                getattr(observation, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
            ),
        )
    ]


def ue_channel_quality_confidence(ue: UeObservation) -> float:
    confidence = getattr(ue, "grant_channel_quality_confidence", None)
    if confidence is None:
        confidence = getattr(ue, "demand_channel_quality_confidence", None)
    if confidence is None:
        return 0.5
    return clamp(float(confidence), 0.0, 1.0)


def ue_hol_delay_ms(ue: UeObservation) -> float:
    return float(getattr(ue, "demand_hol_delay_ms", 0.0) or 0.0)


def is_urllc_ue(ue: UeObservation) -> bool:
    slice_id = str(getattr(ue, "slice_id", "")).strip().lower()
    traffic_type = str(getattr(ue, "traffic_type", "")).strip().lower()
    latency_budget_ms = float(getattr(ue, "latency_budget_ms", 0.0) or 0.0)
    return (
        slice_id == "urllc"
        or traffic_type in {"control", "urllc", "phone", "voice", "telephony"}
        or (latency_budget_ms > 0.0 and latency_budget_ms <= 20.0)
    )


def is_embb_ue(ue: UeObservation) -> bool:
    slice_id = str(getattr(ue, "slice_id", "")).strip().lower()
    traffic_type = str(getattr(ue, "traffic_type", "")).strip().lower()
    return slice_id == "embb" or traffic_type in {"video", "broadband", "embb"}


def ue_demand_bytes(ue: UeObservation) -> int:
    demand_bytes = getattr(ue, "demand_bytes", None)
    if demand_bytes is None:
        demand_bytes = getattr(ue, "queue_bytes", 0)
    return int(max(0, int(demand_bytes)))


def policy_goal(policy_snapshot: PolicySnapshot | None):
    if policy_snapshot is None:
        return None
    return getattr(policy_snapshot, "structured_goal", None)


def goal_risk_posture(policy_snapshot: PolicySnapshot | None) -> str:
    goal = policy_goal(policy_snapshot)
    if goal is None:
        return "balanced"
    return str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()


def goal_kpi_weight(
    policy_snapshot: PolicySnapshot | None,
    kpi_name: str,
    default: float = 0.0,
) -> float:
    goal = policy_goal(policy_snapshot)
    if goal is None:
        return float(default)
    for target in list(getattr(goal, "kpi_targets", []) or []):
        if str(getattr(target, "kpi_name", "")).strip().lower() == str(kpi_name).strip().lower():
            return float(getattr(target, "weight", default) or default)
    return float(default)


def goal_prefers_latency(policy_snapshot: PolicySnapshot | None) -> bool:
    return (
        goal_kpi_weight(policy_snapshot, "latency", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "tail_latency", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "queue_bytes", 0.0) >= 1.0
    )


def goal_prefers_reliability(policy_snapshot: PolicySnapshot | None) -> bool:
    return (
        goal_kpi_weight(policy_snapshot, "avg_bler", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "urllc_reliability", 0.0) > 0.0
        or "reliability" in goal_risk_posture(policy_snapshot)
    )


def goal_prefers_throughput(policy_snapshot: PolicySnapshot | None) -> bool:
    return (
        goal_kpi_weight(policy_snapshot, "total_throughput_mbps", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "spectral_efficiency", 0.0) > 0.0
    )


def goal_prefers_energy(policy_snapshot: PolicySnapshot | None) -> bool:
    return (
        goal_kpi_weight(policy_snapshot, "power_cost", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "energy_efficiency", 0.0) > 0.0
    )


def goal_prefers_mobility(policy_snapshot: PolicySnapshot | None) -> bool:
    return (
        goal_kpi_weight(policy_snapshot, "handover_risk", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "handover_success", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "service_continuity", 0.0) > 0.0
    )


def goal_prefers_interference_control(policy_snapshot: PolicySnapshot | None) -> bool:
    return (
        goal_kpi_weight(policy_snapshot, "sinr", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "interference", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "interference_level", 0.0) > 0.0
        or goal_kpi_weight(policy_snapshot, "prb_utilization", 0.0) > 0.0
    )


def slice_weight(policy_snapshot: PolicySnapshot | None, slice_id: str, default: float = 1.0) -> float:
    weights = getattr(policy_snapshot, "preferred_slice_weight_adjustment", {}) if policy_snapshot is not None else {}
    if not isinstance(weights, dict):
        return float(default)
    return float(weights.get(str(slice_id).strip().lower(), default) or default)


def slice_delay_budget_ms(slice_id: str) -> float:
    normalized = str(slice_id or "").strip().lower()
    if normalized == "urllc":
        return 15.0
    if normalized == "mmtc":
        return 80.0
    if normalized == "embb":
        return 60.0
    return 50.0


def slice_pressures(
    observation: Observation | None,
    *,
    target_cell_id: str | None = None,
) -> dict[str, float]:
    pressures: dict[str, float] = {}
    for ue in iter_ue_observations(observation):
        if target_cell_id is not None and str(getattr(ue, "serving_cell_id", "")) != str(target_cell_id):
            continue
        slice_id = str(getattr(ue, "slice_id", "unknown")).strip().lower() or "unknown"
        queue_pressure = float(ue_demand_bytes(ue))
        latency_pressure = ue_hol_delay_ms(ue)
        harq_pressure = float(getattr(ue, "demand_harq_urgency_score", 0.0) or 0.0) * 300.0
        pressures[slice_id] = pressures.get(slice_id, 0.0) + queue_pressure + latency_pressure + harq_pressure
    return pressures


def dominant_slice_id(
    observation: Observation | None,
    *,
    target_cell_id: str | None = None,
) -> tuple[str, float]:
    pressures = slice_pressures(observation, target_cell_id=target_cell_id)
    if len(pressures) == 0:
        return ("unknown", 0.0)
    total_pressure = max(1.0, float(sum(pressures.values())))
    dominant_id, dominant_value = max(pressures.items(), key=lambda item: float(item[1]))
    return (str(dominant_id), float(dominant_value) / total_pressure)


def cell_load_score(cell: CellObservation) -> float:
    utilization = float(getattr(cell, "resource_grid_utilization", 0.0) or 0.0)
    queue_pressure = float(getattr(cell, "total_queue_bytes", 0.0) or 0.0) / 12000.0
    connected_pressure = float(getattr(cell, "connected_ue_count", 0) or 0) / 16.0
    return clamp(
        max(
            utilization,
            0.55 * utilization + 0.30 * queue_pressure + 0.15 * connected_pressure,
        ),
        0.0,
        2.0,
    )


def cell_interference_score(cell: CellObservation) -> float:
    return max(
        0.0,
        float(getattr(cell, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0),
    )


def select_busiest_cell(observation: Observation | None) -> CellObservation | None:
    cells = iter_cell_observations(observation)
    if len(cells) == 0:
        return None
    return max(cells, key=lambda cell: (cell_load_score(cell), float(getattr(cell, "total_queue_bytes", 0))))


def select_least_loaded_cell(observation: Observation | None) -> CellObservation | None:
    cells = iter_cell_observations(observation)
    if len(cells) == 0:
        return None
    return min(cells, key=lambda cell: (cell_load_score(cell), float(getattr(cell, "total_queue_bytes", 0))))


def select_urlcc_focus_cell(observation: Observation | None) -> CellObservation | None:
    cells = {str(cell.cell_id): cell for cell in iter_cell_observations(observation)}
    if len(cells) == 0:
        return None
    pressure_by_cell: dict[str, float] = {cell_id: 0.0 for cell_id in cells}
    for ue in iter_ue_observations(observation):
        if not is_urllc_ue(ue):
            continue
        cell_id = str(getattr(ue, "serving_cell_id", getattr(observation, "cell_id", "cell_0")))
        pressure_by_cell[cell_id] = pressure_by_cell.get(cell_id, 0.0) + ue_latency_pressure_score(ue)
    target_cell_id = max(pressure_by_cell.items(), key=lambda item: float(item[1]))[0]
    return cells.get(target_cell_id, next(iter(cells.values())))


def select_beam_stressed_cell(observation: Observation | None) -> CellObservation | None:
    cells = {str(cell.cell_id): cell for cell in iter_cell_observations(observation)}
    if len(cells) == 0:
        return None
    stress_by_cell: dict[str, float] = {cell_id: 0.0 for cell_id in cells}
    for ue in iter_ue_observations(observation):
        cell_id = str(getattr(ue, "serving_cell_id", getattr(observation, "cell_id", "cell_0")))
        stress_by_cell[cell_id] = stress_by_cell.get(cell_id, 0.0) + beam_misalignment_risk_score(ue)
    target_cell_id = max(stress_by_cell.items(), key=lambda item: float(item[1]))[0]
    return cells.get(target_cell_id, next(iter(cells.values())))


def best_non_serving_candidate(ue: UeObservation):
    candidates = list(getattr(ue, "candidate_cell_observations", []) or [])
    non_serving = [candidate for candidate in candidates if not bool(getattr(candidate, "is_serving", False))]
    if len(non_serving) == 0:
        return None
    return max(non_serving, key=lambda candidate: float(getattr(candidate, "steering_score_db", -100.0)))


def candidate_cell_ranking(ue: UeObservation, *, top_k: int = 3) -> list[str]:
    candidates = sorted(
        list(getattr(ue, "candidate_cell_observations", []) or []),
        key=lambda candidate: float(getattr(candidate, "steering_score_db", -100.0)),
        reverse=True,
    )
    return [str(candidate.cell_id) for candidate in candidates[:top_k]]


def ue_candidate_gain_db(ue: UeObservation) -> float:
    candidates = list(getattr(ue, "candidate_cell_observations", []) or [])
    serving_candidates = [candidate for candidate in candidates if bool(getattr(candidate, "is_serving", False))]
    serving_score = (
        float(getattr(serving_candidates[0], "steering_score_db", -100.0))
        if len(serving_candidates) > 0
        else float(getattr(ue, "sinr_db", 0.0))
    )
    best_alt = best_non_serving_candidate(ue)
    if best_alt is None:
        return 0.0
    return float(getattr(best_alt, "steering_score_db", -100.0)) - serving_score


def mobility_risk_score(ue: UeObservation) -> float:
    candidate_gain = max(0.0, ue_candidate_gain_db(ue)) / 8.0
    stale_measurement = float(getattr(ue, "measurement_age_slots", 0) or 0) / 8.0
    handover_pending = 1.0 if getattr(ue, "handover_pending_target_cell_id", None) is not None else 0.0
    low_signal = max(0.0, (8.0 - float(getattr(ue, "sinr_db", 0.0) or 0.0)) / 8.0)
    interference = max(
        0.0,
        float(getattr(ue, "inter_cell_interference_penalty_db", 0.0) or 0.0) / 6.0,
    )
    confidence_penalty = 1.0 - ue_channel_quality_confidence(ue)
    risk = (
        0.28 * candidate_gain
        + 0.20 * stale_measurement
        + 0.18 * handover_pending
        + 0.18 * low_signal
        + 0.10 * interference
        + 0.06 * confidence_penalty
    )
    return clamp(risk, 0.0, 1.0)


def beam_misalignment_risk_score(ue: UeObservation) -> float:
    low_gain = max(0.0, (6.0 - float(getattr(ue, "beamforming_gain_db", 0.0) or 0.0)) / 8.0)
    low_post_eq = max(
        0.0,
        (10.0 - float(getattr(ue, "post_eq_sinr_db", getattr(ue, "sinr_db", 0.0)) or 0.0)) / 10.0,
    )
    low_precoder_match = 1.0 - clamp(float(getattr(ue, "precoder_match_score", 0.5) or 0.5), 0.0, 1.0)
    stale_csi = max(0.0, float(getattr(ue, "grant_cqi_validity_age_slots", 0) or 0) / 8.0)
    mobility_risk = mobility_risk_score(ue)
    risk = (
        0.24 * low_gain
        + 0.24 * low_post_eq
        + 0.18 * low_precoder_match
        + 0.14 * stale_csi
        + 0.20 * mobility_risk
    )
    return clamp(risk, 0.0, 1.0)


def future_cqi_estimate(ue: UeObservation) -> float:
    current_cqi = float(getattr(ue, "cqi", 8) or 8)
    age_penalty = 0.25 * float(getattr(ue, "grant_cqi_validity_age_slots", 0) or 0)
    interference_penalty = 0.40 * float(
        getattr(ue, "inter_cell_interference_penalty_db", 0.0) or 0.0
    )
    confidence_bonus = 2.0 * ue_channel_quality_confidence(ue)
    return clamp(current_cqi - age_penalty - interference_penalty + confidence_bonus, 1.0, 15.0)


def ue_latency_pressure_score(ue: UeObservation) -> float:
    queue_pressure = float(ue_demand_bytes(ue)) / 2500.0
    budget_ms = float(getattr(ue, "latency_budget_ms", 0.0) or 0.0)
    hol_delay_ms = ue_hol_delay_ms(ue)
    latency_ratio = (hol_delay_ms / budget_ms) if budget_ms > 0.0 else 0.0
    harq_pressure = float(getattr(ue, "demand_harq_urgency_score", 0.0) or 0.0)
    bler_pressure = float(getattr(ue, "bler", 0.0) or 0.0) * 4.0
    urgency = 0.35 if is_urllc_ue(ue) else 0.0
    score = 0.30 * queue_pressure + 0.35 * latency_ratio + 0.20 * harq_pressure + 0.15 * bler_pressure + urgency
    return clamp(score, 0.0, 2.0)


def make_analysis_constraint(
    *,
    constraint_type: str,
    source_name: str,
    params: dict[str, object],
    scope: str = "cluster",
    target_id: str | None = None,
) -> PolicyConstraint:
    return PolicyConstraint(
        constraint_type=constraint_type,
        scope=scope,
        target_id=target_id,
        params=dict(params),
        source_name=source_name,
    )
