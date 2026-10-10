"""Experiment 2: candidate repair and guidance-contract compilation.

This module evaluates the Non-RT middle layer between a schema-bound planner
and Near-RT execution.  It keeps the original small candidate-repair benchmark
API, but the default config can now exercise the paper-scale Exp. 2 pool:
18 micro chunks x 3 intents x 3 scenario descriptors x H candidates, with the
same candidate pool passed through ablations and literature-inspired baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from experiments.paper_eval_artifacts import configure_matplotlib
from models.intent_orchestration import GoalKpiTarget, RewardParameterRange, StructuredIntentGoal
from models.observation import Observation
from ric.near_rt.xapp_variant_catalog import (
    BASE_XAPP_CAPABILITIES,
    DEFAULT_XAPP_VARIANTS,
    canonical_xapp_category,
)
from ric.non_rt.orchestration_feasibility import OrchestrationFeasibilityProjector


REPAIR_EVAL_VERSION = "candidate_repair_eval_v2_exp2_contract"

LEGACY_REPAIR_MODES = ["deterministic", "reject_only", "schema_only", "no_projection"]
EXP2_REPAIR_MODES = [
    "schema_only",
    "repair_wo_memory",
    "repair_wo_l3",
    "full_repair",
    "orchestran_like",
    "oreo_like",
    "pacifista_like",
    "conservative_fallback",
]
DEFAULT_REPAIR_MODES = EXP2_REPAIR_MODES
ALL_REPAIR_MODES = sorted(set(LEGACY_REPAIR_MODES + EXP2_REPAIR_MODES))

METHOD_LABELS = {
    "schema_only": "Schema-only",
    "repair_wo_memory": "Repair w/o memory",
    "repair_wo_l3": "Repair w/o L3",
    "full_repair": "Full repair",
    "orchestran_like": "OrchestRAN",
    "oreo_like": "OREO",
    "pacifista_like": "PACIFISTA",
    "conservative_fallback": "Conservative fallback",
    "deterministic": "Deterministic repair",
    "reject_only": "Reject-only",
    "no_projection": "No projection",
}


@dataclass(frozen=True)
class CandidateRepairFixture:
    candidate_id: str
    issue_type: str
    requested_xapps: list[str] = field(default_factory=list)
    notes: str = ""
    requested_bounds: list[RewardParameterRange] = field(default_factory=list)
    chunk_id: str = "legacy"
    intent_id: str = "intent:candidate_repair"
    intent_slug: str = "candidate-repair"
    intent_family: str = "repair_audit"
    scenario_id: str = "scenario:legacy"
    scenario_tags: list[str] = field(default_factory=list)
    candidate_source: str = "fixture"
    stress_error_types: list[str] = field(default_factory=list)


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reward_range_to_record(parameter_range: RewardParameterRange) -> dict[str, object]:
    return {
        "field_name": str(getattr(parameter_range, "field_name", "") or ""),
        "target_xapp_name": str(getattr(parameter_range, "target_xapp_name", "") or ""),
        "scope": str(getattr(parameter_range, "scope", "") or "cluster"),
        "lower_bound": getattr(parameter_range, "lower_bound", None),
        "upper_bound": getattr(parameter_range, "upper_bound", None),
        "step": getattr(parameter_range, "step", None),
        "discrete_values": list(getattr(parameter_range, "discrete_values", []) or []),
        "blocked_discrete_values": list(getattr(parameter_range, "blocked_discrete_values", []) or []),
        "max_delta": getattr(parameter_range, "max_delta", None),
        "tuning_mode": str(getattr(parameter_range, "tuning_mode", "") or "heuristic_parameter"),
    }


def reward_range_from_record(record: dict[str, object]) -> RewardParameterRange:
    return RewardParameterRange(
        field_name=str(record.get("field_name", "") or ""),
        target_xapp_name=str(record.get("target_xapp_name", "") or "") or None,
        scope=str(record.get("scope", "") or "cluster"),
        lower_bound=_optional_float(record.get("lower_bound")),
        upper_bound=_optional_float(record.get("upper_bound")),
        step=_optional_float(record.get("step")),
        discrete_values=list(record.get("discrete_values", []) or []),
        blocked_discrete_values=list(record.get("blocked_discrete_values", []) or []),
        max_delta=_optional_float(record.get("max_delta")),
        tuning_mode=str(record.get("tuning_mode", "") or "heuristic_parameter"),
    )


def candidate_fixture_to_record(fixture: CandidateRepairFixture) -> dict[str, object]:
    return {
        "candidate_id": fixture.candidate_id,
        "issue_type": fixture.issue_type,
        "requested_xapps": list(fixture.requested_xapps or []),
        "requested_bounds": [reward_range_to_record(item) for item in list(fixture.requested_bounds or [])],
        "chunk_id": fixture.chunk_id,
        "intent_id": fixture.intent_id,
        "intent_slug": fixture.intent_slug,
        "intent_family": fixture.intent_family,
        "scenario_id": fixture.scenario_id,
        "scenario_tags": list(fixture.scenario_tags or []),
        "candidate_source": fixture.candidate_source,
        "stress_error_types": list(fixture.stress_error_types or []),
        "notes": fixture.notes,
    }


def candidate_fixture_from_record(record: dict[str, object]) -> CandidateRepairFixture:
    return CandidateRepairFixture(
        candidate_id=str(record.get("candidate_id", "") or ""),
        issue_type=str(record.get("issue_type", "") or "llm_candidate"),
        requested_xapps=[str(item).strip() for item in list(record.get("requested_xapps", []) or []) if str(item).strip() != ""],
        requested_bounds=[
            reward_range_from_record(dict(item))
            for item in list(record.get("requested_bounds", []) or [])
            if isinstance(item, dict)
        ],
        chunk_id=str(record.get("chunk_id", "") or "llm"),
        intent_id=str(record.get("intent_id", "") or "intent:llm_candidate"),
        intent_slug=str(record.get("intent_slug", "") or "llm-candidate"),
        intent_family=str(record.get("intent_family", "") or "llm_candidate"),
        scenario_id=str(record.get("scenario_id", "") or "scenario:llm"),
        scenario_tags=[str(item).strip() for item in list(record.get("scenario_tags", []) or []) if str(item).strip() != ""],
        candidate_source=str(record.get("candidate_source", "") or "llm_candidate"),
        stress_error_types=[
            str(item).strip() for item in list(record.get("stress_error_types", []) or []) if str(item).strip() != ""
        ],
        notes=str(record.get("notes", "") or ""),
    )


def write_candidate_fixture_jsonl(path: Path, fixtures: list[CandidateRepairFixture]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for fixture in list(fixtures or []):
            jsonl_file.write(json.dumps(candidate_fixture_to_record(fixture), sort_keys=True) + "\n")


def load_candidate_fixture_jsonl(path: Path) -> list[CandidateRepairFixture]:
    fixtures: list[CandidateRepairFixture] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            stripped = line.strip()
            if stripped == "":
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid candidate JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object candidate at {path}:{line_number}")
            fixtures.append(candidate_fixture_from_record(payload))
    return fixtures


@dataclass(frozen=True)
class Exp2ChunkDescriptor:
    chunk_id: str
    name: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Exp2IntentTemplate:
    intent_slug: str
    intent_id: str
    intent_family: str
    intent_text: str
    kpi_targets: list[GoalKpiTarget] = field(default_factory=list)
    risk_posture: str = "balanced"


@dataclass(frozen=True)
class Exp2ScenarioDescriptor:
    scenario_id: str
    suffix: str
    tags: list[str] = field(default_factory=list)
    observation_overrides: dict[str, object] = field(default_factory=dict)


@dataclass
class FeasibilityCheck:
    violation_types: list[str] = field(default_factory=list)
    executable: bool = False
    hard_conflict: bool = False
    resource_overflow: bool = False
    empty_bound: bool = False
    unsafe_bounds: bool = False
    unsupported_parameter: bool = False
    policy_violation: bool = False
    risk_conflict: bool = False
    soft_conflict: bool = False


@dataclass
class MethodPlanOutput:
    output_xapps: list[str] = field(default_factory=list)
    output_bounds: list[RewardParameterRange] = field(default_factory=list)
    candidate_rejected: bool = False
    projected: bool = False
    fallback_used: bool = False
    projection_notes: list[str] = field(default_factory=list)
    parameter_bound_repaired: bool = False
    soft_arbitration_applied: bool = False
    risk_bound_tightened: bool = False
    l2_penalty_applied: bool = False
    raw_action: str = ""


@dataclass
class ContractMetrics:
    success: bool = False
    projectable_constraint_completeness: float = 0.0
    kpi_guardrail_coverage: float = 0.0
    fallback_profile_valid: bool = False
    admissible_region_volume_ratio: float = 0.0
    over_restriction: bool = False
    inconsistency: bool = False
    fallback_rate_sanity: float = 0.0
    protected_kpi_uncontrolled: bool = False
    near_rt_action_blocked_rate: float = 0.0


@dataclass
class CandidateRepairModeResult:
    mode: str
    candidate_id: str
    issue_type: str
    requested_xapps: list[str]
    output_xapps: list[str]
    raw_violation_types: list[str]
    output_violation_types: list[str]
    schema_valid: bool
    initially_feasible: bool
    final_feasible: bool
    candidate_rejected: bool
    projected: bool
    repair_success: bool
    hard_conflict_escape: bool
    resource_overflow_escape: bool
    unknown_xapp_escape: bool
    average_repair_distance: float
    repair_latency_ms: float
    projection_notes: list[str] = field(default_factory=list)
    requested_bounds: list[RewardParameterRange] = field(default_factory=list)
    output_bounds: list[RewardParameterRange] = field(default_factory=list)
    chunk_id: str = "legacy"
    scenario_id: str = "scenario:legacy"
    scenario_tags: list[str] = field(default_factory=list)
    intent_id: str = "intent:candidate_repair"
    intent_slug: str = "candidate-repair"
    intent_family: str = "repair_audit"
    candidate_source: str = "fixture"
    stress_error_types: list[str] = field(default_factory=list)
    fallback_used: bool = False
    unsupported_parameter_residual: bool = False
    empty_bound_residual: bool = False
    unsafe_bound_residual: bool = False
    policy_violation_residual: bool = False
    soft_conflict_present: bool = False
    soft_arbitration_applied: bool = False
    risk_conflict_present: bool = False
    risk_bound_tightened: bool = False
    l2_penalty_applied: bool = False
    parameter_bound_repaired: bool = False
    protected_kpi_coverage: float = 0.0
    intent_coverage: float = 0.0
    predicted_intent_loss: float = 1.0
    utility_preserving_executable: bool = False
    contract_success: bool = False
    projectable_constraint_completeness: float = 0.0
    kpi_guardrail_coverage: float = 0.0
    fallback_profile_valid: bool = False
    admissible_region_volume_ratio: float = 0.0
    over_restriction: bool = False
    contract_inconsistency: bool = False
    sanity_fallback_rate: float = 0.0
    protected_kpi_uncontrolled: bool = False
    near_rt_action_blocked_rate: float = 0.0
    outcome_bucket: str = "unknown"

    def to_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "method_label": METHOD_LABELS.get(self.mode, self.mode),
            "candidate_id": self.candidate_id,
            "chunk_id": self.chunk_id,
            "scenario_id": self.scenario_id,
            "scenario_tags": "|".join(self.scenario_tags),
            "intent_id": self.intent_id,
            "intent_slug": self.intent_slug,
            "intent_family": self.intent_family,
            "candidate_source": self.candidate_source,
            "issue_type": self.issue_type,
            "stress_error_types": "|".join(self.stress_error_types),
            "requested_xapps": "|".join(self.requested_xapps),
            "output_xapps": "|".join(self.output_xapps),
            "requested_bounds": _bounds_to_compact_text(self.requested_bounds),
            "output_bounds": _bounds_to_compact_text(self.output_bounds),
            "raw_violation_types": "|".join(self.raw_violation_types),
            "output_violation_types": "|".join(self.output_violation_types),
            "schema_valid": int(self.schema_valid),
            "initially_feasible": int(self.initially_feasible),
            "final_feasible": int(self.final_feasible),
            "candidate_rejected": int(self.candidate_rejected),
            "projected": int(self.projected),
            "fallback_used": int(self.fallback_used),
            "repair_success": int(self.repair_success),
            "hard_conflict_escape": int(self.hard_conflict_escape),
            "resource_overflow_escape": int(self.resource_overflow_escape),
            "unknown_xapp_escape": int(self.unknown_xapp_escape),
            "unsupported_parameter_residual": int(self.unsupported_parameter_residual),
            "empty_bound_residual": int(self.empty_bound_residual),
            "unsafe_bound_residual": int(self.unsafe_bound_residual),
            "policy_violation_residual": int(self.policy_violation_residual),
            "soft_conflict_present": int(self.soft_conflict_present),
            "soft_arbitration_applied": int(self.soft_arbitration_applied),
            "risk_conflict_present": int(self.risk_conflict_present),
            "risk_bound_tightened": int(self.risk_bound_tightened),
            "l2_penalty_applied": int(self.l2_penalty_applied),
            "parameter_bound_repaired": int(self.parameter_bound_repaired),
            "protected_kpi_coverage": round(float(self.protected_kpi_coverage), 6),
            "intent_coverage": round(float(self.intent_coverage), 6),
            "predicted_intent_loss": round(float(self.predicted_intent_loss), 6),
            "utility_preserving_executable": int(self.utility_preserving_executable),
            "contract_success": int(self.contract_success),
            "projectable_constraint_completeness": round(float(self.projectable_constraint_completeness), 6),
            "kpi_guardrail_coverage": round(float(self.kpi_guardrail_coverage), 6),
            "fallback_profile_valid": int(self.fallback_profile_valid),
            "admissible_region_volume_ratio": round(float(self.admissible_region_volume_ratio), 6),
            "over_restriction": int(self.over_restriction),
            "contract_inconsistency": int(self.contract_inconsistency),
            "sanity_fallback_rate": round(float(self.sanity_fallback_rate), 6),
            "protected_kpi_uncontrolled": int(self.protected_kpi_uncontrolled),
            "near_rt_action_blocked_rate": round(float(self.near_rt_action_blocked_rate), 6),
            "average_repair_distance": round(float(self.average_repair_distance), 6),
            "repair_latency_ms": round(float(self.repair_latency_ms), 6),
            "projection_notes": "|".join(self.projection_notes),
            "outcome_bucket": self.outcome_bucket,
        }


PARAM_POLICY_LIBRARY: dict[str, dict[str, object]] = {
    "latency_bias": {"lower": 0.95, "upper": 1.85, "step": 0.05},
    "power_cap_dbm": {"lower": 27.0, "upper": 30.5, "step": 0.25},
    "mobility_bias_db": {"lower": 0.0, "upper": 4.5, "step": 0.25},
    "measurement_event_offset_db": {"lower": 0.5, "upper": 2.5, "step": 0.25},
    "steering_hysteresis_db": {"lower": 2.0, "upper": 5.0, "step": 0.25, "l3_lower": 3.0},
    "steering_aggressiveness": {"lower": 0.20, "upper": 0.80, "step": 0.05, "l3_upper": 0.60},
    "load_threshold": {"lower": 0.50, "upper": 0.88, "step": 0.02},
    "cio_bias_db": {"lower": 0.0, "upper": 3.5, "step": 0.25},
    "slice_weight_adjustment": {"lower": 0.85, "upper": 1.35, "step": 0.05},
    "prb_weight": {"lower": 0.80, "upper": 1.55, "step": 0.05},
    "priority_coefficient": {"lower": 1.0, "upper": 2.2, "step": 0.1},
    "delay_threshold_ms": {"lower": 10.0, "upper": 28.0, "step": 1.0},
    "muting_ratio": {"lower": 0.04, "upper": 0.22, "step": 0.02, "l3_lower": 0.10},
    "coordination_threshold": {"lower": 1.2, "upper": 3.5, "step": 0.1},
    "sleep_threshold": {"lower": 0.18, "upper": 0.40, "step": 0.02},
    "wake_up_threshold": {"lower": 0.38, "upper": 0.58, "step": 0.02},
    "dwell_time_slots": {"lower": 20.0, "upper": 55.0, "step": 5.0},
    "scheduler_policy_hint": {"values": ["balanced", "latency_bias", "slice_bias"]},
    "beam_candidate_preference": {"values": ["balanced", "coverage", "capacity"]},
    "traffic_steering_preference": {"values": ["balanced", "edge_offload", "stay_put"]},
    "cell_sleep_mode": {"values": ["active", "micro_sleep"]},
}

SAFE_FALLBACK_CANDIDATE = [
    "urllc_guard_xapp_balanced",
    "latency_scheduler_xapp_balanced",
]

CONTROLLED_STRESS_ERRORS = [
    "unregistered_xapp",
    "unsupported_parameter",
    "missing_dependency",
    "hard_actuator_conflict",
    "soft_kpi_conflict",
    "risk_conflict",
    "resource_overflow",
    "unsafe_bounds",
]


def build_default_available_xapps() -> list[str]:
    names: list[str] = []
    for variants in DEFAULT_XAPP_VARIANTS.values():
        for variant in list(variants or []):
            name = str(variant.get("name", "") or "").strip()
            if name != "":
                names.append(name)
    return sorted(set(names))


def build_default_candidate_fixtures() -> list[CandidateRepairFixture]:
    """Small legacy fixture pool used by the original unit tests."""

    return [
        CandidateRepairFixture(
            candidate_id="valid_balanced_throughput_reliability",
            issue_type="valid",
            requested_xapps=[
                "mobility_predictor_xapp_balanced",
                "traffic_steering_xapp_balanced",
                "latency_scheduler_xapp_balanced",
                "urllc_guard_xapp_balanced",
            ],
            notes="Already executable plan for throughput plus reliability.",
        ),
        CandidateRepairFixture(
            candidate_id="unknown_xapp",
            issue_type="unknown_xapp",
            requested_xapps=[
                "traffic_steering_xapp_balanced",
                "nonexistent_magic_scheduler_xapp",
                "urllc_guard_xapp_balanced",
            ],
            notes="Hallucinated xApp name should be dropped or rejected.",
        ),
        CandidateRepairFixture(
            candidate_id="missing_dependency",
            issue_type="missing_dependency",
            requested_xapps=["traffic_steering_xapp_hotspot_relief", "urllc_guard_xapp_balanced"],
            notes="Traffic steering depends on mobility prediction.",
        ),
        CandidateRepairFixture(
            candidate_id="hard_mutex_sleep_vs_urllc",
            issue_type="hard_actuator_conflict",
            requested_xapps=["cell_sleep_xapp_balanced", "power_saving_xapp_balanced", "urllc_guard_xapp_strict"],
            notes="Cell sleep conflicts with URLLC guard.",
        ),
        CandidateRepairFixture(
            candidate_id="hard_mutex_sleep_vs_latency",
            issue_type="hard_actuator_conflict",
            requested_xapps=[
                "cell_sleep_xapp_aggressive",
                "power_saving_xapp_balanced",
                "latency_scheduler_xapp_urllc_fastpath",
            ],
            notes="Cell sleep conflicts with latency scheduler.",
        ),
        CandidateRepairFixture(
            candidate_id="resource_overflow_many_xapps",
            issue_type="resource_overflow",
            requested_xapps=[
                "mobility_predictor_xapp_high_speed",
                "traffic_steering_xapp_hotspot_relief",
                "latency_scheduler_xapp_urllc_fastpath",
                "urllc_guard_xapp_strict",
                "massive_mimo_beamformer_xapp_capacity_push",
                "spectrum_optimizer_xapp_aggressive",
                "ran_slicing_manager_xapp_sla_strict",
            ],
            notes="Too many active xApps for cardinality and CPU/memory budgets.",
        ),
        CandidateRepairFixture(
            candidate_id="duplicate_category_variants",
            issue_type="duplicate_category",
            requested_xapps=[
                "massive_mimo_beamformer_xapp_capacity_push",
                "massive_mimo_beamformer_xapp_coverage_boost",
                "traffic_steering_xapp_edge_rescue",
                "traffic_steering_xapp_quality_first",
                "urllc_guard_xapp_balanced",
            ],
            notes="Several variants map to the same xApp category.",
        ),
        CandidateRepairFixture(
            candidate_id="qos_guard_missing",
            issue_type="soft_qos_conflict",
            requested_xapps=["massive_mimo_beamformer_xapp_capacity_push", "spectrum_optimizer_xapp_aggressive"],
            notes="Throughput-heavy plan lacks reliability protection for the goal.",
        ),
        CandidateRepairFixture(
            candidate_id="policy_energy_vs_latency",
            issue_type="policy_conflict",
            requested_xapps=[
                "cell_sleep_xapp_deep_saver",
                "power_saving_xapp_energy_max",
                "latency_scheduler_xapp_tail_guard",
                "urllc_guard_xapp_tail_safe",
            ],
            notes="Energy-saving plan violates protected low-latency policy.",
        ),
        CandidateRepairFixture(
            candidate_id="empty_candidate",
            issue_type="schema_invalid",
            requested_xapps=[],
            notes="Empty candidate should not be executed.",
        ),
    ]


def build_default_goal() -> StructuredIntentGoal:
    return StructuredIntentGoal(
        goal_id="goal:candidate_repair",
        intent_id="intent:candidate_repair",
        intent_text="Improve cell-edge throughput while protecting low-latency users.",
        summary="Throughput-reliability intent used for candidate repair evaluation.",
        horizon_slots=20,
        risk_posture="conservative",
        target_entities=["cluster"],
        kpi_targets=[
            GoalKpiTarget(
                kpi_name="total_throughput_mbps",
                objective="maximize",
                target_value=45.0,
                weight=1.2,
                unit="Mbps",
                priority=1,
            ),
            GoalKpiTarget(
                kpi_name="avg_bler",
                objective="minimize",
                target_value=0.10,
                weight=1.4,
                priority=0,
            ),
            GoalKpiTarget(
                kpi_name="scheduler_hol_delay_ms",
                objective="minimize",
                target_value=20.0,
                weight=1.1,
                unit="ms",
                priority=0,
            ),
        ],
    )


def build_default_observation() -> Observation:
    return Observation(
        slot_id=100,
        avg_sinr_db=8.5,
        avg_bler=0.16,
        total_queue_bytes=82000,
        total_throughput_mbps=24.0,
        cell_id="cell_0",
        current_codebook_id=0,
        current_beam_profile_id=0,
        tx_power_dbm=30.0,
        scheduler_mode="round_robin",
        connected_ue_count=42,
        handover_pending_ue_count=6,
        scheduler_stale_csi_count=9,
        resource_grid_utilization=0.88,
        scheduled_ue_ids=[],
        num_scheduled_ues=0,
        ue_observations=[],
    )


def build_exp2_chunk_descriptors() -> list[Exp2ChunkDescriptor]:
    return [
        Exp2ChunkDescriptor("chunk001_mobility", "high mobility handover", ["high_mobility", "handover_edge"]),
        Exp2ChunkDescriptor("chunk002_interference", "interference edge", ["interference_burst", "cell_edge_users"]),
        Exp2ChunkDescriptor("chunk003_stale", "stale CSI memory", ["stale_memory", "stale_csi"]),
        Exp2ChunkDescriptor("chunk004_hotspot", "hotspot shift", ["hotspot_shift", "dense_hotspot"]),
        Exp2ChunkDescriptor("chunk005_blockage", "blockage recovery", ["blockage", "beam_recovery"]),
        Exp2ChunkDescriptor("chunk006_energy", "energy saving", ["low_load", "energy_saving"]),
        Exp2ChunkDescriptor("chunk007_mixed_slice", "mixed slice SLA", ["mixed_slice", "slice_sla"]),
        Exp2ChunkDescriptor("chunk008_premium_urllc", "premium URLLC", ["premium_tenant", "urllc_burst"]),
        Exp2ChunkDescriptor("chunk009_handover_video", "handover video edge", ["high_mobility", "video_edge"]),
        Exp2ChunkDescriptor("chunk010_interference_urllc", "interference URLLC", ["interference_burst", "urllc_burst"]),
        Exp2ChunkDescriptor("chunk011_recovery_handover", "handover recovery", ["recovery", "high_mobility"]),
        Exp2ChunkDescriptor("chunk012_recovery_interference", "interference recovery", ["recovery", "interference_burst"]),
        Exp2ChunkDescriptor("chunk013_handover_blockage", "handover blockage", ["high_mobility", "blockage"]),
        Exp2ChunkDescriptor("chunk014_stale_recovery", "stale memory recovery", ["stale_memory", "recovery"]),
        Exp2ChunkDescriptor("chunk015_premium_policy", "premium policy", ["premium_tenant", "tenant_priority"]),
        Exp2ChunkDescriptor("chunk016_conflict_energy", "throughput energy conflict", ["energy_saving", "throughput_push"]),
        Exp2ChunkDescriptor("chunk017_audit_blockage", "audit blockage repair", ["audit_safe", "blockage", "hotspot_shift"]),
        Exp2ChunkDescriptor("chunk018_load_balance", "multi-cell load balance", ["load_balance", "dense_hotspot"]),
    ]


def build_exp2_intent_templates() -> list[Exp2IntentTemplate]:
    return [
        Exp2IntentTemplate(
            intent_slug="edge-throughput-latency",
            intent_id="intent:edge_throughput_latency",
            intent_family="throughput_latency_tradeoff",
            intent_text=(
                "Increase edge throughput while keeping BLER and scheduler head-of-line "
                "delay within protected bounds."
            ),
            risk_posture="conservative",
            kpi_targets=[
                GoalKpiTarget("total_throughput_mbps", "maximize", 45.0, weight=1.5, unit="Mbps", priority=4, hard=True),
                GoalKpiTarget("avg_bler", "minimize", 0.12, tolerance=0.015, weight=2.0, unit="ratio", priority=5, hard=True),
                GoalKpiTarget("scheduler_hol_delay_ms", "minimize", 18.0, tolerance=3.0, weight=1.7, unit="ms", priority=5, hard=True),
            ],
        ),
        Exp2IntentTemplate(
            intent_slug="urllc-burst-reliability",
            intent_id="intent:urllc_burst_reliability",
            intent_family="urllc_reliability",
            intent_text="Protect URLLC bursts, reduce BLER and backlog, and preserve usable throughput.",
            risk_posture="conservative",
            kpi_targets=[
                GoalKpiTarget("avg_bler", "minimize", 0.10, tolerance=0.012, weight=2.4, unit="ratio", priority=5, hard=True),
                GoalKpiTarget("total_queue_bytes", "minimize", 52000.0, weight=1.8, unit="bytes", priority=5, hard=True),
                GoalKpiTarget("total_throughput_mbps", "maximize", 32.0, weight=1.1, unit="Mbps", priority=3, hard=True),
            ],
        ),
        Exp2IntentTemplate(
            intent_slug="energy-aware-mobility",
            intent_id="intent:energy_aware_mobility",
            intent_family="energy_mobility_tradeoff",
            intent_text="Save energy opportunistically while keeping handover risk and throughput safe.",
            risk_posture="balanced",
            kpi_targets=[
                GoalKpiTarget("power_cost", "minimize", 28.0, weight=1.2, unit="dBm-like", priority=3, hard=False),
                GoalKpiTarget("handover_risk", "minimize", 0.24, tolerance=0.04, weight=1.8, unit="score", priority=5, hard=True),
                GoalKpiTarget("total_throughput_mbps", "maximize", 30.0, weight=1.0, unit="Mbps", priority=3, hard=True),
            ],
        ),
    ]


def build_exp2_scenario_descriptors() -> list[Exp2ScenarioDescriptor]:
    return [
        Exp2ScenarioDescriptor(
            "nominal_probe",
            "nominal",
            ["nominal_probe"],
            {"avg_bler": 0.11, "handover_pending_ue_count": 4, "resource_grid_utilization": 0.62},
        ),
        Exp2ScenarioDescriptor(
            "stress_transfer",
            "stress",
            ["stress_transfer", "urllc_burst", "high_mobility"],
            {"avg_bler": 0.18, "handover_pending_ue_count": 18, "resource_grid_utilization": 0.89},
        ),
        Exp2ScenarioDescriptor(
            "recovery_edge",
            "recovery",
            ["recovery_edge", "cell_edge_users", "interference_burst"],
            {"avg_bler": 0.16, "handover_pending_ue_count": 10, "resource_grid_utilization": 0.81},
        ),
    ]


def structured_goal_from_exp2_intent(intent: Exp2IntentTemplate, *, chunk_id: str) -> StructuredIntentGoal:
    return StructuredIntentGoal(
        goal_id=f"goal:{intent.intent_slug}:{chunk_id}",
        intent_id=intent.intent_id,
        intent_text=intent.intent_text,
        summary=f"Experiment 2 goal for {intent.intent_slug} in {chunk_id}.",
        horizon_slots=20,
        risk_posture=intent.risk_posture,
        target_entities=["cluster"],
        kpi_targets=list(intent.kpi_targets),
        constraints={"experiment": "exp2_candidate_repair"},
        generated_by="exp2_fixture_builder",
    )


def observation_for_scenario(scenario: Exp2ScenarioDescriptor) -> Observation:
    base = build_default_observation()
    for key, value in dict(scenario.observation_overrides or {}).items():
        if hasattr(base, key):
            setattr(base, key, value)
    return base


def _unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in list(items or []):
        normalized = str(item or "").strip()
        if normalized == "" or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _canonical_categories(xapps: list[str]) -> list[str]:
    return _unique_preserving_order([canonical_xapp_category(item) for item in list(xapps or [])])


def _preferred_variant(
    *,
    projector: OrchestrationFeasibilityProjector,
    xapp_name: str,
    available_xapp_names: list[str],
) -> str | None:
    return projector._preferred_variant_name(  # pylint: disable=protected-access
        xapp_name,
        available_xapp_names=available_xapp_names,
    )


def _schema_valid(requested_xapps: list[str]) -> bool:
    return len(list(requested_xapps or [])) > 0 and all(
        isinstance(item, str) and str(item).strip() != ""
        for item in list(requested_xapps or [])
    )


def _schema_valid_with_bounds(requested_xapps: list[str], bounds: list[RewardParameterRange]) -> bool:
    if not _schema_valid(requested_xapps):
        return False
    for item in list(bounds or []):
        field_name = str(getattr(item, "field_name", "") or "").strip()
        target_name = str(getattr(item, "target_xapp_name", "") or "").strip()
        if field_name == "" or target_name == "":
            return False
        lower_bound = getattr(item, "lower_bound", None)
        upper_bound = getattr(item, "upper_bound", None)
        discrete_values = list(getattr(item, "discrete_values", []) or [])
        if lower_bound is None and upper_bound is None and len(discrete_values) == 0:
            return False
    return True


def _schema_only_output(
    *,
    requested_xapps: list[str],
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
) -> list[str]:
    output: list[str] = []
    seen_categories: set[str] = set()
    for xapp_name in list(requested_xapps or []):
        category_name = canonical_xapp_category(xapp_name)
        if category_name in seen_categories:
            continue
        preferred_name = _preferred_variant(
            projector=projector,
            xapp_name=xapp_name,
            available_xapp_names=available_xapp_names,
        )
        if preferred_name is None:
            continue
        output.append(preferred_name)
        seen_categories.add(category_name)
    return output


def _no_projection_output(requested_xapps: list[str]) -> list[str]:
    return [str(item).strip() for item in list(requested_xapps or []) if str(item).strip() != ""]


def _qos_missing_guard_violation(
    *,
    goal: StructuredIntentGoal,
    output_categories: set[str],
) -> bool:
    kpi_names = {
        str(getattr(target, "kpi_name", "") or "").strip().lower()
        for target in list(getattr(goal, "kpi_targets", []) or [])
    }
    reliability_required = "avg_bler" in kpi_names or "scheduler_hol_delay_ms" in kpi_names
    if not reliability_required:
        return False
    return not bool(output_categories & {"urllc_guard_xapp", "latency_scheduler_xapp"})


def _violation_types(
    *,
    xapps: list[str],
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
) -> list[str]:
    check = _check_feasibility(
        xapps=xapps,
        bounds=[],
        projector=projector,
        available_xapp_names=available_xapp_names,
        goal=goal,
        scenario_tags=[],
        soft_arbitration_applied=False,
        risk_bound_tightened=False,
        ignore_missing_contract_bounds=True,
    )
    return check.violation_types


def _is_feasible(violations: list[str]) -> bool:
    hard_violations = {
        "empty_plan",
        "unknown_xapp",
        "unsupported_parameter",
        "missing_dependency",
        "hard_mutex_conflict",
        "cardinality_overflow",
        "resource_overflow",
        "qos_guard_missing",
        "policy_violation",
        "empty_bound",
        "unsafe_bounds",
    }
    return len(set(violations) & hard_violations) == 0


def _repair_distance(requested_xapps: list[str], output_xapps: list[str]) -> float:
    requested_categories = _canonical_categories(requested_xapps)
    output_categories = _canonical_categories(output_xapps)
    symmetric_distance = len(set(requested_categories).symmetric_difference(set(output_categories)))
    length_delta = abs(len(requested_categories) - len(output_categories))
    return float(symmetric_distance + length_delta)


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 0:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * float(percentile)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    lower = ordered[lower_index] * (upper_index - rank)
    upper = ordered[upper_index] * (rank - lower_index)
    return float(lower + upper)


def _variant_names_by_category(category_name: str) -> list[str]:
    return [
        str(item.get("name", ""))
        for item in DEFAULT_XAPP_VARIANTS.get(canonical_xapp_category(category_name), [])
        if str(item.get("name", "")).strip() != ""
    ]


def _choose_variant(category_name: str, token: str = "balanced") -> str:
    variants = _variant_names_by_category(category_name)
    for name in variants:
        if token in name:
            return name
    return variants[0] if len(variants) > 0 else category_name


def _target_tunable_fields(xapp_name: str) -> list[str]:
    category_name = canonical_xapp_category(xapp_name)
    spec = BASE_XAPP_CAPABILITIES.get(category_name, {})
    return [str(item) for item in list(spec.get("tunable_fields", []) or [])]


def _active_projectable_keys(xapps: list[str]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for xapp_name in list(xapps or []):
        for field_name in _target_tunable_fields(xapp_name):
            if field_name in PARAM_POLICY_LIBRARY:
                keys.append((str(xapp_name), str(field_name)))
    return keys


def _range_key(parameter_range: RewardParameterRange) -> tuple[str, str]:
    return (
        str(getattr(parameter_range, "target_xapp_name", "") or "").strip(),
        str(getattr(parameter_range, "field_name", "") or "").strip(),
    )


def _category_range_key(parameter_range: RewardParameterRange) -> tuple[str, str]:
    target_name, field_name = _range_key(parameter_range)
    return (canonical_xapp_category(target_name), field_name)


def _range_by_category_key(bounds: list[RewardParameterRange]) -> dict[tuple[str, str], RewardParameterRange]:
    result: dict[tuple[str, str], RewardParameterRange] = {}
    for item in list(bounds or []):
        key = _category_range_key(item)
        if key[0] == "" or key[1] == "":
            continue
        result[key] = item
    return result


def _make_parameter_range(
    *,
    xapp_name: str,
    field_name: str,
    l3_tight: bool = False,
    conservative: bool = False,
) -> RewardParameterRange:
    policy = dict(PARAM_POLICY_LIBRARY.get(field_name, {}))
    if "values" in policy:
        values = list(policy.get("values", []) or [])
        if conservative and len(values) > 0:
            values = [values[0]]
        return RewardParameterRange(
            field_name=field_name,
            target_xapp_name=xapp_name,
            scope="cluster" if canonical_xapp_category(xapp_name) in {"traffic_steering_xapp", "mobility_predictor_xapp"} else "cell",
            discrete_values=values,
            tuning_mode="safety_envelope",
        )

    lower = float(policy.get("lower", 0.0) or 0.0)
    upper = float(policy.get("upper", lower) or lower)
    if bool(l3_tight):
        if "l3_lower" in policy:
            lower = max(lower, float(policy["l3_lower"]))
        if "l3_upper" in policy:
            upper = min(upper, float(policy["l3_upper"]))
    if bool(conservative):
        midpoint = (lower + upper) * 0.5
        half_width = max((upper - lower) * 0.18, float(policy.get("step", 0.05) or 0.05))
        lower = max(lower, midpoint - half_width)
        upper = min(upper, midpoint + half_width)
    return RewardParameterRange(
        field_name=field_name,
        target_xapp_name=xapp_name,
        scope="cluster" if canonical_xapp_category(xapp_name) in {"traffic_steering_xapp", "mobility_predictor_xapp"} else "cell",
        lower_bound=round(lower, 6),
        upper_bound=round(upper, 6),
        step=float(policy.get("step", 0.05) or 0.05),
        max_delta=float(policy.get("step", 0.05) or 0.05) * 4.0,
        tuning_mode="safety_envelope",
    )


def _make_raw_range(
    *,
    xapp_name: str,
    field_name: str,
    lower: float | None = None,
    upper: float | None = None,
    values: list[object] | None = None,
) -> RewardParameterRange:
    return RewardParameterRange(
        field_name=field_name,
        target_xapp_name=xapp_name,
        lower_bound=lower,
        upper_bound=upper,
        discrete_values=list(values or []),
        tuning_mode="planner_candidate",
    )


def _ranges_equal(left: RewardParameterRange | None, right: RewardParameterRange | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        str(getattr(left, "field_name", "") or "") == str(getattr(right, "field_name", "") or "")
        and canonical_xapp_category(str(getattr(left, "target_xapp_name", "") or ""))
        == canonical_xapp_category(str(getattr(right, "target_xapp_name", "") or ""))
        and getattr(left, "lower_bound", None) == getattr(right, "lower_bound", None)
        and getattr(left, "upper_bound", None) == getattr(right, "upper_bound", None)
        and list(getattr(left, "discrete_values", []) or []) == list(getattr(right, "discrete_values", []) or [])
    )


def _intersect_range(
    requested: RewardParameterRange | None,
    policy_range: RewardParameterRange,
) -> tuple[RewardParameterRange, bool, bool]:
    if requested is None:
        return policy_range, True, False
    if len(list(getattr(policy_range, "discrete_values", []) or [])) > 0:
        allowed = set(getattr(policy_range, "discrete_values", []) or [])
        requested_values = list(getattr(requested, "discrete_values", []) or [])
        if len(requested_values) == 0:
            result_values = list(allowed)
        else:
            result_values = [value for value in requested_values if value in allowed]
        result = RewardParameterRange(
            field_name=policy_range.field_name,
            target_xapp_name=policy_range.target_xapp_name,
            scope=policy_range.scope,
            discrete_values=result_values,
            tuning_mode="safety_envelope",
        )
        return result, not _ranges_equal(requested, result), len(result_values) == 0

    req_lower = getattr(requested, "lower_bound", None)
    req_upper = getattr(requested, "upper_bound", None)
    lower = float(getattr(policy_range, "lower_bound", 0.0) or 0.0) if req_lower is None else max(
        float(req_lower),
        float(getattr(policy_range, "lower_bound", 0.0) or 0.0),
    )
    upper = float(getattr(policy_range, "upper_bound", lower) or lower) if req_upper is None else min(
        float(req_upper),
        float(getattr(policy_range, "upper_bound", req_upper) or req_upper),
    )
    result = RewardParameterRange(
        field_name=policy_range.field_name,
        target_xapp_name=policy_range.target_xapp_name,
        scope=policy_range.scope,
        lower_bound=round(lower, 6),
        upper_bound=round(upper, 6),
        step=getattr(policy_range, "step", None),
        max_delta=getattr(policy_range, "max_delta", None),
        tuning_mode="safety_envelope",
    )
    return result, not _ranges_equal(requested, result), lower > upper


def _bounds_to_compact_text(bounds: list[RewardParameterRange]) -> str:
    parts: list[str] = []
    for item in list(bounds or []):
        field_name = str(getattr(item, "field_name", "") or "")
        target = canonical_xapp_category(str(getattr(item, "target_xapp_name", "") or ""))
        values = list(getattr(item, "discrete_values", []) or [])
        if len(values) > 0:
            parts.append(f"{target}.{field_name}={','.join(str(value) for value in values)}")
        else:
            parts.append(
                f"{target}.{field_name}=[{getattr(item, 'lower_bound', None)},{getattr(item, 'upper_bound', None)}]"
            )
    return ";".join(parts)


def _xapp_bound_supported(xapps: list[str], parameter_range: RewardParameterRange) -> bool:
    target_name = str(getattr(parameter_range, "target_xapp_name", "") or "").strip()
    field_name = str(getattr(parameter_range, "field_name", "") or "").strip()
    if target_name == "" or field_name == "":
        return False
    selected_categories = {canonical_xapp_category(item) for item in list(xapps or [])}
    category_name = canonical_xapp_category(target_name)
    if category_name not in selected_categories:
        return False
    return field_name in _target_tunable_fields(target_name) and field_name in PARAM_POLICY_LIBRARY


def _bound_is_safe(parameter_range: RewardParameterRange) -> tuple[bool, bool]:
    field_name = str(getattr(parameter_range, "field_name", "") or "").strip()
    policy = PARAM_POLICY_LIBRARY.get(field_name)
    if policy is None:
        return False, False
    if "values" in policy:
        values = list(getattr(parameter_range, "discrete_values", []) or [])
        allowed = set(policy.get("values", []) or [])
        return len(values) > 0, len(values) > 0 and set(values).issubset(allowed)
    lower = getattr(parameter_range, "lower_bound", None)
    upper = getattr(parameter_range, "upper_bound", None)
    if lower is None or upper is None:
        return False, False
    lower_f = float(lower)
    upper_f = float(upper)
    non_empty = lower_f <= upper_f
    safe = (
        lower_f >= float(policy.get("lower", lower_f) or lower_f) - 1e-9
        and upper_f <= float(policy.get("upper", upper_f) or upper_f) + 1e-9
    )
    return non_empty, bool(non_empty and safe)


def _has_category(categories: set[str], category_name: str) -> bool:
    return canonical_xapp_category(category_name) in categories


def _scenario_tag_set(tags: list[str]) -> set[str]:
    return {str(tag).strip().lower() for tag in list(tags or []) if str(tag).strip() != ""}


def _find_bound(bounds: list[RewardParameterRange], field_name: str) -> RewardParameterRange | None:
    for item in list(bounds or []):
        if str(getattr(item, "field_name", "") or "") == field_name:
            return item
    return None


def _classify_soft_risk_policy(
    *,
    xapps: list[str],
    bounds: list[RewardParameterRange],
    scenario_tags: list[str],
    goal: StructuredIntentGoal,
) -> tuple[bool, bool, bool]:
    categories = set(_canonical_categories(xapps))
    tags = _scenario_tag_set(scenario_tags)
    xapp_text = " ".join(str(item).lower() for item in list(xapps or []))

    soft_conflict = False
    if _has_category(categories, "power_saving_xapp") and (
        _has_category(categories, "latency_scheduler_xapp") or _has_category(categories, "urllc_guard_xapp")
    ):
        soft_conflict = True
    if (
        _has_category(categories, "massive_mimo_beamformer_xapp")
        and _has_category(categories, "spectrum_optimizer_xapp")
        and _has_category(categories, "urllc_guard_xapp")
        and ("capacity_push" in xapp_text or "aggressive" in xapp_text)
    ):
        soft_conflict = True

    risk_conflict = False
    steering = _find_bound(bounds, "steering_aggressiveness")
    hysteresis = _find_bound(bounds, "steering_hysteresis_db")
    if (
        _has_category(categories, "traffic_steering_xapp")
        and _has_category(categories, "mobility_predictor_xapp")
        and ("high_mobility" in tags or "handover_edge" in tags)
    ):
        steering_upper = float(getattr(steering, "upper_bound", 0.80) or 0.80)
        hysteresis_lower = float(getattr(hysteresis, "lower_bound", 2.0) or 2.0)
        if steering_upper > 0.60 or hysteresis_lower < 3.0 or "load_first" in xapp_text or "handover_push" in xapp_text:
            risk_conflict = True
    if (
        _has_category(categories, "spectrum_optimizer_xapp")
        and _has_category(categories, "massive_mimo_beamformer_xapp")
        and ("interference_burst" in tags or "cell_edge_users" in tags)
        and ("aggressive" in xapp_text or "capacity_push" in xapp_text)
    ):
        risk_conflict = True
    if "stale_memory" in tags and ("aggressive" in xapp_text or "load_first" in xapp_text):
        risk_conflict = True

    policy_violation = False
    if _has_category(categories, "cell_sleep_xapp") and (
        "premium_tenant" in tags
        or "tenant_priority" in tags
        or "coverage_constraint" in tags
        or "urllc_burst" in tags
        or any(
            str(getattr(target, "kpi_name", "") or "").lower() in {"avg_bler", "scheduler_hol_delay_ms"}
            and bool(getattr(target, "hard", False))
            for target in list(getattr(goal, "kpi_targets", []) or [])
        )
    ):
        policy_violation = True
    sleep_mode = _find_bound(bounds, "cell_sleep_mode")
    if sleep_mode is not None:
        values = set(getattr(sleep_mode, "discrete_values", []) or [])
        if values & {"light_sleep", "deep_sleep"} and ("premium_tenant" in tags or "urllc_burst" in tags):
            policy_violation = True

    return soft_conflict, risk_conflict, policy_violation


def _check_feasibility(
    *,
    xapps: list[str],
    bounds: list[RewardParameterRange],
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
    soft_arbitration_applied: bool,
    risk_bound_tightened: bool,
    ignore_missing_contract_bounds: bool = False,
) -> FeasibilityCheck:
    violations: list[str] = []
    selected_xapps = [str(item).strip() for item in list(xapps or []) if str(item).strip() != ""]
    if len(selected_xapps) == 0:
        return FeasibilityCheck(violation_types=["empty_plan"], executable=False)

    available_names = set(available_xapp_names)
    unknown_names = [
        xapp_name
        for xapp_name in selected_xapps
        if xapp_name not in available_names
        and _preferred_variant(
            projector=projector,
            xapp_name=xapp_name,
            available_xapp_names=available_xapp_names,
        )
        is None
    ]
    if len(unknown_names) > 0:
        violations.append("unknown_xapp")

    categories = _canonical_categories(selected_xapps)
    category_set = set(categories)
    if len(categories) < len([canonical_xapp_category(item) for item in selected_xapps]):
        violations.append("duplicate_category")

    for dependent_category, dependency_category in projector.dependency_category_pairs:
        if dependent_category in category_set and dependency_category not in category_set:
            violations.append("missing_dependency")
            break

    hard_conflict = False
    for left_category, right_category in projector.mutex_category_pairs:
        if left_category in category_set and right_category in category_set:
            violations.append("hard_mutex_conflict")
            hard_conflict = True
            break

    if len(category_set) > int(projector.max_selected_xapps):
        violations.append("cardinality_overflow")

    known_costed_xapps = [
        xapp_name
        for xapp_name in selected_xapps
        if "unknown_xapp" not in violations
        or _preferred_variant(
            projector=projector,
            xapp_name=xapp_name,
            available_xapp_names=available_xapp_names,
        )
        is not None
    ]
    cpu_cost, mem_cost = projector._summed_costs(known_costed_xapps)  # pylint: disable=protected-access
    resource_overflow = float(cpu_cost) > float(projector.max_cpu_cost) or float(mem_cost) > float(projector.max_mem_cost)
    if resource_overflow:
        violations.append("resource_overflow")

    if _qos_missing_guard_violation(goal=goal, output_categories=category_set):
        violations.append("qos_guard_missing")

    unsupported_parameter = False
    empty_bound = False
    unsafe_bounds = False
    if not bool(ignore_missing_contract_bounds):
        active_keys = set(_active_projectable_keys(selected_xapps))
        covered_keys = {_range_key(item) for item in list(bounds or [])}
        missing_keys = active_keys - covered_keys
        if len(active_keys) > 0 and len(missing_keys) == len(active_keys):
            violations.append("missing_action_region")
    for parameter_range in list(bounds or []):
        supported = _xapp_bound_supported(selected_xapps, parameter_range)
        if not supported:
            unsupported_parameter = True
            continue
        non_empty, safe = _bound_is_safe(parameter_range)
        if not non_empty:
            empty_bound = True
        if not safe:
            unsafe_bounds = True
    if unsupported_parameter:
        violations.append("unsupported_parameter")
    if empty_bound:
        violations.append("empty_bound")
    if unsafe_bounds:
        violations.append("unsafe_bounds")

    soft_conflict, risk_conflict, policy_violation = _classify_soft_risk_policy(
        xapps=selected_xapps,
        bounds=bounds,
        scenario_tags=scenario_tags,
        goal=goal,
    )
    if soft_conflict and not bool(soft_arbitration_applied):
        violations.append("soft_conflict_unarbitrated")
    if risk_conflict and not bool(risk_bound_tightened):
        violations.append("risk_conflict_unmitigated")
    if policy_violation:
        violations.append("policy_violation")

    unique_violations = _unique_preserving_order(violations)
    return FeasibilityCheck(
        violation_types=unique_violations,
        executable=_is_feasible(unique_violations),
        hard_conflict=hard_conflict,
        resource_overflow=resource_overflow,
        empty_bound=empty_bound,
        unsafe_bounds=unsafe_bounds,
        unsupported_parameter=unsupported_parameter,
        policy_violation=policy_violation,
        risk_conflict=risk_conflict,
        soft_conflict=soft_conflict,
    )


def _goal_kpi_names(goal: StructuredIntentGoal) -> list[str]:
    return [
        str(getattr(target, "kpi_name", "") or "").strip().lower()
        for target in list(getattr(goal, "kpi_targets", []) or [])
        if str(getattr(target, "kpi_name", "") or "").strip() != ""
    ]


def _coverage_for_kpi(kpi_name: str, categories: set[str]) -> bool:
    kpi = str(kpi_name or "").lower()
    if kpi in {"avg_bler", "scheduler_hol_delay_ms", "delay_ms", "total_queue_bytes"}:
        return bool(categories & {"urllc_guard_xapp", "latency_scheduler_xapp", "ran_slicing_manager_xapp"})
    if kpi in {"total_throughput_mbps", "avg_sinr_db", "sinr"}:
        return bool(categories & {"massive_mimo_beamformer_xapp", "spectrum_optimizer_xapp", "traffic_steering_xapp"})
    if kpi in {"handover_risk", "service_continuity"}:
        return bool(categories & {"mobility_predictor_xapp", "traffic_steering_xapp"})
    if kpi in {"power_cost", "energy_efficiency"}:
        return bool(categories & {"power_saving_xapp", "cell_sleep_xapp"})
    return len(categories) > 0


def _protected_kpi_coverage(xapps: list[str], goal: StructuredIntentGoal) -> float:
    protected = [
        str(getattr(target, "kpi_name", "") or "").strip().lower()
        for target in list(getattr(goal, "kpi_targets", []) or [])
        if bool(getattr(target, "hard", False)) or int(getattr(target, "priority", 0) or 0) >= 3
    ]
    if len(protected) == 0:
        protected = _goal_kpi_names(goal)
    if len(protected) == 0:
        return 1.0
    categories = set(_canonical_categories(xapps))
    covered = sum(1 for kpi_name in protected if _coverage_for_kpi(kpi_name, categories))
    return float(covered) / float(len(protected))


def _intent_coverage(xapps: list[str], goal: StructuredIntentGoal) -> float:
    kpis = _goal_kpi_names(goal)
    if len(kpis) == 0:
        return 1.0 if len(xapps) > 0 else 0.0
    categories = set(_canonical_categories(xapps))
    covered = sum(1 for kpi_name in kpis if _coverage_for_kpi(kpi_name, categories))
    return float(covered) / float(len(kpis))


def _predicted_intent_loss(
    *,
    xapps: list[str],
    check: FeasibilityCheck,
    goal: StructuredIntentGoal,
    fallback_used: bool,
) -> float:
    coverage = _intent_coverage(xapps, goal)
    protected_coverage = _protected_kpi_coverage(xapps, goal)
    penalty = 0.0
    if check.resource_overflow:
        penalty += 0.22
    if check.hard_conflict or check.policy_violation:
        penalty += 0.35
    if check.risk_conflict:
        penalty += 0.12
    if check.soft_conflict:
        penalty += 0.08
    if bool(fallback_used):
        penalty += 0.18
    return max(0.0, min(1.0, 1.0 - 0.65 * coverage - 0.25 * protected_coverage + penalty))


def _canonicalize_registered_xapps(
    *,
    requested_xapps: list[str],
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    deduplicate_categories: bool = True,
) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    seen_categories: set[str] = set()
    notes: list[str] = []
    for xapp_name in list(requested_xapps or []):
        normalized = str(xapp_name or "").strip()
        if normalized == "":
            continue
        category = canonical_xapp_category(normalized)
        if bool(deduplicate_categories) and category in seen_categories:
            notes.append(f"deduplicated_category:{category}")
            continue
        preferred = _preferred_variant(
            projector=projector,
            xapp_name=normalized,
            available_xapp_names=available_xapp_names,
        )
        if preferred is None:
            notes.append(f"dropped_unregistered:{normalized}")
            continue
        selected.append(preferred)
        seen_categories.add(category)
    return selected, notes


def _category_priority(
    *,
    projector: OrchestrationFeasibilityProjector,
    category_name: str,
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
    mode: str,
) -> float:
    goal_focus = projector._goal_focus(goal)  # pylint: disable=protected-access
    risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()
    score = projector._category_priority(  # pylint: disable=protected-access
        category_name,
        goal_focus=goal_focus,
        risk_posture=risk_posture,
    )
    tags = _scenario_tag_set(scenario_tags)
    if category_name == "mobility_predictor_xapp" and ("high_mobility" in tags or "handover_edge" in tags):
        score += 0.18
    if category_name == "traffic_steering_xapp" and ("hotspot_shift" in tags or "load_balance" in tags):
        score += 0.14
    if category_name == "urllc_guard_xapp" and "urllc_burst" in tags:
        score += 0.22
    if category_name == "latency_scheduler_xapp" and "urllc_burst" in tags:
        score += 0.16
    if category_name == "power_saving_xapp" and "energy_saving" in tags:
        score += 0.12
    if category_name == "cell_sleep_xapp" and "energy_saving" in tags:
        score += 0.05
    if mode == "orchestran_like":
        cost = projector._resource_cost(category_name)[0]  # pylint: disable=protected-access
        score -= 0.10 * cost
    if mode == "oreo_like":
        shared_bonus = {
            "urllc_guard_xapp": 0.08,
            "ran_slicing_manager_xapp": 0.08,
            "traffic_steering_xapp": 0.05,
            "mobility_predictor_xapp": 0.05,
        }.get(category_name, 0.0)
        score += shared_bonus
    return float(score)


def _remove_lowest_priority_for_mode(
    *,
    selected_xapps: list[str],
    projector: OrchestrationFeasibilityProjector,
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
    mode: str,
    protected_categories: set[str] | None = None,
) -> tuple[list[str], str]:
    protected = set(protected_categories or set())
    removable = [
        item for item in list(selected_xapps or [])
        if canonical_xapp_category(item) not in protected
    ]
    if len(removable) == 0:
        removable = list(selected_xapps or [])
    if len(removable) == 0:
        return [], ""
    victim = sorted(
        removable,
        key=lambda item: (
            _category_priority(
                projector=projector,
                category_name=canonical_xapp_category(item),
                goal=goal,
                scenario_tags=scenario_tags,
                mode=mode,
            ),
            -projector._resource_cost(item)[0],  # pylint: disable=protected-access
            str(item),
        ),
    )[0]
    updated = [item for item in list(selected_xapps or []) if item != victim]
    return updated, victim


def _repair_dependencies(
    *,
    selected_xapps: list[str],
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    notes: list[str],
) -> list[str]:
    selected = list(selected_xapps or [])
    for dependent_category, dependency_category in projector.dependency_category_pairs:
        categories = {canonical_xapp_category(item) for item in selected}
        if dependent_category not in categories or dependency_category in categories:
            continue
        dependency = _preferred_variant(
            projector=projector,
            xapp_name=dependency_category,
            available_xapp_names=available_xapp_names,
        )
        if dependency is None:
            selected = [item for item in selected if canonical_xapp_category(item) != dependent_category]
            notes.append(f"dropped_for_missing_dependency:{dependent_category}->{dependency_category}")
        else:
            selected.append(dependency)
            notes.append(f"added_dependency:{dependent_category}->{dependency_category}")
    return _unique_preserving_order(selected)


def _resolve_hard_mutex(
    *,
    selected_xapps: list[str],
    projector: OrchestrationFeasibilityProjector,
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
    notes: list[str],
    mode: str,
) -> list[str]:
    selected = list(selected_xapps or [])
    for _ in range(8):
        changed = False
        category_map = {canonical_xapp_category(item): item for item in selected}
        for left_category, right_category in projector.mutex_category_pairs:
            if left_category not in category_map or right_category not in category_map:
                continue
            left_score = _category_priority(
                projector=projector,
                category_name=left_category,
                goal=goal,
                scenario_tags=scenario_tags,
                mode=mode,
            )
            right_score = _category_priority(
                projector=projector,
                category_name=right_category,
                goal=goal,
                scenario_tags=scenario_tags,
                mode=mode,
            )
            drop_category = right_category if left_score >= right_score else left_category
            selected = [item for item in selected if canonical_xapp_category(item) != drop_category]
            notes.append(f"resolved_hard_mutex:{left_category}!={right_category}:dropped={drop_category}")
            changed = True
            break
        if not changed:
            break
    return selected


def _resolve_non_overridable_policy(
    *,
    selected_xapps: list[str],
    scenario_tags: list[str],
    goal: StructuredIntentGoal,
    notes: list[str],
) -> list[str]:
    tags = _scenario_tag_set(scenario_tags)
    reliability_hard = any(
        str(getattr(target, "kpi_name", "") or "").lower() in {"avg_bler", "scheduler_hol_delay_ms"}
        and bool(getattr(target, "hard", False))
        for target in list(getattr(goal, "kpi_targets", []) or [])
    )
    if not (
        "premium_tenant" in tags
        or "tenant_priority" in tags
        or "coverage_constraint" in tags
        or "urllc_burst" in tags
        or reliability_hard
    ):
        return selected_xapps
    updated = [
        item for item in list(selected_xapps or [])
        if canonical_xapp_category(item) != "cell_sleep_xapp"
    ]
    if len(updated) != len(list(selected_xapps or [])):
        notes.append("policy_removed_cell_sleep")
    return updated


def _resource_repair(
    *,
    selected_xapps: list[str],
    projector: OrchestrationFeasibilityProjector,
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
    notes: list[str],
    mode: str,
    protected_categories: set[str] | None = None,
) -> list[str]:
    selected = list(selected_xapps or [])
    for _ in range(16):
        cpu_cost, mem_cost = projector._summed_costs(selected)  # pylint: disable=protected-access
        if (
            len(set(_canonical_categories(selected))) <= int(projector.max_selected_xapps)
            and float(cpu_cost) <= float(projector.max_cpu_cost)
            and float(mem_cost) <= float(projector.max_mem_cost)
        ):
            break
        selected, victim = _remove_lowest_priority_for_mode(
            selected_xapps=selected,
            projector=projector,
            goal=goal,
            scenario_tags=scenario_tags,
            mode=mode,
            protected_categories=protected_categories,
        )
        if victim == "":
            break
        notes.append(f"resource_pruned:{victim}")
    return selected


def _fallback_plan(
    *,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
) -> list[str]:
    del goal
    selected: list[str] = []
    for name in SAFE_FALLBACK_CANDIDATE:
        preferred = _preferred_variant(
            projector=projector,
            xapp_name=name,
            available_xapp_names=available_xapp_names,
        )
        if preferred is not None:
            selected.append(preferred)
    if len(selected) == 0:
        preferred = _preferred_variant(
            projector=projector,
            xapp_name="urllc_guard_xapp",
            available_xapp_names=available_xapp_names,
        )
        if preferred is not None:
            selected.append(preferred)
    return selected


def _normalize_bounds(
    *,
    selected_xapps: list[str],
    requested_bounds: list[RewardParameterRange],
    scenario_tags: list[str],
    check_risk_before: bool,
    use_l3: bool,
    conservative: bool = False,
) -> tuple[list[RewardParameterRange], bool, bool, list[str]]:
    requested_by_category = _range_by_category_key(requested_bounds)
    output_bounds: list[RewardParameterRange] = []
    repaired = False
    tightened = False
    notes: list[str] = []
    tags = _scenario_tag_set(scenario_tags)
    high_risk_context = bool(
        check_risk_before
        or "high_mobility" in tags
        or "handover_edge" in tags
        or "interference_burst" in tags
        or "stale_memory" in tags
    )
    for xapp_name, field_name in _active_projectable_keys(selected_xapps):
        l3_tight = bool(use_l3 and high_risk_context and field_name in {"steering_aggressiveness", "steering_hysteresis_db", "muting_ratio"})
        policy_range = _make_parameter_range(
            xapp_name=xapp_name,
            field_name=field_name,
            l3_tight=l3_tight,
            conservative=conservative,
        )
        requested = requested_by_category.get((canonical_xapp_category(xapp_name), field_name))
        result, changed, empty = _intersect_range(requested, policy_range)
        if empty:
            result = policy_range
            changed = True
            notes.append(f"empty_bound_reset:{canonical_xapp_category(xapp_name)}.{field_name}")
        if changed:
            repaired = True
        if l3_tight:
            tightened = True
            notes.append(f"l3_tightened:{canonical_xapp_category(xapp_name)}.{field_name}")
        output_bounds.append(result)
    return output_bounds, repaired, tightened, notes


def _memory_risk_context(*, scenario_tags: list[str], raw_check: FeasibilityCheck) -> bool:
    tags = _scenario_tag_set(scenario_tags)
    return bool(
        raw_check.risk_conflict
        or "high_mobility" in tags
        or "handover_edge" in tags
        or "interference_burst" in tags
        or "cell_edge_users" in tags
        or "stale_memory" in tags
    )


def _drop_category(selected_xapps: list[str], category_name: str) -> list[str]:
    return [
        item for item in list(selected_xapps or [])
        if canonical_xapp_category(item) != canonical_xapp_category(category_name)
    ]


def _l2_memory_risk_avoidance(
    *,
    selected_xapps: list[str],
    scenario_tags: list[str],
    raw_check: FeasibilityCheck,
    notes: list[str],
) -> tuple[list[str], bool]:
    """Candidate-level L2 avoidance used by the w/o-L3 ablation.

    L2 memory can down-rank or remove historically risky xApp combinations, but
    it cannot edit continuous action bounds.  This intentionally separates it
    from L3 counterfactual-supported bound tightening.
    """

    selected = list(selected_xapps or [])
    tags = _scenario_tag_set(scenario_tags)
    if not _memory_risk_context(scenario_tags=scenario_tags, raw_check=raw_check):
        return selected, False

    changed = False
    categories = {canonical_xapp_category(item) for item in selected}
    text = " ".join(str(item).lower() for item in selected)

    if (
        "traffic_steering_xapp" in categories
        and "mobility_predictor_xapp" in categories
        and ("high_mobility" in tags or "handover_edge" in tags or "stale_memory" in tags)
    ):
        selected = _drop_category(selected, "traffic_steering_xapp")
        notes.append("l2_avoid_rule:dropped_traffic_steering_high_mobility")
        changed = True

    categories = {canonical_xapp_category(item) for item in selected}
    if (
        "spectrum_optimizer_xapp" in categories
        and "massive_mimo_beamformer_xapp" in categories
        and ("interference_burst" in tags or "cell_edge_users" in tags)
        and ("aggressive" in text or "capacity_push" in text)
    ):
        selected = _drop_category(selected, "spectrum_optimizer_xapp")
        notes.append("l2_avoid_rule:dropped_spectrum_optimizer_interference_edge")
        changed = True

    categories = {canonical_xapp_category(item) for item in selected}
    if "stale_memory" in tags and "traffic_steering_xapp" in categories and ("load_first" in text or "aggressive" in text):
        selected = _drop_category(selected, "traffic_steering_xapp")
        notes.append("l2_avoid_rule:dropped_stale_aggressive_steering")
        changed = True

    return _unique_preserving_order(selected), changed


def _schema_only_method(
    *,
    fixture: CandidateRepairFixture,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
) -> MethodPlanOutput:
    requested_xapps = list(fixture.requested_xapps or [])
    schema_ok = _schema_valid_with_bounds(requested_xapps, list(fixture.requested_bounds or []))
    output = MethodPlanOutput(raw_action="schema_validate_only")
    if not schema_ok:
        output.candidate_rejected = True
        output.projection_notes.append("schema_only:rejected_schema_invalid")
        return output
    available = set(available_xapp_names)
    all_registered = all(item in available for item in requested_xapps)
    all_supported = all(_xapp_bound_supported(requested_xapps, item) for item in list(fixture.requested_bounds or []))
    if not all_registered or not all_supported:
        output.candidate_rejected = True
        output.projection_notes.append("schema_only:rejected_registry_or_parameter")
        return output
    output.output_xapps = list(requested_xapps)
    output.output_bounds = list(fixture.requested_bounds or [])
    return output


def _projective_repair_method(
    *,
    mode: str,
    fixture: CandidateRepairFixture,
    raw_check: FeasibilityCheck,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
) -> MethodPlanOutput:
    notes: list[str] = []
    selected, canonical_notes = _canonicalize_registered_xapps(
        requested_xapps=list(fixture.requested_xapps or []),
        projector=projector,
        available_xapp_names=available_xapp_names,
        deduplicate_categories=True,
    )
    notes.extend(canonical_notes)

    if len(selected) == 0:
        return MethodPlanOutput(candidate_rejected=True, projection_notes=notes + ["repair:empty_after_registry_grounding"])

    selected = _repair_dependencies(
        selected_xapps=selected,
        projector=projector,
        available_xapp_names=available_xapp_names,
        notes=notes,
    )
    selected = _resolve_non_overridable_policy(
        selected_xapps=selected,
        scenario_tags=scenario_tags,
        goal=goal,
        notes=notes,
    )
    selected = _resolve_hard_mutex(
        selected_xapps=selected,
        projector=projector,
        goal=goal,
        scenario_tags=scenario_tags,
        notes=notes,
        mode=mode,
    )
    selected = _resource_repair(
        selected_xapps=selected,
        projector=projector,
        goal=goal,
        scenario_tags=scenario_tags,
        notes=notes,
        mode=mode,
        protected_categories={"urllc_guard_xapp", "latency_scheduler_xapp"} if mode == "full_repair" else set(),
    )
    selected = _repair_dependencies(
        selected_xapps=selected,
        projector=projector,
        available_xapp_names=available_xapp_names,
        notes=notes,
    )
    selected = _resource_repair(
        selected_xapps=selected,
        projector=projector,
        goal=goal,
        scenario_tags=scenario_tags,
        notes=notes,
        mode=mode,
        protected_categories={"urllc_guard_xapp", "latency_scheduler_xapp"} if mode == "full_repair" else set(),
    )

    l2_selection_adjusted = False
    if mode == "repair_wo_l3":
        selected, l2_selection_adjusted = _l2_memory_risk_avoidance(
            selected_xapps=selected,
            scenario_tags=scenario_tags,
            raw_check=raw_check,
            notes=notes,
        )
        if l2_selection_adjusted:
            selected = _repair_dependencies(
                selected_xapps=selected,
                projector=projector,
                available_xapp_names=available_xapp_names,
                notes=notes,
            )
            selected = _resource_repair(
                selected_xapps=selected,
                projector=projector,
                goal=goal,
                scenario_tags=scenario_tags,
                notes=notes,
                mode=mode,
            )

    if len(selected) == 0:
        return MethodPlanOutput(candidate_rejected=True, projection_notes=notes + ["repair:empty_after_projection"])

    use_l3 = mode == "full_repair"
    bounds, bound_repaired, risk_tightened, bound_notes = _normalize_bounds(
        selected_xapps=selected,
        requested_bounds=list(fixture.requested_bounds or []),
        scenario_tags=scenario_tags,
        check_risk_before=raw_check.risk_conflict,
        use_l3=use_l3,
        conservative=False,
    )
    notes.extend(bound_notes)
    soft_conflict, risk_conflict, _ = _classify_soft_risk_policy(
        xapps=selected,
        bounds=bounds,
        scenario_tags=scenario_tags,
        goal=goal,
    )
    soft_arbitration = bool(soft_conflict)
    if soft_arbitration:
        notes.append("soft_conflict_arbitrated:guardrail_first")
    l2_penalty = (
        (mode == "repair_wo_l3" and bool(l2_selection_adjusted or risk_conflict or raw_check.risk_conflict))
        or (mode == "full_repair" and bool(risk_conflict or raw_check.risk_conflict))
    )
    if l2_penalty:
        notes.append("l2_failure_memory_penalty_applied")

    projected = bool(
        selected != list(fixture.requested_xapps or [])
        or bound_repaired
        or risk_tightened
        or len(notes) > 0
    )
    return MethodPlanOutput(
        output_xapps=selected,
        output_bounds=bounds,
        projected=projected,
        projection_notes=notes,
        parameter_bound_repaired=bound_repaired,
        soft_arbitration_applied=soft_arbitration,
        risk_bound_tightened=risk_tightened,
        l2_penalty_applied=l2_penalty,
        raw_action=mode,
    )


def _orchestran_like_method(
    *,
    fixture: CandidateRepairFixture,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
) -> MethodPlanOutput:
    notes: list[str] = ["orchestran_like:resource_timing_switching_cost"]
    selected, canonical_notes = _canonicalize_registered_xapps(
        requested_xapps=list(fixture.requested_xapps or []),
        projector=projector,
        available_xapp_names=available_xapp_names,
        deduplicate_categories=True,
    )
    notes.extend(canonical_notes)
    selected = _repair_dependencies(
        selected_xapps=selected,
        projector=projector,
        available_xapp_names=available_xapp_names,
        notes=notes,
    )
    selected = _resource_repair(
        selected_xapps=selected,
        projector=projector,
        goal=goal,
        scenario_tags=scenario_tags,
        notes=notes,
        mode="orchestran_like",
    )
    if len(selected) == 0:
        return MethodPlanOutput(candidate_rejected=True, projection_notes=notes + ["orchestran_like:resource_infeasible"])
    bounds, repaired, _, bound_notes = _normalize_bounds(
        selected_xapps=selected,
        requested_bounds=list(fixture.requested_bounds or []),
        scenario_tags=scenario_tags,
        check_risk_before=False,
        use_l3=False,
    )
    notes.extend(bound_notes)
    return MethodPlanOutput(
        output_xapps=selected,
        output_bounds=bounds,
        projected=True,
        projection_notes=notes,
        parameter_bound_repaired=repaired,
        raw_action="orchestran_like",
    )


def _oreo_like_method(
    *,
    fixture: CandidateRepairFixture,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
) -> MethodPlanOutput:
    notes: list[str] = ["oreo_like:service_graph_coverage_sharing"]
    selected, canonical_notes = _canonicalize_registered_xapps(
        requested_xapps=list(fixture.requested_xapps or []),
        projector=projector,
        available_xapp_names=available_xapp_names,
        deduplicate_categories=True,
    )
    notes.extend(canonical_notes)
    selected = _repair_dependencies(
        selected_xapps=selected,
        projector=projector,
        available_xapp_names=available_xapp_names,
        notes=notes,
    )
    selected = _resource_repair(
        selected_xapps=selected,
        projector=projector,
        goal=goal,
        scenario_tags=scenario_tags,
        notes=notes,
        mode="oreo_like",
        protected_categories={"urllc_guard_xapp", "ran_slicing_manager_xapp"},
    )
    if len(selected) == 0:
        return MethodPlanOutput(candidate_rejected=True, projection_notes=notes + ["oreo_like:empty_service_graph"])
    bounds, repaired, _, bound_notes = _normalize_bounds(
        selected_xapps=selected,
        requested_bounds=list(fixture.requested_bounds or []),
        scenario_tags=scenario_tags,
        check_risk_before=False,
        use_l3=False,
    )
    notes.extend(bound_notes)
    return MethodPlanOutput(
        output_xapps=selected,
        output_bounds=bounds,
        projected=True,
        projection_notes=notes,
        parameter_bound_repaired=repaired,
        raw_action="oreo_like",
    )


def _pacifista_like_method(
    *,
    fixture: CandidateRepairFixture,
    raw_check: FeasibilityCheck,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
    scenario_tags: list[str],
) -> MethodPlanOutput:
    notes: list[str] = ["pacifista_like:conflict_profile"]
    if raw_check.hard_conflict:
        return MethodPlanOutput(candidate_rejected=True, projection_notes=notes + ["pacifista_like:reject_hard_conflict"])
    if raw_check.policy_violation:
        return MethodPlanOutput(candidate_rejected=True, projection_notes=notes + ["pacifista_like:reject_policy_conflict"])
    if raw_check.unsupported_parameter or "unknown_xapp" in raw_check.violation_types:
        return MethodPlanOutput(candidate_rejected=True, projection_notes=notes + ["pacifista_like:reject_schema_or_registry"])

    selected, canonical_notes = _canonicalize_registered_xapps(
        requested_xapps=list(fixture.requested_xapps or []),
        projector=projector,
        available_xapp_names=available_xapp_names,
        deduplicate_categories=True,
    )
    notes.extend(canonical_notes)
    selected = _resource_repair(
        selected_xapps=selected,
        projector=projector,
        goal=goal,
        scenario_tags=scenario_tags,
        notes=notes,
        mode="pacifista_like",
    )
    if len(selected) == 0:
        return MethodPlanOutput(candidate_rejected=True, projection_notes=notes + ["pacifista_like:reject_resource_overflow"])
    bounds, repaired, _, bound_notes = _normalize_bounds(
        selected_xapps=selected,
        requested_bounds=list(fixture.requested_bounds or []),
        scenario_tags=scenario_tags,
        check_risk_before=False,
        use_l3=False,
    )
    notes.extend(bound_notes)
    soft_arbitration = bool(raw_check.soft_conflict)
    if raw_check.risk_conflict:
        notes.append("pacifista_like:risk_penalty_monitor_only")
    return MethodPlanOutput(
        output_xapps=selected,
        output_bounds=bounds,
        projected=True,
        projection_notes=notes,
        parameter_bound_repaired=repaired,
        soft_arbitration_applied=soft_arbitration,
        raw_action="pacifista_like",
    )


def _conservative_fallback_method(
    *,
    fixture: CandidateRepairFixture,
    raw_check: FeasibilityCheck,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
) -> MethodPlanOutput:
    if raw_check.executable and _schema_valid_with_bounds(list(fixture.requested_xapps or []), list(fixture.requested_bounds or [])):
        return MethodPlanOutput(
            output_xapps=list(fixture.requested_xapps or []),
            output_bounds=list(fixture.requested_bounds or []),
            raw_action="conservative_passthrough",
        )
    selected = _fallback_plan(projector=projector, available_xapp_names=available_xapp_names, goal=goal)
    bounds, repaired, _, bound_notes = _normalize_bounds(
        selected_xapps=selected,
        requested_bounds=[],
        scenario_tags=[],
        check_risk_before=False,
        use_l3=False,
        conservative=True,
    )
    return MethodPlanOutput(
        output_xapps=selected,
        output_bounds=bounds,
        projected=True,
        fallback_used=True,
        projection_notes=["conservative_fallback:registry_defined_safe_plan"] + bound_notes,
        parameter_bound_repaired=repaired,
        raw_action="conservative_fallback",
    )


def _legacy_mode_output(
    *,
    mode: str,
    fixture: CandidateRepairFixture,
    initially_feasible: bool,
    schema_valid: bool,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
    observation: Observation,
) -> MethodPlanOutput:
    if mode == "deterministic":
        if not schema_valid:
            return MethodPlanOutput(candidate_rejected=True, projection_notes=["schema_bound:rejected_schema_invalid_candidate"])
        plan = projector.project_candidate(
            candidate_id=fixture.candidate_id,
            requested_xapps=list(fixture.requested_xapps or []),
            goal=goal,
            observation=observation,
            available_xapp_names=available_xapp_names,
        )
        summary = getattr(plan, "feasibility_summary", None)
        return MethodPlanOutput(
            output_xapps=list(getattr(plan, "x", []) or []),
            output_bounds=list(getattr(plan, "B", []) or []),
            projected=bool(getattr(summary, "projected", False)),
            projection_notes=list(getattr(summary, "projection_notes", []) or []),
            parameter_bound_repaired=bool(getattr(summary, "projected", False)),
            raw_action="legacy_deterministic",
        )
    if mode == "reject_only":
        output_xapps = _schema_only_output(
            requested_xapps=list(fixture.requested_xapps or []),
            projector=projector,
            available_xapp_names=available_xapp_names,
        )
        if not schema_valid or not initially_feasible:
            return MethodPlanOutput(
                candidate_rejected=True,
                projection_notes=["reject_only:rejected_infeasible_candidate"],
            )
        return MethodPlanOutput(output_xapps=output_xapps, output_bounds=list(fixture.requested_bounds or []), raw_action="reject_only")
    if mode == "schema_only":
        output_xapps = _schema_only_output(
            requested_xapps=list(fixture.requested_xapps or []),
            projector=projector,
            available_xapp_names=available_xapp_names,
        )
        return MethodPlanOutput(
            output_xapps=output_xapps,
            output_bounds=list(fixture.requested_bounds or []),
            projected=output_xapps != list(fixture.requested_xapps or []),
            projection_notes=["schema_only:canonicalized_or_dropped_schema_invalid_fields"] if output_xapps != list(fixture.requested_xapps or []) else [],
            raw_action="legacy_schema_only",
        )
    if mode == "no_projection":
        return MethodPlanOutput(
            output_xapps=_no_projection_output(list(fixture.requested_xapps or [])),
            output_bounds=list(fixture.requested_bounds or []),
            projection_notes=["no_projection:raw_candidate_passed_to_sandbox"],
            raw_action="legacy_no_projection",
        )
    raise ValueError(f"Unsupported legacy mode: {mode}")


def _apply_method(
    *,
    mode: str,
    fixture: CandidateRepairFixture,
    raw_check: FeasibilityCheck,
    initially_feasible: bool,
    schema_valid: bool,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
    observation: Observation,
) -> MethodPlanOutput:
    scenario_tags = list(fixture.scenario_tags or [])
    if mode in {"deterministic", "reject_only", "no_projection"}:
        return _legacy_mode_output(
            mode=mode,
            fixture=fixture,
            initially_feasible=initially_feasible,
            schema_valid=schema_valid,
            projector=projector,
            available_xapp_names=available_xapp_names,
            goal=goal,
            observation=observation,
        )
    if mode == "schema_only":
        if len(list(fixture.requested_bounds or [])) == 0 and fixture.chunk_id == "legacy":
            return _legacy_mode_output(
                mode=mode,
                fixture=fixture,
                initially_feasible=initially_feasible,
                schema_valid=schema_valid,
                projector=projector,
                available_xapp_names=available_xapp_names,
                goal=goal,
                observation=observation,
            )
        return _schema_only_method(fixture=fixture, projector=projector, available_xapp_names=available_xapp_names)
    if mode in {"repair_wo_memory", "repair_wo_l3", "full_repair"}:
        return _projective_repair_method(
            mode=mode,
            fixture=fixture,
            raw_check=raw_check,
            projector=projector,
            available_xapp_names=available_xapp_names,
            goal=goal,
            scenario_tags=scenario_tags,
        )
    if mode == "orchestran_like":
        return _orchestran_like_method(
            fixture=fixture,
            projector=projector,
            available_xapp_names=available_xapp_names,
            goal=goal,
            scenario_tags=scenario_tags,
        )
    if mode == "oreo_like":
        return _oreo_like_method(
            fixture=fixture,
            projector=projector,
            available_xapp_names=available_xapp_names,
            goal=goal,
            scenario_tags=scenario_tags,
        )
    if mode == "pacifista_like":
        return _pacifista_like_method(
            fixture=fixture,
            raw_check=raw_check,
            projector=projector,
            available_xapp_names=available_xapp_names,
            goal=goal,
            scenario_tags=scenario_tags,
        )
    if mode == "conservative_fallback":
        return _conservative_fallback_method(
            fixture=fixture,
            raw_check=raw_check,
            projector=projector,
            available_xapp_names=available_xapp_names,
            goal=goal,
        )
    raise ValueError(f"Unsupported repair evaluation mode: {mode}")


def _contract_volume_ratio(bounds: list[RewardParameterRange], xapps: list[str]) -> float:
    active = _active_projectable_keys(xapps)
    if len(active) == 0:
        return 0.0
    by_key = {_range_key(item): item for item in list(bounds or [])}
    ratios: list[float] = []
    for xapp_name, field_name in active:
        bound = by_key.get((xapp_name, field_name))
        policy = PARAM_POLICY_LIBRARY.get(field_name, {})
        if bound is None:
            ratios.append(0.0)
            continue
        if "values" in policy:
            allowed_values = list(policy.get("values", []) or [])
            values = list(getattr(bound, "discrete_values", []) or [])
            ratios.append(float(len(values)) / float(max(1, len(allowed_values))))
            continue
        lower = getattr(bound, "lower_bound", None)
        upper = getattr(bound, "upper_bound", None)
        if lower is None or upper is None:
            ratios.append(0.0)
            continue
        policy_width = float(policy.get("upper", 0.0) or 0.0) - float(policy.get("lower", 0.0) or 0.0)
        if policy_width <= 0.0:
            ratios.append(0.0)
            continue
        ratios.append(max(0.0, min(1.0, (float(upper) - float(lower)) / policy_width)))
    return sum(ratios) / float(len(ratios))


def _fallback_profile_valid(
    *,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
) -> bool:
    fallback_xapps = _fallback_plan(projector=projector, available_xapp_names=available_xapp_names, goal=goal)
    fallback_bounds, _, _, _ = _normalize_bounds(
        selected_xapps=fallback_xapps,
        requested_bounds=[],
        scenario_tags=[],
        check_risk_before=False,
        use_l3=False,
        conservative=True,
    )
    check = _check_feasibility(
        xapps=fallback_xapps,
        bounds=fallback_bounds,
        projector=projector,
        available_xapp_names=available_xapp_names,
        goal=goal,
        scenario_tags=[],
        soft_arbitration_applied=True,
        risk_bound_tightened=True,
    )
    return bool(check.executable)


def _compile_contract_metrics(
    *,
    mode: str,
    output: MethodPlanOutput,
    final_check: FeasibilityCheck,
    projector: OrchestrationFeasibilityProjector,
    available_xapp_names: list[str],
    goal: StructuredIntentGoal,
) -> ContractMetrics:
    if output.candidate_rejected or not final_check.executable:
        return ContractMetrics()
    active_keys = _active_projectable_keys(output.output_xapps)
    covered_keys = {_range_key(item) for item in list(output.output_bounds or [])}
    completeness = float(len(set(active_keys) & covered_keys)) / float(max(1, len(active_keys)))
    kpi_coverage = _protected_kpi_coverage(output.output_xapps, goal)
    fallback_valid = _fallback_profile_valid(
        projector=projector,
        available_xapp_names=available_xapp_names,
        goal=goal,
    )
    volume_ratio = _contract_volume_ratio(output.output_bounds, output.output_xapps)

    inconsistent = False
    if final_check.unsupported_parameter or final_check.empty_bound or final_check.unsafe_bounds:
        inconsistent = True
    if len(active_keys) > 0 and completeness < 0.50:
        inconsistent = True
    if mode == "schema_only" and len(output.output_bounds) == 0:
        inconsistent = True

    restriction_threshold = 0.40
    over_restriction = bool(volume_ratio < restriction_threshold or len(active_keys) <= 1)
    risk_unmitigated = bool(final_check.risk_conflict and not output.risk_bound_tightened)
    success = bool(
        final_check.executable
        and not inconsistent
        and not risk_unmitigated
        and completeness >= 0.70
        and kpi_coverage >= 0.50
        and fallback_valid
    )
    if mode == "schema_only" and not output.projected:
        success = bool(success and completeness >= 0.90)

    fallback_rate = 0.02
    if final_check.risk_conflict and not output.risk_bound_tightened:
        fallback_rate += 0.16
    if over_restriction:
        fallback_rate += 0.22
    if not success:
        fallback_rate += 0.15
    if output.fallback_used:
        fallback_rate += 0.18
    protected_uncontrolled = bool(kpi_coverage < 0.50 or not success)
    blocked_rate = 0.0
    if over_restriction:
        blocked_rate = min(1.0, 0.25 + max(0.0, restriction_threshold - volume_ratio))
    elif final_check.risk_conflict and not output.risk_bound_tightened:
        blocked_rate = 0.12
    return ContractMetrics(
        success=success,
        projectable_constraint_completeness=completeness,
        kpi_guardrail_coverage=kpi_coverage,
        fallback_profile_valid=fallback_valid,
        admissible_region_volume_ratio=volume_ratio,
        over_restriction=over_restriction,
        inconsistency=inconsistent,
        fallback_rate_sanity=min(1.0, fallback_rate),
        protected_kpi_uncontrolled=protected_uncontrolled,
        near_rt_action_blocked_rate=min(1.0, blocked_rate),
    )


def _outcome_bucket(
    *,
    raw_check: FeasibilityCheck,
    final_check: FeasibilityCheck,
    output: MethodPlanOutput,
) -> str:
    if final_check.executable and raw_check.executable and not output.projected and not output.fallback_used:
        return "executable_without_repair"
    if final_check.executable and (not raw_check.executable or output.projected or output.fallback_used):
        return "repaired_and_executable"
    if not output.candidate_rejected and not final_check.executable:
        if final_check.hard_conflict:
            return "residual_hard_conflict"
        if final_check.resource_overflow:
            return "residual_resource_overflow"
        if final_check.policy_violation:
            return "residual_policy_violation"
        if final_check.empty_bound:
            return "residual_empty_bounds"
        return "residual_other"
    violations = set(raw_check.violation_types + final_check.violation_types)
    if "hard_mutex_conflict" in violations:
        return "rejected_due_to_hard_conflict"
    if "empty_bound" in violations:
        return "rejected_due_to_empty_bounds"
    if "resource_overflow" in violations or "cardinality_overflow" in violations:
        return "rejected_due_to_resource_overflow"
    if "policy_violation" in violations:
        return "rejected_due_to_policy_violation"
    return "rejected_other"


def _bounds_for_candidate(xapps: list[str], *, rng: random.Random, unsafe: bool = False) -> list[RewardParameterRange]:
    bounds: list[RewardParameterRange] = []
    for xapp_name, field_name in _active_projectable_keys(xapps):
        if field_name not in {
            "steering_aggressiveness",
            "steering_hysteresis_db",
            "power_cap_dbm",
            "latency_bias",
            "scheduler_policy_hint",
            "beam_candidate_preference",
            "traffic_steering_preference",
            "cell_sleep_mode",
            "muting_ratio",
        }:
            continue
        policy = PARAM_POLICY_LIBRARY[field_name]
        if "values" in policy:
            values = list(policy.get("values", []) or [])
            if unsafe and field_name == "cell_sleep_mode":
                values = ["deep_sleep"]
            elif rng.random() < 0.50:
                values = values[: max(1, min(len(values), 2))]
            bounds.append(_make_raw_range(xapp_name=xapp_name, field_name=field_name, values=values))
            continue
        lower = float(policy.get("lower", 0.0) or 0.0)
        upper = float(policy.get("upper", lower) or lower)
        if unsafe and field_name == "steering_aggressiveness":
            upper = 0.95
        elif unsafe and field_name == "steering_hysteresis_db":
            lower = 0.5
        elif unsafe and field_name == "power_cap_dbm":
            lower = 20.0
        else:
            span = max(0.01, upper - lower)
            lower = lower + rng.random() * span * 0.15
            upper = upper - rng.random() * span * 0.10
        bounds.append(_make_raw_range(xapp_name=xapp_name, field_name=field_name, lower=round(lower, 4), upper=round(upper, 4)))
    return bounds


def _base_categories_for_context(
    *,
    intent: Exp2IntentTemplate,
    chunk: Exp2ChunkDescriptor,
    scenario: Exp2ScenarioDescriptor,
) -> list[str]:
    tags = set(chunk.tags + scenario.tags)
    categories: list[str] = []
    if intent.intent_family == "throughput_latency_tradeoff":
        categories.extend(["massive_mimo_beamformer_xapp", "spectrum_optimizer_xapp", "latency_scheduler_xapp", "urllc_guard_xapp"])
    elif intent.intent_family == "urllc_reliability":
        categories.extend(["urllc_guard_xapp", "latency_scheduler_xapp", "ran_slicing_manager_xapp"])
    else:
        categories.extend(["power_saving_xapp", "mobility_predictor_xapp", "traffic_steering_xapp"])

    if "high_mobility" in tags or "handover_edge" in tags:
        categories.extend(["mobility_predictor_xapp", "traffic_steering_xapp"])
    if "interference_burst" in tags or "cell_edge_users" in tags:
        categories.extend(["spectrum_optimizer_xapp", "massive_mimo_beamformer_xapp"])
    if "energy_saving" in tags or "low_load" in tags:
        categories.extend(["power_saving_xapp", "cell_sleep_xapp"])
    if "urllc_burst" in tags or "premium_tenant" in tags:
        categories.extend(["urllc_guard_xapp", "latency_scheduler_xapp"])
    if "blockage" in tags or "beam_recovery" in tags:
        categories.extend(["massive_mimo_beamformer_xapp", "traffic_steering_xapp"])
    if "slice_sla" in tags or "mixed_slice" in tags:
        categories.append("ran_slicing_manager_xapp")
    return _unique_preserving_order(categories)


def _variant_token_for_category(category: str, *, tags: set[str], rng: random.Random) -> str:
    if category == "traffic_steering_xapp":
        if "high_mobility" in tags:
            return "load_first" if rng.random() < 0.35 else "balanced"
        if "hotspot_shift" in tags or "load_balance" in tags:
            return "hotspot_relief"
        return "balanced"
    if category == "mobility_predictor_xapp":
        if "high_mobility" in tags:
            return "handover_push" if rng.random() < 0.35 else "high_speed"
        return "balanced"
    if category == "spectrum_optimizer_xapp":
        if "interference_burst" in tags:
            return "aggressive" if rng.random() < 0.30 else "interference_averse"
        return "balanced"
    if category == "massive_mimo_beamformer_xapp":
        if "blockage" in tags or "cell_edge_users" in tags:
            return "capacity_push" if rng.random() < 0.30 else "coverage_boost"
        return "balanced"
    if category == "latency_scheduler_xapp":
        return "urllc_fastpath" if "urllc_burst" in tags else "balanced"
    if category == "urllc_guard_xapp":
        return "strict" if "urllc_burst" in tags else "balanced"
    if category == "power_saving_xapp":
        return "energy_max" if "energy_saving" in tags and rng.random() < 0.35 else "balanced"
    if category == "cell_sleep_xapp":
        return "deep_saver" if "energy_saving" in tags and rng.random() < 0.35 else "balanced"
    return "balanced"


def _natural_candidate(
    *,
    candidate_id: str,
    chunk: Exp2ChunkDescriptor,
    intent: Exp2IntentTemplate,
    scenario: Exp2ScenarioDescriptor,
    rng: random.Random,
    available_xapps: list[str],
) -> CandidateRepairFixture:
    del available_xapps
    tags = set(chunk.tags + scenario.tags)
    categories = _base_categories_for_context(intent=intent, chunk=chunk, scenario=scenario)
    rng.shuffle(categories)
    width = rng.randint(3, min(5, max(3, len(categories))))
    selected_categories = categories[:width]
    selected_xapps = [
        _choose_variant(category, _variant_token_for_category(category, tags=tags, rng=rng))
        for category in selected_categories
    ]
    if rng.random() < 0.08:
        selected_xapps.append("nonexistent_magic_scheduler_xapp")
    if rng.random() < 0.10 and "traffic_steering_xapp" in selected_categories:
        selected_xapps = [item for item in selected_xapps if canonical_xapp_category(item) != "mobility_predictor_xapp"]
    unsafe = bool(("high_mobility" in tags or "interference_burst" in tags) and rng.random() < 0.18)
    bounds = _bounds_for_candidate(selected_xapps, rng=rng, unsafe=unsafe)
    issue_type = "natural_planner"
    if unsafe:
        issue_type = "natural_unsafe_bounds"
    return CandidateRepairFixture(
        candidate_id=candidate_id,
        issue_type=issue_type,
        requested_xapps=selected_xapps,
        requested_bounds=bounds,
        chunk_id=chunk.chunk_id,
        intent_id=intent.intent_id,
        intent_slug=intent.intent_slug,
        intent_family=intent.intent_family,
        scenario_id=f"{chunk.chunk_id}:{scenario.scenario_id}",
        scenario_tags=list(_unique_preserving_order(chunk.tags + scenario.tags)),
        candidate_source="natural_planner",
        stress_error_types=[],
        notes="Deterministic natural planner surrogate candidate.",
    )


def _controlled_stress_candidate(
    *,
    candidate_id: str,
    chunk: Exp2ChunkDescriptor,
    intent: Exp2IntentTemplate,
    scenario: Exp2ScenarioDescriptor,
    error_type: str,
    rng: random.Random,
) -> CandidateRepairFixture:
    tags = set(chunk.tags + scenario.tags)
    base_categories = _base_categories_for_context(intent=intent, chunk=chunk, scenario=scenario)
    selected_xapps = [
        _choose_variant(category, _variant_token_for_category(category, tags=tags, rng=rng))
        for category in base_categories[:4]
    ]
    bounds = _bounds_for_candidate(selected_xapps, rng=rng, unsafe=False)

    if error_type == "unregistered_xapp":
        selected_xapps.append("nonexistent_magic_scheduler_xapp")
    elif error_type == "unsupported_parameter":
        if len(selected_xapps) == 0:
            selected_xapps.append(_choose_variant("latency_scheduler_xapp", "balanced"))
        bounds.append(
            _make_raw_range(
                xapp_name=selected_xapps[0],
                field_name="quantum_scheduler_hint",
                lower=0.0,
                upper=1.0,
            )
        )
    elif error_type == "missing_dependency":
        selected_xapps = [
            _choose_variant("traffic_steering_xapp", "hotspot_relief"),
            _choose_variant("urllc_guard_xapp", "balanced"),
        ]
        bounds = _bounds_for_candidate(selected_xapps, rng=rng, unsafe=False)
    elif error_type == "hard_actuator_conflict":
        selected_xapps = [
            _choose_variant("cell_sleep_xapp", "aggressive"),
            _choose_variant("power_saving_xapp", "balanced"),
            _choose_variant("latency_scheduler_xapp", "urllc_fastpath"),
            _choose_variant("urllc_guard_xapp", "strict"),
        ]
        bounds = _bounds_for_candidate(selected_xapps, rng=rng, unsafe=False)
    elif error_type == "soft_kpi_conflict":
        selected_xapps = [
            _choose_variant("power_saving_xapp", "energy_max"),
            _choose_variant("latency_scheduler_xapp", "tail_guard"),
            _choose_variant("ran_slicing_manager_xapp", "sla_strict"),
        ]
        bounds = _bounds_for_candidate(selected_xapps, rng=rng, unsafe=False)
    elif error_type == "risk_conflict":
        selected_xapps = [
            _choose_variant("traffic_steering_xapp", "load_first"),
            _choose_variant("mobility_predictor_xapp", "handover_push"),
            _choose_variant("massive_mimo_beamformer_xapp", "capacity_push"),
        ]
        bounds = _bounds_for_candidate(selected_xapps, rng=rng, unsafe=True)
        tags.add("high_mobility")
    elif error_type == "resource_overflow":
        selected_xapps = [
            _choose_variant("mobility_predictor_xapp", "high_speed"),
            _choose_variant("traffic_steering_xapp", "hotspot_relief"),
            _choose_variant("latency_scheduler_xapp", "urllc_fastpath"),
            _choose_variant("urllc_guard_xapp", "strict"),
            _choose_variant("massive_mimo_beamformer_xapp", "capacity_push"),
            _choose_variant("spectrum_optimizer_xapp", "aggressive"),
            _choose_variant("ran_slicing_manager_xapp", "sla_strict"),
        ]
        bounds = _bounds_for_candidate(selected_xapps, rng=rng, unsafe=False)
    elif error_type == "unsafe_bounds":
        if not any(canonical_xapp_category(item) == "traffic_steering_xapp" for item in selected_xapps):
            selected_xapps.append(_choose_variant("traffic_steering_xapp", "hotspot_relief"))
        if not any(canonical_xapp_category(item) == "mobility_predictor_xapp" for item in selected_xapps):
            selected_xapps.append(_choose_variant("mobility_predictor_xapp", "high_speed"))
        bounds = _bounds_for_candidate(selected_xapps, rng=rng, unsafe=True)

    return CandidateRepairFixture(
        candidate_id=candidate_id,
        issue_type=error_type,
        requested_xapps=_unique_preserving_order(selected_xapps),
        requested_bounds=bounds,
        chunk_id=chunk.chunk_id,
        intent_id=intent.intent_id,
        intent_slug=intent.intent_slug,
        intent_family=intent.intent_family,
        scenario_id=f"{chunk.chunk_id}:{scenario.scenario_id}",
        scenario_tags=list(_unique_preserving_order(list(tags))),
        candidate_source="controlled_stress",
        stress_error_types=[error_type],
        notes=f"Controlled stress fixture for {error_type}.",
    )


def build_exp2_candidate_pool(
    *,
    candidates_per_context: int = 20,
    seed: int = 20260517,
    available_xapp_names: list[str] | None = None,
) -> list[CandidateRepairFixture]:
    available_xapps = list(available_xapp_names or build_default_available_xapps())
    fixtures: list[CandidateRepairFixture] = []
    chunks = build_exp2_chunk_descriptors()
    intents = build_exp2_intent_templates()
    scenarios = build_exp2_scenario_descriptors()
    h = max(1, int(candidates_per_context))
    natural_count = max(0, h - len(CONTROLLED_STRESS_ERRORS))
    for chunk in chunks:
        for intent in intents:
            for scenario in scenarios:
                context_key = f"{chunk.chunk_id}:{intent.intent_slug}:{scenario.scenario_id}"
                rng = random.Random(f"{seed}:{context_key}")
                for index in range(natural_count):
                    fixtures.append(
                        _natural_candidate(
                            candidate_id=f"{context_key}:nat{index:02d}",
                            chunk=chunk,
                            intent=intent,
                            scenario=scenario,
                            rng=rng,
                            available_xapps=available_xapps,
                        )
                    )
                for index in range(h - natural_count):
                    error_type = CONTROLLED_STRESS_ERRORS[index % len(CONTROLLED_STRESS_ERRORS)]
                    fixtures.append(
                        _controlled_stress_candidate(
                            candidate_id=f"{context_key}:stress{index:02d}:{error_type}",
                            chunk=chunk,
                            intent=intent,
                            scenario=scenario,
                            error_type=error_type,
                            rng=rng,
                        )
                    )
    return fixtures


def _goal_for_fixture(fixture: CandidateRepairFixture, fallback_goal: StructuredIntentGoal) -> StructuredIntentGoal:
    intent_lookup = {item.intent_id: item for item in build_exp2_intent_templates()}
    intent = intent_lookup.get(str(fixture.intent_id or ""))
    if intent is None:
        return fallback_goal
    return structured_goal_from_exp2_intent(intent, chunk_id=str(fixture.chunk_id or "legacy"))


def _observation_for_fixture(fixture: CandidateRepairFixture, fallback_observation: Observation) -> Observation:
    scenario_suffix = str(fixture.scenario_id or "").split(":")[-1]
    for scenario in build_exp2_scenario_descriptors():
        if scenario.scenario_id == scenario_suffix:
            return observation_for_scenario(scenario)
    return fallback_observation


def evaluate_candidate_repair_modes(
    *,
    modes: list[str] | None = None,
    fixtures: list[CandidateRepairFixture] | None = None,
    projector: OrchestrationFeasibilityProjector | None = None,
    available_xapp_names: list[str] | None = None,
    goal: StructuredIntentGoal | None = None,
    observation: Observation | None = None,
    use_exp2_pool: bool = False,
    candidates_per_context: int = 20,
    seed: int = 20260517,
) -> tuple[list[CandidateRepairModeResult], dict[str, dict[str, object]]]:
    selected_modes = [str(item).strip() for item in list(modes or DEFAULT_REPAIR_MODES) if str(item).strip() != ""]
    for mode in selected_modes:
        if mode not in set(ALL_REPAIR_MODES):
            raise ValueError(f"Unsupported repair evaluation mode: {mode}")

    available_names = list(available_xapp_names or build_default_available_xapps())
    if fixtures is not None:
        candidate_fixtures = list(fixtures)
    elif use_exp2_pool or any(mode in set(EXP2_REPAIR_MODES) - {"schema_only"} for mode in selected_modes):
        candidate_fixtures = build_exp2_candidate_pool(
            candidates_per_context=candidates_per_context,
            seed=seed,
            available_xapp_names=available_names,
        )
    else:
        candidate_fixtures = build_default_candidate_fixtures()

    active_projector = projector or OrchestrationFeasibilityProjector(
        max_selected_xapps=4,
        max_cpu_cost=4.6,
        max_mem_cost=4.2,
    )
    default_goal = goal or build_default_goal()
    default_observation = observation or build_default_observation()

    results: list[CandidateRepairModeResult] = []
    for mode in selected_modes:
        normalized_mode = str(mode or "").strip().lower()
        for fixture in candidate_fixtures:
            active_goal = goal or _goal_for_fixture(fixture, default_goal)
            active_observation = observation or _observation_for_fixture(fixture, default_observation)
            requested_xapps = list(fixture.requested_xapps or [])
            requested_bounds = list(fixture.requested_bounds or [])
            raw_check = _check_feasibility(
                xapps=requested_xapps,
                bounds=requested_bounds,
                projector=active_projector,
                available_xapp_names=available_names,
                goal=active_goal,
                scenario_tags=list(fixture.scenario_tags or []),
                soft_arbitration_applied=False,
                risk_bound_tightened=False,
                ignore_missing_contract_bounds=len(requested_bounds) == 0,
            )
            initially_feasible = bool(raw_check.executable)
            schema_valid = _schema_valid_with_bounds(requested_xapps, requested_bounds) if len(requested_bounds) > 0 else _schema_valid(requested_xapps)

            started_at = time.perf_counter()
            output = _apply_method(
                mode=normalized_mode,
                fixture=fixture,
                raw_check=raw_check,
                initially_feasible=initially_feasible,
                schema_valid=schema_valid,
                projector=active_projector,
                available_xapp_names=available_names,
                goal=active_goal,
                observation=active_observation,
            )
            latency_ms = (time.perf_counter() - started_at) * 1000.0

            final_check = _check_feasibility(
                xapps=output.output_xapps,
                bounds=output.output_bounds,
                projector=active_projector,
                available_xapp_names=available_names,
                goal=active_goal,
                scenario_tags=list(fixture.scenario_tags or []),
                soft_arbitration_applied=output.soft_arbitration_applied,
                risk_bound_tightened=output.risk_bound_tightened,
                ignore_missing_contract_bounds=normalized_mode in {"deterministic", "reject_only", "no_projection"} and len(output.output_bounds) == 0,
            )
            final_feasible = bool(final_check.executable and not output.candidate_rejected)
            contract = _compile_contract_metrics(
                mode=normalized_mode,
                output=output,
                final_check=final_check,
                projector=active_projector,
                available_xapp_names=available_names,
                goal=active_goal,
            )
            hard_conflict_escape = bool((not output.candidate_rejected) and final_check.hard_conflict)
            resource_overflow_escape = bool((not output.candidate_rejected) and final_check.resource_overflow)
            unknown_xapp_escape = bool((not output.candidate_rejected) and "unknown_xapp" in final_check.violation_types)
            repair_success = bool((not initially_feasible) and final_feasible)
            intent_coverage = _intent_coverage(output.output_xapps, active_goal)
            protected_coverage = _protected_kpi_coverage(output.output_xapps, active_goal)
            predicted_loss = _predicted_intent_loss(
                xapps=output.output_xapps,
                check=final_check,
                goal=active_goal,
                fallback_used=output.fallback_used,
            )
            utility_preserving = bool(final_feasible and intent_coverage >= 0.55 and predicted_loss <= 0.45)
            outcome = _outcome_bucket(raw_check=raw_check, final_check=final_check, output=output)

            results.append(
                CandidateRepairModeResult(
                    mode=normalized_mode,
                    candidate_id=fixture.candidate_id,
                    issue_type=fixture.issue_type,
                    requested_xapps=requested_xapps,
                    output_xapps=list(output.output_xapps or []),
                    raw_violation_types=list(raw_check.violation_types),
                    output_violation_types=list(final_check.violation_types),
                    schema_valid=schema_valid,
                    initially_feasible=initially_feasible,
                    final_feasible=final_feasible,
                    candidate_rejected=bool(output.candidate_rejected),
                    projected=bool(output.projected),
                    repair_success=repair_success,
                    hard_conflict_escape=hard_conflict_escape,
                    resource_overflow_escape=resource_overflow_escape,
                    unknown_xapp_escape=unknown_xapp_escape,
                    average_repair_distance=_repair_distance(requested_xapps, list(output.output_xapps or [])),
                    repair_latency_ms=latency_ms,
                    projection_notes=list(output.projection_notes),
                    requested_bounds=requested_bounds,
                    output_bounds=list(output.output_bounds or []),
                    chunk_id=fixture.chunk_id,
                    scenario_id=fixture.scenario_id,
                    scenario_tags=list(fixture.scenario_tags or []),
                    intent_id=fixture.intent_id,
                    intent_slug=fixture.intent_slug,
                    intent_family=fixture.intent_family,
                    candidate_source=fixture.candidate_source,
                    stress_error_types=list(fixture.stress_error_types or []),
                    fallback_used=bool(output.fallback_used),
                    unsupported_parameter_residual=bool(final_check.unsupported_parameter),
                    empty_bound_residual=bool(final_check.empty_bound),
                    unsafe_bound_residual=bool(final_check.unsafe_bounds),
                    policy_violation_residual=bool(final_check.policy_violation),
                    soft_conflict_present=bool(raw_check.soft_conflict or final_check.soft_conflict),
                    soft_arbitration_applied=bool(output.soft_arbitration_applied),
                    risk_conflict_present=bool(raw_check.risk_conflict or final_check.risk_conflict),
                    risk_bound_tightened=bool(output.risk_bound_tightened),
                    l2_penalty_applied=bool(output.l2_penalty_applied),
                    parameter_bound_repaired=bool(output.parameter_bound_repaired),
                    protected_kpi_coverage=protected_coverage,
                    intent_coverage=intent_coverage,
                    predicted_intent_loss=predicted_loss,
                    utility_preserving_executable=utility_preserving,
                    contract_success=contract.success,
                    projectable_constraint_completeness=contract.projectable_constraint_completeness,
                    kpi_guardrail_coverage=contract.kpi_guardrail_coverage,
                    fallback_profile_valid=contract.fallback_profile_valid,
                    admissible_region_volume_ratio=contract.admissible_region_volume_ratio,
                    over_restriction=contract.over_restriction,
                    contract_inconsistency=contract.inconsistency,
                    sanity_fallback_rate=contract.fallback_rate_sanity,
                    protected_kpi_uncontrolled=contract.protected_kpi_uncontrolled,
                    near_rt_action_blocked_rate=contract.near_rt_action_blocked_rate,
                    outcome_bucket=outcome,
                )
            )
    return results, summarize_candidate_repair_results(results)


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if len(union) == 0:
        return 1.0
    return float(len(left & right)) / float(len(union))


def _bound_iou(left: RewardParameterRange | None, right: RewardParameterRange | None) -> float:
    if left is None and right is None:
        return 1.0
    if left is None or right is None:
        return 0.0
    field_name = str(getattr(left, "field_name", "") or "")
    if field_name != str(getattr(right, "field_name", "") or ""):
        return 0.0
    left_values = set(getattr(left, "discrete_values", []) or [])
    right_values = set(getattr(right, "discrete_values", []) or [])
    if len(left_values) > 0 or len(right_values) > 0:
        union = left_values | right_values
        return float(len(left_values & right_values)) / float(max(1, len(union)))
    left_lower = getattr(left, "lower_bound", None)
    left_upper = getattr(left, "upper_bound", None)
    right_lower = getattr(right, "lower_bound", None)
    right_upper = getattr(right, "upper_bound", None)
    if None in {left_lower, left_upper, right_lower, right_upper}:
        return 0.0
    intersection = max(0.0, min(float(left_upper), float(right_upper)) - max(float(left_lower), float(right_lower)))
    union = max(float(left_upper), float(right_upper)) - min(float(left_lower), float(right_lower))
    if union <= 0.0:
        return 0.0
    return intersection / union


def _diversity_metrics(mode_results: list[CandidateRepairModeResult], *, max_pairs: int = 50000) -> dict[str, float]:
    feasible = [item for item in list(mode_results or []) if item.final_feasible and len(item.output_xapps) > 0]
    n = len(feasible)
    if n <= 1:
        return {"xapp_diversity": 0.0, "bound_diversity": 0.0, "plan_diversity": 0.0, "diversity_pair_count": 0.0}
    total_pairs = n * (n - 1) // 2
    stride = max(1, int(math.ceil(total_pairs / max(1, max_pairs))))
    pair_index = 0
    used_pairs = 0
    xapp_distance_sum = 0.0
    bound_distance_sum = 0.0
    for i in range(n):
        left = feasible[i]
        left_xapps = set(_canonical_categories(left.output_xapps))
        left_bounds = {_category_range_key(item): item for item in left.output_bounds}
        for j in range(i + 1, n):
            pair_index += 1
            if (pair_index - 1) % stride != 0:
                continue
            right = feasible[j]
            right_xapps = set(_canonical_categories(right.output_xapps))
            xapp_distance_sum += 1.0 - _jaccard_similarity(left_xapps, right_xapps)
            right_bounds = {_category_range_key(item): item for item in right.output_bounds}
            keys = set(left_bounds.keys()) | set(right_bounds.keys())
            if len(keys) == 0:
                bound_distance = 0.0
            else:
                bound_distance = sum(1.0 - _bound_iou(left_bounds.get(key), right_bounds.get(key)) for key in keys) / float(len(keys))
            bound_distance_sum += bound_distance
            used_pairs += 1
    if used_pairs == 0:
        return {"xapp_diversity": 0.0, "bound_diversity": 0.0, "plan_diversity": 0.0, "diversity_pair_count": 0.0}
    xapp_diversity = xapp_distance_sum / float(used_pairs)
    bound_diversity = bound_distance_sum / float(used_pairs)
    return {
        "xapp_diversity": xapp_diversity,
        "bound_diversity": bound_diversity,
        "plan_diversity": 0.6 * xapp_diversity + 0.4 * bound_diversity,
        "diversity_pair_count": float(used_pairs),
    }


def summarize_candidate_repair_results(
    results: list[CandidateRepairModeResult],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[CandidateRepairModeResult]] = {}
    for result in list(results or []):
        grouped.setdefault(result.mode, []).append(result)

    summary: dict[str, dict[str, object]] = {}
    for mode, mode_results in sorted(grouped.items()):
        denominator = max(1, len(mode_results))
        infeasible_inputs = [item for item in mode_results if not item.initially_feasible]
        repair_denominator = max(1, len(infeasible_inputs))
        soft_cases = [item for item in mode_results if item.soft_conflict_present]
        risk_cases = [item for item in mode_results if item.risk_conflict_present]
        feasible_results = [item for item in mode_results if item.final_feasible]
        latencies = [float(item.repair_latency_ms) for item in mode_results]
        diversity = _diversity_metrics(mode_results)
        summary[mode] = {
            "repair_eval_version": REPAIR_EVAL_VERSION,
            "method_label": METHOD_LABELS.get(mode, mode),
            "candidate_count": int(len(mode_results)),
            "schema_validity_rate": sum(1 for item in mode_results if item.schema_valid) / denominator,
            "initial_feasible_rate": sum(1 for item in mode_results if item.initially_feasible) / denominator,
            "feasible_candidate_rate": sum(1 for item in mode_results if item.final_feasible) / denominator,
            "final_executable_plan_ratio": sum(1 for item in mode_results if item.final_feasible) / denominator,
            "candidate_rejection_rate": sum(1 for item in mode_results if item.candidate_rejected) / denominator,
            "repair_success_rate": sum(1 for item in infeasible_inputs if item.repair_success) / repair_denominator,
            "projection_applied_rate": sum(1 for item in mode_results if item.projected) / denominator,
            "hard_conflict_escape_rate": sum(1 for item in mode_results if item.hard_conflict_escape) / denominator,
            "hard_conflict_residual_rate": sum(1 for item in mode_results if item.hard_conflict_escape) / denominator,
            "resource_overflow_escape_rate": sum(1 for item in mode_results if item.resource_overflow_escape) / denominator,
            "resource_overflow_residual_rate": sum(1 for item in mode_results if item.resource_overflow_escape) / denominator,
            "unknown_xapp_escape_rate": sum(1 for item in mode_results if item.unknown_xapp_escape) / denominator,
            "unsupported_parameter_residual_rate": sum(1 for item in mode_results if item.unsupported_parameter_residual) / denominator,
            "empty_bound_rate": sum(1 for item in mode_results if item.empty_bound_residual) / denominator,
            "unsafe_bound_residual_rate": sum(1 for item in mode_results if item.unsafe_bound_residual) / denominator,
            "policy_violation_residual_rate": sum(1 for item in mode_results if item.policy_violation_residual) / denominator,
            "parameter_bound_repair_rate": sum(1 for item in mode_results if item.parameter_bound_repaired) / denominator,
            "soft_conflict_arbitration_coverage": (
                sum(1 for item in soft_cases if item.soft_arbitration_applied) / max(1, len(soft_cases))
            ),
            "risk_bound_tightening_rate": (
                sum(1 for item in risk_cases if item.risk_bound_tightened) / max(1, len(risk_cases))
            ),
            "l2_penalty_applied_rate": sum(1 for item in mode_results if item.l2_penalty_applied) / denominator,
            "protected_kpi_coverage": sum(float(item.protected_kpi_coverage) for item in mode_results) / denominator,
            "mean_intent_coverage": sum(float(item.intent_coverage) for item in mode_results) / denominator,
            "mean_predicted_intent_loss": sum(float(item.predicted_intent_loss) for item in mode_results) / denominator,
            "utility_preserving_executable_ratio": (
                sum(1 for item in mode_results if item.utility_preserving_executable) / denominator
            ),
            "contract_compilation_success_rate": sum(1 for item in mode_results if item.contract_success) / denominator,
            "contract_success_rate": sum(1 for item in mode_results if item.contract_success) / denominator,
            "projectable_constraint_completeness": (
                sum(float(item.projectable_constraint_completeness) for item in feasible_results)
                / max(1, len(feasible_results))
            ),
            "kpi_guardrail_coverage": (
                sum(float(item.kpi_guardrail_coverage) for item in feasible_results) / max(1, len(feasible_results))
            ),
            "fallback_profile_validity_rate": (
                sum(1 for item in feasible_results if item.fallback_profile_valid) / max(1, len(feasible_results))
            ),
            "admissible_region_volume_ratio": (
                sum(float(item.admissible_region_volume_ratio) for item in feasible_results)
                / max(1, len(feasible_results))
            ),
            "over_restriction_rate": sum(1 for item in mode_results if item.over_restriction) / denominator,
            "contract_inconsistency_rate": sum(1 for item in mode_results if item.contract_inconsistency) / denominator,
            "mean_sanity_fallback_rate": sum(float(item.sanity_fallback_rate) for item in mode_results) / denominator,
            "protected_kpi_uncontrolled_rate": sum(1 for item in mode_results if item.protected_kpi_uncontrolled) / denominator,
            "near_rt_action_blocked_rate": sum(float(item.near_rt_action_blocked_rate) for item in mode_results) / denominator,
            "average_repair_distance": sum(float(item.average_repair_distance) for item in mode_results) / denominator,
            "mean_repair_latency_ms": sum(latencies) / max(1, len(latencies)),
            "p95_repair_latency_ms": _percentile(latencies, 0.95),
            "mean_output_xapp_count": sum(len(item.output_xapps) for item in mode_results) / denominator,
            **diversity,
        }
    return summary


def _summary_records(summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return [{"mode": str(mode), **dict(values)} for mode, values in sorted(dict(summary or {}).items())]


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if len(records) == 0:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _make_main_table(summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    columns = [
        "final_executable_plan_ratio",
        "repair_success_rate",
        "hard_conflict_residual_rate",
        "resource_overflow_residual_rate",
        "soft_conflict_arbitration_coverage",
        "protected_kpi_coverage",
        "contract_compilation_success_rate",
        "plan_diversity",
    ]
    return [
        {
            "method": METHOD_LABELS.get(mode, mode),
            **{column: round(float(values.get(column, 0.0) or 0.0), 6) for column in columns},
        }
        for mode, values in sorted(summary.items())
    ]


def _make_contract_table(summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    columns = [
        "contract_compilation_success_rate",
        "projectable_constraint_completeness",
        "kpi_guardrail_coverage",
        "fallback_profile_validity_rate",
        "admissible_region_volume_ratio",
        "over_restriction_rate",
        "contract_inconsistency_rate",
        "utility_preserving_executable_ratio",
    ]
    return [
        {
            "method": METHOD_LABELS.get(mode, mode),
            **{column: round(float(values.get(column, 0.0) or 0.0), 6) for column in columns},
        }
        for mode, values in sorted(summary.items())
    ]


def _plot_candidate_outcomes(results: list[CandidateRepairModeResult], figures_dir: Path) -> None:
    plt = configure_matplotlib()
    modes = [mode for mode in EXP2_REPAIR_MODES if any(item.mode == mode for item in results)]
    if len(modes) == 0:
        modes = sorted({item.mode for item in results})
    buckets = [
        "executable_without_repair",
        "repaired_and_executable",
        "rejected_due_to_hard_conflict",
        "rejected_due_to_empty_bounds",
        "rejected_due_to_resource_overflow",
        "rejected_due_to_policy_violation",
        "residual_other",
        "rejected_other",
    ]
    colors = ["#5E8C61", "#89B66D", "#C8553D", "#E59F3A", "#B95C7D", "#7A5C99", "#8A8F98", "#C3C7CE"]
    counts_by_mode: dict[str, dict[str, float]] = {}
    for mode in modes:
        mode_results = [item for item in results if item.mode == mode]
        denom = max(1, len(mode_results))
        counts_by_mode[mode] = {
            bucket: sum(1 for item in mode_results if item.outcome_bucket == bucket) / denom
            for bucket in buckets
        }
    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    bottom = [0.0 for _ in modes]
    x_positions = list(range(len(modes)))
    for bucket, color in zip(buckets, colors):
        values = [counts_by_mode[mode][bucket] for mode in modes]
        ax.bar(x_positions, values, bottom=bottom, label=bucket.replace("_", " "), color=color)
        bottom = [left + value for left, value in zip(bottom, values)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS.get(mode, mode) for mode in modes], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Candidate fraction")
    ax.set_title("Experiment 2 candidate outcomes by repair method")
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    fig.savefig(figures_dir / "exp2_candidate_outcome_stacked.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures_dir / "exp2_candidate_outcome_stacked.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_residual_violations(summary: dict[str, dict[str, object]], figures_dir: Path) -> None:
    plt = configure_matplotlib()
    modes = [mode for mode in EXP2_REPAIR_MODES if mode in summary]
    metrics = [
        ("hard_conflict_residual_rate", "Hard conflict"),
        ("resource_overflow_residual_rate", "Resource overflow"),
        ("policy_violation_residual_rate", "Policy violation"),
        ("contract_inconsistency_rate", "Contract inconsistency"),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    width = 0.18
    x_positions = list(range(len(modes)))
    colors = ["#C8553D", "#E59F3A", "#7A5C99", "#556070"]
    for offset, (metric, label) in enumerate(metrics):
        values = [float(summary[mode].get(metric, 0.0) or 0.0) for mode in modes]
        ax.bar([x + (offset - 1.5) * width for x in x_positions], values, width=width, label=label, color=colors[offset])
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS.get(mode, mode) for mode in modes], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Residual rate")
    ax.set_title("Experiment 2 residual violation rates")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "exp2_residual_violation_rates.png", dpi=240)
    fig.savefig(figures_dir / "exp2_residual_violation_rates.pdf")
    plt.close(fig)


def _plot_bound_repair_example(results: list[CandidateRepairModeResult], figures_dir: Path) -> None:
    plt = configure_matplotlib()
    full_risk = [
        item for item in results
        if item.mode == "full_repair" and item.risk_conflict_present and item.risk_bound_tightened
    ]
    if len(full_risk) == 0:
        return
    steering_before: list[float] = []
    steering_after: list[float] = []
    hysteresis_before: list[float] = []
    hysteresis_after: list[float] = []
    for item in full_risk[:250]:
        before_steering = _find_bound(item.requested_bounds, "steering_aggressiveness")
        after_steering = _find_bound(item.output_bounds, "steering_aggressiveness")
        before_hysteresis = _find_bound(item.requested_bounds, "steering_hysteresis_db")
        after_hysteresis = _find_bound(item.output_bounds, "steering_hysteresis_db")
        if before_steering is not None and after_steering is not None:
            steering_before.append(float(getattr(before_steering, "upper_bound", 0.0) or 0.0))
            steering_after.append(float(getattr(after_steering, "upper_bound", 0.0) or 0.0))
        if before_hysteresis is not None and after_hysteresis is not None:
            hysteresis_before.append(float(getattr(before_hysteresis, "lower_bound", 0.0) or 0.0))
            hysteresis_after.append(float(getattr(after_hysteresis, "lower_bound", 0.0) or 0.0))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].boxplot([steering_before, steering_after], labels=["Raw upper", "Repaired upper"], patch_artist=True)
    axes[0].set_title("steering_aggressiveness")
    axes[0].set_ylabel("Upper bound")
    axes[1].boxplot([hysteresis_before, hysteresis_after], labels=["Raw lower", "Repaired lower"], patch_artist=True)
    axes[1].set_title("steering_hysteresis_db")
    axes[1].set_ylabel("Lower bound (dB)")
    fig.suptitle("Experiment 2 L3-supported parameter-bound repair")
    fig.tight_layout()
    fig.savefig(figures_dir / "exp2_parameter_bound_repair_example.png", dpi=240)
    fig.savefig(figures_dir / "exp2_parameter_bound_repair_example.pdf")
    plt.close(fig)


def _plot_contract_region_diversity(summary: dict[str, dict[str, object]], figures_dir: Path) -> None:
    plt = configure_matplotlib()
    modes = [mode for mode in EXP2_REPAIR_MODES if mode in summary]
    x_positions = list(range(len(modes)))
    volume = [float(summary[mode].get("admissible_region_volume_ratio", 0.0) or 0.0) for mode in modes]
    diversity = [float(summary[mode].get("plan_diversity", 0.0) or 0.0) for mode in modes]
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    width = 0.36
    ax.bar([x - width / 2 for x in x_positions], volume, width=width, label="Admissible-region volume", color="#5E8C61")
    ax.bar([x + width / 2 for x in x_positions], diversity, width=width, label="Plan diversity", color="#527FA6")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS.get(mode, mode) for mode in modes], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Ratio")
    ax.set_title("Experiment 2 contract region and diversity")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "exp2_contract_region_diversity.png", dpi=240)
    fig.savefig(figures_dir / "exp2_contract_region_diversity.pdf")
    plt.close(fig)


def _svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#24313f}'
        '.title{font-size:18px;font-weight:700}.axis{stroke:#2d3748;stroke-width:1}'
        '.grid{stroke:#d9dee7;stroke-width:1}.label{font-size:11px}</style>'
        f"{body}</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _fallback_svg_stacked_outcomes(results: list[CandidateRepairModeResult], figures_dir: Path) -> None:
    modes = [mode for mode in EXP2_REPAIR_MODES if any(item.mode == mode for item in results)]
    buckets = [
        "executable_without_repair",
        "repaired_and_executable",
        "rejected_due_to_hard_conflict",
        "rejected_due_to_empty_bounds",
        "rejected_due_to_resource_overflow",
        "rejected_due_to_policy_violation",
        "residual_other",
        "rejected_other",
    ]
    colors = ["#5E8C61", "#89B66D", "#C8553D", "#E59F3A", "#B95C7D", "#7A5C99", "#8A8F98", "#C3C7CE"]
    width, height = 1180, 620
    left, top, plot_w, plot_h = 70, 70, 900, 380
    gap = 18
    bar_w = max(24, int((plot_w - gap * max(0, len(modes) - 1)) / max(1, len(modes))))
    body = [f'<text class="title" x="{left}" y="34">Experiment 2 candidate outcomes by repair method</text>']
    body.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for tick in range(0, 6):
        y = top + plot_h - tick * plot_h / 5
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text class="label" x="30" y="{y + 4:.1f}">{tick / 5:.1f}</text>')
    for mode_index, mode in enumerate(modes):
        mode_results = [item for item in results if item.mode == mode]
        denom = max(1, len(mode_results))
        x = left + mode_index * (bar_w + gap)
        y_cursor = top + plot_h
        for bucket, color in zip(buckets, colors):
            value = sum(1 for item in mode_results if item.outcome_bucket == bucket) / denom
            h = value * plot_h
            y_cursor -= h
            body.append(f'<rect x="{x}" y="{y_cursor:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}"/>')
        label = METHOD_LABELS.get(mode, mode)
        body.append(
            f'<text class="label" transform="translate({x + bar_w / 2:.1f},{top + plot_h + 88}) rotate(-35)" '
            f'text-anchor="end">{_svg_escape(label)}</text>'
        )
    legend_x, legend_y = 1000, 80
    for index, (bucket, color) in enumerate(zip(buckets, colors)):
        y = legend_y + index * 24
        body.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{color}"/>')
        body.append(f'<text class="label" x="{legend_x + 22}" y="{y + 12}">{_svg_escape(bucket.replace("_", " "))}</text>')
    _write_svg(figures_dir / "exp2_candidate_outcome_stacked.svg", width, height, "".join(body))


def _fallback_svg_grouped_bars(
    *,
    summary: dict[str, dict[str, object]],
    figures_dir: Path,
    filename: str,
    title: str,
    metrics: list[tuple[str, str, str]],
) -> None:
    modes = [mode for mode in EXP2_REPAIR_MODES if mode in summary]
    width, height = 1180, 560
    left, top, plot_w, plot_h = 70, 70, 900, 330
    group_gap = 22
    group_w = max(50, (plot_w - group_gap * max(0, len(modes) - 1)) / max(1, len(modes)))
    bar_w = max(8, (group_w - 8) / max(1, len(metrics)))
    body = [f'<text class="title" x="{left}" y="34">{_svg_escape(title)}</text>']
    body.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for tick in range(0, 6):
        y = top + plot_h - tick * plot_h / 5
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text class="label" x="30" y="{y + 4:.1f}">{tick / 5:.1f}</text>')
    for mode_index, mode in enumerate(modes):
        x0 = left + mode_index * (group_w + group_gap)
        for metric_index, (metric, _label, color) in enumerate(metrics):
            value = max(0.0, min(1.0, float(summary[mode].get(metric, 0.0) or 0.0)))
            h = value * plot_h
            x = x0 + metric_index * bar_w
            y = top + plot_h - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(
            f'<text class="label" transform="translate({x0 + group_w / 2:.1f},{top + plot_h + 88}) rotate(-35)" '
            f'text-anchor="end">{_svg_escape(METHOD_LABELS.get(mode, mode))}</text>'
        )
    legend_x, legend_y = 995, 82
    for index, (_metric, label, color) in enumerate(metrics):
        y = legend_y + index * 24
        body.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{color}"/>')
        body.append(f'<text class="label" x="{legend_x + 22}" y="{y + 12}">{_svg_escape(label)}</text>')
    _write_svg(figures_dir / filename, width, height, "".join(body))


def _fallback_svg_bound_repair(results: list[CandidateRepairModeResult], figures_dir: Path) -> None:
    full_risk = [
        item for item in results
        if item.mode == "full_repair" and item.risk_conflict_present and item.risk_bound_tightened
    ]
    if len(full_risk) == 0:
        return
    def _mean(values: list[float]) -> float:
        return sum(values) / float(max(1, len(values)))

    steering_before: list[float] = []
    steering_after: list[float] = []
    hysteresis_before: list[float] = []
    hysteresis_after: list[float] = []
    for item in full_risk:
        before_steering = _find_bound(item.requested_bounds, "steering_aggressiveness")
        after_steering = _find_bound(item.output_bounds, "steering_aggressiveness")
        before_hysteresis = _find_bound(item.requested_bounds, "steering_hysteresis_db")
        after_hysteresis = _find_bound(item.output_bounds, "steering_hysteresis_db")
        if before_steering is not None and after_steering is not None:
            steering_before.append(float(getattr(before_steering, "upper_bound", 0.0) or 0.0))
            steering_after.append(float(getattr(after_steering, "upper_bound", 0.0) or 0.0))
        if before_hysteresis is not None and after_hysteresis is not None:
            hysteresis_before.append(float(getattr(before_hysteresis, "lower_bound", 0.0) or 0.0))
            hysteresis_after.append(float(getattr(after_hysteresis, "lower_bound", 0.0) or 0.0))
    values = [
        ("steering upper", _mean(steering_before), _mean(steering_after), "#527FA6"),
        ("hysteresis lower", _mean(hysteresis_before), _mean(hysteresis_after), "#C8553D"),
    ]
    width, height = 820, 360
    body = ['<text class="title" x="50" y="34">Experiment 2 L3-supported bound repair</text>']
    x1, x2 = 230, 560
    for index, (label, before, after, color) in enumerate(values):
        y = 130 + index * 110
        body.append(f'<text x="50" y="{y + 5}">{_svg_escape(label)}</text>')
        body.append(f'<circle cx="{x1}" cy="{y}" r="20" fill="{color}" opacity="0.85"/>')
        body.append(f'<circle cx="{x2}" cy="{y}" r="20" fill="{color}" opacity="0.45"/>')
        body.append(f'<line x1="{x1 + 24}" y1="{y}" x2="{x2 - 24}" y2="{y}" stroke="{color}" stroke-width="3" marker-end="url(#arrow)"/>')
        body.append(f'<text x="{x1}" y="{y + 48}" text-anchor="middle">raw {before:.2f}</text>')
        body.append(f'<text x="{x2}" y="{y + 48}" text-anchor="middle">repaired {after:.2f}</text>')
    defs = '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#24313f"/></marker></defs>'
    _write_svg(figures_dir / "exp2_parameter_bound_repair_example.svg", width, height, defs + "".join(body))


def _write_svg_fallback_figures(
    results: list[CandidateRepairModeResult],
    summary: dict[str, dict[str, object]],
    figures_dir: Path,
) -> None:
    _fallback_svg_stacked_outcomes(results, figures_dir)
    _fallback_svg_grouped_bars(
        summary=summary,
        figures_dir=figures_dir,
        filename="exp2_residual_violation_rates.svg",
        title="Experiment 2 residual violation rates",
        metrics=[
            ("hard_conflict_residual_rate", "Hard conflict", "#C8553D"),
            ("resource_overflow_residual_rate", "Resource overflow", "#E59F3A"),
            ("policy_violation_residual_rate", "Policy violation", "#7A5C99"),
            ("contract_inconsistency_rate", "Contract inconsistency", "#556070"),
        ],
    )
    _fallback_svg_bound_repair(results, figures_dir)
    _fallback_svg_grouped_bars(
        summary=summary,
        figures_dir=figures_dir,
        filename="exp2_contract_region_diversity.svg",
        title="Experiment 2 contract region and diversity",
        metrics=[
            ("admissible_region_volume_ratio", "Admissible region", "#5E8C61"),
            ("plan_diversity", "Plan diversity", "#527FA6"),
        ],
    )


def _write_figures(results: list[CandidateRepairModeResult], summary: dict[str, dict[str, object]], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plotting_error = figures_dir / "plotting_error.txt"
    if plotting_error.exists():
        plotting_error.unlink()
    try:
        _plot_candidate_outcomes(results, figures_dir)
        _plot_residual_violations(summary, figures_dir)
        _plot_bound_repair_example(results, figures_dir)
        _plot_contract_region_diversity(summary, figures_dir)
    except Exception as exc:  # pragma: no cover - plotting should not hide CSV results.
        _write_svg_fallback_figures(results, summary, figures_dir)
        with (figures_dir / "plotting_error.txt").open("w", encoding="utf-8") as error_file:
            error_file.write(f"matplotlib path unavailable; wrote SVG fallback figures. Details: {exc}")


def write_candidate_repair_outputs(
    *,
    output_dir: str | Path,
    results: list[CandidateRepairModeResult],
    summary: dict[str, dict[str, object]],
    config: dict[str, Any] | None = None,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    records = [result.to_record() for result in list(results or [])]
    _write_csv(path / "case_results.csv", records)

    with (path / "summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)

    summary_records = _summary_records(summary)
    _write_csv(path / "summary.csv", summary_records)
    _write_csv(path / "main_table.csv", _make_main_table(summary))
    _write_csv(path / "contract_table.csv", _make_contract_table(summary))

    if config is not None:
        with (path / "config_snapshot.json").open("w", encoding="utf-8") as json_file:
            json.dump(config, json_file, indent=2, sort_keys=True)

    if len(results) > 0:
        _write_figures(results, summary, path)
