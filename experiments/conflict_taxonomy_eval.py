"""Candidate-level conflict taxonomy ablation benchmark.

The benchmark compares simple conflict handling baselines against a typed
hard/soft/risk/policy taxonomy. It is a sandbox experiment: plans are evaluated
for detection and decision quality, not executed on the RAN simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
import time
from typing import Any

from experiments.candidate_repair_eval import (
    _canonical_categories,
    _schema_valid,
    build_default_available_xapps,
)
from ric.near_rt.xapp_variant_catalog import canonical_xapp_category
from ric.non_rt.orchestration_feasibility import OrchestrationFeasibilityProjector


CONFLICT_EVAL_VERSION = "conflict_taxonomy_eval_v1"
DEFAULT_CONFLICT_MODES = [
    "no_conflict_check",
    "binary_conflict_flag",
    "pacifista_like_profiler",
    "typed_taxonomy",
]


@dataclass(frozen=True)
class ConflictFixture:
    case_id: str
    conflict_family: str
    requested_xapps: list[str] = field(default_factory=list)
    scenario_tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class ConflictFinding:
    conflict_type: str
    conflict_id: str
    severity: float
    reason: str


@dataclass
class ConflictEvalResult:
    mode: str
    case_id: str
    conflict_family: str
    requested_xapps: list[str]
    scenario_tags: list[str]
    true_conflict_types: list[str]
    detected_conflict_types: list[str]
    action: str
    accepted: bool
    rejected: bool
    arbitration_applied: bool
    risk_mitigation_applied: bool
    hard_conflict_escape: bool
    policy_conflict_escape: bool
    risk_conflict_realized: bool
    protected_kpi_violation: bool
    unsafe_plan: bool
    over_rejected: bool
    soft_arbitration_success: bool
    intent_fulfillment_proxy: float
    total_cpu_cost: float
    total_mem_cost: float
    active_xapp_count: int
    decision_latency_ms: float
    decision_notes: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "case_id": self.case_id,
            "conflict_family": self.conflict_family,
            "requested_xapps": "|".join(self.requested_xapps),
            "scenario_tags": "|".join(self.scenario_tags),
            "true_conflict_types": "|".join(self.true_conflict_types),
            "detected_conflict_types": "|".join(self.detected_conflict_types),
            "action": self.action,
            "accepted": int(self.accepted),
            "rejected": int(self.rejected),
            "arbitration_applied": int(self.arbitration_applied),
            "risk_mitigation_applied": int(self.risk_mitigation_applied),
            "hard_conflict_escape": int(self.hard_conflict_escape),
            "policy_conflict_escape": int(self.policy_conflict_escape),
            "risk_conflict_realized": int(self.risk_conflict_realized),
            "protected_kpi_violation": int(self.protected_kpi_violation),
            "unsafe_plan": int(self.unsafe_plan),
            "over_rejected": int(self.over_rejected),
            "soft_arbitration_success": int(self.soft_arbitration_success),
            "intent_fulfillment_proxy": round(float(self.intent_fulfillment_proxy), 6),
            "total_cpu_cost": round(float(self.total_cpu_cost), 6),
            "total_mem_cost": round(float(self.total_mem_cost), 6),
            "active_xapp_count": int(self.active_xapp_count),
            "decision_latency_ms": round(float(self.decision_latency_ms), 6),
            "decision_notes": "|".join(self.decision_notes),
        }


def build_default_conflict_fixtures() -> list[ConflictFixture]:
    return [
        ConflictFixture(
            case_id="valid_throughput_reliability",
            conflict_family="none",
            requested_xapps=[
                "massive_mimo_beamformer_xapp_balanced",
                "spectrum_optimizer_xapp_balanced",
                "urllc_guard_xapp_balanced",
            ],
            scenario_tags=["medium_load", "mixed_traffic"],
            notes="A feasible throughput-reliability portfolio.",
        ),
        ConflictFixture(
            case_id="valid_energy_period",
            conflict_family="none",
            requested_xapps=[
                "power_saving_xapp_balanced",
                "cell_sleep_xapp_macro_guarded",
            ],
            scenario_tags=["low_load", "no_latency_slice"],
            notes="Energy-saving plan outside latency-sensitive periods.",
        ),
        ConflictFixture(
            case_id="hard_sleep_vs_urllc",
            conflict_family="hard",
            requested_xapps=[
                "cell_sleep_xapp_balanced",
                "power_saving_xapp_balanced",
                "urllc_guard_xapp_strict",
            ],
            scenario_tags=["urllc_burst", "latency_sensitive"],
            notes="Cell sleep and URLLC guard issue incompatible actuator intents.",
        ),
        ConflictFixture(
            case_id="hard_sleep_vs_latency_scheduler",
            conflict_family="hard",
            requested_xapps=[
                "cell_sleep_xapp_aggressive",
                "power_saving_xapp_balanced",
                "latency_scheduler_xapp_urllc_fastpath",
            ],
            scenario_tags=["latency_sensitive"],
            notes="Cell sleep conflicts with latency-protective scheduling.",
        ),
        ConflictFixture(
            case_id="soft_energy_vs_latency",
            conflict_family="soft",
            requested_xapps=[
                "power_saving_xapp_urlcc_safe",
                "latency_scheduler_xapp_tail_guard",
                "ran_slicing_manager_xapp_sla_strict",
            ],
            scenario_tags=["mixed_traffic", "latency_sensitive"],
            notes="Energy saving and latency protection can coexist after arbitration.",
        ),
        ConflictFixture(
            case_id="soft_capacity_vs_reliability",
            conflict_family="soft",
            requested_xapps=[
                "massive_mimo_beamformer_xapp_capacity_push",
                "spectrum_optimizer_xapp_aggressive",
                "urllc_guard_xapp_tail_safe",
            ],
            scenario_tags=["cell_edge_users", "urllc_burst"],
            notes="Capacity push can pressure reliability unless guardrails arbitrate.",
        ),
        ConflictFixture(
            case_id="risk_aggressive_steering_high_mobility",
            conflict_family="risk",
            requested_xapps=[
                "traffic_steering_xapp_load_first",
                "mobility_predictor_xapp_handover_push",
                "massive_mimo_beamformer_xapp_capacity_push",
            ],
            scenario_tags=["dense_hotspot", "high_mobility"],
            notes="Aggressive steering may realize as ping-pong handover risk.",
        ),
        ConflictFixture(
            case_id="risk_interference_packing_burst",
            conflict_family="risk",
            requested_xapps=[
                "spectrum_optimizer_xapp_aggressive",
                "massive_mimo_beamformer_xapp_capacity_push",
                "traffic_steering_xapp_balanced",
                "mobility_predictor_xapp_balanced",
            ],
            scenario_tags=["interference_burst", "cell_edge_users"],
            notes="Aggressive spectrum packing may degrade edge SINR under interference.",
        ),
        ConflictFixture(
            case_id="policy_sleep_premium_urllc",
            conflict_family="policy",
            requested_xapps=[
                "cell_sleep_xapp_deep_saver",
                "power_saving_xapp_energy_max",
                "ran_slicing_manager_xapp_sla_strict",
            ],
            scenario_tags=["premium_tenant", "urllc_burst", "tenant_priority"],
            notes="Tenant priority forbids deep sleep during premium URLLC demand.",
        ),
        ConflictFixture(
            case_id="policy_sleep_coverage_constraint",
            conflict_family="policy",
            requested_xapps=[
                "cell_sleep_xapp_night_mode",
                "power_saving_xapp_energy_max",
            ],
            scenario_tags=["coverage_constraint", "cell_edge_users"],
            notes="Coverage policy forbids sleep for a cell-edge coverage anchor.",
        ),
    ]


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        text = str(item or "").strip()
        if text == "" or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _has_category(categories: set[str], category_name: str) -> bool:
    return canonical_xapp_category(category_name) in categories


def classify_typed_conflicts(
    fixture: ConflictFixture,
    *,
    projector: OrchestrationFeasibilityProjector,
) -> list[ConflictFinding]:
    xapps = list(fixture.requested_xapps or [])
    categories = set(_canonical_categories(xapps))
    tags = {str(tag).strip().lower() for tag in list(fixture.scenario_tags or [])}
    xapp_text = " ".join(str(item).lower() for item in xapps)
    findings: list[ConflictFinding] = []

    for left_category, right_category in projector.mutex_category_pairs:
        if left_category in categories and right_category in categories:
            findings.append(
                ConflictFinding(
                    conflict_type="hard",
                    conflict_id=f"hard:{left_category}!={right_category}",
                    severity=0.98,
                    reason="Incompatible actuator control surfaces.",
                )
            )

    if _has_category(categories, "cell_sleep_xapp") and (
        "premium_tenant" in tags
        or "tenant_priority" in tags
        or "coverage_constraint" in tags
    ):
        findings.append(
            ConflictFinding(
                conflict_type="policy",
                conflict_id="policy:cell_sleep_tenant_or_coverage_guard",
                severity=0.92,
                reason="Cell sleep violates tenant priority or coverage constraints.",
            )
        )

    if _has_category(categories, "power_saving_xapp") and _has_category(categories, "latency_scheduler_xapp"):
        findings.append(
            ConflictFinding(
                conflict_type="soft",
                conflict_id="soft:energy_latency_tradeoff",
                severity=0.55,
                reason="Energy-saving pressure can be arbitrated with latency protection.",
            )
        )

    if (
        _has_category(categories, "massive_mimo_beamformer_xapp")
        and _has_category(categories, "spectrum_optimizer_xapp")
        and _has_category(categories, "urllc_guard_xapp")
        and ("capacity_push" in xapp_text or "aggressive" in xapp_text)
    ):
        findings.append(
            ConflictFinding(
                conflict_type="soft",
                conflict_id="soft:capacity_reliability_tradeoff",
                severity=0.58,
                reason="Capacity-oriented radio actions need reliability arbitration.",
            )
        )

    if (
        _has_category(categories, "traffic_steering_xapp")
        and _has_category(categories, "mobility_predictor_xapp")
        and "high_mobility" in tags
        and ("load_first" in xapp_text or "handover_push" in xapp_text)
    ):
        findings.append(
            ConflictFinding(
                conflict_type="risk",
                conflict_id="risk:steering_mobility_pingpong",
                severity=0.74,
                reason="Aggressive steering can realize as ping-pong under high mobility.",
            )
        )

    if (
        _has_category(categories, "spectrum_optimizer_xapp")
        and _has_category(categories, "massive_mimo_beamformer_xapp")
        and "interference_burst" in tags
        and ("aggressive" in xapp_text or "capacity_push" in xapp_text)
    ):
        findings.append(
            ConflictFinding(
                conflict_type="risk",
                conflict_id="risk:interference_capacity_edge_sinr",
                severity=0.67,
                reason="Aggressive packing may turn into edge-SINR degradation.",
            )
        )

    return findings


def classify_pacifista_like_conflicts(
    fixture: ConflictFixture,
    *,
    projector: OrchestrationFeasibilityProjector,
) -> list[ConflictFinding]:
    typed_findings = classify_typed_conflicts(fixture, projector=projector)
    profiled_findings: list[ConflictFinding] = []
    for finding in typed_findings:
        if finding.conflict_type == "policy":
            continue
        if finding.conflict_type == "hard":
            profiled_findings.append(finding)
        elif finding.conflict_type == "soft":
            profiled_findings.append(
                ConflictFinding(
                    conflict_type="profiled",
                    conflict_id=f"pacifista:{finding.conflict_id}",
                    severity=float(finding.severity),
                    reason="Historical xApp-pair profile flags medium-severity degradation.",
                )
            )
        elif finding.conflict_type == "risk":
            profiled_findings.append(
                ConflictFinding(
                    conflict_type="profiled",
                    conflict_id=f"pacifista:{finding.conflict_id}",
                    severity=float(finding.severity) + 0.03,
                    reason="Historical pair profile predicts scenario-sensitive risk.",
                )
            )
    return profiled_findings


def _finding_types(findings: list[ConflictFinding]) -> list[str]:
    return _unique([finding.conflict_type for finding in list(findings or [])])


def _resource_cost(
    *,
    xapps: list[str],
    projector: OrchestrationFeasibilityProjector,
) -> tuple[float, float]:
    return projector._summed_costs(xapps)  # pylint: disable=protected-access


def _decision_for_mode(
    *,
    mode: str,
    true_findings: list[ConflictFinding],
    detected_findings: list[ConflictFinding],
    schema_valid: bool,
) -> tuple[str, bool, bool, bool, list[str]]:
    if not schema_valid:
        return "reject_schema_invalid", False, False, False, ["schema_invalid"]

    true_types = set(_finding_types(true_findings))
    if mode == "no_conflict_check":
        return "accept_without_conflict_check", True, False, False, ["basic_schema_only"]

    if mode == "binary_conflict_flag":
        if len(true_findings) > 0:
            return "reject_any_conflict", False, False, False, ["binary_conflict_flag"]
        return "accept_no_conflict", True, False, False, ["binary_no_conflict"]

    if mode == "pacifista_like_profiler":
        if len(detected_findings) == 0:
            return "accept_profile_no_conflict", True, False, False, ["profile_no_conflict"]
        max_severity = max(float(finding.severity) for finding in detected_findings)
        if max_severity >= 0.76:
            return "reject_profile_high_severity", False, False, False, [f"profile_severity={max_severity:.2f}"]
        if max_severity >= 0.45:
            return "accept_profile_monitor", True, False, False, [f"profile_monitor_severity={max_severity:.2f}"]
        return "accept_profile_low_severity", True, False, False, [f"profile_low_severity={max_severity:.2f}"]

    if mode == "typed_taxonomy":
        if "hard" in true_types:
            return "reject_hard_conflict", False, False, False, ["typed_hard_reject"]
        if "policy" in true_types:
            return "reject_policy_conflict", False, False, False, ["typed_policy_reject"]
        if "soft" in true_types:
            return "accept_soft_arbitrated", True, True, False, ["typed_soft_arbitration"]
        if "risk" in true_types:
            return "accept_risk_monitored", True, False, True, ["typed_risk_monitor_and_fallback"]
        return "accept_no_conflict", True, False, False, ["typed_no_conflict"]

    raise ValueError(f"Unsupported conflict evaluation mode: {mode}")


def evaluate_conflict_taxonomy_modes(
    *,
    modes: list[str] | None = None,
    fixtures: list[ConflictFixture] | None = None,
    projector: OrchestrationFeasibilityProjector | None = None,
    available_xapp_names: list[str] | None = None,
) -> tuple[list[ConflictEvalResult], dict[str, dict[str, object]]]:
    selected_modes = [str(item).strip() for item in list(modes or DEFAULT_CONFLICT_MODES) if str(item).strip() != ""]
    for mode in selected_modes:
        if mode not in set(DEFAULT_CONFLICT_MODES):
            raise ValueError(f"Unsupported conflict evaluation mode: {mode}")

    active_fixtures = list(fixtures or build_default_conflict_fixtures())
    active_projector = projector or OrchestrationFeasibilityProjector(
        max_selected_xapps=4,
        max_cpu_cost=4.6,
        max_mem_cost=4.2,
    )
    available_names = set(available_xapp_names or build_default_available_xapps())
    results: list[ConflictEvalResult] = []

    for mode in selected_modes:
        for fixture in active_fixtures:
            requested_xapps = [str(item).strip() for item in list(fixture.requested_xapps or []) if str(item).strip() != ""]
            schema_valid = _schema_valid(requested_xapps) and all(
                xapp_name in available_names
                for xapp_name in requested_xapps
            )
            started_at = time.perf_counter()
            true_findings = classify_typed_conflicts(fixture, projector=active_projector)
            if mode == "pacifista_like_profiler":
                detected_findings = classify_pacifista_like_conflicts(fixture, projector=active_projector)
            elif mode in {"binary_conflict_flag", "typed_taxonomy"}:
                detected_findings = list(true_findings)
            else:
                detected_findings = []
            action, accepted, arbitration_applied, risk_mitigation_applied, decision_notes = _decision_for_mode(
                mode=mode,
                true_findings=true_findings,
                detected_findings=detected_findings,
                schema_valid=schema_valid,
            )
            decision_latency_ms = (time.perf_counter() - started_at) * 1000.0

            true_types = set(_finding_types(true_findings))
            detected_types = _finding_types(detected_findings)
            rejected = not bool(accepted)
            hard_escape = bool(accepted and "hard" in true_types)
            policy_escape = bool(accepted and "policy" in true_types)
            risk_realized = bool(accepted and "risk" in true_types and not risk_mitigation_applied)
            soft_unarbitrated = bool(accepted and "soft" in true_types and not arbitration_applied)
            protected_kpi_violation = bool(hard_escape or policy_escape or risk_realized or soft_unarbitrated)
            unsafe_plan = bool(hard_escape or policy_escape or protected_kpi_violation)
            over_rejected = bool(rejected and len(true_types & {"hard", "policy"}) == 0)
            soft_arbitration_success = bool(
                accepted
                and "soft" in true_types
                and arbitration_applied
                and not protected_kpi_violation
            )
            intent_fulfillment_proxy = 1.0 if bool(accepted and not protected_kpi_violation) else 0.0
            cpu_cost, mem_cost = _resource_cost(xapps=requested_xapps, projector=active_projector)

            results.append(
                ConflictEvalResult(
                    mode=mode,
                    case_id=fixture.case_id,
                    conflict_family=fixture.conflict_family,
                    requested_xapps=requested_xapps,
                    scenario_tags=list(fixture.scenario_tags),
                    true_conflict_types=_finding_types(true_findings),
                    detected_conflict_types=detected_types,
                    action=action,
                    accepted=accepted,
                    rejected=rejected,
                    arbitration_applied=arbitration_applied,
                    risk_mitigation_applied=risk_mitigation_applied,
                    hard_conflict_escape=hard_escape,
                    policy_conflict_escape=policy_escape,
                    risk_conflict_realized=risk_realized,
                    protected_kpi_violation=protected_kpi_violation,
                    unsafe_plan=unsafe_plan,
                    over_rejected=over_rejected,
                    soft_arbitration_success=soft_arbitration_success,
                    intent_fulfillment_proxy=intent_fulfillment_proxy,
                    total_cpu_cost=cpu_cost,
                    total_mem_cost=mem_cost,
                    active_xapp_count=len(requested_xapps) if accepted else 0,
                    decision_latency_ms=decision_latency_ms,
                    decision_notes=decision_notes,
                )
            )
    return results, summarize_conflict_taxonomy_results(results)


def summarize_conflict_taxonomy_results(
    results: list[ConflictEvalResult],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[ConflictEvalResult]] = {}
    for result in list(results or []):
        grouped.setdefault(result.mode, []).append(result)

    summary: dict[str, dict[str, object]] = {}
    for mode, mode_results in sorted(grouped.items()):
        denominator = max(1, len(mode_results))
        hard_cases = [item for item in mode_results if "hard" in set(item.true_conflict_types)]
        soft_cases = [item for item in mode_results if "soft" in set(item.true_conflict_types)]
        risk_cases = [item for item in mode_results if "risk" in set(item.true_conflict_types)]
        policy_cases = [item for item in mode_results if "policy" in set(item.true_conflict_types)]
        accepted_results = [item for item in mode_results if item.accepted]
        summary[mode] = {
            "conflict_eval_version": CONFLICT_EVAL_VERSION,
            "case_count": len(mode_results),
            "acceptance_rate": sum(1 for item in mode_results if item.accepted) / denominator,
            "rejection_rate": sum(1 for item in mode_results if item.rejected) / denominator,
            "hard_conflict_attempt_rate": len(hard_cases) / denominator,
            "hard_conflict_escape_rate": (
                sum(1 for item in hard_cases if item.hard_conflict_escape) / max(1, len(hard_cases))
            ),
            "policy_conflict_escape_rate": (
                sum(1 for item in policy_cases if item.policy_conflict_escape) / max(1, len(policy_cases))
            ),
            "soft_conflict_arbitration_success_rate": (
                sum(1 for item in soft_cases if item.soft_arbitration_success) / max(1, len(soft_cases))
            ),
            "risk_conflict_realization_rate": (
                sum(1 for item in risk_cases if item.risk_conflict_realized) / max(1, len(risk_cases))
            ),
            "unsafe_plan_rate": sum(1 for item in mode_results if item.unsafe_plan) / denominator,
            "over_rejection_rate": sum(1 for item in mode_results if item.over_rejected) / denominator,
            "protected_kpi_violation_rate": sum(1 for item in mode_results if item.protected_kpi_violation) / denominator,
            "intent_fulfillment_proxy": sum(float(item.intent_fulfillment_proxy) for item in mode_results) / denominator,
            "mean_active_xapp_count": (
                sum(int(item.active_xapp_count) for item in accepted_results) / max(1, len(accepted_results))
            ),
            "mean_resource_cpu_cost": (
                sum(float(item.total_cpu_cost) for item in accepted_results) / max(1, len(accepted_results))
            ),
            "mean_resource_mem_cost": (
                sum(float(item.total_mem_cost) for item in accepted_results) / max(1, len(accepted_results))
            ),
            "mean_decision_latency_ms": (
                sum(float(item.decision_latency_ms) for item in mode_results) / denominator
            ),
        }
    return summary


def write_conflict_taxonomy_outputs(
    *,
    output_dir: str | Path,
    results: list[ConflictEvalResult],
    summary: dict[str, dict[str, object]],
    config: dict[str, Any] | None = None,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    records = [result.to_record() for result in list(results or [])]
    fieldnames = list(records[0].keys()) if len(records) > 0 else []
    with (path / "case_results.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    with (path / "summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)

    summary_records = [
        {"mode": str(mode), **dict(values)}
        for mode, values in sorted(dict(summary or {}).items())
    ]
    if len(summary_records) > 0:
        summary_fieldnames: list[str] = []
        for record in summary_records:
            for key in record.keys():
                if key not in summary_fieldnames:
                    summary_fieldnames.append(key)
        with (path / "summary.csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=summary_fieldnames)
            writer.writeheader()
            for record in summary_records:
                writer.writerow(record)

    if config is not None:
        with (path / "config_snapshot.json").open("w", encoding="utf-8") as json_file:
            json.dump(config, json_file, indent=2, sort_keys=True)
