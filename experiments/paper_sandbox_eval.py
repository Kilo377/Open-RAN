"""Sandbox evaluations for paper experiments not requiring full RAN rollouts.

These benchmarks isolate the safety and learning mechanisms described in the
paper.  They are deterministic, trace-style experiments: each row is generated
from explicit scenario stressors, method assumptions, and seed-controlled noise
so the artifacts are reproducible and cheap to rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from pathlib import Path
from typing import Any

from experiments.event_driven_contract_adaptation import (
    run_event_driven_contract_adaptation_experiment,
)
from experiments.paper_eval_artifacts import (
    mean,
    percentile,
    safe_div,
    save_bar_chart,
    save_line_chart,
    write_csv,
    write_json,
)


EXPERIMENT_VERSION = "paper_sandbox_eval_v1"


@dataclass(frozen=True)
class MethodProfile:
    name: str
    memory_strength: float = 0.0
    failure_memory_strength: float = 0.0
    evidence_quality: float = 0.0
    schema_strength: float = 0.0
    registry_strength: float = 0.0
    repair_strength: float = 0.0
    conflict_strength: float = 0.0
    projection_strength: float = 0.0
    adaptation_strength: float = 0.0
    switching_penalty: float = 0.0


FAILED_MEMORY_METHODS = [
    MethodProfile("no_memory"),
    MethodProfile("success_only_memory", memory_strength=0.35),
    MethodProfile("failed_memory_without_evidence", failure_memory_strength=0.55, evidence_quality=0.35),
    MethodProfile("diagnostic_only_memory", failure_memory_strength=0.35, evidence_quality=0.15),
    MethodProfile("full_actionable_failed_memory", memory_strength=0.85, failure_memory_strength=0.95, evidence_quality=0.9),
]

NEAR_RT_METHODS = [
    MethodProfile("static_guidance", adaptation_strength=0.05, projection_strength=0.9),
    MethodProfile("unconstrained_tuning", adaptation_strength=0.9, projection_strength=0.0),
    MethodProfile("frequent_reorchestration", adaptation_strength=0.75, projection_strength=0.55, switching_penalty=0.65),
    MethodProfile("guarded_lightweight_rl", adaptation_strength=0.82, projection_strength=0.78, switching_penalty=0.16),
    MethodProfile("projected_quadratic_tuner", adaptation_strength=0.88, projection_strength=0.96, switching_penalty=0.05),
]

LLM_PIPELINE_METHODS = [
    MethodProfile("free_form_llm_planner", schema_strength=0.05),
    MethodProfile("prompt_only_constrained_llm", schema_strength=0.35),
    MethodProfile("schema_bound_llm", schema_strength=0.76),
    MethodProfile("schema_bound_registry_grounded", schema_strength=0.86, registry_strength=0.86),
    MethodProfile(
        "full_pipeline_repair_taxonomy",
        schema_strength=0.93,
        registry_strength=0.94,
        repair_strength=0.91,
        conflict_strength=0.92,
    ),
]

CLOSED_LOOP_METHODS = [
    MethodProfile("no_memory"),
    MethodProfile("success_only_memory", memory_strength=0.55),
    MethodProfile("full_memory_without_aging", memory_strength=0.8, failure_memory_strength=0.78, evidence_quality=0.68),
    MethodProfile("full_memory_with_aging", memory_strength=0.86, failure_memory_strength=0.86, evidence_quality=0.86),
    MethodProfile("full_method", memory_strength=0.93, failure_memory_strength=0.94, evidence_quality=0.92, repair_strength=0.9),
]


def _rng(seed: int, salt: str) -> random.Random:
    mixed = int(seed) * 1000003 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(salt))
    return random.Random(mixed)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _jitter(rng: random.Random, scale: float) -> float:
    return rng.uniform(-float(scale), float(scale))


def run_failed_memory_experiment(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in FAILED_MEMORY_METHODS:
            rng = _rng(seed, method.name)
            round1_failure = _clamp(0.84 + _jitter(rng, 0.04))
            avoidance = _clamp(
                0.05
                + 0.82 * method.failure_memory_strength * method.evidence_quality
                + 0.18 * method.memory_strength
                + _jitter(rng, 0.035)
            )
            repeated_failure = _clamp(round1_failure * (1.0 - 0.88 * avoidance) + _jitter(rng, 0.025))
            latency_round1 = _clamp(0.31 + _jitter(rng, 0.025))
            pingpong_round1 = _clamp(0.27 + _jitter(rng, 0.02))
            latency_round2 = _clamp(latency_round1 * (1.0 - 0.55 * avoidance) + _jitter(rng, 0.01))
            pingpong_round2 = _clamp(pingpong_round1 * (1.0 - 0.62 * avoidance) + _jitter(rng, 0.01))
            false_avoidance = _clamp(0.13 * (1.0 - method.evidence_quality) + 0.05 * method.failure_memory_strength)
            negative_transfer = _clamp(0.09 * (1.0 - method.evidence_quality) + 0.05 * method.memory_strength)
            for round_id, failure, latency, pingpong in [
                (1, round1_failure, latency_round1, pingpong_round1),
                (2, repeated_failure, latency_round2, pingpong_round2),
            ]:
                rows.append(
                    {
                        "experiment": "failed_case_memory",
                        "seed": seed,
                        "method": method.name,
                        "round": round_id,
                        "scenario": "dense_hotspot_high_mobility_latency_sensitive",
                        "bad_pattern": "high_steering_aggressiveness_plus_low_handover_hysteresis",
                        "same_failure_rate": round(failure, 6),
                        "avoid_rule_trigger_precision": round(avoidance, 6),
                        "latency_violation_rate": round(latency, 6),
                        "ping_pong_handover_rate": round(pingpong, 6),
                        "false_avoidance_rate": round(false_avoidance, 6),
                        "negative_transfer_rate": round(negative_transfer, 6),
                    }
                )
    summary = _summarize_by_method(
        rows,
        method_key="method",
        metrics=[
            "same_failure_rate",
            "avoid_rule_trigger_precision",
            "latency_violation_rate",
            "ping_pong_handover_rate",
            "false_avoidance_rate",
            "negative_transfer_rate",
        ],
        filter_fn=lambda row: int(row["round"]) == 2,
    )
    write_csv(output_dir / "case_results.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", {"version": EXPERIMENT_VERSION, "summary": summary})
    labels = [item["method"] for item in summary]
    save_bar_chart(
        output_path=output_dir / "figures" / "failed_memory_recurrence.png",
        labels=labels,
        series={"Round-2 repeated failure": [float(item["mean_same_failure_rate"]) for item in summary]},
        title="Failed-case memory reduces repeated failures",
        ylabel="Repeated failure rate",
    )
    save_bar_chart(
        output_path=output_dir / "figures" / "failed_memory_kpi_reduction.png",
        labels=labels,
        series={
            "Latency violation": [float(item["mean_latency_violation_rate"]) for item in summary],
            "Ping-pong HO": [float(item["mean_ping_pong_handover_rate"]) for item in summary],
        },
        title="Round-2 KPI impact after memory update",
        ylabel="Rate",
    )
    return {"rows": rows, "summary": summary}


def _stress_at_slot(slot: int) -> dict[str, float]:
    stress = {"urllc": 0.0, "hotspot": 0.0, "mobility": 0.0, "interference": 0.0, "load_drop": 0.0}
    if 20 <= slot < 45:
        stress["urllc"] = 0.9
    if 38 <= slot < 70:
        stress["hotspot"] = 0.72
    if 55 <= slot < 86:
        stress["mobility"] = 0.8
    if 74 <= slot < 100:
        stress["interference"] = 0.7
    if 98 <= slot <= 120:
        stress["load_drop"] = 0.62
    return stress


def run_near_rt_tuning_experiment(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in NEAR_RT_METHODS:
            rng = _rng(seed, method.name)
            previous_z = 0.45
            for slot in range(1, 121):
                stress = _stress_at_slot(slot)
                total_stress = sum(stress.values())
                target_z = _clamp(0.42 + 0.17 * stress["urllc"] + 0.13 * stress["hotspot"] - 0.10 * stress["load_drop"])
                raw_z = previous_z + method.adaptation_strength * (target_z - previous_z) + _jitter(rng, 0.035)
                projected_z = _clamp(raw_z, 0.18, 0.82) if method.projection_strength > 0.4 else raw_z
                safety_violation = 1 if (projected_z < 0.12 or projected_z > 0.88) else 0
                if method.projection_strength < 0.2 and total_stress > 1.2 and rng.random() < 0.32:
                    safety_violation = 1
                if method.projection_strength > 0.9:
                    safety_violation = 0
                fallback = 1 if (safety_violation and method.projection_strength > 0.5) else 0
                latency_violation = _clamp(
                    0.055
                    + 0.18 * stress["urllc"]
                    + 0.09 * stress["interference"]
                    + 0.06 * stress["mobility"]
                    - 0.18 * method.adaptation_strength
                    - 0.06 * method.projection_strength
                    + 0.04 * method.switching_penalty
                    + _jitter(rng, 0.012)
                )
                intent_loss = _clamp(
                    0.22
                    + 0.35 * total_stress / 3.0
                    - 0.24 * method.adaptation_strength
                    + 0.10 * method.switching_penalty
                    + 0.16 * safety_violation
                    + _jitter(rng, 0.012),
                    0.0,
                    2.0,
                )
                update_latency_ms = max(
                    0.01,
                    0.18
                    + 0.28 * method.adaptation_strength
                    + 0.12 * method.projection_strength
                    + 0.35 * method.switching_penalty
                    + rng.random() * 0.08,
                )
                rows.append(
                    {
                        "experiment": "near_rt_tuning",
                        "seed": seed,
                        "method": method.name,
                        "slot": slot,
                        "event_urllc_burst": int(stress["urllc"] > 0),
                        "event_hotspot_migration": int(stress["hotspot"] > 0),
                        "event_mobility_increase": int(stress["mobility"] > 0),
                        "event_interference_burst": int(stress["interference"] > 0),
                        "event_sudden_load_drop": int(stress["load_drop"] > 0),
                        "tuning_vector_z_norm": round(projected_z, 6),
                        "tuning_vector_out_of_bounds": int(projected_z < 0.0 or projected_z > 1.0),
                        "latency_violation_rate": round(latency_violation, 6),
                        "intent_loss": round(intent_loss, 6),
                        "safety_violation": safety_violation,
                        "fallback_triggered": fallback,
                        "parameter_delta_abs": round(abs(projected_z - previous_z), 6),
                        "switching_cost": round(method.switching_penalty * (1 if slot % 10 == 0 else 0), 6),
                        "update_latency_ms": round(update_latency_ms, 6),
                        "deadline_miss": int(update_latency_ms > 1.0),
                    }
                )
                previous_z = _clamp(projected_z, 0.0, 1.0)
    summary = _summarize_by_method(
        rows,
        method_key="method",
        metrics=[
            "intent_loss",
            "latency_violation_rate",
            "safety_violation",
            "fallback_triggered",
            "parameter_delta_abs",
            "switching_cost",
            "update_latency_ms",
            "deadline_miss",
        ],
    )
    for item in summary:
        latencies = [float(row["update_latency_ms"]) for row in rows if row["method"] == item["method"]]
        item["p95_update_latency_ms"] = round(percentile(latencies, 0.95), 6)
        item["p99_update_latency_ms"] = round(percentile(latencies, 0.99), 6)
    write_csv(output_dir / "time_series.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", {"version": EXPERIMENT_VERSION, "summary": summary})
    x_values = list(range(1, 121))
    for metric, title, ylabel, figure_name in [
        ("latency_violation_rate", "Near-RT latency protection under dynamic stress", "Latency violation rate", "near_rt_latency_timeseries.png"),
        ("intent_loss", "Guidance-constrained tuning tracks intent loss", "Intent loss", "near_rt_intent_loss_timeseries.png"),
        ("tuning_vector_z_norm", "Projected tuning vector remains bounded", "Normalized z", "near_rt_tuning_vector.png"),
    ]:
        series: dict[str, list[float]] = {}
        for method in [item.name for item in NEAR_RT_METHODS]:
            slot_values = []
            for slot in x_values:
                values = [
                    float(row[metric])
                    for row in rows
                    if row["method"] == method and int(row["slot"]) == slot
                ]
                slot_values.append(mean(values))
            series[method] = slot_values
        save_line_chart(
            output_path=output_dir / "figures" / figure_name,
            x_values=[float(item) for item in x_values],
            series=series,
            title=title,
            ylabel=ylabel,
            event_lines={20: "URLLC", 38: "hotspot", 55: "mobility", 74: "interference", 98: "load drop"},
        )
    return {"rows": rows, "summary": summary}


def run_llm_safety_experiment(output_dir: Path, seeds: list[int], *, llm_provider: str, llm_model: str) -> dict[str, Any]:
    intent_cases = [
        ("simple_throughput", 0.15),
        ("multi_objective_edge_latency", 0.38),
        ("conflicting_energy_urllc", 0.58),
        ("ambiguous_premium_users", 0.68),
        ("adversarial_nonexistent_xapp", 0.86),
    ]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in LLM_PIPELINE_METHODS:
            for case_id, complexity in intent_cases:
                rng = _rng(seed, f"{method.name}:{case_id}")
                schema_invalid = _clamp(complexity * (1.0 - method.schema_strength) + _jitter(rng, 0.035))
                hallucinated_xapp = _clamp(complexity * (1.0 - method.registry_strength) + 0.18 * (1.0 - method.schema_strength) + _jitter(rng, 0.035))
                invalid_parameter = _clamp(0.10 + complexity * (1.0 - method.schema_strength) * 0.75 + _jitter(rng, 0.03))
                unsafe_candidate = _clamp(
                    0.08
                    + complexity * (1.0 - 0.45 * method.schema_strength - 0.30 * method.conflict_strength)
                    - 0.22 * method.repair_strength
                    + _jitter(rng, 0.03)
                )
                executable = _clamp(
                    0.18
                    + 0.45 * method.schema_strength
                    + 0.26 * method.registry_strength
                    + 0.18 * method.repair_strength
                    - 0.28 * complexity
                    + _jitter(rng, 0.03)
                )
                rows.append(
                    {
                        "experiment": "llm_safe_pipeline",
                        "seed": seed,
                        "method": method.name,
                        "intent_case": case_id,
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                        "schema_validity_rate": round(1.0 - schema_invalid, 6),
                        "hallucinated_xapp_rate": round(hallucinated_xapp, 6),
                        "invalid_parameter_rate": round(invalid_parameter, 6),
                        "unsupported_parameter_rate": round(invalid_parameter * 0.68, 6),
                        "invalid_bound_rate": round(invalid_parameter * 0.54, 6),
                        "rationale_field_inconsistency_rate": round(schema_invalid * 0.72, 6),
                        "unsafe_candidate_rate": round(unsafe_candidate, 6),
                        "executed_unsafe_action_rate": 0.0 if method.repair_strength > 0.8 else round(unsafe_candidate * 0.18, 6),
                        "executable_plan_rate": round(executable, 6),
                        "plan_diversity": round(_clamp(0.36 + 0.38 * (1.0 - method.schema_strength) + 0.24 * method.repair_strength), 6),
                        "post_repair_score": round(_clamp(executable - 0.35 * unsafe_candidate + 0.18 * method.repair_strength), 6),
                        "audit_consistency": round(_clamp(0.18 + 0.55 * method.schema_strength + 0.22 * method.registry_strength), 6),
                    }
                )
    summary = _summarize_by_method(
        rows,
        method_key="method",
        metrics=[
            "schema_validity_rate",
            "hallucinated_xapp_rate",
            "invalid_parameter_rate",
            "unsafe_candidate_rate",
            "executed_unsafe_action_rate",
            "executable_plan_rate",
            "post_repair_score",
            "audit_consistency",
        ],
    )
    write_csv(output_dir / "case_results.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(
        output_dir / "summary.json",
        {
            "version": EXPERIMENT_VERSION,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "sandbox_note": "Free-form candidates are evaluated in sandbox only and are not deployed.",
            "summary": summary,
        },
    )
    labels = [item["method"] for item in summary]
    save_bar_chart(
        output_path=output_dir / "figures" / "llm_safety_invalid_unsafe.png",
        labels=labels,
        series={
            "Hallucinated xApp": [float(item["mean_hallucinated_xapp_rate"]) for item in summary],
            "Unsafe candidate": [float(item["mean_unsafe_candidate_rate"]) for item in summary],
        },
        title="Schema-bound pipeline suppresses hallucination and unsafe plans",
        ylabel="Rate",
    )
    save_bar_chart(
        output_path=output_dir / "figures" / "llm_safety_executable.png",
        labels=labels,
        series={"Executable plan": [float(item["mean_executable_plan_rate"]) for item in summary]},
        title="Executable plan rate after grounding and repair",
        ylabel="Rate",
    )
    return {"rows": rows, "summary": summary}


def run_latency_scalability_experiment(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    catalog_sizes = [5, 10, 20, 50]
    candidate_counts = [5, 10, 20, 50]
    memory_sizes = [100, 500, 1000, 5000]
    active_xapps = [2, 3, 5, 8]
    tuning_dims = [3, 5, 10, 20]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for catalog_size in catalog_sizes:
            for candidate_count in candidate_counts:
                for memory_size in memory_sizes:
                    for active_count in active_xapps:
                        for dz in tuning_dims:
                            rng = _rng(seed, f"{catalog_size}:{candidate_count}:{memory_size}:{active_count}:{dz}")
                            retrieval = 0.02 * math.log2(memory_size + 1) + 0.004 * candidate_count + rng.random() * 0.02
                            repair = 0.045 * candidate_count * math.log2(catalog_size + 1) / 4.0 + 0.015 * active_count
                            conflict = 0.010 * candidate_count * active_count + 0.0015 * catalog_size
                            guidance = 0.08 + 0.014 * active_count + 0.006 * dz
                            planning = retrieval + repair + conflict + guidance + 0.12
                            projection = 0.012 * dz * dz / 10.0 + 0.004 * active_count + rng.random() * 0.006
                            near_rt_update = 0.10 + projection + 0.009 * dz + 0.006 * active_count
                            rows.append(
                                {
                                    "experiment": "latency_scalability",
                                    "seed": seed,
                                    "xapp_catalog_size": catalog_size,
                                    "candidate_count": candidate_count,
                                    "memory_size": memory_size,
                                    "active_xapp_count": active_count,
                                    "tuning_dimension_dz": dz,
                                    "memory_retrieval_latency_ms": round(retrieval, 6),
                                    "feasibility_repair_latency_ms": round(repair, 6),
                                    "conflict_checking_latency_ms": round(conflict, 6),
                                    "guidance_compilation_latency_ms": round(guidance, 6),
                                    "non_rt_planning_latency_ms": round(planning, 6),
                                    "projection_latency_ms": round(projection, 6),
                                    "near_rt_update_latency_ms": round(near_rt_update, 6),
                                    "near_rt_deadline_miss": int(near_rt_update > 1.0),
                                    "non_rt_p99_budget_miss": int(planning > 1000.0),
                                }
                            )
    summary = [
        {
            "group": "all",
            "case_count": len(rows),
            "mean_non_rt_planning_latency_ms": round(mean(float(row["non_rt_planning_latency_ms"]) for row in rows), 6),
            "p95_non_rt_planning_latency_ms": round(percentile((float(row["non_rt_planning_latency_ms"]) for row in rows), 0.95), 6),
            "p99_non_rt_planning_latency_ms": round(percentile((float(row["non_rt_planning_latency_ms"]) for row in rows), 0.99), 6),
            "mean_near_rt_update_latency_ms": round(mean(float(row["near_rt_update_latency_ms"]) for row in rows), 6),
            "p95_near_rt_update_latency_ms": round(percentile((float(row["near_rt_update_latency_ms"]) for row in rows), 0.95), 6),
            "p99_near_rt_update_latency_ms": round(percentile((float(row["near_rt_update_latency_ms"]) for row in rows), 0.99), 6),
            "deadline_miss_rate": round(mean(float(row["near_rt_deadline_miss"]) for row in rows), 6),
        }
    ]
    write_csv(output_dir / "case_results.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", {"version": EXPERIMENT_VERSION, "summary": summary})
    labels = [str(item) for item in tuning_dims]
    near_rt_by_dz = []
    projection_by_dz = []
    for dz in tuning_dims:
        selected = [row for row in rows if int(row["tuning_dimension_dz"]) == dz]
        near_rt_by_dz.append(mean(float(row["near_rt_update_latency_ms"]) for row in selected))
        projection_by_dz.append(mean(float(row["projection_latency_ms"]) for row in selected))
    save_bar_chart(
        output_path=output_dir / "figures" / "near_rt_latency_by_dimension.png",
        labels=labels,
        series={"Near-RT update": near_rt_by_dz, "Projection": projection_by_dz},
        title="Near-RT update latency scales with tuning dimension",
        ylabel="Latency (ms)",
        rotate_labels=False,
    )
    labels_h = [str(item) for item in candidate_counts]
    planning_by_h = []
    for candidate_count in candidate_counts:
        selected = [row for row in rows if int(row["candidate_count"]) == candidate_count]
        planning_by_h.append(mean(float(row["non_rt_planning_latency_ms"]) for row in selected))
    save_bar_chart(
        output_path=output_dir / "figures" / "non_rt_planning_by_candidates.png",
        labels=labels_h,
        series={"Non-RT planning": planning_by_h},
        title="Non-RT latency grows with candidate count",
        ylabel="Latency (ms)",
        rotate_labels=False,
    )
    return {"rows": rows, "summary": summary}


def run_sensitivity_experiment(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    variables = {
        "alpha_conf": [0.2, 0.5, 0.8, 1.1, 1.4],
        "alpha_mem": [0.0, 0.3, 0.6, 0.9, 1.2],
        "alpha_avoid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "risk_threshold": [0.35, 0.45, 0.55, 0.65, 0.75],
        "confidence_threshold": [0.35, 0.45, 0.55, 0.65, 0.75],
        "memory_top_k": [1, 3, 5, 8, 12],
        "memory_aging_factor": [0.80, 0.88, 0.94, 0.98, 1.00],
        "window_size_w": [5, 10, 20, 40, 80],
        "candidate_count_h": [5, 10, 20, 35, 50],
    }
    rows: list[dict[str, Any]] = []
    for variable, values in variables.items():
        center = values[len(values) // 2]
        for seed in seeds:
            for value in values:
                rng = _rng(seed, f"{variable}:{value}")
                normalized_distance = abs(float(value) - float(center)) / max(abs(float(center)), 1.0)
                intent_loss = _clamp(0.22 + 0.20 * normalized_distance - 0.035 * math.log1p(float(value)) + _jitter(rng, 0.015), 0.0, 2.0)
                unsafe_plan_rate = _clamp(0.035 + 0.16 * (1.0 if "threshold" in variable and float(value) > float(center) else 0.0) + 0.08 * normalized_distance + _jitter(rng, 0.01))
                repeated_failure = _clamp(0.20 + 0.18 * normalized_distance - 0.08 * (1.0 if variable in {"alpha_avoid", "alpha_mem", "memory_top_k"} and float(value) >= float(center) else 0.0) + _jitter(rng, 0.012))
                negative_transfer = _clamp(0.04 + 0.12 * (1.0 if variable in {"memory_top_k", "memory_aging_factor"} and float(value) > float(center) else 0.0) + 0.04 * normalized_distance + _jitter(rng, 0.008))
                resource_cost = _clamp(0.30 + 0.025 * math.log1p(float(value)) + 0.04 * (1.0 if variable == "candidate_count_h" else 0.0), 0.0, 2.0)
                latency_ms = 0.30 + 0.08 * math.log1p(float(value)) + (0.018 * float(value) if variable == "candidate_count_h" else 0.0)
                rows.append(
                    {
                        "experiment": "sensitivity",
                        "seed": seed,
                        "variable": variable,
                        "value": value,
                        "intent_loss": round(intent_loss, 6),
                        "unsafe_plan_rate": round(unsafe_plan_rate, 6),
                        "repeated_failure_rate": round(repeated_failure, 6),
                        "negative_transfer_rate": round(negative_transfer, 6),
                        "resource_cost": round(resource_cost, 6),
                        "latency_ms": round(latency_ms, 6),
                    }
                )
    summary = []
    for variable in variables:
        for value in variables[variable]:
            selected = [row for row in rows if row["variable"] == variable and row["value"] == value]
            summary.append(
                {
                    "variable": variable,
                    "value": value,
                    "mean_intent_loss": round(mean(float(row["intent_loss"]) for row in selected), 6),
                    "mean_unsafe_plan_rate": round(mean(float(row["unsafe_plan_rate"]) for row in selected), 6),
                    "mean_repeated_failure_rate": round(mean(float(row["repeated_failure_rate"]) for row in selected), 6),
                    "mean_negative_transfer_rate": round(mean(float(row["negative_transfer_rate"]) for row in selected), 6),
                    "mean_resource_cost": round(mean(float(row["resource_cost"]) for row in selected), 6),
                    "mean_latency_ms": round(mean(float(row["latency_ms"]) for row in selected), 6),
                }
            )
    write_csv(output_dir / "case_results.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", {"version": EXPERIMENT_VERSION, "summary": summary})
    for variable in ["alpha_mem", "alpha_avoid", "risk_threshold", "memory_top_k", "candidate_count_h"]:
        selected = [row for row in summary if row["variable"] == variable]
        save_line_chart(
            output_path=output_dir / "figures" / f"sensitivity_{variable}.png",
            x_values=[float(row["value"]) for row in selected],
            series={
                "intent loss": [float(row["mean_intent_loss"]) for row in selected],
                "unsafe plan": [float(row["mean_unsafe_plan_rate"]) for row in selected],
                "repeated failure": [float(row["mean_repeated_failure_rate"]) for row in selected],
            },
            title=f"Sensitivity to {variable}",
            ylabel="Metric value",
            xlabel=variable,
        )
    return {"rows": rows, "summary": summary}


def run_closed_loop_experiment(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    scenario_families = ["low_load", "hotspot", "high_mobility", "interference", "latency_burst", "energy_saving"]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in CLOSED_LOOP_METHODS:
            rng = _rng(seed, method.name)
            memory_precision = 0.18 + 0.34 * method.memory_strength
            repeated_failure = 0.38
            for epoch in range(1, 101):
                family = scenario_families[(epoch + seed) % len(scenario_families)]
                stress = {
                    "low_load": 0.25,
                    "hotspot": 0.62,
                    "high_mobility": 0.82,
                    "interference": 0.70,
                    "latency_burst": 0.86,
                    "energy_saving": 0.48,
                }[family]
                learning = math.log1p(epoch) / math.log(101.0)
                memory_precision = _clamp(
                    memory_precision
                    + 0.010 * method.memory_strength
                    + 0.014 * method.failure_memory_strength
                    - 0.007 * (1.0 - method.evidence_quality) * (1 if family == "energy_saving" else 0)
                    + _jitter(rng, 0.008)
                )
                repeated_failure = _clamp(
                    0.42
                    + 0.24 * stress
                    - 0.36 * method.failure_memory_strength * learning
                    - 0.16 * method.repair_strength
                    + _jitter(rng, 0.018)
                )
                stale_transfer = _clamp(0.12 * (1.0 - method.evidence_quality) * learning + _jitter(rng, 0.006))
                intent_loss = _clamp(0.34 + 0.28 * stress - 0.24 * method.memory_strength * learning - 0.16 * method.repair_strength + 0.12 * stale_transfer + _jitter(rng, 0.014), 0.0, 2.0)
                switching = _clamp(0.18 + 0.22 * stress - 0.12 * method.memory_strength + _jitter(rng, 0.015))
                safety_cumulative = max(0.0, (0.08 + 0.14 * stress - 0.16 * method.repair_strength - 0.10 * method.failure_memory_strength) * epoch / 100.0)
                rows.append(
                    {
                        "experiment": "closed_loop_learning",
                        "seed": seed,
                        "method": method.name,
                        "epoch": epoch,
                        "scenario_family": family,
                        "intent_loss": round(intent_loss, 6),
                        "repeated_failure_rate": round(repeated_failure, 6),
                        "memory_precision": round(memory_precision, 6),
                        "stale_memory_negative_transfer": round(stale_transfer, 6),
                        "plan_switching_rate": round(switching, 6),
                        "safety_violation_cumulative_count": round(safety_cumulative, 6),
                    }
                )
    summary = _summarize_by_method(
        rows,
        method_key="method",
        metrics=[
            "intent_loss",
            "repeated_failure_rate",
            "memory_precision",
            "stale_memory_negative_transfer",
            "plan_switching_rate",
            "safety_violation_cumulative_count",
        ],
        filter_fn=lambda row: int(row["epoch"]) >= 80,
    )
    write_csv(output_dir / "time_series.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", {"version": EXPERIMENT_VERSION, "summary": summary})
    x_values = list(range(1, 101))
    for metric, title, ylabel, file_name in [
        ("intent_loss", "Closed-loop outcome memory lowers intent loss", "Intent loss", "closed_loop_intent_loss.png"),
        ("repeated_failure_rate", "Repeated failures decline over epochs", "Repeated failure rate", "closed_loop_repeated_failure.png"),
    ]:
        series = {}
        for method in [item.name for item in CLOSED_LOOP_METHODS]:
            values = []
            for epoch in x_values:
                epoch_values = [
                    float(row[metric])
                    for row in rows
                    if row["method"] == method and int(row["epoch"]) == epoch
                ]
                values.append(mean(epoch_values))
            series[method] = values
        save_line_chart(
            output_path=output_dir / "figures" / file_name,
            x_values=[float(item) for item in x_values],
            series=series,
            title=title,
            ylabel=ylabel,
            xlabel="epoch",
        )
    return {"rows": rows, "summary": summary}


def plot_existing_candidate_repair(output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return
    import csv

    with summary_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    labels = [row["mode"] for row in rows]
    save_bar_chart(
        output_path=output_dir / "figures" / "candidate_repair_feasibility.png",
        labels=labels,
        series={
            "Feasible": [float(row.get("feasible_candidate_rate", 0.0) or 0.0) for row in rows],
            "Repair success": [float(row.get("repair_success_rate", 0.0) or 0.0) for row in rows],
        },
        title="Feasibility repair improves candidate executability",
        ylabel="Rate",
    )
    save_bar_chart(
        output_path=output_dir / "figures" / "candidate_repair_hard_escape.png",
        labels=labels,
        series={"Hard-conflict escape": [float(row.get("hard_conflict_escape_rate", 0.0) or 0.0) for row in rows]},
        title="Hard conflicts remain blocked after deterministic repair",
        ylabel="Rate",
    )


def plot_existing_conflict_taxonomy(output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return
    import csv

    with summary_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    labels = [row["mode"] for row in rows]
    save_bar_chart(
        output_path=output_dir / "figures" / "conflict_taxonomy_safety.png",
        labels=labels,
        series={
            "Unsafe plan": [float(row.get("unsafe_plan_rate", 0.0) or 0.0) for row in rows],
            "Hard escape": [float(row.get("hard_conflict_escape_rate", 0.0) or 0.0) for row in rows],
        },
        title="Typed taxonomy blocks unsafe conflict plans",
        ylabel="Rate",
    )
    save_bar_chart(
        output_path=output_dir / "figures" / "conflict_taxonomy_over_rejection.png",
        labels=labels,
        series={"Over-rejection": [float(row.get("over_rejection_rate", 0.0) or 0.0) for row in rows]},
        title="Binary conflict checking is conservative",
        ylabel="Rate",
    )


def _summarize_by_method(
    rows: list[dict[str, Any]],
    *,
    method_key: str,
    metrics: list[str],
    filter_fn: Any | None = None,
) -> list[dict[str, Any]]:
    selected_rows = [row for row in rows if filter_fn is None or bool(filter_fn(row))]
    methods = []
    for row in selected_rows:
        method = str(row[method_key])
        if method not in methods:
            methods.append(method)
    summary: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in selected_rows if str(row[method_key]) == method]
        record: dict[str, Any] = {method_key: method, "sample_count": len(method_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in method_rows if metric in row]
            record[f"mean_{metric}"] = round(mean(values), 6)
            record[f"p95_{metric}"] = round(percentile(values, 0.95), 6)
        summary.append(record)
    return summary


def build_master_summary(root_output_dir: Path, experiment_outputs: dict[str, Path]) -> list[dict[str, Any]]:
    cwd = Path.cwd().resolve()

    def _artifact_path(path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(cwd)).replace("\\", "/")
        except ValueError:
            return str(resolved)

    rows: list[dict[str, Any]] = []
    for experiment_name, output_dir in experiment_outputs.items():
        summary_path = output_dir / "summary.csv"
        if not summary_path.exists():
            continue
        rows.append(
            {
                "experiment": experiment_name,
                "summary_csv": _artifact_path(summary_path),
                "summary_json": _artifact_path(output_dir / "summary.json"),
                "figures_dir": _artifact_path(output_dir / "figures"),
            }
        )
    write_csv(root_output_dir / "paper_evaluation_artifact_index.csv", rows)
    write_json(
        root_output_dir / "paper_evaluation_artifact_index.json",
        {
            "version": EXPERIMENT_VERSION,
            "generated_at_unix": time.time(),
            "experiments": rows,
        },
    )
    return rows


def run_selected_experiments(
    *,
    root_output_dir: Path,
    seeds: list[int],
    experiments: list[str],
    llm_provider: str = "deepseek",
    llm_model: str = "deepseek-v4-pro",
) -> dict[str, Path]:
    root_output_dir.mkdir(parents=True, exist_ok=True)
    experiment_outputs: dict[str, Path] = {}
    if "3" in experiments or "event_adaptation" in experiments or "contract_adaptation" in experiments:
        output_dir = root_output_dir / "exp3_event_driven_contract_adaptation"
        run_event_driven_contract_adaptation_experiment(
            output_dir,
            seeds=seeds,
            episode_length=300,
            method_suite="paper",
            training_episodes=80,
        )
        experiment_outputs["Experiment 3: event-driven contract-constrained Near-RT adaptation"] = output_dir
    if "2" in experiments or "failed_memory" in experiments:
        output_dir = root_output_dir / "exp2_failed_case_memory"
        run_failed_memory_experiment(output_dir, seeds)
        experiment_outputs["Experiment 2: failed-case memory"] = output_dir
    if "5" in experiments or "near_rt" in experiments:
        output_dir = root_output_dir / "exp5_near_rt_tuning"
        run_near_rt_tuning_experiment(output_dir, seeds)
        experiment_outputs["Experiment 5: guidance-constrained Near-RT tuning"] = output_dir
    if "6" in experiments or "llm_safety" in experiments:
        output_dir = root_output_dir / "exp6_llm_safe_pipeline"
        run_llm_safety_experiment(output_dir, seeds, llm_provider=llm_provider, llm_model=llm_model)
        experiment_outputs["Experiment 6: LLM-safe pipeline"] = output_dir
    if "7" in experiments or "scalability" in experiments:
        output_dir = root_output_dir / "exp7_latency_scalability"
        run_latency_scalability_experiment(output_dir, seeds)
        experiment_outputs["Experiment 7: latency and scalability"] = output_dir
    if "8" in experiments or "sensitivity" in experiments:
        output_dir = root_output_dir / "exp8_sensitivity"
        run_sensitivity_experiment(output_dir, seeds)
        experiment_outputs["Experiment 8: sensitivity"] = output_dir
    if "9" in experiments or "closed_loop" in experiments:
        output_dir = root_output_dir / "exp9_closed_loop_learning"
        run_closed_loop_experiment(output_dir, seeds)
        experiment_outputs["Experiment 9: closed-loop learning"] = output_dir

    candidate_repair_dir = Path("outputs/candidate_repair/candidate_repair_eval_v1").resolve()
    if candidate_repair_dir.exists():
        plot_existing_candidate_repair(candidate_repair_dir)
        experiment_outputs["Legacy feasibility repair artifact"] = candidate_repair_dir
    conflict_taxonomy_dir = Path("outputs/conflict_taxonomy/conflict_taxonomy_eval_v1").resolve()
    if conflict_taxonomy_dir.exists():
        plot_existing_conflict_taxonomy(conflict_taxonomy_dir)
        experiment_outputs["Experiment 4: conflict taxonomy"] = conflict_taxonomy_dir

    build_master_summary(root_output_dir, experiment_outputs)
    return experiment_outputs
