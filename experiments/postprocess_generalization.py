"""Post-process cross-scenario generalization experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from core.config_loader import load_yaml_file
from experiments.eval_metrics import compute_cross_scenario_generalization_gap
from experiments.scenario_scan_utils import flatten_nested_record


CASE_METRIC_FIELDS = {
    "intent_loss_raw": "summary_experimental_metrics_intent_loss_mean_raw",
    "intent_loss_normalized": "summary_experimental_metrics_intent_loss_mean_normalized",
    "safety_violation_rate": "summary_experimental_metrics_safety_violation_rate",
    "stability_penalty_mean": "summary_experimental_metrics_stability_penalty_mean",
    "xapp_switching_cost_mean": "summary_experimental_metrics_xapp_switching_cost_mean",
    "mean_observable_throughput_mbps": "summary_experimental_metrics_kpi_summary_mean_observable_throughput_mbps",
    "final_observable_bler": "summary_experimental_metrics_kpi_summary_final_observable_bler",
}

METHOD_CONFIG_FIELDS = [
    "variant_name",
    "validator_enabled",
    "rl_guidance_enabled",
    "safety_shield_enabled",
    "static_xapp_selection_enabled",
    "reflection_enabled",
    "memory_mode",
    "selection_mode",
    "repair_mode",
    "conflict_mode",
    "llm_pipeline_mode",
    "near_rt_mode",
]


CASE_DIR_VARIANT_NAMES = {
    "orch": "orchestran_like_resource_aware",
    "hrl": "intent_hrl_like",
    "oreo": "oreo_like_service_graph",
}


def _write_csv(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(records) == 0:
        output_path.write_text("", encoding="utf-8")
        return
    field_names: list[str] = []
    for record in records:
        for field_name in record.keys():
            if field_name not in field_names:
                field_names.append(field_name)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(records)


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for row in reader:
            normalized_row = {
                str(key).strip(): str(value).strip()
                for key, value in dict(row).items()
            }
            if normalized_row.get("scenario_id", "") == "":
                continue
            rows.append(normalized_row)
    return rows


def _manifest_index(manifest_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        str(row["scenario_id"]).strip(): dict(row)
        for row in manifest_rows
        if str(row.get("scenario_id", "")).strip() != ""
    }


def _parse_case_dir_name(case_dir_name: str) -> tuple[str, str, str, int]:
    normalized = str(case_dir_name).strip()
    parts = normalized.split("__")
    if len(parts) >= 4 and str(parts[-3]).startswith("intent-"):
        scenario_id = "__".join(parts[:-3])
        intent_slug = str(parts[-3]).replace("intent-", "", 1)
        variant_name = str(parts[-2])
        seed_part = str(parts[-1])
    else:
        scenario_id, variant_name, seed_part = normalized.rsplit("__", 2)
        intent_slug = "edge-throughput-latency"
    seed = int(str(seed_part).replace("seed", "").strip())
    return scenario_id, intent_slug, CASE_DIR_VARIANT_NAMES.get(variant_name, variant_name), seed


def _mean(values: list[float]) -> float | None:
    if len(values) == 0:
        return None
    return float(sum(values) / float(len(values)))


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_case_records(
    *,
    experiment_dir: Path,
    manifest_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    manifest_by_scenario = _manifest_index(manifest_rows)
    case_records: list[dict[str, Any]] = []
    cases_dir = experiment_dir / "cases"
    if not cases_dir.exists():
        return case_records

    for case_dir in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
        final_summary_path = case_dir / "final_summary.json"
        if not final_summary_path.exists():
            continue
        scenario_id, intent_slug, variant_name, seed = _parse_case_dir_name(case_dir.name)
        with final_summary_path.open("r", encoding="utf-8") as json_file:
            final_summary = json.load(json_file)
        record = {
            "scenario_id": str(scenario_id),
            "scenario_case_id": str(scenario_id),
            "intent_slug": str(intent_slug),
            "variant_name": str(variant_name),
            "seed": int(seed),
            "case_output_dir": str(case_dir),
        }
        record.update(manifest_by_scenario.get(str(scenario_id), {}))
        record.update(flatten_nested_record(final_summary, prefix="summary_"))
        method_config = dict(final_summary.get("method_config", {}) or {})
        for field_name in METHOD_CONFIG_FIELDS:
            record[field_name] = method_config.get(field_name)
        record["active_intent_id"] = str(
            final_summary.get("final_policy_snapshot_summary", {}).get("active_intent_id", "") or ""
        )
        active_intent_text = str(
            final_summary.get("final_policy_snapshot_summary", {}).get("active_intent_text", "")
            or final_summary.get("active_intent_text", "")
            or ""
        )
        if active_intent_text != "":
            record["intent_text"] = active_intent_text
        case_records.append(record)
    return case_records


def collect_orchestration_behavior(
    *,
    experiment_dir: Path,
    case_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    behavior_rows: list[dict[str, Any]] = []
    for record in case_records:
        case_dir = Path(str(record.get("case_output_dir", "")))
        history_path = case_dir / "orchestration_epoch_history.json"
        if not history_path.exists():
            continue
        with history_path.open("r", encoding="utf-8") as json_file:
            epoch_history = json.load(json_file)
        for epoch_entry in list(epoch_history or []):
            row = {
                "scenario_id": str(record.get("scenario_id", "")),
                "split": str(record.get("split", "")),
                "topology_class": str(record.get("topology_class", "")),
                "load_class": str(record.get("load_class", "")),
                "user_distribution_class": str(record.get("user_distribution_class", "")),
                "xapp_pool_class": str(record.get("xapp_pool_class", "")),
                "variant_name": str(record.get("variant_name", "")),
                "seed": int(record.get("seed", 0) or 0),
                "active_intent_id": str(record.get("active_intent_id", "")),
                "intent_slug": str(record.get("intent_slug", "")),
                "intent_text": str(record.get("intent_text", "")),
                "case_output_dir": str(case_dir),
                "epoch_id": int(epoch_entry.get("epoch_id", 0) or 0),
                "slot_id": int(epoch_entry.get("slot_id", 0) or 0),
                "intent_changed_this_epoch": bool(epoch_entry.get("intent_changed_this_epoch", False)),
                "translation_refreshed_this_epoch": bool(
                    epoch_entry.get("translation_refreshed_this_epoch", False)
                ),
                "translation_query_id": str(epoch_entry.get("translation_query_id", "") or ""),
                "translation_query_encoding_version": str(
                    epoch_entry.get("translation_query_encoding_version", "") or ""
                ),
                "translation_query_retriever_backend_name": str(
                    epoch_entry.get("translation_query_retriever_backend_name", "") or ""
                ),
                "translation_query_embedding_dim": int(
                    epoch_entry.get("translation_query_embedding_dim", 0) or 0
                ),
                "translation_context_signature": json.dumps(
                    epoch_entry.get("translation_context_signature", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "translation_retrieved_context_count": int(
                    epoch_entry.get("translation_retrieved_context_count", 0) or 0
                ),
                "translation_retrieved_context_source_ids": json.dumps(
                    epoch_entry.get("translation_retrieved_context_source_ids", []),
                    ensure_ascii=False,
                ),
                "translation_retrieved_context_summary": json.dumps(
                    epoch_entry.get("translation_retrieved_context_summary", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "selected_xapps": json.dumps(epoch_entry.get("selected_xapps", []), ensure_ascii=False),
                "selection_query_id": str(epoch_entry.get("selection_query_id", "") or ""),
                "selection_query_slot_id": epoch_entry.get("selection_query_slot_id"),
                "selection_query_encoding_version": str(
                    epoch_entry.get("selection_query_encoding_version", "") or ""
                ),
                "selection_query_retriever_backend_name": str(
                    epoch_entry.get("selection_query_retriever_backend_name", "") or ""
                ),
                "selection_query_embedding_dim": int(
                    epoch_entry.get("selection_query_embedding_dim", 0) or 0
                ),
                "selection_query_kpis": json.dumps(
                    epoch_entry.get("selection_query_kpis", []),
                    ensure_ascii=False,
                ),
                "selection_query_target_profile": str(
                    epoch_entry.get("selection_query_target_profile", "") or ""
                ),
                "selection_query_previous_xapp_categories": json.dumps(
                    epoch_entry.get("selection_query_previous_xapp_categories", []),
                    ensure_ascii=False,
                ),
                "guidance_id": str(epoch_entry.get("guidance_id", "") or ""),
                "guidance_mode": (
                    "rl_guidance_contract"
                    if str(epoch_entry.get("guidance_id", "") or "").strip() != ""
                    else "legacy_reward_spec"
                ),
                "guidance_source_epoch": epoch_entry.get("guidance_source_epoch"),
                "guidance_fallback_used": bool(epoch_entry.get("guidance_fallback_used", False)),
                "selected_rule_ids": json.dumps(epoch_entry.get("selected_rule_ids", []), ensure_ascii=False),
                "hard_safety_envelope_count": int(epoch_entry.get("hard_safety_envelope_count", 0) or 0),
                "hard_safety_envelope_fields": json.dumps(
                    epoch_entry.get("hard_safety_envelope_fields", []),
                    ensure_ascii=False,
                ),
                "guided_action_region_count": int(epoch_entry.get("guided_action_region_count", 0) or 0),
                "guided_action_region_fields": json.dumps(
                    epoch_entry.get("guided_action_region_fields", []),
                    ensure_ascii=False,
                ),
                "forbidden_combination_count": int(epoch_entry.get("forbidden_combination_count", 0) or 0),
                "exploration_health_gate": json.dumps(
                    epoch_entry.get("exploration_health_gate", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "validator_enabled": bool(epoch_entry.get("validator_enabled", False)),
                "rl_guidance_enabled": bool(epoch_entry.get("rl_guidance_enabled", False)),
                "safety_shield_enabled": bool(epoch_entry.get("safety_shield_enabled", False)),
                "static_xapp_selection_enabled": bool(
                    epoch_entry.get("static_xapp_selection_enabled", False)
                ),
                "memory_mode": str(epoch_entry.get("memory_mode", "") or ""),
                "selection_mode": str(epoch_entry.get("selection_mode", "") or ""),
                "repair_mode": str(epoch_entry.get("repair_mode", "") or ""),
                "conflict_mode": str(epoch_entry.get("conflict_mode", "") or ""),
                "llm_pipeline_mode": str(epoch_entry.get("llm_pipeline_mode", "") or ""),
                "near_rt_mode": str(epoch_entry.get("near_rt_mode", "") or ""),
                "validator_schema_ok": epoch_entry.get("validator_schema_ok"),
                "validator_admissibility_ok": epoch_entry.get("validator_admissibility_ok"),
                "validator_range_ok": epoch_entry.get("validator_range_ok"),
                "selection_plan_id": str(epoch_entry.get("selection_plan_id", "") or ""),
                "selection_status": str(epoch_entry.get("selection_status", "") or ""),
                "selection_locked": bool(epoch_entry.get("selection_locked", False)),
                "selection_generation": int(epoch_entry.get("selection_generation", 0) or 0),
                "last_failure_reason": str(epoch_entry.get("last_failure_reason", "") or ""),
                "last_failure_gap_score": epoch_entry.get("last_failure_gap_score"),
                "last_retrieved_success_case_ids": json.dumps(
                    epoch_entry.get("last_retrieved_success_case_ids", []),
                    ensure_ascii=False,
                ),
                "last_retrieved_failure_case_ids": json.dumps(
                    epoch_entry.get("last_retrieved_failure_case_ids", []),
                    ensure_ascii=False,
                ),
                "last_retrieved_success_traces": json.dumps(
                    epoch_entry.get("last_retrieved_success_traces", []),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "last_retrieved_failure_traces": json.dumps(
                    epoch_entry.get("last_retrieved_failure_traces", []),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "candidate_plan_count": int(epoch_entry.get("candidate_plan_count", 0) or 0),
                "selected_candidate_id": str(epoch_entry.get("selected_candidate_id", "") or ""),
                "selected_candidate_score_version": str(
                    epoch_entry.get("selected_candidate_score_version", "") or ""
                ),
                "selected_candidate_score_direction": str(
                    epoch_entry.get("selected_candidate_score_direction", "") or ""
                ),
                "selected_candidate_predictor_name": str(
                    epoch_entry.get("selected_candidate_predictor_name", "") or ""
                ),
                "selected_candidate_total_score": epoch_entry.get("selected_candidate_total_score"),
                "selected_candidate_predicted_intent_loss": epoch_entry.get(
                    "selected_candidate_predicted_intent_loss"
                ),
                "selected_candidate_conflict_cost": epoch_entry.get(
                    "selected_candidate_conflict_cost"
                ),
                "selected_candidate_stability_risk": epoch_entry.get(
                    "selected_candidate_stability_risk"
                ),
                "selected_candidate_resource_pressure": epoch_entry.get(
                    "selected_candidate_resource_pressure"
                ),
                "selected_candidate_eta_profile_name": str(
                    epoch_entry.get("selected_candidate_eta_profile_name", "") or ""
                ),
                "selected_candidate_chi_profile_name": str(
                    epoch_entry.get("selected_candidate_chi_profile_name", "") or ""
                ),
                "selected_candidate_parameter_bound_count": int(
                    epoch_entry.get("selected_candidate_parameter_bound_count", 0) or 0
                ),
                "selected_candidate_feasible": bool(
                    epoch_entry.get("selected_candidate_feasible", False)
                ),
                "selected_candidate_projection_applied": bool(
                    epoch_entry.get("selected_candidate_projection_applied", False)
                ),
                "selected_candidate_total_cpu_cost": epoch_entry.get(
                    "selected_candidate_total_cpu_cost"
                ),
                "selected_candidate_total_mem_cost": epoch_entry.get(
                    "selected_candidate_total_mem_cost"
                ),
                "selected_candidate_rationale": str(
                    epoch_entry.get("selected_candidate_rationale", "") or ""
                ),
                "eta_profile_name": str(epoch_entry.get("eta_profile_name", "") or ""),
                "eta_goal_focus": str(epoch_entry.get("eta_goal_focus", "") or ""),
                "eta_payload": json.dumps(
                    epoch_entry.get("eta_payload", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "composition_profile_name": str(epoch_entry.get("composition_profile_name", "") or ""),
                "composition_mode": str(epoch_entry.get("composition_mode", "") or ""),
                "arbitration_mode": str(epoch_entry.get("arbitration_mode", "") or ""),
                "override_order": json.dumps(epoch_entry.get("override_order", []), ensure_ascii=False),
                "chi_defaulted": bool(epoch_entry.get("chi_defaulted", True)),
                "chi_notes": json.dumps(epoch_entry.get("chi_notes", []), ensure_ascii=False),
                "feasible": bool(epoch_entry.get("feasible", True)),
                "projection_applied": bool(epoch_entry.get("projection_applied", False)),
                "total_cpu_cost": epoch_entry.get("total_cpu_cost"),
                "total_mem_cost": epoch_entry.get("total_mem_cost"),
                "max_cpu_cost": epoch_entry.get("max_cpu_cost"),
                "max_mem_cost": epoch_entry.get("max_mem_cost"),
                "dependency_pairs": json.dumps(epoch_entry.get("dependency_pairs", []), ensure_ascii=False),
                "mutex_pairs": json.dumps(epoch_entry.get("mutex_pairs", []), ensure_ascii=False),
                "projection_notes": json.dumps(epoch_entry.get("projection_notes", []), ensure_ascii=False),
                "goal_id": str(epoch_entry.get("goal_id", "") or ""),
                "intent_loss_raw": epoch_entry.get("intent_loss_raw"),
                "intent_loss_normalized": epoch_entry.get("intent_loss_normalized"),
            }
            behavior_rows.append(row)
    return behavior_rows


def _intent_specs_from_config(config_snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_suite = config_snapshot.get("intent_suite")
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw_suite, list):
        for item in raw_suite:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("intent_slug", "") or item.get("intent_id", "") or "").strip()
            if slug == "":
                continue
            result[slug] = {
                "intent_slug": slug,
                "intent_id": str(item.get("intent_id", slug) or slug),
                "intent_family": str(item.get("intent_family", "") or ""),
                "intent_text": str(item.get("intent_text", "") or ""),
                "macro_weight": float(item.get("macro_weight", 1.0) or 1.0),
            }
    if len(result) == 0:
        result["edge-throughput-latency"] = {
            "intent_slug": "edge-throughput-latency",
            "intent_id": str(config_snapshot.get("intent_id", "intent:edge_throughput_latency")),
            "intent_family": str(config_snapshot.get("intent_family", "edge_throughput_latency")),
            "intent_text": str(config_snapshot.get("intent_text", "")),
            "macro_weight": 1.0,
        }
    return result


def _split_records(
    case_records: list[dict[str, Any]],
    *,
    variant_name: str,
    split: str | None = None,
) -> list[dict[str, Any]]:
    filtered = [
        record
        for record in case_records
        if str(record.get("variant_name", "")) == str(variant_name)
    ]
    if split is None:
        return filtered
    return [
        record
        for record in filtered
        if str(record.get("split", "")).strip().lower() == str(split).strip().lower()
    ]


def _metric_mean(case_records: list[dict[str, Any]], metric_field: str) -> float | None:
    values = [
        float(value)
        for value in (_safe_float(record.get(metric_field)) for record in case_records)
        if value is not None
    ]
    return _mean(values)


def aggregate_generalization_results(
    *,
    case_records: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
    variant_names: list[str],
    reference_scenario_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_splits = {
        "seen": int(sum(1 for row in manifest_rows if str(row.get("split", "")).lower() == "seen")),
        "unseen": int(sum(1 for row in manifest_rows if str(row.get("split", "")).lower() == "unseen")),
    }
    aggregated_rows: list[dict[str, Any]] = []
    aggregated_json: dict[str, Any] = {
        "reference_scenario_id": str(reference_scenario_id),
        "manifest_case_counts": manifest_splits,
        "variants": {},
    }

    for variant_name in variant_names:
        variant_records = _split_records(case_records, variant_name=variant_name)
        seen_records = _split_records(case_records, variant_name=variant_name, split="seen")
        unseen_records = _split_records(case_records, variant_name=variant_name, split="unseen")
        row = {
            "variant_name": str(variant_name),
            "case_count": int(len(variant_records)),
            "seen_case_count": int(len(seen_records)),
            "unseen_case_count": int(len(unseen_records)),
        }
        for metric_name, metric_field in CASE_METRIC_FIELDS.items():
            seen_value = _metric_mean(seen_records, metric_field)
            unseen_value = _metric_mean(unseen_records, metric_field)
            all_value = _metric_mean(variant_records, metric_field)
            row[f"seen_mean_{metric_name}"] = seen_value
            row[f"unseen_mean_{metric_name}"] = unseen_value
            row[f"all_mean_{metric_name}"] = all_value

        seen_raw = row.get("seen_mean_intent_loss_raw")
        unseen_raw = row.get("unseen_mean_intent_loss_raw")
        seen_normalized = row.get("seen_mean_intent_loss_normalized")
        unseen_normalized = row.get("unseen_mean_intent_loss_normalized")
        row["transfer_gap_intent_loss_raw"] = (
            float(unseen_raw) - float(seen_raw)
            if seen_raw is not None and unseen_raw is not None
            else None
        )
        row["transfer_gap_intent_loss_normalized"] = (
            float(unseen_normalized) - float(seen_normalized)
            if seen_normalized is not None and unseen_normalized is not None
            else None
        )
        method_config = {
            field_name: next(
                (
                    record.get(field_name)
                    for record in variant_records
                    if record.get(field_name) is not None
                ),
                None,
            )
            for field_name in METHOD_CONFIG_FIELDS
        }
        row.update({field_name: method_config[field_name] for field_name in METHOD_CONFIG_FIELDS})
        generalization_gap = compute_cross_scenario_generalization_gap(
            variant_records,
            reference_scenario_name=str(reference_scenario_id),
            metric_field=CASE_METRIC_FIELDS["intent_loss_normalized"],
        )
        row["reference_scenario_id"] = str(reference_scenario_id)
        row["reference_case_metric_intent_loss_normalized"] = generalization_gap.get(
            "reference_metric_value"
        )
        row["mean_reference_gap_intent_loss_normalized"] = generalization_gap.get("mean_gap")
        aggregated_rows.append(row)
        aggregated_json["variants"][str(variant_name)] = {
            "method_config": method_config,
            "summary": dict(row),
            "cross_scenario_generalization_gap": generalization_gap,
        }
    return aggregated_rows, aggregated_json


def aggregate_intent_results(
    *,
    case_records: list[dict[str, Any]],
    variant_names: list[str],
    intent_specs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    intent_slugs = sorted(
        {
            str(record.get("intent_slug", "edge-throughput-latency") or "edge-throughput-latency")
            for record in case_records
        }
        | set(intent_specs.keys())
    )
    rows: list[dict[str, Any]] = []
    for intent_slug in intent_slugs:
        intent_records = [
            record
            for record in case_records
            if str(record.get("intent_slug", "edge-throughput-latency") or "edge-throughput-latency") == intent_slug
        ]
        intent_spec = dict(intent_specs.get(intent_slug, {}) or {})
        for variant_name in variant_names:
            variant_records = [
                record
                for record in intent_records
                if str(record.get("variant_name", "")) == str(variant_name)
            ]
            row: dict[str, Any] = {
                "intent_slug": intent_slug,
                "intent_id": str(intent_spec.get("intent_id", intent_slug)),
                "intent_family": str(intent_spec.get("intent_family", "")),
                "intent_text": str(
                    intent_spec.get("intent_text", "")
                    or next((record.get("intent_text", "") for record in variant_records if record.get("intent_text", "")), "")
                ),
                "intent_macro_weight": float(intent_spec.get("macro_weight", 1.0) or 1.0),
                "variant_name": str(variant_name),
                "case_count": int(len(variant_records)),
            }
            for metric_name, metric_field in CASE_METRIC_FIELDS.items():
                row[f"mean_{metric_name}"] = _metric_mean(variant_records, metric_field)
            rows.append(row)
    return rows


def aggregate_macro_average_results(
    *,
    intent_rows: list[dict[str, Any]],
    variant_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    macro_rows: list[dict[str, Any]] = []
    macro_json: dict[str, Any] = {"variants": {}}
    for variant_name in variant_names:
        rows = [
            row
            for row in intent_rows
            if str(row.get("variant_name", "")) == str(variant_name)
            and int(row.get("case_count", 0) or 0) > 0
        ]
        total_weight = sum(float(row.get("intent_macro_weight", 1.0) or 1.0) for row in rows)
        total_weight = total_weight if total_weight > 0.0 else float(len(rows) or 1)
        record: dict[str, Any] = {
            "variant_name": str(variant_name),
            "intent_count": int(len(rows)),
            "case_count": int(sum(int(row.get("case_count", 0) or 0) for row in rows)),
            "macro_weight_sum": float(total_weight),
        }
        for metric_name in CASE_METRIC_FIELDS.keys():
            weighted_values = [
                (
                    _safe_float(row.get(f"mean_{metric_name}")),
                    float(row.get("intent_macro_weight", 1.0) or 1.0),
                )
                for row in rows
            ]
            numerator = sum(float(value) * float(weight) for value, weight in weighted_values if value is not None)
            denominator = sum(float(weight) for value, weight in weighted_values if value is not None)
            record[f"macro_mean_{metric_name}"] = (
                numerator / denominator if denominator > 0.0 else None
            )
        macro_rows.append(record)
        macro_json["variants"][str(variant_name)] = dict(record)
    return macro_rows, macro_json


def postprocess_generalization_artifacts(
    *,
    experiment_dir: Path,
    manifest_path: Path,
    reference_scenario_id: str,
    variant_names: list[str] | None = None,
) -> dict[str, Any]:
    manifest_rows = load_manifest(manifest_path)
    case_records = collect_case_records(
        experiment_dir=experiment_dir,
        manifest_rows=manifest_rows,
    )
    if variant_names is None:
        variant_names = sorted(
            {
                str(record.get("variant_name", ""))
                for record in case_records
                if str(record.get("variant_name", "")).strip() != ""
            }
        )
    behavior_rows = collect_orchestration_behavior(
        experiment_dir=experiment_dir,
        case_records=case_records,
    )
    config_snapshot = load_yaml_file(str(experiment_dir / "config_snapshot.yaml"))
    intent_specs = _intent_specs_from_config(config_snapshot)
    aggregated_rows, aggregated_json = aggregate_generalization_results(
        case_records=case_records,
        manifest_rows=manifest_rows,
        variant_names=list(variant_names),
        reference_scenario_id=str(reference_scenario_id),
    )
    intent_rows = aggregate_intent_results(
        case_records=case_records,
        variant_names=list(variant_names),
        intent_specs=intent_specs,
    )
    macro_rows, macro_json = aggregate_macro_average_results(
        intent_rows=intent_rows,
        variant_names=list(variant_names),
    )

    _write_csv(experiment_dir / "case_results.csv", case_records)
    _write_csv(experiment_dir / "orchestration_behavior.csv", behavior_rows)
    _write_csv(experiment_dir / "aggregated_results.csv", aggregated_rows)
    _write_csv(experiment_dir / "intent_aggregated_results.csv", intent_rows)
    _write_csv(experiment_dir / "macro_average_results.csv", macro_rows)
    with (experiment_dir / "aggregated_results.json").open("w", encoding="utf-8") as json_file:
        json.dump(aggregated_json, json_file, indent=2, ensure_ascii=False)
    with (experiment_dir / "macro_average_results.json").open("w", encoding="utf-8") as json_file:
        json.dump(macro_json, json_file, indent=2, ensure_ascii=False)

    plots_input_dir = experiment_dir / "plots_input"
    plots_input_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (
        "case_results.csv",
        "orchestration_behavior.csv",
        "aggregated_results.csv",
        "intent_aggregated_results.csv",
        "macro_average_results.csv",
    ):
        shutil.copy2(experiment_dir / file_name, plots_input_dir / file_name)

    return {
        "manifest_rows": manifest_rows,
        "case_records": case_records,
        "behavior_rows": behavior_rows,
        "aggregated_rows": aggregated_rows,
        "intent_rows": intent_rows,
        "macro_rows": macro_rows,
        "aggregated_json": aggregated_json,
        "macro_json": macro_json,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process cross-scenario generalization outputs.")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment output directory produced by run_cross_scenario_generalization.py",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest path. Defaults to <experiment-dir>/scenario_manifest.csv",
    )
    parser.add_argument(
        "--reference-scenario-id",
        type=str,
        default="",
        help="Override the reference scenario used for transfer-gap reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment_dir = Path(str(args.experiment_dir)).resolve()
    manifest_path = (
        Path(str(args.manifest)).resolve()
        if str(args.manifest).strip() != ""
        else experiment_dir / "scenario_manifest.csv"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    config_snapshot = load_yaml_file(str(experiment_dir / "config_snapshot.yaml"))
    default_reference = str(config_snapshot.get("reference_scenario_id", "benchmark_nominal_cluster"))
    reference_scenario_id = str(args.reference_scenario_id or default_reference or "benchmark_nominal_cluster")
    variant_names = list(config_snapshot.get("variants", []) or [])

    artifacts = postprocess_generalization_artifacts(
        experiment_dir=experiment_dir,
        manifest_path=manifest_path,
        reference_scenario_id=reference_scenario_id,
        variant_names=variant_names or None,
    )
    print(
        f"Post-processed {len(artifacts['case_records'])} case records into "
        f"{experiment_dir / 'aggregated_results.csv'} and {experiment_dir / 'orchestration_behavior.csv'}."
    )


if __name__ == "__main__":
    main()
