"""Warm-up/transfer runner for Experiment 1 cross-scenario orchestration."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from core.config_loader import load_yaml_file
from core.debug import DebugConfig
from experiments.experiment_variants import build_minimal_paper_variants
from experiments.postprocess_generalization import postprocess_generalization_artifacts
from experiments.run_cross_scenario_generalization import (
    _apply_xapp_pool_profile,
    _assert_preflight_requirements,
    _build_case_record,
    _intent_suite_from_config,
    _load_intent_orchestration_configs,
    _load_manifest,
    _prepare_configs_for_run,
    _resolve_manifest_cases,
    _save_config_snapshot,
    _save_manifest_copy,
    _save_run_plan,
    _write_csv,
)
from experiments.scenario_scan_utils import run_matrix_case


CASE_DIR_VARIANT_ALIASES = {
    "orchestran_like_resource_aware": "orch",
    "intent_hrl_like": "hrl",
    "oreo_like_service_graph": "oreo",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 warm-up/transfer evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/experiment1_cross_scenario_transfer.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Only regenerate aggregate outputs from completed case directories.",
    )
    return parser.parse_args()


def _write_transfer_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_path = output_dir / "transfer_case_manifest.csv"
    field_names: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in field_names:
                field_names.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _case_json_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
        epoch_count = int(
            dict(payload.get("orchestration_epoch_history_summary", {}) or {}).get(
                "epoch_count", 0
            )
            or 0
        )
        if epoch_count <= 0:
            return False
        return True
    except Exception:
        return False


def _case_output_dir(
    *,
    root_output_dir: Path,
    phase_label: str,
    scenario_id: str,
    intent_slug: str,
    variant_name: str,
    seed: int,
) -> Path:
    legacy_output_dir = (
        root_output_dir
        / "cases"
        / f"{phase_label}__{scenario_id}__intent-{intent_slug}__{variant_name}__seed{int(seed)}"
    )
    if _case_json_is_valid(legacy_output_dir / "final_summary.json"):
        return legacy_output_dir

    case_variant_name = CASE_DIR_VARIANT_ALIASES.get(str(variant_name), str(variant_name))
    if case_variant_name == str(variant_name):
        return legacy_output_dir
    return (
        root_output_dir
        / "cases"
        / f"{phase_label}__{scenario_id}__intent-{intent_slug}__{case_variant_name}__seed{int(seed)}"
    )


def _memory_file_token(raw_value: object, *, fallback: str) -> str:
    raw = str(raw_value or "").strip() or str(fallback)
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return token.strip("._-") or str(fallback)


def _memory_path(
    *,
    root_output_dir: Path,
    variant_name: str,
    intent_slug: str,
    seed: int,
    task_id: str = "",
    memory_transfer: dict[str, Any] | None = None,
) -> str:
    transfer_memory = dict(memory_transfer or {})
    cumulative_enabled = bool(
        transfer_memory.get(
            "cumulative_memory_enabled",
            transfer_memory.get("shared_memory_enabled", False),
        )
    )
    if cumulative_enabled:
        raw_dir = str(
            transfer_memory.get("cumulative_memory_dir", "")
            or transfer_memory.get("shared_memory_dir", "")
            or ""
        ).strip()
        base_dir = (
            Path(raw_dir)
            if raw_dir != ""
            else root_output_dir.parent / "exp1_cumulative_memory"
        )
        scope = str(transfer_memory.get("shared_memory_scope", "seed") or "seed").strip().lower()
        if scope == "global":
            file_name = "global.json"
        elif scope in {"task_seed", "task-seed", "task"}:
            task_token = _memory_file_token(task_id, fallback=str(intent_slug or "task"))
            file_name = f"{task_token}__seed{int(seed)}.json"
        elif scope == "intent":
            file_name = f"intent-{intent_slug}__seed{int(seed)}.json"
        else:
            file_name = f"seed{int(seed)}.json"
        return str(base_dir / variant_name / file_name)
    return str(root_output_dir / "memory" / variant_name / f"intent-{intent_slug}__seed{int(seed)}.json")


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip() != ""]
    text = str(value or "").strip()
    return [text] if text != "" else []


def _bootstrap_source_variants(
    *,
    variant_name: str,
    memory_transfer: dict[str, Any],
) -> list[str]:
    explicit_sources = dict(memory_transfer.get("bootstrap_variant_sources", {}) or {})
    candidates: list[str] = []
    candidates.extend(_as_string_list(explicit_sources.get(str(variant_name), "")))
    candidates.append(str(variant_name))
    if str(variant_name) != "success_only_memory":
        candidates.append("full_method")
    candidates.append("success_only_memory")
    return list(dict.fromkeys(item for item in candidates if str(item).strip() != ""))


def _bootstrap_source_file_names(*, seed: int, task_id: str, intent_slug: str) -> list[str]:
    task_token = _memory_file_token(task_id, fallback=str(intent_slug or "task"))
    names = [
        f"{task_token}__seed{int(seed)}.json",
        f"{task_token}__seed1.json",
        f"seed{int(seed)}.json",
        "seed1.json",
    ]
    return list(dict.fromkeys(names))


def _ensure_bootstrap_memory_store(
    *,
    store_path: str,
    variant_name: str,
    intent_slug: str,
    seed: int,
    task_id: str,
    transfer_config: dict[str, Any],
    memory_transfer: dict[str, Any],
) -> None:
    target_path = Path(str(store_path))
    if target_path.exists():
        return

    protocol_config = dict(transfer_config.get("experiment1_protocol", {}) or {})
    raw_source_dir = str(
        memory_transfer.get("bootstrap_memory_source", "")
        or transfer_config.get("bootstrap_memory_source", "")
        or protocol_config.get("bootstrap_memory_source", "")
        or ""
    ).strip()
    if raw_source_dir == "":
        return
    source_dir = Path(raw_source_dir)
    if not source_dir.exists():
        return

    source_variants = _bootstrap_source_variants(
        variant_name=variant_name,
        memory_transfer=memory_transfer,
    )
    source_file_names = _bootstrap_source_file_names(
        seed=int(seed),
        task_id=task_id,
        intent_slug=intent_slug,
    )
    for source_variant in source_variants:
        for file_name in source_file_names:
            source_path = source_dir / source_variant / file_name
            if not source_path.exists() or source_path.resolve() == target_path.resolve():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            return


def _apply_transfer_memory_overrides(
    *,
    ric_config: dict[str, Any],
    variant_name: str,
    intent_slug: str,
    seed: int,
    root_output_dir: Path,
    transfer_config: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(dict(ric_config or {}))
    memory_transfer = dict(transfer_config.get("memory_transfer", {}) or {})
    persistent_variants = {
        str(item)
        for item in list(memory_transfer.get("persistent_variants", []) or [])
        if str(item).strip() != ""
    }
    persistent_enabled = bool(memory_transfer.get("enabled", True)) and variant_name in persistent_variants
    if not persistent_enabled:
        return updated

    protocol_config = dict(transfer_config.get("experiment1_protocol", {}) or {})
    task_id = str(protocol_config.get("task_id", "") or "")
    store_path = _memory_path(
        root_output_dir=root_output_dir,
        variant_name=variant_name,
        intent_slug=intent_slug,
        seed=seed,
        task_id=task_id,
        memory_transfer=memory_transfer,
    )
    _ensure_bootstrap_memory_store(
        store_path=store_path,
        variant_name=variant_name,
        intent_slug=intent_slug,
        seed=seed,
        task_id=task_id,
        transfer_config=transfer_config,
        memory_transfer=memory_transfer,
    )
    global_overrides = {
        "persistent_memory_enabled": True,
        "memory_store_path": store_path,
        "success_memory_size": int(memory_transfer.get("success_memory_size", 48)),
        "reflection_memory_size": int(memory_transfer.get("reflection_memory_size", 48)),
        "retrieval_top_k": int(memory_transfer.get("retrieval_top_k", 8)),
        "failure_gap_threshold": float(memory_transfer.get("failure_gap_threshold", 0.12)),
        "convergence_gap_threshold": float(memory_transfer.get("convergence_gap_threshold", 0.12)),
        "success_gap_threshold": float(memory_transfer.get("success_gap_threshold", 0.10)),
        "gap_patience_rounds": int(memory_transfer.get("gap_patience_rounds", 1)),
        "stagnation_patience_rounds": int(memory_transfer.get("stagnation_patience_rounds", 1)),
    }
    updated.update(global_overrides)
    for rapp_item in list(updated.get("rapps", []) or []):
        if str(rapp_item.get("type", "")) != "xapp_selection_rapp":
            continue
        rapp_item.update(global_overrides)
    return updated


def _apply_literature_baseline_state_overrides(
    *,
    ric_config: dict[str, Any],
    variant_name: str,
    intent_slug: str,
    seed: int,
    root_output_dir: Path,
    phase: str,
    repeat_index: int,
) -> dict[str, Any]:
    updated = deepcopy(dict(ric_config or {}))
    state_dir = root_output_dir / "literature_baseline_state" / str(variant_name)
    state_path = state_dir / f"intent-{intent_slug}__seed{int(seed)}.json"
    baseline_overrides = {
        "experiment_transfer_phase": str(phase),
        "experiment_phase_repeat": int(repeat_index),
        "literature_baseline_state_path": str(state_path),
        "literature_baseline_state_enabled": str(variant_name)
        in {
            "orchestran_like_resource_aware",
            "intent_hrl_like",
            "oreo_like_service_graph",
        },
        "intent_hrl_train_on_warmup": str(phase) == "warmup",
        "intent_hrl_freeze_on_transfer": str(phase) == "transfer",
    }
    updated.update(baseline_overrides)
    for rapp_item in list(updated.get("rapps", []) or []):
        if str(rapp_item.get("type", "")) != "xapp_selection_rapp":
            continue
        rapp_item.update(baseline_overrides)
    return updated


def _phase_rows(
    *,
    resolved_rows: list[dict[str, Any]],
    phase: str,
    selected_scenario_ids: list[str],
) -> list[dict[str, Any]]:
    selected = {str(item) for item in list(selected_scenario_ids or [])}
    rows: list[dict[str, Any]] = []
    for item in resolved_rows:
        scenario_id = str(item["manifest"].get("scenario_id", item["case"].case_id))
        if scenario_id in selected:
            row = dict(item)
            row["transfer_phase"] = str(phase)
            rows.append(row)
    return rows


def _with_forced_case_sim_overrides(
    *,
    case: Any,
    force_sim_overrides: dict[str, Any],
) -> Any:
    if len(dict(force_sim_overrides or {})) == 0:
        return case
    updated_case = deepcopy(case)
    updated_case.sim_overrides = {
        **dict(getattr(case, "sim_overrides", {}) or {}),
        **dict(force_sim_overrides or {}),
    }
    return updated_case


def _aggregate_transfer_headline(root_output_dir: Path) -> None:
    records_path = root_output_dir / "transfer_case_results.csv"
    if not records_path.exists():
        return
    with records_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    warmup_rows = [row for row in rows if row.get("transfer_phase") == "warmup"]
    transfer_rows = [row for row in rows if row.get("transfer_phase") == "transfer"]
    if len(transfer_rows) == 0:
        return

    metrics = [
        ("intent_loss_normalized", "summary_experimental_metrics_intent_loss_mean_normalized"),
        ("intent_loss_raw", "summary_experimental_metrics_intent_loss_mean_raw"),
        ("intent_target_fulfillment", "summary_experimental_metrics_intent_target_fulfillment_mean"),
        ("hard_target_fulfillment", "summary_experimental_metrics_hard_target_fulfillment_mean"),
        ("hard_intent_fulfillment", "summary_experimental_metrics_hard_intent_fulfillment_ratio"),
        ("hard_target_violation_count", "summary_experimental_metrics_hard_target_violation_count_mean"),
        ("safety_violation_rate", "summary_experimental_metrics_safety_violation_rate"),
        ("switching_cost", "summary_experimental_metrics_xapp_switching_cost_mean"),
        ("throughput_mbps", "summary_experimental_metrics_kpi_summary_mean_observable_throughput_mbps"),
        ("final_throughput_mbps", "summary_experimental_metrics_kpi_summary_final_observable_throughput_mbps"),
        ("mean_bler", "summary_experimental_metrics_kpi_summary_mean_observable_bler"),
        ("final_bler", "summary_experimental_metrics_kpi_summary_final_observable_bler"),
        ("health_gate_trigger_rate", "summary_experimental_metrics_health_gate_trigger_rate"),
        ("safe_baseline_fallback_rate", "summary_experimental_metrics_safe_baseline_fallback_rate"),
        ("forced_non_rt_reselection_count", "summary_orchestration_epoch_history_summary_forced_non_rt_reselection_count"),
        ("epoch_count", "summary_orchestration_epoch_history_summary_epoch_count"),
        ("success_memory_count", "summary_orchestration_epoch_history_summary_last_epoch_success_memory_count"),
        ("failure_memory_count", "summary_orchestration_epoch_history_summary_last_epoch_failure_memory_count"),
    ]
    by_variant: dict[str, list[dict[str, str]]] = {}
    by_intent_variant: dict[tuple[str, str], list[dict[str, str]]] = {}
    by_scenario_variant: dict[tuple[str, str], list[dict[str, str]]] = {}
    by_family_variant: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in transfer_rows:
        by_variant.setdefault(str(row.get("variant_name", "")), []).append(row)
        by_intent_variant.setdefault(
            (str(row.get("intent_slug", "")), str(row.get("variant_name", ""))),
            [],
        ).append(row)
        by_scenario_variant.setdefault(
            (str(row.get("scenario_id", "")), str(row.get("variant_name", ""))),
            [],
        ).append(row)
        by_family_variant.setdefault(
            (str(row.get("intent_family", "")), str(row.get("variant_name", ""))),
            [],
        ).append(row)

    def _mean(items: list[dict[str, str]], field_name: str) -> float:
        values: list[float] = []
        for item in items:
            try:
                values.append(float(item.get(field_name, 0.0) or 0.0))
            except ValueError:
                continue
        return sum(values) / float(len(values)) if values else 0.0

    def _std(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        avg = sum(values) / float(len(values))
        return (sum((value - avg) ** 2 for value in values) / float(len(values))) ** 0.5

    def _case_safety_first_score(row: dict[str, str]) -> float:
        intent_loss = _mean([row], "summary_experimental_metrics_intent_loss_mean_normalized")
        safety = _mean([row], "summary_experimental_metrics_safety_violation_rate")
        hard_shortfall = 1.0 - _mean([row], "summary_experimental_metrics_hard_target_fulfillment_mean")
        bler = _mean([row], "summary_experimental_metrics_kpi_summary_mean_observable_bler")
        throughput = _mean([row], "summary_experimental_metrics_kpi_summary_mean_observable_throughput_mbps")
        switching = _mean([row], "summary_experimental_metrics_xapp_switching_cost_mean")
        return (
            intent_loss
            + 0.25 * max(0.0, hard_shortfall)
            + 0.35 * safety
            + 0.05 * bler
            + 0.02 * switching
            - 0.001 * throughput
        )

    def _is_memory_sensitive_row(row: dict[str, str]) -> bool:
        text = (
            str(row.get("intent_slug", ""))
            + " "
            + str(row.get("intent_family", ""))
            + " "
            + str(row.get("scenario_id", ""))
            + " "
            + str(row.get("stress", ""))
            + " "
            + str(row.get("notes", ""))
        ).lower()
        tokens = (
            "mobility",
            "handover",
            "ping",
            "conflict",
            "interference",
            "blockage",
            "hotspot_shift",
            "stale",
            "negative_transfer",
            "policy",
            "safety",
        )
        return any(token in text for token in tokens)

    def _tail_mean(values: list[float], *, fraction: float = 0.10) -> float:
        if len(values) == 0:
            return 0.0
        sorted_values = sorted(values, reverse=True)
        count = max(1, int(round(float(len(sorted_values)) * float(fraction))))
        return sum(sorted_values[:count]) / float(count)

    def _add_transfer_scores(out: dict[str, Any]) -> None:
        intent_loss = float(out.get("transfer_mean_intent_loss_normalized", 0.0) or 0.0)
        hard_shortfall = max(
            0.0,
            1.0 - float(out.get("transfer_mean_hard_target_fulfillment", 1.0) or 1.0),
        )
        safety = float(out.get("transfer_mean_safety_violation_rate", 0.0) or 0.0)
        bler = float(out.get("transfer_mean_mean_bler", 0.0) or 0.0)
        throughput = float(out.get("transfer_mean_throughput_mbps", 0.0) or 0.0)
        switching = float(out.get("transfer_mean_switching_cost", 0.0) or 0.0)
        epoch_count = max(1.0, float(out.get("transfer_mean_epoch_count", 1.0) or 1.0))
        failure_memory_count = float(out.get("transfer_mean_failure_memory_count", 0.0) or 0.0)
        out["transfer_cross_scenario_score"] = (
            intent_loss
            + 0.18 * hard_shortfall
            + 0.10 * safety
            + 0.02 * bler
            - 0.004 * throughput
        )
        out["transfer_safety_first_score"] = (
            intent_loss
            + 0.25 * hard_shortfall
            + 0.35 * safety
            + 0.05 * bler
            + 0.02 * switching
            - 0.001 * throughput
        )
        out["transfer_safety_success_rate"] = max(0.0, 1.0 - safety)
        out["transfer_failure_memory_density"] = failure_memory_count / epoch_count

    def _write_rows(path: Path, selected_rows: list[dict[str, Any]]) -> None:
        field_names: list[str] = []
        for selected_row in selected_rows:
            for key in selected_row.keys():
                if key not in field_names:
                    field_names.append(key)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(selected_rows)

    def _phase_summary(selected_rows: list[dict[str, str]]) -> dict[str, float]:
        out: dict[str, Any] = {}
        for metric_name, field_name in metrics:
            out[f"transfer_mean_{metric_name}"] = _mean(selected_rows, field_name)
        _add_transfer_scores(out)
        return {
            "intent_loss_normalized": float(out.get("transfer_mean_intent_loss_normalized", 0.0) or 0.0),
            "hard_target_fulfillment": float(out.get("transfer_mean_hard_target_fulfillment", 0.0) or 0.0),
            "hard_intent_fulfillment": float(out.get("transfer_mean_hard_intent_fulfillment", 0.0) or 0.0),
            "safety_violation_rate": float(out.get("transfer_mean_safety_violation_rate", 0.0) or 0.0),
            "safety_first_score": float(out.get("transfer_safety_first_score", 0.0) or 0.0),
            "tail_robustness_score": float(out.get("transfer_tail_robustness_score", 0.0) or 0.0),
            "throughput_mbps": float(out.get("transfer_mean_throughput_mbps", 0.0) or 0.0),
            "switching_cost": float(out.get("transfer_mean_switching_cost", 0.0) or 0.0),
            "safe_baseline_fallback_rate": float(
                out.get("transfer_mean_safe_baseline_fallback_rate", 0.0) or 0.0
            ),
            "forced_non_rt_reselection_count": float(
                out.get("transfer_mean_forced_non_rt_reselection_count", 0.0) or 0.0
            ),
        }

    def _risk_bucket(row: dict[str, str]) -> str:
        text = (
            str(row.get("intent_slug", ""))
            + " "
            + str(row.get("intent_family", ""))
            + " "
            + str(row.get("scenario_id", ""))
            + " "
            + str(row.get("stress", ""))
            + " "
            + str(row.get("notes", ""))
        ).lower()
        if any(token in text for token in ("stale", "negative_transfer")):
            return "stale_memory_shift"
        if any(token in text for token in ("blockage", "hotspot_shift")):
            return "blockage_or_hotspot_shift"
        if any(token in text for token in ("mobility", "handover", "ping-pong", "ping_pong")):
            return "mobility_handover_risk"
        if any(token in text for token in ("interference", "sinr", "cell-edge", "edge")):
            return "interference_edge_risk"
        if any(token in text for token in ("urllc", "latency", "reliability", "premium")):
            return "latency_reliability_risk"
        if any(token in text for token in ("safety", "policy", "conflict")):
            return "policy_safety_risk"
        return ""

    def _linear_slope(points: list[tuple[float, float]]) -> float:
        clean_points = [(float(x), float(y)) for x, y in points if x is not None and y is not None]
        if len(clean_points) <= 1:
            return 0.0
        mean_x = sum(x for x, _ in clean_points) / float(len(clean_points))
        mean_y = sum(y for _, y in clean_points) / float(len(clean_points))
        denom = sum((x - mean_x) ** 2 for x, _ in clean_points)
        if denom <= 1e-12:
            return 0.0
        return sum((x - mean_x) * (y - mean_y) for x, y in clean_points) / denom

    phase_summary_by_variant: dict[str, dict[str, Any]] = {}
    degradation_rows: list[dict[str, Any]] = []
    all_variant_names = sorted({str(row.get("variant_name", "")) for row in rows if str(row.get("variant_name", ""))})
    for variant_name in all_variant_names:
        selected_warmup_rows = [
            row for row in warmup_rows if str(row.get("variant_name", "")) == variant_name
        ]
        selected_transfer_rows = [
            row for row in transfer_rows if str(row.get("variant_name", "")) == variant_name
        ]
        warmup_summary = _phase_summary(selected_warmup_rows)
        transfer_summary = _phase_summary(selected_transfer_rows)
        intent_loss_delta = (
            transfer_summary["intent_loss_normalized"] - warmup_summary["intent_loss_normalized"]
        )
        safety_delta = (
            transfer_summary["safety_violation_rate"] - warmup_summary["safety_violation_rate"]
        )
        safety_first_delta = (
            transfer_summary["safety_first_score"] - warmup_summary["safety_first_score"]
        )
        hard_target_drop = (
            warmup_summary["hard_target_fulfillment"] - transfer_summary["hard_target_fulfillment"]
        )
        degradation_score = (
            max(0.0, safety_first_delta)
            + 0.30 * max(0.0, safety_delta)
            + 0.25 * max(0.0, hard_target_drop)
            + 0.15 * max(0.0, intent_loss_delta)
        )
        adaptation_score = (
            transfer_summary["safety_first_score"]
            + 0.35 * max(0.0, safety_first_delta)
            + 0.20 * max(0.0, safety_delta)
            + 0.15 * max(0.0, hard_target_drop)
            + 0.10 * max(0.0, intent_loss_delta)
        )
        degradation_row: dict[str, Any] = {
            "variant_name": variant_name,
            "warmup_case_count": len(selected_warmup_rows),
            "transfer_case_count": len(selected_transfer_rows),
            "warmup_intent_loss_normalized": warmup_summary["intent_loss_normalized"],
            "transfer_intent_loss_normalized": transfer_summary["intent_loss_normalized"],
            "degradation_intent_loss_delta": intent_loss_delta,
            "warmup_safety_violation_rate": warmup_summary["safety_violation_rate"],
            "transfer_safety_violation_rate": transfer_summary["safety_violation_rate"],
            "degradation_safety_violation_delta": safety_delta,
            "warmup_hard_target_fulfillment": warmup_summary["hard_target_fulfillment"],
            "transfer_hard_target_fulfillment": transfer_summary["hard_target_fulfillment"],
            "degradation_hard_target_fulfillment_drop": hard_target_drop,
            "warmup_safety_first_score": warmup_summary["safety_first_score"],
            "transfer_safety_first_score": transfer_summary["safety_first_score"],
            "degradation_safety_first_delta": safety_first_delta,
            "cross_scenario_degradation_score": degradation_score,
            "cross_scenario_adaptation_score": adaptation_score,
        }
        phase_summary_by_variant[variant_name] = degradation_row
        degradation_rows.append(degradation_row)

    high_risk_rows_with_bucket: list[dict[str, str]] = []
    for row in transfer_rows:
        bucket = _risk_bucket(row)
        if bucket == "":
            continue
        tagged = dict(row)
        tagged["risk_bucket"] = bucket
        high_risk_rows_with_bucket.append(tagged)

    high_risk_summary_by_variant: dict[str, dict[str, Any]] = {}
    high_risk_summary_rows: list[dict[str, Any]] = []
    by_risk_variant: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in high_risk_rows_with_bucket:
        by_risk_variant.setdefault(
            (str(row.get("risk_bucket", "")), str(row.get("variant_name", ""))),
            [],
        ).append(row)
    for (risk_bucket, variant_name), selected_rows in sorted(by_risk_variant.items()):
        phase_summary = _phase_summary(selected_rows)
        out = {
            "risk_bucket": risk_bucket,
            "variant_name": variant_name,
            "case_count": len(selected_rows),
            "intent_loss_normalized": phase_summary["intent_loss_normalized"],
            "hard_target_fulfillment": phase_summary["hard_target_fulfillment"],
            "safety_violation_rate": phase_summary["safety_violation_rate"],
            "safety_first_score": phase_summary["safety_first_score"],
            "throughput_mbps": phase_summary["throughput_mbps"],
            "safe_baseline_fallback_rate": phase_summary["safe_baseline_fallback_rate"],
            "forced_non_rt_reselection_count": phase_summary["forced_non_rt_reselection_count"],
        }
        high_risk_summary_rows.append(out)
    for variant_name in all_variant_names:
        selected_rows = [
            row for row in high_risk_rows_with_bucket if str(row.get("variant_name", "")) == variant_name
        ]
        phase_summary = _phase_summary(selected_rows)
        high_risk_summary_by_variant[variant_name] = {
            "transfer_high_risk_case_count": int(len(selected_rows)),
            "transfer_high_risk_intent_loss_normalized": phase_summary["intent_loss_normalized"],
            "transfer_high_risk_hard_target_fulfillment": phase_summary["hard_target_fulfillment"],
            "transfer_high_risk_safety_violation_rate": phase_summary["safety_violation_rate"],
            "transfer_high_risk_safety_first_score": phase_summary["safety_first_score"],
        }

    repeat_summary_rows: list[dict[str, Any]] = []
    recovery_summary_by_variant: dict[str, dict[str, Any]] = {}
    by_repeat_variant: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in transfer_rows:
        try:
            repeat_id = int(float(row.get("phase_repeat", 1) or 1))
        except ValueError:
            repeat_id = 1
        by_repeat_variant.setdefault(
            (str(row.get("variant_name", "")), repeat_id),
            [],
        ).append(row)
    for variant_name in all_variant_names:
        selected_repeat_ids = sorted(
            repeat_id for (candidate_variant, repeat_id) in by_repeat_variant if candidate_variant == variant_name
        )
        if len(selected_repeat_ids) == 0:
            continue
        first_summary = _phase_summary(by_repeat_variant[(variant_name, selected_repeat_ids[0])])
        first_score = first_summary["safety_first_score"]
        score_points: list[tuple[float, float]] = []
        final_summary = first_summary
        for repeat_id in selected_repeat_ids:
            phase_summary = _phase_summary(by_repeat_variant[(variant_name, repeat_id)])
            final_summary = phase_summary
            score_points.append((float(repeat_id), phase_summary["safety_first_score"]))
            repeat_summary_rows.append(
                {
                    "variant_name": variant_name,
                    "phase_repeat": int(repeat_id),
                    "case_count": len(by_repeat_variant[(variant_name, repeat_id)]),
                    "intent_loss_normalized": phase_summary["intent_loss_normalized"],
                    "hard_target_fulfillment": phase_summary["hard_target_fulfillment"],
                    "safety_violation_rate": phase_summary["safety_violation_rate"],
                    "safety_first_score": phase_summary["safety_first_score"],
                    "safe_baseline_fallback_rate": phase_summary["safe_baseline_fallback_rate"],
                    "forced_non_rt_reselection_count": phase_summary["forced_non_rt_reselection_count"],
                    "safety_first_gain_vs_first_repeat": first_score
                    - phase_summary["safety_first_score"],
                    "intent_loss_gain_vs_first_repeat": first_summary["intent_loss_normalized"]
                    - phase_summary["intent_loss_normalized"],
                    "safety_violation_reduction_vs_first_repeat": first_summary["safety_violation_rate"]
                    - phase_summary["safety_violation_rate"],
                }
            )
        recovery_summary_by_variant[variant_name] = {
            "transfer_recovery_repeat_count": int(len(selected_repeat_ids)),
            "transfer_recovery_gain_final_vs_first": first_score - final_summary["safety_first_score"],
            "transfer_recovery_intent_loss_gain_final_vs_first": first_summary["intent_loss_normalized"]
            - final_summary["intent_loss_normalized"],
            "transfer_recovery_safety_violation_reduction_final_vs_first": first_summary[
                "safety_violation_rate"
            ]
            - final_summary["safety_violation_rate"],
            "transfer_recovery_safety_first_slope": _linear_slope(score_points),
        }

    summary_rows: list[dict[str, Any]] = []
    for variant_name, variant_rows in sorted(by_variant.items()):
        out: dict[str, Any] = {
            "variant_name": variant_name,
            "case_count": len(variant_rows),
            "intent_count": len({row.get("intent_slug", "") for row in variant_rows}),
        }
        for metric_name, field_name in metrics:
            out[f"transfer_mean_{metric_name}"] = _mean(variant_rows, field_name)
        _add_transfer_scores(out)
        case_scores = [_case_safety_first_score(row) for row in variant_rows]
        out["transfer_worst_case_safety_first_score"] = max(case_scores) if case_scores else 0.0
        out["transfer_p90_case_safety_first_score"] = _tail_mean(case_scores, fraction=0.10)
        out["transfer_std_safety_first_score"] = _std(case_scores)
        out["transfer_tail_robustness_score"] = (
            float(out.get("transfer_safety_first_score", 0.0) or 0.0)
            + 0.25 * float(out.get("transfer_worst_case_safety_first_score", 0.0) or 0.0)
            + 0.10 * float(out.get("transfer_std_safety_first_score", 0.0) or 0.0)
        )
        memory_sensitive_rows = [row for row in variant_rows if _is_memory_sensitive_row(row)]
        out["transfer_memory_sensitive_case_count"] = int(len(memory_sensitive_rows))
        out["transfer_memory_sensitive_mean_intent_loss_normalized"] = _mean(
            memory_sensitive_rows,
            "summary_experimental_metrics_intent_loss_mean_normalized",
        )
        out["transfer_memory_sensitive_mean_safety_violation_rate"] = _mean(
            memory_sensitive_rows,
            "summary_experimental_metrics_safety_violation_rate",
        )
        memory_sensitive_scores = [_case_safety_first_score(row) for row in memory_sensitive_rows]
        out["transfer_memory_sensitive_tail_robustness_score"] = (
            _tail_mean(memory_sensitive_scores, fraction=0.20)
            if len(memory_sensitive_scores) > 0
            else 0.0
        )
        out.update(phase_summary_by_variant.get(variant_name, {}))
        out.update(high_risk_summary_by_variant.get(variant_name, {}))
        out.update(recovery_summary_by_variant.get(variant_name, {}))
        summary_rows.append(out)

    no_memory_row = next(
        (row for row in summary_rows if str(row.get("variant_name", "")) == "no_memory_planner"),
        None,
    )
    if no_memory_row is not None:
        no_memory_score = float(no_memory_row.get("transfer_safety_first_score", 0.0) or 0.0)
        no_memory_safety = float(no_memory_row.get("transfer_mean_safety_violation_rate", 0.0) or 0.0)
        no_memory_loss = float(no_memory_row.get("transfer_mean_intent_loss_normalized", 0.0) or 0.0)
        for row in summary_rows:
            row["transfer_safety_first_gain_vs_no_memory"] = (
                no_memory_score - float(row.get("transfer_safety_first_score", 0.0) or 0.0)
            )
            row["transfer_safety_violation_reduction_vs_no_memory"] = (
                no_memory_safety - float(row.get("transfer_mean_safety_violation_rate", 0.0) or 0.0)
            )
            row["transfer_intent_loss_reduction_vs_no_memory"] = (
                no_memory_loss - float(row.get("transfer_mean_intent_loss_normalized", 0.0) or 0.0)
            )
        no_memory_degradation = float(
            no_memory_row.get("cross_scenario_degradation_score", 0.0) or 0.0
        )
        no_memory_adaptation = float(
            no_memory_row.get("cross_scenario_adaptation_score", 0.0) or 0.0
        )
        no_memory_high_risk_score = float(
            no_memory_row.get("transfer_high_risk_safety_first_score", 0.0) or 0.0
        )
        for row in summary_rows:
            row["transfer_degradation_gain_vs_no_memory"] = no_memory_degradation - float(
                row.get("cross_scenario_degradation_score", 0.0) or 0.0
            )
            row["transfer_adaptation_gain_vs_no_memory"] = no_memory_adaptation - float(
                row.get("cross_scenario_adaptation_score", 0.0) or 0.0
            )
            row["transfer_high_risk_gain_vs_no_memory"] = no_memory_high_risk_score - float(
                row.get("transfer_high_risk_safety_first_score", 0.0) or 0.0
            )

    _write_rows(root_output_dir / "warmup_to_transfer_degradation.csv", degradation_rows)
    _write_rows(root_output_dir / "transfer_high_risk_results.csv", high_risk_summary_rows)
    _write_rows(root_output_dir / "transfer_repeat_recovery.csv", repeat_summary_rows)

    field_names: list[str] = []
    for row in summary_rows:
        for key in row.keys():
            if key not in field_names:
                field_names.append(key)
    with (root_output_dir / "transfer_headline_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(summary_rows)

    metric_rank_specs = [
        ("intent", "transfer_mean_intent_loss_normalized", "lower"),
        ("intent", "transfer_mean_intent_loss_raw", "lower"),
        ("safety", "transfer_mean_safety_violation_rate", "lower"),
        ("safety", "transfer_safety_success_rate", "higher"),
        ("safety", "transfer_mean_mean_bler", "lower"),
        ("near_rt_recovery", "transfer_mean_health_gate_trigger_rate", "lower"),
        ("near_rt_recovery", "transfer_mean_safe_baseline_fallback_rate", "lower"),
        ("near_rt_recovery", "transfer_mean_forced_non_rt_reselection_count", "lower"),
        ("ran_kpi", "transfer_mean_throughput_mbps", "higher"),
        ("ran_kpi", "transfer_mean_final_throughput_mbps", "higher"),
        ("stability", "transfer_mean_switching_cost", "lower"),
        ("memory", "transfer_mean_success_memory_count", "higher"),
        ("memory", "transfer_mean_failure_memory_count", "higher"),
        ("memory", "transfer_failure_memory_density", "higher"),
        ("memory_sensitive", "transfer_memory_sensitive_mean_intent_loss_normalized", "lower"),
        ("memory_sensitive", "transfer_memory_sensitive_mean_safety_violation_rate", "lower"),
        ("memory_sensitive", "transfer_memory_sensitive_tail_robustness_score", "lower"),
        ("high_risk", "transfer_high_risk_intent_loss_normalized", "lower"),
        ("high_risk", "transfer_high_risk_safety_violation_rate", "lower"),
        ("high_risk", "transfer_high_risk_safety_first_score", "lower"),
        ("cross_scenario_degradation", "degradation_intent_loss_delta", "lower"),
        ("cross_scenario_degradation", "degradation_safety_violation_delta", "lower"),
        ("cross_scenario_degradation", "degradation_safety_first_delta", "lower"),
        ("cross_scenario_degradation", "cross_scenario_degradation_score", "lower"),
        ("cross_scenario_degradation", "cross_scenario_adaptation_score", "lower"),
        ("transfer_recovery", "transfer_recovery_gain_final_vs_first", "higher"),
        ("transfer_recovery", "transfer_recovery_safety_violation_reduction_final_vs_first", "higher"),
        ("transfer_recovery", "transfer_recovery_safety_first_slope", "lower"),
        ("summary_optional", "transfer_safety_first_score", "lower"),
        ("summary_optional", "transfer_p90_case_safety_first_score", "lower"),
        ("summary_optional", "transfer_tail_robustness_score", "lower"),
    ]
    ranking_rows: list[dict[str, Any]] = []
    for metric_group, metric_name, direction in metric_rank_specs:
        present_rows = [
            row
            for row in summary_rows
            if metric_name in row and str(row.get(metric_name, "")).strip() != ""
        ]
        reverse = direction == "higher"
        sorted_rows = sorted(
            present_rows,
            key=lambda row: float(row.get(metric_name, 0.0) or 0.0),
            reverse=reverse,
        )
        for rank_index, row in enumerate(sorted_rows, start=1):
            ranking_rows.append(
                {
                    "metric_group": metric_group,
                    "metric_name": metric_name,
                    "direction": direction,
                    "rank": rank_index,
                    "variant_name": row.get("variant_name", ""),
                    "value": float(row.get(metric_name, 0.0) or 0.0),
                }
            )
    with (root_output_dir / "transfer_metric_rankings.csv").open("w", newline="", encoding="utf-8") as csv_file:
        field_names = ["metric_group", "metric_name", "direction", "rank", "variant_name", "value"]
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(ranking_rows)

    intent_rows: list[dict[str, Any]] = []
    for (intent_slug, variant_name), selected_rows in sorted(by_intent_variant.items()):
        out = {
            "intent_slug": intent_slug,
            "variant_name": variant_name,
            "case_count": len(selected_rows),
        }
        for metric_name, field_name in metrics:
            out[f"transfer_mean_{metric_name}"] = _mean(selected_rows, field_name)
        _add_transfer_scores(out)
        case_scores = [_case_safety_first_score(row) for row in selected_rows]
        out["transfer_worst_case_safety_first_score"] = max(case_scores) if case_scores else 0.0
        out["transfer_std_safety_first_score"] = _std(case_scores)
        out["transfer_tail_robustness_score"] = (
            float(out.get("transfer_safety_first_score", 0.0) or 0.0)
            + 0.25 * float(out.get("transfer_worst_case_safety_first_score", 0.0) or 0.0)
            + 0.10 * float(out.get("transfer_std_safety_first_score", 0.0) or 0.0)
        )
        intent_rows.append(out)
    field_names = []
    for row in intent_rows:
        for key in row.keys():
            if key not in field_names:
                field_names.append(key)
    with (root_output_dir / "transfer_intent_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(intent_rows)

    scenario_rows: list[dict[str, Any]] = []
    for (scenario_id, variant_name), selected_rows in sorted(by_scenario_variant.items()):
        out = {
            "scenario_id": scenario_id,
            "variant_name": variant_name,
            "case_count": len(selected_rows),
        }
        for metric_name, field_name in metrics:
            out[f"transfer_mean_{metric_name}"] = _mean(selected_rows, field_name)
        _add_transfer_scores(out)
        case_scores = [_case_safety_first_score(row) for row in selected_rows]
        out["transfer_worst_case_safety_first_score"] = max(case_scores) if case_scores else 0.0
        out["transfer_p90_case_safety_first_score"] = _tail_mean(case_scores, fraction=0.20)
        out["transfer_std_safety_first_score"] = _std(case_scores)
        scenario_rows.append(out)
    field_names = []
    for row in scenario_rows:
        for key in row.keys():
            if key not in field_names:
                field_names.append(key)
    with (root_output_dir / "transfer_scenario_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(scenario_rows)

    family_rows: list[dict[str, Any]] = []
    for (intent_family, variant_name), selected_rows in sorted(by_family_variant.items()):
        out = {
            "intent_family": intent_family,
            "variant_name": variant_name,
            "case_count": len(selected_rows),
        }
        for metric_name, field_name in metrics:
            out[f"transfer_mean_{metric_name}"] = _mean(selected_rows, field_name)
        _add_transfer_scores(out)
        case_scores = [_case_safety_first_score(row) for row in selected_rows]
        out["transfer_worst_case_safety_first_score"] = max(case_scores) if case_scores else 0.0
        out["transfer_p90_case_safety_first_score"] = _tail_mean(case_scores, fraction=0.20)
        out["transfer_std_safety_first_score"] = _std(case_scores)
        family_rows.append(out)
    field_names = []
    for row in family_rows:
        for key in row.keys():
            if key not in field_names:
                field_names.append(key)
    with (root_output_dir / "transfer_intent_family_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(family_rows)


def main() -> None:
    args = _parse_args()
    DebugConfig.DEBUG = False
    DebugConfig.ORCHESTRATION_SUMMARY = False
    DebugConfig.CONCISE_SLOT_METRICS = False
    config_path = (project_root / str(args.config)).resolve()
    experiment_config = load_yaml_file(str(config_path))
    manifest_path = (project_root / str(experiment_config["manifest_path"])).resolve()
    manifest_rows = _load_manifest(manifest_path)
    default_matrix_name = str(
        experiment_config.get("matrix_defaults", {}).get("default_matrix_name", "oran_paper_eval_suite")
    )
    resolved_rows = _resolve_manifest_cases(
        manifest_rows=manifest_rows,
        default_matrix_name=default_matrix_name,
    )
    warmup_rows = _phase_rows(
        resolved_rows=resolved_rows,
        phase="warmup",
        selected_scenario_ids=list(experiment_config.get("warmup_scenario_ids", []) or []),
    )
    transfer_rows = _phase_rows(
        resolved_rows=resolved_rows,
        phase="transfer",
        selected_scenario_ids=list(experiment_config.get("transfer_scenario_ids", []) or []),
    )
    if len(warmup_rows) == 0 or len(transfer_rows) == 0:
        raise ValueError("Experiment 1 transfer config must define non-empty warmup and transfer scenario ids.")
    memory_transfer_config = dict(experiment_config.get("memory_transfer", {}) or {})
    warmup_repeats = max(1, int(memory_transfer_config.get("warmup_repeats", 1) or 1))
    transfer_repeats = max(1, int(memory_transfer_config.get("transfer_repeats", 1) or 1))
    phase_batches = (
        ("warmup", warmup_rows, warmup_repeats),
        ("transfer", transfer_rows, transfer_repeats),
    )
    global_sim_overrides = dict(experiment_config.get("global_sim_overrides", {}) or {})
    global_ric_overrides = dict(experiment_config.get("global_ric_overrides", {}) or {})

    base_configs, phy_config = _load_intent_orchestration_configs()
    if not bool(args.postprocess_only):
        _assert_preflight_requirements(
            experiment_config=experiment_config,
            base_configs=base_configs,
            dry_run=bool(args.dry_run),
        )

    available_variants = build_minimal_paper_variants()
    variant_names = [str(item) for item in list(experiment_config.get("variants", []) or [])]
    selected_variants = [available_variants[name] for name in variant_names]
    seeds = [int(item) for item in list(experiment_config.get("seeds", []) or [1])]
    intent_suite = _intent_suite_from_config(experiment_config)
    intent_subset = {
        str(item)
        for item in list(experiment_config.get("intent_subset", []) or [])
        if str(item).strip() != ""
    }
    if len(intent_subset) > 0:
        intent_suite = [
            item
            for item in intent_suite
            if str(item.get("intent_slug", "")) in intent_subset
        ]

    experiment_name = str(experiment_config.get("experiment_name", "experiment1_cross_scenario_transfer"))
    root_output_dir = (
        project_root
        / str(experiment_config.get("base_output_dir", "outputs/paper_eval"))
        / experiment_name
    ).resolve()
    root_output_dir.mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(root_output_dir, experiment_config)
    _save_manifest_copy(root_output_dir, manifest_path)
    _save_run_plan(
        output_dir=root_output_dir,
        resolved_rows=warmup_rows + transfer_rows,
        variants=selected_variants,
        seeds=seeds,
        intent_suite=intent_suite,
    )

    plan_rows: list[dict[str, Any]] = []
    for intent_spec in intent_suite:
        for seed in seeds:
            for variant in selected_variants:
                for phase, phase_items, phase_repeats in phase_batches:
                    for repeat_index in range(1, int(phase_repeats) + 1):
                        for item in phase_items:
                            scenario_id = str(item["manifest"].get("scenario_id", item["case"].case_id))
                            plan_rows.append(
                                {
                                    "phase": phase,
                                    "phase_repeat": int(repeat_index),
                                    "intent_slug": str(intent_spec.get("intent_slug", "intent")),
                                    "seed": int(seed),
                                    "variant_name": variant.name,
                                    "scenario_id": scenario_id,
                                }
                            )
    _write_transfer_manifest(root_output_dir, plan_rows)

    if args.dry_run:
        print(
            f"[dry-run] Planned {len(plan_rows)} warm-up/transfer case runs under {root_output_dir}. "
            f"Headline metrics will use transfer rows only."
        )
        return

    if args.postprocess_only:
        _aggregate_transfer_headline(root_output_dir)
        postprocess_generalization_artifacts(
            experiment_dir=root_output_dir,
            manifest_path=root_output_dir / "scenario_manifest.csv",
            reference_scenario_id=str(experiment_config.get("reference_scenario_id", "")),
            variant_names=variant_names,
        )
        print(f"Post-processing refreshed under {root_output_dir}.")
        return

    case_records: list[dict[str, Any]] = []
    xapp_pool_profiles = dict(experiment_config.get("xapp_pool_profiles", {}) or {})
    force_sim_overrides = dict(experiment_config.get("force_sim_overrides", {}) or {})
    max_runs = max(0, int(args.max_runs or 0))
    completed_runs = 0
    os.environ.setdefault("ORAN_WRITE_SLOT_SNAPSHOTS", "0")

    for intent_spec in intent_suite:
        intent_slug = str(intent_spec.get("intent_slug", "intent") or "intent")
        intent_text = str(intent_spec.get("intent_text", "") or "")
        for seed in seeds:
            for variant in selected_variants:
                for phase, phase_items, phase_repeats in phase_batches:
                    for repeat_index in range(1, int(phase_repeats) + 1):
                        for item in phase_items:
                            if max_runs > 0 and completed_runs >= max_runs:
                                _write_csv(root_output_dir / "transfer_case_results.csv", case_records)
                                _aggregate_transfer_headline(root_output_dir)
                                print(f"Stopped after max_runs={max_runs}. Partial outputs are under {root_output_dir}.")
                                return

                            manifest = dict(item["manifest"])
                            scenario_id = str(manifest.get("scenario_id", item["case"].case_id))
                            xapp_pool_class = str(manifest.get("xapp_pool_class", "") or "")
                            phase_label = (
                                str(phase)
                                if int(phase_repeats) == 1
                                else f"{phase}r{int(repeat_index)}"
                            )
                            case_output_dir = _case_output_dir(
                                root_output_dir=root_output_dir,
                                phase_label=phase_label,
                                scenario_id=scenario_id,
                                intent_slug=intent_slug,
                                variant_name=variant.name,
                                seed=int(seed),
                            )
                            final_summary_path = case_output_dir / "final_summary.json"
                            print(
                                f"[exp1-transfer] {completed_runs + 1}/{len(plan_rows)} phase={phase} "
                                f"repeat={int(repeat_index)} intent={intent_slug} scenario={scenario_id} "
                                f"variant={variant.name} seed={int(seed)}"
                            )

                            if bool(args.skip_completed) and _case_json_is_valid(final_summary_path):
                                with final_summary_path.open("r", encoding="utf-8") as summary_file:
                                    final_summary = json.load(summary_file)
                            else:
                                run_configs = _prepare_configs_for_run(
                                    base_configs=base_configs,
                                    variant=variant,
                                    seed=int(seed),
                                    intent_text=intent_text,
                                    xapp_pool_profile_name=xapp_pool_class,
                                    xapp_pool_profiles=xapp_pool_profiles,
                                    intent_spec=intent_spec,
                                    global_sim_overrides=global_sim_overrides,
                                    global_ric_overrides=global_ric_overrides,
                                )
                                run_configs["ric"] = _apply_transfer_memory_overrides(
                                    ric_config=run_configs["ric"],
                                    variant_name=variant.name,
                                    intent_slug=intent_slug,
                                    seed=int(seed),
                                    root_output_dir=root_output_dir,
                                    transfer_config=experiment_config,
                                )
                                run_configs["ric"] = _apply_literature_baseline_state_overrides(
                                    ric_config=run_configs["ric"],
                                    variant_name=variant.name,
                                    intent_slug=intent_slug,
                                    seed=int(seed),
                                    root_output_dir=root_output_dir,
                                    phase=phase,
                                    repeat_index=int(repeat_index),
                                )
                                case_output_dir.mkdir(parents=True, exist_ok=True)
                                runtime_case = _with_forced_case_sim_overrides(
                                    case=item["case"],
                                    force_sim_overrides=force_sim_overrides,
                                )
                                final_summary = run_matrix_case(
                                    base_configs=run_configs,
                                    phy_config=phy_config,
                                    case=runtime_case,
                                    output_dir=str(case_output_dir),
                                    active_intent_text=intent_text,
                                )

                            record = _build_case_record(
                                resolved_row=item,
                                final_summary=final_summary,
                                output_dir=case_output_dir,
                                variant=variant,
                                seed=int(seed),
                                intent_spec=intent_spec,
                            )
                            record["transfer_phase"] = phase
                            record["phase_repeat"] = int(repeat_index)
                            record["headline_split"] = (
                                phase == str(experiment_config.get("memory_transfer", {}).get("headline_split", "transfer"))
                            )
                            case_records.append(record)
                            completed_runs += 1

    _write_csv(root_output_dir / "transfer_case_results.csv", case_records)
    shutil.copy2(root_output_dir / "transfer_case_results.csv", root_output_dir / "case_results.csv")
    _aggregate_transfer_headline(root_output_dir)
    postprocess_generalization_artifacts(
        experiment_dir=root_output_dir,
        manifest_path=root_output_dir / "scenario_manifest.csv",
        reference_scenario_id=str(experiment_config.get("reference_scenario_id", "")),
        variant_names=variant_names,
    )
    print(f"Completed Experiment 1 warm-up/transfer benchmark under {root_output_dir}.")


if __name__ == "__main__":
    main()
