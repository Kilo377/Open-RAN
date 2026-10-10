"""Config-driven runner for the cross-scenario generalization benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from core.config_loader import dump_yaml_file, load_all_configs, load_yaml_file
from core.debug import DebugConfig
from experiments.experiment_variants import ExperimentVariant, build_minimal_paper_variants
from experiments.postprocess_generalization import postprocess_generalization_artifacts
from experiments.scenario_scan_utils import build_case_summary_record, run_matrix_case
from ric.near_rt.xapp_variant_catalog import canonical_xapp_category
from scenarios.scenario_matrix_registry import ScenarioMatrixRegistry
from scenarios.scenario_registry import deep_merge_dict


def _slugify(text: str, *, fallback: str = "intent") -> str:
    normalized = str(text or "").strip().lower()
    chars: list[str] = []
    previous_dash = False
    for char in normalized:
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    slug = "".join(chars).strip("-")
    return slug if slug != "" else fallback


def _intent_suite_from_config(experiment_config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_suite = experiment_config.get("intent_suite")
    if isinstance(raw_suite, list) and len(raw_suite) > 0:
        intents: list[dict[str, Any]] = []
        for index, item in enumerate(raw_suite, start=1):
            if not isinstance(item, dict):
                continue
            intent_text = str(item.get("intent_text", "") or "").strip()
            if intent_text == "":
                continue
            intent_slug = str(
                item.get("intent_slug", "")
                or item.get("intent_id", "")
                or _slugify(intent_text, fallback=f"intent-{index}")
            ).strip()
            intents.append(
                {
                    "intent_slug": _slugify(intent_slug, fallback=f"intent-{index}"),
                    "intent_id": str(item.get("intent_id", intent_slug) or intent_slug),
                    "intent_family": str(item.get("intent_family", "mixed") or "mixed"),
                    "intent_text": intent_text,
                    "macro_weight": float(item.get("macro_weight", 1.0) or 1.0),
                    "notes": str(item.get("notes", "") or ""),
                    "hard_kpi_targets": [
                        dict(target)
                        for target in list(item.get("hard_kpi_targets", []) or [])
                        if isinstance(target, dict)
                    ],
                }
            )
        if len(intents) > 0:
            return intents

    intent_text = str(experiment_config.get("intent_text", "") or "").strip()
    if intent_text == "":
        raise ValueError("Experiment config must define either intent_text or a non-empty intent_suite.")
    return [
        {
            "intent_slug": str(experiment_config.get("intent_slug", "edge-throughput-latency") or "edge-throughput-latency"),
            "intent_id": str(experiment_config.get("intent_id", "intent:edge_throughput_latency") or "intent:edge_throughput_latency"),
            "intent_family": str(experiment_config.get("intent_family", "edge_throughput_latency") or "edge_throughput_latency"),
            "intent_text": intent_text,
            "macro_weight": 1.0,
            "notes": "Legacy single-intent configuration.",
            "hard_kpi_targets": [
                dict(target)
                for target in list(experiment_config.get("hard_kpi_targets", []) or [])
                if isinstance(target, dict)
            ],
        }
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cross-scenario generalization benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/cross_scenario_generalization.yaml",
        help="Experiment configuration YAML path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, manifest, case resolution, and output layout without executing simulations.",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip simulation and only regenerate aggregated outputs from existing case results.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent case workers to run in parallel. Use 1 for serial execution.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse case directories that already contain final_summary.json.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional limit on the number of planned case runs, useful for smoke testing.",
    )
    return parser.parse_args()


def _load_intent_orchestration_configs() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    configs = load_all_configs(str(project_root / "configs"))
    configs["phy"] = load_yaml_file(str(project_root / "configs" / "phy.yaml"))
    for override_name in ("scenario", "sim", "cell", "ue", "beamforming", "ric", "phy"):
        override_path = project_root / "configs" / f"{override_name}_intent_orchestration.yaml"
        if not override_path.exists():
            continue
        configs[override_name] = deep_merge_dict(
            dict(configs.get(override_name, {}) or {}),
            load_yaml_file(str(override_path)),
        )
    phy_config = dict(configs.get("phy", {}) or {})
    return configs, phy_config


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for row in reader:
            normalized = {
                str(key).strip(): str(value).strip()
                for key, value in dict(row).items()
            }
            if normalized.get("scenario_id", "") == "":
                continue
            rows.append(normalized)
    return rows


def _inject_intent_text(
    ric_config: dict[str, Any],
    intent_text: str,
    hard_kpi_targets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    updated_ric_config = deepcopy(dict(ric_config or {}))
    for rapp_item in list(updated_ric_config.get("rapps", []) or []):
        if str(rapp_item.get("type", "")) == "intent_goal_translator_rapp":
            rapp_item["intent_text"] = str(intent_text)
            if hard_kpi_targets is not None:
                rapp_item["hard_kpi_targets"] = [
                    dict(item)
                    for item in list(hard_kpi_targets or [])
                    if isinstance(item, dict)
                ]
    return updated_ric_config


def _xapp_allowed_by_pool(xapp_item: dict[str, Any], allowed_categories: set[str]) -> bool:
    xapp_type = str(xapp_item.get("type", "")).strip()
    xapp_name = str(xapp_item.get("name", "")).strip()
    if xapp_type == "goal_parameter_tuning_xapp" or xapp_name == "goal_parameter_tuning_xapp":
        return True
    category_name = canonical_xapp_category(xapp_name or xapp_type)
    return (
        xapp_type in allowed_categories
        or xapp_name in allowed_categories
        or category_name in allowed_categories
    )


def _apply_xapp_pool_profile(
    *,
    ric_config: dict[str, Any],
    xapp_pool_profile_name: str,
    xapp_pool_profiles: dict[str, Any],
) -> dict[str, Any]:
    updated_ric_config = deepcopy(dict(ric_config or {}))
    profile = dict(xapp_pool_profiles.get(str(xapp_pool_profile_name), {}) or {})
    allowed_categories = {
        str(item).strip()
        for item in list(profile.get("allowed_xapp_categories", []) or [])
        if str(item).strip() != ""
    }
    if len(allowed_categories) == 0:
        return updated_ric_config

    updated_ric_config["available_xapp_names"] = sorted(allowed_categories)
    updated_ric_config["xapps"] = [
        deepcopy(xapp_item)
        for xapp_item in list(updated_ric_config.get("xapps", []) or [])
        if _xapp_allowed_by_pool(xapp_item, allowed_categories)
    ]
    for rapp_item in list(updated_ric_config.get("rapps", []) or []):
        if str(rapp_item.get("type", "")) == "xapp_selection_rapp":
            rapp_item["available_xapp_names"] = sorted(allowed_categories)
    return updated_ric_config


def _prepare_configs_for_run(
    *,
    base_configs: dict[str, dict[str, Any]],
    variant: ExperimentVariant,
    seed: int,
    intent_text: str,
    xapp_pool_profile_name: str,
    xapp_pool_profiles: dict[str, Any],
    intent_spec: dict[str, Any] | None = None,
    global_sim_overrides: dict[str, Any] | None = None,
    global_ric_overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    run_configs = deepcopy(base_configs)
    global_ric_overrides = dict(global_ric_overrides or {})
    run_configs["sim"] = deep_merge_dict(
        dict(run_configs.get("sim", {}) or {}),
        {
            "rng_seed": int(seed),
            **dict(global_sim_overrides or {}),
            **variant.sim_overrides(),
        },
    )
    run_configs["ric"] = _inject_intent_text(
        deep_merge_dict(
            deep_merge_dict(
                dict(run_configs.get("ric", {}) or {}),
                global_ric_overrides,
            ),
            variant.ric_overrides(),
        ),
        intent_text,
        hard_kpi_targets=list(dict(intent_spec or {}).get("hard_kpi_targets", []) or []),
    )
    rapp_deadline_keys = {
        "llm_timeout_s",
        "llm_startup_timeout_s",
        "llm_request_retries",
        "llm_max_output_tokens",
    }
    deadline_overrides = {
        key: global_ric_overrides[key]
        for key in rapp_deadline_keys
        if key in global_ric_overrides
    }
    if deadline_overrides:
        for rapp_item in list(run_configs["ric"].get("rapps", []) or []):
            if str(rapp_item.get("type", "")).endswith("_rapp"):
                rapp_item.update(deadline_overrides)
    run_configs["ric"] = _apply_xapp_pool_profile(
        ric_config=run_configs["ric"],
        xapp_pool_profile_name=xapp_pool_profile_name,
        xapp_pool_profiles=xapp_pool_profiles,
    )
    return run_configs


def _resolve_manifest_cases(
    *,
    manifest_rows: list[dict[str, str]],
    default_matrix_name: str,
) -> list[dict[str, Any]]:
    matrix_registry = ScenarioMatrixRegistry()
    case_cache: dict[str, dict[str, Any]] = {}
    resolved_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        source_type = str(row.get("source_type", "matrix_case") or "matrix_case")
        if source_type != "matrix_case":
            raise ValueError(f"Unsupported source_type in manifest: {source_type}")
        matrix_name = str(row.get("source_matrix_name", "") or default_matrix_name)
        cache_key = str(matrix_name)
        if cache_key not in case_cache:
            case_cache[cache_key] = {
                case.case_id: case
                for case in matrix_registry.expand_matrix(matrix_name)
            }
        source_case_id = str(row.get("source_case_id", "") or row.get("scenario_id", ""))
        case = case_cache[cache_key].get(source_case_id)
        if case is None:
            raise KeyError(
                f"Scenario manifest references unknown case_id='{source_case_id}' in matrix '{matrix_name}'."
            )
        resolved_rows.append({"manifest": dict(row), "case": case, "matrix_name": matrix_name})
    return resolved_rows


def _save_config_snapshot(output_dir: Path, experiment_config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml_file(experiment_config, str(output_dir / "config_snapshot.yaml"))


def _save_manifest_copy(output_dir: Path, manifest_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, output_dir / "scenario_manifest.csv")


def _save_run_plan(
    *,
    output_dir: Path,
    resolved_rows: list[dict[str, Any]],
    variants: list[ExperimentVariant],
    seeds: list[int],
    intent_suite: list[dict[str, Any]],
) -> None:
    run_plan = {
        "intent_text": str(intent_suite[0]["intent_text"]) if len(intent_suite) == 1 else "",
        "intent_suite": list(intent_suite),
        "variant_names": [variant.name for variant in variants],
        "seeds": list(seeds),
        "scenario_ids": [str(item["manifest"]["scenario_id"]) for item in resolved_rows],
        "planned_run_count": int(len(intent_suite) * len(resolved_rows) * len(variants) * len(seeds)),
        "scenarios": [
            {
                "scenario_id": str(item["manifest"]["scenario_id"]),
                "split": str(item["manifest"].get("split", "")),
                "xapp_pool_class": str(item["manifest"].get("xapp_pool_class", "")),
                "matrix_name": str(item["matrix_name"]),
                "source_case_id": str(item["manifest"].get("source_case_id", "")),
            }
            for item in resolved_rows
        ],
    }
    with (output_dir / "run_plan.json").open("w", encoding="utf-8") as json_file:
        json.dump(run_plan, json_file, indent=2, ensure_ascii=False)


def _assert_preflight_requirements(
    *,
    experiment_config: dict[str, Any],
    base_configs: dict[str, dict[str, Any]],
    dry_run: bool,
) -> None:
    preflight = dict(experiment_config.get("preflight", {}) or {})
    if dry_run:
        return
    if not bool(preflight.get("require_api_key", True)):
        return
    ric_config = dict(base_configs.get("ric", {}) or {})
    llm_enabled = any(
        bool(rapp_item.get("llm_enabled", rapp_item.get("ollama_enabled", False)))
        for rapp_item in list(ric_config.get("rapps", []) or [])
    )
    if not llm_enabled:
        return
    env_var_name = str(preflight.get("llm_api_key_env_var", "DASHSCOPE_API_KEY"))
    if str(os.environ.get(env_var_name, "")).strip() != "":
        return
    raise RuntimeError(
        f"Missing required API key environment variable '{env_var_name}'. "
        f"Set it before running the cross-scenario benchmark, or use --dry-run to validate the setup only."
    )


def _build_case_record(
    *,
    resolved_row: dict[str, Any],
    final_summary: dict[str, Any],
    output_dir: Path,
    variant: ExperimentVariant,
    seed: int,
    intent_spec: dict[str, Any],
) -> dict[str, Any]:
    case = resolved_row["case"]
    manifest = dict(resolved_row["manifest"])
    record = build_case_summary_record(
        case=case,
        final_summary=final_summary,
        output_dir=str(output_dir),
    )
    scenario_id = str(manifest.get("scenario_id", case.case_id))
    intent_slug = str(intent_spec.get("intent_slug", "intent") or "intent")
    record["case_id"] = f"{scenario_id}__intent-{intent_slug}__{variant.name}__seed{int(seed)}"
    record["scenario_id"] = scenario_id
    record["scenario_case_id"] = scenario_id
    record["intent_slug"] = intent_slug
    record["intent_family"] = str(intent_spec.get("intent_family", "") or "")
    record["intent_text"] = str(intent_spec.get("intent_text", "") or "")
    record["intent_macro_weight"] = float(intent_spec.get("macro_weight", 1.0) or 1.0)
    record["variant_name"] = str(variant.name)
    record["seed"] = int(seed)
    record["active_intent_id"] = str(
        final_summary.get("final_policy_snapshot_summary", {}).get("active_intent_id", "") or ""
    )
    for key, value in manifest.items():
        record[str(key)] = value
    record["validator_enabled"] = bool(variant.validator_enabled)
    record["rl_guidance_enabled"] = bool(variant.rl_guidance_enabled)
    record["safety_shield_enabled"] = bool(variant.safety_shield_enabled)
    record["static_xapp_selection_enabled"] = bool(variant.static_xapp_selection_enabled)
    record["reflection_enabled"] = bool(variant.reflection_enabled)
    return record


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


def _build_case_tasks(
    *,
    resolved_rows: list[dict[str, Any]],
    variants: list[ExperimentVariant],
    seeds: list[int],
    intent_suite: list[dict[str, Any]],
    root_output_dir: Path,
    max_runs: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for intent_spec in intent_suite:
        intent_slug = str(intent_spec.get("intent_slug", "intent") or "intent")
        for seed in seeds:
            for variant in variants:
                for item in resolved_rows:
                    manifest = dict(item["manifest"])
                    scenario_id = str(manifest.get("scenario_id", item["case"].case_id))
                    xapp_pool_class = str(manifest.get("xapp_pool_class", "") or "")
                    case_output_dir = (
                        root_output_dir
                        / "cases"
                        / f"{scenario_id}__intent-{intent_slug}__{variant.name}__seed{int(seed)}"
                    )
                    tasks.append(
                        {
                            "resolved_row": item,
                            "variant": variant,
                            "seed": int(seed),
                            "intent_spec": dict(intent_spec),
                            "intent_slug": intent_slug,
                            "intent_text": str(intent_spec.get("intent_text", "") or ""),
                            "scenario_id": scenario_id,
                            "split": str(manifest.get("split", "") or ""),
                            "xapp_pool_class": xapp_pool_class,
                            "case_output_dir": str(case_output_dir),
                        }
                    )
                    if max_runs > 0 and len(tasks) >= int(max_runs):
                        return tasks
    return tasks


def _run_case_task(task: dict[str, Any]) -> dict[str, Any]:
    DebugConfig.DEBUG = False
    DebugConfig.ORCHESTRATION_SUMMARY = False
    DebugConfig.CONCISE_SLOT_METRICS = False

    case_output_dir = Path(str(task["case_output_dir"]))
    case_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_row = task["resolved_row"]
    variant = task["variant"]
    seed = int(task["seed"])
    intent_text = str(task["intent_text"])
    intent_spec = dict(task["intent_spec"])
    final_summary_path = case_output_dir / "final_summary.json"
    worker_error_path = case_output_dir / "worker_error.json"

    try:
        if bool(task.get("skip_completed", False)) and final_summary_path.exists():
            try:
                with final_summary_path.open("r", encoding="utf-8") as summary_file:
                    final_summary = json.load(summary_file)
                skipped = True
            except json.JSONDecodeError:
                final_summary_path.unlink(missing_ok=True)
                if worker_error_path.exists():
                    worker_error_path.unlink()
                skipped = False
                run_configs = _prepare_configs_for_run(
                    base_configs=task["base_configs"],
                    variant=variant,
                    seed=seed,
                    intent_text=intent_text,
                    xapp_pool_profile_name=str(task["xapp_pool_class"]),
                    xapp_pool_profiles=task["xapp_pool_profiles"],
                    intent_spec=intent_spec,
                )
                stdout_path = case_output_dir / "worker_stdout.log"
                stderr_path = case_output_dir / "worker_stderr.log"
                with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr_file:
                    with redirect_stdout(stdout_file), redirect_stderr(stderr_file):
                        final_summary = run_matrix_case(
                            base_configs=run_configs,
                            phy_config=task["phy_config"],
                            case=resolved_row["case"],
                            output_dir=str(case_output_dir),
                            active_intent_text=intent_text,
                        )
        else:
            if worker_error_path.exists():
                worker_error_path.unlink()
            run_configs = _prepare_configs_for_run(
                base_configs=task["base_configs"],
                variant=variant,
                seed=seed,
                intent_text=intent_text,
                xapp_pool_profile_name=str(task["xapp_pool_class"]),
                xapp_pool_profiles=task["xapp_pool_profiles"],
                intent_spec=intent_spec,
            )
            stdout_path = case_output_dir / "worker_stdout.log"
            stderr_path = case_output_dir / "worker_stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                with redirect_stdout(stdout_file), redirect_stderr(stderr_file):
                    final_summary = run_matrix_case(
                        base_configs=run_configs,
                        phy_config=task["phy_config"],
                        case=resolved_row["case"],
                        output_dir=str(case_output_dir),
                        active_intent_text=intent_text,
                    )
            skipped = False
        if worker_error_path.exists():
            worker_error_path.unlink()

        return {
            "ok": True,
            "skipped": skipped,
            "case_id": case_output_dir.name,
            "record": _build_case_record(
                resolved_row=resolved_row,
                final_summary=final_summary,
                output_dir=case_output_dir,
                variant=variant,
                seed=seed,
                intent_spec=intent_spec,
            ),
        }
    except Exception as exc:
        error_payload = {
            "case_id": case_output_dir.name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        with worker_error_path.open("w", encoding="utf-8") as error_file:
            json.dump(error_payload, error_file, indent=2, ensure_ascii=False)
        return {
            "ok": False,
            "case_id": case_output_dir.name,
            "error": error_payload,
        }


def _execute_case_tasks(
    *,
    tasks: list[dict[str, Any]],
    jobs: int,
) -> list[dict[str, Any]]:
    if len(tasks) == 0:
        return []

    case_records: list[dict[str, Any]] = []
    if jobs <= 1:
        for index, task in enumerate(tasks, start=1):
            print(
                f"[generalization] {index}/{len(tasks)} intent={task['intent_slug']} "
                f"scenario={task['scenario_id']} split={task['split']} "
                f"variant={task['variant'].name} seed={int(task['seed'])} "
                f"xapp_pool={task['xapp_pool_class']}"
            )
            result = _run_case_task(task)
            if not bool(result.get("ok", False)):
                raise RuntimeError(
                    f"Case failed: {result.get('case_id')} ({result.get('error', {}).get('error')})"
                )
            case_records.append(dict(result["record"]))
        return case_records

    with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
        future_to_task = {executor.submit(_run_case_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            result = future.result()
            status = "skipped" if bool(result.get("skipped", False)) else "done"
            if not bool(result.get("ok", False)):
                error = dict(result.get("error", {}) or {})
                raise RuntimeError(
                    f"Case failed: {result.get('case_id')} ({error.get('error_type')}: {error.get('error')}). "
                    f"See {Path(str(task['case_output_dir'])) / 'worker_error.json'}"
                )
            case_records.append(dict(result["record"]))
            print(
                f"[generalization] {completed}/{len(tasks)} {status} intent={task['intent_slug']} "
                f"scenario={task['scenario_id']} variant={task['variant'].name} seed={int(task['seed'])}"
            )
    return case_records


def main() -> None:
    args = _parse_args()
    DebugConfig.DEBUG = False
    DebugConfig.ORCHESTRATION_SUMMARY = False
    DebugConfig.CONCISE_SLOT_METRICS = False

    experiment_config_path = (project_root / str(args.config)).resolve()
    experiment_config = load_yaml_file(str(experiment_config_path))
    if len(experiment_config) == 0:
        raise FileNotFoundError(f"Experiment config not found or empty: {experiment_config_path}")

    manifest_path = (project_root / str(experiment_config["manifest_path"])).resolve()
    manifest_rows = _load_manifest(manifest_path)
    if len(manifest_rows) == 0:
        raise ValueError(f"Scenario manifest is empty: {manifest_path}")

    available_variants = build_minimal_paper_variants()
    variant_names = [str(item) for item in list(experiment_config.get("variants", []) or [])]
    selected_variants = [available_variants[name] for name in variant_names]
    seeds = [int(item) for item in list(experiment_config.get("seeds", []) or [1])]
    intent_suite = _intent_suite_from_config(experiment_config)
    reference_scenario_id = str(
        experiment_config.get("reference_scenario_id", "benchmark_nominal_cluster") or "benchmark_nominal_cluster"
    )
    default_matrix_name = str(
        experiment_config.get("matrix_defaults", {}).get("default_matrix_name", "orchestration_benchmark_suite")
    )

    resolved_rows = _resolve_manifest_cases(
        manifest_rows=manifest_rows,
        default_matrix_name=default_matrix_name,
    )
    base_configs, phy_config = _load_intent_orchestration_configs()
    _assert_preflight_requirements(
        experiment_config=experiment_config,
        base_configs=base_configs,
        dry_run=bool(args.dry_run),
    )

    experiment_name = str(experiment_config.get("experiment_name", "cross_scenario_generalization"))
    root_output_dir = (project_root / str(experiment_config.get("base_output_dir", "outputs/generalization")) / experiment_name).resolve()
    _save_config_snapshot(root_output_dir, experiment_config)
    _save_manifest_copy(root_output_dir, manifest_path)
    _save_run_plan(
        output_dir=root_output_dir,
        resolved_rows=resolved_rows,
        variants=selected_variants,
        seeds=seeds,
        intent_suite=intent_suite,
    )

    if args.postprocess_only:
        postprocess_generalization_artifacts(
            experiment_dir=root_output_dir,
            manifest_path=root_output_dir / "scenario_manifest.csv",
            reference_scenario_id=reference_scenario_id,
            variant_names=variant_names,
        )
        print(f"Post-processing refreshed under {root_output_dir}.")
        return

    if args.dry_run:
        print(
            f"[dry-run] Planned {len(resolved_rows) * len(selected_variants) * len(seeds)} runs "
            f"per intent, {len(intent_suite) * len(resolved_rows) * len(selected_variants) * len(seeds)} total runs "
            f"for experiment '{experiment_name}'. Outputs will be written under {root_output_dir}."
        )
        return

    xapp_pool_profiles = dict(experiment_config.get("xapp_pool_profiles", {}) or {})
    jobs = max(1, int(args.jobs or 1))
    tasks = _build_case_tasks(
        resolved_rows=resolved_rows,
        variants=selected_variants,
        seeds=seeds,
        intent_suite=intent_suite,
        root_output_dir=root_output_dir,
        max_runs=max(0, int(args.max_runs or 0)),
    )
    for task in tasks:
        task["base_configs"] = base_configs
        task["phy_config"] = phy_config
        task["xapp_pool_profiles"] = xapp_pool_profiles
        task["skip_completed"] = bool(args.skip_completed)

    print(
        f"[generalization] Executing {len(tasks)} case runs with jobs={jobs} "
        f"skip_completed={bool(args.skip_completed)}."
    )
    case_records = _execute_case_tasks(tasks=tasks, jobs=jobs)

    _write_csv(root_output_dir / "case_results.csv", case_records)
    postprocess_generalization_artifacts(
        experiment_dir=root_output_dir,
        manifest_path=root_output_dir / "scenario_manifest.csv",
        reference_scenario_id=reference_scenario_id,
        variant_names=variant_names,
    )
    print(
        f"Completed cross-scenario generalization benchmark. "
        f"Aggregated results are under {root_output_dir}."
    )


if __name__ == "__main__":
    main()
