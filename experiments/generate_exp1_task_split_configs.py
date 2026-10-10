"""Generate split Experiment 1 configs from a frozen task plan."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config_loader import dump_yaml_file, load_yaml_file


DEFAULT_PLAN = PROJECT_ROOT / "experiments" / "configs" / "exp1_main_a_bootstrap_task_plan.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "configs" / "exp1_main_a_bootstrap_split"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate split Experiment 1 configs from a task plan.")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=[],
        help="Optional task ids to emit. Omit to generate every task in the plan.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[],
        help="Optional seed override. Omit to use the task plan seeds.",
    )
    parser.add_argument("--experiment-prefix", default="")
    parser.add_argument("--memory-dir", default="")
    parser.add_argument(
        "--memory-scope",
        default="",
        help="Optional memory sharing scope override, e.g. seed or task_seed.",
    )
    parser.add_argument("--base-output-dir", default="")
    parser.add_argument("--total-slots", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="Optional variant override. Omit to use the task plan variants.",
    )
    parser.add_argument(
        "--persistent-variants",
        nargs="*",
        default=[],
        help="Optional memory-backed variant override for split configs.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = load_yaml_file(str(path))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return dict(loaded)


def _resolve_plan_path(raw_path: object, *, plan_path: Path) -> Path:
    path = Path(str(raw_path or "").strip())
    if path.is_absolute():
        return path
    plan_relative = (plan_path.parent / path).resolve()
    if plan_relative.exists():
        return plan_relative
    return (PROJECT_ROOT / path).resolve()


def _slugify(value: object) -> str:
    text = []
    for char in str(value or "").strip().lower():
        if char.isalnum():
            text.append(char)
        elif char in {"-", "_"}:
            text.append("-")
        else:
            text.append("-")
    normalized = "".join(text)
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-") or "task"


def _merge_dicts(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        merged.update(dict(mapping or {}))
    return merged


def _planned_runs(config: dict[str, Any]) -> int:
    memory_transfer = dict(config.get("memory_transfer", {}) or {})
    warmup_repeats = max(1, int(memory_transfer.get("warmup_repeats", 1) or 1))
    transfer_repeats = max(1, int(memory_transfer.get("transfer_repeats", 1) or 1))
    scenario_runs = (
        len(list(config.get("warmup_scenario_ids", []) or [])) * warmup_repeats
        + len(list(config.get("transfer_scenario_ids", []) or [])) * transfer_repeats
    )
    return int(len(list(config.get("variants", []) or [])) * scenario_runs)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_tasks(plan: dict[str, Any], task_ids: list[str]) -> list[dict[str, Any]]:
    tasks = [dict(item) for item in list(plan.get("tasks", []) or []) if isinstance(item, dict)]
    if len(task_ids) == 0:
        return tasks
    requested = {str(item).strip() for item in task_ids if str(item).strip() != ""}
    selected = [task for task in tasks if str(task.get("task_id", "")).strip() in requested]
    missing = sorted(requested - {str(task.get("task_id", "")).strip() for task in selected})
    if missing:
        raise ValueError(f"Unknown task ids in plan: {missing}")
    return selected


def _selected_seeds(plan: dict[str, Any], seed_override: list[int]) -> list[int]:
    seeds = [int(seed) for seed in list(seed_override or plan.get("seeds", []) or [1])]
    return list(dict.fromkeys(seeds))


def _task_config(
    *,
    base_config: dict[str, Any],
    plan: dict[str, Any],
    plan_path: Path,
    task: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    task_id = str(task.get("task_id", "task") or "task").strip()
    intent_slug = str(task.get("intent_slug", "") or "").strip()
    if intent_slug == "":
        raise ValueError(f"Task {task_id} is missing intent_slug.")

    prefix = str(args.experiment_prefix or plan.get("experiment_prefix", "exp1_main_a_bootstrap_v1")).strip()
    config["experiment_name"] = f"{prefix}_{_slugify(task_id)}_seed{int(seed)}"
    config["base_output_dir"] = str(
        args.base_output_dir or plan.get("base_output_dir") or config.get("base_output_dir", "outputs/paper_eval")
    )
    config["reference_scenario_id"] = str(
        task.get("reference_scenario_id")
        or plan.get("reference_scenario_id")
        or config.get("reference_scenario_id", "")
    )
    config["intent_subset"] = [intent_slug]
    config["warmup_scenario_ids"] = list(task.get("warmup_scenario_ids", []) or [])
    config["transfer_scenario_ids"] = list(task.get("transfer_scenario_ids", []) or [])
    config["variants"] = list(args.variants or plan.get("variants", []) or config.get("variants", []) or [])
    config["seeds"] = [int(seed)]

    memory_transfer = _merge_dicts(
        dict(config.get("memory_transfer", {}) or {}),
        dict(plan.get("memory_transfer", {}) or {}),
        dict(task.get("memory_transfer", {}) or {}),
    )
    memory_dir = str(args.memory_dir or memory_transfer.get("cumulative_memory_dir", "") or "").strip()
    bootstrap_memory_source = str(
        memory_transfer.get("bootstrap_memory_source", "")
        or plan.get("bootstrap_memory_source", "")
        or ""
    ).strip()
    memory_scope = str(
        args.memory_scope
        or memory_transfer.get("shared_memory_scope", "seed")
        or "seed"
    ).strip()
    memory_transfer.update(
        {
            "enabled": True,
            "cumulative_memory_enabled": True,
            "shared_memory_scope": memory_scope,
        }
    )
    if memory_dir != "":
        memory_transfer["cumulative_memory_dir"] = memory_dir
    if bootstrap_memory_source != "":
        memory_transfer["bootstrap_memory_source"] = bootstrap_memory_source
    if args.persistent_variants:
        memory_transfer["persistent_variants"] = list(args.persistent_variants)
    config["memory_transfer"] = memory_transfer

    force_sim_overrides = _merge_dicts(
        dict(config.get("force_sim_overrides", {}) or {}),
        dict(plan.get("force_sim_overrides", {}) or {}),
        dict(task.get("force_sim_overrides", {}) or {}),
    )
    total_slots = int(args.total_slots or force_sim_overrides.get("total_slots", 0) or 0)
    if total_slots > 0:
        force_sim_overrides["total_slots"] = total_slots
    if force_sim_overrides:
        config["force_sim_overrides"] = force_sim_overrides

    config["global_sim_overrides"] = _merge_dicts(
        dict(config.get("global_sim_overrides", {}) or {}),
        dict(plan.get("global_sim_overrides", {}) or {}),
        dict(task.get("global_sim_overrides", {}) or {}),
    )
    config["global_ric_overrides"] = _merge_dicts(
        dict(config.get("global_ric_overrides", {}) or {}),
        dict(plan.get("global_ric_overrides", {}) or {}),
        dict(task.get("global_ric_overrides", {}) or {}),
    )
    config["experiment1_protocol"] = {
        "plan_path": str(plan_path),
        "protocol_name": str(plan.get("protocol_name", "") or ""),
        "task_id": task_id,
        "task_label": str(task.get("label", task_id) or task_id),
        "seed": int(seed),
        "bootstrap_memory_source": str(plan.get("bootstrap_memory_source", "") or ""),
        "headline_role": str(task.get("headline_role", "exp1_a_main") or "exp1_a_main"),
        "notes": str(task.get("notes", "") or ""),
    }
    return config


def main() -> None:
    args = _parse_args()
    plan_path = Path(str(args.plan)).resolve()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = _load_yaml(plan_path)
    base_config_path = _resolve_plan_path(plan.get("base_config", ""), plan_path=plan_path)
    base_config = _load_yaml(base_config_path)
    tasks = _selected_tasks(plan, list(args.task_ids or []))
    seeds = _selected_seeds(plan, list(args.seeds or []))

    manifest_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id", "task") or "task").strip()
        for seed in seeds:
            config = _task_config(
                base_config=base_config,
                plan=plan,
                plan_path=plan_path,
                task=task,
                seed=int(seed),
                args=args,
            )
            config_path = output_dir / f"{config['experiment_name']}.yaml"
            dump_yaml_file(config, str(config_path))
            manifest_rows.append(
                {
                    "task_id": task_id,
                    "label": str(task.get("label", task_id) or task_id),
                    "intent_slug": str(task.get("intent_slug", "") or ""),
                    "seed": int(seed),
                    "config_path": str(config_path),
                    "output_dir": str(Path(str(config["base_output_dir"])) / str(config["experiment_name"])),
                    "warmup_scenario_ids": ";".join(list(config.get("warmup_scenario_ids", []) or [])),
                    "transfer_scenario_ids": ";".join(list(config.get("transfer_scenario_ids", []) or [])),
                    "variant_count": len(list(config.get("variants", []) or [])),
                    "planned_runs": _planned_runs(config),
                    "total_slots": int(dict(config.get("force_sim_overrides", {}) or {}).get("total_slots", 0) or 0),
                    "memory_dir": str(
                        dict(config.get("memory_transfer", {}) or {}).get("cumulative_memory_dir", "") or ""
                    ),
                    "memory_scope": str(
                        dict(config.get("memory_transfer", {}) or {}).get("shared_memory_scope", "") or ""
                    ),
                }
            )

    manifest_path = output_dir / "task_manifest.csv"
    _write_csv(manifest_path, manifest_rows)
    print(
        f"Generated {len(manifest_rows)} Experiment 1 split configs under {output_dir}. "
        f"Planned case runs: {sum(int(row['planned_runs']) for row in manifest_rows)}. "
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
