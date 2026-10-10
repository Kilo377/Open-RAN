"""Generate small, recombinable Experiment 1 chunk configs.

The default chunking strategy creates one config per (intent, seed), while
keeping all seven methods and all warm-up/transfer scenarios in each chunk.
This makes every chunk locally comparable and keeps reruns small.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BASE_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "experiment1_cross_scenario_transfer.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "configs" / "exp1_chunks"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Experiment 1 chunk configs.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--mode",
        choices=["intent-seed", "intent"],
        default="intent-seed",
        help="intent-seed creates one small config per intent and seed; intent keeps all seeds per intent.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="exp1_transfer_v16_chunks",
        help="Prefix for generated experiment_name values.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        loaded = yaml.safe_load(input_file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return loaded


def _slugify(value: object) -> str:
    raw = str(value or "").strip().lower()
    normalized = []
    for char in raw:
        if char.isalnum():
            normalized.append(char)
        elif char in {"-", "_"}:
            normalized.append("-")
        else:
            normalized.append("-")
    text = "".join(normalized)
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "chunk"


def _intent_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    subset = [str(item) for item in list(config.get("intent_subset", []) or [])]
    suite = list(config.get("intent_suite", []) or [])
    if len(subset) == 0:
        return [dict(item) for item in suite if isinstance(item, dict)]
    by_slug = {
        str(item.get("intent_slug", "") or ""): dict(item)
        for item in suite
        if isinstance(item, dict)
    }
    missing = [slug for slug in subset if slug not in by_slug]
    if missing:
        raise ValueError(f"Missing intent_suite definitions for: {missing}")
    return [by_slug[slug] for slug in subset]


def _planned_runs_per_chunk(config: dict[str, Any], *, intent_count: int, seed_count: int) -> int:
    variants = list(config.get("variants", []) or [])
    warmup = list(config.get("warmup_scenario_ids", []) or [])
    transfer = list(config.get("transfer_scenario_ids", []) or [])
    memory_transfer = dict(config.get("memory_transfer", {}) or {})
    warmup_repeats = int(memory_transfer.get("warmup_repeats", 1) or 1)
    transfer_repeats = int(memory_transfer.get("transfer_repeats", 1) or 1)
    scenario_runs = int(len(warmup) * warmup_repeats + len(transfer) * transfer_repeats)
    return int(intent_count * seed_count * len(variants) * scenario_runs)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _chunk_output_dir(config: dict[str, Any]) -> str:
    base_output_dir = Path(str(config.get("base_output_dir", "outputs/paper_eval") or "outputs/paper_eval"))
    return str(base_output_dir / str(config.get("experiment_name", "experiment1_chunk")))


def main() -> None:
    args = _parse_args()
    base_config_path = Path(str(args.base_config)).resolve()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_yaml(base_config_path)
    intent_specs = _intent_specs(base_config)
    seeds = [int(seed) for seed in list(base_config.get("seeds", []) or [1])]

    rows: list[dict[str, Any]] = []
    generated_count = 0
    if args.mode == "intent-seed":
        chunk_index = 0
        for intent_spec in intent_specs:
            intent_slug = str(intent_spec.get("intent_slug", "intent") or "intent")
            for seed in seeds:
                chunk_index += 1
                config = deepcopy(base_config)
                chunk_id = f"chunk{chunk_index:03d}_{_slugify(intent_slug)}_seed{int(seed)}"
                config["experiment_name"] = f"{args.experiment_prefix}_{chunk_id}"
                config["intent_subset"] = [intent_slug]
                config["seeds"] = [int(seed)]
                config_path = output_dir / f"{config['experiment_name']}.yaml"
                config_path.write_text(
                    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                generated_count += 1
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "config_path": str(config_path),
                        "output_dir": _chunk_output_dir(config),
                        "intent_slug": intent_slug,
                        "seed": int(seed),
                        "intent_count": 1,
                        "seed_count": 1,
                        "variant_count": len(list(config.get("variants", []) or [])),
                        "planned_runs": _planned_runs_per_chunk(config, intent_count=1, seed_count=1),
                    }
                )
    else:
        for chunk_index, intent_spec in enumerate(intent_specs, start=1):
            intent_slug = str(intent_spec.get("intent_slug", "intent") or "intent")
            config = deepcopy(base_config)
            chunk_id = f"chunk{chunk_index:03d}_{_slugify(intent_slug)}_allseeds"
            config["experiment_name"] = f"{args.experiment_prefix}_{chunk_id}"
            config["intent_subset"] = [intent_slug]
            config_path = output_dir / f"{config['experiment_name']}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            generated_count += 1
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "config_path": str(config_path),
                    "output_dir": _chunk_output_dir(config),
                    "intent_slug": intent_slug,
                    "seed": "all",
                    "intent_count": 1,
                    "seed_count": len(seeds),
                    "variant_count": len(list(config.get("variants", []) or [])),
                    "planned_runs": _planned_runs_per_chunk(
                        config,
                        intent_count=1,
                        seed_count=len(seeds),
                    ),
                }
            )

    manifest_path = output_dir / "chunk_manifest.csv"
    _write_manifest(manifest_path, rows)
    total_planned = sum(int(row["planned_runs"]) for row in rows)
    print(
        f"Generated {generated_count} Experiment 1 chunk configs under {output_dir}. "
        f"Total planned runs across chunks: {total_planned}. Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
