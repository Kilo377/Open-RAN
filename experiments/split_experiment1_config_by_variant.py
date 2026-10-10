"""Split an Experiment 1 transfer config into one config per method variant."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


VARIANT_SLUGS = {
    "fixed_plan": "fixed",
    "no_memory_planner": "nomem",
    "success_only_memory": "succ",
    "orchestran_like_resource_aware": "orch",
    "intent_hrl_like": "hrl",
    "oreo_like_service_graph": "oreo",
    "full_method": "full",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Experiment 1 transfer config by variant.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="experiments/configs/exp1_micro_split")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(str(args.config)).resolve()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_experiment_name = str(base.get("experiment_name", config_path.stem) or config_path.stem)
    variants = [str(item) for item in list(base.get("variants", []) or [])]
    for variant_name in variants:
        slug = VARIANT_SLUGS.get(variant_name, variant_name.replace("_", "-"))
        config = deepcopy(base)
        config["experiment_name"] = f"{base_experiment_name}_{slug}"
        config["variants"] = [variant_name]
        memory_transfer = dict(config.get("memory_transfer", {}) or {})
        if variant_name in {"success_only_memory", "full_method"}:
            memory_transfer["persistent_variants"] = [variant_name]
        else:
            memory_transfer["persistent_variants"] = []
        config["memory_transfer"] = memory_transfer
        output_path = output_dir / f"{base_experiment_name}_{slug}.yaml"
        output_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(output_path)


if __name__ == "__main__":
    main()
