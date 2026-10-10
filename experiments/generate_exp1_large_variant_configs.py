from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


BASE_CONFIG = Path("experiments/configs/experiment1_cross_scenario_transfer.yaml")
OUTPUT_DIR = Path("experiments/configs")

VARIANT_SLUGS = {
    "fixed_plan": "fixed",
    "no_memory_planner": "nomem",
    "success_only_memory": "succ",
    "orchestran_like_resource_aware": "orch",
    "intent_hrl_like": "hrl",
    "oreo_like_service_graph": "oreo",
    "full_method": "full",
}


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    base_experiment_name = str(base.get("experiment_name", "exp1_transfer_v16_literature_aligned"))
    for variant_name, slug in VARIANT_SLUGS.items():
        config = deepcopy(base)
        config["experiment_name"] = f"{base_experiment_name}_{slug}"
        config["variants"] = [variant_name]
        memory_transfer = dict(config.get("memory_transfer", {}) or {})
        if variant_name in {"success_only_memory", "full_method"}:
            memory_transfer["persistent_variants"] = [variant_name]
        else:
            memory_transfer["persistent_variants"] = []
        config["memory_transfer"] = memory_transfer
        output_path = OUTPUT_DIR / f"{base_experiment_name}_{slug}.yaml"
        output_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(output_path)


if __name__ == "__main__":
    main()
