"""
Utilities for scenario-matrix expansion, execution, and aggregation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.factory import build_controller_registry, build_phy_adapter, build_scheduler
from core.simulator import Simulator
from experiments.eval_metrics import compute_cross_scenario_generalization_gap
from models.scenario import ScenarioConfig
from models.scenario_matrix import ScenarioMatrixCase
from ric.non_rt.active_intent_manager import ActiveIntentManager
from scenarios.scenario_generator import ScenarioGenerator
from scenarios.scenario_registry import ScenarioRegistry, deep_merge_dict


def build_runtime_scenario_config(
    *,
    scenario_name: str,
    scenario_type: str,
    sim_config: dict[str, Any],
    cell_config: dict[str, Any],
    ric_config: dict[str, Any],
) -> ScenarioConfig:
    return ScenarioConfig(
        total_slots=int(sim_config.get("total_slots", 10)),
        slot_duration_ms=float(sim_config.get("slot_duration_ms", 1.0)),
        cell_id=str(cell_config.get("cell_id", "cell_0")),
        xapp_interval_slots=int(ric_config.get("xapp_interval_slots", 5)),
        rapp_interval_slots=int(ric_config.get("rapp_interval_slots", 10)),
        run_rapps_on_start=bool(ric_config.get("run_rapps_on_start", False)),
        orchestration_warmup_slots=int(ric_config.get("orchestration_warmup_slots", 0)),
        scenario_name=str(scenario_name),
        scenario_type=str(scenario_type),
        obs_delay_slots=int(sim_config.get("obs_delay_slots", 0)),
        config_actuation_delay_slots=int(sim_config.get("config_actuation_delay_slots", 0)),
        command_actuation_delay_slots=int(sim_config.get("command_actuation_delay_slots", 0)),
        control_reject_prob=float(sim_config.get("control_reject_prob", 0.0)),
        rng_seed=int(sim_config.get("rng_seed", 1)),
        measurement_queue_delay_slots=int(sim_config.get("measurement_queue_delay_slots", 1)),
        measurement_sinr_window_slots=int(sim_config.get("measurement_sinr_window_slots", 8)),
        measurement_sinr_noise_std_db=float(sim_config.get("measurement_sinr_noise_std_db", 2.0)),
        measurement_bler_window_tbs=int(sim_config.get("measurement_bler_window_tbs", 20)),
        measurement_throughput_granularity_slots=int(sim_config.get("measurement_throughput_granularity_slots", 10)),
        variant_name=str(
            ric_config.get(
                "experiment_variant_name",
                sim_config.get("variant_name", "full_method"),
            )
        ),
        validator_enabled=bool(ric_config.get("validator_enabled", True)),
        rl_guidance_enabled=bool(ric_config.get("rl_guidance_enabled", True)),
        safety_shield_enabled=bool(ric_config.get("safety_shield_enabled", True)),
        static_xapp_selection_enabled=bool(
            ric_config.get("static_xapp_selection_enabled", False)
        ),
        reflection_enabled=bool(ric_config.get("reflection_enabled", True)),
        memory_mode=str(ric_config.get("memory_mode", "full") or "full"),
        selection_mode=str(ric_config.get("selection_mode", "reflective") or "reflective"),
        repair_mode=str(ric_config.get("repair_mode", "deterministic") or "deterministic"),
        conflict_mode=str(ric_config.get("conflict_mode", "typed_taxonomy") or "typed_taxonomy"),
        llm_pipeline_mode=str(ric_config.get("llm_pipeline_mode", "schema_bound") or "schema_bound"),
        near_rt_mode=str(ric_config.get("near_rt_mode", "projected_tuning") or "projected_tuning"),
        non_rt_reselection_enabled=bool(ric_config.get("non_rt_reselection_enabled", False)),
        non_rt_reselection_fallback_threshold=int(
            ric_config.get("non_rt_reselection_fallback_threshold", 2)
        ),
        non_rt_reselection_min_gap_slots=int(
            ric_config.get("non_rt_reselection_min_gap_slots", 10)
        ),
    )


def apply_matrix_case(
    *,
    base_configs: dict[str, dict[str, Any]],
    case: ScenarioMatrixCase,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    scenario_config = {
        "scenario_name": case.scenario_name,
        "template_overrides": dict(case.template_overrides),
        "overrides": dict(case.scenario_overrides),
    }
    sim_config = deep_merge_dict(base_configs["sim"], case.sim_overrides)
    cell_config = deep_merge_dict(base_configs["cell"], case.cell_overrides)
    ue_config = deep_merge_dict(base_configs["ue"], case.ue_overrides)
    ric_config = deep_merge_dict(base_configs["ric"], case.ric_overrides)
    beamforming_config = deep_merge_dict(base_configs["beamforming"], case.beamforming_overrides)
    return scenario_config, sim_config, cell_config, ue_config, ric_config, beamforming_config


def run_matrix_case(
    *,
    base_configs: dict[str, dict[str, Any]],
    phy_config: dict[str, Any],
    case: ScenarioMatrixCase,
    output_dir: str,
    active_intent_text: str = "",
) -> dict[str, Any]:
    scenario_config, sim_config, cell_config, ue_config, ric_config, beamforming_config = apply_matrix_case(
        base_configs=base_configs,
        case=case,
    )
    scenario_template = ScenarioRegistry().resolve(scenario_config)
    scenario_bundle = ScenarioGenerator().generate(
        template=scenario_template,
        scenario_config=scenario_config,
        sim_config=sim_config,
        cell_config=cell_config,
        ue_config=ue_config,
        beamforming_config=beamforming_config,
    )
    runtime_config = build_runtime_scenario_config(
        scenario_name=scenario_bundle.scenario_name,
        scenario_type=scenario_bundle.scenario_template.channel.scenario_type,
        sim_config=scenario_bundle.sim_config,
        cell_config=scenario_bundle.cell_config,
        ric_config=ric_config,
    )
    controller_registry = build_controller_registry(ric_config)
    normalized_intent_text = str(active_intent_text or "").strip()
    if normalized_intent_text != "":
        controller_registry.active_intent_manager = ActiveIntentManager(
            initial_intent_text=normalized_intent_text,
            initial_epoch=1,
            initial_change_reason="experiment_runner_initial_intent",
            source_name="run_matrix_case",
        )

    simulator = Simulator(
        config=runtime_config,
        controller_registry=controller_registry,
        scheduler=build_scheduler(scenario_bundle.sim_config),
        phy_adapter=build_phy_adapter(phy_config, scenario_bundle.beamforming_config),
        cell_config=scenario_bundle.cell_config,
        ue_config=scenario_bundle.ue_config,
        beamforming_config=scenario_bundle.beamforming_config,
        output_dir=output_dir,
    )
    simulator.run()

    final_summary_path = Path(output_dir) / "final_summary.json"
    with final_summary_path.open("r", encoding="utf-8") as summary_file:
        final_summary = json.load(summary_file)
    return final_summary


def flatten_nested_record(record: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        field_name = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_nested_record(value, prefix=f"{field_name}_"))
        elif isinstance(value, (list, tuple)):
            flattened[field_name] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[field_name] = value
    return flattened


def build_case_summary_record(
    *,
    case: ScenarioMatrixCase,
    final_summary: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    record = {
        "case_id": case.case_id,
        "scenario_case_id": case.case_id,
        "scenario_name": case.scenario_name,
        "description": case.description,
        "tags": json.dumps(case.tags, ensure_ascii=False),
        "coordinates": json.dumps(case.coordinates, ensure_ascii=False),
        "output_dir": str(output_dir),
    }
    record.update(flatten_nested_record(final_summary, prefix="summary_"))
    return record


def aggregate_case_records(
    case_records: list[dict[str, Any]],
    *,
    reference_scenario_name: str = "benchmark_nominal_cluster",
) -> dict[str, Any]:
    numeric_fields: list[str] = []
    for record in case_records:
        for field_name, value in record.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if field_name not in numeric_fields:
                    numeric_fields.append(field_name)
    aggregates: dict[str, Any] = {
        "case_count": int(len(case_records)),
        "case_ids": [record["case_id"] for record in case_records],
        "reference_scenario_name": str(reference_scenario_name),
    }
    for field_name in numeric_fields:
        values = [
            float(record[field_name])
            for record in case_records
            if record.get(field_name) is not None and record.get(field_name) != ""
        ]
        if len(values) == 0:
            continue
        aggregates[f"{field_name}_mean"] = sum(values) / float(len(values))
        aggregates[f"{field_name}_min"] = min(values)
        aggregates[f"{field_name}_max"] = max(values)
    aggregates["cross_scenario_generalization_gap"] = compute_cross_scenario_generalization_gap(
        case_records,
        reference_scenario_name=reference_scenario_name,
    )
    return aggregates


def write_matrix_summary(
    *,
    output_dir: str,
    matrix_name: str,
    matrix_description: str,
    case_records: list[dict[str, Any]],
    reference_scenario_name: str = "benchmark_nominal_cluster",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "matrix_name": matrix_name,
        "description": matrix_description,
        "reference_scenario_name": str(reference_scenario_name),
        "aggregates": aggregate_case_records(
            case_records,
            reference_scenario_name=reference_scenario_name,
        ),
        "cases": case_records,
    }
    with (output_path / "matrix_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary_payload, json_file, indent=4, ensure_ascii=False)

    if len(case_records) == 0:
        return

    field_names: list[str] = []
    for record in case_records:
        for field_name in record.keys():
            if field_name not in field_names:
                field_names.append(field_name)

    with (output_path / "matrix_cases.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(case_records)
