"""Run the intent-driven rApp/xApp orchestration pipeline."""

import argparse
import os
from pathlib import Path
import sys
from typing import Any


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.config_loader import load_all_configs, load_yaml_file
from core.debug import DebugConfig
from core.factory import build_controller_registry, build_phy_adapter, build_scheduler
from core.simulator import Simulator
from models.scenario import ScenarioConfig
from ric.non_rt.active_intent_manager import ActiveIntentManager
from scenarios.scenario_generator import ScenarioGenerator
from scenarios.scenario_registry import ScenarioRegistry, deep_merge_dict


def _configure_utf8_console() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intent-driven RAN orchestration.")
    parser.add_argument("--intent", type=str, default="", help="User intent text for the orchestration pipeline.")
    parser.add_argument("--intent-file", type=str, default="", help="Optional text file containing the user intent.")
    parser.add_argument(
        "--total-slots",
        type=int,
        default=None,
        help="Override total slots. Use 0 or a negative value to run until interrupted.",
    )
    parser.add_argument(
        "--run-forever",
        action="store_true",
        help="Run continuously until interrupted with Ctrl+C.",
    )
    return parser.parse_args()


def _resolve_intent_text(args: argparse.Namespace) -> str:
    if str(getattr(args, "intent", "") or "").strip() != "":
        return str(args.intent).strip()
    intent_file = str(getattr(args, "intent_file", "") or "").strip()
    if intent_file != "":
        return Path(intent_file).read_text(encoding="utf-8").strip()
    return input("Enter intent for the orchestration run: ").strip()


def _load_intent_specific_overrides(
    configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    configs = dict(configs)
    configs["phy"] = load_yaml_file(str(project_root / "configs" / "phy.yaml"))
    override_names = ("scenario", "sim", "cell", "ue", "beamforming", "ric", "phy")
    for override_name in override_names:
        override_path = project_root / "configs" / f"{override_name}_intent_orchestration.yaml"
        if not override_path.exists():
            continue
        configs[override_name] = deep_merge_dict(
            dict(configs.get(override_name, {}) or {}),
            load_yaml_file(str(override_path)),
        )
    return configs


def main() -> None:
    _configure_utf8_console()
    args = _parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    DebugConfig.DEBUG = False
    DebugConfig.ORCHESTRATION_SUMMARY = True
    DebugConfig.CONCISE_SLOT_METRICS = True

    configs = load_all_configs(str(project_root / "configs"))
    configs = _load_intent_specific_overrides(configs)
    intent_text = _resolve_intent_text(args)
    if intent_text == "":
        raise ValueError("Intent text is required for run_intent_orchestration.py")
    for rapp_item in list(configs["ric"].get("rapps", []) or []):
        if str(rapp_item.get("type", "")) == "intent_goal_translator_rapp":
            rapp_item["intent_text"] = intent_text
    phy_config = dict(configs.get("phy", {}) or {})

    scenario_template = ScenarioRegistry().resolve(configs.get("scenario"))
    scenario_bundle = ScenarioGenerator().generate(
        template=scenario_template,
        scenario_config=configs.get("scenario"),
        sim_config=configs["sim"],
        cell_config=configs["cell"],
        ue_config=configs["ue"],
        beamforming_config=configs["beamforming"],
    )

    total_slots = int(scenario_bundle.sim_config["total_slots"])
    if bool(configs["ric"].get("run_forever", False)):
        total_slots = -1
    if args.total_slots is not None:
        total_slots = int(args.total_slots)
    if args.run_forever:
        total_slots = -1

    scenario_config = ScenarioConfig(
        total_slots=total_slots,
        slot_duration_ms=scenario_bundle.sim_config["slot_duration_ms"],
        cell_id=scenario_bundle.cell_config["cell_id"],
        xapp_interval_slots=configs["ric"]["xapp_interval_slots"],
        rapp_interval_slots=configs["ric"]["rapp_interval_slots"],
        run_rapps_on_start=bool(configs["ric"].get("run_rapps_on_start", True)),
        orchestration_warmup_slots=int(configs["ric"].get("orchestration_warmup_slots", 0)),
        scenario_name=scenario_bundle.scenario_name,
        scenario_type=scenario_bundle.scenario_template.channel.scenario_type,
        obs_delay_slots=int(scenario_bundle.sim_config.get("obs_delay_slots", 0)),
        config_actuation_delay_slots=int(scenario_bundle.sim_config.get("config_actuation_delay_slots", 0)),
        command_actuation_delay_slots=int(scenario_bundle.sim_config.get("command_actuation_delay_slots", 0)),
        control_reject_prob=float(scenario_bundle.sim_config.get("control_reject_prob", 0.0)),
        rng_seed=int(scenario_bundle.sim_config.get("rng_seed", 1)),
        measurement_queue_delay_slots=int(scenario_bundle.sim_config.get("measurement_queue_delay_slots", 1)),
        measurement_sinr_window_slots=int(scenario_bundle.sim_config.get("measurement_sinr_window_slots", 8)),
        measurement_sinr_noise_std_db=float(scenario_bundle.sim_config.get("measurement_sinr_noise_std_db", 2.0)),
        measurement_bler_window_tbs=int(scenario_bundle.sim_config.get("measurement_bler_window_tbs", 20)),
        measurement_throughput_granularity_slots=int(
            scenario_bundle.sim_config.get("measurement_throughput_granularity_slots", 10)
        ),
    )

    controller_registry = build_controller_registry(configs["ric"])
    controller_registry.active_intent_manager = ActiveIntentManager(
        initial_intent_text=intent_text,
        initial_epoch=1,
        initial_change_reason="initial_cli_input",
        source_name="run_intent_orchestration_cli",
    )

    simulator = Simulator(
        config=scenario_config,
        controller_registry=controller_registry,
        scheduler=build_scheduler(scenario_bundle.sim_config),
        phy_adapter=build_phy_adapter(phy_config, scenario_bundle.beamforming_config),
        cell_config=scenario_bundle.cell_config,
        ue_config=scenario_bundle.ue_config,
        beamforming_config=scenario_bundle.beamforming_config,
        output_dir=str(project_root / "outputs" / "intent_orchestration"),
    )
    simulator.run()


if __name__ == "__main__":
    main()
