"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: factory.py
Description: This file defines factory helper functions for creating
controllers, schedulers, and PHY adapters from YAML configuration.
"""

from phy.custom_phy_adapter import CustomPhyAdapter
from phy.link_adapter import FakePhyAdapter
from phy.sionna_phy_adapter import SionnaPhyAdapter
from ran.scheduler import SimpleScheduler
from ric.controller_registry import ControllerRegistry
from ric.near_rt.beam_xapp import BeamXApp
from ric.near_rt.cell_sleep_xapp import CellSleepXApp
from ric.near_rt.goal_parameter_tuning_xapp import GoalParameterTuningXApp
from ric.near_rt.latency_scheduler_xapp import LatencySchedulerXApp
from ric.near_rt.massive_mimo_beamformer_xapp import MassiveMIMOBeamformerXApp
from ric.near_rt.mobility_predictor_xapp import MobilityPredictorXApp
from ric.near_rt.power_saving_xapp import PowerSavingXApp
from ric.near_rt.ran_slicing_manager_xapp import RANSlicingManagerXApp
from ric.near_rt.spectrum_optimizer_xapp import SpectrumOptimizerXApp
from ric.near_rt.traffic_steering_xapp import TrafficSteeringXApp
from ric.near_rt.urllc_guard_xapp import URLLCGuardXApp
from ric.near_rt.xapp_variant_catalog import expand_default_variant_xapp_configs
from ric.non_rt.codebook_rapp import CodebookRApp
from ric.non_rt.intent_goal_translator_rapp import IntentGoalTranslatorRApp
from ric.non_rt.reward_shaping_rapp import RewardShapingRApp
from ric.non_rt.slicing_rapp import SlicingRApp
from ric.non_rt.xapp_selection_rapp import XAppSelectionRApp


def _rapp_llm_kwargs(
    rapp_item: dict,
    *,
    default_max_output_tokens: int,
    default_startup_timeout_s: float = 600.0,
) -> dict:
    return {
        "llm_enabled": bool(rapp_item.get("llm_enabled", rapp_item.get("ollama_enabled", True))),
        "llm_provider": str(rapp_item.get("llm_provider", "ollama")),
        "llm_api_key": rapp_item.get("llm_api_key"),
        "llm_api_key_env_var": str(rapp_item.get("llm_api_key_env_var", "DASHSCOPE_API_KEY")),
        "llm_base_url": rapp_item.get("llm_base_url", rapp_item.get("ollama_base_url")),
        "llm_model": rapp_item.get("llm_model", rapp_item.get("ollama_model")),
        "llm_timeout_s": float(rapp_item.get("llm_timeout_s", rapp_item.get("ollama_timeout_s", 120.0))),
        "llm_startup_timeout_s": float(
            rapp_item.get(
                "llm_startup_timeout_s",
                rapp_item.get("ollama_startup_timeout_s", default_startup_timeout_s),
            )
        ),
        "llm_max_output_tokens": int(
            rapp_item.get(
                "llm_max_output_tokens",
                rapp_item.get("ollama_max_output_tokens", default_max_output_tokens),
            )
        ),
        "llm_request_retries": int(
            rapp_item.get("llm_request_retries", rapp_item.get("ollama_request_retries", 1))
        ),
        "llm_temperature": float(rapp_item.get("llm_temperature", 0.0)),
        "llm_enable_thinking": bool(rapp_item.get("llm_enable_thinking", False)),
        "llm_reasoning_effort": str(rapp_item.get("llm_reasoning_effort", "")),
        "llm_extra_body": dict(rapp_item.get("llm_extra_body", {}) or {}),
    }


def build_controller_registry(ric_config: dict) -> ControllerRegistry:
    """
    Builds the controller registry from the RIC configuration dictionary.
    INPUT
    ric_config: The RIC configuration dictionary loaded from YAML.
    OUTPUT
    One ControllerRegistry object.
    EXAMPLE OUTPUT
    controller_registry = build_controller_registry(ric_config)
    """
    controller_registry = ControllerRegistry()
    global_validator_enabled = bool(ric_config.get("validator_enabled", True))
    global_rl_guidance_enabled = bool(ric_config.get("rl_guidance_enabled", True))
    global_safety_shield_enabled = bool(ric_config.get("safety_shield_enabled", True))
    global_static_xapp_selection_enabled = bool(
        ric_config.get("static_xapp_selection_enabled", False)
    )
    global_reflection_enabled = bool(ric_config.get("reflection_enabled", True))
    global_memory_mode = str(ric_config.get("memory_mode", "full") or "full")
    global_selection_mode = str(ric_config.get("selection_mode", "reflective") or "reflective")
    global_memory_age_decay_enabled = bool(ric_config.get("memory_age_decay_enabled", True))
    global_memory_retrieval_mode = str(ric_config.get("memory_retrieval_mode", "ranked") or "ranked")
    global_reflective_transfer_guard_enabled = bool(
        ric_config.get("reflective_transfer_guard_enabled", True)
    )
    global_max_selected_xapps = int(ric_config.get("max_selected_xapps", 4))
    global_success_memory_size = int(ric_config.get("success_memory_size", 12))
    global_reflection_memory_size = int(ric_config.get("reflection_memory_size", 8))
    global_retrieval_top_k = int(ric_config.get("retrieval_top_k", 3))
    global_persistent_memory_enabled = bool(ric_config.get("persistent_memory_enabled", False))
    global_memory_store_path = ric_config.get("memory_store_path")
    global_failure_gap_threshold = float(ric_config.get("failure_gap_threshold", 0.35))
    global_gap_patience_rounds = int(ric_config.get("gap_patience_rounds", 2))
    global_convergence_gap_threshold = float(ric_config.get("convergence_gap_threshold", 0.18))
    global_success_gap_threshold = ric_config.get("success_gap_threshold")
    global_stagnation_patience_rounds = int(ric_config.get("stagnation_patience_rounds", 2))
    global_stagnation_epsilon = float(ric_config.get("stagnation_epsilon", 0.03))
    global_experiment_transfer_phase = str(ric_config.get("experiment_transfer_phase", "") or "")
    global_experiment_phase_repeat = int(ric_config.get("experiment_phase_repeat", 0) or 0)
    global_literature_baseline_state_path = str(
        ric_config.get("literature_baseline_state_path", "") or ""
    )
    global_literature_baseline_state_enabled = bool(
        ric_config.get("literature_baseline_state_enabled", False)
    )
    global_intent_hrl_train_on_warmup = bool(ric_config.get("intent_hrl_train_on_warmup", True))
    global_intent_hrl_freeze_on_transfer = bool(
        ric_config.get("intent_hrl_freeze_on_transfer", True)
    )

    for rapp_item in ric_config.get("rapps", []):
        rapp_type = rapp_item["type"]

        if rapp_type == "codebook_rapp":
            controller_registry.register_rapp(
                CodebookRApp(
                    name=rapp_item["name"],
                    bler_threshold=rapp_item["bler_threshold"],
                    throughput_threshold_mbps=rapp_item["throughput_threshold_mbps"],
                )
            )
        elif rapp_type == "slicing_rapp":
            controller_registry.register_rapp(
                SlicingRApp(
                    name=rapp_item["name"],
                    queue_threshold_bytes=rapp_item["queue_threshold_bytes"],
                )
            )
        elif rapp_type == "intent_goal_translator_rapp":
            controller_registry.register_rapp(
                IntentGoalTranslatorRApp(
                    name=rapp_item["name"],
                    intent_text=str(rapp_item.get("intent_text", "")),
                    validator_enabled=bool(
                        rapp_item.get("validator_enabled", global_validator_enabled)
                    ),
                    hard_kpi_targets=list(rapp_item.get("hard_kpi_targets", []) or []),
                    translation_embedding_backend_name=str(
                        rapp_item.get("translation_embedding_backend_name", "hashed_feature_embedding_v1")
                    ),
                    translation_embedding_backend_state_path=rapp_item.get(
                        "translation_embedding_backend_state_path"
                    ),
                    **_rapp_llm_kwargs(
                        rapp_item,
                        default_max_output_tokens=1024,
                        default_startup_timeout_s=600.0,
                    ),
                )
            )
        elif rapp_type == "xapp_selection_rapp":
            controller_registry.register_rapp(
                XAppSelectionRApp(
                    name=rapp_item["name"],
                    max_selected_xapps=int(global_max_selected_xapps),
                    success_memory_size=int(global_success_memory_size),
                    reflection_memory_size=int(global_reflection_memory_size),
                    retrieval_top_k=int(global_retrieval_top_k),
                    max_cpu_cost=float(rapp_item.get("max_cpu_cost", 6.0)),
                    max_mem_cost=float(rapp_item.get("max_mem_cost", 5.5)),
                    mutex_category_pairs=list(rapp_item.get("mutex_category_pairs", []) or []),
                    dependency_category_pairs=list(
                        rapp_item.get("dependency_category_pairs", []) or []
                    ),
                    persistent_memory_enabled=bool(global_persistent_memory_enabled),
                    memory_store_path=global_memory_store_path,
                    available_xapp_names=list(
                        rapp_item.get(
                            "available_xapp_names",
                            ric_config.get("available_xapp_names", []),
                        )
                        or []
                    ),
                    static_xapp_selection_enabled=bool(global_static_xapp_selection_enabled),
                    freeze_learning=bool(rapp_item.get('freeze_learning',ric_config.get('freeze_learning',False))),
                    reflection_enabled=bool(global_reflection_enabled),
                    memory_mode=str(global_memory_mode),
                    selection_mode=str(global_selection_mode),
                    memory_age_decay_enabled=bool(global_memory_age_decay_enabled),
                    memory_retrieval_mode=str(global_memory_retrieval_mode),
                    reflective_transfer_guard_enabled=bool(
                        global_reflective_transfer_guard_enabled
                    ),
                    failure_gap_threshold=float(global_failure_gap_threshold),
                    gap_patience_rounds=int(global_gap_patience_rounds),
                    convergence_gap_threshold=float(global_convergence_gap_threshold),
                    success_gap_threshold=global_success_gap_threshold,
                    stagnation_patience_rounds=int(global_stagnation_patience_rounds),
                    stagnation_epsilon=float(global_stagnation_epsilon),
                    experiment_transfer_phase=str(
                        rapp_item.get("experiment_transfer_phase", global_experiment_transfer_phase)
                    ),
                    experiment_phase_repeat=int(
                        rapp_item.get("experiment_phase_repeat", global_experiment_phase_repeat)
                    ),
                    literature_baseline_state_enabled=bool(
                        rapp_item.get(
                            "literature_baseline_state_enabled",
                            global_literature_baseline_state_enabled,
                        )
                    ),
                    literature_baseline_state_path=str(
                        rapp_item.get(
                            "literature_baseline_state_path",
                            global_literature_baseline_state_path,
                        )
                        or ""
                    ),
                    intent_hrl_train_on_warmup=bool(
                        rapp_item.get("intent_hrl_train_on_warmup", global_intent_hrl_train_on_warmup)
                    ),
                    intent_hrl_freeze_on_transfer=bool(
                        rapp_item.get(
                            "intent_hrl_freeze_on_transfer",
                            global_intent_hrl_freeze_on_transfer,
                        )
                    ),
                    selection_embedding_backend_name=str(
                        rapp_item.get("selection_embedding_backend_name", "hashed_feature_embedding_v1")
                    ),
                    selection_embedding_backend_state_path=rapp_item.get(
                        "selection_embedding_backend_state_path"
                    ),
                    surrogate_predictor_name=str(
                        rapp_item.get("surrogate_predictor_name", "memory_calibrated_surrogate_predictor_v1")
                    ),
                    surrogate_predictor_state_path=rapp_item.get(
                        "surrogate_predictor_state_path"
                    ),
                    **_rapp_llm_kwargs(
                        rapp_item,
                        default_max_output_tokens=1536,
                        default_startup_timeout_s=600.0,
                    ),
                )
            )
        elif rapp_type == "reward_shaping_rapp":
            controller_registry.register_rapp(
                RewardShapingRApp(
                    name=rapp_item["name"],
                    rl_guidance_enabled=bool(
                        rapp_item.get("rl_guidance_enabled", global_rl_guidance_enabled)
                    ),
                    **_rapp_llm_kwargs(
                        rapp_item,
                        default_max_output_tokens=1536,
                        default_startup_timeout_s=900.0,
                    ),
                )
            )
        else:
            raise ValueError(f"Unsupported rApp type: {rapp_type}")

    expanded_xapp_items = expand_default_variant_xapp_configs(
        list(ric_config.get("xapps", []) or [])
    )

    for xapp_item in expanded_xapp_items:
        xapp_type = xapp_item["type"]

        if xapp_type == "xapp_instance":
            from ric.near_rt.xapp_library import XAppInstance
            controller_registry.register_xapp(XAppInstance(
                xapp_item['instance_id'],xapp_item['implementation_id'],xapp_item['cell_ids'],
                parameters=xapp_item.get('parameters'),priority=xapp_item.get('priority'),
                slot_duration_ms=xapp_item.get('slot_duration_ms',.5),initial_bindings=xapp_item.get('initial_bindings'),context_cell_ids=xapp_item.get('context_cell_ids')))
        elif xapp_type == "rule_based_v1":
            from ric.near_rt.rule_based_v1 import RuleXApp
            controller_registry.register_xapp(RuleXApp(
                xapp_item['kind'],name=xapp_item.get('name'),priority=xapp_item.get('priority'),
                parameters=xapp_item.get('parameters'),cell_ids=xapp_item.get('cell_ids',()),
                slot_duration_ms=xapp_item.get('slot_duration_ms',.5)))
        elif xapp_type == "rule_kpi_monitor":
            from ric.near_rt.rule_based_v1 import KpiMonitor
            controller_registry.register_xapp(KpiMonitor())
        elif xapp_type == "beam_xapp":
            controller_registry.register_xapp(
                BeamXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    sinr_threshold_db=xapp_item["sinr_threshold_db"],
                    queue_threshold_bytes=xapp_item["queue_threshold_bytes"],
                )
            )
        elif xapp_type == "latency_scheduler_xapp":
            controller_registry.register_xapp(
                LatencySchedulerXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    queue_threshold_bytes=xapp_item["queue_threshold_bytes"],
                )
            )
        elif xapp_type == "mobility_predictor_xapp":
            controller_registry.register_xapp(
                MobilityPredictorXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    risk_threshold=float(xapp_item.get("risk_threshold", 0.55)),
                )
            )
        elif xapp_type == "traffic_steering_xapp":
            controller_registry.register_xapp(
                TrafficSteeringXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    load_threshold=float(xapp_item.get("load_threshold", 0.70)),
                    edge_sinr_threshold_db=float(
                        xapp_item.get("edge_sinr_threshold_db", 5.0)
                    ),
                )
            )
        elif xapp_type == "power_saving_xapp":
            controller_registry.register_xapp(
                PowerSavingXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    low_load_threshold=float(xapp_item.get("low_load_threshold", 0.35)),
                    queue_threshold_bytes=int(xapp_item.get("queue_threshold_bytes", 3000)),
                    min_power_cap_dbm=float(xapp_item.get("min_power_cap_dbm", 24.0)),
                )
            )
        elif xapp_type == "spectrum_optimizer_xapp":
            controller_registry.register_xapp(
                SpectrumOptimizerXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    utilization_threshold=float(
                        xapp_item.get("utilization_threshold", 0.65)
                    ),
                    interference_threshold_db=float(
                        xapp_item.get("interference_threshold_db", 2.5)
                    ),
                )
            )
        elif xapp_type == "massive_mimo_beamformer_xapp":
            controller_registry.register_xapp(
                MassiveMIMOBeamformerXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    beam_risk_threshold=float(xapp_item.get("beam_risk_threshold", 0.50)),
                    coverage_sinr_threshold_db=float(
                        xapp_item.get("coverage_sinr_threshold_db", 7.0)
                    ),
                )
            )
        elif xapp_type == "ran_slicing_manager_xapp":
            controller_registry.register_xapp(
                RANSlicingManagerXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    slice_dominance_threshold=float(
                        xapp_item.get("slice_dominance_threshold", 0.45)
                    ),
                )
            )
        elif xapp_type == "urllc_guard_xapp":
            controller_registry.register_xapp(
                URLLCGuardXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    queue_threshold_bytes=int(xapp_item.get("queue_threshold_bytes", 1200)),
                    bler_threshold=float(xapp_item.get("bler_threshold", 0.12)),
                    latency_budget_threshold_ms=float(
                        xapp_item.get("latency_budget_threshold_ms", 20.0)
                    ),
                )
            )
        elif xapp_type == "goal_parameter_tuning_xapp":
            controller_registry.register_xapp(
                GoalParameterTuningXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    safety_shield_enabled=bool(
                        xapp_item.get(
                            "safety_shield_enabled",
                            global_safety_shield_enabled,
                        )
                    ),
                )
            )
        elif xapp_type == "cell_sleep_xapp":
            controller_registry.register_xapp(
                CellSleepXApp(
                    name=xapp_item["name"],
                    priority=xapp_item["priority"],
                    micro_sleep_load_threshold=float(
                        xapp_item.get("micro_sleep_load_threshold", 0.28)
                    ),
                    light_sleep_load_threshold=float(
                        xapp_item.get("light_sleep_load_threshold", 0.18)
                    ),
                    deep_sleep_load_threshold=float(
                        xapp_item.get("deep_sleep_load_threshold", 0.08)
                    ),
                    max_queue_bytes_for_sleep=int(
                        xapp_item.get("max_queue_bytes_for_sleep", 4000)
                    ),
                )
            )
        else:
            raise ValueError(f"Unsupported xApp type: {xapp_type}")

    return controller_registry


def build_scheduler(sim_config: dict):
    """
    Builds the scheduler object from the simulator configuration dictionary.
    INPUT
    sim_config: The simulator configuration dictionary loaded from YAML.
    OUTPUT
    One scheduler object implementing the scheduler base interface.
    EXAMPLE OUTPUT
    scheduler = build_scheduler(sim_config)
    """
    scheduler_type = sim_config.get("scheduler_type", "simple_scheduler")

    if scheduler_type in ("simple_scheduler", "rt_sys_scheduler"):
        scheduler_class = SimpleScheduler
        if scheduler_type == "rt_sys_scheduler":
            from phy.rt_sys.scheduler import RtSysScheduler
            scheduler_class = RtSysScheduler
        return scheduler_class(
            total_prbs_per_slot=int(sim_config.get("scheduler_total_prbs_per_slot", 48)),
            total_symbols_per_slot=int(sim_config.get("scheduler_total_symbols_per_slot", 14)),
            max_ues_per_slot=int(sim_config.get("scheduler_max_ues_per_slot", 2)),
            min_prbs_per_grant=int(sim_config.get("scheduler_min_prbs_per_grant", 4)),
            min_symbols_per_grant=int(sim_config.get("scheduler_min_symbols_per_grant", 4)),
            retx_weight_bonus=float(sim_config.get("scheduler_retx_weight_bonus", 1000.0)),
            packing_strategy=str(sim_config.get("scheduler_packing_strategy", "frequency_first")),
            enable_spatial_multiplexing=bool(sim_config.get("scheduler_enable_spatial_multiplexing", False)),
            max_mu_users_per_group=int(sim_config.get("scheduler_max_mu_users_per_group", 2)),
            min_mu_beam_separation_deg=float(
                sim_config.get("scheduler_min_mu_beam_separation_deg", 45.0)
            ),
            max_total_spatial_layers=sim_config.get("scheduler_max_total_spatial_layers", None),
        )

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def build_phy_adapter(phy_config: dict, beamforming_config: dict):
    """
    Builds the PHY adapter object from the PHY configuration dictionary.
    INPUT
    phy_config: The PHY configuration dictionary loaded from YAML.
    beamforming_config: The PHY configuration dictionary loaded from YAML.
    OUTPUT
    One PHY adapter object implementing the PHY base interface.
    EXAMPLE OUTPUT
    phy_adapter = build_phy_adapter(phy_config, beamforming_config)
    """
    phy_type = phy_config.get("phy_type", "fake_phy")

    if phy_type == "sionna_rt_sys":
        from phy.rt_sys.adapter import RtSysPhyAdapter
        return RtSysPhyAdapter(phy_config.get("rt_sys", {}))

    if phy_type == "fake_phy":
        return FakePhyAdapter()

    if phy_type == "custom_phy":
        custom_phy_config = phy_config.get("custom_phy", {})

        raw_codebooks = beamforming_config.get("codebooks", {})
        raw_profiles = beamforming_config.get("beam_profiles", {})

        codebook_map = {int(key): value for key, value in raw_codebooks.items()}
        beam_profile_map = {int(key): value for key, value in raw_profiles.items()}

        return CustomPhyAdapter(
            codebook_map=codebook_map,
            beam_profile_map=beam_profile_map,
            noise_floor_dbm=custom_phy_config.get("noise_floor_dbm", -94.0),
            base_pathloss_compensation_db=custom_phy_config.get(
                "base_pathloss_compensation_db",
                98.0,
            ),
            bler_target=custom_phy_config.get("bler_target", 0.1),
            max_mcs_index=custom_phy_config.get("max_mcs_index", 27),
            min_valid_sinr_db=custom_phy_config.get("min_valid_sinr_db", -40.0),
            max_valid_sinr_db=custom_phy_config.get("max_valid_sinr_db", 80.0),
        )

    if phy_type == "sionna_phy":
        sionna_phy_config = phy_config.get("sionna_phy", {})
        return SionnaPhyAdapter(
            sionna_config=sionna_phy_config,
        )

    raise ValueError(f"Unsupported PHY type: {phy_type}")
