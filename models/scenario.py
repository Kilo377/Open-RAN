"""
Scenario configuration and scene-template data models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioConfig:
    """
    Runtime simulator control knobs derived from configs and scene templates.

    `total_slots <= 0` means run continuously until interrupted.
    """

    total_slots: int = 10
    slot_duration_ms: float = 1.0
    cell_id: str = "cell_0"
    xapp_interval_slots: int = 5
    rapp_interval_slots: int = 10
    run_rapps_on_start: bool = False
    orchestration_warmup_slots: int = 0
    scenario_name: str = "baseline_light_load_static"
    scenario_type: str = "umi_street_canyon"

    # Realism knobs.
    obs_delay_slots: int = 0
    config_actuation_delay_slots: int = 0
    command_actuation_delay_slots: int = 0
    control_reject_prob: float = 0.0
    rng_seed: int = 1

    # Measurement pipeline knobs.
    measurement_queue_delay_slots: int = 1
    measurement_sinr_window_slots: int = 8
    measurement_sinr_noise_std_db: float = 2.0
    measurement_bler_window_tbs: int = 20
    measurement_throughput_granularity_slots: int = 10

    # Experiment metadata knobs.
    variant_name: str = "full_method"
    validator_enabled: bool = True
    rl_guidance_enabled: bool = True
    safety_shield_enabled: bool = True
    static_xapp_selection_enabled: bool = False
    reflection_enabled: bool = True
    memory_mode: str = "full"
    selection_mode: str = "reflective"
    repair_mode: str = "deterministic"
    conflict_mode: str = "typed_taxonomy"
    llm_pipeline_mode: str = "schema_bound"
    near_rt_mode: str = "projected_tuning"
    non_rt_reselection_enabled: bool = False
    non_rt_reselection_fallback_threshold: int = 2
    non_rt_reselection_min_gap_slots: int = 10


@dataclass
class TopologyProfile:
    num_cells: int = 1
    cell_radius_m: float = 250.0
    inter_site_distance_m: float = 500.0
    cell_height_m: float = 25.0
    default_ue_height_m: float = 1.5
    default_velocity_mps: float = 0.0
    default_heading_deg: float = 0.0
    geometry_layout: str = "radial"
    indoor_ratio: float = 0.0
    candidate_cell_limit: int = 3
    handover_margin_db: float = 3.0
    handover_delay_slots: int = 2


@dataclass
class RadioProfile:
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    subcarrier_spacing_hz: float = 30e3
    cyclic_prefix_type: str = "normal"
    gnb_num_tx_ant: int = 4
    default_ue_num_rx_ant: int = 2
    gnb_panel_layout: str = "2x2"
    default_ue_panel_layout: str = "1x2"
    polarization: str = "single"


@dataclass
class ChannelProfile:
    scenario_type: str = "umi_street_canyon"
    los_mode: str = "auto"
    default_channel_model_type: str = "tdl_c"
    shadowing_sigma_db_los: float = 4.0
    shadowing_sigma_db_nlos: float = 7.0
    delay_spread_ns_los: float = 120.0
    delay_spread_ns_nlos: float = 300.0
    k_factor_db_los: float = 9.0
    k_factor_db_nlos: float = 0.0
    angle_spread_deg_los: float = 20.0
    angle_spread_deg_nlos: float = 35.0
    spatial_correlation_level_los: float = 0.25
    spatial_correlation_level_nlos: float = 0.55
    o2i_penetration_loss_db: float = 18.0
    fast_fading_seed_offset: int = 1000
    spatial_corr_seed_offset: int = 2000


@dataclass
class TrafficProfile:
    name: str = "mixed_baseline"
    arrival_rate_scale: float = 1.0
    burstiness_scale: float = 1.0


@dataclass
class ControlProfile:
    obs_delay_slots: int = 1
    config_actuation_delay_slots: int = 1
    command_actuation_delay_slots: int = 0
    measurement_queue_delay_slots: int = 1
    measurement_sinr_window_slots: int = 8
    measurement_sinr_noise_std_db: float = 2.0
    measurement_bler_window_tbs: int = 20
    measurement_throughput_granularity_slots: int = 10


@dataclass
class ScenarioTemplate:
    name: str
    description: str
    topology: TopologyProfile = field(default_factory=TopologyProfile)
    radio: RadioProfile = field(default_factory=RadioProfile)
    channel: ChannelProfile = field(default_factory=ChannelProfile)
    traffic: TrafficProfile = field(default_factory=TrafficProfile)
    control: ControlProfile = field(default_factory=ControlProfile)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UeGeometry:
    ue_id: str
    x_m: float
    y_m: float
    z_m: float
    distance_2d_m: float
    distance_3d_m: float
    velocity_mps: float
    heading_deg: float
    is_indoor: bool = False


@dataclass
class GeneratedScenarioBundle:
    scenario_name: str
    scenario_template: ScenarioTemplate
    sim_config: dict[str, Any]
    cell_config: dict[str, Any]
    ue_config: dict[str, Any]
    beamforming_config: dict[str, Any]
    ue_geometries: dict[str, UeGeometry] = field(default_factory=dict)
    scenario_metadata: dict[str, Any] = field(default_factory=dict)
