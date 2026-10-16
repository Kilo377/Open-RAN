"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: simulator.py
Description: This file defines the main slot-driven simulator. The
simulator maintains the global state, advances the simulation clock,
interacts with multiple Non-RT and Near-RT controllers, applies actions
through an executor, synchronizes beamforming state, runs the scheduler,
builds standardized PHY input objects, executes the PHY layer, and
delegates state update work to the state updater.
"""

from models.observation import Observation
from models.action import CommandAction, ConfigAction
from models.oran_lifecycle import E2ControlOutcome, OranLifecycleResources
from models.phy_input import PhyInput, PhyResourceGridSlice, PhyUeInput
from models.phy_result import PhyResult, PhyUeResult
from models.phy_runtime_context import PhyRuntimeContext
from models.scenario import ScenarioConfig
from models.scheduling import SchedulingDecision, SchedulingStageTrace
from core.clock import SimulationClock
from core.metrics import build_observation
from core.debug import DebugConfig, debug_print
from core.rf_interference import (
    build_frequency_interval_hz,
    compute_frequency_coupling_fraction,
    resolve_cell_rf_profile,
)
from core.result_writer import ResultWriter
from core.runtime_views import get_cluster_scheduled_ue_ids
from core.mobility_manager import MobilityManager
from core.state_updater import StateUpdater
from core.measurement_updater import MeasurementUpdater
from traffic.traffic_engine import TrafficEngine
from ran.local_executor import LocalExecutor
from ran.command_consumer import CommandConsumer
from ric.controller_manager import ControllerManager
from ric.interfaces.a1_like import A1LikeInterface
from ric.interfaces.e2_like import E2LikeInterface
from state.antenna_state import AntennaState
from state.beam_state import BeamState
from state.beamforming_state import BeamformingState
from state.candidate_cell_state import CandidateCellState
from state.cell_state import CellState
from state.channel_state import ChannelState
from state.connection_state import ConnectionState
from state.control_state import ControlState
from state.global_state import GlobalState
from state.link_state import LinkState
from state.logical_channel_group_state import LogicalChannelGroupState
from state.logical_channel_state import LogicalChannelState
from state.managed_cell_runtime_state import ManagedCellRuntimeState
from state.propagation_state import LargeScaleLinkState, SmallScaleLinkState
from state.resource_grid_state import ResourceGridState
from state.scheduler_state import SchedulerState
from state.ue_state import UeState
from phy.custom_phy_adapter import CustomPhyAdapter
from phy.mimo_link_model import SingleUserMimoModel
from phy.nr_reference import (
    derive_nr_frequency_range,
    derive_nr_numerology_index,
    lookup_nr_max_prbs,
)

import math
import random
from collections import deque
from types import SimpleNamespace
from typing import Any, Sequence, TypedDict


class NumerologyConfig(TypedDict):
    carrier_frequency_hz: float
    bandwidth_hz: float
    subcarrier_spacing_hz: float
    cyclic_prefix_type: str
    nr_frequency_range: str
    nr_numerology_index: int
    subcarriers_per_prb: int
    n_prbs_total: int
    n_ofdm_symbols_per_slot: int
    fft_size: int
    usable_subcarrier_mask: list[int]
    total_resource_elements: int
    prb_lookup_source: str


def _coerce_float_config(value: object | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


class Simulator:
    """
    Implements the slot-driven simulator main loop.
    INPUT
    config: The scenario configuration object.
    controller_registry: The registry that stores controller instances.
    scheduler: The scheduler object.
    phy_adapter: The PHY adapter object.
    cell_config: The cell configuration dictionary loaded from YAML.
    ue_config: The UE configuration dictionary loaded from YAML.
    beamforming_config: The beamforming configuration dictionary loaded from YAML.
    output_dir: The output directory used for result files.
    OUTPUT
    A simulator object that can run the experiment slot by slot.
    EXAMPLE OUTPUT
    sim = Simulator(config, controller_registry, scheduler, phy_adapter, ...)
    sim.run()
    """

    def __init__(
        self,
        config: ScenarioConfig,
        controller_registry,
        scheduler,
        phy_adapter,
        cell_config: dict,
        ue_config: dict,
        beamforming_config: dict,
        output_dir: str = "outputs",
    ) -> None:
        """
        Initializes the simulator and builds the initial state.
        INPUT
        config: The scenario configuration object.
        controller_registry: The registry that stores controller instances.
        scheduler: The scheduler object.
        phy_adapter: The PHY adapter object.
        cell_config: The cell configuration dictionary.
        ue_config: The UE configuration dictionary.
        beamforming_config: The beamforming configuration dictionary.
        output_dir: The output directory used for result files.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.state is initialized with one cell and several UEs.
        """
        self.config = config
        self.controller_registry = controller_registry
        self.cell_config = cell_config
        self.ue_config = ue_config
        self.beamforming_config = beamforming_config

        self.clock = SimulationClock(slot_duration_ms=config.slot_duration_ms)
        self.executor = LocalExecutor()
        self.command_consumer = CommandConsumer()
        self.scheduler = scheduler
        self.phy_adapter = phy_adapter
        self.shadow_phy_adapter = self._build_shadow_phy_adapter()
        self.shadow_phy_label = "custom_phy" if self.shadow_phy_adapter is not None else None
        self.single_user_mimo_model = SingleUserMimoModel()
        self.mobility_manager = MobilityManager(
            slot_duration_ms=float(getattr(config, "slot_duration_ms", 1.0)),
            rng_seed=int(getattr(config, "rng_seed", 1)),
        )
        self.state_updater = StateUpdater()
        self.measurement_updater = MeasurementUpdater(
            rng_seed=int(getattr(config, "rng_seed", 1)),
            slot_duration_ms=float(getattr(config, "slot_duration_ms", 1.0)),
            queue_delay_slots=int(getattr(config, "measurement_queue_delay_slots", 1)),
            sinr_window_slots=int(getattr(config, "measurement_sinr_window_slots", 8)),
            sinr_noise_std_db=float(getattr(config, "measurement_sinr_noise_std_db", 2.0)),
            bler_window_tbs=int(getattr(config, "measurement_bler_window_tbs", 20)),
            throughput_granularity_slots=int(
                getattr(config, "measurement_throughput_granularity_slots", 10)
            ),
        )

        self.traffic_engine = TrafficEngine(
            ue_config=ue_config,
            slot_duration_ms=float(getattr(config, "slot_duration_ms", 1.0)),
            rng_seed=int(getattr(config, "rng_seed", 1)),
        )
        self.result_writer = ResultWriter(output_dir=output_dir)
        self.slot_records = []
        self.state = self._build_initial_state()
        self._rt_sys_mode = bool(getattr(self.phy_adapter, "cluster_execution", False))
        self.state.rt_tx_contracts = self._rt_sys_mode and getattr(
            getattr(self.phy_adapter, 'abstraction', None), 'feedback_mode', None) == 'nr_direct'
        if self._rt_sys_mode:
            from phy.rt_sys.mobility import RtMobilityManager
            self.traffic_engine.enable_packet_tracking(self.state)
            if self.phy_adapter.config.get('paper_interfaces',{}).get('enabled',False):
                from phy.rt_sys.paper_environment import PaperEnvironment
                self._paper_ue_config=ue_config
                self.state.paper=PaperEnvironment(self,self.phy_adapter.config['paper_interfaces'])
            for ue in self.state.ues:
                ue.rt_feedback_mode = self.phy_adapter.config.get('feedback_mode', 'sys_abstraction')
            self.mobility_manager = RtMobilityManager(self.phy_adapter,
                slot_duration_ms=config.slot_duration_ms, rng_seed=config.rng_seed,
                delay_slots=config.measurement_queue_delay_slots,
                noise_std_db=config.measurement_sinr_noise_std_db)
            self.phy_adapter.bind_state(self.state)
            for runtime in self.state.managed_cells.values():
                runtime.scheduler.enable_spatial_multiplexing=self.scheduler.enable_spatial_multiplexing
                runtime.scheduler.max_mu_users_per_group=self.scheduler.max_mu_users_per_group
                runtime.scheduler.max_total_spatial_layers=self.scheduler.max_total_spatial_layers
            self.mobility_manager.update(self.state)
            if self.phy_adapter.config.get('initial_association') == 'strongest_rt_reference':
                self.mobility_manager.initialize_association(self.state)
            self._refresh_managed_cell_runtime_views(self.state)
            from phy.rt_sys.measurements import reference_measurements
            self.state.rt_reference_measurements = reference_measurements(self.phy_adapter.current_snapshot, self.state,
                self.phy_adapter.config.get('noise_figure_db', 7))
            self.measurement_updater.update(self.state)


        self._rng = random.Random(int(getattr(config, "rng_seed", 1)))
        self._obs_delay_slots = int(getattr(config, "obs_delay_slots", 0))
        self._config_delay_slots = int(getattr(config, "config_actuation_delay_slots", 0))
        self._command_delay_slots = int(getattr(config, "command_actuation_delay_slots", 0))
        self._control_reject_prob = float(getattr(config, "control_reject_prob", 0.0))
        phy_runtime_config = getattr(self.phy_adapter, "sionna_config", None)
        if not isinstance(phy_runtime_config, dict):
            phy_runtime_config = {}
        self._enable_cross_frequency_interference = bool(
            phy_runtime_config.get("enable_cross_frequency_interference", True)
        )
        self._inter_cell_adjacent_channel_leakage_factor = float(
            phy_runtime_config.get(
                "inter_cell_adjacent_channel_leakage_factor",
                phy_runtime_config.get("adjacent_prb_leakage_factor", 0.05),
            )
        )
        self._inter_cell_frequency_guard_decay_scale = max(
            1e-6,
            float(phy_runtime_config.get("inter_cell_frequency_guard_decay_scale", 1.0)),
        )
        self._inter_cell_far_frequency_leakage_floor = max(
            0.0,
            float(phy_runtime_config.get("inter_cell_far_frequency_leakage_floor", 1e-7)),
        )

        self._actuation_seq = 0
        self._pending_config_actuations: list[tuple[int, int, ConfigAction]] = []
        self._pending_command_actuations: list[tuple[int, int, CommandAction]] = []
        self._observation_queue: deque[Observation] = deque()

        e2_interfaces = [E2LikeInterface(xapp) for xapp in controller_registry.xapps]
        a1_interfaces = [A1LikeInterface(rapp) for rapp in controller_registry.rapps]

        self.controller_manager = ControllerManager(
            a1_interfaces=a1_interfaces,
            e2_interfaces=e2_interfaces,
            xapp_interval_slots=config.xapp_interval_slots,
            rapp_interval_slots=config.rapp_interval_slots,
            run_rapps_on_start=bool(getattr(config, "run_rapps_on_start", False)),
            orchestration_warmup_slots=int(getattr(config, "orchestration_warmup_slots", 0) or 0),
            active_intent_manager=getattr(controller_registry, "active_intent_manager", None),
            variant_name=str(getattr(config, "variant_name", "full_method") or "full_method"),
            validator_enabled=bool(getattr(config, "validator_enabled", True)),
            rl_guidance_enabled=bool(getattr(config, "rl_guidance_enabled", True)),
            safety_shield_enabled=bool(getattr(config, "safety_shield_enabled", True)),
            static_xapp_selection_enabled=bool(
                getattr(config, "static_xapp_selection_enabled", False)
            ),
            reflection_enabled=bool(getattr(config, "reflection_enabled", True)),
            memory_mode=str(getattr(config, "memory_mode", "full") or "full"),
            selection_mode=str(getattr(config, "selection_mode", "reflective") or "reflective"),
            repair_mode=str(getattr(config, "repair_mode", "deterministic") or "deterministic"),
            conflict_mode=str(getattr(config, "conflict_mode", "typed_taxonomy") or "typed_taxonomy"),
            llm_pipeline_mode=str(getattr(config, "llm_pipeline_mode", "schema_bound") or "schema_bound"),
            near_rt_mode=str(getattr(config, "near_rt_mode", "projected_tuning") or "projected_tuning"),
            non_rt_reselection_enabled=bool(getattr(config, "non_rt_reselection_enabled", False)),
            non_rt_reselection_fallback_threshold=int(
                getattr(config, "non_rt_reselection_fallback_threshold", 2) or 2
            ),
            non_rt_reselection_min_gap_slots=int(
                getattr(config, "non_rt_reselection_min_gap_slots", 10) or 10
            ),
        )

    def _build_method_config_summary(self) -> dict[str, object]:
        return {
            "variant_name": str(getattr(self.config, "variant_name", "full_method") or "full_method"),
            "validator_enabled": bool(getattr(self.config, "validator_enabled", True)),
            "rl_guidance_enabled": bool(getattr(self.config, "rl_guidance_enabled", True)),
            "safety_shield_enabled": bool(getattr(self.config, "safety_shield_enabled", True)),
            "static_xapp_selection_enabled": bool(
                getattr(self.config, "static_xapp_selection_enabled", False)
            ),
            "reflection_enabled": bool(getattr(self.config, "reflection_enabled", True)),
            "memory_mode": str(getattr(self.config, "memory_mode", "full") or "full"),
            "selection_mode": str(getattr(self.config, "selection_mode", "reflective") or "reflective"),
            "repair_mode": str(getattr(self.config, "repair_mode", "deterministic") or "deterministic"),
            "conflict_mode": str(getattr(self.config, "conflict_mode", "typed_taxonomy") or "typed_taxonomy"),
            "llm_pipeline_mode": str(getattr(self.config, "llm_pipeline_mode", "schema_bound") or "schema_bound"),
            "near_rt_mode": str(getattr(self.config, "near_rt_mode", "projected_tuning") or "projected_tuning"),
            "non_rt_reselection_enabled": bool(getattr(self.config, "non_rt_reselection_enabled", False)),
            "seed": int(getattr(self.config, "rng_seed", 1) or 1),
            "scenario_name": str(getattr(self.config, "scenario_name", "") or ""),
        }

    def _build_orchestration_epoch_history_summary(
        self,
        orchestration_epoch_history: list[dict[str, object]],
    ) -> dict[str, object]:
        if len(orchestration_epoch_history) == 0:
            return {
                "epoch_count": 0,
                "active_intent_ids": [],
                "translation_refresh_count": 0,
                "last_epoch": {},
            }
        active_intent_ids = sorted(
            {
                str(entry.get("active_intent_id", "") or "")
                for entry in orchestration_epoch_history
                if str(entry.get("active_intent_id", "") or "").strip() != ""
            }
        )
        translation_refresh_count = int(
            sum(
                1
                for entry in orchestration_epoch_history
                if bool(entry.get("translation_refreshed_this_epoch", False))
            )
        )
        forced_reselection_count = int(
            sum(
                1
                for entry in orchestration_epoch_history
                if bool(entry.get("forced_non_rt_reselection", False))
            )
        )
        return {
            "epoch_count": int(len(orchestration_epoch_history)),
            "active_intent_ids": active_intent_ids,
            "translation_refresh_count": translation_refresh_count,
            "forced_non_rt_reselection_count": forced_reselection_count,
            "last_epoch": dict(orchestration_epoch_history[-1]),
        }

    def _build_shadow_phy_adapter(self) -> CustomPhyAdapter | None:
        sionna_config = getattr(self.phy_adapter, "sionna_config", None)
        if not isinstance(sionna_config, dict):
            return None

        if not bool(sionna_config.get("shadow_compare_enabled", False)):
            return None

        raw_codebooks = self.beamforming_config.get("codebooks", {})
        raw_profiles = self.beamforming_config.get("beam_profiles", {})
        codebook_map = {int(key): value for key, value in raw_codebooks.items()}
        beam_profile_map = {int(key): value for key, value in raw_profiles.items()}
        return CustomPhyAdapter(
            codebook_map=codebook_map,
            beam_profile_map=beam_profile_map,
            noise_floor_dbm=float(sionna_config.get("shadow_compare_noise_floor_dbm", -94.0)),
            base_pathloss_compensation_db=float(
                sionna_config.get("shadow_compare_base_pathloss_compensation_db", 98.0)
            ),
            bler_target=float(sionna_config.get("shadow_compare_bler_target", 0.1)),
            max_mcs_index=int(sionna_config.get("shadow_compare_max_mcs_index", 15)),
        )

    def _derive_beam_gain_from_config(
        self,
        *,
        codebook_id: int,
        beam_profile_id: int,
        serving_beam_id: int,
        codebook_map: dict[int, dict],
        beam_profile_map: dict[int, dict],
    ) -> float:
        codebook_entry = codebook_map.get(int(codebook_id), {})
        beam_profile_entry = beam_profile_map.get(int(beam_profile_id), {})
        base_gain_db = float(codebook_entry.get("base_gain_db", 0.0))
        beam_gain_db = float(codebook_entry.get("beams", {}).get(int(serving_beam_id), 0.0))
        profile_gain_db = float(beam_profile_entry.get("profile_gain_db", 0.0))
        return base_gain_db + beam_gain_db + profile_gain_db

    def _get_single_user_mimo_model(self) -> SingleUserMimoModel:
        mimo_model = getattr(self, "single_user_mimo_model", None)
        if mimo_model is None:
            mimo_model = SingleUserMimoModel()
            self.single_user_mimo_model = mimo_model
        return mimo_model

    def _next_power_of_two(self, value: int) -> int:
        normalized_value = max(1, int(value))
        return 1 << (normalized_value - 1).bit_length()

    def _build_usable_subcarrier_mask(
        self,
        *,
        fft_size: int,
        usable_subcarrier_count: int,
    ) -> list[int]:
        normalized_fft_size = max(1, int(fft_size))
        normalized_usable_count = max(0, min(int(usable_subcarrier_count), normalized_fft_size))
        mask = [0 for _ in range(normalized_fft_size)]
        if normalized_usable_count <= 0:
            return mask

        start_index = max(0, (normalized_fft_size - normalized_usable_count) // 2)
        end_index = min(normalized_fft_size, start_index + normalized_usable_count)
        for index in range(start_index, end_index):
            mask[index] = 1
        return mask

    def _build_phy_resource_grid_slice(
        self,
        *,
        grant: object | None,
    ) -> PhyResourceGridSlice:
        resource_grid = getattr(self.state, "resource_grid", None)
        subcarriers_per_prb = int(getattr(resource_grid, "n_subcarriers_per_prb", 12))
        total_prbs = int(getattr(resource_grid, "n_prbs_total", 0))
        total_symbols = int(getattr(resource_grid, "n_ofdm_symbols_per_slot", 0))

        if grant is None:
            return PhyResourceGridSlice()

        start_prb = max(0, int(getattr(grant, "start_prb", 0)))
        allocated_prbs = max(0, int(getattr(grant, "allocated_prbs", 0)))
        start_symbol = max(0, int(getattr(grant, "start_symbol", 0)))
        allocated_symbols = max(0, int(getattr(grant, "allocated_symbols", 0)))

        resource_block_indices = list(getattr(grant, "resource_block_indices", []) or [])
        if not resource_block_indices:
            end_prb = min(total_prbs if total_prbs > 0 else start_prb + allocated_prbs, start_prb + allocated_prbs)
            resource_block_indices = list(range(start_prb, max(start_prb, end_prb)))

        allocated_symbol_indices = list(getattr(grant, "ofdm_symbol_indices", []) or [])
        if not allocated_symbol_indices:
            end_symbol = min(
                total_symbols if total_symbols > 0 else start_symbol + allocated_symbols,
                start_symbol + allocated_symbols,
            )
            allocated_symbol_indices = list(range(start_symbol, max(start_symbol, end_symbol)))

        allocated_subcarrier_indices: list[int] = []
        for prb_index in resource_block_indices:
            base_subcarrier_index = int(prb_index) * subcarriers_per_prb
            allocated_subcarrier_indices.extend(
                range(base_subcarrier_index, base_subcarrier_index + subcarriers_per_prb)
            )

        resource_elements = int(len(allocated_subcarrier_indices) * len(allocated_symbol_indices))
        return PhyResourceGridSlice(
            start_prb=start_prb,
            allocated_prbs=allocated_prbs,
            start_symbol=start_symbol,
            allocated_symbols=allocated_symbols,
            resource_block_indices=resource_block_indices,
            allocated_subcarrier_indices=allocated_subcarrier_indices,
            allocated_symbol_indices=allocated_symbol_indices,
            dmrs_pattern_id=getattr(grant, "dmrs_pattern_id", None),
            resource_elements=resource_elements,
        )

    def _derive_numerology_config(
        self,
        *,
        carrier_frequency_hz: float,
        bandwidth_hz: float,
        subcarrier_spacing_hz: float,
        cyclic_prefix_type: str,
    ) -> NumerologyConfig:
        normalized_carrier_frequency_hz = max(1.0, float(carrier_frequency_hz))
        normalized_bandwidth_hz = max(1.0, float(bandwidth_hz))
        normalized_scs_hz = max(1.0, float(subcarrier_spacing_hz))
        normalized_cp = str(cyclic_prefix_type).strip().lower() or "normal"
        nr_frequency_range = derive_nr_frequency_range(normalized_carrier_frequency_hz)
        nr_numerology_index = derive_nr_numerology_index(normalized_scs_hz)

        subcarriers_per_prb = 12
        symbols_per_nr_slot = 12 if normalized_cp == "extended" else 14
        nr_slot_duration_ms = 1.0 * (15e3 / normalized_scs_hz)
        nr_slots_per_sim_slot = max(
            1,
            int(round(float(self.config.slot_duration_ms) / max(nr_slot_duration_ms, 1e-6))),
        )
        n_ofdm_symbols_per_slot = int(symbols_per_nr_slot * nr_slots_per_sim_slot)

        n_prbs_total, prb_lookup_source = lookup_nr_max_prbs(
            bandwidth_hz=normalized_bandwidth_hz,
            subcarrier_spacing_hz=normalized_scs_hz,
            carrier_frequency_hz=normalized_carrier_frequency_hz,
        )
        usable_subcarrier_count = int(n_prbs_total * subcarriers_per_prb)
        fft_size = self._next_power_of_two(int(math.ceil(float(usable_subcarrier_count) * 1.25)))
        usable_subcarrier_mask = self._build_usable_subcarrier_mask(
            fft_size=fft_size,
            usable_subcarrier_count=usable_subcarrier_count,
        )
        total_resource_elements = int(usable_subcarrier_count * n_ofdm_symbols_per_slot)

        return {
            "carrier_frequency_hz": normalized_carrier_frequency_hz,
            "bandwidth_hz": normalized_bandwidth_hz,
            "subcarrier_spacing_hz": normalized_scs_hz,
            "cyclic_prefix_type": normalized_cp,
            "nr_frequency_range": str(nr_frequency_range),
            "nr_numerology_index": int(nr_numerology_index),
            "subcarriers_per_prb": int(subcarriers_per_prb),
            "n_prbs_total": int(n_prbs_total),
            "n_ofdm_symbols_per_slot": int(n_ofdm_symbols_per_slot),
            "fft_size": int(fft_size),
            "usable_subcarrier_mask": usable_subcarrier_mask,
            "total_resource_elements": int(total_resource_elements),
            "prb_lookup_source": str(prb_lookup_source),
        }

    def _sync_scheduler_budget_with_numerology(
        self,
        numerology: NumerologyConfig,
    ) -> None:
        if hasattr(self.scheduler, "total_prbs_per_slot"):
            self.scheduler.total_prbs_per_slot = int(numerology["n_prbs_total"])
        if hasattr(self.scheduler, "total_symbols_per_slot"):
            self.scheduler.total_symbols_per_slot = int(numerology["n_ofdm_symbols_per_slot"])

    def _build_resource_grid_state_from_numerology(
        self,
        numerology: NumerologyConfig,
    ) -> ResourceGridState:
        return ResourceGridState(
            total_prbs=int(numerology["n_prbs_total"]),
            total_symbols=int(numerology["n_ofdm_symbols_per_slot"]),
            subcarriers_per_prb=int(numerology["subcarriers_per_prb"]),
            n_prbs_total=int(numerology["n_prbs_total"]),
            n_subcarriers_per_prb=int(numerology["subcarriers_per_prb"]),
            n_ofdm_symbols_per_slot=int(numerology["n_ofdm_symbols_per_slot"]),
            fft_size=int(numerology["fft_size"]),
            usable_subcarrier_mask=list(numerology["usable_subcarrier_mask"]),
            carrier_frequency_hz=float(numerology["carrier_frequency_hz"]),
            bandwidth_hz=float(numerology["bandwidth_hz"]),
            subcarrier_spacing_hz=float(numerology["subcarrier_spacing_hz"]),
            cyclic_prefix_type=str(numerology["cyclic_prefix_type"]),
            nr_frequency_range=str(numerology["nr_frequency_range"]),
            nr_numerology_index=int(numerology["nr_numerology_index"]),
            total_resource_elements=int(numerology["total_resource_elements"]),
            allocated_resource_elements=0,
            utilization=0.0,
            overlap_fraction_mean=0.0,
            interference_penalty_db_mean=0.0,
            scheduled_ue_count=0,
            scheduled_layer_count_total=0,
            mu_mimo_overlap_pair_count=0,
            allocations=[],
        )

    def _build_managed_cell_runtime_states(
        self,
        *,
        primary_cell: CellState,
        primary_scheduler: SchedulerState,
        primary_channel: ChannelState,
        primary_beam: BeamState,
        primary_resource_grid: ResourceGridState,
    ) -> dict[str, ManagedCellRuntimeState]:
        managed_cells: dict[str, ManagedCellRuntimeState] = {}
        cell_radio_profiles = dict(primary_cell.cell_radio_profiles or {})
        primary_cell_id = str(primary_cell.cell_id)
        managed_cell_ids = list(primary_cell.managed_cell_ids or [primary_cell_id])
        if primary_cell_id not in managed_cell_ids:
            managed_cell_ids.insert(0, primary_cell_id)

        primary_profile = resolve_cell_rf_profile(
            dict(cell_radio_profiles.get(primary_cell_id, {}) or {})
        )
        managed_cells[primary_cell_id] = ManagedCellRuntimeState(
            cell_id=primary_cell_id,
            cell=primary_cell,
            scheduler=primary_scheduler,
            channel=primary_channel,
            beam=primary_beam,
            resource_grid=primary_resource_grid,
            radio_profile=primary_profile,
            is_primary=True,
        )

        for managed_cell_id in managed_cell_ids:
            if str(managed_cell_id) == primary_cell_id:
                continue
            radio_profile = resolve_cell_rf_profile(
                dict(cell_radio_profiles.get(str(managed_cell_id), {}) or {})
            )
            antenna_defaults = dict(self.beamforming_config.get("antenna", {}) or {})
            carrier_frequency_hz_value = _coerce_float_config(
                radio_profile.get(
                    "carrier_frequency_hz",
                    antenna_defaults.get("carrier_frequency_hz", 3.5e9),
                ),
                3.5e9,
            )
            bandwidth_hz_value = _coerce_float_config(
                radio_profile.get(
                    "bandwidth_hz",
                    antenna_defaults.get("bandwidth_hz", 20e6),
                ),
                20e6,
            )
            subcarrier_spacing_hz_value = _coerce_float_config(
                radio_profile.get(
                    "subcarrier_spacing_hz",
                    antenna_defaults.get("subcarrier_spacing_hz", 30e3),
                ),
                30e3,
            )
            cyclic_prefix_type_value = str(
                radio_profile.get(
                    "cyclic_prefix_type",
                    antenna_defaults.get("cyclic_prefix_type", "normal"),
                )
            )
            tx_power_dbm_value = _coerce_float_config(
                radio_profile.get("tx_power_dbm", primary_cell.tx_power_dbm),
                float(primary_cell.tx_power_dbm),
            )
            numerology = self._derive_numerology_config(
                carrier_frequency_hz=carrier_frequency_hz_value,
                bandwidth_hz=bandwidth_hz_value,
                subcarrier_spacing_hz=subcarrier_spacing_hz_value,
                cyclic_prefix_type=cyclic_prefix_type_value,
            )
            runtime_cell = CellState(
                cell_id=str(managed_cell_id),
                tx_power_dbm=tx_power_dbm_value,
                current_codebook_id=int(primary_cell.current_codebook_id),
                current_beam_profile_id=int(primary_cell.current_beam_profile_id),
                beam_candidate_preference=str(primary_cell.beam_candidate_preference),
                power_cap_dbm=primary_cell.power_cap_dbm,
                traffic_steering_preference=str(primary_cell.traffic_steering_preference),
                mobility_bias_db=float(primary_cell.mobility_bias_db),
                handover_margin_db=float(primary_cell.handover_margin_db),
                handover_delay_slots=int(primary_cell.handover_delay_slots),
                handover_cost_db=float(primary_cell.handover_cost_db),
                handover_interruption_slots=int(primary_cell.handover_interruption_slots),
                steering_hysteresis_db=float(primary_cell.steering_hysteresis_db),
                measurement_event_trigger=str(primary_cell.measurement_event_trigger),
                measurement_event_offset_db=float(primary_cell.measurement_event_offset_db),
                managed_cell_ids=list(managed_cell_ids),
                neighbor_cell_positions=dict(primary_cell.neighbor_cell_positions),
                cell_radio_profiles=dict(primary_cell.cell_radio_profiles),
            )
            runtime_scheduler = SchedulerState(
                scheduler_mode=str(primary_scheduler.scheduler_mode),
                packing_strategy=str(primary_scheduler.packing_strategy),
                enable_spatial_multiplexing=bool(primary_scheduler.enable_spatial_multiplexing),
                max_mu_users_per_group=int(primary_scheduler.max_mu_users_per_group),
                min_mu_beam_separation_deg=float(primary_scheduler.min_mu_beam_separation_deg),
                max_total_spatial_layers=primary_scheduler.max_total_spatial_layers,
                scheduler_policy_hint=str(primary_scheduler.scheduler_policy_hint),
                latency_bias=primary_scheduler.latency_bias,
                slice_weight_adjustment=dict(primary_scheduler.slice_weight_adjustment),
            )
            runtime_channel = ChannelState(
                pathloss_db=float(primary_channel.pathloss_db),
                shadowing_db=float(primary_channel.shadowing_db),
                channel_profile=str(
                    radio_profile.get("default_channel_model_type", primary_channel.channel_profile)
                ),
                doppler_hz=float(primary_channel.doppler_hz),
            )
            runtime_beam = BeamState(
                codebook_id=int(runtime_cell.current_codebook_id),
                beam_profile_id=int(runtime_cell.current_beam_profile_id),
                selected_beam_id=0,
                beam_gain_db=0.0,
                codebook_map=dict(primary_beam.codebook_map),
                beam_profile_map=dict(primary_beam.beam_profile_map),
            )
            managed_cells[str(managed_cell_id)] = ManagedCellRuntimeState(
                cell_id=str(managed_cell_id),
                cell=runtime_cell,
                scheduler=runtime_scheduler,
                channel=runtime_channel,
                beam=runtime_beam,
                resource_grid=self._build_resource_grid_state_from_numerology(numerology),
                radio_profile=radio_profile,
                is_primary=False,
            )
        return managed_cells

    def _refresh_managed_cell_runtime_views(self, state: GlobalState) -> None:
        managed_cells = getattr(state, "managed_cells", {})
        if not isinstance(managed_cells, dict) or len(managed_cells) == 0:
            return

        for cell_id, runtime_state in managed_cells.items():
            runtime_state.is_primary = str(cell_id) == str(getattr(state.cell, "cell_id", ""))
            runtime_state.active_ue_ids = []
            runtime_state.scheduled_ue_ids = list(
                getattr(runtime_state.scheduler, "scheduled_ue_ids", []) or []
            )
            runtime_state.beam.codebook_id = int(getattr(runtime_state.cell, "current_codebook_id", 0))
            runtime_state.beam.beam_profile_id = int(getattr(runtime_state.cell, "current_beam_profile_id", 0))

        for ue in getattr(state, "ues", []):
            ue_id = str(getattr(ue, "ue_id", ""))
            connection = getattr(state, "connections", {}).get(ue_id, None)
            serving_cell_id = str(
                getattr(connection, "serving_cell_id", getattr(state.cell, "cell_id", ""))
            )
            runtime_state = managed_cells.get(serving_cell_id)
            if runtime_state is not None:
                runtime_state.active_ue_ids.append(ue_id)

        primary_cell_runtime = managed_cells.get(str(getattr(state.cell, "cell_id", "")))
        if primary_cell_runtime is not None:
            primary_cell_runtime.scheduled_ue_ids = list(
                getattr(state.scheduler, "scheduled_ue_ids", []) or []
            )
        self._refresh_beamforming_views_by_serving_cell(state)

    def _build_runtime_antenna_state(
        self,
        state: GlobalState | SimpleNamespace,
        runtime_state: ManagedCellRuntimeState,
    ) -> AntennaState:
        base_antenna = getattr(state, "antenna", None)
        radio_profile = dict(getattr(runtime_state, "radio_profile", {}) or {})
        resource_grid = getattr(runtime_state, "resource_grid", None)
        return AntennaState(
            gnb_num_tx_ant=int(
                radio_profile.get(
                    "gnb_num_tx_ant",
                    getattr(base_antenna, "gnb_num_tx_ant", 4),
                )
            ),
            ue_num_rx_ant_by_ue=dict(getattr(base_antenna, "ue_num_rx_ant_by_ue", {}) or {}),
            carrier_frequency_hz=float(
                radio_profile.get(
                    "carrier_frequency_hz",
                    getattr(resource_grid, "carrier_frequency_hz", getattr(base_antenna, "carrier_frequency_hz", 3.5e9)),
                )
            ),
            bandwidth_hz=float(
                radio_profile.get(
                    "bandwidth_hz",
                    getattr(resource_grid, "bandwidth_hz", getattr(base_antenna, "bandwidth_hz", 20e6)),
                )
            ),
            subcarrier_spacing_hz=float(
                radio_profile.get(
                    "subcarrier_spacing_hz",
                    getattr(resource_grid, "subcarrier_spacing_hz", getattr(base_antenna, "subcarrier_spacing_hz", 30e3)),
                )
            ),
            cyclic_prefix_type=str(
                radio_profile.get(
                    "cyclic_prefix_type",
                    getattr(resource_grid, "cyclic_prefix_type", getattr(base_antenna, "cyclic_prefix_type", "normal")),
                )
            ),
            num_ofdm_symbols_per_slot=int(
                getattr(
                    resource_grid,
                    "n_ofdm_symbols_per_slot",
                    getattr(base_antenna, "num_ofdm_symbols_per_slot", 14),
                )
            ),
            gnb_panel_layout=str(
                radio_profile.get(
                    "gnb_panel_layout",
                    getattr(base_antenna, "gnb_panel_layout", "2x2"),
                )
            ),
            ue_panel_layout_by_ue=dict(
                getattr(base_antenna, "ue_panel_layout_by_ue", {}) or {}
            ),
            polarization=str(
                radio_profile.get(
                    "polarization",
                    getattr(base_antenna, "polarization", "single"),
                )
            ),
        )

    def _refresh_beamforming_views_by_serving_cell(self, state: GlobalState) -> None:
        managed_cells = getattr(state, "managed_cells", {})
        if not isinstance(managed_cells, dict) or len(managed_cells) == 0:
            return

        beamforming_map = getattr(state, "beamforming", {})
        if not isinstance(beamforming_map, dict):
            return

        for ue in getattr(state, "ues", []):
            ue_id = str(getattr(ue, "ue_id", ""))
            if ue_id == "":
                continue
            connection = getattr(state, "connections", {}).get(ue_id, None)
            serving_cell_id = str(
                getattr(connection, "serving_cell_id", getattr(state.cell, "cell_id", ""))
            )
            runtime_state = managed_cells.get(serving_cell_id)
            if runtime_state is None:
                runtime_state = managed_cells.get(str(getattr(state.cell, "cell_id", "")))
            if runtime_state is None:
                continue
            antenna_state = self._build_runtime_antenna_state(state, runtime_state)
            previous_state = beamforming_map.get(ue_id)
            updated_state = self._derive_beamforming_state_for_ue(
                ue=ue,
                cell=runtime_state.cell,
                antenna_state=antenna_state,
                codebook_map=dict(getattr(runtime_state.beam, "codebook_map", {}) or {}),
                beam_profile_map=dict(getattr(runtime_state.beam, "beam_profile_map", {}) or {}),
            )
            if previous_state is not None:
                updated_state.precoder_match_score = float(
                    getattr(previous_state, "precoder_match_score", updated_state.precoder_match_score)
                )
                updated_state.post_eq_sinr_db = _coerce_float_config(
                    getattr(previous_state, "post_eq_sinr_db", updated_state.post_eq_sinr_db),
                    _coerce_float_config(updated_state.post_eq_sinr_db, 0.0),
                )
            beamforming_map[ue_id] = updated_state

    def _ordered_runtime_states(
        self,
        state: GlobalState,
    ) -> list[ManagedCellRuntimeState]:
        managed_cells = getattr(state, "managed_cells", {})
        if not isinstance(managed_cells, dict):
            return []
        return [
            runtime_state
            for _, runtime_state in sorted(
                managed_cells.items(),
                key=lambda item: (
                    0 if bool(getattr(item[1], "is_primary", False)) else 1,
                    str(item[0]),
                ),
            )
        ]

    def _build_cell_scoped_state(
        self,
        state: GlobalState,
        runtime_state: ManagedCellRuntimeState,
    ) -> SimpleNamespace:
        active_ue_id_set = {
            str(ue_id) for ue_id in list(getattr(runtime_state, "active_ue_ids", []) or [])
        }
        scoped_ues = [
            ue for ue in getattr(state, "ues", []) if str(getattr(ue, "ue_id", "")) in active_ue_id_set
        ]
        scoped_connections = {
            ue_id: connection
            for ue_id, connection in dict(getattr(state, "connections", {}) or {}).items()
            if str(ue_id) in active_ue_id_set
        }
        scoped_links = {
            ue_id: link
            for ue_id, link in dict(getattr(state, "links", {}) or {}).items()
            if str(ue_id) in active_ue_id_set
        }
        scoped_beamforming = {
            ue_id: beamforming
            for ue_id, beamforming in dict(getattr(state, "beamforming", {}) or {}).items()
            if str(ue_id) in active_ue_id_set
        }
        return SimpleNamespace(
            slot_id=int(getattr(state, "slot_id", -1)),
            sim_time_ms=float(getattr(state, "sim_time_ms", 0.0)),
            cell=runtime_state.cell,
            scheduler=runtime_state.scheduler,
            channel=runtime_state.channel,
            beam=runtime_state.beam,
            antenna=self._build_runtime_antenna_state(state, runtime_state),
            resource_grid=runtime_state.resource_grid,
            ues=scoped_ues,
            control=getattr(state, "control", None),
            measurement=getattr(state, "measurement", None),
            connections=scoped_connections,
            links=scoped_links,
            beamforming=scoped_beamforming,
            runtime_mode=str(getattr(state, "runtime_mode", "single_active_cell")),
            managed_cells=getattr(state, "managed_cells", {}),
            paper=getattr(state,'paper',None),
            rt_du_observation=getattr(state,'rt_du_observation',None),
            rt_tx_contracts=self._rt_sys_mode and getattr(self.phy_adapter.abstraction,'feedback_mode',None)=='nr_direct',
            rt_channel_snapshot=self.phy_adapter.scheduler_snapshot if self._rt_sys_mode else None,
            rt_harq_combining=self.phy_adapter.config.get('harq_combining','none') if self._rt_sys_mode else 'none',
        )

    def _aggregate_cluster_stage_trace(
        self,
        per_cell_stage_trace: dict[str, SchedulingStageTrace],
    ) -> SchedulingStageTrace:
        cluster_stage_trace = SchedulingStageTrace(demand_source="multi_cell_cluster")
        cluster_stage_trace.per_cell_stage_trace = {}
        for cell_id, stage_trace in per_cell_stage_trace.items():
            cluster_stage_trace.demand_source = getattr(
                stage_trace,
                "demand_source",
                cluster_stage_trace.demand_source,
            )
            for field_name in (
                "feasible_candidate_ue_ids",
                "filtered_out_ue_ids",
                "retx_candidate_ue_ids",
                "new_tx_candidate_ue_ids",
                "selected_retx_ue_ids",
                "selected_new_tx_ue_ids",
                "grant_candidate_ue_ids",
                "granted_ue_ids",
            ):
                current_values = list(getattr(cluster_stage_trace, field_name, []) or [])
                new_values = list(getattr(stage_trace, field_name, []) or [])
                setattr(
                    cluster_stage_trace,
                    field_name,
                    list(dict.fromkeys(current_values + [str(item) for item in new_values])),
                )
            cluster_stage_trace.filtered_out_reasons.update(
                dict(getattr(stage_trace, "filtered_out_reasons", {}) or {})
            )
            cluster_stage_trace.per_cell_stage_trace[str(cell_id)] = {
                "feasible_candidate_count": int(
                    len(getattr(stage_trace, "feasible_candidate_ue_ids", []) or [])
                ),
                "selected_retx_count": int(
                    len(getattr(stage_trace, "selected_retx_ue_ids", []) or [])
                ),
                "selected_new_tx_count": int(
                    len(getattr(stage_trace, "selected_new_tx_ue_ids", []) or [])
                ),
                "granted_ue_count": int(len(getattr(stage_trace, "granted_ue_ids", []) or [])),
            }
        return cluster_stage_trace

    def _store_cluster_scheduler_summary(self, state: GlobalState) -> None:
        runtime_states = self._ordered_runtime_states(state)
        cluster_grants = []
        cluster_demands = []
        cluster_scheduled_ue_ids: list[str] = []
        per_cell_stage_trace: dict[str, SchedulingStageTrace] = {}
        total_prbs = 0
        total_symbols = 0
        unallocated_prbs = 0
        unallocated_symbols = 0

        for runtime_state in runtime_states:
            runtime_scheduler = getattr(runtime_state, "scheduler", None)
            runtime_decision = getattr(runtime_scheduler, "last_decision", None)
            runtime_grants = list(getattr(runtime_scheduler, "grants", []) or [])
            runtime_demands = list(getattr(runtime_scheduler, "last_demands", []) or [])
            runtime_stage_trace = getattr(runtime_scheduler, "last_stage_trace", None)
            cluster_grants.extend(runtime_grants)
            cluster_demands.extend(runtime_demands)
            cluster_scheduled_ue_ids.extend(
                [str(ue_id) for ue_id in list(getattr(runtime_scheduler, "scheduled_ue_ids", []) or [])]
            )
            if isinstance(runtime_stage_trace, SchedulingStageTrace):
                per_cell_stage_trace[str(runtime_state.cell_id)] = runtime_stage_trace
            total_prbs += int(getattr(runtime_decision, "total_prbs", getattr(runtime_state.resource_grid, "n_prbs_total", 0)))
            total_symbols = max(
                total_symbols,
                int(
                    getattr(
                        runtime_decision,
                        "total_symbols",
                        getattr(runtime_state.resource_grid, "n_ofdm_symbols_per_slot", 0),
                    )
                ),
            )
            unallocated_prbs += int(getattr(runtime_decision, "unallocated_prbs", 0))
            unallocated_symbols = max(
                unallocated_symbols,
                int(getattr(runtime_decision, "unallocated_symbols", 0)),
            )

        cluster_stage_trace = self._aggregate_cluster_stage_trace(per_cell_stage_trace)
        state.cluster_scheduled_ue_ids = list(dict.fromkeys(cluster_scheduled_ue_ids))
        state.cluster_grants = list(cluster_grants)
        state.cluster_last_demands = list(cluster_demands)
        state.cluster_last_stage_trace = cluster_stage_trace
        state.cluster_last_decision = SchedulingDecision(
            slot_id=int(getattr(state, "slot_id", -1)),
            scheduler_mode="multi_cell_cluster" if len(runtime_states) > 1 else str(getattr(state.scheduler, "scheduler_mode", "round_robin")),
            total_prbs=int(total_prbs),
            total_symbols=int(total_symbols),
            grants=list(cluster_grants),
            demands=list(cluster_demands),
            stage_trace=cluster_stage_trace,
            unallocated_prbs=int(unallocated_prbs),
            unallocated_symbols=int(unallocated_symbols),
        )

    def _combine_phy_results(
        self,
        *,
        slot_id: int,
        phy_results: Sequence[PhyResult],
        combined_cell_id: str,
    ) -> PhyResult:
        ue_results: list[PhyUeResult] = []
        for phy_result in phy_results:
            ue_results.extend(list(getattr(phy_result, "ue_results", []) or []))
        scheduled_results = [
            ue_result
            for ue_result in ue_results
            if bool(getattr(ue_result, "is_scheduled", False))
        ]
        if len(scheduled_results) > 0:
            avg_sinr_db = float(
                sum(float(getattr(item, "sinr_db", 0.0)) for item in scheduled_results)
                / float(len(scheduled_results))
            )
            avg_bler = float(
                sum(float(getattr(item, "bler", 0.0)) for item in scheduled_results)
                / float(len(scheduled_results))
            )
            total_tb_size_bytes = int(
                sum(int(getattr(item, "tb_size_bytes", 0)) for item in scheduled_results)
            )
        else:
            avg_sinr_db = 0.0
            avg_bler = 0.0
            total_tb_size_bytes = 0
        return PhyResult(
            slot_id=int(slot_id),
            cell_id=str(combined_cell_id),
            ue_results=ue_results,
            avg_sinr_db=avg_sinr_db,
            avg_bler=avg_bler,
            total_tb_size_bytes=int(total_tb_size_bytes),
        )

    def _dbm_to_mw(self, dbm_value: float) -> float:
        clamped_dbm = max(-200.0, min(120.0, float(dbm_value)))
        return 10.0 ** (clamped_dbm / 10.0)

    def _mw_to_dbm(self, mw_value: float) -> float:
        if float(mw_value) <= 0.0:
            return -120.0
        return 10.0 * math.log10(float(mw_value))

    def _compute_interval_overlap_fraction(
        self,
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> float:
        width_a = max(0.0, float(end_a) - float(start_a))
        if width_a <= 0.0:
            return 0.0
        overlap = max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))
        return max(0.0, min(1.0, overlap / width_a))

    def _compute_interval_guard_band(
        self,
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> float:
        if float(end_a) < float(start_b):
            return max(0.0, float(start_b) - float(end_a))
        if float(end_b) < float(start_a):
            return max(0.0, float(start_a) - float(end_b))
        return 0.0

    def _compute_frequency_coupling_fraction(
        self,
        *,
        target_interval_hz: tuple[float, float],
        interferer_interval_hz: tuple[float, float],
        target_rf_profile: dict[str, Any] | None,
        interferer_rf_profile: dict[str, Any] | None,
    ) -> tuple[float, float, float, float]:
        return compute_frequency_coupling_fraction(
            target_interval_hz=target_interval_hz,
            interferer_interval_hz=interferer_interval_hz,
            target_rf_profile=target_rf_profile,
            interferer_rf_profile=interferer_rf_profile,
            enable_cross_frequency_interference=self._enable_cross_frequency_interference,
            far_frequency_leakage_floor=self._inter_cell_far_frequency_leakage_floor,
            guard_decay_scale=self._inter_cell_frequency_guard_decay_scale,
        )

    def _compute_beam_coupling_factor(
        self,
        *,
        target_candidate_beam_id: int | None,
        interferer_beam_id: int | None,
    ) -> float:
        if target_candidate_beam_id is None or interferer_beam_id is None:
            return 0.35
        normalized_target = int(target_candidate_beam_id) % 8
        normalized_interferer = int(interferer_beam_id) % 8
        beam_gap = abs(normalized_target - normalized_interferer)
        beam_gap = min(beam_gap, 8 - beam_gap)
        return max(0.2, 1.0 - 0.2 * float(beam_gap))

    def _build_phy_grant_frequency_interval_hz(
        self,
        *,
        phy_input: PhyInput,
        ue_input: PhyUeInput,
    ) -> tuple[float, float]:
        runtime_context = getattr(phy_input, "runtime_context", None)
        center_frequency_hz = float(
            getattr(runtime_context, "carrier_frequency_hz", 3.5e9) if runtime_context is not None else 3.5e9
        )
        bandwidth_hz = max(
            1.0,
            float(getattr(runtime_context, "bandwidth_hz", 20e6) if runtime_context is not None else 20e6),
        )
        total_prbs = max(
            1,
            int(getattr(runtime_context, "n_prbs_total", 0) if runtime_context is not None else 0)
            or int(getattr(ue_input, "allocated_prbs", 0))
            or 1,
        )
        resource_block_indices = list(getattr(ue_input, "resource_block_indices", []) or [])
        if len(resource_block_indices) > 0:
            start_prb = min(int(item) for item in resource_block_indices)
            end_prb = max(int(item) for item in resource_block_indices) + 1
        else:
            start_prb = max(0, int(getattr(ue_input, "start_prb", 0)))
            end_prb = start_prb + max(0, int(getattr(ue_input, "allocated_prbs", 0)))
        if end_prb <= start_prb:
            return center_frequency_hz, center_frequency_hz
        prb_bandwidth_hz = bandwidth_hz / float(total_prbs)
        lower_hz = center_frequency_hz - 0.5 * bandwidth_hz + float(start_prb) * prb_bandwidth_hz
        upper_hz = center_frequency_hz - 0.5 * bandwidth_hz + float(end_prb) * prb_bandwidth_hz
        return build_frequency_interval_hz(
            carrier_frequency_hz=float(0.5 * (lower_hz + upper_hz)),
            bandwidth_hz=float(max(1.0, upper_hz - lower_hz)),
        )

    def _build_phy_grant_time_interval_ms(
        self,
        *,
        phy_input: PhyInput,
        ue_input: PhyUeInput,
    ) -> tuple[float, float]:
        runtime_context = getattr(phy_input, "runtime_context", None)
        slot_duration_ms = max(
            1e-6,
            float(
                getattr(runtime_context, "slot_duration_ms", self.config.slot_duration_ms)
                if runtime_context is not None
                else self.config.slot_duration_ms
            ),
        )
        total_symbols = max(
            1,
            int(
                getattr(runtime_context, "num_ofdm_symbols_per_slot", 0)
                if runtime_context is not None
                else 0
            )
            or int(getattr(ue_input, "allocated_symbols", 0))
            or 1,
        )
        allocated_symbol_indices = list(getattr(ue_input, "allocated_symbol_indices", []) or [])
        if len(allocated_symbol_indices) > 0:
            start_symbol = min(int(item) for item in allocated_symbol_indices)
            end_symbol = max(int(item) for item in allocated_symbol_indices) + 1
        else:
            start_symbol = max(0, int(getattr(ue_input, "start_symbol", 0)))
            end_symbol = start_symbol + max(0, int(getattr(ue_input, "allocated_symbols", 0)))
        if end_symbol <= start_symbol:
            return 0.0, 0.0
        symbol_duration_ms = slot_duration_ms / float(total_symbols)
        return (
            float(start_symbol) * symbol_duration_ms,
            float(end_symbol) * symbol_duration_ms,
        )

    def _annotate_phy_inputs_with_inter_cell_interference(
        self,
        *,
        state: GlobalState,
        phy_inputs: Sequence[PhyInput],
    ) -> None:
        if len(phy_inputs) <= 1:
            return

        connection_map = getattr(state, "connections", {})
        if not isinstance(connection_map, dict):
            connection_map = {}
        managed_cells = getattr(state, "managed_cells", {})
        if not isinstance(managed_cells, dict):
            managed_cells = {}
        radio_profile_by_cell_id = {
            str(cell_id): resolve_cell_rf_profile(
                dict(getattr(runtime_state, "radio_profile", {}) or {})
            )
            for cell_id, runtime_state in managed_cells.items()
        }

        active_entries: list[dict[str, Any]] = []
        for phy_input in phy_inputs:
            runtime_context = getattr(phy_input, "runtime_context", None)
            if runtime_context is None:
                continue
            total_resource_elements = max(
                1,
                int(getattr(runtime_context, "n_prbs_total", 0))
                * int(getattr(runtime_context, "num_ofdm_symbols_per_slot", 0))
                * 12,
            )
            for ue_input in list(getattr(phy_input, "ue_inputs", []) or []):
                if not bool(getattr(ue_input, "is_scheduled", False)):
                    continue
                frequency_interval_hz = self._build_phy_grant_frequency_interval_hz(
                    phy_input=phy_input,
                    ue_input=ue_input,
                )
                time_interval_ms = self._build_phy_grant_time_interval_ms(
                    phy_input=phy_input,
                    ue_input=ue_input,
                )
                active_entries.append(
                    {
                        "phy_input": phy_input,
                        "ue_input": ue_input,
                        "cell_id": str(getattr(phy_input, "cell_id", "cell_0")),
                        "rf_profile": radio_profile_by_cell_id.get(
                            str(getattr(phy_input, "cell_id", "cell_0")),
                            resolve_cell_rf_profile(
                                {
                                    "cell_type": "macro",
                                    "carrier_frequency_hz": getattr(runtime_context, "carrier_frequency_hz", 3.5e9),
                                }
                            ),
                        ),
                        "frequency_interval_hz": frequency_interval_hz,
                        "time_interval_ms": time_interval_ms,
                        "resource_share": float(
                            max(
                                0.0,
                                float(getattr(getattr(ue_input, "resource_grid_slice", None), "resource_elements", 0))
                                / float(total_resource_elements),
                            )
                        ),
                    }
                )

        for target_entry in active_entries:
            target_phy_input = target_entry["phy_input"]
            target_ue_input = target_entry["ue_input"]
            target_cell_id = str(target_entry["cell_id"])
            target_ue_id = str(getattr(target_ue_input, "ue_id", ""))
            connection = connection_map.get(target_ue_id)
            candidate_cells = {
                str(getattr(candidate, "cell_id", "")): candidate
                for candidate in list(getattr(connection, "candidate_cells", []) or [])
                if str(getattr(candidate, "cell_id", "")) != ""
            } if connection is not None else {}

            interference_mw = 0.0
            overlap_fraction_sum = 0.0
            frequency_coupling_fraction_sum = 0.0
            interfering_cell_ids: set[str] = set()
            target_frequency_interval_hz = target_entry["frequency_interval_hz"]
            target_time_interval_ms = target_entry["time_interval_ms"]

            for interferer_entry in active_entries:
                interferer_cell_id = str(interferer_entry["cell_id"])
                if interferer_cell_id == target_cell_id:
                    continue

                interferer_frequency_interval_hz = interferer_entry["frequency_interval_hz"]
                interferer_time_interval_ms = interferer_entry["time_interval_ms"]
                (
                    frequency_coupling_fraction,
                    frequency_overlap_fraction,
                    _,
                    _,
                ) = self._compute_frequency_coupling_fraction(
                    target_interval_hz=(
                        float(target_frequency_interval_hz[0]),
                        float(target_frequency_interval_hz[1]),
                    ),
                    interferer_interval_hz=(
                        float(interferer_frequency_interval_hz[0]),
                        float(interferer_frequency_interval_hz[1]),
                    ),
                    target_rf_profile=dict(target_entry.get("rf_profile", {}) or {}),
                    interferer_rf_profile=dict(interferer_entry.get("rf_profile", {}) or {}),
                )
                time_overlap_fraction = self._compute_interval_overlap_fraction(
                    float(target_time_interval_ms[0]),
                    float(target_time_interval_ms[1]),
                    float(interferer_time_interval_ms[0]),
                    float(interferer_time_interval_ms[1]),
                )
                coupling_fraction = float(frequency_coupling_fraction * time_overlap_fraction)
                overlap_fraction = float(frequency_overlap_fraction * time_overlap_fraction)
                if coupling_fraction <= 0.0:
                    continue

                interferer_phy_input = interferer_entry["phy_input"]
                interferer_ue_input = interferer_entry["ue_input"]
                candidate = candidate_cells.get(interferer_cell_id)
                candidate_beam_id = (
                    int(getattr(candidate, "candidate_beam_id", 0))
                    if candidate is not None
                    else None
                )
                received_power_dbm = (
                    float(getattr(candidate, "rsrp_dbm", -120.0))
                    if candidate is not None
                    else float(getattr(interferer_phy_input, "tx_power_dbm", 30.0))
                    - float(getattr(target_ue_input, "link_pathloss_db", 130.0))
                    - float(getattr(target_ue_input, "link_shadowing_db", 0.0))
                    - float(getattr(target_ue_input, "link_o2i_penetration_loss_db", 0.0))
                )
                beam_coupling_factor = self._compute_beam_coupling_factor(
                    target_candidate_beam_id=candidate_beam_id,
                    interferer_beam_id=getattr(interferer_ue_input, "serving_beam_id", None),
                )
                codebook_factor = 1.0 + 0.10 * float(
                    max(0, int(getattr(interferer_phy_input, "current_codebook_id", 0)))
                )
                profile_factor = 1.0 + 0.08 * float(
                    max(0, int(getattr(interferer_phy_input, "current_beam_profile_id", 0)))
                )
                coupling_scale = max(
                    0.0,
                    float(coupling_fraction)
                    * float(beam_coupling_factor)
                    * float(codebook_factor)
                    * float(profile_factor),
                )
                if coupling_scale <= 0.0:
                    continue
                interference_mw += self._dbm_to_mw(received_power_dbm) * coupling_scale
                overlap_fraction_sum += overlap_fraction
                frequency_coupling_fraction_sum += coupling_fraction
                interfering_cell_ids.add(interferer_cell_id)

            target_runtime_context = getattr(target_phy_input, "runtime_context", None)
            target_bandwidth_hz = float(
                getattr(target_runtime_context, "bandwidth_hz", 20e6)
                if target_runtime_context is not None
                else 20e6
            )
            thermal_noise_dbm = -174.0 + 10.0 * math.log10(max(1.0, target_bandwidth_hz)) + 7.0
            thermal_noise_mw = self._dbm_to_mw(thermal_noise_dbm)
            inter_cell_interference_dbm = self._mw_to_dbm(interference_mw)
            inter_cell_interference_penalty_db = (
                10.0 * math.log10(1.0 + interference_mw / max(thermal_noise_mw, 1e-12))
                if interference_mw > 0.0
                else 0.0
            )

            target_ue_input.inter_cell_interference_dbm = float(inter_cell_interference_dbm)
            target_ue_input.inter_cell_interference_penalty_db = float(
                inter_cell_interference_penalty_db
            )
            target_ue_input.inter_cell_interference_overlap_fraction = float(
                min(1.0, overlap_fraction_sum)
            )
            target_ue_input.inter_cell_frequency_coupling_fraction = float(
                min(1.0, frequency_coupling_fraction_sum)
            )
            target_ue_input.inter_cell_interference_source_count = int(len(interfering_cell_ids))

    def _run_managed_cell_schedulers(
        self,
        state: GlobalState,
    ) -> list[tuple[ManagedCellRuntimeState, SimpleNamespace | GlobalState]]:
        runtime_states = self._ordered_runtime_states(state)
        if len(runtime_states) <= 1:
            self.scheduler.run(state)
            self._refresh_managed_cell_runtime_views(state)
            self._store_cluster_scheduler_summary(state)
            primary_runtime = runtime_states[0] if len(runtime_states) == 1 else None
            return (
                [(primary_runtime, state)]
                if primary_runtime is not None
                else []
            )

        original_total_prbs = int(getattr(self.scheduler, "total_prbs_per_slot", 0))
        original_total_symbols = int(getattr(self.scheduler, "total_symbols_per_slot", 0))
        scoped_states: list[tuple[ManagedCellRuntimeState, SimpleNamespace | GlobalState]] = []
        try:
            for runtime_state in runtime_states:
                scoped_state = self._build_cell_scoped_state(state, runtime_state)
                scoped_states.append((runtime_state, scoped_state))
                if hasattr(self.scheduler, "total_prbs_per_slot"):
                    self.scheduler.total_prbs_per_slot = int(
                        getattr(runtime_state.resource_grid, "n_prbs_total", original_total_prbs)
                    )
                if hasattr(self.scheduler, "total_symbols_per_slot"):
                    self.scheduler.total_symbols_per_slot = int(
                        getattr(
                            runtime_state.resource_grid,
                            "n_ofdm_symbols_per_slot",
                            original_total_symbols,
                        )
                    )
                self.scheduler.run(scoped_state)
        finally:
            if hasattr(self.scheduler, "total_prbs_per_slot"):
                self.scheduler.total_prbs_per_slot = int(original_total_prbs)
            if hasattr(self.scheduler, "total_symbols_per_slot"):
                self.scheduler.total_symbols_per_slot = int(original_total_symbols)

        self._refresh_managed_cell_runtime_views(state)
        self._store_cluster_scheduler_summary(state)
        return scoped_states

    def _run_managed_cell_phy(
        self,
        state: GlobalState,
        scoped_states: Sequence[tuple[ManagedCellRuntimeState, SimpleNamespace | GlobalState]],
    ) -> tuple[PhyResult, PhyResult | None]:
        per_cell_phy_results: list[PhyResult] = []
        per_cell_shadow_results: list[PhyResult] = []
        phy_inputs_by_runtime: list[tuple[ManagedCellRuntimeState, SimpleNamespace | GlobalState, PhyInput]] = []
        for runtime_state, scoped_state in scoped_states:
            phy_inputs_by_runtime.append((runtime_state, scoped_state, self._build_phy_input(scoped_state)))

        cluster_results = None
        if self._rt_sys_mode:
            self.phy_adapter.current_snapshot.metadata["sleep_modes"] = {
                r.cell_id: str(r.cell.cell_sleep_mode) for r, _, _ in phy_inputs_by_runtime}
            cluster_results = self.phy_adapter.run_cluster_slot(
                [item[2] for item in phy_inputs_by_runtime], self.phy_adapter.current_snapshot)
        else:
            self._annotate_phy_inputs_with_inter_cell_interference(
                state=state, phy_inputs=[item[2] for item in phy_inputs_by_runtime])

        for runtime_state, scoped_state, phy_input in phy_inputs_by_runtime:
            phy_result = (cluster_results[runtime_state.cell_id] if cluster_results is not None
                          else self.phy_adapter.run_slot(phy_input))
            shadow_phy_result = (
                self.shadow_phy_adapter.run_slot(phy_input)
                if self.shadow_phy_adapter is not None
                else None
            )
            runtime_state.last_phy_result = phy_result
            runtime_state.last_shadow_phy_result = shadow_phy_result
            self.state_updater.apply_phy_result(scoped_state, phy_result)
            if self._rt_sys_mode:
                runtime_state.cell.rf_tx_power_w = self.phy_adapter.trace['cells'][runtime_state.cell_id]['average_used_w']
                runtime_state.cell.rf_power_produced_at_slot = state.slot_id
                # StateUpdater replaces the scoped grid after PHY accounting.
                # Publish that grid to the views used by telemetry and the RIC.
                runtime_state.resource_grid = scoped_state.resource_grid
                # MU shares REs across users. Report occupied physical REs,
                # rather than double-counting per-user allocations.
                runtime_state.resource_grid.allocated_resource_elements=int(
                    self.phy_adapter.trace['cells'][runtime_state.cell_id]['occupied_re'])
                runtime_state.resource_grid.utilization=runtime_state.resource_grid.allocated_resource_elements/max(
                    1,runtime_state.resource_grid.total_resource_elements)
                if runtime_state.is_primary:
                    state.resource_grid = runtime_state.resource_grid
            per_cell_phy_results.append(phy_result)
            if shadow_phy_result is not None:
                per_cell_shadow_results.append(shadow_phy_result)

        if self._rt_sys_mode:self.phy_adapter.release_inactive_contexts(state)
        combined_phy_result = self._combine_phy_results(
            slot_id=int(getattr(state, "slot_id", -1)),
            phy_results=per_cell_phy_results,
            combined_cell_id="cluster" if len(per_cell_phy_results) > 1 else str(getattr(state.cell, "cell_id", "cell_0")),
        )
        combined_shadow_phy_result = (
            self._combine_phy_results(
                slot_id=int(getattr(state, "slot_id", -1)),
                phy_results=per_cell_shadow_results,
                combined_cell_id="cluster_shadow",
            )
            if len(per_cell_shadow_results) > 0
            else None
        )
        state.last_phy_result = combined_phy_result
        state.last_shadow_phy_result = combined_shadow_phy_result
        state.last_shadow_phy_label = self.shadow_phy_label
        state.avg_sinr_db = float(getattr(combined_phy_result, "avg_sinr_db", 0.0))
        state.avg_bler = float(getattr(combined_phy_result, "avg_bler", 0.0))
        return combined_phy_result, combined_shadow_phy_result

    def _derive_beamforming_state_for_ue(
        self,
        *,
        ue: UeState,
        cell: CellState,
        antenna_state: AntennaState,
        codebook_map: dict[int, dict],
        beam_profile_map: dict[int, dict],
    ) -> BeamformingState:
        ue_id = str(ue.ue_id)
        num_rx_ant = int(antenna_state.ue_num_rx_ant_by_ue.get(ue_id, 1))
        max_layers = max(1, min(int(antenna_state.gnb_num_tx_ant), int(num_rx_ant)))
        rank = max(1, min(int(getattr(ue, "rank", 1)), max_layers))
        layer_count = max(1, rank)
        serving_beam_id = int(getattr(ue, "serving_beam_id", 0))
        codebook_id = int(cell.current_codebook_id)
        beam_profile_id = int(cell.current_beam_profile_id)
        beam_gain_db = self._derive_beam_gain_from_config(
            codebook_id=codebook_id,
            beam_profile_id=beam_profile_id,
            serving_beam_id=serving_beam_id,
            codebook_map=codebook_map,
            beam_profile_map=beam_profile_map,
        )
        beam_direction_deg = float(serving_beam_id) * 30.0
        precoder_id = int(codebook_id)
        combiner_id = int(getattr(ue, "pmi", 0))
        mimo_model = self._get_single_user_mimo_model()
        precoder_matrix = mimo_model.build_precoder_matrix(
            num_tx_ant=int(antenna_state.gnb_num_tx_ant),
            layer_count=int(layer_count),
            precoder_id=int(precoder_id),
        )
        combiner_vector = mimo_model.build_combiner_vector(
            num_rx_ant=int(num_rx_ant),
            combiner_id=int(combiner_id),
        )
        return BeamformingState(
            ue_id=ue_id,
            codebook_id=codebook_id,
            beam_profile_id=beam_profile_id,
            serving_beam_id=serving_beam_id,
            precoder_id=precoder_id,
            combiner_id=combiner_id,
            rank=rank,
            layer_count=layer_count,
            beam_gain_db=float(beam_gain_db),
            beam_direction_deg=beam_direction_deg,
            precoding_matrix_label=f"cb{codebook_id}-r{rank}-b{serving_beam_id}",
            precoder_matrix=precoder_matrix,
            combiner_vector=combiner_vector,
            precoder_match_score=1.0,
            post_eq_sinr_db=float(getattr(ue, "sinr_db", 0.0)),
        )

    def _build_logical_channels_for_ue(
        self,
        *,
        ue_item: dict,
    ) -> list[LogicalChannelState]:
        raw_logical_channels = list(ue_item.get("logical_channels", []) or [])
        ue_priority = int(ue_item.get("priority", 0))
        ue_latency_budget_ms = float(ue_item.get("latency_budget_ms", 0.0))
        ue_slice_id = str(ue_item.get("slice_id", ""))
        ue_traffic_type = str(ue_item.get("traffic_type", ""))

        if len(raw_logical_channels) == 0:
            return [
                LogicalChannelState(
                    logical_channel_id=str(ue_item.get("default_logical_channel_id", "lc_0")),
                    drb_id=int(ue_item.get("drb_id", 1)),
                    lcg_id=int(ue_item.get("lcg_id", 0)),
                    qos_flow_id=(
                        int(ue_item["qos_flow_id"])
                        if ("qos_flow_id" in ue_item and ue_item["qos_flow_id"] is not None)
                        else None
                    ),
                    queue_bytes=int(ue_item.get("queue_bytes", 0)),
                    priority=ue_priority,
                    packet_delay_budget_ms=ue_latency_budget_ms,
                    slice_id=ue_slice_id,
                    traffic_type=ue_traffic_type,
                    prioritized_bit_rate_bps=(
                        float(ue_item["prioritized_bit_rate_bps"])
                        if (
                            "prioritized_bit_rate_bps" in ue_item
                            and ue_item["prioritized_bit_rate_bps"] is not None
                        )
                        else None
                    ),
                    bucket_size_duration_ms=float(ue_item.get("bucket_size_duration_ms", 10.0)),
                    max_queue_bytes=(
                        int(ue_item["max_queue_bytes"])
                        if ("max_queue_bytes" in ue_item and ue_item["max_queue_bytes"] is not None)
                        else None
                    ),
                )
            ]

        logical_channels: list[LogicalChannelState] = []
        top_level_queue_bytes = int(ue_item.get("queue_bytes", 0))
        declared_queue_bytes = 0
        explicit_queue_declared = False

        for index, logical_channel_item in enumerate(raw_logical_channels):
            queue_bytes = int(logical_channel_item.get("queue_bytes", 0))
            if "queue_bytes" in logical_channel_item and logical_channel_item["queue_bytes"] is not None:
                explicit_queue_declared = True
            declared_queue_bytes += queue_bytes
            logical_channels.append(
                LogicalChannelState(
                    logical_channel_id=str(
                        logical_channel_item.get("logical_channel_id", f"lc_{index}")
                    ),
                    drb_id=int(logical_channel_item.get("drb_id", index + 1)),
                    lcg_id=int(logical_channel_item.get("lcg_id", 0)),
                    qos_flow_id=(
                        int(logical_channel_item["qos_flow_id"])
                        if (
                            "qos_flow_id" in logical_channel_item
                            and logical_channel_item["qos_flow_id"] is not None
                        )
                        else None
                    ),
                    queue_bytes=queue_bytes,
                    priority=int(logical_channel_item.get("priority", ue_priority)),
                    packet_delay_budget_ms=float(
                        logical_channel_item.get("latency_budget_ms", ue_latency_budget_ms)
                    ),
                    slice_id=str(logical_channel_item.get("slice_id", ue_slice_id)),
                    traffic_type=str(logical_channel_item.get("traffic_type", ue_traffic_type)),
                    prioritized_bit_rate_bps=(
                        float(logical_channel_item["prioritized_bit_rate_bps"])
                        if (
                            "prioritized_bit_rate_bps" in logical_channel_item
                            and logical_channel_item["prioritized_bit_rate_bps"] is not None
                        )
                        else None
                    ),
                    bucket_size_duration_ms=float(
                        logical_channel_item.get(
                            "bucket_size_duration_ms",
                            ue_item.get("bucket_size_duration_ms", 10.0),
                        )
                    ),
                    max_queue_bytes=(
                        int(logical_channel_item["max_queue_bytes"])
                        if (
                            "max_queue_bytes" in logical_channel_item
                            and logical_channel_item["max_queue_bytes"] is not None
                        )
                        else None
                    ),
                )
            )

        if not explicit_queue_declared and len(logical_channels) > 0:
            logical_channels[0].queue_bytes = int(top_level_queue_bytes)
        elif len(logical_channels) > 0 and top_level_queue_bytes > declared_queue_bytes:
            logical_channels[0].queue_bytes += int(top_level_queue_bytes - declared_queue_bytes)

        return logical_channels

    def _build_logical_channel_groups_for_ue(
        self,
        *,
        logical_channels: Sequence[LogicalChannelState],
    ) -> list[LogicalChannelGroupState]:
        groups_by_lcg_id: dict[int, LogicalChannelGroupState] = {}
        for logical_channel in logical_channels:
            logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
            if logical_channel_id == "":
                continue
            lcg_id = int(getattr(logical_channel, "lcg_id", 0))
            group = groups_by_lcg_id.setdefault(
                lcg_id,
                LogicalChannelGroupState(lcg_id=lcg_id),
            )
            group.logical_channel_ids.append(logical_channel_id)
            drb_id = getattr(logical_channel, "drb_id", None)
            if drb_id is not None and int(drb_id) not in group.drb_ids:
                group.drb_ids.append(int(drb_id))
            queue_bytes = int(getattr(logical_channel, "queue_bytes", 0))
            group.queue_bytes += int(queue_bytes)
            if getattr(logical_channel, "prioritized_bit_rate_bps", None) is not None:
                group.prioritized_queue_bytes += int(queue_bytes)
            group.pbr_credit_bytes += float(getattr(logical_channel, "pbr_credit_bytes", 0.0) or 0.0)
            group.scheduled_bytes += int(getattr(logical_channel, "scheduled_bytes", 0))
            group.prioritized_scheduled_bytes += int(
                getattr(logical_channel, "prioritized_scheduled_bytes", 0)
            )
            group.delivered_bytes += int(getattr(logical_channel, "delivered_bytes", 0))
            group.bsr_reported_bytes = int(group.queue_bytes)
            group.bsr_pending_bytes = 0
            group.bsr_age_slots = 0
            group.last_bsr_report_slot = 0
            group.last_bsr_trigger_slot = 0
            group.bsr_triggered = bool(group.queue_bytes > 0)
            group.bsr_trigger_reason = "initial" if group.queue_bytes > 0 else None

        groups = list(groups_by_lcg_id.values())
        for group in groups:
            group.logical_channel_ids = sorted(set(group.logical_channel_ids))
            group.drb_ids = sorted(set(group.drb_ids))
        return sorted(groups, key=lambda group: int(group.lcg_id))

    def _aggregate_arrivals_by_logical_channel_group(
        self,
        *,
        arrivals_by_logical_channel: dict[str, int],
    ) -> dict[str, int]:
        arrivals_by_logical_channel_group: dict[str, int] = {}
        for ue in getattr(self.state, "ues", []):
            ue_id = str(getattr(ue, "ue_id", ""))
            if ue_id == "":
                continue
            logical_channel_to_lcg = {
                str(getattr(logical_channel, "logical_channel_id", "")): int(
                    getattr(logical_channel, "lcg_id", 0)
                )
                for logical_channel in list(getattr(ue, "logical_channels", []) or [])
                if str(getattr(logical_channel, "logical_channel_id", "")) != ""
            }
            for logical_channel_id, lcg_id in logical_channel_to_lcg.items():
                channel_key = f"{ue_id}:{logical_channel_id}"
                if channel_key not in arrivals_by_logical_channel:
                    continue
                group_key = f"{ue_id}:lcg_{int(lcg_id)}"
                arrivals_by_logical_channel_group[group_key] = (
                    int(arrivals_by_logical_channel_group.get(group_key, 0))
                    + int(arrivals_by_logical_channel[channel_key])
                )
        return arrivals_by_logical_channel_group

    def _build_initial_state(self) -> GlobalState:
        """
        Builds the initial simulation state from YAML-loaded configuration.
        INPUT
        No explicit input.
        OUTPUT
        A GlobalState object with initialized cell and UE states.
        EXAMPLE OUTPUT
        GlobalState(...)
        """
        cell = CellState(
            cell_id=self.cell_config["cell_id"],
            tx_power_dbm=self.cell_config["tx_power_dbm"],
            current_codebook_id=self.cell_config["current_codebook_id"],
            current_beam_profile_id=self.cell_config["current_beam_profile_id"],
            managed_cell_ids=list(self.cell_config.get("managed_cell_ids", []) or []),
            neighbor_cell_positions=dict(self.cell_config.get("neighbor_cell_positions", {}) or {}),
            cell_radio_profiles=dict(self.cell_config.get("cell_radio_profiles", {}) or {}),
            handover_margin_db=float(self.cell_config.get("handover_margin_db", 3.0)),
            handover_delay_slots=int(self.cell_config.get("handover_delay_slots", 2)),
            handover_cost_db=float(self.cell_config.get("handover_cost_db", 0.0)),
            handover_interruption_slots=int(self.cell_config.get("handover_interruption_slots", 0)),
            steering_hysteresis_db=float(self.cell_config.get("steering_hysteresis_db", 0.0)),
            measurement_event_trigger=str(self.cell_config.get("measurement_event_trigger", "a3_offset")),
            measurement_event_offset_db=float(self.cell_config.get("measurement_event_offset_db", 0.0)),
        )
        if len(cell.managed_cell_ids) == 0:
            cell.managed_cell_ids = [str(cell.cell_id)]
        if str(cell.cell_id) not in cell.neighbor_cell_positions:
            cell.neighbor_cell_positions[str(cell.cell_id)] = {
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": float(self.cell_config.get("cell_height_m", 25.0)),
            }

        scheduler = SchedulerState()
        channel = ChannelState()
        channel.pathloss_db = 130.0
        channel.shadowing_db = 6.0
        channel.channel_profile = "tdl_c"
        channel.doppler_hz = 30.0

        raw_codebooks = self.beamforming_config.get("codebooks", {})
        raw_profiles = self.beamforming_config.get("beam_profiles", {})

        codebook_map = {int(key): value for key, value in raw_codebooks.items()}
        beam_profile_map = {int(key): value for key, value in raw_profiles.items()}

        beam = BeamState(
            codebook_id=cell.current_codebook_id,
            beam_profile_id=cell.current_beam_profile_id,
            selected_beam_id=0,
            beam_gain_db=0.0,
            codebook_map=codebook_map,
            beam_profile_map=beam_profile_map,
        )
        antenna_config = dict(self.beamforming_config.get("antenna", {}))

        ues = []
        for ue_item in self.ue_config["ues"]:
            logical_channels = self._build_logical_channels_for_ue(ue_item=ue_item)
            logical_channel_groups = self._build_logical_channel_groups_for_ue(
                logical_channels=logical_channels
            )
            aggregate_queue_bytes = int(sum(int(item.queue_bytes) for item in logical_channels))
            ues.append(
                UeState(
                    ue_id=ue_item["ue_id"],
                    queue_bytes=aggregate_queue_bytes,
                    throughput_mbps=ue_item["throughput_mbps"],
                    sinr_db=ue_item["sinr_db"],
                    radio_quality_db=ue_item["radio_quality_db"],
                    bler=ue_item["bler"],
                    serving_beam_id=ue_item["serving_beam_id"],
                    slice_id=ue_item["slice_id"],
                    traffic_type=ue_item["traffic_type"],
                    latency_budget_ms=ue_item["latency_budget_ms"],
                    priority=ue_item["priority"],
                    logical_channels=logical_channels,
                    logical_channel_groups=logical_channel_groups,
                )
            )
        antenna_state = AntennaState(
            gnb_num_tx_ant=int(antenna_config.get("gnb_num_tx_ant", 4)),
            ue_num_rx_ant_by_ue={
                str(ue.ue_id): int(self.ue_config["ues"][index].get("num_rx_ant", antenna_config.get("default_ue_num_rx_ant", 2)))
                for index, ue in enumerate(ues)
            },
            carrier_frequency_hz=float(antenna_config.get("carrier_frequency_hz", 3.5e9)),
            bandwidth_hz=float(antenna_config.get("bandwidth_hz", 20e6)),
            subcarrier_spacing_hz=float(antenna_config.get("subcarrier_spacing_hz", 30e3)),
            cyclic_prefix_type=str(antenna_config.get("cyclic_prefix_type", "normal")),
            num_ofdm_symbols_per_slot=int(antenna_config.get("num_ofdm_symbols_per_slot", 14)),
            gnb_panel_layout=str(antenna_config.get("gnb_panel_layout", "2x2")),
            ue_panel_layout_by_ue={
                str(ue.ue_id): str(self.ue_config["ues"][index].get("ue_panel_layout", antenna_config.get("default_ue_panel_layout", "1x2")))
                for index, ue in enumerate(ues)
            },
            polarization=str(antenna_config.get("polarization", "single")),
        )
        numerology = self._derive_numerology_config(
            carrier_frequency_hz=antenna_state.carrier_frequency_hz,
            bandwidth_hz=antenna_state.bandwidth_hz,
            subcarrier_spacing_hz=antenna_state.subcarrier_spacing_hz,
            cyclic_prefix_type=antenna_state.cyclic_prefix_type,
        )
        antenna_state.num_ofdm_symbols_per_slot = int(numerology["n_ofdm_symbols_per_slot"])
        antenna_state.nr_frequency_range = str(numerology["nr_frequency_range"])
        antenna_state.nr_numerology_index = int(numerology["nr_numerology_index"])
        channel_profiles = [
            str(item.get("channel_model_type", "")).strip().lower()
            for item in self.ue_config.get("ues", [])
            if str(item.get("channel_model_type", "")).strip() != ""
        ]
        channel.channel_profile = (
            channel_profiles[0]
            if len(channel_profiles) > 0
            else ("tdl_d" if antenna_state.carrier_frequency_hz >= 6.0e9 else "tdl_c")
        )
        if len(self.ue_config.get("ues", [])) > 0:
            ue_items = list(self.ue_config.get("ues", []) or [])
            channel.pathloss_db = float(
                sum(float(item.get("pathloss_db", 130.0)) for item in ue_items) / float(len(ue_items))
            )
            channel.shadowing_db = float(
                sum(float(item.get("shadowing_db", 6.0)) for item in ue_items) / float(len(ue_items))
            )
            channel.doppler_hz = float(
                sum(float(item.get("doppler_hz", 30.0)) for item in ue_items) / float(len(ue_items))
            )
        self._sync_scheduler_budget_with_numerology(numerology)
        connections = {
            str(ue.ue_id): ConnectionState(
                ue_id=str(ue.ue_id),
                serving_cell_id=str(self.ue_config["ues"][index].get("serving_cell_id", cell.cell_id)),
                connection_state="connected",
                c_rnti=4096 + index,
                serving_beam_id=int(ue.serving_beam_id),
                rank_indicator=int(getattr(ue, "rank", 1)),
                pmi=int(getattr(ue, "pmi", 0)),
                cqi=int(getattr(ue, "cqi", 0)),
                active_harq_processes=[],
                candidate_cells=[
                    CandidateCellState(
                        cell_id=str(candidate.get("cell_id", "")),
                        cell_type=str(candidate.get("cell_type", "macro")),
                        x_m=float(candidate.get("x_m", 0.0)),
                        y_m=float(candidate.get("y_m", 0.0)),
                        z_m=float(candidate.get("z_m", 25.0)),
                        distance_2d_m=float(candidate.get("distance_2d_m", 0.0)),
                        distance_3d_m=float(candidate.get("distance_3d_m", 0.0)),
                        carrier_frequency_hz=float(candidate.get("carrier_frequency_hz", antenna_state.carrier_frequency_hz)),
                        bandwidth_hz=float(candidate.get("bandwidth_hz", antenna_state.bandwidth_hz)),
                        subcarrier_spacing_hz=float(candidate.get("subcarrier_spacing_hz", antenna_state.subcarrier_spacing_hz)),
                        tx_power_dbm=float(candidate.get("tx_power_dbm", cell.tx_power_dbm)),
                        pathloss_db=float(candidate.get("pathloss_db", 130.0)),
                        shadowing_db=float(candidate.get("shadowing_db", 0.0)),
                        rsrp_dbm=float(candidate.get("rsrp_dbm", -100.0)),
                        quality_score_db=float(candidate.get("quality_score_db", -100.0)),
                        candidate_beam_id=int(candidate.get("candidate_beam_id", 0)),
                        is_serving=bool(candidate.get("cell_id", "") == self.ue_config["ues"][index].get("serving_cell_id", cell.cell_id)),
                    )
                    for candidate in list(self.ue_config["ues"][index].get("candidate_cells", []) or [])
                ],
            )
            for index, ue in enumerate(ues)
        }
        beamforming = {
            str(ue.ue_id): self._derive_beamforming_state_for_ue(
                ue=ue,
                cell=cell,
                antenna_state=antenna_state,
                codebook_map=codebook_map,
                beam_profile_map=beam_profile_map,
            )
            for ue in ues
        }
        links = {
            str(ue.ue_id): LinkState(
                ue_id=str(ue.ue_id),
                serving_cell_id=str(self.ue_config["ues"][index].get("serving_cell_id", cell.cell_id)),
                scenario_type=str(self.ue_config["ues"][index].get("scenario_type", "umi_street_canyon")),
                los_state=str(self.ue_config["ues"][index].get("los_state", "los")),
                x_m=float(self.ue_config["ues"][index].get("x_m", 0.0)),
                y_m=float(self.ue_config["ues"][index].get("y_m", 0.0)),
                z_m=float(self.ue_config["ues"][index].get("z_m", 1.5)),
                distance_2d_m=float(
                    self.ue_config["ues"][index].get(
                        "distance_2d_m",
                        self.ue_config["ues"][index].get("large_scale", {}).get("distance_2d_m", 0.0),
                    )
                ),
                distance_3d_m=float(
                    self.ue_config["ues"][index].get(
                        "distance_3d_m",
                        self.ue_config["ues"][index].get("large_scale", {}).get("distance_3d_m", 0.0),
                    )
                ),
                velocity_mps=float(self.ue_config["ues"][index].get("velocity_mps", 0.0)),
                heading_deg=float(self.ue_config["ues"][index].get("heading_deg", 0.0)),
                is_indoor=bool(self.ue_config["ues"][index].get("is_indoor", False)),
                pathloss_db=float(
                    self.ue_config["ues"][index].get(
                        "pathloss_db",
                        self.ue_config["ues"][index].get("large_scale", {}).get("pathloss_db", channel.pathloss_db),
                    )
                ),
                shadowing_db=float(
                    self.ue_config["ues"][index].get(
                        "shadowing_db",
                        self.ue_config["ues"][index].get("large_scale", {}).get("shadowing_db", channel.shadowing_db),
                    )
                ),
                shadowing_sigma_db=float(
                    self.ue_config["ues"][index].get(
                        "shadowing_sigma_db",
                        self.ue_config["ues"][index].get("large_scale", {}).get(
                            "shadowing_sigma_db",
                            abs(
                                float(
                                    self.ue_config["ues"][index].get(
                                        "shadowing_db",
                                        self.ue_config["ues"][index].get("large_scale", {}).get(
                                            "shadowing_db",
                                            channel.shadowing_db,
                                        ),
                                    )
                                )
                            ),
                        ),
                    )
                ),
                o2i_penetration_loss_db=float(
                    self.ue_config["ues"][index].get(
                        "o2i_penetration_loss_db",
                        self.ue_config["ues"][index].get("large_scale", {}).get("o2i_penetration_loss_db", 0.0),
                    )
                ),
                doppler_hz=float(
                    self.ue_config["ues"][index].get(
                        "doppler_hz",
                        self.ue_config["ues"][index].get("small_scale", {}).get("doppler_hz", channel.doppler_hz),
                    )
                ),
                delay_spread_ns=float(
                    self.ue_config["ues"][index].get(
                        "delay_spread_ns",
                        self.ue_config["ues"][index].get("small_scale", {}).get("delay_spread_ns", 300.0),
                    )
                ),
                k_factor_db=float(
                    self.ue_config["ues"][index].get(
                        "k_factor_db",
                        self.ue_config["ues"][index].get("small_scale", {}).get("k_factor_db", 0.0),
                    )
                ),
                angle_spread_deg=float(
                    self.ue_config["ues"][index].get(
                        "angle_spread_deg",
                        self.ue_config["ues"][index].get("small_scale", {}).get("angle_spread_deg", 35.0),
                    )
                ),
                spatial_correlation_level=float(
                    self.ue_config["ues"][index].get(
                        "spatial_correlation_level",
                        self.ue_config["ues"][index].get("small_scale", {}).get("spatial_correlation_level", 0.5),
                    )
                ),
                channel_model_type=str(
                    self.ue_config["ues"][index].get(
                        "channel_model_type",
                        self.ue_config["ues"][index].get("small_scale", {}).get("channel_model_type", channel.channel_profile),
                    )
                ),
                fast_fading_seed=int(
                    self.ue_config["ues"][index].get(
                        "fast_fading_seed",
                        self.ue_config["ues"][index].get("small_scale", {}).get("fast_fading_seed", 1000 + index),
                    )
                ),
                spatial_corr_seed=int(
                    self.ue_config["ues"][index].get(
                        "spatial_corr_seed",
                        self.ue_config["ues"][index].get("small_scale", {}).get("spatial_corr_seed", 2000 + index),
                    )
                ),
                large_scale=LargeScaleLinkState(
                    distance_2d_m=float(
                        self.ue_config["ues"][index].get("large_scale", {}).get(
                            "distance_2d_m",
                            self.ue_config["ues"][index].get("distance_2d_m", 0.0),
                        )
                    ),
                    distance_3d_m=float(
                        self.ue_config["ues"][index].get("large_scale", {}).get(
                            "distance_3d_m",
                            self.ue_config["ues"][index].get("distance_3d_m", 0.0),
                        )
                    ),
                    pathloss_db=float(
                        self.ue_config["ues"][index].get("large_scale", {}).get(
                            "pathloss_db",
                            self.ue_config["ues"][index].get("pathloss_db", channel.pathloss_db),
                        )
                    ),
                    shadowing_db=float(
                        self.ue_config["ues"][index].get("large_scale", {}).get(
                            "shadowing_db",
                            self.ue_config["ues"][index].get("shadowing_db", channel.shadowing_db),
                        )
                    ),
                    shadowing_sigma_db=float(
                        self.ue_config["ues"][index].get("large_scale", {}).get(
                            "shadowing_sigma_db",
                            self.ue_config["ues"][index].get(
                                "shadowing_sigma_db",
                                abs(
                                    float(
                                        self.ue_config["ues"][index].get("large_scale", {}).get(
                                            "shadowing_db",
                                            self.ue_config["ues"][index].get(
                                                "shadowing_db",
                                                channel.shadowing_db,
                                            ),
                                        )
                                    )
                                ),
                            ),
                        )
                    ),
                    los_state=str(
                        self.ue_config["ues"][index].get("large_scale", {}).get(
                            "los_state",
                            self.ue_config["ues"][index].get("los_state", "los"),
                        )
                    ),
                    o2i_penetration_loss_db=float(
                        self.ue_config["ues"][index].get("large_scale", {}).get(
                            "o2i_penetration_loss_db",
                            self.ue_config["ues"][index].get("o2i_penetration_loss_db", 0.0),
                        )
                    ),
                ),
                small_scale=SmallScaleLinkState(
                    doppler_hz=float(
                        self.ue_config["ues"][index].get("small_scale", {}).get(
                            "doppler_hz",
                            self.ue_config["ues"][index].get("doppler_hz", channel.doppler_hz),
                        )
                    ),
                    delay_spread_ns=float(
                        self.ue_config["ues"][index].get("small_scale", {}).get(
                            "delay_spread_ns",
                            self.ue_config["ues"][index].get("delay_spread_ns", 300.0),
                        )
                    ),
                    k_factor_db=float(
                        self.ue_config["ues"][index].get("small_scale", {}).get(
                            "k_factor_db",
                            self.ue_config["ues"][index].get("k_factor_db", 0.0),
                        )
                    ),
                    angle_spread_deg=float(
                        self.ue_config["ues"][index].get("small_scale", {}).get(
                            "angle_spread_deg",
                            self.ue_config["ues"][index].get("angle_spread_deg", 35.0),
                        )
                    ),
                    spatial_correlation_level=float(
                        self.ue_config["ues"][index].get("small_scale", {}).get(
                            "spatial_correlation_level",
                            self.ue_config["ues"][index].get("spatial_correlation_level", 0.5),
                        )
                    ),
                    channel_model_type=str(
                        self.ue_config["ues"][index].get("small_scale", {}).get(
                            "channel_model_type",
                            self.ue_config["ues"][index].get("channel_model_type", channel.channel_profile),
                        )
                    ),
                    fast_fading_seed=int(
                        self.ue_config["ues"][index].get("small_scale", {}).get(
                            "fast_fading_seed",
                            self.ue_config["ues"][index].get("fast_fading_seed", 1000 + index),
                        )
                    ),
                    spatial_corr_seed=int(
                        self.ue_config["ues"][index].get("small_scale", {}).get(
                            "spatial_corr_seed",
                            self.ue_config["ues"][index].get("spatial_corr_seed", 2000 + index),
                        )
                    ),
                ),
                last_post_eq_sinr_db=float(getattr(ue, "sinr_db", 0.0)),
                cqi=int(getattr(ue, "cqi", 0)),
                ri=int(getattr(ue, "ri", 1)),
                pmi=int(getattr(ue, "pmi", 0)),
            )
            for index, ue in enumerate(ues)
        }

        control = ControlState()
        resource_grid = self._build_resource_grid_state_from_numerology(numerology)
        managed_cells = self._build_managed_cell_runtime_states(
            primary_cell=cell,
            primary_scheduler=scheduler,
            primary_channel=channel,
            primary_beam=beam,
            primary_resource_grid=resource_grid,
        )

        state = GlobalState(
            slot_id=0,
            sim_time_ms=0.0,
            slot_duration_ms=self.config.slot_duration_ms,
            cell=cell,
            scheduler=scheduler,
            channel=channel,
            beam=beam,
            antenna=antenna_state,
            beamforming=beamforming,
            ues=ues,
            control=control,
            runtime_mode=("multi_cell_ready" if len(managed_cells) > 1 else "single_active_cell"),
            connections=connections,
            links=links,
            resource_grid=resource_grid,
            managed_cells=managed_cells,
        )

        # Initialize measurement view for controllers.
        if not bool(getattr(self.phy_adapter,'cluster_execution',False)):
            self.mobility_manager.update(state)
        self._refresh_managed_cell_runtime_views(state)
        if not bool(getattr(self.phy_adapter,'cluster_execution',False)):
            self.measurement_updater.update(state)
        return state

    def _sync_beam_state_from_cell(self) -> None:
        """
        Synchronizes beam-related runtime state from the current cell control
        state after actions are applied.
        INPUT
        No explicit input.
        OUTPUT
        None. The beam state is updated in place.
        EXAMPLE OUTPUT
        state.beam.codebook_id becomes state.cell.current_codebook_id
        """
        self.state.beam.codebook_id = self.state.cell.current_codebook_id
        self.state.beam.beam_profile_id = self.state.cell.current_beam_profile_id
        beamforming_map = getattr(self.state, "beamforming", {})
        if not isinstance(beamforming_map, dict):
            return

        for ue in getattr(self.state, "ues", []):
            ue_id = str(getattr(ue, "ue_id", ""))
            current_state = beamforming_map.get(ue_id)
            if current_state is None:
                continue
            updated_state = self._derive_beamforming_state_for_ue(
                ue=ue,
                cell=self.state.cell,
                antenna_state=self.state.antenna,
                codebook_map=self.state.beam.codebook_map,
                beam_profile_map=self.state.beam.beam_profile_map,
            )
            updated_state.rank = int(getattr(current_state, "rank", updated_state.rank))
            updated_state.layer_count = int(getattr(current_state, "layer_count", updated_state.layer_count))
            updated_state.combiner_id = int(getattr(current_state, "combiner_id", updated_state.combiner_id))
            mimo_model = self._get_single_user_mimo_model()
            updated_state.precoder_matrix = mimo_model.build_precoder_matrix(
                num_tx_ant=int(getattr(self.state.antenna, "gnb_num_tx_ant", 1)),
                layer_count=int(updated_state.layer_count),
                precoder_id=int(updated_state.precoder_id),
            )
            updated_state.combiner_vector = mimo_model.build_combiner_vector(
                num_rx_ant=int(getattr(self.state.antenna, "ue_num_rx_ant_by_ue", {}).get(ue_id, 1)),
                combiner_id=int(updated_state.combiner_id),
            )
            updated_state.precoder_match_score = float(
                getattr(current_state, "precoder_match_score", updated_state.precoder_match_score)
            )
            updated_state.post_eq_sinr_db = getattr(current_state, "post_eq_sinr_db", updated_state.post_eq_sinr_db)
            beamforming_map[ue_id] = updated_state

    def _resolve_config_target_cell_ids(self, action: ConfigAction) -> list[str]:
        resolve_target_cell_ids = getattr(self.executor, "resolve_target_cell_ids", None)
        if callable(resolve_target_cell_ids):
            resolved_target_cell_ids = resolve_target_cell_ids(self.state, action)
            if isinstance(resolved_target_cell_ids, (list, tuple, set)):
                return [str(cell_id) for cell_id in list(resolved_target_cell_ids)]

        managed_cells = getattr(self.state, "managed_cells", {})
        if not isinstance(managed_cells, dict) or len(managed_cells) == 0:
            return [str(getattr(self.state.cell, "cell_id", "cell_0"))]

        apply_scope = str(getattr(action, "apply_scope", "cell")).strip().lower()
        target_cell_id = getattr(action, "target_cell_id", None)
        if apply_scope == "cluster":
            return [str(cell_id) for cell_id in sorted(managed_cells.keys(), key=lambda item: str(item))]
        if target_cell_id is not None and str(target_cell_id).strip() != "":
            normalized_target_cell_id = str(target_cell_id).strip()
            if normalized_target_cell_id in managed_cells:
                return [normalized_target_cell_id]
        return [str(getattr(self.state.cell, "cell_id", "cell_0"))]

    def _refresh_runtime_views_after_config_apply(self) -> None:
        self._sync_beam_state_from_cell()
        self._refresh_managed_cell_runtime_views(self.state)

    def _apply_rt_config_action(self, action, apply_slot):
        provenances=getattr(action,'_library_provenances',[])
        if any(not self.controller_manager.instance_selection.valid(p) for p in provenances):
            outcome=self._build_config_actuation_outcome(action=action,apply_slot=int(self.state.slot_id),
                status='rejected',reason='stale_xapp_selection',target_cell_ids=self._resolve_config_target_cell_ids(action))
            outcome.update(slot_id=self.state.slot_id,provenance=provenances)
            self.state.control.actuation_outcomes.append(outcome)
            self._update_e2_control_request_status(action.control_request_id,status='rejected',reason='stale_xapp_selection')
            return
        owners=getattr(action,'_rule_owners',())
        if owners and any(not self.controller_manager._is_xapp_enabled(owner) for owner in owners):
            self.state.control.actuation_outcomes.append(dict(kind='config',status='rejected',reason='rule_withdrawn_before_actuation',
                slot_id=self.state.slot_id,source=action.source_name,owners=list(owners)))
            return
        from phy.rt_sys.control import audit_action,runtime_control_values
        fields=audit_action(action)
        scope=str(getattr(action,'apply_scope','cell')).lower()
        target=getattr(action,'target_cell_id',None)
        reason=None
        if scope not in ('cell','cluster'):reason='unsupported_scope'
        elif scope=='cell' and target is not None and target not in self.state.managed_cells:reason='unknown_target_cell'
        elif any(f['route']=='unsupported' for f in fields):reason='unsupported_fields:'+','.join(f['field'] for f in fields if f['route']=='unsupported')
        elif action.effective_until_slot is not None and self.state.slot_id>action.effective_until_slot:reason='expired_before_actuation'
        elif self._rng.random()<self._control_reject_prob:reason='probabilistic_reject'
        targets=[] if reason in ('unknown_target_cell','unsupported_scope') else self._resolve_config_target_cell_ids(action)
        before=runtime_control_values(self.state,targets)
        from phy.rt_sys.paper_controls import validate_transaction
        if reason is None:reason=validate_transaction(self.state,action,targets)
        if reason is None:
            action._paper_hysteresis_changed=action.steering_hysteresis_db is not None and any(self.state.managed_cells[c].cell.steering_hysteresis_db!=action.steering_hysteresis_db for c in targets)
            applied=self.executor.apply_action(self.state,action)
            if applied and getattr(self.state,'paper',None):
                from phy.rt_sys.paper_controls import apply_values
                apply_values(self.state,action,applied)
            if not applied:reason='no_runtime_consumer_applied'
            self._refresh_runtime_views_after_config_apply()
        else:applied=[]
        outcome=self._build_config_actuation_outcome(action=action,apply_slot=int(self.state.slot_id),
            status='rejected' if reason else 'applied',reason=reason,target_cell_ids=applied if not reason else targets)
        outcome.update(requested_apply_slot=int(apply_slot),fields=fields,before=before,
            after=runtime_control_values(self.state,targets),validity_semantics='Persistent target state; expiry bounds admission, not automatic rollback')
        if provenances:outcome['provenance']=provenances
        self.state.control.actuation_outcomes.append(outcome)

    def _build_config_actuation_outcome(
        self,
        *,
        action: ConfigAction,
        apply_slot: int,
        status: str,
        reason: str | None = None,
        target_cell_ids: list[str] | None = None,
    ) -> dict[str, object]:
        metadata = getattr(action, "metadata", None)
        outcome = {
            "kind": "config",
            "issued_at_slot": int(getattr(metadata, "issued_at_slot", -1) if getattr(metadata, "issued_at_slot", -1) is not None else -1),
            "apply_slot": int(apply_slot),
            "source": str(getattr(metadata, "source", getattr(action, "source_name", "unknown"))),
            "decision_id": str(getattr(metadata, "decision_id", "")),
            "control_request_id": getattr(action, "control_request_id", None),
            "status": str(status),
            "apply_scope": str(getattr(action, "apply_scope", "cell")),
            "target_cell_id": getattr(action, "target_cell_id", None),
            "target_cell_ids": list(target_cell_ids or []),
        }
        if reason is not None and str(reason).strip() != "":
            outcome["reason"] = str(reason)
        return outcome

    def _update_e2_control_request_status(
        self,
        request_id: object | None,
        *,
        status: str,
        reason: str = "",
    ) -> None:
        resources = getattr(self.state.control, "oran_lifecycle_resources", None)
        if not isinstance(resources, OranLifecycleResources):
            return
        normalized_request_id = "" if request_id is None else str(request_id).strip()
        if normalized_request_id == "":
            return
        for request in list(getattr(resources, "e2_control_requests", []) or []):
            if str(getattr(request, "request_id", "")) != normalized_request_id:
                continue
            request.status = str(status)
            request.reason = str(reason)
            request.updated_at_slot = int(self.state.slot_id)
            return

    def _build_e2_control_outcome(
        self,
        *,
        outcome: dict[str, object],
        index: int,
    ) -> E2ControlOutcome | None:
        request_id = outcome.get("control_request_id", outcome.get("decision_id"))
        if request_id is None or str(request_id).strip() == "":
            return None
        apply_slot = outcome.get("apply_slot", self.state.slot_id)
        if not isinstance(apply_slot, int):
            try:
                apply_slot = int(str(apply_slot))
            except (TypeError, ValueError):
                apply_slot = int(self.state.slot_id)
        raw_target_cell_ids = outcome.get("target_cell_ids", [])
        target_cell_ids = (
            list(raw_target_cell_ids)
            if isinstance(raw_target_cell_ids, (list, tuple, set))
            else []
        )
        return E2ControlOutcome(
            outcome_id=f"e2_control_outcome:{int(self.state.slot_id)}:{int(index)}",
            request_id=str(request_id),
            kind=str(outcome.get("kind", "config")),
            source_name=str(outcome.get("source", "unknown")),
            status=str(outcome.get("status", "unknown")),
            applied_at_slot=int(apply_slot),
            target_cell_ids=[str(item) for item in target_cell_ids],
            reason=str(outcome.get("reason", "")),
            decision_id=(
                str(outcome.get("decision_id"))
                if outcome.get("decision_id", None) is not None
                else None
            ),
        )

    def _refresh_control_lifecycle_outcomes(self) -> None:
        resources = getattr(self.state.control, "oran_lifecycle_resources", None)
        if not isinstance(resources, OranLifecycleResources):
            return
        resources.e2_control_outcomes = []
        for index, outcome in enumerate(list(getattr(self.state.control, "actuation_outcomes", []) or [])):
            e2_outcome = self._build_e2_control_outcome(
                outcome=dict(outcome),
                index=int(index),
            )
            if e2_outcome is None:
                continue
            resources.e2_control_outcomes.append(e2_outcome)
            self._update_e2_control_request_status(
                e2_outcome.request_id,
                status=str(e2_outcome.status),
                reason=str(e2_outcome.reason),
            )

    def _build_phy_runtime_context(
        self,
        scoped_state: GlobalState | SimpleNamespace | None = None,
    ) -> PhyRuntimeContext:
        """
        Builds one PHY runtime context object from the current global state.
        INPUT
        No explicit input.
        OUTPUT
        One PhyRuntimeContext object for the current slot.
        EXAMPLE OUTPUT
        PhyRuntimeContext(slot_id=1, current_codebook_id=0)
        """
        state = self.state if scoped_state is None else scoped_state
        return PhyRuntimeContext(
            slot_id=state.slot_id,
            sim_time_ms=state.sim_time_ms,
            slot_duration_ms=self.config.slot_duration_ms,
            cell_id=state.cell.cell_id,
            scheduler_mode=state.scheduler.scheduler_mode,
            tx_power_dbm=state.cell.tx_power_dbm,
            current_codebook_id=state.cell.current_codebook_id,
            current_beam_profile_id=state.cell.current_beam_profile_id,
            carrier_frequency_hz=float(getattr(state.antenna, "carrier_frequency_hz", 3.5e9)),
            bandwidth_hz=float(getattr(state.antenna, "bandwidth_hz", 20e6)),
            subcarrier_spacing_hz=float(getattr(state.antenna, "subcarrier_spacing_hz", 30e3)),
            cyclic_prefix_type=str(getattr(state.antenna, "cyclic_prefix_type", "normal")),
            num_ofdm_symbols_per_slot=int(getattr(state.antenna, "num_ofdm_symbols_per_slot", 14)),
            n_prbs_total=int(getattr(getattr(state, "resource_grid", None), "n_prbs_total", 0)),
            pathloss_db=state.channel.pathloss_db,
            shadowing_db=state.channel.shadowing_db,
            channel_profile=state.channel.channel_profile,
            codebook_id=state.beam.codebook_id,
            beam_profile_id=state.beam.beam_profile_id,
            doppler_hz=state.channel.doppler_hz,
        )

    def _build_phy_input(
        self,
        scoped_state: GlobalState | SimpleNamespace | None = None,
    ) -> PhyInput:
        """
        Builds one standardized PHY input object from the current global state.
        INPUT
        No explicit input.
        OUTPUT
        One PhyInput object for the current slot.
        EXAMPLE OUTPUT
        PhyInput(slot_id=1, ue_inputs=[...])
        """
        state = self.state if scoped_state is None else scoped_state
        scheduling_decision = getattr(state.scheduler, "last_decision", None)
        grants = list(getattr(state.scheduler, "grants", []))
        grant_map = {str(grant.ue_id): grant for grant in grants}
        scheduled_ue_set = set(str(item) for item in getattr(state.scheduler, "scheduled_ue_ids", []))

        ue_inputs = []
        for ue in state.ues:
            link_state = getattr(state, "links", {}).get(str(ue.ue_id), None)
            beamforming_state = getattr(state, "beamforming", {}).get(str(ue.ue_id), None)
            grant = grant_map.get(str(ue.ue_id))
            resource_grid_slice = self._build_phy_resource_grid_slice(grant=grant)
            link_large_scale = getattr(link_state, "large_scale", None)
            link_small_scale = getattr(link_state, "small_scale", None)
            ue_inputs.append(
                PhyUeInput(
                    ue_id=ue.ue_id,
                    queue_bytes=ue.queue_bytes,
                    sinr_db=ue.sinr_db,
                    radio_quality_db=ue.radio_quality_db,
                    bler=ue.bler,
                    serving_beam_id=ue.serving_beam_id,
                    is_scheduled=str(ue.ue_id) in scheduled_ue_set,
                    slice_id=ue.slice_id,
                    traffic_type=ue.traffic_type,
                    latency_budget_ms=ue.latency_budget_ms,
                    priority=ue.priority,
                    allocated_prbs=int(getattr(grant, "allocated_prbs", 0)) if grant is not None else 0,
                    allocated_symbols=int(getattr(grant, "allocated_symbols", 0)) if grant is not None else 0,
                    start_prb=int(getattr(grant, "start_prb", 0)) if grant is not None else 0,
                    start_symbol=int(getattr(grant, "start_symbol", 0)) if grant is not None else 0,
                    resource_block_indices=list(resource_grid_slice.resource_block_indices),
                    allocated_subcarrier_indices=list(resource_grid_slice.allocated_subcarrier_indices),
                    allocated_symbol_indices=list(resource_grid_slice.allocated_symbol_indices),
                    resource_grid_slice=resource_grid_slice,
                    is_retx=bool(getattr(grant, "is_retx", False)) if grant is not None else False,
                    tx_contract=getattr(grant, "tx_contract", None),
                    requested_bytes=int(getattr(grant, "requested_bytes", 0)) if grant is not None else 0,
                    grant_bytes_budget=int(getattr(grant, "grant_bytes_budget", 0)) if grant is not None else 0,
                    priority_score=float(getattr(grant, "priority_score", 0.0)) if grant is not None else 0.0,
                    target_mcs=getattr(grant, "target_mcs", None) if grant is not None else None,
                    mcs_cap=getattr(grant, "mcs_cap", None) if grant is not None else None,
                    mcs_table_id=str(getattr(grant, "mcs_table_id", "qam64")) if grant is not None else "qam64",
                    tbs_selection_method=str(getattr(grant, "tbs_selection_method", "resource_elements"))
                    if grant is not None else "resource_elements",
                    dmrs_pattern_id=getattr(grant, "dmrs_pattern_id", None) if grant is not None else None,
                    harq_process_id=getattr(grant, "harq_process_id", None) if grant is not None else None,
                    harq_redundancy_version=getattr(grant, "harq_redundancy_version", None) if grant is not None else None,
                    harq_ndi=getattr(grant, "harq_ndi", None) if grant is not None else None,
                    harq_retransmission_round=int(getattr(grant, "harq_retransmission_round", 0)) if grant is not None else 0,
                    cqi_validity_age_slots=int(getattr(grant, "cqi_validity_age_slots", 0)) if grant is not None else 0,
                    ri_validity_age_slots=int(getattr(grant, "ri_validity_age_slots", 0)) if grant is not None else 0,
                    pmi_validity_age_slots=int(getattr(grant, "pmi_validity_age_slots", 0)) if grant is not None else 0,
                    channel_quality_confidence=(
                        float(getattr(grant, "channel_quality_confidence", 1.0))
                        if grant is not None and getattr(grant, "channel_quality_confidence", None) is not None
                        else None
                    ),
                    link_adaptation_reason=(
                        str(getattr(grant, "link_adaptation_reason", ""))
                        if grant is not None and getattr(grant, "link_adaptation_reason", None) is not None
                        else None
                    ),
                    spatial_group_id=getattr(grant, "spatial_group_id", None) if grant is not None else None,
                    spatial_group_size=int(getattr(grant, "spatial_group_size", 1)) if grant is not None else 1,
                    spatial_compatibility_score=float(getattr(grant, "spatial_compatibility_score", 0.0))
                    if grant is not None else 0.0,
                    link_pathloss_db=float(
                        getattr(link_large_scale, "pathloss_db", getattr(link_state, "pathloss_db", state.channel.pathloss_db))
                    )
                    if link_state is not None else float(state.channel.pathloss_db),
                    link_shadowing_db=float(
                        getattr(link_large_scale, "shadowing_db", getattr(link_state, "shadowing_db", state.channel.shadowing_db))
                    )
                    if link_state is not None else float(state.channel.shadowing_db),
                    link_los_state=str(
                        getattr(link_large_scale, "los_state", getattr(link_state, "los_state", "los"))
                    ) if link_state is not None else "los",
                    link_o2i_penetration_loss_db=float(
                        getattr(
                            link_large_scale,
                            "o2i_penetration_loss_db",
                            getattr(link_state, "o2i_penetration_loss_db", 0.0),
                        )
                    ) if link_state is not None else 0.0,
                    link_doppler_hz=float(
                        getattr(link_small_scale, "doppler_hz", getattr(link_state, "doppler_hz", state.channel.doppler_hz))
                    )
                    if link_state is not None else float(state.channel.doppler_hz),
                    link_delay_spread_ns=float(
                        getattr(link_small_scale, "delay_spread_ns", getattr(link_state, "delay_spread_ns", 300.0))
                    )
                    if link_state is not None else 300.0,
                    link_k_factor_db=float(
                        getattr(link_small_scale, "k_factor_db", getattr(link_state, "k_factor_db", 0.0))
                    ) if link_state is not None else 0.0,
                    link_angle_spread_deg=float(
                        getattr(
                            link_small_scale,
                            "angle_spread_deg",
                            getattr(link_state, "angle_spread_deg", 35.0),
                        )
                    ) if link_state is not None else 35.0,
                    link_spatial_correlation_level=float(
                        getattr(
                            link_small_scale,
                            "spatial_correlation_level",
                            getattr(link_state, "spatial_correlation_level", 0.5),
                        )
                    ) if link_state is not None else 0.5,
                    link_channel_model_type=str(
                        getattr(
                            link_small_scale,
                            "channel_model_type",
                            getattr(link_state, "channel_model_type", state.channel.channel_profile),
                        )
                    )
                    if link_state is not None else str(state.channel.channel_profile),
                    link_fast_fading_seed=int(
                        getattr(link_small_scale, "fast_fading_seed", getattr(link_state, "fast_fading_seed", 0))
                    )
                    if link_state is not None else 0,
                    link_spatial_corr_seed=int(
                        getattr(
                            link_small_scale,
                            "spatial_corr_seed",
                            getattr(link_state, "spatial_corr_seed", 0),
                        )
                    )
                    if link_state is not None else 0,
                    reported_cqi=int(getattr(link_state, "cqi", getattr(ue, "cqi", 0)))
                    if link_state is not None else int(getattr(ue, "cqi", 0)),
                    reported_ri=int(getattr(link_state, "ri", getattr(ue, "ri", 1)))
                    if link_state is not None else int(getattr(ue, "ri", 1)),
                    reported_pmi=int(getattr(link_state, "pmi", getattr(ue, "pmi", 0)))
                    if link_state is not None else int(getattr(ue, "pmi", 0)),
                    num_tx_ant=int(getattr(state.antenna, "gnb_num_tx_ant", 1)),
                    num_rx_ant=int(
                        getattr(state.antenna, "ue_num_rx_ant_by_ue", {}).get(str(ue.ue_id), 1)
                    ),
                    rank_indicator=int(getattr(beamforming_state, "rank", getattr(ue, "rank", 1)))
                    if beamforming_state is not None else int(getattr(ue, "rank", 1)),
                    layer_count=(grant.tx_contract.layers if getattr(grant, 'tx_contract', None) is not None else
                        (int(getattr(beamforming_state, "layer_count", getattr(ue, "ri", 1)))
                         if beamforming_state is not None else int(getattr(ue, "ri", 1)))) ,
                    precoder_id=int(getattr(beamforming_state, "precoder_id", state.cell.current_codebook_id))
                    if beamforming_state is not None else int(state.cell.current_codebook_id),
                    combiner_id=int(getattr(beamforming_state, "combiner_id", getattr(ue, "pmi", 0)))
                    if beamforming_state is not None else int(getattr(ue, "pmi", 0)),
                    precoder_matrix=list(getattr(beamforming_state, "precoder_matrix", []) or [])
                    if beamforming_state is not None else [],
                    combiner_vector=list(getattr(beamforming_state, "combiner_vector", []) or [])
                    if beamforming_state is not None else [],
                    precoding_matrix_label=str(getattr(beamforming_state, "precoding_matrix_label", ""))
                    if beamforming_state is not None else None,
                    beam_direction_deg=float(getattr(beamforming_state, "beam_direction_deg", 0.0))
                    if beamforming_state is not None else 0.0,
                    beamforming_gain_db=float(getattr(beamforming_state, "beam_gain_db", 0.0))
                    if beamforming_state is not None else 0.0,
                )
            )

        runtime_context = self._build_phy_runtime_context(state)

        return PhyInput(
            slot_id=state.slot_id,
            sim_time_ms=state.sim_time_ms,
            cell_id=state.cell.cell_id,
            tx_power_dbm=runtime_context.tx_power_dbm,
            current_codebook_id=runtime_context.current_codebook_id,
            current_beam_profile_id=runtime_context.current_beam_profile_id,
            scheduled_ue_ids=list(state.scheduler.scheduled_ue_ids),
            ue_inputs=ue_inputs,
            runtime_context=runtime_context,
            scheduling_decision=scheduling_decision,
        )

    def _stamp_observation_delivery(
        self,
        observation: Observation,
        *,
        delivered_at_slot: int,
    ) -> Observation:
        delivered_slot = int(delivered_at_slot)

        if observation.produced_at_slot is None:
            observation.produced_at_slot = observation.measurement_produced_at_slot
        observation.report_delay_slots = self._obs_delay_slots
        observation.delivered_at_slot = delivered_slot

        if observation.measurement_produced_at_slot is None:
            observation.measurement_produced_at_slot = observation.produced_at_slot
        observation.measurement_delivered_at_slot = delivered_slot

        if observation.produced_at_slot is not None:
            observation.measurement_age_slots = max(
                0,
                delivered_slot - int(observation.measurement_produced_at_slot or observation.produced_at_slot),
            )

        for ue_observation in observation.ue_observations:
            if ue_observation.measurement_produced_at_slot is None:
                ue_observation.measurement_produced_at_slot = observation.measurement_produced_at_slot
            ue_observation.measurement_delivered_at_slot = delivered_slot
            for candidate in ue_observation.candidate_cell_observations:
                candidate.measurement_delivered_at_slot = delivered_slot
            if ue_observation.measurement_produced_at_slot is not None:
                ue_observation.measurement_age_slots = max(
                    0,
                    delivered_slot - int(ue_observation.measurement_produced_at_slot),
                )

        return observation

    def _extract_observation_trace(self, observation: Observation | None) -> dict[str, object]:
        if observation is None:
            return {}

        return {
            "observation_slot_id": int(getattr(observation, "slot_id", -1)),
            "observation_produced_at_slot": getattr(observation, "produced_at_slot", None),
            "observation_delivered_at_slot": getattr(observation, "delivered_at_slot", None),
            "measurement_produced_at_slot": getattr(observation, "measurement_produced_at_slot", None),
            "measurement_delivered_at_slot": getattr(observation, "measurement_delivered_at_slot", None),
            "measurement_age_slots": getattr(observation, "measurement_age_slots", None),
            "measurement_queue_delay_slots": getattr(observation, "measurement_queue_delay_slots", None),
            "measurement_sinr_window_slots": getattr(observation, "measurement_sinr_window_slots", None),
            "measurement_sinr_noise_std_db_applied": getattr(
                observation,
                "measurement_sinr_noise_std_db_applied",
                None,
            ),
            "measurement_bler_window_tbs": getattr(observation, "measurement_bler_window_tbs", None),
            "measurement_throughput_granularity_slots": getattr(
                observation,
                "measurement_throughput_granularity_slots",
                None,
            ),
            "measurement_queue_age_slots_mean": getattr(
                observation,
                "measurement_queue_age_slots_mean",
                None,
            ),
            "measurement_sinr_effective_sample_count_mean": getattr(
                observation,
                "measurement_sinr_effective_sample_count_mean",
                None,
            ),
            "measurement_bler_sample_count_mean": getattr(
                observation,
                "measurement_bler_sample_count_mean",
                None,
            ),
            "measurement_throughput_sample_count_mean": getattr(
                observation,
                "measurement_throughput_sample_count_mean",
                None,
            ),
        }

    def _summarize_numeric_series(self, values: Sequence[object | None]) -> dict[str, object]:
        normalized_values: list[float] = []
        for value in values:
            if value is None:
                continue

            if isinstance(value, bool):
                normalized_values.append(float(value))
                continue

            if isinstance(value, (int, float)):
                normalized_values.append(float(value))
                continue

            if isinstance(value, str):
                try:
                    normalized_values.append(float(value))
                except ValueError:
                    continue
                continue

        if len(normalized_values) == 0:
            return {
                "count": 0,
                "min": None,
                "mean": None,
                "max": None,
            }

        return {
            "count": int(len(normalized_values)),
            "min": float(min(normalized_values)),
            "mean": float(sum(normalized_values) / float(len(normalized_values))),
            "max": float(max(normalized_values)),
        }

    def _build_trace_stats(
        self,
        records: list[dict],
        *,
        trace_key: str,
        field_names: list[str],
    ) -> dict[str, dict[str, object]]:
        trace_stats: dict[str, dict[str, object]] = {}
        for field_name in field_names:
            values = [
                dict(record.get(trace_key, {})).get(field_name)
                for record in records
            ]
            trace_stats[field_name] = self._summarize_numeric_series(values)
        return trace_stats

    def _build_final_policy_snapshot_summary(self) -> dict[str, object]:
        controller_manager = getattr(self, "controller_manager", None)
        if controller_manager is None or not hasattr(controller_manager, "policy_store"):
            return {
                "policy_epoch_id": 0,
                "active_intent_id": "",
                "intent_changed_this_epoch": False,
                "translation_refreshed_this_epoch": False,
                "structured_goal_id": "",
                "validated_template": {},
                "orchestration_plan_view": {},
                "selection_plan_id": "",
                "reward_spec": {},
                "rl_guidance": {},
            }
        policy_snapshot = controller_manager.policy_store.get_snapshot()
        validated_template = getattr(policy_snapshot, "validated_intent_template", None)
        validation_report = getattr(validated_template, "validation_report", None)
        projection_report = getattr(validated_template, "projection_report", None)
        fallback_report = getattr(validated_template, "fallback_report", None)
        orchestration_plan_view = getattr(policy_snapshot, "orchestration_plan_view", None)
        rl_guidance = getattr(policy_snapshot, "rl_guidance", None)
        selected_safety_rule_set = getattr(rl_guidance, "selected_safety_rule_set", None)
        reward_spec = getattr(policy_snapshot, "reward_spec", None)
        plan = getattr(policy_snapshot, "xapp_selection_plan", None)
        goal = getattr(policy_snapshot, "structured_goal", None)
        return {
            "policy_epoch_id": int(getattr(policy_snapshot, "policy_epoch_id", 0) or 0),
            "active_intent_id": str(getattr(policy_snapshot, "active_intent_id", "") or ""),
            "intent_changed_this_epoch": bool(
                getattr(policy_snapshot, "intent_changed_this_epoch", False)
            ),
            "translation_refreshed_this_epoch": bool(
                getattr(policy_snapshot, "translation_refreshed_this_epoch", False)
            ),
            "structured_goal_id": str(getattr(goal, "goal_id", "") or ""),
            "validated_template": {
                "template_id": str(getattr(validated_template, "template_id", "") or ""),
                "raw_kpi_count": int(
                    len(
                        list(
                            dict(getattr(validated_template, "raw_llm_candidate", {}) or {}).get(
                                "kpi_targets",
                                [],
                            )
                            or []
                        )
                    )
                )
                if validated_template is not None
                else 0,
                "final_kpi_count": int(
                    len(list(getattr(validated_template, "kpi_template", []) or []))
                )
                if validated_template is not None
                else 0,
                "schema_ok": getattr(validation_report, "schema_ok", None),
                "admissibility_ok": getattr(validation_report, "admissibility_ok", None),
                "direction_ok": getattr(validation_report, "direction_ok", None),
                "range_ok": getattr(validation_report, "range_ok", None),
                "projected": bool(getattr(projection_report, "projected", False))
                if projection_report is not None
                else False,
                "fallback_used": bool(getattr(fallback_report, "fallback_used", False))
                if fallback_report is not None
                else False,
                "fallback_reason": str(getattr(fallback_report, "fallback_reason", "") or "")
                if fallback_report is not None
                else "",
            },
            "orchestration_plan_view": {
                "plan_id": str(getattr(orchestration_plan_view, "plan_id", "") or ""),
                "selected_xapps": list(getattr(orchestration_plan_view, "x", []) or []),
                "action_bound_count": int(len(list(getattr(orchestration_plan_view, "B", []) or []))),
                "chi_profile_name": getattr(getattr(orchestration_plan_view, "chi", None), "profile_name", None),
                "chi_defaulted": bool(getattr(getattr(orchestration_plan_view, "chi", None), "defaulted", True))
                if getattr(orchestration_plan_view, "chi", None) is not None
                else True,
            },
            "selection_plan_id": str(getattr(plan, "plan_id", "") or ""),
            "reward_spec": {
                "reward_id": str(getattr(reward_spec, "reward_id", "") or ""),
                "reward_term_keys": sorted(
                    str(key) for key in dict(getattr(reward_spec, "reward_terms", {}) or {}).keys()
                ),
            },
            "rl_guidance": {
                "guidance_id": str(getattr(rl_guidance, "guidance_id", "") or ""),
                "source_epoch": getattr(rl_guidance, "source_epoch", None),
                "derived_from_intent_id": str(getattr(rl_guidance, "derived_from_intent_id", "") or ""),
                "fallback_used": bool(getattr(rl_guidance, "fallback_used", False)),
                "guided_action_region_count": int(
                    len(list(getattr(rl_guidance, "guided_action_region", []) or []))
                )
                if rl_guidance is not None
                else 0,
                "selected_rule_ids": [
                    str(getattr(rule_ref, "rule_id", "") or "")
                    for rule_ref in list(
                        getattr(selected_safety_rule_set, "selected_rule_refs", []) or []
                    )
                ]
                if selected_safety_rule_set is not None
                else [],
            },
        }

    def _build_final_summary(self) -> dict:
        """
        Builds the final simulation summary dictionary.
        INPUT
        No explicit input.
        OUTPUT
        One summary dictionary for JSON export.
        EXAMPLE OUTPUT
        {"total_slots": 30, "final_avg_sinr_db": 13.5}
        """
        latest_record = self.slot_records[-1] if len(self.slot_records) > 0 else None
        final_observation = None if latest_record is not None else build_observation(self.state)

        last_active_record = None
        for record in reversed(self.slot_records):
            true_metrics = dict(record.get("true_metrics", {}))
            if str(true_metrics.get("scheduled_ue_ids", "")) or int(true_metrics.get("granted_ue_count", 0)) > 0:
                last_active_record = record
                break

        total_pending_retx_bytes = sum(ue.harq.pending_retx_bytes for ue in self.state.ues)
        ack_count = sum(1 for ue in self.state.ues if ue.harq.last_decision == "ACK")
        nack_count = sum(1 for ue in self.state.ues if ue.harq.last_decision == "NACK")
        ack_rate = (
            float(ack_count) / float(ack_count + nack_count)
            if (ack_count + nack_count) > 0 else 0.0
        )
        avg_radio_quality_db = (
            sum(ue.radio_quality_db for ue in self.state.ues) / float(len(self.state.ues))
            if len(self.state.ues) > 0 else 0.0
        )
        cluster_scheduled_ue_ids = get_cluster_scheduled_ue_ids(self.state)

        fallback_true_metrics = {
            "queue_bytes": int(sum(ue.queue_bytes for ue in self.state.ues)),
            "throughput_mbps": float(sum(ue.throughput_mbps for ue in self.state.ues)),
            "scheduled_ue_ids": ";".join(str(item) for item in cluster_scheduled_ue_ids),
            "scheduled_ue_count": int(len(cluster_scheduled_ue_ids)),
            "connected_ue_count": int(
                sum(
                    1
                    for connection in getattr(self.state, "connections", {}).values()
                    if str(getattr(connection, "connection_state", "connected")) == "connected"
                )
            ) if getattr(self.state, "connections", None) else int(len(self.state.ues)),
            "out_of_sync_ue_count": int(
                sum(
                    1
                    for connection in getattr(self.state, "connections", {}).values()
                    if str(getattr(connection, "connection_state", "connected")) == "out_of_sync"
                )
            ) if getattr(self.state, "connections", None) else 0,
            "scheduler_mode": self.state.scheduler.scheduler_mode,
            "tx_power_dbm": float(self.state.cell.tx_power_dbm),
            "carrier_frequency_hz": float(getattr(getattr(self.state, "antenna", None), "carrier_frequency_hz", 0.0)),
            "bandwidth_hz": float(getattr(getattr(self.state, "antenna", None), "bandwidth_hz", 0.0)),
            "subcarrier_spacing_hz": float(getattr(getattr(self.state, "antenna", None), "subcarrier_spacing_hz", 0.0)),
            "cyclic_prefix_type": str(getattr(getattr(self.state, "antenna", None), "cyclic_prefix_type", "normal")),
            "num_ofdm_symbols_per_slot": int(getattr(getattr(self.state, "antenna", None), "num_ofdm_symbols_per_slot", 0)),
            "n_prbs_total": int(getattr(getattr(self.state, "resource_grid", None), "n_prbs_total", 0)),
            "current_codebook_id": int(self.state.cell.current_codebook_id),
            "current_beam_profile_id": int(self.state.cell.current_beam_profile_id),
            "total_pending_retx_bytes": int(total_pending_retx_bytes),
            "ack_count_this_slot": int(ack_count),
            "nack_count_this_slot": int(nack_count),
            "ack_rate_this_slot": float(ack_rate),
            "avg_radio_quality_db_all": float(avg_radio_quality_db),
        }
        fallback_observable_metrics = {
            "queue_bytes": int(getattr(final_observation, "total_queue_bytes", 0)),
            "throughput_mbps": float(getattr(final_observation, "total_throughput_mbps", 0.0)),
            "sinr_db": getattr(final_observation, "avg_sinr_db", None),
            "bler": getattr(final_observation, "avg_bler", None),
            "sinr_db_scheduled": getattr(final_observation, "avg_sinr_db_scheduled", None),
            "bler_scheduled": getattr(final_observation, "avg_bler_scheduled", None),
            "sinr_db_all": getattr(final_observation, "avg_sinr_db_all", None),
            "radio_quality_db_all": getattr(final_observation, "avg_radio_quality_db_all", None),
            "scheduled_ue_ids": ";".join(
                str(item) for item in getattr(final_observation, "scheduled_ue_ids", [])
            ),
            "scheduled_ue_count": int(getattr(final_observation, "num_scheduled_ues", 0)),
            "connected_ue_count": int(getattr(final_observation, "connected_ue_count", 0)),
            "out_of_sync_ue_count": int(getattr(final_observation, "out_of_sync_ue_count", 0)),
            "carrier_frequency_hz": getattr(final_observation, "carrier_frequency_hz", None),
            "bandwidth_hz": getattr(final_observation, "bandwidth_hz", None),
            "subcarrier_spacing_hz": getattr(final_observation, "subcarrier_spacing_hz", None),
            "cyclic_prefix_type": getattr(final_observation, "cyclic_prefix_type", None),
            "num_ofdm_symbols_per_slot": getattr(final_observation, "num_ofdm_symbols_per_slot", None),
            "n_prbs_total": getattr(final_observation, "resource_grid_n_prbs_total", None),
        }
        fallback_measurement_trace = {
            "observation_slot_id": int(getattr(final_observation, "slot_id", -1)),
            "observation_produced_at_slot": getattr(final_observation, "produced_at_slot", None),
            "observation_delivered_at_slot": getattr(final_observation, "delivered_at_slot", None),
            "produced_at_slot": getattr(final_observation, "measurement_produced_at_slot", None),
            "delivered_at_slot": getattr(final_observation, "measurement_delivered_at_slot", None),
            "age_slots": getattr(final_observation, "measurement_age_slots", None),
            "queue_delay_slots_applied": getattr(final_observation, "measurement_queue_delay_slots", None),
            "queue_age_slots_mean": getattr(final_observation, "measurement_queue_age_slots_mean", None),
            "sinr_window_slots": getattr(final_observation, "measurement_sinr_window_slots", None),
            "sinr_noise_std_db_applied": getattr(
                final_observation,
                "measurement_sinr_noise_std_db_applied",
                None,
            ),
            "sinr_effective_sample_count_mean": getattr(
                final_observation,
                "measurement_sinr_effective_sample_count_mean",
                None,
            ),
            "bler_window_tbs": getattr(final_observation, "measurement_bler_window_tbs", None),
            "bler_sample_count_mean": getattr(
                final_observation,
                "measurement_bler_sample_count_mean",
                None,
            ),
            "throughput_granularity_slots": getattr(
                final_observation,
                "measurement_throughput_granularity_slots",
                None,
            ),
            "throughput_sample_count_mean": getattr(
                final_observation,
                "measurement_throughput_sample_count_mean",
                None,
            ),
        }
        fallback_phy_comparison_trace = {}
        measurement_trace_stats = self._build_trace_stats(
            self.slot_records,
            trace_key="measurement_trace",
            field_names=[
                "age_slots",
                "queue_age_slots_mean",
                "sinr_effective_sample_count_mean",
                "bler_sample_count_mean",
                "throughput_sample_count_mean",
            ],
        )
        control_input_measurement_trace_stats = self._build_trace_stats(
            self.slot_records,
            trace_key="control_trace",
            field_names=[
                "input_measurement_age_slots",
                "input_queue_age_slots_mean",
                "input_sinr_effective_sample_count_mean",
                "input_bler_sample_count_mean",
                "input_throughput_sample_count_mean",
            ],
        )
        phy_comparison_trace_stats = self._build_trace_stats(
            self.slot_records,
            trace_key="phy_comparison_trace",
            field_names=[
                "compared_ue_count",
                "ack_match_rate",
                "mean_abs_sinr_gap_db",
                "mean_abs_post_eq_sinr_gap_db",
                "max_abs_post_eq_sinr_gap_db",
                "mean_abs_bler_gap",
                "mean_abs_throughput_gap_mbps",
                "mean_abs_tb_size_gap_bytes",
                "mean_abs_spatial_multiplexing_gain_gap_db",
                "spatial_mode_match_rate",
            ],
        )
        final_per_ue_snapshot = []
        if hasattr(self.state, "snapshots") and len(getattr(self.state, "snapshots", [])) > 0:
            final_per_ue_snapshot = list(self.state.snapshots[-1].get("ues", []))

        executed_slots = int(len(self.slot_records))
        configured_total_slots = int(getattr(self.config, "total_slots", 0) or 0)
        controller_manager = getattr(self, "controller_manager", None)
        orchestration_epoch_history = (
            controller_manager.get_orchestration_epoch_history()
            if controller_manager is not None
            and hasattr(controller_manager, "get_orchestration_epoch_history")
            else []
        )
        summary = {
            "total_slots": executed_slots,
            "configured_total_slots": configured_total_slots,
            "run_forever_configured": bool(configured_total_slots <= 0),
            "slot_duration_ms": self.config.slot_duration_ms,
            "cell_id": self.state.cell.cell_id,
            "num_ues": len(self.state.ues),
            "carrier_frequency_hz": float(getattr(getattr(self.state, "antenna", None), "carrier_frequency_hz", 0.0)),
            "bandwidth_hz": float(getattr(getattr(self.state, "antenna", None), "bandwidth_hz", 0.0)),
            "subcarrier_spacing_hz": float(getattr(getattr(self.state, "antenna", None), "subcarrier_spacing_hz", 0.0)),
            "cyclic_prefix_type": str(getattr(getattr(self.state, "antenna", None), "cyclic_prefix_type", "normal")),
            "num_ofdm_symbols_per_slot": int(getattr(getattr(self.state, "antenna", None), "num_ofdm_symbols_per_slot", 0)),
            "n_prbs_total": int(getattr(getattr(self.state, "resource_grid", None), "n_prbs_total", 0)),
            "final_true_metrics": dict(latest_record.get("true_metrics", {}))
            if latest_record is not None else fallback_true_metrics,
            "final_observable_metrics": dict(latest_record.get("observable_metrics", {}))
            if latest_record is not None else fallback_observable_metrics,
            "final_measurement_trace": dict(latest_record.get("measurement_trace", {}))
            if latest_record is not None else fallback_measurement_trace,
            "final_control_trace": dict(latest_record.get("control_trace", {}))
            if latest_record is not None else {},
            "final_phy_comparison_trace": dict(latest_record.get("phy_comparison_trace", {}))
            if latest_record is not None else fallback_phy_comparison_trace,
            "measurement_trace_stats": measurement_trace_stats,
            "control_input_measurement_trace_stats": control_input_measurement_trace_stats,
            "phy_comparison_trace_stats": phy_comparison_trace_stats,
            "final_per_ue_snapshot": final_per_ue_snapshot,
        }

        if last_active_record is not None:
            summary.update(
                {
                    "last_active_slot_id": int(last_active_record.get("slot_id", -1)),
                    "last_active_sim_time_ms": float(last_active_record.get("sim_time_ms", 0.0)),
                    "last_active_true_metrics": dict(last_active_record.get("true_metrics", {})),
                    "last_active_observable_metrics": dict(last_active_record.get("observable_metrics", {})),
                    "last_active_measurement_trace": dict(last_active_record.get("measurement_trace", {})),
                    "last_active_control_trace": dict(last_active_record.get("control_trace", {})),
                    "last_active_phy_comparison_trace": dict(last_active_record.get("phy_comparison_trace", {})),
                }
            )

        selection_modes = [
            str(record.get("control_trace", {}).get("rl_selection_mode", "") or "")
            for record in self.slot_records
            if str(record.get("control_trace", {}).get("rl_selection_mode", "") or "") != ""
        ]
        guidance_modes = [
            str(record.get("control_trace", {}).get("rl_guidance_mode", "") or "")
            for record in self.slot_records
            if str(record.get("control_trace", {}).get("rl_guidance_mode", "") or "") != ""
        ]
        safe_baseline_fallback_slot_count = int(
            sum(
                1
                for record in self.slot_records
                if bool(record.get("control_trace", {}).get("rl_safe_baseline_fallback", False))
            )
        )
        if len(selection_modes) > 0:
            summary["near_rt_selection_mode_counts"] = {
                mode: int(sum(1 for candidate in selection_modes if candidate == mode))
                for mode in sorted(set(selection_modes))
            }
        if len(guidance_modes) > 0:
            summary["near_rt_guidance_mode_counts"] = {
                mode: int(sum(1 for candidate in guidance_modes if candidate == mode))
                for mode in sorted(set(guidance_modes))
            }
        summary["safe_baseline_fallback_slot_count"] = safe_baseline_fallback_slot_count
        summary["safe_baseline_fallback_slot_fraction"] = (
            float(safe_baseline_fallback_slot_count) / float(executed_slots)
            if executed_slots > 0
            else 0.0
        )
        summary["method_config"] = self._build_method_config_summary()
        summary["orchestration_epoch_history_summary"] = self._build_orchestration_epoch_history_summary(
            orchestration_epoch_history
        )
        summary["final_policy_snapshot_summary"] = self._build_final_policy_snapshot_summary()
        from experiments.eval_metrics import build_run_metric_summary

        summary["experimental_metrics"] = build_run_metric_summary(
            slot_records=self.slot_records,
            orchestration_epoch_history=orchestration_epoch_history,
        )

        return summary

    def step(self) -> Observation:
        """
        Advances the simulator by one slot.
        INPUT
        No explicit input.
        OUTPUT
        One Observation object for the current slot.
        EXAMPLE OUTPUT
        Observation(slot_id=3, avg_sinr_db=12.3, avg_bler=0.08)
        """
        self.clock.tick()
        self.state.slot_id = self.clock.current_slot
        self.state.sim_time_ms = self.clock.current_time_ms

        # Traffic arrival: bytes arrive into UE queues before control/scheduling.
        arrivals_by_ue, arrivals_by_logical_channel = self.traffic_engine.apply_arrivals(self.state)
        if self._rt_sys_mode:
            self.phy_adapter.bind_state(self.state)
        self.mobility_manager.update(self.state)
        self._refresh_managed_cell_runtime_views(self.state)
        self.state_updater.refresh_logical_channel_groups(self.state)
        arrivals_by_logical_channel_group = self._aggregate_arrivals_by_logical_channel_group(
            arrivals_by_logical_channel=arrivals_by_logical_channel
        )
        self.state.traffic.produced_at_slot = int(self.state.slot_id)
        self.state.traffic.arrival_bytes_by_ue = dict(arrivals_by_ue)
        self.state.traffic.arrival_bytes_by_logical_channel = dict(arrivals_by_logical_channel)
        self.state.traffic.arrival_bytes_by_logical_channel_group = dict(
            arrivals_by_logical_channel_group
        )
        self.state.traffic.arrival_bytes_total = int(sum(arrivals_by_ue.values()))

        debug_print(f"Starting slot {self.state.slot_id}.")

        # Apply delayed control-plane actuations scheduled for this slot.
        self.state.control.actuation_outcomes = []

        due_configs = [item for item in self._pending_config_actuations if item[0] <= self.state.slot_id]
        self._pending_config_actuations = [item for item in self._pending_config_actuations if item[0] > self.state.slot_id]
        due_configs.sort(key=lambda x: (x[0], x[1]))

        for apply_slot, _, action in due_configs:
            if self._rt_sys_mode:
                self._apply_rt_config_action(action,apply_slot)
                continue
            target_cell_ids = self._resolve_config_target_cell_ids(action)
            rejected = self._rng.random() < self._control_reject_prob
            if rejected:
                self.state.control.actuation_outcomes.append(
                    self._build_config_actuation_outcome(
                        action=action,
                        apply_slot=int(apply_slot),
                        status="rejected",
                        reason="probabilistic_reject",
                        target_cell_ids=target_cell_ids,
                    )
                )
                continue

            applied_target_cell_ids = self.executor.apply_action(self.state, action)
            self._refresh_runtime_views_after_config_apply()
            self.state.control.actuation_outcomes.append(
                self._build_config_actuation_outcome(
                    action=action,
                    apply_slot=int(apply_slot),
                    status="applied",
                    target_cell_ids=applied_target_cell_ids,
                )
            )

        commands_to_apply: list[CommandAction] = []

        # Deliver one delayed observation to controllers (if configured).
        delivered_observation = None
        if self._obs_delay_slots <= 0:
            if len(self._observation_queue) > 0:
                delivered_observation = self._observation_queue[-1]
                delivered_observation = self._stamp_observation_delivery(
                    delivered_observation,
                    delivered_at_slot=int(self.state.slot_id),
                )
            else:
                delivered_observation = build_observation(self.state)
                delivered_observation.produced_at_slot = int(self.state.slot_id)
                delivered_observation = self._stamp_observation_delivery(
                    delivered_observation,
                    delivered_at_slot=int(self.state.slot_id),
                )
        else:
            if len(self._observation_queue) >= self._obs_delay_slots and len(self._observation_queue) > 0:
                delivered_observation = self._observation_queue.popleft()
                delivered_observation = self._stamp_observation_delivery(
                    delivered_observation,
                    delivered_at_slot=int(self.state.slot_id),
                )
            else:
                # Warm-up: fall back to a best-effort observation.
                delivered_observation = build_observation(self.state)
                delivered_observation.produced_at_slot = int(self.state.slot_id)
                delivered_observation = self._stamp_observation_delivery(
                    delivered_observation,
                    delivered_at_slot=int(self.state.slot_id),
                )

        self.state.rt_du_observation = delivered_observation
        control_decision = self.controller_manager.update(
            slot_id=self.state.slot_id,
            observation=delivered_observation,
        )

        # Store issued control-plane decision for this slot (debug/analysis).
        if self._rt_sys_mode:
            from phy.rt_sys.control import audit_action
            self._rt_control_audit = [dict(target_cell_id=a.target_cell_id, fields=audit_action(a))
                                     for a in control_decision.config_actions]
        self.state.control.active_config = control_decision.config_action
        self.state.control.active_config_actions = list(control_decision.config_actions)
        self.state.control.pending_commands = list(control_decision.commands)
        self.state.control.active_policy_constraints = list(control_decision.policy_constraints)
        self.state.control.decision_metadata = control_decision.metadata
        self.state.control.arbitration_trace = control_decision.trace
        self.state.control.oran_lifecycle_resources = control_decision.oran_lifecycle_resources
        self.state.control.input_observation_trace = self._extract_observation_trace(
            delivered_observation
        )
        self._refresh_control_lifecycle_outcomes()

        # Apply/enqueue actuation effects (config/commands) with delay.
        if self._rt_sys_mode:
            for config_action in control_decision.config_actions:
                apply_slot=max(int(self.state.slot_id)+int(self._config_delay_slots),
                    int(config_action.effective_from_slot) if config_action.effective_from_slot is not None else int(self.state.slot_id))
                if apply_slot<=self.state.slot_id:self._apply_rt_config_action(config_action,apply_slot)
                else:
                    self._actuation_seq+=1
                    self._pending_config_actuations.append((apply_slot,self._actuation_seq,config_action))
                    self._update_e2_control_request_status(config_action.control_request_id,status='queued',reason='waiting_for_actuation_or_effective_from')
        elif int(self._config_delay_slots) <= 0:
            for config_action in list(control_decision.config_actions):
                target_cell_ids = self._resolve_config_target_cell_ids(config_action)
                rejected = self._rng.random() < self._control_reject_prob
                if rejected:
                    self.state.control.actuation_outcomes.append(
                        self._build_config_actuation_outcome(
                            action=config_action,
                            apply_slot=int(self.state.slot_id),
                            status="rejected",
                            reason="probabilistic_reject",
                            target_cell_ids=target_cell_ids,
                        )
                    )
                    continue
                applied_target_cell_ids = self.executor.apply_action(self.state, config_action)
                self._refresh_runtime_views_after_config_apply()
                self.state.control.actuation_outcomes.append(
                    self._build_config_actuation_outcome(
                        action=config_action,
                        apply_slot=int(self.state.slot_id),
                        status="applied",
                        target_cell_ids=applied_target_cell_ids,
                    )
                )
        else:
            config_apply_slot = int(self.state.slot_id) + int(self._config_delay_slots)
            for config_action in list(control_decision.config_actions):
                self._update_e2_control_request_status(
                    getattr(config_action, "control_request_id", None),
                    status="queued",
                    reason="waiting_for_config_delay",
                )
                self._actuation_seq += 1
                self._pending_config_actuations.append(
                    (config_apply_slot, self._actuation_seq, config_action)
                )

        if int(self._command_delay_slots) <= 0:
            for command in list(control_decision.commands):
                rejected = self._rng.random() < self._control_reject_prob
                if rejected:
                    self.state.control.actuation_outcomes.append(
                        {
                            "kind": "command",
                            "issued_at_slot": int(getattr(getattr(command, "metadata", None), "issued_at_slot", -1)),
                            "apply_slot": int(self.state.slot_id),
                            "source": str(getattr(getattr(command, "metadata", None), "source", getattr(command, "source_name", "unknown"))),
                            "decision_id": str(getattr(getattr(command, "metadata", None), "decision_id", "")),
                            "control_request_id": getattr(command, "control_request_id", None),
                            "status": "rejected",
                            "reason": "probabilistic_reject",
                        }
                    )
                    continue
                commands_to_apply.append(command)
                self.state.control.actuation_outcomes.append(
                    {
                        "kind": "command",
                        "issued_at_slot": int(getattr(getattr(command, "metadata", None), "issued_at_slot", -1)),
                        "apply_slot": int(self.state.slot_id),
                        "source": str(getattr(getattr(command, "metadata", None), "source", getattr(command, "source_name", "unknown"))),
                        "decision_id": str(getattr(getattr(command, "metadata", None), "decision_id", "")),
                        "control_request_id": getattr(command, "control_request_id", None),
                        "status": "applied",
                    }
                )
        else:
            for command in list(control_decision.commands):
                self._update_e2_control_request_status(
                    getattr(command, "control_request_id", None),
                    status="queued",
                    reason="waiting_for_command_delay",
                )
                self._actuation_seq += 1
                cmd_apply_slot = int(self.state.slot_id) + int(self._command_delay_slots)
                self._pending_command_actuations.append((cmd_apply_slot, self._actuation_seq, command))

        due_commands = [item for item in self._pending_command_actuations if item[0] <= self.state.slot_id]
        self._pending_command_actuations = [item for item in self._pending_command_actuations if item[0] > self.state.slot_id]
        due_commands.sort(key=lambda x: (x[0], x[1]))

        for apply_slot, _, command in due_commands:
            provenance=getattr(command,'_library_provenance',None)
            if provenance and not self.controller_manager.instance_selection.valid(provenance):
                self.state.control.actuation_outcomes.append(dict(kind='command',status='rejected',reason='stale_xapp_selection',
                    slot_id=self.state.slot_id,control_request_id=command.control_request_id,provenance=provenance))
                self._update_e2_control_request_status(command.control_request_id,status='rejected',reason='stale_xapp_selection')
                continue
            rejected = self._rng.random() < self._control_reject_prob
            if rejected:
                self.state.control.actuation_outcomes.append(
                    {
                        "kind": "command",
                        "issued_at_slot": int(getattr(getattr(command, "metadata", None), "issued_at_slot", -1)),
                        "apply_slot": int(apply_slot),
                        "source": str(getattr(getattr(command, "metadata", None), "source", getattr(command, "source_name", "unknown"))),
                        "decision_id": str(getattr(getattr(command, "metadata", None), "decision_id", "")),
                        "control_request_id": getattr(command, "control_request_id", None),
                        "status": "rejected",
                        "reason": "probabilistic_reject",
                    }
                )
                continue
            commands_to_apply.append(command)
            self.state.control.actuation_outcomes.append(
                {
                    "kind": "command",
                    "issued_at_slot": int(getattr(getattr(command, "metadata", None), "issued_at_slot", -1)),
                    "apply_slot": int(apply_slot),
                    "source": str(getattr(getattr(command, "metadata", None), "source", getattr(command, "source_name", "unknown"))),
                    "decision_id": str(getattr(getattr(command, "metadata", None), "decision_id", "")),
                    "control_request_id": getattr(command, "control_request_id", None),
                    "status": "applied",
                }
            )

        self._refresh_control_lifecycle_outcomes()

        # Consume one-shot commands before scheduling.
        if getattr(self.state,'paper',None):
            commands_to_apply=self.state.paper.consume_commands(commands_to_apply)
            self.state.paper.before_scheduling()
        self.command_consumer.apply_commands(self.state, commands_to_apply)

        self._refresh_managed_cell_runtime_views(self.state)
        scoped_states = self._run_managed_cell_schedulers(self.state)
        phy_result, shadow_phy_result = self._run_managed_cell_phy(self.state, scoped_states)

        # Update controller-visible measurements after truth evolves.
        if self._rt_sys_mode:
            from phy.rt_sys.measurements import reference_measurements
            self.state.rt_reference_measurements = reference_measurements(self.phy_adapter.current_snapshot, self.state,
                self.phy_adapter.config.get('noise_figure_db', 7))
        self.measurement_updater.update(self.state)

        if getattr(self.state,'paper',None):self.state.paper.after_phy()
        post_phy_observation = build_observation(self.state)
        if getattr(self.state,'paper',None):self.state.paper.attach_observation(post_phy_observation)
        post_phy_observation.produced_at_slot = int(self.state.slot_id)
        post_phy_observation.delivered_at_slot = None
        self._observation_queue.append(post_phy_observation)

        max_obs_cache = max(4, int(self._obs_delay_slots) + 2)
        while len(self._observation_queue) > max_obs_cache:
            self._observation_queue.popleft()
        self.state_updater.record_snapshot(self.state, observation=post_phy_observation)
        self.state_updater.record_slot_metrics(
            state=self.state,
            observation=post_phy_observation,
            slot_records=self.slot_records,
            phy_result=phy_result,
            shadow_phy_result=shadow_phy_result,
            shadow_phy_label=self.shadow_phy_label,
        )

        debug_print(
            f"Finished slot {self.state.slot_id}. "
            f"avg_sinr_db={(f'{post_phy_observation.avg_sinr_db:.2f}' if post_phy_observation.avg_sinr_db is not None else 'N/A')}, "
            f"avg_bler={(f'{post_phy_observation.avg_bler:.4f}' if post_phy_observation.avg_bler is not None else 'N/A')}."
        )

        if self._rt_sys_mode and self.slot_records:
            self.phy_adapter.trace.update(
                mobility_events=list(self.mobility_manager.events),
                handover_counts={uid:c.handover_count for uid,c in self.state.connections.items()},
                connection_states={uid:c.connection_state for uid,c in self.state.connections.items()})
            self.slot_records[-1]["rt_sys"] = dict(self.phy_adapter.trace)
            self.slot_records[-1]["rt_sys"]["control_audit"] = self._rt_control_audit
        return post_phy_observation

    def run(self) -> None:
        """
        Runs the full simulation for the configured number of slots.
        INPUT
        No explicit input.
        OUTPUT
        None. Slot-level results are printed to stdout and written to files.
        EXAMPLE OUTPUT
        [slot=001] avg_sinr=10.00 dB, avg_bler=0.180
        """
        configured_total_slots = int(getattr(self.config, "total_slots", 0) or 0)
        run_forever = configured_total_slots <= 0
        executed_slots = 0
        interrupted = False

        try:
            while run_forever or executed_slots < configured_total_slots:
                observation = self.step()
                executed_slots += 1
                latest_record = self.slot_records[-1] if len(self.slot_records) > 0 else {}
                true_metrics = dict(latest_record.get("true_metrics", {}))
                observable_metrics = dict(latest_record.get("observable_metrics", {}))

                observable_sinr = observable_metrics.get("sinr_db", observation.avg_sinr_db)
                observable_bler = observable_metrics.get("bler", observation.avg_bler)
                observable_queue = observable_metrics.get("queue_bytes", observation.total_queue_bytes)
                observable_throughput = observable_metrics.get(
                    "throughput_mbps",
                    observation.total_throughput_mbps,
                )

                observable_sinr_text = (
                    f"{float(observable_sinr):.2f}"
                    if observable_sinr is not None
                    else "N/A"
                )
                observable_bler_text = (
                    f"{float(observable_bler):.3f}"
                    if observable_bler is not None
                    else "N/A"
                )
                if not DebugConfig.SLOT_METRICS:
                    continue
                if DebugConfig.CONCISE_SLOT_METRICS:
                    print(
                        f"[slot={observation.slot_id:03d}] "
                        f"throughput={float(observable_throughput):.2f} Mbps, "
                        f"queue={int(observable_queue)} bytes, "
                        f"sinr={observable_sinr_text} dB, "
                        f"bler={observable_bler_text}"
                    )
                else:
                    print(
                        f"[slot={observation.slot_id:03d}] "
                        f"observable(sinr={observable_sinr_text} dB, "
                        f"bler={observable_bler_text}, "
                        f"queue={int(observable_queue)} bytes, "
                        f"throughput={float(observable_throughput):.2f} Mbps) "
                        f"| true(queue={int(true_metrics.get('queue_bytes', 0))} bytes, "
                        f"new_tx={int(true_metrics.get('new_tx_bytes_this_slot', 0))} bytes, "
                        f"retx_tx={int(true_metrics.get('retx_tx_bytes_this_slot', 0))} bytes, "
                        f"ack={int(true_metrics.get('ack_count_this_slot', 0))}/"
                        f"{int(true_metrics.get('tx_tb_count_this_slot', 0))})"
                    )
        except KeyboardInterrupt:
            interrupted = True
            print("\n[INFO] Simulation interrupted by user. Writing partial results...")
        finally:
            if len(self.slot_records) > 0 or interrupted:
                final_summary = self._build_final_summary()
                controller_manager = getattr(self, "controller_manager", None)
                orchestration_epoch_history = (
                    controller_manager.get_orchestration_epoch_history()
                    if controller_manager is not None
                    and hasattr(controller_manager, "get_orchestration_epoch_history")
                    else []
                )
                self.result_writer.write_slot_metrics(self.slot_records)
                self.result_writer.write_final_summary(final_summary)
                self.result_writer.write_snapshots(list(getattr(self.state, "snapshots", [])))
                self.result_writer.write_orchestration_epoch_history(orchestration_epoch_history)

