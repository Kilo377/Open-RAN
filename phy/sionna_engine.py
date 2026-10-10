"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: sionna_engine.py
Description: This file defines the TensorFlow-based batched Sionna PHY
engine wrapper. The current version supports batched UE execution and a
smooth BLER approximation so that the simulator can later migrate to GPU
execution with minimal upper-layer refactoring.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Dict, List

from phy.mcs_profiles import (
    estimate_tbs_bytes,
    select_mcs_index_from_sinr,
    get_mcs_profile,
    normalize_mcs_table_id,
    normalize_tbs_selection_method,
)
from phy.cdl_su_mimo_model import CdlSuMimoModel
from phy.mimo_link_model import SingleUserMimoModel


class SionnaEngine:
    """
    Wraps the TensorFlow-based batched PHY execution backend.
    INPUT
    config: The Sionna PHY configuration dictionary.
    OUTPUT
    One reusable Sionna engine object.
    EXAMPLE OUTPUT
    engine = SionnaEngine(config)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the Sionna engine object.
        INPUT
        config: The Sionna PHY configuration dictionary.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.batch_size = 8
        """
        self.config = config
        self.backend = config.get("backend", "sionna_tensorflow")
        self.device = self._normalize_device(config.get("device", "cpu"))
        self.gpu_strict = bool(config.get("gpu_strict", True))
        self.gpu_memory_growth = bool(config.get("gpu_memory_growth", True))
        gpu_memory_limit_mb = config.get("gpu_memory_limit_mb", None)
        self.gpu_memory_limit_mb = (
            int(gpu_memory_limit_mb)
            if gpu_memory_limit_mb is not None and str(gpu_memory_limit_mb).strip() != ""
            else None
        )
        self._tf_device_configured = False
        self.active_device = "cpu"
        self.batch_size = int(config.get("batch_size", 1))
        self.num_bits_per_block = int(config.get("num_bits_per_block", 4096))
        self.max_sionna_codeblock_info_bits = int(
            config.get("max_sionna_codeblock_info_bits", 8448)
        )
        self.default_modulation = config.get("default_modulation", "qpsk")
        self.noise_floor_dbm = float(config.get("noise_floor_dbm", -94.0))
        self.code_block_size_bits = int(config.get("code_block_size_bits", 128))

        self.awgn_channel_config = config.get("awgn_channel", {})
        self.link_adaptation_config = config.get("link_adaptation", {})
        self.gain_config = config.get("gains", {})
        self.bler_model_config = config.get("bler_model", {})

        self.noise_power_mode = str(config.get("noise_power_mode", "thermal")).lower()
        self.bandwidth_hz = float(config.get("bandwidth_hz", 20e6))
        self.noise_figure_db = float(config.get("noise_figure_db", 7.0))
        self.interference_margin_db = float(config.get("interference_margin_db", 0.0))
        self.sinr_margin_db = float(config.get("sinr_margin_db", 0.0))
        self.num_iter_decoder = int(config.get("num_iter_decoder", 20))
        self.bler_smoothing_alpha = float(config.get("bler_smoothing_alpha", 0.0))
        self.bler_smoothing_beta = float(config.get("bler_smoothing_beta", 0.0))
        self.min_supported_sionna_coderate = float(
            config.get("min_supported_sionna_coderate", 0.34)
        )
        self.min_valid_sinr_db = float(config.get("min_valid_sinr_db", -40.0))
        self.max_valid_sinr_db = float(config.get("max_valid_sinr_db", 80.0))
        self.max_mcs_index = int(config.get("max_mcs_index", 27))

        self.num_mod_symbols_per_slot = int(config.get("num_mod_symbols_per_slot", 5120))
        self.subcarriers_per_prb = int(config.get("subcarriers_per_prb", 12))
        self.max_ofdm_symbols_per_slot = int(config.get("max_ofdm_symbols_per_slot", 14))
        self.enable_overlap_interference = bool(config.get("enable_overlap_interference", True))
        self.min_spatial_coupling = float(config.get("min_spatial_coupling", 0.05))
        self.same_precoder_coupling_multiplier = float(
            config.get("same_precoder_coupling_multiplier", 1.25)
        )
        self.adjacent_prb_leakage_factor = float(
            config.get("adjacent_prb_leakage_factor", 0.15)
        )
        self.same_group_interference_reduction = float(
            config.get("same_group_interference_reduction", 0.7)
        )
        self.mu_mimo_bonus_db_max = float(config.get("mu_mimo_bonus_db_max", 2.0))
        self.phy_mode = self._normalize_phy_mode(
            config.get("phy_mode", "system_level_approx")
        )
        self.cdl_su_mimo_config = dict(config.get("cdl_su_mimo", {}) or {})

        self.small_scale_model = str(config.get("small_scale_model", "gaussian")).lower()
        self.small_scale_std_db = float(config.get("small_scale_std_db", 2.0))

        self.tx_power_reference_dbm = float(
            self.gain_config.get("tx_power_reference_dbm", 30.0)
        )
        self.codebook_gain_step_db = float(
            self.gain_config.get("codebook_gain_step_db", 1.0)
        )
        self.beam_profile_gain_step_db = float(
            self.gain_config.get("beam_profile_gain_step_db", 0.8)
        )

        self.device_name = "/GPU:0" if self.device == "gpu" else "/CPU:0"
        self.single_user_mimo_model = SingleUserMimoModel()
        self.cdl_su_mimo_model = CdlSuMimoModel(self.cdl_su_mimo_config)

        # Cache for expensive Sionna objects keyed by (bits_per_symbol, num_info_bits, coderate, num_iter_decoder)
        self._coded_awgn_channel_cache: Dict[tuple[int, int, float, int], Any] = {}

    def _normalize_device(self, device: Any) -> str:
        normalized = str(device or "cpu").strip().lower()
        if normalized in {"gpu", "cuda", "cuda:0", "/gpu:0"}:
            return "gpu"
        if normalized in {"auto", "tensorflow_auto"}:
            return "auto"
        return "cpu"

    def _normalize_phy_mode(self, phy_mode: Any) -> str:
        normalized = str(phy_mode or "system_level_approx").strip().lower()
        if normalized in {"sionna_cdl_su_mimo", "cdl_su_mimo", "sionna_cdl", "sionna_cdl_su"}:
            return "sionna_cdl_su_mimo"
        return "system_level_approx"

    def _compute_noise_power_dbm(self, bandwidth_hz: float | None = None) -> float:
        """Computes noise power over the configured bandwidth.

        thermal: -174 dBm/Hz + 10log10(BW) + NF
        fixed: use configured noise_floor_dbm
        """
        mode = self.noise_power_mode
        if mode == "fixed":
            return float(self.noise_floor_dbm)

        bw_hz = max(1.0, float(self.bandwidth_hz if bandwidth_hz is None else bandwidth_hz))
        return -174.0 + 10.0 * math.log10(bw_hz) + float(self.noise_figure_db)

    def _select_codeblock_info_bits(self, tb_info_bits: int) -> int:
        """Selects a supported LDPC information-block size >= tb_info_bits.

        For Sionna's 5G-NR LDPC implementation, many information block sizes are
        supported as long as they are multiples of 32 and within the maximum
        configured limit.
        """
        tb_info_bits_int = int(max(1, tb_info_bits))

        max_bits = int(max(32, int(self.max_sionna_codeblock_info_bits)))
        max_bits = (max_bits // 32) * 32

        rounded = ((tb_info_bits_int + 31) // 32) * 32
        rounded = max(32, rounded)
        return int(min(max_bits, rounded))

    def _split_transport_block_info_bits(self, tb_info_bits: int) -> list[int]:
        """Splits one transport block into Sionna-supported LDPC code blocks."""
        remaining_bits = int(max(1, tb_info_bits))
        codeblock_cap_bits = self._select_codeblock_info_bits(
            int(self.max_sionna_codeblock_info_bits)
        )
        codeblock_sizes: list[int] = []

        while remaining_bits > 0:
            current_block_target_bits = min(remaining_bits, codeblock_cap_bits)
            current_codeblock_bits = self._select_codeblock_info_bits(current_block_target_bits)
            codeblock_sizes.append(int(current_codeblock_bits))
            remaining_bits -= min(remaining_bits, current_block_target_bits)

        return codeblock_sizes

    def _get_numpy(self):
        try:
            return importlib.import_module("numpy")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing dependency 'numpy'. Install it to use SionnaEngine."
            ) from exc

    def _get_tf(self):
        try:
            tf = importlib.import_module("tensorflow")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing dependency 'tensorflow'. Install it to use SionnaEngine."
            ) from exc
        self._configure_tensorflow_device(tf)
        return tf

    def _configure_tensorflow_device(self, tf: Any) -> None:
        if self._tf_device_configured:
            return

        requested_device = self.device
        physical_gpus = list(tf.config.list_physical_devices("GPU") or [])
        if requested_device == "auto":
            requested_device = "gpu" if len(physical_gpus) > 0 else "cpu"

        if requested_device == "cpu":
            self.active_device = "cpu"
            self.device_name = "/CPU:0"
            self._tf_device_configured = True
            return

        if len(physical_gpus) == 0:
            if self.gpu_strict:
                raise RuntimeError(
                    "SionnaEngine was configured with sionna_phy.device='gpu', "
                    "but TensorFlow does not list any physical GPU devices. "
                    "Install a GPU-enabled TensorFlow/Sionna runtime or switch "
                    "sionna_phy.device to 'cpu'."
                )
            self.active_device = "cpu"
            self.device_name = "/CPU:0"
            self._tf_device_configured = True
            return

        first_gpu = physical_gpus[0]
        try:
            if self.gpu_memory_limit_mb is not None:
                tf.config.set_logical_device_configuration(
                    first_gpu,
                    [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=int(self.gpu_memory_limit_mb)
                        )
                    ],
                )
            elif self.gpu_memory_growth:
                for gpu_device in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu_device, True)
        except RuntimeError:
            # TensorFlow may already be initialized by another component. In that
            # case the visible device remains usable, so we keep running on GPU.
            pass

        self.active_device = "gpu"
        self.device_name = "/GPU:0"
        self._tf_device_configured = True

    def runtime_device_summary(self) -> dict[str, object]:
        return {
            "requested_device": self.device,
            "active_device": self.active_device,
            "device_name": self.device_name,
            "gpu_strict": self.gpu_strict,
            "gpu_memory_growth": self.gpu_memory_growth,
            "gpu_memory_limit_mb": self.gpu_memory_limit_mb,
            "tf_device_configured": self._tf_device_configured,
        }

    def _get_coded_awgn_channel_class(self):
        """Returns Sionna's 5G-NR coded AWGN channel class."""
        try:
            module = importlib.import_module("sionna.sys.phy_abstraction")
            return getattr(module, "CodedAWGNChannelNR")
        except ModuleNotFoundError:
            try:
                importlib.import_module("sionna")
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Missing dependency 'sionna'. Install NVIDIA Sionna to use the real Sionna PHY backend."
                ) from exc
        except Exception:
            pass

        try:
            module = importlib.import_module("sionna.phy.nr.utils")
            return getattr(module, "CodedAWGNChannelNR")
        except Exception as exc:
            # Fallback for older/newer package layouts
            raise ImportError(
                "Failed to import CodedAWGNChannelNR from Sionna. "
                "Your installed 'sionna' package may be incompatible."
            ) from exc

    def _set_sionna_seed(self, seed: int) -> None:
        """Sets Sionna/TF seeds for repeatable link simulation."""
        tf = self._get_tf()
        try:
            sionna_module = importlib.import_module("sionna")
            phy_module = getattr(sionna_module, "phy", None)
            config = getattr(phy_module, "config", None)
            if config is not None and hasattr(config, "seed"):
                setattr(config, "seed", int(seed))
                return
        except Exception:
            # Fall back to TensorFlow global seed.
            pass
        tf.random.set_seed(int(seed))

    def _get_coded_awgn_channel(
        self,
        *,
        bits_per_symbol: int,
        num_info_bits: int,
        target_coderate: float,
    ) -> Any:
        key = (int(bits_per_symbol), int(num_info_bits), float(target_coderate), int(self.num_iter_decoder))
        if key in self._coded_awgn_channel_cache:
            return self._coded_awgn_channel_cache[key]

        channel_cls = self._get_coded_awgn_channel_class()
        channel = channel_cls(
            num_bits_per_symbol=int(bits_per_symbol),
            num_info_bits=int(num_info_bits),
            target_coderate=float(target_coderate),
            num_iter_decoder=int(self.num_iter_decoder),
        )
        self._coded_awgn_channel_cache[key] = channel
        return channel

    def _normalize_target_coderate_for_sionna(self, target_coderate: float) -> float:
        """Clamps coderate into the range Sionna's NR LDPC abstraction supports."""
        normalized = max(1e-3, min(0.99, float(target_coderate)))
        minimum = max(1e-3, min(0.99, float(self.min_supported_sionna_coderate)))
        return max(normalized, minimum)

    def _clip_sinr_db(self, sinr_db: float) -> float:
        minimum = min(float(self.min_valid_sinr_db), float(self.max_valid_sinr_db))
        maximum = max(float(self.min_valid_sinr_db), float(self.max_valid_sinr_db))
        return max(minimum, min(maximum, float(sinr_db)))

    def build_engine_input(
        self,
        phy_input,
        ue_input,
    ) -> Dict[str, Any]:
        """
        Builds one standardized low-level engine input dictionary from the
        simulator PHY input and one UE PHY input object.
        INPUT
        phy_input: The slot-level PHY input object.
        ue_input: The per-UE PHY input object.
        OUTPUT
        One engine input dictionary.
        EXAMPLE OUTPUT
        {"slot_id": 1, "ue_id": "ue_0", "tx_power_dbm": 30.0}
        """
        link_pathloss_db = getattr(ue_input, "link_pathloss_db", phy_input.runtime_context.pathloss_db)
        if link_pathloss_db is None:
            link_pathloss_db = phy_input.runtime_context.pathloss_db
        link_shadowing_db = getattr(ue_input, "link_shadowing_db", phy_input.runtime_context.shadowing_db)
        if link_shadowing_db is None:
            link_shadowing_db = phy_input.runtime_context.shadowing_db
        link_doppler_hz = getattr(
            ue_input,
            "link_doppler_hz",
            getattr(phy_input.runtime_context, "doppler_hz", 0.0),
        )
        if link_doppler_hz is None:
            link_doppler_hz = getattr(phy_input.runtime_context, "doppler_hz", 0.0)
        link_delay_spread_ns = getattr(ue_input, "link_delay_spread_ns", 300.0)
        if link_delay_spread_ns is None:
            link_delay_spread_ns = 300.0
        link_fast_fading_seed = int(getattr(ue_input, "link_fast_fading_seed", 0) or 0)
        link_los_state = getattr(ue_input, "link_los_state", "los")
        if link_los_state is None:
            link_los_state = "los"
        link_o2i_penetration_loss_db = getattr(ue_input, "link_o2i_penetration_loss_db", 0.0)
        if link_o2i_penetration_loss_db is None:
            link_o2i_penetration_loss_db = 0.0
        link_k_factor_db = getattr(ue_input, "link_k_factor_db", 0.0)
        if link_k_factor_db is None:
            link_k_factor_db = 0.0
        link_angle_spread_deg = getattr(ue_input, "link_angle_spread_deg", 35.0)
        if link_angle_spread_deg is None:
            link_angle_spread_deg = 35.0
        link_spatial_correlation_level = getattr(ue_input, "link_spatial_correlation_level", 0.5)
        if link_spatial_correlation_level is None:
            link_spatial_correlation_level = 0.5
        link_spatial_corr_seed = int(getattr(ue_input, "link_spatial_corr_seed", 0) or 0)

        return {
            "slot_id": phy_input.slot_id,
            "sim_time_ms": phy_input.sim_time_ms,
            "slot_duration_ms": float(
                getattr(getattr(phy_input, "runtime_context", None), "slot_duration_ms", 1.0)
            ),
            "cell_id": phy_input.cell_id,
            "ue_id": ue_input.ue_id,
            "tx_power_dbm": phy_input.tx_power_dbm,
            "codebook_id": phy_input.current_codebook_id,
            "beam_profile_id": phy_input.current_beam_profile_id,
            "carrier_frequency_hz": float(
                getattr(phy_input.runtime_context, "carrier_frequency_hz", 3.5e9)
            ),
            "bandwidth_hz": float(getattr(phy_input.runtime_context, "bandwidth_hz", self.bandwidth_hz)),
            "subcarrier_spacing_hz": float(
                getattr(phy_input.runtime_context, "subcarrier_spacing_hz", 30e3)
            ),
            "cyclic_prefix_type": str(
                getattr(phy_input.runtime_context, "cyclic_prefix_type", "normal")
            ),
            "num_ofdm_symbols_per_slot": int(
                getattr(phy_input.runtime_context, "num_ofdm_symbols_per_slot", self.max_ofdm_symbols_per_slot)
            ),
            "n_prbs_total": int(getattr(phy_input.runtime_context, "n_prbs_total", 0)),
            "serving_beam_id": ue_input.serving_beam_id,
            "radio_quality_db": ue_input.radio_quality_db,
            "queue_bytes": ue_input.queue_bytes,
            "requested_bytes": int(getattr(ue_input, "requested_bytes", ue_input.queue_bytes)),
            "grant_bytes_budget": int(getattr(ue_input, "grant_bytes_budget", 0)),
            "is_scheduled": ue_input.is_scheduled,
            "allocated_prbs": int(getattr(ue_input, "allocated_prbs", 0)),
            "allocated_symbols": int(getattr(ue_input, "allocated_symbols", 0)),
            "start_prb": int(getattr(ue_input, "start_prb", 0)),
            "start_symbol": int(getattr(ue_input, "start_symbol", 0)),
            "resource_block_indices": list(getattr(ue_input, "resource_block_indices", []) or []),
            "allocated_subcarrier_indices": list(
                getattr(ue_input, "allocated_subcarrier_indices", []) or []
            ),
            "allocated_symbol_indices": list(getattr(ue_input, "allocated_symbol_indices", []) or []),
            "resource_grid_slice": getattr(ue_input, "resource_grid_slice", None),
            "is_retx": bool(getattr(ue_input, "is_retx", False)),
            "priority_score": float(getattr(ue_input, "priority_score", 0.0)),
            "target_mcs": getattr(ue_input, "target_mcs", None),
            "mcs_cap": getattr(ue_input, "mcs_cap", None),
            "mcs_table_id": str(getattr(ue_input, "mcs_table_id", "qam64")),
            "tbs_selection_method": str(getattr(ue_input, "tbs_selection_method", "resource_elements")),
            "dmrs_pattern_id": getattr(ue_input, "dmrs_pattern_id", None),
            "harq_process_id": getattr(ue_input, "harq_process_id", None),
            "harq_redundancy_version": getattr(ue_input, "harq_redundancy_version", None),
            "harq_ndi": getattr(ue_input, "harq_ndi", None),
            "harq_retransmission_round": int(getattr(ue_input, "harq_retransmission_round", 0) or 0),
            "cqi_validity_age_slots": int(getattr(ue_input, "cqi_validity_age_slots", 0) or 0),
            "ri_validity_age_slots": int(getattr(ue_input, "ri_validity_age_slots", 0) or 0),
            "pmi_validity_age_slots": int(getattr(ue_input, "pmi_validity_age_slots", 0) or 0),
            "channel_quality_confidence": (
                float(getattr(ue_input, "channel_quality_confidence", 1.0))
                if getattr(ue_input, "channel_quality_confidence", None) is not None
                else None
            ),
            "link_adaptation_reason": getattr(ue_input, "link_adaptation_reason", None),
            "spatial_group_id": getattr(ue_input, "spatial_group_id", None),
            "spatial_group_size": int(getattr(ue_input, "spatial_group_size", 1) or 1),
            "spatial_compatibility_score": float(
                getattr(ue_input, "spatial_compatibility_score", 0.0) or 0.0
            ),
            "slice_id": ue_input.slice_id,
            "traffic_type": ue_input.traffic_type,
            "latency_budget_ms": ue_input.latency_budget_ms,
            "priority": ue_input.priority,
            "channel_profile": getattr(
                ue_input,
                "link_channel_model_type",
                phy_input.runtime_context.channel_profile,
            ),
            "pathloss_db": float(link_pathloss_db),
            "shadowing_db": float(link_shadowing_db),
            "los_state": str(link_los_state),
            "o2i_penetration_loss_db": float(link_o2i_penetration_loss_db),
            "doppler_hz": float(link_doppler_hz),
            "delay_spread_ns": float(link_delay_spread_ns),
            "k_factor_db": float(link_k_factor_db),
            "angle_spread_deg": float(link_angle_spread_deg),
            "spatial_correlation_level": float(link_spatial_correlation_level),
            "fast_fading_seed": link_fast_fading_seed,
            "spatial_corr_seed": link_spatial_corr_seed,
            "reported_cqi": getattr(ue_input, "reported_cqi", None),
            "reported_ri": getattr(ue_input, "reported_ri", None),
            "reported_pmi": getattr(ue_input, "reported_pmi", None),
            "num_tx_ant": int(getattr(ue_input, "num_tx_ant", 1) or 1),
            "num_rx_ant": int(getattr(ue_input, "num_rx_ant", 1) or 1),
            "rank_indicator": int(getattr(ue_input, "rank_indicator", 1) or 1),
            "layer_count": int(getattr(ue_input, "layer_count", 1) or 1),
            "precoder_id": int(getattr(ue_input, "precoder_id", phy_input.current_codebook_id) or phy_input.current_codebook_id),
            "combiner_id": int(getattr(ue_input, "combiner_id", 0) or 0),
            "precoder_matrix": list(getattr(ue_input, "precoder_matrix", []) or []),
            "combiner_vector": list(getattr(ue_input, "combiner_vector", []) or []),
            "precoding_matrix_label": getattr(ue_input, "precoding_matrix_label", None),
            "beam_direction_deg": float(getattr(ue_input, "beam_direction_deg", 0.0) or 0.0),
            "beamforming_gain_db": float(getattr(ue_input, "beamforming_gain_db", 0.0) or 0.0),
            "inter_cell_interference_dbm": float(
                getattr(ue_input, "inter_cell_interference_dbm", -120.0) or -120.0
            ),
            "inter_cell_interference_penalty_db": float(
                getattr(ue_input, "inter_cell_interference_penalty_db", 0.0) or 0.0
            ),
            "inter_cell_interference_overlap_fraction": float(
                getattr(ue_input, "inter_cell_interference_overlap_fraction", 0.0) or 0.0
            ),
            "inter_cell_frequency_coupling_fraction": float(
                getattr(ue_input, "inter_cell_frequency_coupling_fraction", 0.0) or 0.0
            ),
            "inter_cell_interference_source_count": int(
                getattr(ue_input, "inter_cell_interference_source_count", 0) or 0
            ),
            "slot_total_prbs": int(
                getattr(getattr(phy_input, "scheduling_decision", None), "total_prbs", 0)
                or getattr(phy_input.runtime_context, "n_prbs_total", 0)
                or 0
            ),
            "slot_total_symbols": int(
                getattr(getattr(phy_input, "scheduling_decision", None), "total_symbols", 0)
                or getattr(phy_input.runtime_context, "num_ofdm_symbols_per_slot", self.max_ofdm_symbols_per_slot)
                or 0
            ),
        }

    def _compute_repeatable_small_scale_db(
        self,
        slot_id: int,
        ue_id: str,
        *,
        fast_fading_seed: int = 0,
        doppler_hz: float = 0.0,
        k_factor_db: float = 0.0,
        angle_spread_deg: float = 35.0,
        spatial_correlation_level: float = 0.5,
    ) -> float:
        """
        Computes one repeatable small-scale channel fluctuation term.
        INPUT
        slot_id: The current slot index.
        ue_id: The UE identifier.
        OUTPUT
        One small-scale variation value in dB.
        EXAMPLE OUTPUT
        1.2
        """
        if self.small_scale_model == "legacy":
            ue_hash = sum(ord(character) for character in ue_id) + int(fast_fading_seed)
            phase_index = (slot_id + ue_hash) % 9
            doppler_scale = 1.0 + min(1.0, max(0.0, float(doppler_hz) / 120.0))
            return 0.6 * (float(phase_index) - 4.0) * doppler_scale

        # Default: Gaussian small-scale fading term (in dB), repeatable per (slot, UE).
        std_db = max(0.0, float(self.small_scale_std_db))
        if std_db <= 0.0:
            return 0.0

        ue_hash = sum(ord(character) for character in ue_id)
        seed = 20000 + 37 * int(slot_id) + 19 * int(ue_hash) + int(fast_fading_seed)
        np = self._get_numpy()
        random_state = np.random.RandomState(seed)
        doppler_scale = 1.0 + min(1.0, max(0.0, float(doppler_hz) / 120.0))
        k_factor_damping = max(0.3, 1.0 - min(0.7, max(0.0, float(k_factor_db)) / 20.0))
        angle_spread_scale = 0.75 + min(0.35, max(0.0, float(angle_spread_deg)) / 180.0)
        spatial_corr_scale = max(
            0.55,
            1.0 - 0.35 * max(0.0, min(1.0, float(spatial_correlation_level))),
        )
        effective_std_db = std_db * doppler_scale * k_factor_damping * angle_spread_scale * spatial_corr_scale
        return float(random_state.normal(loc=0.0, scale=effective_std_db))

    def _compute_link_sinr_db(
        self,
        engine_input: Dict[str, Any],
    ) -> float:
        """
        Computes one effective link SINR in dB.
        INPUT
        engine_input: The low-level engine input dictionary.
        OUTPUT
        One effective SINR value in dB.
        EXAMPLE OUTPUT
        11.5
        """
        tx_power_dbm = float(engine_input["tx_power_dbm"])
        ue_rx_gain_db = float(engine_input["radio_quality_db"])
        carrier_frequency_hz = max(1.0, float(engine_input.get("carrier_frequency_hz", 3.5e9)))
        precoder_id = int(engine_input.get("precoder_id", engine_input["codebook_id"]))
        layer_count = max(1, int(engine_input.get("layer_count", 1)))
        rank_indicator = max(1, int(engine_input.get("rank_indicator", layer_count)))
        num_tx_ant = max(1, int(engine_input.get("num_tx_ant", 1)))
        num_rx_ant = max(1, int(engine_input.get("num_rx_ant", 1)))
        codebook_gain_db = self.codebook_gain_step_db * float(precoder_id)
        beam_gain_db = self.beam_profile_gain_step_db * float(engine_input["beam_profile_id"])
        spatial_array_gain_db = 10.0 * math.log10(float(max(1, min(num_tx_ant, num_rx_ant))))
        layer_penalty_db = max(
            0.0,
            10.0 * math.log10(float(max(1, layer_count))) - 10.0 * math.log10(float(rank_indicator)),
        )
        explicit_beamforming_gain_db = float(engine_input.get("beamforming_gain_db", 0.0))
        small_scale_db = self._compute_repeatable_small_scale_db(
            slot_id=int(engine_input["slot_id"]),
            ue_id=str(engine_input["ue_id"]),
            fast_fading_seed=int(engine_input.get("fast_fading_seed", 0)),
            doppler_hz=float(engine_input.get("doppler_hz", 0.0)),
            k_factor_db=float(engine_input.get("k_factor_db", 0.0)),
            angle_spread_deg=float(engine_input.get("angle_spread_deg", 35.0)),
            spatial_correlation_level=float(engine_input.get("spatial_correlation_level", 0.5)),
        )

        pathloss_db = float(engine_input["pathloss_db"])
        shadowing_db = float(engine_input["shadowing_db"])
        o2i_penetration_loss_db = float(engine_input.get("o2i_penetration_loss_db", 0.0))
        frequency_pathloss_adjustment_db = 20.0 * math.log10(carrier_frequency_hz / 3.5e9)

        rx_power_dbm = (
            tx_power_dbm
            + ue_rx_gain_db
            + codebook_gain_db
            + beam_gain_db
            + spatial_array_gain_db
            + explicit_beamforming_gain_db
            - layer_penalty_db
            + small_scale_db
            - pathloss_db
            - shadowing_db
            - o2i_penetration_loss_db
            - frequency_pathloss_adjustment_db
        )

        noise_power_dbm = self._compute_noise_power_dbm(
            bandwidth_hz=float(engine_input.get("bandwidth_hz", self.bandwidth_hz))
        )
        return rx_power_dbm - noise_power_dbm - float(self.interference_margin_db)

    def _evaluate_single_user_mimo(
        self,
        engine_input: Dict[str, Any],
        *,
        pre_eq_sinr_db: float,
    ) -> Dict[str, Any]:
        evaluation = self.single_user_mimo_model.evaluate(
            pre_eq_sinr_db=float(pre_eq_sinr_db),
            num_tx_ant=int(engine_input.get("num_tx_ant", 1)),
            num_rx_ant=int(engine_input.get("num_rx_ant", 1)),
            layer_count=int(engine_input.get("layer_count", 1)),
            rank_indicator=int(engine_input.get("rank_indicator", 1)),
            precoder_id=int(engine_input.get("precoder_id", engine_input.get("codebook_id", 0))),
            combiner_id=int(engine_input.get("combiner_id", 0)),
            reported_ri=engine_input.get("reported_ri", None),
            reported_pmi=engine_input.get("reported_pmi", None),
            delay_spread_ns=float(engine_input.get("delay_spread_ns", 300.0)),
            doppler_hz=float(engine_input.get("doppler_hz", 0.0)),
            channel_model_type=str(engine_input.get("channel_profile", "tdl_c")),
            k_factor_db=float(engine_input.get("k_factor_db", 0.0)),
            angle_spread_deg=float(engine_input.get("angle_spread_deg", 35.0)),
            spatial_correlation_level=float(engine_input.get("spatial_correlation_level", 0.5)),
        )
        return {
            "pre_eq_sinr_db": float(evaluation.pre_eq_sinr_db),
            "post_eq_sinr_db": float(evaluation.post_eq_sinr_db),
            "channel_rank_limit": int(evaluation.channel_rank_limit),
            "precoder_match_score": float(evaluation.precoder_match_score),
            "spatial_mode": str(evaluation.spatial_mode),
        }

    def _build_mcs_config_from_index(
        self,
        mcs_index: int,
        *,
        mcs_table_id: str | None = None,
        num_layers: int = 1,
    ) -> Dict[str, Any]:
        mcs_profile = get_mcs_profile(
            mcs_index,
            normalize_mcs_table_id(mcs_table_id),
            num_layers=max(1, int(num_layers)),
        )
        return {
            "modulation": str(mcs_profile.modulation_name),
            "bits_per_symbol": int(mcs_profile.bits_per_symbol),
            "mcs_index": int(mcs_profile.mcs_index),
            "target_coderate": float(mcs_profile.code_rate),
            "mcs_table_id": str(mcs_profile.mcs_table_id),
        }

    def _select_modulation_and_mcs(
        self,
        effective_sinr_db: float,
        *,
        target_mcs: int | None = None,
        mcs_cap: int | None = None,
        mcs_table_id: str | None = None,
        num_layers: int = 1,
    ) -> Dict[str, Any]:
        """
        Selects a minimal modulation and MCS configuration from effective SINR.
        INPUT
        effective_sinr_db: The effective SINR in dB.
        OUTPUT
        One modulation configuration dictionary.
        EXAMPLE OUTPUT
        {"modulation": "qpsk", "bits_per_symbol": 2, "mcs_index": 5}
        """
        normalized_table_id = normalize_mcs_table_id(mcs_table_id)
        base_mcs_index = select_mcs_index_from_sinr(
            float(effective_sinr_db),
            mcs_table_id=normalized_table_id,
            num_layers=max(1, int(num_layers)),
            max_mcs_index=int(self.max_mcs_index),
            bler_target=0.1,
            selection_margin_db=float(
                self.link_adaptation_config.get("selection_margin_db", 1.5)
            ),
        )

        if target_mcs is not None:
            base_mcs_index = int(target_mcs)
        if mcs_cap is not None:
            base_mcs_index = min(int(base_mcs_index), int(mcs_cap))
        base_mcs_index = max(0, min(int(self.max_mcs_index), int(base_mcs_index)))

        return self._build_mcs_config_from_index(
            base_mcs_index,
            mcs_table_id=normalized_table_id,
            num_layers=num_layers,
        )

    def _compute_num_mod_symbols(
        self,
        engine_input: Dict[str, Any],
    ) -> int:
        allocated_subcarrier_indices = list(engine_input.get("allocated_subcarrier_indices", []) or [])
        allocated_symbol_indices = list(engine_input.get("allocated_symbol_indices", []) or [])
        if allocated_subcarrier_indices and allocated_symbol_indices:
            return max(1, len(allocated_subcarrier_indices) * len(allocated_symbol_indices))

        resource_block_indices = list(engine_input.get("resource_block_indices", []) or [])
        if resource_block_indices and allocated_symbol_indices:
            return max(1, len(resource_block_indices) * self.subcarriers_per_prb * len(allocated_symbol_indices))

        allocated_prbs = int(max(0, int(engine_input.get("allocated_prbs", 0))))
        allocated_symbols = int(max(0, int(engine_input.get("allocated_symbols", 0))))
        if allocated_prbs > 0 and allocated_symbols > 0:
            return max(1, allocated_prbs * allocated_symbols * self.subcarriers_per_prb)
        n_prbs_total = int(engine_input.get("n_prbs_total", 0) or 0)
        n_symbols_total = int(engine_input.get("num_ofdm_symbols_per_slot", 0) or 0)
        if n_prbs_total > 0 and n_symbols_total > 0:
            return max(1, n_prbs_total * n_symbols_total * self.subcarriers_per_prb)
        return int(max(1, int(self.num_mod_symbols_per_slot)))

    def _build_resource_grid_view(self, engine_input: Dict[str, Any]) -> Dict[str, Any]:
        resource_grid_slice = engine_input.get("resource_grid_slice", None)
        if resource_grid_slice is not None:
            resource_block_indices = list(getattr(resource_grid_slice, "resource_block_indices", []) or [])
            allocated_subcarrier_indices = list(
                getattr(resource_grid_slice, "allocated_subcarrier_indices", []) or []
            )
            allocated_symbol_indices = list(
                getattr(resource_grid_slice, "allocated_symbol_indices", []) or []
            )
            resource_elements = int(getattr(resource_grid_slice, "resource_elements", 0))
            start_prb = int(getattr(resource_grid_slice, "start_prb", 0))
            start_symbol = int(getattr(resource_grid_slice, "start_symbol", 0))
            prb_count = int(getattr(resource_grid_slice, "allocated_prbs", len(resource_block_indices)))
            symbol_count = int(getattr(resource_grid_slice, "allocated_symbols", len(allocated_symbol_indices)))
        else:
            resource_block_indices = list(engine_input.get("resource_block_indices", []) or [])
            allocated_subcarrier_indices = list(engine_input.get("allocated_subcarrier_indices", []) or [])
            allocated_symbol_indices = list(engine_input.get("allocated_symbol_indices", []) or [])
            resource_elements = 0
            start_prb = int(engine_input.get("start_prb", 0) or 0)
            start_symbol = int(engine_input.get("start_symbol", 0) or 0)
            prb_count = int(engine_input.get("allocated_prbs", 0) or 0)
            symbol_count = int(engine_input.get("allocated_symbols", 0) or 0)

        slot_total_prbs_raw = int(engine_input.get("slot_total_prbs", 0) or 0)
        slot_total_symbols_raw = int(engine_input.get("slot_total_symbols", 0) or 0)
        slot_total_prbs = (
            slot_total_prbs_raw
            if slot_total_prbs_raw > 0
            else max(prb_count, 1)
        )
        slot_total_symbols = (
            slot_total_symbols_raw
            if slot_total_symbols_raw > 0
            else max(symbol_count, int(self.max_ofdm_symbols_per_slot))
        )
        start_prb = max(0, start_prb)
        start_symbol = max(0, start_symbol)
        prb_count = max(0, prb_count)
        symbol_count = max(0, symbol_count)
        if not resource_block_indices:
            end_prb = min(slot_total_prbs, start_prb + prb_count)
            resource_block_indices = list(range(start_prb, max(start_prb, end_prb)))
        else:
            end_prb = min(slot_total_prbs, max(resource_block_indices) + 1)

        if not allocated_symbol_indices:
            end_symbol = min(slot_total_symbols, start_symbol + symbol_count)
            allocated_symbol_indices = list(range(start_symbol, max(start_symbol, end_symbol)))
        else:
            end_symbol = min(slot_total_symbols, max(allocated_symbol_indices) + 1)

        if not allocated_subcarrier_indices and resource_block_indices:
            for prb_index in resource_block_indices:
                base_subcarrier = int(prb_index) * self.subcarriers_per_prb
                allocated_subcarrier_indices.extend(
                    range(base_subcarrier, base_subcarrier + self.subcarriers_per_prb)
                )

        end_prb = min(slot_total_prbs, start_prb + prb_count)
        end_symbol = min(slot_total_symbols, start_symbol + symbol_count)
        effective_prb_count = max(0, end_prb - start_prb)
        effective_symbol_count = max(0, end_symbol - start_symbol)
        if resource_elements <= 0:
            if allocated_subcarrier_indices and allocated_symbol_indices:
                resource_elements = len(allocated_subcarrier_indices) * len(allocated_symbol_indices)
            else:
                resource_elements = effective_prb_count * effective_symbol_count * self.subcarriers_per_prb
        occupancy_fraction = (
            float(resource_elements)
            / float(max(1, slot_total_prbs * slot_total_symbols * self.subcarriers_per_prb))
        )
        return {
            "start_prb": start_prb,
            "end_prb": end_prb,
            "start_symbol": start_symbol,
            "end_symbol": end_symbol,
            "prb_count": effective_prb_count,
            "symbol_count": effective_symbol_count,
            "resource_block_indices": resource_block_indices,
            "allocated_subcarrier_indices": allocated_subcarrier_indices,
            "allocated_symbol_indices": allocated_symbol_indices,
            "resource_elements": int(resource_elements),
            "occupancy_fraction": float(occupancy_fraction),
        }

    def _compute_resource_overlap_fraction(
        self,
        target_view: Dict[str, Any],
        other_view: Dict[str, Any],
    ) -> float:
        target_subcarriers = set(int(item) for item in target_view.get("allocated_subcarrier_indices", []))
        other_subcarriers = set(int(item) for item in other_view.get("allocated_subcarrier_indices", []))
        target_symbols = set(int(item) for item in target_view.get("allocated_symbol_indices", []))
        other_symbols = set(int(item) for item in other_view.get("allocated_symbol_indices", []))

        if target_subcarriers and other_subcarriers:
            overlap_subcarriers = len(target_subcarriers & other_subcarriers)
        else:
            overlap_prbs = max(
                0,
                min(int(target_view["end_prb"]), int(other_view["end_prb"]))
                - max(int(target_view["start_prb"]), int(other_view["start_prb"])),
            )
            overlap_subcarriers = overlap_prbs * self.subcarriers_per_prb

        if target_symbols and other_symbols:
            overlap_symbols = len(target_symbols & other_symbols)
        else:
            overlap_symbols = max(
                0,
                min(int(target_view["end_symbol"]), int(other_view["end_symbol"]))
                - max(int(target_view["start_symbol"]), int(other_view["start_symbol"])),
            )

        overlap_res = overlap_subcarriers * overlap_symbols
        target_res = max(1, int(target_view["resource_elements"]))
        return float(overlap_res) / float(target_res)

    def _compute_adjacent_prb_leakage_fraction(
        self,
        target_view: Dict[str, Any],
        other_view: Dict[str, Any],
    ) -> float:
        target_symbol_count = max(1, int(target_view.get("symbol_count", 0)))
        other_symbol_count = max(1, int(other_view.get("symbol_count", 0)))
        overlap_symbols = max(
            0,
            min(int(target_view["end_symbol"]), int(other_view["end_symbol"]))
            - max(int(target_view["start_symbol"]), int(other_view["start_symbol"])),
        )
        if overlap_symbols <= 0:
            return 0.0

        overlap_prbs = max(
            0,
            min(int(target_view["end_prb"]), int(other_view["end_prb"]))
            - max(int(target_view["start_prb"]), int(other_view["start_prb"])),
        )
        if overlap_prbs > 0:
            return 0.0

        left_end_prb = min(int(target_view["end_prb"]), int(other_view["end_prb"]))
        right_start_prb = max(int(target_view["start_prb"]), int(other_view["start_prb"]))
        if right_start_prb - left_end_prb != 0:
            return 0.0

        symbol_overlap_fraction = float(overlap_symbols) / float(target_symbol_count)
        occupancy_scale = min(
            1.0,
            float(other_view.get("occupancy_fraction", 0.0))
            / max(1e-9, float(target_view.get("occupancy_fraction", 0.0))),
        )
        return max(0.0, float(self.adjacent_prb_leakage_factor) * symbol_overlap_fraction * occupancy_scale)

    def _compute_beam_separation_deg(self, target_input: Dict[str, Any], other_input: Dict[str, Any]) -> float:
        target_direction = float(target_input.get("beam_direction_deg", 0.0))
        other_direction = float(other_input.get("beam_direction_deg", 0.0))
        raw_delta = abs(target_direction - other_direction) % 360.0
        return min(raw_delta, 360.0 - raw_delta)

    def _compute_mu_mimo_group_context(
        self,
        target_input: Dict[str, Any],
        all_engine_inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        spatial_group_id = target_input.get("spatial_group_id", None)
        if spatial_group_id is None:
            return {
                "group_id": None,
                "group_size": 1,
                "compatibility_score": 0.0,
                "multiplexing_gain_db": 0.0,
            }

        group_members = [
            item
            for item in all_engine_inputs
            if item is not target_input
            and bool(item.get("is_scheduled", False))
            and item.get("spatial_group_id", None) == spatial_group_id
        ]
        if len(group_members) == 0:
            return {
                "group_id": spatial_group_id,
                "group_size": 1,
                "compatibility_score": float(target_input.get("spatial_compatibility_score", 0.0)),
                "multiplexing_gain_db": 0.0,
            }

        compatibility_values = [float(target_input.get("spatial_compatibility_score", 0.0))]
        for group_member in group_members:
            beam_separation_deg = self._compute_beam_separation_deg(target_input, group_member)
            normalized_separation = min(1.0, beam_separation_deg / 180.0)
            diversity_factor = (
                1.0
                if int(group_member.get("precoder_id", -1)) != int(target_input.get("precoder_id", -1))
                else 0.85
            )
            compatibility_values.append(
                max(
                    0.0,
                    min(
                        1.0,
                        normalized_separation
                        * diversity_factor
                        * float(group_member.get("spatial_compatibility_score", 1.0)),
                    ),
                )
            )

        compatibility_score = max(0.0, min(1.0, sum(compatibility_values) / float(len(compatibility_values))))
        group_size = 1 + len(group_members)
        multiplexing_gain_db = min(
            float(self.mu_mimo_bonus_db_max),
            float(self.mu_mimo_bonus_db_max) * compatibility_score * float(group_size - 1) / float(group_size),
        )
        return {
            "group_id": spatial_group_id,
            "group_size": group_size,
            "compatibility_score": compatibility_score,
            "multiplexing_gain_db": multiplexing_gain_db,
        }

    def _compute_overlap_interference_penalty_db(
        self,
        target_input: Dict[str, Any],
        all_engine_inputs: List[Dict[str, Any]],
    ) -> tuple[float, float, float, float, int]:
        if not self.enable_overlap_interference or not bool(target_input.get("is_scheduled", False)):
            return 0.0, 0.0, 0.0, 0.0, 1

        target_view = self._build_resource_grid_view(target_input)
        target_num_rx_ant = max(1, int(target_input.get("num_rx_ant", 1)))
        total_interference_ratio = 0.0
        max_overlap_fraction = 0.0
        mu_context = self._compute_mu_mimo_group_context(target_input, all_engine_inputs)

        for other_input in all_engine_inputs:
            if other_input is target_input or not bool(other_input.get("is_scheduled", False)):
                continue
            other_view = self._build_resource_grid_view(other_input)
            overlap_fraction = self._compute_resource_overlap_fraction(target_view, other_view)
            leakage_fraction = 0.0
            if overlap_fraction <= 0.0:
                leakage_fraction = self._compute_adjacent_prb_leakage_fraction(target_view, other_view)
            effective_coupling_fraction = max(overlap_fraction, leakage_fraction)
            if effective_coupling_fraction <= 0.0:
                continue

            max_overlap_fraction = max(max_overlap_fraction, overlap_fraction)
            beam_separation_deg = self._compute_beam_separation_deg(target_input, other_input)
            normalized_separation = min(1.0, beam_separation_deg / 180.0)
            beam_coupling = max(self.min_spatial_coupling, 1.0 - normalized_separation)
            if int(other_input.get("precoder_id", -1)) == int(target_input.get("precoder_id", -2)):
                beam_coupling *= self.same_precoder_coupling_multiplier
            if (
                mu_context["group_id"] is not None
                and other_input.get("spatial_group_id", None) == mu_context["group_id"]
            ):
                beam_coupling *= max(
                    0.15,
                    1.0 - float(self.same_group_interference_reduction) * float(mu_context["compatibility_score"]),
                )

            layer_pressure = float(max(1, int(other_input.get("layer_count", 1)))) / float(target_num_rx_ant)
            total_interference_ratio += (
                effective_coupling_fraction * beam_coupling * max(0.25, layer_pressure)
            )

        if total_interference_ratio <= 0.0:
            return (
                0.0,
                max_overlap_fraction,
                float(mu_context["multiplexing_gain_db"]),
                float(mu_context["compatibility_score"]),
                int(mu_context["group_size"]),
            )

        return (
            10.0 * math.log10(1.0 + total_interference_ratio),
            max_overlap_fraction,
            float(mu_context["multiplexing_gain_db"]),
            float(mu_context["compatibility_score"]),
            int(mu_context["group_size"]),
        )


    def _generate_bits(
        self,
        slot_id: int,
        ue_id: str,
        num_bits: int,
    ) -> Any:
        """
        Generates one repeatable bit vector.
        INPUT
        slot_id: The current slot index.
        ue_id: The UE identifier.
        num_bits: The number of bits.
        OUTPUT
        One TensorFlow bit tensor.
        EXAMPLE OUTPUT
        tf.Tensor([...], dtype=int32)
        """
        seed = 1000 + 37 * int(slot_id) + 19 * sum(ord(character) for character in ue_id)
        np = self._get_numpy()
        random_state = np.random.RandomState(seed)
        return random_state.randint(0, 2, size=num_bits).astype(np.int32)

    def _modulate_qpsk(self, bits: Any) -> Any:
        """
        Maps one bit vector to QPSK symbols.
        INPUT
        bits: The input bit tensor.
        OUTPUT
        One complex QPSK symbol tensor.
        EXAMPLE OUTPUT
        tf.Tensor([...], dtype=complex64)
        """
        np = self._get_numpy()
        bit_pairs = np.reshape(bits, (-1, 2))
        inphase = 1.0 - 2.0 * bit_pairs[:, 0].astype(np.float32)
        quadrature = 1.0 - 2.0 * bit_pairs[:, 1].astype(np.float32)
        return (inphase + 1j * quadrature) / math.sqrt(2.0)

    def _demodulate_qpsk(self, rx_symbols: Any) -> Any:
        """
        Hard-demodulates QPSK symbols to bits.
        INPUT
        rx_symbols: The received symbol tensor.
        OUTPUT
        One recovered bit tensor.
        EXAMPLE OUTPUT
        tf.Tensor([...], dtype=int32)
        """
        np = self._get_numpy()
        real_bits = (np.real(rx_symbols) < 0.0).astype(np.int32)
        imag_bits = (np.imag(rx_symbols) < 0.0).astype(np.int32)
        stacked_bits = np.stack([real_bits, imag_bits], axis=1)
        return np.reshape(stacked_bits, (-1,))

    def _gray_bits_to_binary_indices(self, gray_bits: Any) -> Any:
        np = self._get_numpy()
        binary_bits = np.asarray(gray_bits, dtype=np.int32).copy()
        for bit_offset in range(1, binary_bits.shape[1]):
            binary_bits[:, bit_offset] = np.bitwise_xor(
                binary_bits[:, bit_offset - 1],
                binary_bits[:, bit_offset],
            )

        bit_weights = 1 << np.arange(binary_bits.shape[1] - 1, -1, -1, dtype=np.int32)
        return np.sum(binary_bits * bit_weights, axis=1, dtype=np.int32)

    def _binary_indices_to_gray_bits(self, binary_indices: Any, bits_per_axis: int) -> Any:
        np = self._get_numpy()
        bit_shifts = np.arange(bits_per_axis - 1, -1, -1, dtype=np.int32)
        binary_bits = ((np.asarray(binary_indices, dtype=np.int32)[:, None] >> bit_shifts) & 1).astype(np.int32)
        gray_bits = np.empty_like(binary_bits)
        gray_bits[:, 0] = binary_bits[:, 0]
        for bit_offset in range(1, bits_per_axis):
            gray_bits[:, bit_offset] = np.bitwise_xor(
                binary_bits[:, bit_offset - 1],
                binary_bits[:, bit_offset],
            )
        return gray_bits

    def _square_qam_normalization(self, bits_per_axis: int) -> float:
        axis_level_count = 1 << int(bits_per_axis)
        return math.sqrt((2.0 / 3.0) * float(axis_level_count**2 - 1))

    def _modulate_square_qam(self, bits: Any, bits_per_symbol: int) -> Any:
        np = self._get_numpy()
        if int(bits_per_symbol) < 2 or int(bits_per_symbol) % 2 != 0:
            raise ValueError("Square QAM requires an even bits_per_symbol >= 2.")

        bits_per_axis = int(bits_per_symbol) // 2
        bit_groups = np.reshape(np.asarray(bits, dtype=np.int32), (-1, int(bits_per_symbol)))
        real_indices = self._gray_bits_to_binary_indices(bit_groups[:, :bits_per_axis])
        imag_indices = self._gray_bits_to_binary_indices(bit_groups[:, bits_per_axis:])
        axis_level_count = 1 << bits_per_axis
        normalization = self._square_qam_normalization(bits_per_axis)

        real_levels = (axis_level_count - 1 - 2 * real_indices).astype(np.float32)
        imag_levels = (axis_level_count - 1 - 2 * imag_indices).astype(np.float32)
        return (real_levels + 1j * imag_levels) / normalization

    def _demodulate_square_qam(self, rx_symbols: Any, bits_per_symbol: int) -> Any:
        np = self._get_numpy()
        if int(bits_per_symbol) < 2 or int(bits_per_symbol) % 2 != 0:
            raise ValueError("Square QAM requires an even bits_per_symbol >= 2.")

        bits_per_axis = int(bits_per_symbol) // 2
        axis_level_count = 1 << bits_per_axis
        normalization = self._square_qam_normalization(bits_per_axis)
        real_value = np.real(rx_symbols) * normalization
        imag_value = np.imag(rx_symbols) * normalization

        real_indices = np.clip(
            np.rint((float(axis_level_count - 1) - real_value) / 2.0),
            0,
            axis_level_count - 1,
        ).astype(np.int32)
        imag_indices = np.clip(
            np.rint((float(axis_level_count - 1) - imag_value) / 2.0),
            0,
            axis_level_count - 1,
        ).astype(np.int32)

        real_bits = self._binary_indices_to_gray_bits(real_indices, bits_per_axis)
        imag_bits = self._binary_indices_to_gray_bits(imag_indices, bits_per_axis)
        output_bits = np.concatenate([real_bits, imag_bits], axis=1)
        return np.reshape(output_bits, (-1,))

    def _modulate_16qam(self, bits: Any) -> Any:
        """
        Maps one bit vector to 16QAM symbols.
        INPUT
        bits: The input bit tensor.
        OUTPUT
        One complex 16QAM symbol tensor.
        EXAMPLE OUTPUT
        tf.Tensor([...], dtype=complex64)
        """
        return self._modulate_square_qam(bits, bits_per_symbol=4)

    def _demodulate_16qam(self, rx_symbols: Any) -> Any:
        """
        Hard-demodulates 16QAM symbols to bits.
        INPUT
        rx_symbols: The received symbol tensor.
        OUTPUT
        One recovered bit tensor.
        EXAMPLE OUTPUT
        tf.Tensor([...], dtype=int32)
        """
        return self._demodulate_square_qam(rx_symbols, bits_per_symbol=4)

    def _modulate_64qam(self, bits: Any) -> Any:
        return self._modulate_square_qam(bits, bits_per_symbol=6)

    def _demodulate_64qam(self, rx_symbols: Any) -> Any:
        return self._demodulate_square_qam(rx_symbols, bits_per_symbol=6)

    def _modulate_256qam(self, bits: Any) -> Any:
        return self._modulate_square_qam(bits, bits_per_symbol=8)

    def _demodulate_256qam(self, rx_symbols: Any) -> Any:
        return self._demodulate_square_qam(rx_symbols, bits_per_symbol=8)

    def _modulate(self, bits: Any, modulation: str) -> Any:
        """
        Maps bits to symbols according to modulation scheme.
        INPUT
        bits: The input bit tensor.
        modulation: The modulation scheme.
        OUTPUT
        One symbol tensor.
        EXAMPLE OUTPUT
        tf.Tensor([...], dtype=complex64)
        """
        if modulation == "16qam":
            return self._modulate_16qam(bits)
        if modulation == "64qam":
            return self._modulate_64qam(bits)
        if modulation == "256qam":
            return self._modulate_256qam(bits)
        return self._modulate_qpsk(bits)

    def _demodulate(self, rx_symbols: Any, modulation: str) -> Any:
        """
        Demodulates received symbols according to modulation scheme.
        INPUT
        rx_symbols: The received symbol tensor.
        modulation: The modulation scheme.
        OUTPUT
        One recovered bit tensor.
        EXAMPLE OUTPUT
        tf.Tensor([...], dtype=int32)
        """
        if modulation == "16qam":
            return self._demodulate_16qam(rx_symbols)
        if modulation == "64qam":
            return self._demodulate_64qam(rx_symbols)
        if modulation == "256qam":
            return self._demodulate_256qam(rx_symbols)
        return self._demodulate_qpsk(rx_symbols)

    def _apply_awgn(
        self,
        tx_symbols: Any,
        effective_sinr_db: float,
        slot_id: int,
        ue_id: str,
    ) -> Any:
        """
        Applies one AWGN channel to transmitted symbols.
        INPUT
        tx_symbols: The transmitted symbol tensor.
        effective_sinr_db: The effective SINR in dB.
        slot_id: The current slot index.
        ue_id: The UE identifier.
        OUTPUT
        One noisy symbol tensor.
        EXAMPLE OUTPUT
        tf.Tensor([...], dtype=complex64)
        """
        snr_linear = 10.0 ** (effective_sinr_db / 10.0)
        noise_variance = 1.0 / max(snr_linear, 1e-6)

        seed = 2000 + 41 * int(slot_id) + 23 * sum(ord(character) for character in ue_id)
        np = self._get_numpy()
        random_state = np.random.RandomState(seed)

        symbol_count = int(len(tx_symbols))

        real_noise_np = random_state.normal(
            loc=0.0,
            scale=math.sqrt(noise_variance / 2.0),
            size=symbol_count,
        ).astype(np.float32)

        imag_noise_np = random_state.normal(
            loc=0.0,
            scale=math.sqrt(noise_variance / 2.0),
            size=symbol_count,
        ).astype(np.float32)

        complex_noise = real_noise_np + 1j * imag_noise_np
        return tx_symbols + complex_noise.astype(np.complex64)

    def _compute_ber(self, tx_bits: Any, rx_bits: Any) -> float:
        """
        Computes BER from transmitted and recovered bits.
        INPUT
        tx_bits: The transmitted bit tensor.
        rx_bits: The recovered bit tensor.
        OUTPUT
        One BER value.
        EXAMPLE OUTPUT
        0.01
        """
        np = self._get_numpy()
        tx_bits_np = np.asarray(tx_bits)
        rx_bits_np = np.asarray(rx_bits)
        total_bits = int(tx_bits_np.size)
        if total_bits <= 0:
            return 0.0
        bit_errors = int(np.sum(tx_bits_np != rx_bits_np))
        return float(bit_errors) / float(total_bits)

    def _estimate_bler_from_sinr(
        self,
        effective_sinr_db: float,
    ) -> float:
        """
        Estimates BLER using one smooth sigmoid-like mapping from SINR.
        INPUT
        effective_sinr_db: The effective SINR in dB.
        OUTPUT
        One BLER estimate in [0, 1].
        EXAMPLE OUTPUT
        0.18
        """
        center_sinr_db = float(
            self.bler_model_config.get("center_sinr_db", 12.0)
        )
        slope = float(self.bler_model_config.get("slope", 0.9))
        x_value = (effective_sinr_db - center_sinr_db) / max(slope, 1e-6)
        bler = 1.0 / (1.0 + math.exp(x_value))
        return max(0.0, min(1.0, bler))

    def _estimate_throughput_mbps(
        self,
        num_bits: int,
        estimated_bler: float,
        slot_duration_ms: float,
    ) -> float:
        """
        Estimates throughput in Mbps from block size and BLER.
        INPUT
        num_bits: The number of transmitted bits.
        estimated_bler: The estimated BLER.
        OUTPUT
        One throughput estimate in Mbps.
        EXAMPLE OUTPUT
        8.2
        """
        successful_bits = float(num_bits) * max(0.0, min(1.0, 1.0 - estimated_bler))
        slot_duration_s = max(float(slot_duration_ms), 1e-6) / 1000.0
        return successful_bits / 1e6 / slot_duration_s

    def run_link_simulation(
        self,
        engine_input: Dict[str, Any],
        *,
        interference_penalty_db: float = 0.0,
        resource_grid_overlap_fraction: float = 0.0,
        spatial_multiplexing_gain_db: float = 0.0,
        spatial_compatibility_score: float = 0.0,
        spatial_group_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Runs one minimal link-level simulation for one UE.
        INPUT
        engine_input: One engine input dictionary.
        OUTPUT
        One engine result dictionary.
        EXAMPLE OUTPUT
        {"effective_sinr_db": 12.0, "bler": 0.14}
        """
        resource_grid_view = self._build_resource_grid_view(engine_input)
        pre_eq_sinr_db = self._clip_sinr_db(self._compute_link_sinr_db(engine_input))
        mimo_state = self._evaluate_single_user_mimo(
            engine_input,
            pre_eq_sinr_db=pre_eq_sinr_db,
        )
        effective_sinr_db = float(mimo_state["post_eq_sinr_db"])
        effective_sinr_db = (
            float(effective_sinr_db)
            - float(self.sinr_margin_db)
            - float(interference_penalty_db)
            - float(engine_input.get("inter_cell_interference_penalty_db", 0.0) or 0.0)
            + float(spatial_multiplexing_gain_db)
        )
        effective_sinr_db = self._clip_sinr_db(effective_sinr_db)

        beam_gain_db = (
            self.codebook_gain_step_db * float(engine_input.get("precoder_id", engine_input["codebook_id"]))
            + self.beam_profile_gain_step_db * float(engine_input["beam_profile_id"])
            + float(engine_input.get("beamforming_gain_db", 0.0))
        )

        slot_duration_ms = float(engine_input.get("slot_duration_ms", 1.0))
        allocated_prbs = int(engine_input.get("allocated_prbs", 0))
        allocated_symbols = int(engine_input.get("allocated_symbols", 0))
        start_prb = int(engine_input.get("start_prb", 0))
        start_symbol = int(engine_input.get("start_symbol", 0))
        is_retx = bool(engine_input.get("is_retx", False))

        if not engine_input["is_scheduled"]:
            phy_mode = self._normalize_phy_mode(engine_input.get("phy_mode", self.phy_mode))
            return {
                "sinr_db": effective_sinr_db,
                "effective_sinr_db": effective_sinr_db,
                "pre_eq_sinr_db": float(mimo_state["pre_eq_sinr_db"]),
                "post_eq_sinr_db": effective_sinr_db,
                "beam_gain_db": beam_gain_db,
                "allocated_prbs": allocated_prbs,
                "allocated_symbols": allocated_symbols,
                "start_prb": start_prb,
                "start_symbol": start_symbol,
                "is_retx": is_retx,
                "requested_bytes": int(engine_input.get("requested_bytes", 0)),
                "grant_bytes_budget": int(engine_input.get("grant_bytes_budget", 0)),
                "mcs_table_id": str(engine_input.get("mcs_table_id", "qam64")),
                "tbs_selection_method": str(engine_input.get("tbs_selection_method", "resource_elements")),
                "dmrs_pattern_id": engine_input.get("dmrs_pattern_id", None),
                "harq_process_id": engine_input.get("harq_process_id", None),
                "harq_redundancy_version": engine_input.get("harq_redundancy_version", None),
                "harq_ndi": engine_input.get("harq_ndi", None),
                "harq_retransmission_round": int(engine_input.get("harq_retransmission_round", 0) or 0),
                "cqi_validity_age_slots": int(engine_input.get("cqi_validity_age_slots", 0) or 0),
                "ri_validity_age_slots": int(engine_input.get("ri_validity_age_slots", 0) or 0),
                "pmi_validity_age_slots": int(engine_input.get("pmi_validity_age_slots", 0) or 0),
                "channel_quality_confidence": engine_input.get("channel_quality_confidence", None),
                "link_adaptation_reason": engine_input.get("link_adaptation_reason", None),
                "scheduled_bytes": 0,
                "delivered_bytes": 0,
                "mcs_index": 0,
                "tb_size_bytes": 0,
                "transport_block_size_bytes": 0,
                "ack": False,
                "bler": 0.0,
                "throughput_mbps": 0.0,
                "ber": 0.0,
                "cqi": 0,
                "rank": int(mimo_state["channel_rank_limit"]),
                "pmi": int(engine_input.get("reported_pmi", engine_input.get("combiner_id", 0))),
                "ri": int(engine_input.get("reported_ri", engine_input.get("layer_count", 1))),
                "channel_rank_limit": int(mimo_state["channel_rank_limit"]),
                "precoder_match_score": float(mimo_state["precoder_match_score"]),
                "spatial_mode": "mu_mimo" if int(spatial_group_size) > 1 else str(mimo_state["spatial_mode"]),
                "spatial_group_id": engine_input.get("spatial_group_id", None),
                "spatial_group_size": int(spatial_group_size),
                "spatial_compatibility_score": float(spatial_compatibility_score),
                "spatial_multiplexing_gain_db": float(spatial_multiplexing_gain_db),
                "resource_grid_re_count": int(resource_grid_view["resource_elements"]),
                "resource_grid_overlap_fraction": float(resource_grid_overlap_fraction),
                "interference_penalty_db": float(
                    float(interference_penalty_db)
                    + float(engine_input.get("inter_cell_interference_penalty_db", 0.0) or 0.0)
                ),
                "intra_cell_interference_penalty_db": float(interference_penalty_db),
                "inter_cell_interference_dbm": float(
                    engine_input.get("inter_cell_interference_dbm", -120.0) or -120.0
                ),
                "inter_cell_interference_penalty_db": float(
                    engine_input.get("inter_cell_interference_penalty_db", 0.0) or 0.0
                ),
                "inter_cell_interference_overlap_fraction": float(
                    engine_input.get("inter_cell_interference_overlap_fraction", 0.0) or 0.0
                ),
                "inter_cell_frequency_coupling_fraction": float(
                    engine_input.get("inter_cell_frequency_coupling_fraction", 0.0) or 0.0
                ),
                "inter_cell_interference_source_count": int(
                    engine_input.get("inter_cell_interference_source_count", 0) or 0
                ),
                "phy_mode": str(phy_mode),
                "waveform_backend": None,
                "effective_data_re_count": int(resource_grid_view["resource_elements"]),
                "dmrs_overhead_fraction": 0.0,
            }

        phy_mode = self._normalize_phy_mode(engine_input.get("phy_mode", self.phy_mode))
        modulation_config = self._select_modulation_and_mcs(
            effective_sinr_db,
            target_mcs=engine_input.get("target_mcs"),
            mcs_cap=engine_input.get("mcs_cap"),
            mcs_table_id=engine_input.get("mcs_table_id"),
            num_layers=max(1, int(engine_input.get("layer_count", 1))),
        )
        bits_per_symbol = int(modulation_config["bits_per_symbol"])
        mcs_index = int(modulation_config["mcs_index"])
        target_coderate = float(modulation_config.get("target_coderate", 0.5))
        target_coderate = max(1e-3, min(0.99, target_coderate))
        sionna_target_coderate = self._normalize_target_coderate_for_sionna(target_coderate)
        layer_count = max(1, int(engine_input.get("layer_count", 1)))
        mcs_table_id = normalize_mcs_table_id(engine_input.get("mcs_table_id"))
        tbs_selection_method = normalize_tbs_selection_method(engine_input.get("tbs_selection_method"))
        dmrs_pattern_id = engine_input.get("dmrs_pattern_id", None)

        queue_bytes = int(max(0, int(engine_input.get("queue_bytes", 0))))
        requested_bytes = int(max(0, int(engine_input.get("requested_bytes", queue_bytes))))
        grant_bytes_budget = int(max(0, int(engine_input.get("grant_bytes_budget", requested_bytes))))
        harq_retransmission_round = int(engine_input.get("harq_retransmission_round", 0) or 0)
        effective_grant_budget = grant_bytes_budget if grant_bytes_budget > 0 else requested_bytes
        max_tb_size_bytes = int(max(1, int(self.num_bits_per_block) // 8))
        waveform_backend = None
        dmrs_overhead_fraction = 0.0
        effective_data_re_count = self._compute_num_mod_symbols(engine_input)
        if phy_mode == "sionna_cdl_su_mimo":
            waveform_context = self.cdl_su_mimo_model.build_waveform_context(
                engine_input=engine_input,
                resource_grid_view=resource_grid_view,
                pre_eq_sinr_db=float(mimo_state["pre_eq_sinr_db"]),
                mimo_state=mimo_state,
            )
            effective_sinr_db += float(waveform_context.sinr_adjustment_db)
            effective_sinr_db = self._clip_sinr_db(effective_sinr_db)
            waveform_backend = str(waveform_context.waveform_backend)
            dmrs_overhead_fraction = float(waveform_context.dmrs_overhead_fraction)
            effective_data_re_count = int(waveform_context.effective_data_resource_elements)
        num_mod_symbols = int(max(1, effective_data_re_count))
        tb_size_bytes_budget = estimate_tbs_bytes(
            resource_elements=int(num_mod_symbols),
            mcs_index=mcs_index,
            mcs_table_id=mcs_table_id,
            num_layers=layer_count,
            tbs_selection_method=tbs_selection_method,
            dmrs_pattern_id=dmrs_pattern_id,
        )

        tb_size_bytes = int(min(queue_bytes, requested_bytes, effective_grant_budget, max_tb_size_bytes, tb_size_bytes_budget))
        tb_info_bits = int(max(1, tb_size_bytes * 8))
        codeblock_info_bits_list = self._split_transport_block_info_bits(tb_info_bits)

        ebno_db = float(effective_sinr_db) - 10.0 * math.log10(
            max(1e-6, float(bits_per_symbol) * float(sionna_target_coderate))
        )

        ue_id_str = str(engine_input.get("ue_id", "ue"))
        ue_hash = sum(ord(character) for character in ue_id_str)
        seed = 1000 + 37 * int(engine_input["slot_id"]) + 19 * int(ue_hash)
        self._set_sionna_seed(seed)

        tf = self._get_tf()
        batch_size = max(1, int(self.batch_size))
        total_bit_error_count = 0.0
        total_bit_count = 0.0
        transport_block_errors = None
        for codeblock_info_bits in codeblock_info_bits_list:
            channel = self._get_coded_awgn_channel(
                bits_per_symbol=bits_per_symbol,
                num_info_bits=int(codeblock_info_bits),
                target_coderate=sionna_target_coderate,
            )
            with tf.device(self.device_name):
                b, b_hat = channel(
                    batch_size=batch_size,
                    ebno_db=tf.constant(ebno_db, dtype=tf.float32),
                )

            b_int = tf.cast(b, tf.int32)
            b_hat_int = tf.cast(tf.round(b_hat), tf.int32)
            bit_errors = tf.not_equal(b_int, b_hat_int)
            block_errors = tf.reduce_any(bit_errors, axis=1)
            total_bit_error_count += float(tf.reduce_sum(tf.cast(bit_errors, tf.float32)).numpy())
            total_bit_count += float(batch_size * int(codeblock_info_bits))
            if transport_block_errors is None:
                transport_block_errors = block_errors
            else:
                transport_block_errors = tf.logical_or(transport_block_errors, block_errors)

        if transport_block_errors is None:
            transport_block_errors = tf.zeros((batch_size,), dtype=tf.bool)
        ber = total_bit_error_count / max(1.0, total_bit_count)
        block_error_count = float(tf.reduce_sum(tf.cast(transport_block_errors, tf.float32)).numpy())
        block_count = float(batch_size)
        if self.bler_smoothing_alpha > 0.0 or self.bler_smoothing_beta > 0.0:
            bler = (block_error_count + float(self.bler_smoothing_alpha)) / (
                block_count + float(self.bler_smoothing_alpha) + float(self.bler_smoothing_beta)
            )
        else:
            bler = block_error_count / block_count if block_count > 0.0 else 0.0

        np = self._get_numpy()
        ack_seed = 4000 + 29 * int(engine_input["slot_id"]) + 31 * sum(
            ord(character) for character in str(engine_input.get("ue_id", "ue"))
        ) + 17 * int(harq_retransmission_round)
        ack_random = float(np.random.RandomState(ack_seed).rand())
        ack = bool(ack_random <= max(0.0, min(1.0, 1.0 - float(bler))))

        throughput_mbps = (
            float(tb_size_bytes * 8) / 1e6 / max(slot_duration_ms, 1e-6) * 1000.0
            if ack else 0.0
        )
        delivered_bytes = int(tb_size_bytes) if ack else 0

        cqi = int(max(0, min(15, round((effective_sinr_db + 5.0) / 2.0))))

        return {
            "sinr_db": effective_sinr_db,
            "effective_sinr_db": effective_sinr_db,
            "pre_eq_sinr_db": float(mimo_state["pre_eq_sinr_db"]),
            "post_eq_sinr_db": effective_sinr_db,
            "beam_gain_db": beam_gain_db,
            "allocated_prbs": allocated_prbs,
            "allocated_symbols": allocated_symbols,
            "start_prb": start_prb,
            "start_symbol": start_symbol,
            "is_retx": is_retx,
            "requested_bytes": requested_bytes,
            "grant_bytes_budget": grant_bytes_budget,
            "mcs_table_id": mcs_table_id,
            "tbs_selection_method": tbs_selection_method,
            "dmrs_pattern_id": dmrs_pattern_id,
            "harq_process_id": engine_input.get("harq_process_id", None),
            "harq_redundancy_version": engine_input.get("harq_redundancy_version", None),
            "harq_ndi": engine_input.get("harq_ndi", None),
            "harq_retransmission_round": harq_retransmission_round,
            "cqi_validity_age_slots": int(engine_input.get("cqi_validity_age_slots", 0) or 0),
            "ri_validity_age_slots": int(engine_input.get("ri_validity_age_slots", 0) or 0),
            "pmi_validity_age_slots": int(engine_input.get("pmi_validity_age_slots", 0) or 0),
            "channel_quality_confidence": engine_input.get("channel_quality_confidence", None),
            "link_adaptation_reason": engine_input.get("link_adaptation_reason", None),
            "scheduled_bytes": tb_size_bytes,
            "delivered_bytes": delivered_bytes,
            "mcs_index": mcs_index,
            "modulation_order": bits_per_symbol,
            "code_rate": sionna_target_coderate,
            "num_layers": layer_count,
            "tb_size_bytes": tb_size_bytes,
            "transport_block_size_bytes": tb_size_bytes,
            "ack": ack,
            "bler": bler,
            "throughput_mbps": throughput_mbps,
            "ber": ber,
            "cqi": cqi,
            "rank": int(mimo_state["channel_rank_limit"]),
            "pmi": int(engine_input.get("reported_pmi", engine_input.get("combiner_id", 0))),
            "ri": int(engine_input.get("reported_ri", layer_count)),
            "channel_rank_limit": int(mimo_state["channel_rank_limit"]),
            "precoder_match_score": float(mimo_state["precoder_match_score"]),
            "spatial_mode": "mu_mimo" if int(spatial_group_size) > 1 else str(mimo_state["spatial_mode"]),
            "spatial_group_id": engine_input.get("spatial_group_id", None),
            "spatial_group_size": int(spatial_group_size),
            "spatial_compatibility_score": float(spatial_compatibility_score),
            "spatial_multiplexing_gain_db": float(spatial_multiplexing_gain_db),
            "resource_grid_re_count": int(resource_grid_view["resource_elements"]),
            "resource_grid_overlap_fraction": float(resource_grid_overlap_fraction),
            "interference_penalty_db": float(
                float(interference_penalty_db)
                + float(engine_input.get("inter_cell_interference_penalty_db", 0.0) or 0.0)
            ),
            "intra_cell_interference_penalty_db": float(interference_penalty_db),
            "inter_cell_interference_dbm": float(
                engine_input.get("inter_cell_interference_dbm", -120.0) or -120.0
            ),
            "inter_cell_interference_penalty_db": float(
                engine_input.get("inter_cell_interference_penalty_db", 0.0) or 0.0
            ),
            "inter_cell_interference_overlap_fraction": float(
                engine_input.get("inter_cell_interference_overlap_fraction", 0.0) or 0.0
            ),
            "inter_cell_frequency_coupling_fraction": float(
                engine_input.get("inter_cell_frequency_coupling_fraction", 0.0) or 0.0
            ),
            "inter_cell_interference_source_count": int(
                engine_input.get("inter_cell_interference_source_count", 0) or 0
            ),
            "phy_mode": str(phy_mode),
            "waveform_backend": waveform_backend,
            "effective_data_re_count": int(effective_data_re_count),
            "dmrs_overhead_fraction": float(dmrs_overhead_fraction),
        }


    def run_batch_link_simulation(
        self,
        engine_inputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Runs one batched PHY execution over multiple UE engine inputs.
        INPUT
        engine_inputs: One list of engine input dictionaries.
        OUTPUT
        One list of engine result dictionaries.
        EXAMPLE OUTPUT
        [result_0, result_1, result_2]
        """
        results = []
        for engine_input in engine_inputs:
            (
                interference_penalty_db,
                overlap_fraction,
                spatial_multiplexing_gain_db,
                spatial_compatibility_score,
                spatial_group_size,
            ) = self._compute_overlap_interference_penalty_db(
                engine_input,
                engine_inputs,
            )
            results.append(
                self.run_link_simulation(
                    engine_input,
                    interference_penalty_db=interference_penalty_db,
                    resource_grid_overlap_fraction=overlap_fraction,
                    spatial_multiplexing_gain_db=spatial_multiplexing_gain_db,
                    spatial_compatibility_score=spatial_compatibility_score,
                    spatial_group_size=spatial_group_size,
                )
            )
        return results



