"""
Waveform-aware SU-MIMO/CDL helper logic used by the Sionna PHY dual path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict


@dataclass(frozen=True)
class CdlSuMimoWaveformContext:
    phy_mode: str
    waveform_backend: str
    dmrs_overhead_fraction: float
    effective_data_resource_elements: int
    sinr_adjustment_db: float
    delay_selectivity_penalty_db: float
    cdl_richness_bonus_db: float
    cp_duration_ns: float
    delay_spread_to_cp_ratio: float


class CdlSuMimoModel:
    """
    Builds a waveform-aware execution context for a single-user CDL-like OFDM path.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        normalized_config = dict(config or {})
        self.config = normalized_config
        self.waveform_backend = str(
            normalized_config.get("waveform_backend", "cdl_su_mimo_proxy")
        )
        self.delay_penalty_scale_db = float(
            normalized_config.get("delay_penalty_scale_db", 1.5)
        )
        self.k_factor_bonus_scale_db = float(
            normalized_config.get("k_factor_bonus_scale_db", 0.8)
        )
        self.angle_spread_bonus_scale_db = float(
            normalized_config.get("angle_spread_bonus_scale_db", 0.6)
        )
        self.spatial_correlation_penalty_scale_db = float(
            normalized_config.get("spatial_correlation_penalty_scale_db", 1.0)
        )
        self.los_bonus_db = float(normalized_config.get("los_bonus_db", 0.25))
        self.dmrs_overhead_fraction_default = float(
            normalized_config.get("dmrs_overhead_fraction_default", 0.0714)
        )
        self.dmrs_overhead_fraction_dense = float(
            normalized_config.get("dmrs_overhead_fraction_dense", 0.1428)
        )

    def _estimate_cp_duration_ns(
        self,
        *,
        subcarrier_spacing_hz: float,
        cyclic_prefix_type: str,
    ) -> float:
        normalized_scs_hz = max(1.0, float(subcarrier_spacing_hz))
        cp_type = str(cyclic_prefix_type).strip().lower()
        base_cp_ns = 16670.0 if cp_type == "extended" else 4700.0
        return float(base_cp_ns * (15e3 / normalized_scs_hz))

    def _estimate_dmrs_overhead_fraction(
        self,
        *,
        dmrs_pattern_id: str | None,
        num_layers: int,
    ) -> float:
        normalized_pattern_id = str(dmrs_pattern_id or "").strip().lower()
        if "dense" in normalized_pattern_id or "double" in normalized_pattern_id:
            base_fraction = self.dmrs_overhead_fraction_dense
        elif normalized_pattern_id != "":
            base_fraction = self.dmrs_overhead_fraction_default * 1.1
        else:
            base_fraction = self.dmrs_overhead_fraction_default

        layer_factor = 1.0 + 0.08 * max(0, int(num_layers) - 1)
        return float(min(0.3, max(0.02, base_fraction * layer_factor)))

    def build_waveform_context(
        self,
        *,
        engine_input: Dict[str, Any],
        resource_grid_view: Dict[str, Any],
        pre_eq_sinr_db: float,
        mimo_state: Dict[str, Any],
    ) -> CdlSuMimoWaveformContext:
        num_layers = max(1, int(engine_input.get("layer_count", 1)))
        dmrs_overhead_fraction = self._estimate_dmrs_overhead_fraction(
            dmrs_pattern_id=engine_input.get("dmrs_pattern_id", None),
            num_layers=num_layers,
        )

        raw_resource_elements = max(1, int(resource_grid_view.get("resource_elements", 0)))
        effective_data_resource_elements = max(
            1,
            int(round(float(raw_resource_elements) * (1.0 - dmrs_overhead_fraction))),
        )

        cp_duration_ns = self._estimate_cp_duration_ns(
            subcarrier_spacing_hz=float(engine_input.get("subcarrier_spacing_hz", 30e3)),
            cyclic_prefix_type=str(engine_input.get("cyclic_prefix_type", "normal")),
        )
        delay_spread_ns = max(0.0, float(engine_input.get("delay_spread_ns", 300.0)))
        delay_spread_to_cp_ratio = float(delay_spread_ns / max(cp_duration_ns, 1.0))
        delay_selectivity_penalty_db = self.delay_penalty_scale_db * min(
            1.0,
            max(0.0, delay_spread_to_cp_ratio),
        )

        k_factor_db = max(0.0, float(engine_input.get("k_factor_db", 0.0)))
        angle_spread_deg = max(0.0, float(engine_input.get("angle_spread_deg", 35.0)))
        spatial_correlation_level = max(
            0.0,
            min(1.0, float(engine_input.get("spatial_correlation_level", 0.5))),
        )
        los_state = str(engine_input.get("los_state", "los")).strip().lower()
        channel_profile = str(engine_input.get("channel_profile", "tdl_c")).strip().lower()

        k_factor_bonus_db = self.k_factor_bonus_scale_db * min(1.0, k_factor_db / 12.0)
        angle_spread_bonus_db = self.angle_spread_bonus_scale_db * min(1.0, angle_spread_deg / 60.0)
        spatial_corr_penalty_db = (
            self.spatial_correlation_penalty_scale_db
            * spatial_correlation_level
            * max(0, num_layers - 1)
        )
        los_bonus_db = self.los_bonus_db if los_state == "los" else 0.0
        channel_richness_bonus_db = 0.4 if ("cdl" in channel_profile or channel_profile.endswith("_d")) else 0.15
        waveform_bonus_db = 0.15 * min(
            1.0,
            max(0.0, float(mimo_state.get("precoder_match_score", 1.0))),
        )

        sinr_adjustment_db = (
            los_bonus_db
            + k_factor_bonus_db
            + angle_spread_bonus_db
            + channel_richness_bonus_db
            + waveform_bonus_db
            - delay_selectivity_penalty_db
            - spatial_corr_penalty_db
        )

        return CdlSuMimoWaveformContext(
            phy_mode="sionna_cdl_su_mimo",
            waveform_backend=str(self.waveform_backend),
            dmrs_overhead_fraction=float(dmrs_overhead_fraction),
            effective_data_resource_elements=int(effective_data_resource_elements),
            sinr_adjustment_db=float(sinr_adjustment_db),
            delay_selectivity_penalty_db=float(delay_selectivity_penalty_db),
            cdl_richness_bonus_db=float(
                los_bonus_db
                + k_factor_bonus_db
                + angle_spread_bonus_db
                + channel_richness_bonus_db
                + waveform_bonus_db
            ),
            cp_duration_ns=float(cp_duration_ns),
            delay_spread_to_cp_ratio=float(delay_spread_to_cp_ratio),
        )
