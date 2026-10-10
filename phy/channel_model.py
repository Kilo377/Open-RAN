"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: channel_model.py
Description: This file defines a simplified channel model used by the
custom PHY adapter. The current implementation separates large-scale and
small-scale channel effects and produces one slot-level effective channel
quality estimate from a stable UE radio-quality baseline.
"""

from models.phy_input import PhyUeInput
from models.phy_runtime_context import PhyRuntimeContext


class SimpleChannelModel:
    """
    Implements a simplified effective channel model.
    INPUT
    No explicit input during initialization.
    OUTPUT
    A reusable channel model object.
    EXAMPLE OUTPUT
    channel_model = SimpleChannelModel()
    effective_sinr_db = channel_model.compute_effective_sinr(ue_input, runtime_context)
    """

    def _compute_large_scale_loss_db(
        self,
        ue_input: PhyUeInput,
        runtime_context: PhyRuntimeContext,
    ) -> float:
        """
        Computes one large-scale channel loss term.
        INPUT
        runtime_context: The PHY runtime context object.
        OUTPUT
        One large-scale loss value in dB.
        EXAMPLE OUTPUT
        9.2
        """
        pathloss_db = getattr(ue_input, "link_pathloss_db", runtime_context.pathloss_db)
        if pathloss_db is None:
            pathloss_db = runtime_context.pathloss_db
        shadowing_db = getattr(ue_input, "link_shadowing_db", runtime_context.shadowing_db)
        if shadowing_db is None:
            shadowing_db = runtime_context.shadowing_db
        o2i_penetration_loss_db = getattr(ue_input, "link_o2i_penetration_loss_db", 0.0)
        if o2i_penetration_loss_db is None:
            o2i_penetration_loss_db = 0.0
        pathloss_db = float(pathloss_db)
        shadowing_db = float(shadowing_db)
        o2i_penetration_loss_db = float(o2i_penetration_loss_db)
        pathloss_component_db = 0.04 * (pathloss_db - 100.0)
        shadowing_component_db = 0.25 * shadowing_db
        o2i_component_db = 0.04 * o2i_penetration_loss_db
        return pathloss_component_db + shadowing_component_db + o2i_component_db

    def _compute_small_scale_variation_db(
        self,
        ue_input: PhyUeInput,
        runtime_context: PhyRuntimeContext,
    ) -> float:
        """
        Computes one simplified small-scale fading variation term.
        INPUT
        ue_input: The PHY UE input object.
        runtime_context: The PHY runtime context object.
        OUTPUT
        One small-scale variation value in dB.
        EXAMPLE OUTPUT
        1.2
        """
        fast_fading_seed = int(getattr(ue_input, "link_fast_fading_seed", 0) or 0)
        doppler_hz = getattr(ue_input, "link_doppler_hz", getattr(runtime_context, "doppler_hz", 0.0))
        if doppler_hz is None:
            doppler_hz = getattr(runtime_context, "doppler_hz", 0.0)
        k_factor_db = getattr(ue_input, "link_k_factor_db", 0.0)
        if k_factor_db is None:
            k_factor_db = 0.0
        angle_spread_deg = getattr(ue_input, "link_angle_spread_deg", 35.0)
        if angle_spread_deg is None:
            angle_spread_deg = 35.0
        spatial_correlation_level = getattr(ue_input, "link_spatial_correlation_level", 0.5)
        if spatial_correlation_level is None:
            spatial_correlation_level = 0.5
        doppler_hz = float(doppler_hz)
        k_factor_db = float(k_factor_db)
        angle_spread_deg = float(angle_spread_deg)
        spatial_correlation_level = float(spatial_correlation_level)
        slot_phase = (runtime_context.slot_id + ue_input.serving_beam_id + fast_fading_seed) % 7
        centered_variation_db = float(slot_phase) - 3.0

        traffic_bias_db = 0.3 if ue_input.traffic_type == "control" else 0.0
        doppler_bias_db = min(1.5, max(0.0, doppler_hz / 120.0))
        k_factor_damping = max(0.3, 1.0 - min(0.7, max(0.0, k_factor_db) / 20.0))
        angle_spread_scale = 0.75 + min(0.35, max(0.0, angle_spread_deg) / 180.0)
        spatial_corr_scale = max(
            0.55,
            1.0 - 0.35 * max(0.0, min(1.0, spatial_correlation_level)),
        )

        return (
            0.8 * centered_variation_db * k_factor_damping * angle_spread_scale * spatial_corr_scale
            + traffic_bias_db
            + doppler_bias_db
        )

    def compute_effective_sinr(
        self,
        ue_input: PhyUeInput,
        runtime_context: PhyRuntimeContext,
    ) -> float:
        """
        Computes one slot-level effective SINR estimate for one UE before
        beamforming and link adaptation decisions.
        INPUT
        ue_input: The PHY UE input object.
        runtime_context: The PHY runtime context object.
        OUTPUT
        One effective SINR value in dB.
        EXAMPLE OUTPUT
        12.5
        """
        large_scale_loss_db = self._compute_large_scale_loss_db(ue_input, runtime_context)
        small_scale_variation_db = self._compute_small_scale_variation_db(
            ue_input=ue_input,
            runtime_context=runtime_context,
        )

        baseline_radio_quality_db = ue_input.radio_quality_db

        effective_sinr_db = (
            baseline_radio_quality_db
            + small_scale_variation_db
            - large_scale_loss_db
        )

        return effective_sinr_db
