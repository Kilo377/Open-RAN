"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: beamforming.py
Description: This file defines simplified beamforming helper logic used by
the custom PHY adapter. The current implementation converts configured
codebook and beam profile selections into one beamforming gain value.
"""

import math

from models.phy_input import PhyUeInput
from models.phy_runtime_context import PhyRuntimeContext


class SimpleBeamformingModel:
    """
    Implements a simplified beamforming gain model.
    INPUT
    codebook_map: The configured codebook mapping dictionary.
    beam_profile_map: The configured beam profile mapping dictionary.
    OUTPUT
    A reusable beamforming model object.
    EXAMPLE OUTPUT
    beamforming_model = SimpleBeamformingModel(codebook_map, beam_profile_map)
    beam_gain_db = beamforming_model.compute_beam_gain(ue_input, runtime_context)
    """

    def __init__(
        self,
        codebook_map: dict,
        beam_profile_map: dict,
    ) -> None:
        """
        Initializes the beamforming model.
        INPUT
        codebook_map: The configured codebook mapping dictionary.
        beam_profile_map: The configured beam profile mapping dictionary.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.codebook_map = {...}
        """
        self.codebook_map = codebook_map
        self.beam_profile_map = beam_profile_map

    def compute_beam_gain(
        self,
        ue_input: PhyUeInput,
        runtime_context: PhyRuntimeContext,
    ) -> float:
        """
        Computes one beamforming gain estimate for one UE based on the
        active codebook, beam profile, and serving beam selection.
        INPUT
        ue_input: The PHY UE input object.
        runtime_context: The PHY runtime context object.
        OUTPUT
        One beamforming gain value in dB.
        EXAMPLE OUTPUT
        2.8
        """
        codebook_id = int(getattr(ue_input, "precoder_id", runtime_context.codebook_id) or runtime_context.codebook_id)
        beam_profile_id = int(runtime_context.beam_profile_id)
        codebook_entry = self.codebook_map.get(codebook_id, {})
        beam_profile_entry = self.beam_profile_map.get(beam_profile_id, {})

        codebook_base_gain_db = float(codebook_entry.get("base_gain_db", 0.0))
        beam_gain_map = codebook_entry.get("beams", {})
        serving_beam_gain_db = float(beam_gain_map.get(ue_input.serving_beam_id, 0.0))
        profile_gain_db = float(beam_profile_entry.get("profile_gain_db", 0.0))
        num_tx_ant = max(1, int(getattr(ue_input, "num_tx_ant", 1) or 1))
        num_rx_ant = max(1, int(getattr(ue_input, "num_rx_ant", 1) or 1))
        rank_indicator = max(1, int(getattr(ue_input, "rank_indicator", 1) or 1))
        layer_count = max(1, int(getattr(ue_input, "layer_count", rank_indicator) or rank_indicator))
        array_gain_db = 10.0 * math.log10(float(max(1, min(num_tx_ant, num_rx_ant))))
        layer_penalty_db = 10.0 * math.log10(float(max(1, layer_count))) - 10.0 * math.log10(float(rank_indicator))
        explicit_beam_gain_db = getattr(ue_input, "beamforming_gain_db", None)
        urllc_bias_db = 0.3 if ue_input.slice_id.lower() == "urllc" else 0.0

        if explicit_beam_gain_db is not None:
            return float(explicit_beam_gain_db) + array_gain_db - max(0.0, layer_penalty_db) + urllc_bias_db

        return (
            codebook_base_gain_db
            + serving_beam_gain_db
            + profile_gain_db
            + array_gain_db
            - max(0.0, layer_penalty_db)
            + urllc_bias_db
        )
