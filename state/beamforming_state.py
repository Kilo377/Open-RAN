"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: beamforming_state.py
Description: This file defines the explicit per-UE beamforming state used
by the simulator. It stores precoder / combiner identifiers, rank and
layer configuration, and the latest beamforming gain estimate.
"""

from dataclasses import dataclass, field


@dataclass
class BeamformingState:
    """
    Stores the beamforming runtime state for one UE.
    """

    ue_id: str
    codebook_id: int = 0
    beam_profile_id: int = 0
    serving_beam_id: int = 0
    precoder_id: int = 0
    combiner_id: int = 0
    rank: int = 1
    layer_count: int = 1
    beam_gain_db: float = 0.0
    beam_direction_deg: float = 0.0
    precoding_matrix_label: str = "cb0-r1-b0"
    precoder_matrix: list[list[float]] = field(default_factory=list)
    combiner_vector: list[float] = field(default_factory=list)
    precoder_match_score: float = 1.0
    post_eq_sinr_db: float | None = None
    spatial_group_id: int | None = None
