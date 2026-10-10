"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: beam_state.py
Description: This file defines the beamforming-related state used by the
simulator. The beam state stores codebook, beam profile, selected beam,
beam gain, and beam configuration mappings that can be consumed by PHY
adapters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BeamState:
    """
    Stores the beamforming-related runtime state.
    INPUT
    codebook_id: The active codebook identifier.
    beam_profile_id: The active beam profile identifier.
    selected_beam_id: The selected beam identifier.
    beam_gain_db: The current beamforming gain in dB.
    codebook_map: The configured codebook mapping dictionary.
    beam_profile_map: The configured beam profile mapping dictionary.
    OUTPUT
    One beam state object.
    EXAMPLE OUTPUT
    BeamState(
        codebook_id=0,
        beam_profile_id=0,
        selected_beam_id=0,
        beam_gain_db=0.0
    )
    """
    codebook_id: int = 0
    beam_profile_id: int = 0
    selected_beam_id: int = 0
    beam_gain_db: float = 0.0
    codebook_map: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    beam_profile_map: Dict[int, Dict[str, Any]] = field(default_factory=dict)