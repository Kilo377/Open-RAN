"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: phy_runtime_context.py
Description: This file defines the PHY runtime context object used by the
simulator. The runtime context groups together cell, scheduler, channel,
and beamforming information needed by a PHY adapter at one slot.
"""

from dataclasses import dataclass


@dataclass
class PhyRuntimeContext:
    """
    Stores runtime context information required by the PHY adapter.
    INPUT
    slot_id: The current slot index.
    sim_time_ms: The current simulation time in milliseconds.
    cell_id: The serving cell identifier.
    scheduler_mode: The scheduler mode active in the current slot.
    tx_power_dbm: The transmit power in dBm.
    current_codebook_id: The active codebook identifier.
    current_beam_profile_id: The active beam profile identifier.
    pathloss_db: The large-scale pathloss value in dB.
    shadowing_db: The shadowing value in dB.
    doppler_hz: The Doppler spread in Hz.
    channel_profile: The channel profile name.
    codebook_id: The beam state codebook identifier.
    beam_profile_id: The beam state beam profile identifier.
    OUTPUT
    One PHY runtime context object.
    EXAMPLE OUTPUT
    PhyRuntimeContext(
        slot_id=1,
        sim_time_ms=1.0,
        cell_id="cell_0",
        scheduler_mode="latency_aware",
        tx_power_dbm=30.0,
        current_codebook_id=1,
        current_beam_profile_id=1,
        pathloss_db=100.0,
        shadowing_db=3.0,
        doppler_hz=30.0,
        channel_profile="tdl_c",
        codebook_id=1,
        beam_profile_id=1
    )
    """
    slot_id: int
    sim_time_ms: float
    slot_duration_ms: float
    cell_id: str
    scheduler_mode: str
    tx_power_dbm: float
    current_codebook_id: int
    current_beam_profile_id: int
    carrier_frequency_hz: float
    bandwidth_hz: float
    subcarrier_spacing_hz: float
    cyclic_prefix_type: str
    num_ofdm_symbols_per_slot: int
    n_prbs_total: int
    pathloss_db: float
    shadowing_db: float
    channel_profile: str
    codebook_id: int
    beam_profile_id: int
    doppler_hz: float = 0.0
