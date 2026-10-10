"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: channel_state.py
Description: This file defines the channel-related state used by the
simulator. The channel state stores long-timescale and runtime channel
parameters that can be consumed by PHY adapters.
"""

from dataclasses import dataclass


@dataclass
class ChannelState:
    """
    Stores the channel-related runtime state.
    INPUT
    pathloss_db: The large-scale pathloss value in dB.
    shadowing_db: The shadowing value in dB.
    channel_profile: The channel profile name.
    doppler_hz: The Doppler spread in Hz.
    OUTPUT
    One channel state object.
    EXAMPLE OUTPUT
    ChannelState(
        pathloss_db=100.0,
        shadowing_db=3.0,
        channel_profile="tdl_c",
        doppler_hz=30.0
    )
    """
    pathloss_db: float = 130.0
    shadowing_db: float = 6.0
    channel_profile: str = "tdl_c"
    doppler_hz: float = 30.0
