"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: antenna_state.py
Description: This file defines the antenna-related runtime state used by
the simulator. The antenna state captures gNB-side array capability and
per-UE receive-antenna capability so that beamforming and rank decisions
can be represented explicitly. In 3GPP-style terms, this state carries the
NR carrier/numerology parameters consumed by the downlink PHY/MAC chain.
"""

from dataclasses import dataclass, field


@dataclass
class AntennaState:
    """
    Stores antenna-array capability for the current simulation.
    """

    gnb_num_tx_ant: int = 4
    ue_num_rx_ant_by_ue: dict[str, int] = field(default_factory=dict)
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    subcarrier_spacing_hz: float = 30e3
    cyclic_prefix_type: str = "normal"
    num_ofdm_symbols_per_slot: int = 14
    nr_frequency_range: str = "FR1"
    nr_numerology_index: int = 1
    gnb_panel_layout: str = "2x2"
    ue_panel_layout_by_ue: dict[str, str] = field(default_factory=dict)
    polarization: str = "single"
