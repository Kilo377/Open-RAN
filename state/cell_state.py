"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: cell_state.py
Description: This file defines the cell-level runtime state used by the
simulator. It stores the current transmission power, codebook selection,
and beam profile information of the serving cell.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CellState:
    """
    Stores the runtime state of one serving cell.
    """
    cell_id: str
    tx_power_dbm: float = 30.0
    nominal_tx_power_dbm: float = 30.0
    current_codebook_id: int = 0
    current_beam_profile_id: int = 0

    # Capability-limited control surface state.
    beam_candidate_preference: str = "balanced"
    power_cap_dbm: Optional[float] = None
    cell_sleep_mode: str = "active"
    traffic_steering_preference: str = "balanced"
    mobility_bias_db: float = 0.0
    handover_margin_db: float = 3.0
    handover_delay_slots: int = 2
    handover_cost_db: float = 0.0
    handover_interruption_slots: int = 0
    steering_hysteresis_db: float = 0.0
    measurement_event_trigger: str = "a3_offset"
    measurement_event_offset_db: float = 0.0
    managed_cell_ids: list[str] = field(default_factory=list)
    neighbor_cell_positions: dict[str, dict[str, float]] = field(default_factory=dict)
    cell_radio_profiles: dict[str, dict[str, object]] = field(default_factory=dict)
