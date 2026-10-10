"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: connection_state.py
Description: This file defines the explicit gNB-UE connection state used by
the simulator. The connection state captures whether a UE is currently
attached to the serving cell, whether it remains schedulable, and which
beam / CQI / HARQ context is active for that UE.
"""

from dataclasses import dataclass, field

from state.candidate_cell_state import CandidateCellState


@dataclass
class ConnectionState:
    """
    Stores the explicit connection state of one UE relative to one serving
    cell.
    """

    ue_id: str
    serving_cell_id: str
    connection_state: str = "connected"
    c_rnti: int = 0
    last_scheduled_slot: int | None = None
    last_ack_slot: int | None = None
    serving_beam_id: int = 0
    rank_indicator: int = 1
    pmi: int = 0
    cqi: int = 0
    timing_advance: int | None = None
    ul_bsr_seen: bool = False
    active_harq_processes: list[int] = field(default_factory=list)
    candidate_cells: list[CandidateCellState] = field(default_factory=list)
    preferred_target_cell_id: str | None = None
    handover_pending_target_cell_id: str | None = None
    handover_started_slot: int | None = None
    last_handover_slot: int | None = None
    handover_count: int = 0
    handover_cost_db: float = 0.0
    last_handover_cost_db: float = 0.0
    handover_interruption_slots: int = 0
    interruption_remaining_slots: int = 0
    steering_hysteresis_db: float = 0.0
    measurement_event_trigger: str = "a3_offset"
    measurement_event_offset_db: float = 0.0
    last_measurement_event_slot: int | None = None
    last_measurement_event_type: str | None = None
    last_measurement_event_margin_db: float | None = None
    last_trigger_reason: str | None = None
    traffic_steering_bias_db: float = 0.0
    consecutive_nack_count: int = 0
    out_of_sync_reason: str | None = None
