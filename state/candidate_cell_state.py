"""
Candidate-cell state used by lightweight mobility and traffic steering.
"""

from dataclasses import dataclass


@dataclass
class CandidateCellState:
    cell_id: str
    cell_type: str = "macro"
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 25.0
    distance_2d_m: float = 0.0
    distance_3d_m: float = 0.0
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    subcarrier_spacing_hz: float = 30e3
    rf_band: str = "sub6"
    adjacent_channel_leakage_ratio_db: float = 45.0
    adjacent_channel_selectivity_db: float = 33.0
    out_of_band_isolation_db: float = 42.0
    tx_power_dbm: float = 30.0
    connected_ue_count: int = 0
    scheduled_ue_count: int = 0
    queued_bytes: int = 0
    estimated_load: float = 0.0
    load_penalty_db: float = 0.0
    estimated_interference_db: float = -120.0
    interference_penalty_db: float = 0.0
    signal_margin_db: float = -120.0
    handover_cost_db: float = 0.0
    steering_hysteresis_db: float = 0.0
    hysteresis_adjusted_score_db: float = -100.0
    measurement_event_trigger: str = "a3_offset"
    measurement_event_offset_db: float = 0.0
    measurement_event_met: bool = False
    trigger_margin_db: float = -100.0
    steering_score_db: float = -100.0
    pathloss_db: float = 130.0
    shadowing_db: float = 0.0
    rsrp_dbm: float = -100.0
    quality_score_db: float = -100.0
    candidate_beam_id: int = 0
    is_serving: bool = False
    is_target: bool = False
