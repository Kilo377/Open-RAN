"""state/measurement_state.py

Defines the controller-visible measurement state.

This is intentionally separate from the internal truth in GlobalState. The
measurement values can include reporting delay, windowing, and smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UeMeasurement:
    ue_id: str
    queue_bytes: int = 0
    throughput_mbps: float = 0.0
    sinr_db: float = 0.0
    bler: float = 0.0
    is_scheduled: bool = False

    # Optional measurement extras.
    ack: bool = False
    tb_size_bytes: int = 0

    # Measurement quality / provenance.
    produced_at_slot: int = 0
    delivered_at_slot: int | None = None
    age_slots: int = 0
    queue_age_slots: int = 0
    queue_delay_slots_applied: int = 0
    queue_produced_at_slot: int = 0
    sinr_window_len: int = 1
    sinr_window_slots: int = 1
    sinr_effective_sample_count: int = 1
    sinr_noise_std_db_applied: float = 0.0
    bler_window_tbs: int = 1
    bler_window_sample_count: int = 0
    bler_sample_count: int = 0
    throughput_window_slots: int = 0
    throughput_granularity_slots: int = 1
    last_measurement_update_slot: int = 0


@dataclass
class MeasurementState:
    """Stores the latest produced measurement view for controllers."""

    produced_at_slot: int = 0
    delivered_at_slot: int | None = None
    age_slots: int = 0
    last_measurement_update_slot: int = 0

    queue_delay_slots: int = 0
    sinr_window_slots: int = 1
    sinr_noise_std_db_applied: float = 0.0
    bler_window_tbs: int = 1
    throughput_granularity_slots: int = 1

    ue_measurements: dict[str, UeMeasurement] = field(default_factory=dict)
