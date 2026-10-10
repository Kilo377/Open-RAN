"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: logical_channel_state.py
Description: This file defines the per-UE logical-channel / DRB state used
by the simulator. It enables one UE to carry multiple queue entities so
the scheduler can first allocate a UE grant and then multiplex bytes among
logical channels within that UE.
"""

from dataclasses import dataclass


@dataclass
class LogicalChannelState:
    """
    Stores one logical-channel queue attached to one UE.
    """

    logical_channel_id: str
    drb_id: int = 1
    lcg_id: int = 0
    qos_flow_id: int | None = None
    queue_bytes: int = 0
    priority: int = 1
    packet_delay_budget_ms: float = 100.0
    slice_id: str = ""
    traffic_type: str = ""
    prioritized_bit_rate_bps: float | None = None
    bucket_size_duration_ms: float = 10.0
    pbr_credit_bytes: float = 0.0
    last_credit_update_slot: int | None = None
    max_queue_bytes: int | None = None
    scheduled_bytes: int = 0
    delivered_bytes: int = 0
    prioritized_scheduled_bytes: int = 0
    last_scheduled_slot: int | None = None
    last_delivered_slot: int | None = None
