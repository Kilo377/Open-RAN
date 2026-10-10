"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: logical_channel_group_state.py
Description: This file defines the per-UE logical-channel-group state used
by the simulator. The state groups multiple logical channels by LCG so
controllers and experiments can observe bearer aggregation instead of only
individual queues. The naming follows 3GPP NR MAC concepts such as LCG and
BSR-style queue exposure.
"""

from dataclasses import dataclass, field


@dataclass
class LogicalChannelGroupState:
    """
    Stores one logical-channel group aggregation for one UE.
    """

    lcg_id: int
    logical_channel_ids: list[str] = field(default_factory=list)
    drb_ids: list[int] = field(default_factory=list)
    queue_bytes: int = 0
    prioritized_queue_bytes: int = 0
    pbr_credit_bytes: float = 0.0
    scheduled_bytes: int = 0
    prioritized_scheduled_bytes: int = 0
    delivered_bytes: int = 0
    bsr_reported_bytes: int = 0
    bsr_pending_bytes: int = 0
    bsr_age_slots: int | None = None
    last_bsr_report_slot: int | None = None
    last_bsr_trigger_slot: int | None = None
    bsr_triggered: bool = False
    bsr_trigger_reason: str | None = None
    bsr_periodic_timer_slots: int = 5
