"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: harq_state.py
Description: This file defines the HARQ-related runtime state used by the
simulator. The HARQ state stores retransmission status, ACK/NACK result,
and pending retransmission bytes for one UE. The terminology follows
3GPP NR HARQ process / RV / NDI style MAC-PHY interaction.
"""

from dataclasses import dataclass, field


@dataclass
class HarqLogicalChannelState:
    """
    Stores retransmission bytes that belong to one logical channel.
    """

    logical_channel_id: str
    drb_id: int | None = None
    lcg_id: int | None = None
    pending_bytes: int = 0
    priority: int = 0
    packet_delay_budget_ms: float = 0.0
    slice_id: str = ""
    traffic_type: str = ""
    process_id: int | None = None
    redundancy_version: int = 0
    ndi: int = 0
    retransmission_round: int = 0


@dataclass
class HarqProcessState:
    """
    Stores one HARQ process context tracked across retransmissions.
    """

    process_id: int
    is_busy: bool = False
    ndi: int = 0
    tb_id: str | None = None
    redundancy_version: int = 0
    retransmission_round: int = 0
    pending_bytes: int = 0
    last_tx_bytes: int = 0
    last_delivered_bytes: int = 0
    is_waiting_ack: bool = False
    rv_sequence: list[int] = field(default_factory=lambda: [0, 2, 3, 1])
    retx_deadline_slot: int | None = None
    timeout_slot: int | None = None
    flush_reason: str | None = None
    service_type: str | None = None
    max_retransmission_rounds: int = 4
    reserved_at_slot: int | None = None
    last_tx_slot: int | None = None
    pending_logical_channels: list[HarqLogicalChannelState] = field(default_factory=list)
    tx_contract: object | None = None
    mcs_index: int | None = None
    mcs_table_id: str | None = None


@dataclass
class HarqState:
    """
    Stores the HARQ-related runtime state of one UE.
    INPUT
    process_id: The HARQ process identifier.
    is_waiting_ack: Whether the current process is waiting for ACK/NACK.
    last_decision: The latest HARQ decision, such as ACK or NACK.
    retransmission_count: The number of retransmissions already used.
    pending_retx_bytes: The number of pending retransmission bytes.
    last_tx_bytes: The transport bytes transmitted in the latest attempt.
    OUTPUT
    One HARQ state object.
    EXAMPLE OUTPUT
    HarqState(
        process_id=0,
        is_waiting_ack=False,
        last_decision="IDLE",
        retransmission_count=0,
        pending_retx_bytes=0,
        last_tx_bytes=0
    )
    """
    process_id: int = 0
    is_waiting_ack: bool = False
    last_decision: str = "IDLE"
    retransmission_count: int = 0
    pending_retx_bytes: int = 0
    last_tx_bytes: int = 0
    last_tx_was_retx: bool = False
    last_delivered_bytes: int = 0
    redundancy_version: int = 0
    ndi: int = 0
    last_harq_process_id: int | None = None
    last_retransmission_round: int = 0
    process_pool_size: int = 4
    next_new_tx_process_id: int = 0
    ndi_by_process_id: dict[int, int] = field(default_factory=dict)
    processes: list[HarqProcessState] = field(default_factory=list)
    pending_retx_logical_channels: list[HarqLogicalChannelState] = field(default_factory=list)
    harq_flush_count: int = 0
    last_flush_reason: str | None = None
    last_timeout_slot: int | None = None
    empty_process_cleanup_count: int = 0
    events: list[dict] = field(default_factory=list)
