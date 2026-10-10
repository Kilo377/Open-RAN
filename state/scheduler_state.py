"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: scheduler_state.py
Description: This file defines the scheduler runtime state used by the
simulator. It stores the current scheduler mode, the grant list, a
slot-level scheduling decision snapshot, and a round-robin pointer for
future MAC-layer extensions.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from models.scheduling import SchedulingDecision, SchedulingStageTrace, UeGrant, UeSchedulingDemand


@dataclass
class SchedulerState:
    """
    Stores the runtime state of the scheduler.
    """
    scheduler_mode: str = "round_robin"
    packing_strategy: str = "frequency_first"
    enable_spatial_multiplexing: bool = False
    max_mu_users_per_group: int = 2
    min_mu_beam_separation_deg: float = 45.0
    max_total_spatial_layers: int | None = None
    scheduled_ue_ids: List[str] = field(default_factory=list)
    grants: List[UeGrant] = field(default_factory=list)
    last_decision: SchedulingDecision | None = None
    last_demands: List[UeSchedulingDemand] = field(default_factory=list)
    last_stage_trace: SchedulingStageTrace | None = None
    rr_pointer: int = 0

    # Capability-limited control surface state.
    scheduler_policy_hint: str = "balanced"
    latency_bias: Optional[float] = None
    slice_weight_adjustment: dict[str, float] = field(default_factory=dict)

    # One-shot command context (cleared after scheduling).
    priority_boost_by_ue: dict[str, float] = field(default_factory=dict)

    # Scheduler runtime history for constraint consumption.
    last_served_slot_by_ue: dict[str, int] = field(default_factory=dict)
