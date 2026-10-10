"""
Per-cell runtime container for multi-cell-ready simulation mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from state.beam_state import BeamState
from state.cell_state import CellState
from state.channel_state import ChannelState
from state.resource_grid_state import ResourceGridState
from state.scheduler_state import SchedulerState


@dataclass
class ManagedCellRuntimeState:
    cell_id: str
    cell: CellState
    scheduler: SchedulerState
    channel: ChannelState
    beam: BeamState
    resource_grid: ResourceGridState
    radio_profile: dict[str, Any] = field(default_factory=dict)
    is_primary: bool = False
    active_ue_ids: list[str] = field(default_factory=list)
    scheduled_ue_ids: list[str] = field(default_factory=list)
    last_phy_result: object | None = None
    last_shadow_phy_result: object | None = None
