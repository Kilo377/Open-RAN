"""state/traffic_state.py

Stores traffic arrival bookkeeping for the current slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrafficState:
    produced_at_slot: int = 0
    arrival_bytes_total: int = 0
    arrival_bytes_by_ue: dict[str, int] = field(default_factory=dict)
    arrival_bytes_by_logical_channel: dict[str, int] = field(default_factory=dict)
    arrival_bytes_by_logical_channel_group: dict[str, int] = field(default_factory=dict)
