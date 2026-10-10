"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: global_state.py
Description: This file defines the global simulator state. The global
state stores the current slot index, simulation time, cell state,
scheduler state, channel state, beam state, UE states, and state history.
"""

from dataclasses import asdict, dataclass, field

from state.antenna_state import AntennaState
from state.beam_state import BeamState
from state.beamforming_state import BeamformingState
from state.cell_state import CellState
from state.channel_state import ChannelState
from state.connection_state import ConnectionState
from state.control_state import ControlState
from state.link_state import LinkState
from state.managed_cell_runtime_state import ManagedCellRuntimeState
from state.measurement_state import MeasurementState
from state.resource_grid_state import ResourceGridState
from state.scheduler_state import SchedulerState
from state.traffic_state import TrafficState
from state.ue_state import UeState


@dataclass
class GlobalState:
    """
    Stores the full runtime state of the simulator.
    INPUT
    slot_id: The current slot index.
    sim_time_ms: The current simulation time in milliseconds.
    cell: The cell-level runtime state.
    scheduler: The scheduler runtime state.
    channel: The channel runtime state.
    beam: The beamforming runtime state.
    ues: The list of UE runtime states.
    history: The list of saved state snapshots.
    OUTPUT
    One global simulation state object.
    EXAMPLE OUTPUT
    GlobalState(
        slot_id=0,
        sim_time_ms=0.0,
        cell=CellState(...),
        scheduler=SchedulerState(...),
        channel=ChannelState(...),
        beam=BeamState(...),
        ues=[UeState(...)]
    )
    """
    slot_id: int
    sim_time_ms: float
    cell: CellState
    scheduler: SchedulerState
    channel: ChannelState
    beam: BeamState
    antenna: AntennaState
    beamforming: dict[str, BeamformingState]
    ues: list[UeState]
    slot_duration_ms: float
    control: ControlState
    runtime_mode: str = "single_active_cell"
    traffic: TrafficState = field(default_factory=TrafficState)
    measurement: MeasurementState | None = None
    connections: dict[str, ConnectionState] = field(default_factory=dict)
    links: dict[str, LinkState] = field(default_factory=dict)
    resource_grid: ResourceGridState = field(default_factory=ResourceGridState)
    managed_cells: dict[str, ManagedCellRuntimeState] = field(default_factory=dict)
    cluster_scheduled_ue_ids: list[str] = field(default_factory=list)
    cluster_grants: list[object] = field(default_factory=list)
    cluster_last_demands: list[object] = field(default_factory=list)
    cluster_last_stage_trace: object | None = None
    cluster_last_decision: object | None = None
    last_phy_result: object | None = None
    last_shadow_phy_result: object | None = None
    last_shadow_phy_label: str | None = None
    avg_sinr_db: float | None = None
    avg_bler: float | None = None
    history: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)

    def record_snapshot(self) -> None:
        """
        Records one full snapshot of the current simulator state.
        INPUT
        No explicit input.
        OUTPUT
        None. One snapshot dictionary is appended to history.
        EXAMPLE OUTPUT
        self.history.append({...})
        """
        snapshot = {
            "slot_id": self.slot_id,
            "sim_time_ms": self.sim_time_ms,
            "cell": asdict(self.cell),
            "scheduler": asdict(self.scheduler),
            "channel": asdict(self.channel),
            "beam": asdict(self.beam),
            "antenna": asdict(self.antenna),
            "beamforming": {
                str(ue_id): asdict(beamforming)
                for ue_id, beamforming in self.beamforming.items()
            },
            "control": asdict(self.control),
            "traffic": asdict(self.traffic),
            "measurement": asdict(self.measurement) if self.measurement is not None else None,
            "connections": {
                str(ue_id): asdict(connection)
                for ue_id, connection in self.connections.items()
            },
            "links": {
                str(ue_id): asdict(link)
                for ue_id, link in self.links.items()
            },
            "resource_grid": asdict(self.resource_grid),
            "runtime_mode": str(self.runtime_mode),
            "managed_cells": {
                str(cell_id): asdict(runtime_state)
                for cell_id, runtime_state in self.managed_cells.items()
            },
            "ues": [asdict(ue) for ue in self.ues],
        }
        self.history.append(snapshot)
