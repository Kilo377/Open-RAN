"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: resource_grid_state.py
Description: This file defines the explicit slot-level OFDM resource-grid
state used by the simulator. It captures the total slot grid size and
the per-UE allocation rectangles actually consumed by the PHY layer.
Within the simulator this grid corresponds to a downlink NR/PDSCH-style
resource grid that is scheduled by the MAC layer over DL-SCH traffic.
"""

from dataclasses import dataclass, field


@dataclass
class ResourceAllocationState:
    """
    Stores one UE allocation rectangle on the slot resource grid.
    """

    ue_id: str
    start_prb: int = 0
    allocated_prbs: int = 0
    start_symbol: int = 0
    allocated_symbols: int = 0
    resource_block_indices: list[int] = field(default_factory=list)
    allocated_subcarrier_indices: list[int] = field(default_factory=list)
    allocated_symbol_indices: list[int] = field(default_factory=list)
    interference_group_id: int | None = None
    spatial_group_id: int | None = None
    layer_count: int = 1
    resource_elements: int = 0
    overlap_fraction: float = 0.0
    interference_penalty_db: float = 0.0


@dataclass
class ResourceGridState:
    """
    Stores the explicit OFDM slot grid summary visible to the simulator.
    """

    total_prbs: int = 0
    total_symbols: int = 0
    subcarriers_per_prb: int = 12
    n_prbs_total: int = 0
    n_subcarriers_per_prb: int = 12
    n_ofdm_symbols_per_slot: int = 0
    fft_size: int = 0
    usable_subcarrier_mask: list[int] = field(default_factory=list)
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    subcarrier_spacing_hz: float = 30e3
    cyclic_prefix_type: str = "normal"
    nr_frequency_range: str = "FR1"
    nr_numerology_index: int = 1
    downlink_transport_channel: str = "DL-SCH"
    downlink_physical_channel: str = "PDSCH"
    reference_signal_label: str = "PDSCH DM-RS"
    total_resource_elements: int = 0
    allocated_resource_elements: int = 0
    utilization: float = 0.0
    overlap_fraction_mean: float = 0.0
    interference_penalty_db_mean: float = 0.0
    scheduled_ue_count: int = 0
    scheduled_layer_count_total: int = 0
    mu_mimo_overlap_pair_count: int = 0
    mu_mimo_group_count: int = 0
    mu_mimo_ue_count: int = 0
    adjacent_interference_pair_count: int = 0
    interference_groups: list[list[str]] = field(default_factory=list)
    spatial_groups: list[list[str]] = field(default_factory=list)
    prb_symbol_occupancy_map: list[list[int]] = field(default_factory=list)
    ue_resource_slices: dict[str, ResourceAllocationState] = field(default_factory=dict)
    allocations: list[ResourceAllocationState] = field(default_factory=list)
