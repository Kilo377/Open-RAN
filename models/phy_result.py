"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: phy_result.py
Description: This file defines PHY result data models used by the simulator
to store UE-level and slot-level PHY execution outputs.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PhyUeResult:
    """
    L1 Design:
    This data class stores one UE-level PHY execution result for one slot.
    """
    ue_id: str

    # Scheduling / delivery semantics
    is_scheduled: bool = False
    allocated_prbs: int = 0
    allocated_symbols: int = 0
    start_prb: int = 0
    start_symbol: int = 0
    is_retx: bool = False
    requested_bytes: int = 0
    grant_bytes_budget: int = 0
    mcs_table_id: str = "qam64"
    tbs_selection_method: str = "resource_elements"
    dmrs_pattern_id: str | None = None
    harq_process_id: int | None = None
    harq_redundancy_version: int | None = None
    harq_ndi: int | None = None
    harq_retransmission_round: int = 0
    cqi_validity_age_slots: int = 0
    ri_validity_age_slots: int = 0
    pmi_validity_age_slots: int = 0
    channel_quality_confidence: float | None = None
    link_adaptation_reason: str | None = None
    scheduled_bytes: int = 0
    delivered_bytes: int = 0
    ack: bool = False
    reference_tb_ack: bool | None = None

    # Core PHY metrics
    sinr_db: float = 0.0
    effective_sinr_db: float = 0.0
    pre_eq_sinr_db: float = 0.0
    post_eq_sinr_db: float = 0.0
    bler: float = 0.0
    ber: float = 0.0

    # Link adaptation / MIMO hints (optional for early-stage PHY)
    mcs_index: int = 0
    modulation_order: int = 0
    code_rate: float = 0.0
    num_layers: int = 1

    # Transport block & throughput
    tb_size_bytes: int = 0
    transport_block_size_bytes: int = 0
    throughput_mbps: float = 0.0
    queue_bytes: int = 0

    # CSI-like fields (optional)
    cqi: int = 0
    rank: int = 1
    pmi: int = 0
    ri: int = 1

    # Beamforming and HARQ extensions (optional)
    beam_gain_db: float = 0.0
    channel_rank_limit: int = 1
    precoder_match_score: float = 1.0
    spatial_mode: str = "su_siso"
    spatial_group_id: int | None = None
    spatial_group_size: int = 1
    spatial_compatibility_score: float = 0.0
    spatial_multiplexing_gain_db: float = 0.0
    harq_retransmission_count: int = 0
    resource_grid_re_count: int = 0
    resource_grid_overlap_fraction: float = 0.0
    interference_penalty_db: float = 0.0
    intra_cell_interference_penalty_db: float = 0.0
    inter_cell_interference_dbm: float = -120.0
    inter_cell_interference_penalty_db: float = 0.0
    inter_cell_interference_overlap_fraction: float = 0.0
    inter_cell_frequency_coupling_fraction: float = 0.0
    inter_cell_interference_source_count: int = 0
    phy_mode: str = "system_level_approx"
    waveform_backend: str | None = None
    effective_data_re_count: int = 0
    dmrs_overhead_fraction: float = 0.0


@dataclass
class PhyResult:
    """
    L1 Design:
    This data class stores one slot-level PHY execution result, including
    all UE PHY results and aggregate PHY metrics.
    """
    slot_id: int
    cell_id: str = "cell_0"
    ue_results: List[PhyUeResult] = field(default_factory=list)
    avg_sinr_db: float = 0.0
    avg_bler: float = 0.0
    total_tb_size_bytes: int = 0


