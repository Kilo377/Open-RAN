"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: metrics.py
Description: This file defines helper functions for building slot-level
observations from the current global state. The Observation is built from
the controller-visible measurement view when available, and it includes
both scheduled-only (TX-only) and all-UE aggregates to avoid selection
bias.
"""

from typing import Any

from core.runtime_views import (
    get_cluster_resource_grid_summary,
    get_cluster_scheduler_demands,
    get_cluster_scheduler_stage_trace,
    get_cluster_scheduled_ue_ids,
)
from models.observation import (
    CandidateCellObservation,
    CellObservation,
    LogicalChannelGroupObservation,
    LogicalChannelObservation,
    Observation,
    UeObservation,
)
from state.global_state import GlobalState


def build_observation(state: GlobalState) -> Observation:
    """
    Builds one observation object from the current global simulation state.
    INPUT
    state: The global simulation state.
    OUTPUT
    One Observation object summarizing the current slot state.
    EXAMPLE OUTPUT
    Observation(slot_id=5, avg_sinr_db=12.3, avg_bler=0.14, ...)
    """
    scheduled_ue_ids = get_cluster_scheduled_ue_ids(state)
    scheduled_ue_id_set = set(scheduled_ue_ids)
    connection_map = getattr(state, "connections", {})
    if not isinstance(connection_map, dict):
        connection_map = {}
    link_map = getattr(state, "links", {})
    if not isinstance(link_map, dict):
        link_map = {}
    beamforming_map = getattr(state, "beamforming", {})
    if not isinstance(beamforming_map, dict):
        beamforming_map = {}
    antenna_state = getattr(state, "antenna", None)
    resource_grid_state = getattr(state, "resource_grid", None)
    cluster_resource_grid_summary = get_cluster_resource_grid_summary(state)
    scheduler_demands = get_cluster_scheduler_demands(state)
    scheduler_stage_trace = get_cluster_scheduler_stage_trace(state)
    scheduler_demand_by_ue_id = {
        str(getattr(demand, "ue_id", "")): demand
        for demand in scheduler_demands
        if str(getattr(demand, "ue_id", "")) != ""
    }
    managed_cells = getattr(state, "managed_cells", {})
    if not isinstance(managed_cells, dict):
        managed_cells = {}
    resource_grid_allocations = {}
    if resource_grid_state is not None:
        for allocation in list(getattr(resource_grid_state, "allocations", [])):
            ue_id = str(getattr(allocation, "ue_id", ""))
            if ue_id != "":
                resource_grid_allocations[ue_id] = allocation

    measurement = getattr(state, "measurement", None)

    def _mean_or_none(values: list[float]) -> float | None:
        if len(values) == 0:
            return None
        return float(sum(values) / float(len(values)))

    def _logical_channel_group_value(
        logical_channel_group: Any,
        field_name: str,
        default: Any,
    ) -> Any:
        if isinstance(logical_channel_group, dict):
            return logical_channel_group.get(field_name, default)
        return getattr(logical_channel_group, field_name, default)

    scheduled_ues = [ue for ue in state.ues if ue.ue_id in scheduled_ue_id_set]

    if len(scheduled_ues) > 0:
        if measurement is not None and hasattr(measurement, "ue_measurements"):
            m = measurement.ue_measurements
            scheduled_sinrs = [float(m[str(ue.ue_id)].sinr_db) for ue in scheduled_ues if str(ue.ue_id) in m]
            scheduled_blers = [float(m[str(ue.ue_id)].bler) for ue in scheduled_ues if str(ue.ue_id) in m]
            avg_sinr_db_scheduled = (
                sum(scheduled_sinrs) / float(len(scheduled_sinrs))
                if len(scheduled_sinrs) > 0 else None
            )
            avg_bler_scheduled = (
                sum(scheduled_blers) / float(len(scheduled_blers))
                if len(scheduled_blers) > 0 else None
            )
        else:
            avg_sinr_db_scheduled = sum(ue.sinr_db for ue in scheduled_ues) / float(len(scheduled_ues))
            avg_bler_scheduled = sum(ue.bler for ue in scheduled_ues) / float(len(scheduled_ues))
    else:
        avg_sinr_db_scheduled = None
        avg_bler_scheduled = None

    if len(state.ues) > 0:
        if measurement is not None and hasattr(measurement, "ue_measurements"):
            m = measurement.ue_measurements
            all_sinrs = [float(m[str(ue.ue_id)].sinr_db) for ue in state.ues if str(ue.ue_id) in m]
            avg_sinr_db_all = sum(all_sinrs) / float(len(all_sinrs)) if len(all_sinrs) > 0 else None
            avg_radio_quality_db_all = sum(float(getattr(ue, "radio_quality_db", 0.0)) for ue in state.ues) / float(len(state.ues))
        else:
            avg_sinr_db_all = sum(float(getattr(ue, "sinr_db", 0.0)) for ue in state.ues) / float(len(state.ues))
            avg_radio_quality_db_all = sum(float(getattr(ue, "radio_quality_db", 0.0)) for ue in state.ues) / float(len(state.ues))
    else:
        avg_sinr_db_all = None
        avg_radio_quality_db_all = None

    # Keep backward-compatible fields but make their scope explicit:
    # - avg_sinr_db: all-UE average (less selection bias)
    # - avg_bler: scheduled-only average (TX-only metric)
    avg_sinr_db = avg_sinr_db_all
    avg_bler = avg_bler_scheduled

    if measurement is not None and hasattr(measurement, "ue_measurements"):
        m = measurement.ue_measurements
        total_queue_bytes = sum(int(m[str(ue.ue_id)].queue_bytes) for ue in state.ues if str(ue.ue_id) in m)
        total_throughput_mbps = sum(float(m[str(ue.ue_id)].throughput_mbps) for ue in state.ues if str(ue.ue_id) in m)
    else:
        total_queue_bytes = sum(ue.queue_bytes for ue in state.ues)
        total_throughput_mbps = sum(ue.throughput_mbps for ue in state.ues)

    if len(connection_map) > 0:
        connected_ue_count = int(
            sum(
                1
                for connection in connection_map.values()
                if str(getattr(connection, "connection_state", "connected")) == "connected"
            )
        )
        out_of_sync_ue_count = int(
            sum(
                1
                for connection in connection_map.values()
                if str(getattr(connection, "connection_state", "connected")) == "out_of_sync"
            )
        )
        handover_pending_ue_count = int(
            sum(
                1
                for connection in connection_map.values()
                if str(getattr(connection, "connection_state", "connected")) == "handover_pending"
            )
        )
        handover_completed_ue_count = int(
            sum(1 for connection in connection_map.values() if getattr(connection, "last_handover_slot", None) is not None)
        )
        handover_interruption_ue_count = int(
            sum(
                1
                for connection in connection_map.values()
                if str(getattr(connection, "connection_state", "connected")) == "handover_interruption"
            )
        )
    else:
        connected_ue_count = int(len(state.ues))
        out_of_sync_ue_count = 0
        handover_pending_ue_count = 0
        handover_completed_ue_count = 0
        handover_interruption_ue_count = 0

    ue_observations = []
    for ue in state.ues:
        ue_id = str(getattr(ue, "ue_id", "unknown_ue"))
        connection = connection_map.get(ue_id)
        link = link_map.get(ue_id)
        beamforming = beamforming_map.get(ue_id)
        serving_cell_id = (
            str(getattr(connection, "serving_cell_id", getattr(state.cell, "cell_id", "")))
            if connection is not None
            else str(getattr(state.cell, "cell_id", ""))
        )
        candidate_cell_observations = [
            CandidateCellObservation(
                cell_id=str(getattr(candidate, "cell_id", "")),
                cell_type=str(getattr(candidate, "cell_type", "macro")),
                distance_2d_m=float(getattr(candidate, "distance_2d_m", 0.0)),
                carrier_frequency_hz=float(getattr(candidate, "carrier_frequency_hz", 3.5e9)),
                bandwidth_hz=float(getattr(candidate, "bandwidth_hz", 20e6)),
                subcarrier_spacing_hz=float(getattr(candidate, "subcarrier_spacing_hz", 30e3)),
                rf_band=str(getattr(candidate, "rf_band", "sub6")),
                adjacent_channel_leakage_ratio_db=float(
                    getattr(candidate, "adjacent_channel_leakage_ratio_db", 45.0)
                ),
                adjacent_channel_selectivity_db=float(
                    getattr(candidate, "adjacent_channel_selectivity_db", 33.0)
                ),
                out_of_band_isolation_db=float(
                    getattr(candidate, "out_of_band_isolation_db", 42.0)
                ),
                tx_power_dbm=float(getattr(candidate, "tx_power_dbm", 30.0)),
                connected_ue_count=int(getattr(candidate, "connected_ue_count", 0)),
                scheduled_ue_count=int(getattr(candidate, "scheduled_ue_count", 0)),
                queued_bytes=int(getattr(candidate, "queued_bytes", 0)),
                estimated_load=float(getattr(candidate, "estimated_load", 0.0)),
                load_penalty_db=float(getattr(candidate, "load_penalty_db", 0.0)),
                estimated_interference_db=float(getattr(candidate, "estimated_interference_db", -120.0)),
                interference_penalty_db=float(getattr(candidate, "interference_penalty_db", 0.0)),
                signal_margin_db=float(getattr(candidate, "signal_margin_db", -120.0)),
                handover_cost_db=float(getattr(candidate, "handover_cost_db", 0.0)),
                steering_hysteresis_db=float(getattr(candidate, "steering_hysteresis_db", 0.0)),
                hysteresis_adjusted_score_db=float(
                    getattr(candidate, "hysteresis_adjusted_score_db", -100.0)
                ),
                measurement_event_trigger=str(
                    getattr(candidate, "measurement_event_trigger", "a3_offset")
                ),
                measurement_event_offset_db=float(
                    getattr(candidate, "measurement_event_offset_db", 0.0)
                ),
                measurement_event_met=bool(getattr(candidate, "measurement_event_met", False)),
                trigger_margin_db=float(getattr(candidate, "trigger_margin_db", -100.0)),
                steering_score_db=float(getattr(candidate, "steering_score_db", -100.0)),
                pathloss_db=float(getattr(candidate, "pathloss_db", 130.0)),
                rsrp_dbm=float(getattr(candidate, "rsrp_dbm", -100.0)),
                quality_score_db=float(getattr(candidate, "quality_score_db", -100.0)),
                candidate_beam_id=int(getattr(candidate, "candidate_beam_id", 0)),
                is_serving=bool(getattr(candidate, "is_serving", False)),
                is_target=bool(getattr(candidate, "is_target", False)),
            )
            for candidate in list(getattr(connection, "candidate_cells", []) or [])
        ] if connection is not None else []
        connection_state = (
            str(getattr(connection, "connection_state", "connected"))
            if connection is not None
            else "connected"
        )
        c_rnti = getattr(connection, "c_rnti", None) if connection is not None else None
        rank_indicator = getattr(connection, "rank_indicator", getattr(ue, "rank", None))
        pmi = getattr(connection, "pmi", getattr(ue, "pmi", None))
        cqi = getattr(connection, "cqi", getattr(ue, "cqi", None))
        num_tx_ant = getattr(antenna_state, "gnb_num_tx_ant", None) if antenna_state is not None else None
        ue_num_rx_ant_map = getattr(antenna_state, "ue_num_rx_ant_by_ue", {}) if antenna_state is not None else {}
        num_rx_ant = ue_num_rx_ant_map.get(ue_id) if isinstance(ue_num_rx_ant_map, dict) else None
        beamforming_rank = getattr(beamforming, "rank", rank_indicator) if beamforming is not None else rank_indicator
        beamforming_layer_count = getattr(beamforming, "layer_count", getattr(ue, "ri", None))
        precoder_id = getattr(beamforming, "precoder_id", getattr(state.cell, "current_codebook_id", None))
        combiner_id = getattr(beamforming, "combiner_id", pmi)
        precoding_matrix_label = getattr(beamforming, "precoding_matrix_label", None) if beamforming is not None else None
        post_eq_sinr_db = (
            getattr(link, "last_post_eq_sinr_db", getattr(beamforming, "post_eq_sinr_db", getattr(ue, "post_eq_sinr_db", None)))
            if beamforming is not None or connection is not None
            else getattr(ue, "post_eq_sinr_db", None)
        )
        precoder_match_score = getattr(beamforming, "precoder_match_score", getattr(ue, "precoder_match_score", None)) if beamforming is not None else getattr(ue, "precoder_match_score", None)
        beam_direction_deg = getattr(beamforming, "beam_direction_deg", None) if beamforming is not None else None
        beamforming_gain_db = getattr(beamforming, "beam_gain_db", None) if beamforming is not None else None
        scheduler_demand = scheduler_demand_by_ue_id.get(ue_id)
        grant_target_mcs = getattr(ue, "grant_target_mcs", None)
        grant_mcs_table_id = getattr(ue, "grant_mcs_table_id", None)
        grant_tbs_selection_method = getattr(ue, "grant_tbs_selection_method", None)
        grant_dmrs_pattern_id = getattr(ue, "grant_dmrs_pattern_id", None)
        grant_cqi_validity_age_slots = getattr(ue, "grant_cqi_validity_age_slots", None)
        grant_ri_validity_age_slots = getattr(ue, "grant_ri_validity_age_slots", None)
        grant_pmi_validity_age_slots = getattr(ue, "grant_pmi_validity_age_slots", None)
        grant_channel_quality_confidence = getattr(ue, "grant_channel_quality_confidence", None)
        grant_link_adaptation_reason = getattr(ue, "grant_link_adaptation_reason", None)
        logical_channels = list(getattr(ue, "logical_channels", []) or [])
        logical_channel_groups = list(getattr(ue, "logical_channel_groups", []) or [])
        if len(logical_channel_groups) == 0 and len(logical_channels) > 0:
            groups_by_lcg_id = {}
            for logical_channel in logical_channels:
                logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
                if logical_channel_id == "":
                    continue
                lcg_id = int(getattr(logical_channel, "lcg_id", 0))
                group = groups_by_lcg_id.setdefault(
                    lcg_id,
                    {
                        "lcg_id": lcg_id,
                        "logical_channel_ids": [],
                        "drb_ids": [],
                        "queue_bytes": 0,
                        "prioritized_queue_bytes": 0,
                        "pbr_credit_bytes": 0.0,
                        "scheduled_bytes": 0,
                        "prioritized_scheduled_bytes": 0,
                        "delivered_bytes": 0,
                    },
                )
                group["logical_channel_ids"].append(logical_channel_id)
                drb_id = getattr(logical_channel, "drb_id", None)
                if drb_id is not None:
                    group["drb_ids"].append(int(drb_id))
                queue_bytes_value = int(getattr(logical_channel, "queue_bytes", 0))
                group["queue_bytes"] += queue_bytes_value
                if getattr(logical_channel, "prioritized_bit_rate_bps", None) is not None:
                    group["prioritized_queue_bytes"] += queue_bytes_value
                group["pbr_credit_bytes"] += float(
                    getattr(logical_channel, "pbr_credit_bytes", 0.0) or 0.0
                )
                group["scheduled_bytes"] += int(getattr(logical_channel, "scheduled_bytes", 0))
                group["prioritized_scheduled_bytes"] += int(
                    getattr(logical_channel, "prioritized_scheduled_bytes", 0)
                )
                group["delivered_bytes"] += int(getattr(logical_channel, "delivered_bytes", 0))
            logical_channel_groups = list(groups_by_lcg_id.values())
        dominant_logical_channel_id = None
        if len(logical_channels) > 0:
            dominant_logical_channel = max(
                logical_channels,
                key=lambda logical_channel: int(getattr(logical_channel, "queue_bytes", 0)),
            )
            dominant_logical_channel_id = str(
                getattr(dominant_logical_channel, "logical_channel_id", "")
            )
        dominant_lcg_id = None
        if len(logical_channel_groups) > 0:
            dominant_logical_channel_group = max(
                logical_channel_groups,
                key=lambda logical_channel_group: int(
                    _logical_channel_group_value(logical_channel_group, "queue_bytes", 0)
                ),
            )
            dominant_lcg_id = int(
                _logical_channel_group_value(dominant_logical_channel_group, "lcg_id", 0)
            )
        scheduled_logical_channel_ids = list(
            getattr(ue, "scheduled_logical_channel_ids", []) or []
        )
        scheduled_lcg_ids = list(getattr(ue, "scheduled_lcg_ids", []) or [])
        logical_channel_observations = [
            LogicalChannelObservation(
                logical_channel_id=str(getattr(logical_channel, "logical_channel_id", "")),
                drb_id=(
                    int(getattr(logical_channel, "drb_id", 0))
                    if getattr(logical_channel, "drb_id", None) is not None
                    else None
                ),
                lcg_id=(
                    int(getattr(logical_channel, "lcg_id", 0))
                    if getattr(logical_channel, "lcg_id", None) is not None
                    else None
                ),
                queue_bytes=int(getattr(logical_channel, "queue_bytes", 0)),
                priority=int(getattr(logical_channel, "priority", 0)),
                packet_delay_budget_ms=float(
                    getattr(logical_channel, "packet_delay_budget_ms", 0.0)
                ),
                slice_id=str(getattr(logical_channel, "slice_id", "")),
                traffic_type=str(getattr(logical_channel, "traffic_type", "")),
                prioritized_bit_rate_bps=(
                    float(getattr(logical_channel, "prioritized_bit_rate_bps", 0.0))
                    if getattr(logical_channel, "prioritized_bit_rate_bps", None) is not None
                    else None
                ),
                pbr_credit_bytes=float(getattr(logical_channel, "pbr_credit_bytes", 0.0) or 0.0),
                is_scheduled=str(getattr(logical_channel, "logical_channel_id", "")) in scheduled_logical_channel_ids,
                scheduled_bytes=int(getattr(logical_channel, "scheduled_bytes", 0)),
                prioritized_scheduled_bytes=int(
                    getattr(logical_channel, "prioritized_scheduled_bytes", 0)
                ),
                delivered_bytes=int(getattr(logical_channel, "delivered_bytes", 0)),
            )
            for logical_channel in logical_channels
        ]
        logical_channel_group_observations = [
            LogicalChannelGroupObservation(
                lcg_id=int(
                    _logical_channel_group_value(logical_channel_group, "lcg_id", 0)
                ),
                logical_channel_ids=sorted(
                    set(
                        list(
                            _logical_channel_group_value(
                                logical_channel_group,
                                "logical_channel_ids",
                                [],
                            )
                        )
                    )
                ),
                drb_ids=sorted(
                    set(
                        int(item)
                        for item in list(
                            _logical_channel_group_value(logical_channel_group, "drb_ids", [])
                        )
                    )
                ),
                queue_bytes=int(
                    _logical_channel_group_value(logical_channel_group, "queue_bytes", 0)
                ),
                prioritized_queue_bytes=int(
                    _logical_channel_group_value(
                        logical_channel_group,
                        "prioritized_queue_bytes",
                        0,
                    )
                ),
                pbr_credit_bytes=float(
                    _logical_channel_group_value(logical_channel_group, "pbr_credit_bytes", 0.0)
                ),
                is_scheduled=int(
                    _logical_channel_group_value(logical_channel_group, "lcg_id", 0)
                ) in scheduled_lcg_ids,
                scheduled_bytes=int(
                    _logical_channel_group_value(logical_channel_group, "scheduled_bytes", 0)
                ),
                prioritized_scheduled_bytes=int(
                    _logical_channel_group_value(
                        logical_channel_group,
                        "prioritized_scheduled_bytes",
                        0,
                    )
                ),
                delivered_bytes=int(
                    _logical_channel_group_value(logical_channel_group, "delivered_bytes", 0)
                ),
                bsr_reported_bytes=int(
                    _logical_channel_group_value(logical_channel_group, "bsr_reported_bytes", 0)
                ),
                bsr_pending_bytes=int(
                    _logical_channel_group_value(logical_channel_group, "bsr_pending_bytes", 0)
                ),
                bsr_age_slots=(
                    int(_logical_channel_group_value(logical_channel_group, "bsr_age_slots", 0))
                    if _logical_channel_group_value(logical_channel_group, "bsr_age_slots", None)
                    is not None
                    else None
                ),
                last_bsr_report_slot=(
                    int(_logical_channel_group_value(logical_channel_group, "last_bsr_report_slot", 0))
                    if _logical_channel_group_value(
                        logical_channel_group,
                        "last_bsr_report_slot",
                        None,
                    ) is not None
                    else None
                ),
                last_bsr_trigger_slot=(
                    int(_logical_channel_group_value(logical_channel_group, "last_bsr_trigger_slot", 0))
                    if _logical_channel_group_value(
                        logical_channel_group,
                        "last_bsr_trigger_slot",
                        None,
                    ) is not None
                    else None
                ),
                bsr_triggered=bool(
                    _logical_channel_group_value(logical_channel_group, "bsr_triggered", False)
                ),
                bsr_trigger_reason=(
                    str(_logical_channel_group_value(logical_channel_group, "bsr_trigger_reason", ""))
                    if _logical_channel_group_value(
                        logical_channel_group,
                        "bsr_trigger_reason",
                        None,
                    ) is not None
                    else None
                ),
            )
            for logical_channel_group in logical_channel_groups
        ]
        resource_grid_allocation = resource_grid_allocations.get(ue_id)
        grant_start_prb = getattr(resource_grid_allocation, "start_prb", getattr(ue, "start_prb", None))
        grant_start_symbol = getattr(resource_grid_allocation, "start_symbol", getattr(ue, "start_symbol", None))
        grant_allocated_prbs = getattr(resource_grid_allocation, "allocated_prbs", getattr(ue, "allocated_prbs", None))
        grant_allocated_symbols = getattr(resource_grid_allocation, "allocated_symbols", getattr(ue, "allocated_symbols", None))
        grant_resource_block_indices = list(
            getattr(resource_grid_allocation, "resource_block_indices", getattr(ue, "resource_block_indices", [])) or []
        )
        grant_allocated_symbol_indices = list(
            getattr(resource_grid_allocation, "allocated_symbol_indices", getattr(ue, "allocated_symbol_indices", [])) or []
        )
        resource_grid_re_count = getattr(resource_grid_allocation, "resource_elements", getattr(ue, "resource_grid_re_count", None))
        resource_grid_overlap_fraction = getattr(
            resource_grid_allocation,
            "overlap_fraction",
            getattr(ue, "resource_grid_overlap_fraction", None),
        )
        interference_penalty_db = getattr(
            resource_grid_allocation,
            "interference_penalty_db",
            getattr(ue, "interference_penalty_db", None),
        )
        intra_cell_interference_penalty_db = getattr(
            ue,
            "intra_cell_interference_penalty_db",
            None,
        )
        inter_cell_interference_dbm = getattr(ue, "inter_cell_interference_dbm", None)
        inter_cell_interference_penalty_db = getattr(
            ue,
            "inter_cell_interference_penalty_db",
            None,
        )
        inter_cell_interference_overlap_fraction = getattr(
            ue,
            "inter_cell_interference_overlap_fraction",
            None,
        )
        inter_cell_frequency_coupling_fraction = getattr(
            ue,
            "inter_cell_frequency_coupling_fraction",
            None,
        )
        inter_cell_interference_source_count = getattr(
            ue,
            "inter_cell_interference_source_count",
            None,
        )
        interference_group_id = getattr(resource_grid_allocation, "interference_group_id", None)
        spatial_group_id = getattr(resource_grid_allocation, "spatial_group_id", getattr(ue, "spatial_group_id", None))
        spatial_group_size = getattr(ue, "spatial_group_size", None)
        spatial_compatibility_score = getattr(ue, "spatial_compatibility_score", None)
        spatial_multiplexing_gain_db = getattr(ue, "spatial_multiplexing_gain_db", None)
        spatial_mode = getattr(ue, "spatial_mode", None)
        active_harq_process_count = (
            int(len(getattr(connection, "active_harq_processes", [])))
            if connection is not None
            else 0
        )
        if measurement is not None and hasattr(measurement, "ue_measurements") and ue_id in measurement.ue_measurements:
            mm = measurement.ue_measurements[ue_id]
            queue_bytes = int(mm.queue_bytes)
            throughput_mbps = float(mm.throughput_mbps)
            sinr_db = float(mm.sinr_db)
            if getattr(mm, 'sinr_kind', '') == 'rt_full_band_reference':
                post_eq_sinr_db = mm.data_sinr_db if mm.data_sinr_db is not None else mm.reference_sinr_db
                beamforming_gain_db = mm.reference_beam_gain_db
            bler = float(mm.bler)
            queue_age_slots = int(getattr(mm, "queue_age_slots", 0))
            queue_delay_slots_applied = int(getattr(mm, "queue_delay_slots_applied", queue_age_slots))
            queue_produced_at_slot = int(getattr(mm, "queue_produced_at_slot", state.slot_id - queue_age_slots))
            sinr_window_len = int(getattr(mm, "sinr_window_len", 1))
            sinr_window_slots = int(getattr(mm, "sinr_window_slots", sinr_window_len))
            sinr_effective_sample_count = int(getattr(mm, "sinr_effective_sample_count", 1))
            sinr_noise_std_db_applied = float(getattr(mm, "sinr_noise_std_db_applied", 0.0))
            bler_window_tbs = int(getattr(mm, "bler_window_tbs", 1))
            bler_window_sample_count = int(getattr(mm, "bler_window_sample_count", 0))
            bler_sample_count = int(getattr(mm, "bler_sample_count", bler_window_sample_count))
            throughput_window_slots = int(getattr(mm, "throughput_window_slots", 0))
            throughput_granularity_slots = int(
                getattr(mm, "throughput_granularity_slots", throughput_window_slots)
            )
            last_measurement_update_slot = int(getattr(mm, "last_measurement_update_slot", state.slot_id))
            measurement_produced_at_slot = int(getattr(mm, "produced_at_slot", state.slot_id))
            measurement_delivered_at_slot = getattr(mm, "delivered_at_slot", None)
            measurement_age_slots = int(getattr(mm, "age_slots", 0))
        else:
            queue_bytes = int(getattr(ue, "queue_bytes", 0))
            throughput_mbps = float(getattr(ue, "throughput_mbps", 0.0))
            sinr_db = float(getattr(ue, "sinr_db", 0.0))
            bler = float(getattr(ue, "bler", 0.0))
            queue_age_slots = None
            queue_delay_slots_applied = None
            queue_produced_at_slot = None
            sinr_window_len = None
            sinr_window_slots = None
            sinr_effective_sample_count = None
            sinr_noise_std_db_applied = None
            bler_window_tbs = None
            bler_window_sample_count = None
            bler_sample_count = None
            throughput_window_slots = None
            throughput_granularity_slots = None
            last_measurement_update_slot = None
            measurement_produced_at_slot = None
            measurement_delivered_at_slot = None
            measurement_age_slots = None

        ue_observations.append(
            UeObservation(
                ue_id=ue_id,
                queue_bytes=queue_bytes,
                throughput_mbps=throughput_mbps,
                sinr_db=sinr_db,
                bler=bler,
                serving_beam_id=int(getattr(ue, "serving_beam_id", 0)),
                is_scheduled=ue_id in scheduled_ue_id_set,
                slice_id=str(getattr(ue, "slice_id", "")),
                traffic_type=str(getattr(ue, "traffic_type", "")),
                latency_budget_ms=float(getattr(ue, "latency_budget_ms", 0.0)),
                priority=int(getattr(ue, "priority", 0)),
                demand_true_backlog_bytes=(
                    int(getattr(scheduler_demand, "true_backlog_bytes", getattr(scheduler_demand, "backlog_bytes", 0)))
                    if scheduler_demand is not None
                    else None
                ),
                demand_observed_backlog_bytes=(
                    int(getattr(scheduler_demand, "observed_backlog_bytes", getattr(scheduler_demand, "backlog_bytes", 0)))
                    if scheduler_demand is not None
                    else None
                ),
                demand_bytes=(
                    int(getattr(scheduler_demand, "demand_bytes", getattr(scheduler_demand, "backlog_bytes", 0)))
                    if scheduler_demand is not None
                    else None
                ),
                demand_hol_delay_ms=(
                    float(getattr(scheduler_demand, "hol_delay_ms", 0.0))
                    if scheduler_demand is not None
                    else None
                ),
                demand_qos_class=(
                    str(getattr(scheduler_demand, "qos_class", ""))
                    if scheduler_demand is not None
                    else None
                ),
                demand_five_qi=(
                    int(getattr(scheduler_demand, "five_qi", 0))
                    if scheduler_demand is not None and getattr(scheduler_demand, "five_qi", None) is not None
                    else None
                ),
                demand_bsr_like_bytes=(
                    int(getattr(scheduler_demand, "bsr_like_bytes", 0))
                    if scheduler_demand is not None
                    else None
                ),
                demand_bsr_reported_bytes=(
                    int(getattr(scheduler_demand, "bsr_reported_bytes", 0))
                    if scheduler_demand is not None
                    else None
                ),
                demand_bsr_pending_bytes=(
                    int(getattr(scheduler_demand, "bsr_pending_bytes", 0))
                    if scheduler_demand is not None
                    else None
                ),
                demand_harq_urgency_score=(
                    float(getattr(scheduler_demand, "harq_urgency_score", 0.0))
                    if scheduler_demand is not None
                    else None
                ),
                demand_channel_quality_confidence=(
                    float(getattr(scheduler_demand, "channel_quality_confidence", 1.0))
                    if scheduler_demand is not None
                    else None
                ),
                demand_reason=(
                    str(getattr(scheduler_demand, "demand_reason", ""))
                    if scheduler_demand is not None
                    else None
                ),
                logical_channel_count=int(len(logical_channels)),
                logical_channel_group_count=int(len(logical_channel_groups)),
                logical_channel_demand_count=(
                    int(len(list(getattr(scheduler_demand, "logical_channel_demands", []) or [])))
                    if scheduler_demand is not None
                    else None
                ),
                logical_channel_group_demand_count=(
                    int(len(list(getattr(scheduler_demand, "logical_channel_group_demands", []) or [])))
                    if scheduler_demand is not None
                    else None
                ),
                dominant_logical_channel_id=dominant_logical_channel_id,
                dominant_lcg_id=dominant_lcg_id,
                scheduled_logical_channel_ids=scheduled_logical_channel_ids,
                scheduled_drb_ids=list(getattr(ue, "scheduled_drb_ids", []) or []),
                scheduled_lcg_ids=scheduled_lcg_ids,
                logical_channel_observations=logical_channel_observations,
                logical_channel_group_observations=logical_channel_group_observations,
                serving_cell_id=serving_cell_id,
                candidate_cell_observations=candidate_cell_observations,
                connection_state=connection_state,
                c_rnti=(int(c_rnti) if c_rnti is not None else None),
                preferred_target_cell_id=(
                    str(getattr(connection, "preferred_target_cell_id", ""))
                    if connection is not None and getattr(connection, "preferred_target_cell_id", None) is not None
                    else None
                ),
                handover_pending_target_cell_id=(
                    str(getattr(connection, "handover_pending_target_cell_id", ""))
                    if connection is not None and getattr(connection, "handover_pending_target_cell_id", None) is not None
                    else None
                ),
                handover_started_slot=(
                    int(getattr(connection, "handover_started_slot", 0))
                    if connection is not None and getattr(connection, "handover_started_slot", None) is not None
                    else None
                ),
                last_handover_slot=(
                    int(getattr(connection, "last_handover_slot", 0))
                    if connection is not None and getattr(connection, "last_handover_slot", None) is not None
                    else None
                ),
                handover_count=(
                    int(getattr(connection, "handover_count", 0))
                    if connection is not None
                    else None
                ),
                handover_cost_db=(
                    float(getattr(connection, "handover_cost_db", 0.0))
                    if connection is not None
                    else None
                ),
                last_handover_cost_db=(
                    float(getattr(connection, "last_handover_cost_db", 0.0))
                    if connection is not None
                    else None
                ),
                handover_interruption_slots=(
                    int(getattr(connection, "handover_interruption_slots", 0))
                    if connection is not None
                    else None
                ),
                interruption_remaining_slots=(
                    int(getattr(connection, "interruption_remaining_slots", 0))
                    if connection is not None
                    else None
                ),
                steering_hysteresis_db=(
                    float(getattr(connection, "steering_hysteresis_db", 0.0))
                    if connection is not None
                    else None
                ),
                measurement_event_trigger=(
                    str(getattr(connection, "measurement_event_trigger", "a3_offset"))
                    if connection is not None
                    else None
                ),
                measurement_event_offset_db=(
                    float(getattr(connection, "measurement_event_offset_db", 0.0))
                    if connection is not None
                    else None
                ),
                last_measurement_event_slot=(
                    int(getattr(connection, "last_measurement_event_slot", 0))
                    if connection is not None and getattr(connection, "last_measurement_event_slot", None) is not None
                    else None
                ),
                last_measurement_event_type=(
                    str(getattr(connection, "last_measurement_event_type", ""))
                    if connection is not None and getattr(connection, "last_measurement_event_type", None) is not None
                    else None
                ),
                last_measurement_event_margin_db=(
                    float(getattr(connection, "last_measurement_event_margin_db", 0.0))
                    if connection is not None and getattr(connection, "last_measurement_event_margin_db", None) is not None
                    else None
                ),
                last_trigger_reason=(
                    str(getattr(connection, "last_trigger_reason", ""))
                    if connection is not None and getattr(connection, "last_trigger_reason", None) is not None
                    else None
                ),
                traffic_steering_bias_db=(
                    float(getattr(connection, "traffic_steering_bias_db", 0.0))
                    if connection is not None
                    else None
                ),
                rank_indicator=(int(rank_indicator) if rank_indicator is not None else None),
                pmi=(int(pmi) if pmi is not None else None),
                cqi=(int(cqi) if cqi is not None else None),
                active_harq_process_count=int(active_harq_process_count),
                num_tx_ant=(int(num_tx_ant) if num_tx_ant is not None else None),
                num_rx_ant=(int(num_rx_ant) if num_rx_ant is not None else None),
                beamforming_rank=(int(beamforming_rank) if beamforming_rank is not None else None),
                beamforming_layer_count=(
                    int(beamforming_layer_count) if beamforming_layer_count is not None else None
                ),
                precoder_id=(int(precoder_id) if precoder_id is not None else None),
                combiner_id=(int(combiner_id) if combiner_id is not None else None),
                precoding_matrix_label=(str(precoding_matrix_label) if precoding_matrix_label is not None else None),
                post_eq_sinr_db=(float(post_eq_sinr_db) if post_eq_sinr_db is not None else None),
                hol_delay_ms=getattr(measurement.ue_measurements.get(ue_id), 'hol_delay_ms', None) if measurement is not None else None,
                rsrp_dbm=getattr(measurement.ue_measurements.get(ue_id), 'rsrp_dbm', None) if measurement is not None else None,
                reference_sinr_db=getattr(measurement.ue_measurements.get(ue_id), 'reference_sinr_db', None) if measurement is not None else None,
                data_sinr_db=getattr(measurement.ue_measurements.get(ue_id), 'data_sinr_db', None) if measurement is not None else None,
                sinr_kind=getattr(measurement.ue_measurements.get(ue_id), 'sinr_kind', 'legacy_empirical') if measurement is not None else 'legacy_empirical',
                post_eq_sinr_source=('data_measurement' if mm.data_sinr_db is not None else 'reference_hypothesis') if measurement is not None and ue_id in measurement.ue_measurements and getattr(mm,'sinr_kind','') == 'rt_full_band_reference' else 'legacy_truth',
                precoder_match_score=(
                    float(precoder_match_score) if precoder_match_score is not None else None
                ),
                beam_direction_deg=(
                    float(beam_direction_deg) if beam_direction_deg is not None else None
                ),
                beamforming_gain_db=(
                    float(beamforming_gain_db) if beamforming_gain_db is not None else None
                ),
                grant_target_mcs=(int(grant_target_mcs) if grant_target_mcs is not None else None),
                grant_mcs_table_id=(
                    str(grant_mcs_table_id) if grant_mcs_table_id is not None else None
                ),
                grant_tbs_selection_method=(
                    str(grant_tbs_selection_method)
                    if grant_tbs_selection_method is not None
                    else None
                ),
                grant_dmrs_pattern_id=(
                    str(grant_dmrs_pattern_id) if grant_dmrs_pattern_id is not None else None
                ),
                grant_cqi_validity_age_slots=(
                    int(grant_cqi_validity_age_slots)
                    if grant_cqi_validity_age_slots is not None
                    else None
                ),
                grant_ri_validity_age_slots=(
                    int(grant_ri_validity_age_slots)
                    if grant_ri_validity_age_slots is not None
                    else None
                ),
                grant_pmi_validity_age_slots=(
                    int(grant_pmi_validity_age_slots)
                    if grant_pmi_validity_age_slots is not None
                    else None
                ),
                grant_channel_quality_confidence=(
                    float(grant_channel_quality_confidence)
                    if grant_channel_quality_confidence is not None
                    else None
                ),
                grant_link_adaptation_reason=(
                    str(grant_link_adaptation_reason)
                    if grant_link_adaptation_reason is not None
                    else None
                ),
                grant_start_prb=(int(grant_start_prb) if grant_start_prb is not None else None),
                grant_start_symbol=(int(grant_start_symbol) if grant_start_symbol is not None else None),
                grant_allocated_prbs=(
                    int(grant_allocated_prbs) if grant_allocated_prbs is not None else None
                ),
                grant_allocated_symbols=(
                    int(grant_allocated_symbols) if grant_allocated_symbols is not None else None
                ),
                grant_resource_block_indices=grant_resource_block_indices,
                grant_allocated_symbol_indices=grant_allocated_symbol_indices,
                resource_grid_re_count=(
                    int(resource_grid_re_count) if resource_grid_re_count is not None else None
                ),
                resource_grid_overlap_fraction=(
                    float(resource_grid_overlap_fraction)
                    if resource_grid_overlap_fraction is not None
                    else None
                ),
                interference_penalty_db=(
                    float(interference_penalty_db) if interference_penalty_db is not None else None
                ),
                intra_cell_interference_penalty_db=(
                    float(intra_cell_interference_penalty_db)
                    if intra_cell_interference_penalty_db is not None
                    else None
                ),
                inter_cell_interference_dbm=(
                    float(inter_cell_interference_dbm)
                    if inter_cell_interference_dbm is not None
                    else None
                ),
                inter_cell_interference_penalty_db=(
                    float(inter_cell_interference_penalty_db)
                    if inter_cell_interference_penalty_db is not None
                    else None
                ),
                inter_cell_interference_overlap_fraction=(
                    float(inter_cell_interference_overlap_fraction)
                    if inter_cell_interference_overlap_fraction is not None
                    else None
                ),
                inter_cell_frequency_coupling_fraction=(
                    float(inter_cell_frequency_coupling_fraction)
                    if inter_cell_frequency_coupling_fraction is not None
                    else None
                ),
                inter_cell_interference_source_count=(
                    int(inter_cell_interference_source_count)
                    if inter_cell_interference_source_count is not None
                    else None
                ),
                interference_group_id=(
                    int(interference_group_id) if interference_group_id is not None else None
                ),
                spatial_group_id=(
                    int(spatial_group_id) if spatial_group_id is not None else None
                ),
                spatial_group_size=(
                    int(spatial_group_size) if spatial_group_size is not None else None
                ),
                spatial_compatibility_score=(
                    float(spatial_compatibility_score)
                    if spatial_compatibility_score is not None
                    else None
                ),
                spatial_multiplexing_gain_db=(
                    float(spatial_multiplexing_gain_db)
                    if spatial_multiplexing_gain_db is not None
                    else None
                ),
                spatial_mode=(str(spatial_mode) if spatial_mode is not None else None),
                queue_age_slots=queue_age_slots,
                sinr_window_len=sinr_window_len,
                sinr_effective_sample_count=sinr_effective_sample_count,
                bler_window_sample_count=bler_window_sample_count,
                throughput_window_slots=throughput_window_slots,
                last_measurement_update_slot=last_measurement_update_slot,
                measurement_produced_at_slot=measurement_produced_at_slot,
                measurement_delivered_at_slot=measurement_delivered_at_slot,
                measurement_age_slots=measurement_age_slots,
                queue_delay_slots_applied=queue_delay_slots_applied,
                queue_produced_at_slot=queue_produced_at_slot,
                sinr_window_slots=sinr_window_slots,
                sinr_noise_std_db_applied=sinr_noise_std_db_applied,
                bler_window_tbs=bler_window_tbs,
                bler_sample_count=bler_sample_count,
                throughput_granularity_slots=throughput_granularity_slots,
            )
        )

    measurement_queue_age_slots_mean = _mean_or_none(
        [
            float(item.queue_age_slots)
            for item in ue_observations
            if item.queue_age_slots is not None
        ]
    )
    measurement_sinr_effective_sample_count_mean = _mean_or_none(
        [
            float(item.sinr_effective_sample_count)
            for item in ue_observations
            if item.sinr_effective_sample_count is not None
        ]
    )
    measurement_bler_sample_count_mean = _mean_or_none(
        [
            float(item.bler_sample_count)
            for item in ue_observations
            if item.bler_sample_count is not None
        ]
    )
    measurement_throughput_sample_count_mean = _mean_or_none(
        [
            float(item.throughput_window_slots)
            for item in ue_observations
            if item.throughput_window_slots is not None
        ]
    )
    spatial_multiplexing_gain_db_mean = _mean_or_none(
        [
            float(item.spatial_multiplexing_gain_db)
            for item in ue_observations
            if item.spatial_multiplexing_gain_db is not None and item.is_scheduled
        ]
    )
    inter_cell_interference_penalty_db_mean = _mean_or_none(
        [
            float(item.inter_cell_interference_penalty_db)
            for item in ue_observations
            if item.inter_cell_interference_penalty_db is not None and item.is_scheduled
        ]
    )
    inter_cell_interference_source_count_mean = _mean_or_none(
        [
            float(item.inter_cell_interference_source_count)
            for item in ue_observations
            if item.inter_cell_interference_source_count is not None and item.is_scheduled
        ]
    )
    inter_cell_frequency_coupling_fraction_mean = _mean_or_none(
        [
            float(item.inter_cell_frequency_coupling_fraction)
            for item in ue_observations
            if item.inter_cell_frequency_coupling_fraction is not None and item.is_scheduled
        ]
    )

    cell_observations: list[CellObservation] = []
    if len(managed_cells) > 0:
        for managed_cell_id, runtime_state in sorted(managed_cells.items(), key=lambda item: str(item[0])):
            runtime_cell = getattr(runtime_state, "cell", None)
            runtime_scheduler = getattr(runtime_state, "scheduler", None)
            runtime_grid = getattr(runtime_state, "resource_grid", None)
            radio_profile = dict(getattr(runtime_state, "radio_profile", {}) or {})
            active_ue_ids = [
                str(ue_id)
                for ue_id in list(getattr(runtime_state, "active_ue_ids", []) or [])
            ]
            if len(active_ue_ids) == 0:
                active_ue_ids = [
                    str(ue.ue_id)
                    for ue in state.ues
                    if str(
                        getattr(
                            connection_map.get(str(ue.ue_id)),
                            "serving_cell_id",
                            getattr(state.cell, "cell_id", ""),
                        )
                    )
                    == str(managed_cell_id)
                ]
            scheduled_cell_ue_ids = [
                str(ue_id)
                for ue_id in list(getattr(runtime_state, "scheduled_ue_ids", []) or [])
            ]
            if len(scheduled_cell_ue_ids) == 0 and runtime_scheduler is not None:
                scheduled_cell_ue_ids = [
                    str(ue_id)
                    for ue_id in list(getattr(runtime_scheduler, "scheduled_ue_ids", []) or [])
                ]
            active_ue_id_set = set(active_ue_ids)
            total_queue_bytes_by_cell = int(
                sum(
                    int(getattr(ue, "queue_bytes", 0))
                    for ue in state.ues
                    if str(ue.ue_id) in active_ue_id_set
                )
            )
            scheduled_ue_observations = [
                item for item in ue_observations if str(item.ue_id) in set(scheduled_cell_ue_ids)
            ]
            cell_observations.append(
                CellObservation(
                    cell_id=str(managed_cell_id),
                    cell_type=str(radio_profile.get("cell_type", "macro")),
                    is_primary=bool(getattr(runtime_state, "is_primary", False)),
                    carrier_frequency_hz=float(
                        radio_profile.get(
                            "carrier_frequency_hz",
                            getattr(runtime_grid, "carrier_frequency_hz", getattr(antenna_state, "carrier_frequency_hz", 3.5e9)),
                        )
                    ),
                    bandwidth_hz=float(
                        radio_profile.get(
                            "bandwidth_hz",
                            getattr(runtime_grid, "bandwidth_hz", getattr(antenna_state, "bandwidth_hz", 20e6)),
                        )
                    ),
                    subcarrier_spacing_hz=float(
                        radio_profile.get(
                            "subcarrier_spacing_hz",
                            getattr(runtime_grid, "subcarrier_spacing_hz", getattr(antenna_state, "subcarrier_spacing_hz", 30e3)),
                        )
                    ),
                    rf_band=str(radio_profile.get("rf_band", "sub6")),
                    adjacent_channel_leakage_ratio_db=float(
                        radio_profile.get("adjacent_channel_leakage_ratio_db", 45.0)
                    ),
                    adjacent_channel_selectivity_db=float(
                        radio_profile.get("adjacent_channel_selectivity_db", 33.0)
                    ),
                    out_of_band_isolation_db=float(
                        radio_profile.get("out_of_band_isolation_db", 42.0)
                    ),
                    tx_power_dbm=float(
                        getattr(runtime_cell, "tx_power_dbm", getattr(state.cell, "tx_power_dbm", 30.0))
                    ),
                    current_codebook_id=int(
                        getattr(runtime_cell, "current_codebook_id", getattr(state.cell, "current_codebook_id", 0))
                    ),
                    current_beam_profile_id=int(
                        getattr(runtime_cell, "current_beam_profile_id", getattr(state.cell, "current_beam_profile_id", 0))
                    ),
                    scheduler_mode=str(
                        getattr(runtime_scheduler, "scheduler_mode", getattr(state.scheduler, "scheduler_mode", "round_robin"))
                    ),
                    scheduler_policy_hint=str(
                        getattr(
                            runtime_scheduler,
                            "scheduler_policy_hint",
                            getattr(state.scheduler, "scheduler_policy_hint", "balanced"),
                        )
                    ),
                    scheduler_packing_strategy=str(
                        getattr(runtime_scheduler, "packing_strategy", "frequency_first")
                    ),
                    latency_bias=(
                        float(getattr(runtime_scheduler, "latency_bias", 0.0))
                        if getattr(runtime_scheduler, "latency_bias", None) is not None
                        else None
                    ),
                    slice_weight_adjustment=dict(
                        getattr(runtime_scheduler, "slice_weight_adjustment", {}) or {}
                    ),
                    beam_candidate_preference=str(
                        getattr(
                            runtime_cell,
                            "beam_candidate_preference",
                            getattr(state.cell, "beam_candidate_preference", "balanced"),
                        )
                    ),
                    power_cap_dbm=(
                        float(getattr(runtime_cell, "power_cap_dbm", 0.0))
                        if getattr(runtime_cell, "power_cap_dbm", None) is not None
                        else None
                    ),
                    cell_sleep_mode=str(
                        getattr(runtime_cell, "cell_sleep_mode", "active")
                    ),
                    traffic_steering_preference=str(
                        getattr(
                            runtime_cell,
                            "traffic_steering_preference",
                            getattr(state.cell, "traffic_steering_preference", "balanced"),
                        )
                    ),
                    mobility_bias_db=(
                        float(getattr(runtime_cell, "mobility_bias_db", 0.0))
                        if getattr(runtime_cell, "mobility_bias_db", None) is not None
                        else None
                    ),
                    handover_margin_db=(
                        float(getattr(runtime_cell, "handover_margin_db", 0.0))
                        if getattr(runtime_cell, "handover_margin_db", None) is not None
                        else None
                    ),
                    handover_cost_db=(
                        float(getattr(runtime_cell, "handover_cost_db", 0.0))
                        if getattr(runtime_cell, "handover_cost_db", None) is not None
                        else None
                    ),
                    handover_interruption_slots=(
                        int(getattr(runtime_cell, "handover_interruption_slots", 0))
                        if getattr(runtime_cell, "handover_interruption_slots", None) is not None
                        else None
                    ),
                    steering_hysteresis_db=(
                        float(getattr(runtime_cell, "steering_hysteresis_db", 0.0))
                        if getattr(runtime_cell, "steering_hysteresis_db", None) is not None
                        else None
                    ),
                    measurement_event_trigger=str(
                        getattr(runtime_cell, "measurement_event_trigger", "a3_offset")
                    ) if runtime_cell is not None else None,
                    measurement_event_offset_db=(
                        float(getattr(runtime_cell, "measurement_event_offset_db", 0.0))
                        if getattr(runtime_cell, "measurement_event_offset_db", None) is not None
                        else None
                    ),
                    connected_ue_count=len(active_ue_ids),
                    scheduled_ue_count=len(scheduled_cell_ue_ids),
                    total_queue_bytes=total_queue_bytes_by_cell,
                    resource_grid_total_re=int(getattr(runtime_grid, "total_resource_elements", 0)),
                    resource_grid_allocated_re=int(getattr(runtime_grid, "allocated_resource_elements", 0)),
                    resource_grid_utilization=float(getattr(runtime_grid, "utilization", 0.0)),
                    interference_penalty_db_mean=_mean_or_none(
                        [
                            float(item.interference_penalty_db)
                            for item in scheduled_ue_observations
                            if item.interference_penalty_db is not None
                        ]
                    ),
                    inter_cell_interference_penalty_db_mean=_mean_or_none(
                        [
                            float(item.inter_cell_interference_penalty_db)
                            for item in scheduled_ue_observations
                            if item.inter_cell_interference_penalty_db is not None
                        ]
                    ),
                    inter_cell_frequency_coupling_fraction_mean=_mean_or_none(
                        [
                            float(item.inter_cell_frequency_coupling_fraction)
                            for item in scheduled_ue_observations
                            if item.inter_cell_frequency_coupling_fraction is not None
                        ]
                    ),
                    inter_cell_interference_source_count_mean=_mean_or_none(
                        [
                            float(item.inter_cell_interference_source_count)
                            for item in scheduled_ue_observations
                            if item.inter_cell_interference_source_count is not None
                        ]
                    ),
                    active_ue_ids=active_ue_ids,
                    scheduled_ue_ids=scheduled_cell_ue_ids,
                )
            )
    else:
        cell_observations.append(
            CellObservation(
                cell_id=str(getattr(state.cell, "cell_id", "cell_0")),
                cell_type="macro",
                is_primary=True,
                carrier_frequency_hz=float(getattr(antenna_state, "carrier_frequency_hz", 3.5e9)),
                bandwidth_hz=float(getattr(antenna_state, "bandwidth_hz", 20e6)),
                subcarrier_spacing_hz=float(getattr(antenna_state, "subcarrier_spacing_hz", 30e3)),
                rf_band="sub6",
                adjacent_channel_leakage_ratio_db=45.0,
                adjacent_channel_selectivity_db=33.0,
                out_of_band_isolation_db=42.0,
                tx_power_dbm=float(getattr(state.cell, "tx_power_dbm", 30.0)),
                current_codebook_id=int(getattr(state.cell, "current_codebook_id", 0)),
                current_beam_profile_id=int(getattr(state.cell, "current_beam_profile_id", 0)),
                scheduler_mode=str(getattr(state.scheduler, "scheduler_mode", "round_robin")),
                scheduler_policy_hint=str(
                    getattr(state.scheduler, "scheduler_policy_hint", "balanced")
                ),
                scheduler_packing_strategy=str(
                    getattr(getattr(state, "scheduler", None), "packing_strategy", "frequency_first")
                ),
                latency_bias=(
                    float(getattr(state.scheduler, "latency_bias", 0.0))
                    if getattr(state.scheduler, "latency_bias", None) is not None
                    else None
                ),
                slice_weight_adjustment=dict(
                    getattr(state.scheduler, "slice_weight_adjustment", {}) or {}
                ),
                beam_candidate_preference=str(
                    getattr(state.cell, "beam_candidate_preference", "balanced")
                ),
                power_cap_dbm=(
                    float(getattr(state.cell, "power_cap_dbm", 0.0))
                    if getattr(state.cell, "power_cap_dbm", None) is not None
                    else None
                ),
                cell_sleep_mode=str(getattr(state.cell, "cell_sleep_mode", "active")),
                traffic_steering_preference=str(
                    getattr(state.cell, "traffic_steering_preference", "balanced")
                ),
                mobility_bias_db=(
                    float(getattr(state.cell, "mobility_bias_db", 0.0))
                    if getattr(state.cell, "mobility_bias_db", None) is not None
                    else None
                ),
                handover_margin_db=(
                    float(getattr(state.cell, "handover_margin_db", 0.0))
                    if getattr(state.cell, "handover_margin_db", None) is not None
                    else None
                ),
                handover_cost_db=(
                    float(getattr(state.cell, "handover_cost_db", 0.0))
                    if getattr(state.cell, "handover_cost_db", None) is not None
                    else None
                ),
                handover_interruption_slots=(
                    int(getattr(state.cell, "handover_interruption_slots", 0))
                    if getattr(state.cell, "handover_interruption_slots", None) is not None
                    else None
                ),
                steering_hysteresis_db=(
                    float(getattr(state.cell, "steering_hysteresis_db", 0.0))
                    if getattr(state.cell, "steering_hysteresis_db", None) is not None
                    else None
                ),
                measurement_event_trigger=str(
                    getattr(state.cell, "measurement_event_trigger", "a3_offset")
                ),
                measurement_event_offset_db=(
                    float(getattr(state.cell, "measurement_event_offset_db", 0.0))
                    if getattr(state.cell, "measurement_event_offset_db", None) is not None
                    else None
                ),
                connected_ue_count=connected_ue_count,
                scheduled_ue_count=len(scheduled_ue_ids),
                total_queue_bytes=int(total_queue_bytes),
                resource_grid_total_re=int(getattr(resource_grid_state, "total_resource_elements", 0)),
                resource_grid_allocated_re=int(getattr(resource_grid_state, "allocated_resource_elements", 0)),
                resource_grid_utilization=float(getattr(resource_grid_state, "utilization", 0.0)),
                interference_penalty_db_mean=_mean_or_none(
                    [
                        float(item.interference_penalty_db)
                        for item in ue_observations
                        if item.interference_penalty_db is not None and item.is_scheduled
                    ]
                ),
                inter_cell_interference_penalty_db_mean=inter_cell_interference_penalty_db_mean,
                inter_cell_frequency_coupling_fraction_mean=inter_cell_frequency_coupling_fraction_mean,
                inter_cell_interference_source_count_mean=inter_cell_interference_source_count_mean,
                active_ue_ids=[str(ue.ue_id) for ue in state.ues],
                scheduled_ue_ids=[str(ue_id) for ue_id in scheduled_ue_ids],
            )
        )

    return Observation(
        slot_id=state.slot_id,
        runtime_mode=str(getattr(state, "runtime_mode", "single_active_cell")),
        cell_id=state.cell.cell_id,
        avg_sinr_db=avg_sinr_db,
        avg_bler=avg_bler,
        avg_sinr_db_scheduled=avg_sinr_db_scheduled,
        avg_bler_scheduled=avg_bler_scheduled,
        avg_sinr_db_all=avg_sinr_db_all,
        avg_radio_quality_db_all=avg_radio_quality_db_all,
        total_queue_bytes=total_queue_bytes,
        total_throughput_mbps=total_throughput_mbps,
        current_codebook_id=state.cell.current_codebook_id,
        current_beam_profile_id=state.cell.current_beam_profile_id,
        tx_power_dbm=state.cell.tx_power_dbm,
        scheduler_mode=state.scheduler.scheduler_mode,
        scheduler_packing_strategy=str(
            getattr(getattr(state, "scheduler", None), "packing_strategy", "frequency_first")
        ),
        scheduler_demand_source=(
            str(getattr(scheduler_stage_trace, "demand_source", ""))
            if scheduler_stage_trace is not None
            else None
        ),
        scheduler_demand_count=int(len(scheduler_demands)),
        scheduler_true_backlog_bytes_total=int(
            sum(
                int(getattr(demand, "true_backlog_bytes", getattr(demand, "backlog_bytes", 0)))
                for demand in scheduler_demands
            )
        ),
        scheduler_demand_bytes_total=int(
            sum(
                int(getattr(demand, "demand_bytes", getattr(demand, "backlog_bytes", 0)))
                for demand in scheduler_demands
            )
        ),
        scheduler_observed_backlog_bytes_total=int(
            sum(
                int(getattr(demand, "observed_backlog_bytes", getattr(demand, "backlog_bytes", 0)))
                for demand in scheduler_demands
            )
        ),
        scheduler_bsr_bytes_total=int(
            sum(int(getattr(demand, "bsr_like_bytes", 0)) for demand in scheduler_demands)
        ),
        scheduler_low_confidence_channel_count=int(
            len(getattr(scheduler_stage_trace, "low_confidence_channel_ue_ids", []) or [])
        ) if scheduler_stage_trace is not None else 0,
        scheduler_stale_csi_count=int(
            len(getattr(scheduler_stage_trace, "stale_csi_ue_ids", []) or [])
        ) if scheduler_stage_trace is not None else 0,
        scheduled_ue_ids=scheduled_ue_ids,
        num_scheduled_ues=len(scheduled_ue_ids),
        ue_observations=ue_observations,
        cell_observations=cell_observations,
        connected_ue_count=connected_ue_count,
        out_of_sync_ue_count=out_of_sync_ue_count,
        handover_pending_ue_count=handover_pending_ue_count,
        handover_completed_ue_count=handover_completed_ue_count,
        handover_interruption_ue_count=handover_interruption_ue_count,
        gnb_num_tx_ant=int(getattr(antenna_state, "gnb_num_tx_ant", 0))
        if antenna_state is not None else None,
        carrier_frequency_hz=float(getattr(antenna_state, "carrier_frequency_hz", 0.0))
        if antenna_state is not None else None,
        bandwidth_hz=float(getattr(antenna_state, "bandwidth_hz", 0.0))
        if antenna_state is not None else None,
        subcarrier_spacing_hz=float(getattr(antenna_state, "subcarrier_spacing_hz", 0.0))
        if antenna_state is not None else None,
        cyclic_prefix_type=str(getattr(antenna_state, "cyclic_prefix_type", "normal"))
        if antenna_state is not None else None,
        num_ofdm_symbols_per_slot=int(getattr(antenna_state, "num_ofdm_symbols_per_slot", 0))
        if antenna_state is not None else None,
        resource_grid_total_re=int(cluster_resource_grid_summary["total_resource_elements"])
        if resource_grid_state is not None else None,
        resource_grid_allocated_re=int(cluster_resource_grid_summary["allocated_resource_elements"])
        if resource_grid_state is not None else None,
        resource_grid_utilization=float(cluster_resource_grid_summary["utilization"])
        if resource_grid_state is not None else None,
        resource_grid_n_prbs_total=int(cluster_resource_grid_summary["n_prbs_total"])
        if resource_grid_state is not None else None,
        resource_grid_n_ofdm_symbols_per_slot=int(
            cluster_resource_grid_summary["n_ofdm_symbols_per_slot"]
        ) if resource_grid_state is not None else None,
        resource_grid_fft_size=int(cluster_resource_grid_summary["fft_size"])
        if resource_grid_state is not None else None,
        resource_grid_overlap_fraction_mean=float(
            cluster_resource_grid_summary["overlap_fraction_mean"]
        ) if resource_grid_state is not None else None,
        resource_grid_mu_mimo_overlap_pair_count=int(
            cluster_resource_grid_summary["mu_mimo_overlap_pair_count"]
        ) if resource_grid_state is not None else None,
        resource_grid_mu_mimo_group_count=int(
            cluster_resource_grid_summary["mu_mimo_group_count"]
        ) if resource_grid_state is not None else None,
        resource_grid_mu_mimo_ue_count=int(
            cluster_resource_grid_summary["mu_mimo_ue_count"]
        ) if resource_grid_state is not None else None,
        resource_grid_adjacent_interference_pair_count=int(
            cluster_resource_grid_summary["adjacent_interference_pair_count"]
        ) if resource_grid_state is not None else None,
        resource_grid_interference_group_count=int(
            cluster_resource_grid_summary["interference_group_count"]
        ) if resource_grid_state is not None else None,
        interference_penalty_db_mean=float(
            cluster_resource_grid_summary["interference_penalty_db_mean"]
        ) if resource_grid_state is not None else None,
        inter_cell_interference_penalty_db_mean=inter_cell_interference_penalty_db_mean,
        inter_cell_frequency_coupling_fraction_mean=inter_cell_frequency_coupling_fraction_mean,
        inter_cell_interference_source_count_mean=inter_cell_interference_source_count_mean,
        spatial_multiplexing_gain_db_mean=spatial_multiplexing_gain_db_mean,
        measurement_produced_at_slot=int(getattr(measurement, "produced_at_slot", state.slot_id))
        if measurement is not None else None,
        measurement_delivered_at_slot=getattr(measurement, "delivered_at_slot", None)
        if measurement is not None else None,
        measurement_age_slots=int(getattr(measurement, "age_slots", 0))
        if measurement is not None else None,
        measurement_queue_delay_slots=int(getattr(measurement, "queue_delay_slots", 0))
        if measurement is not None else None,
        measurement_sinr_window_slots=int(getattr(measurement, "sinr_window_slots", 1))
        if measurement is not None else None,
        measurement_sinr_noise_std_db_applied=float(
            getattr(measurement, "sinr_noise_std_db_applied", 0.0)
        ) if measurement is not None else None,
        measurement_bler_window_tbs=int(getattr(measurement, "bler_window_tbs", 1))
        if measurement is not None else None,
        measurement_throughput_granularity_slots=int(getattr(measurement, "throughput_granularity_slots", 1))
        if measurement is not None else None,
        measurement_queue_age_slots_mean=measurement_queue_age_slots_mean,
        measurement_sinr_effective_sample_count_mean=measurement_sinr_effective_sample_count_mean,
        measurement_bler_sample_count_mean=measurement_bler_sample_count_mean,
        measurement_throughput_sample_count_mean=measurement_throughput_sample_count_mean,
    )
