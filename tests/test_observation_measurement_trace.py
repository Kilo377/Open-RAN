import unittest
from types import SimpleNamespace
from typing import Any, cast

from core.metrics import build_observation
from models.scheduling import LogicalChannelDemand, LogicalChannelGroupDemand, SchedulingStageTrace, UeSchedulingDemand
from state.antenna_state import AntennaState
from state.beamforming_state import BeamformingState
from state.connection_state import ConnectionState
from state.logical_channel_group_state import LogicalChannelGroupState
from state.logical_channel_state import LogicalChannelState
from state.measurement_state import MeasurementState, UeMeasurement
from state.resource_grid_state import ResourceAllocationState, ResourceGridState


class TestObservationMeasurementTrace(unittest.TestCase):
    def test_build_observation_exposes_measurement_metadata(self) -> None:
        state = SimpleNamespace(
            slot_id=9,
            runtime_mode="multi_cell_ready",
            cell=SimpleNamespace(
                cell_id="cell_0",
                current_codebook_id=1,
                current_beam_profile_id=1,
                tx_power_dbm=30.0,
                beam_candidate_preference="coverage",
                power_cap_dbm=31.0,
                traffic_steering_preference="edge_offload",
                mobility_bias_db=2.0,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="round_robin",
                scheduler_policy_hint="latency_bias",
                latency_bias=1.5,
                slice_weight_adjustment={"urllc": 1.2},
                packing_strategy="frequency_first",
                scheduled_ue_ids=["ue_0"],
                last_demands=[
                    UeSchedulingDemand(
                        ue_id="ue_0",
                        true_backlog_bytes=200,
                        demand_bytes=200,
                        backlog_bytes=200,
                        observed_backlog_bytes=240,
                        hol_delay_ms=2.0,
                        priority=5,
                        qos_class="urllc",
                        five_qi=80,
                        slice_id="urllc",
                        traffic_type="control",
                        bsr_like_bytes=200,
                        bsr_reported_bytes=180,
                        bsr_pending_bytes=20,
                        harq_pending_bytes=0,
                        harq_urgency_score=0.0,
                        cqi=9,
                        ri=2,
                        pmi=1,
                        cqi_validity_age_slots=3,
                        ri_validity_age_slots=3,
                        pmi_validity_age_slots=3,
                        measurement_age_slots=0,
                        channel_quality_confidence=0.82,
                        demand_source="mac_state+measurement_metadata",
                        demand_reason="bsr+measurement",
                        logical_channel_demands=[
                            LogicalChannelDemand(
                                logical_channel_id="lc_ctrl",
                                drb_id=2,
                                lcg_id=0,
                                true_backlog_bytes=40,
                                observed_backlog_bytes=48,
                                demand_bytes=40,
                                backlog_bytes=40,
                                hol_delay_ms=2.0,
                                priority=5,
                                packet_delay_budget_ms=20.0,
                                qos_class="urllc",
                                five_qi=80,
                                slice_id="urllc",
                                traffic_type="control",
                                bsr_like_bytes=60,
                                bsr_reported_bytes=40,
                                bsr_pending_bytes=20,
                                bsr_age_slots=2,
                                measurement_age_slots=0,
                            ),
                            LogicalChannelDemand(
                                logical_channel_id="lc_video",
                                drb_id=1,
                                lcg_id=0,
                                true_backlog_bytes=160,
                                observed_backlog_bytes=192,
                                demand_bytes=160,
                                backlog_bytes=160,
                                hol_delay_ms=2.0,
                                priority=1,
                                packet_delay_budget_ms=80.0,
                                qos_class="embb",
                                five_qi=9,
                                slice_id="embb",
                                traffic_type="video",
                                bsr_like_bytes=140,
                                bsr_reported_bytes=140,
                                bsr_pending_bytes=0,
                                bsr_age_slots=2,
                                measurement_age_slots=0,
                                pbr_credit_bytes=32.0,
                            ),
                        ],
                        logical_channel_group_demands=[
                            LogicalChannelGroupDemand(
                                lcg_id=0,
                                logical_channel_ids=["lc_ctrl", "lc_video"],
                                drb_ids=[1, 2],
                                true_backlog_bytes=200,
                                observed_backlog_bytes=240,
                                demand_bytes=200,
                                prioritized_queue_bytes=160,
                                dominant_priority=5,
                                min_packet_delay_budget_ms=20.0,
                                qos_class="urllc",
                                five_qi=80,
                                bsr_reported_bytes=180,
                                bsr_pending_bytes=20,
                                bsr_age_slots=2,
                                pbr_credit_bytes=32.0,
                            )
                        ],
                    )
                ],
                last_stage_trace=SchedulingStageTrace(
                    demand_source="mac_state+measurement_metadata",
                    demand_visible_ue_ids=["ue_0"],
                    feasible_candidate_ue_ids=["ue_0"],
                ),
            ),
            ues=[
                SimpleNamespace(
                    ue_id="ue_0",
                    queue_bytes=200,
                    throughput_mbps=0.5,
                    sinr_db=4.0,
                    bler=0.1,
                    serving_beam_id=0,
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=50.0,
                    priority=1,
                    radio_quality_db=3.0,
                    logical_channels=[
                        LogicalChannelState(
                            logical_channel_id="lc_ctrl",
                            drb_id=2,
                            lcg_id=0,
                            queue_bytes=40,
                            priority=5,
                            packet_delay_budget_ms=20.0,
                            slice_id="urllc",
                            traffic_type="control",
                        ),
                        LogicalChannelState(
                            logical_channel_id="lc_video",
                            drb_id=1,
                            lcg_id=0,
                            queue_bytes=160,
                            priority=1,
                            packet_delay_budget_ms=80.0,
                            slice_id="embb",
                            traffic_type="video",
                            prioritized_bit_rate_bps=500000,
                            pbr_credit_bytes=32.0,
                        ),
                    ],
                    logical_channel_groups=[
                        LogicalChannelGroupState(
                            lcg_id=0,
                            logical_channel_ids=["lc_ctrl", "lc_video"],
                            drb_ids=[1, 2],
                            queue_bytes=200,
                            prioritized_queue_bytes=160,
                            pbr_credit_bytes=32.0,
                            scheduled_bytes=120,
                            prioritized_scheduled_bytes=32,
                            delivered_bytes=96,
                            bsr_reported_bytes=180,
                            bsr_pending_bytes=20,
                            bsr_age_slots=2,
                            last_bsr_report_slot=7,
                            last_bsr_trigger_slot=7,
                            bsr_triggered=False,
                            bsr_trigger_reason="buffer_stale",
                        )
                    ],
                    scheduled_logical_channel_ids=["lc_ctrl", "lc_video"],
                    scheduled_drb_ids=[1, 2],
                    scheduled_lcg_ids=[0],
                    grant_target_mcs=8,
                    grant_mcs_table_id="qam256",
                    grant_tbs_selection_method="latency_guarded",
                    grant_dmrs_pattern_id="dmrs_type1_len2",
                    grant_cqi_validity_age_slots=3,
                    grant_ri_validity_age_slots=3,
                    grant_pmi_validity_age_slots=3,
                    grant_channel_quality_confidence=0.72,
                    grant_link_adaptation_reason="latency_guarded",
                )
            ],
            measurement=MeasurementState(
                produced_at_slot=9,
                delivered_at_slot=None,
                age_slots=0,
                queue_delay_slots=2,
                sinr_window_slots=8,
                sinr_noise_std_db_applied=1.5,
                bler_window_tbs=12,
                throughput_granularity_slots=5,
                ue_measurements={
                    "ue_0": UeMeasurement(
                        ue_id="ue_0",
                        queue_bytes=240,
                        throughput_mbps=0.7,
                        sinr_db=5.5,
                        bler=0.2,
                        produced_at_slot=9,
                        age_slots=0,
                        queue_age_slots=2,
                        queue_delay_slots_applied=2,
                        queue_produced_at_slot=7,
                        sinr_window_len=8,
                        sinr_window_slots=8,
                        sinr_effective_sample_count=4,
                        sinr_noise_std_db_applied=1.5,
                        bler_window_tbs=12,
                        bler_window_sample_count=3,
                        bler_sample_count=3,
                        throughput_window_slots=5,
                        throughput_granularity_slots=5,
                        last_measurement_update_slot=9,
                    )
                },
            ),
            connections={
                "ue_0": ConnectionState(
                    ue_id="ue_0",
                    serving_cell_id="cell_0",
                    connection_state="connected",
                    c_rnti=4100,
                    serving_beam_id=0,
                    rank_indicator=1,
                    pmi=0,
                    cqi=5,
                    active_harq_processes=[0],
                )
            },
            antenna=AntennaState(
                gnb_num_tx_ant=8,
                ue_num_rx_ant_by_ue={"ue_0": 2},
                carrier_frequency_hz=3.5e9,
                bandwidth_hz=40e6,
                subcarrier_spacing_hz=30e3,
                cyclic_prefix_type="normal",
                num_ofdm_symbols_per_slot=28,
                gnb_panel_layout="4x2",
                ue_panel_layout_by_ue={"ue_0": "1x2"},
                polarization="dual",
            ),
            beamforming={
                "ue_0": BeamformingState(
                    ue_id="ue_0",
                    codebook_id=0,
                    beam_profile_id=0,
                    serving_beam_id=0,
                    precoder_id=2,
                    combiner_id=1,
                    rank=2,
                    layer_count=2,
                    beam_gain_db=4.5,
                    beam_direction_deg=15.0,
                    precoding_matrix_label="cb2-r2-b0",
                )
            },
            resource_grid=ResourceGridState(
                total_prbs=12,
                total_symbols=14,
                subcarriers_per_prb=12,
                n_prbs_total=106,
                n_subcarriers_per_prb=12,
                n_ofdm_symbols_per_slot=28,
                fft_size=2048,
                usable_subcarrier_mask=[1] * 1272,
                carrier_frequency_hz=3.5e9,
                bandwidth_hz=40e6,
                subcarrier_spacing_hz=30e3,
                cyclic_prefix_type="normal",
                total_resource_elements=2016,
                allocated_resource_elements=576,
                utilization=576.0 / 2016.0,
                overlap_fraction_mean=0.0,
                interference_penalty_db_mean=0.2,
                scheduled_ue_count=1,
                scheduled_layer_count_total=2,
                mu_mimo_overlap_pair_count=0,
                allocations=[
                    ResourceAllocationState(
                        ue_id="ue_0",
                        start_prb=2,
                        allocated_prbs=6,
                        start_symbol=1,
                        allocated_symbols=8,
                        layer_count=2,
                        resource_elements=576,
                        overlap_fraction=0.0,
                        interference_penalty_db=0.2,
                    )
                ],
            ),
            cluster_last_demands=[
                UeSchedulingDemand(
                    ue_id="ue_0",
                    true_backlog_bytes=200,
                    demand_bytes=200,
                    backlog_bytes=200,
                    observed_backlog_bytes=240,
                    hol_delay_ms=2.0,
                    priority=5,
                    qos_class="urllc",
                    five_qi=80,
                    slice_id="urllc",
                    traffic_type="control",
                    bsr_like_bytes=200,
                    bsr_reported_bytes=180,
                    bsr_pending_bytes=20,
                    harq_pending_bytes=0,
                    harq_urgency_score=0.0,
                    cqi=9,
                    ri=2,
                    pmi=1,
                    cqi_validity_age_slots=3,
                    ri_validity_age_slots=3,
                    pmi_validity_age_slots=3,
                    measurement_age_slots=0,
                    channel_quality_confidence=0.82,
                    demand_source="mac_state+measurement_metadata",
                    demand_reason="bsr+measurement",
                    logical_channel_demands=[
                        LogicalChannelDemand(
                            logical_channel_id="lc_ctrl",
                            drb_id=2,
                            lcg_id=0,
                            true_backlog_bytes=40,
                            observed_backlog_bytes=48,
                            demand_bytes=40,
                            backlog_bytes=40,
                            hol_delay_ms=2.0,
                            priority=5,
                            packet_delay_budget_ms=20.0,
                            qos_class="urllc",
                            five_qi=80,
                            slice_id="urllc",
                            traffic_type="control",
                            bsr_like_bytes=60,
                            bsr_reported_bytes=40,
                            bsr_pending_bytes=20,
                            bsr_age_slots=2,
                            measurement_age_slots=0,
                        ),
                        LogicalChannelDemand(
                            logical_channel_id="lc_video",
                            drb_id=1,
                            lcg_id=0,
                            true_backlog_bytes=160,
                            observed_backlog_bytes=192,
                            demand_bytes=160,
                            backlog_bytes=160,
                            hol_delay_ms=2.0,
                            priority=1,
                            packet_delay_budget_ms=80.0,
                            qos_class="embb",
                            five_qi=9,
                            slice_id="embb",
                            traffic_type="video",
                            bsr_like_bytes=140,
                            bsr_reported_bytes=140,
                            bsr_pending_bytes=0,
                            bsr_age_slots=2,
                            measurement_age_slots=0,
                            pbr_credit_bytes=32.0,
                        ),
                    ],
                    logical_channel_group_demands=[
                        LogicalChannelGroupDemand(
                            lcg_id=0,
                            logical_channel_ids=["lc_ctrl", "lc_video"],
                            drb_ids=[1, 2],
                            true_backlog_bytes=200,
                            observed_backlog_bytes=240,
                            demand_bytes=200,
                            prioritized_queue_bytes=160,
                            dominant_priority=5,
                            min_packet_delay_budget_ms=20.0,
                            qos_class="urllc",
                            five_qi=80,
                            bsr_reported_bytes=180,
                            bsr_pending_bytes=20,
                            bsr_age_slots=2,
                            pbr_credit_bytes=32.0,
                        )
                    ],
                )
            ],
            cluster_last_stage_trace=SchedulingStageTrace(
                demand_source="mac_state+measurement_metadata",
                demand_visible_ue_ids=["ue_0"],
                feasible_candidate_ue_ids=["ue_0"],
            ),
            managed_cells={
                "cell_0": SimpleNamespace(
                    cell=SimpleNamespace(
                        cell_id="cell_0",
                        tx_power_dbm=30.0,
                        current_codebook_id=1,
                        current_beam_profile_id=1,
                        beam_candidate_preference="coverage",
                        power_cap_dbm=31.0,
                        traffic_steering_preference="edge_offload",
                        mobility_bias_db=2.0,
                    ),
                    scheduler=SimpleNamespace(
                        scheduler_mode="round_robin",
                        scheduler_policy_hint="latency_bias",
                        latency_bias=1.5,
                        slice_weight_adjustment={"urllc": 1.2},
                        packing_strategy="frequency_first",
                        scheduled_ue_ids=["ue_0"],
                    ),
                    resource_grid=ResourceGridState(
                        n_prbs_total=106,
                        n_ofdm_symbols_per_slot=28,
                        fft_size=2048,
                        overlap_fraction_mean=0.0,
                        interference_penalty_db_mean=0.2,
                        mu_mimo_overlap_pair_count=0,
                        mu_mimo_group_count=0,
                        mu_mimo_ue_count=0,
                        adjacent_interference_pair_count=0,
                        interference_groups=[],
                        total_resource_elements=2016,
                        allocated_resource_elements=576,
                        utilization=576.0 / 2016.0,
                    ),
                    radio_profile={"cell_type": "macro", "carrier_frequency_hz": 3.5e9, "bandwidth_hz": 40e6, "subcarrier_spacing_hz": 30e3},
                    is_primary=True,
                    active_ue_ids=["ue_0"],
                    scheduled_ue_ids=["ue_0"],
                ),
                "cell_1": SimpleNamespace(
                    cell=SimpleNamespace(
                        cell_id="cell_1",
                        tx_power_dbm=28.0,
                        current_codebook_id=2,
                        current_beam_profile_id=1,
                        beam_candidate_preference="capacity",
                        power_cap_dbm=26.0,
                        traffic_steering_preference="balanced",
                        mobility_bias_db=-1.5,
                    ),
                    scheduler=SimpleNamespace(
                        scheduler_mode="round_robin",
                        scheduler_policy_hint="balanced",
                        latency_bias=None,
                        slice_weight_adjustment={},
                        packing_strategy="time_first",
                        scheduled_ue_ids=[],
                    ),
                    resource_grid=ResourceGridState(
                        total_resource_elements=6720,
                        allocated_resource_elements=0,
                        utilization=0.0,
                    ),
                    radio_profile={"cell_type": "mmwave_hotspot", "carrier_frequency_hz": 28.0e9, "bandwidth_hz": 100e6, "subcarrier_spacing_hz": 120e3},
                    is_primary=False,
                    active_ue_ids=[],
                    scheduled_ue_ids=[],
                ),
            },
        )

        observation = build_observation(cast(Any, state))
        ue_observation = observation.ue_observations[0]

        self.assertEqual(observation.runtime_mode, "multi_cell_ready")
        self.assertEqual(observation.measurement_produced_at_slot, 9)
        self.assertEqual(observation.measurement_queue_delay_slots, 2)
        self.assertEqual(observation.measurement_sinr_noise_std_db_applied, 1.5)
        self.assertEqual(observation.measurement_bler_sample_count_mean, 3.0)
        self.assertEqual(observation.connected_ue_count, 1)
        self.assertEqual(observation.gnb_num_tx_ant, 8)
        self.assertEqual(observation.carrier_frequency_hz, 3.5e9)
        self.assertEqual(observation.bandwidth_hz, 40e6)
        self.assertEqual(observation.subcarrier_spacing_hz, 30e3)
        self.assertEqual(observation.cyclic_prefix_type, "normal")
        self.assertEqual(observation.num_ofdm_symbols_per_slot, 28)
        self.assertEqual(observation.resource_grid_total_re, 8736)
        self.assertEqual(observation.resource_grid_allocated_re, 576)
        self.assertAlmostEqual(observation.resource_grid_utilization or 0.0, 576.0 / 8736.0)
        self.assertEqual(observation.resource_grid_n_prbs_total, 106)
        self.assertEqual(observation.resource_grid_n_ofdm_symbols_per_slot, 28)
        self.assertEqual(observation.resource_grid_fft_size, 2048)
        self.assertEqual(observation.resource_grid_mu_mimo_overlap_pair_count, 0)
        self.assertEqual(observation.interference_penalty_db_mean, 0.1)
        self.assertEqual(len(observation.cell_observations), 2)
        self.assertEqual(observation.cell_observations[0].cell_id, "cell_0")
        self.assertEqual(observation.cell_observations[0].current_codebook_id, 1)
        self.assertEqual(observation.cell_observations[0].scheduler_policy_hint, "latency_bias")
        self.assertEqual(observation.cell_observations[0].traffic_steering_preference, "edge_offload")
        self.assertEqual(observation.cell_observations[1].cell_type, "mmwave_hotspot")
        self.assertEqual(observation.cell_observations[1].bandwidth_hz, 100e6)
        self.assertEqual(ue_observation.queue_delay_slots_applied, 2)
        self.assertEqual(ue_observation.queue_produced_at_slot, 7)
        self.assertEqual(ue_observation.sinr_window_slots, 8)
        self.assertEqual(ue_observation.bler_window_tbs, 12)
        self.assertEqual(ue_observation.bler_sample_count, 3)
        self.assertEqual(ue_observation.throughput_granularity_slots, 5)
        self.assertEqual(ue_observation.logical_channel_count, 2)
        self.assertEqual(ue_observation.logical_channel_group_count, 1)
        self.assertEqual(ue_observation.dominant_logical_channel_id, "lc_video")
        self.assertEqual(ue_observation.dominant_lcg_id, 0)
        self.assertEqual(ue_observation.scheduled_lcg_ids, [0])
        self.assertEqual(len(ue_observation.logical_channel_observations), 2)
        self.assertEqual(len(ue_observation.logical_channel_group_observations), 1)
        self.assertEqual(
            ue_observation.logical_channel_group_observations[0].prioritized_queue_bytes,
            160,
        )
        self.assertEqual(
            ue_observation.logical_channel_group_observations[0].bsr_reported_bytes,
            180,
        )
        self.assertEqual(
            ue_observation.logical_channel_group_observations[0].bsr_pending_bytes,
            20,
        )
        self.assertEqual(
            ue_observation.logical_channel_group_observations[0].last_bsr_report_slot,
            7,
        )
        self.assertEqual(
            ue_observation.logical_channel_group_observations[0].bsr_trigger_reason,
            "buffer_stale",
        )
        self.assertEqual(ue_observation.connection_state, "connected")
        self.assertEqual(ue_observation.c_rnti, 4100)
        self.assertEqual(ue_observation.num_tx_ant, 8)
        self.assertEqual(ue_observation.num_rx_ant, 2)
        self.assertEqual(ue_observation.beamforming_rank, 2)
        self.assertEqual(ue_observation.beamforming_layer_count, 2)
        self.assertEqual(ue_observation.precoder_id, 2)
        self.assertEqual(ue_observation.combiner_id, 1)
        self.assertEqual(ue_observation.beam_direction_deg, 15.0)
        self.assertEqual(ue_observation.beamforming_gain_db, 4.5)
        self.assertEqual(ue_observation.grant_start_prb, 2)
        self.assertEqual(ue_observation.grant_start_symbol, 1)
        self.assertEqual(ue_observation.grant_allocated_prbs, 6)
        self.assertEqual(ue_observation.grant_allocated_symbols, 8)
        self.assertEqual(ue_observation.grant_target_mcs, 8)
        self.assertEqual(ue_observation.grant_mcs_table_id, "qam256")
        self.assertEqual(ue_observation.grant_tbs_selection_method, "latency_guarded")
        self.assertEqual(ue_observation.grant_dmrs_pattern_id, "dmrs_type1_len2")
        self.assertEqual(ue_observation.grant_cqi_validity_age_slots, 3)
        self.assertEqual(ue_observation.grant_ri_validity_age_slots, 3)
        self.assertEqual(ue_observation.grant_pmi_validity_age_slots, 3)
        self.assertAlmostEqual(ue_observation.grant_channel_quality_confidence or 0.0, 0.72, places=2)
        self.assertEqual(ue_observation.grant_link_adaptation_reason, "latency_guarded")
        self.assertEqual(ue_observation.demand_true_backlog_bytes, 200)
        self.assertEqual(ue_observation.demand_observed_backlog_bytes, 240)
        self.assertEqual(ue_observation.demand_bytes, 200)
        self.assertEqual(ue_observation.demand_qos_class, "urllc")
        self.assertEqual(ue_observation.demand_five_qi, 80)
        self.assertEqual(ue_observation.demand_bsr_reported_bytes, 180)
        self.assertEqual(ue_observation.demand_bsr_pending_bytes, 20)
        self.assertEqual(ue_observation.logical_channel_demand_count, 2)
        self.assertEqual(ue_observation.logical_channel_group_demand_count, 1)
        self.assertEqual(ue_observation.resource_grid_re_count, 576)
        self.assertEqual(ue_observation.resource_grid_overlap_fraction, 0.0)
        self.assertEqual(ue_observation.interference_penalty_db, 0.2)
        self.assertEqual(observation.scheduler_demand_source, "mac_state+measurement_metadata")
        self.assertEqual(observation.scheduler_demand_count, 1)
        self.assertEqual(observation.scheduler_demand_bytes_total, 200)
        self.assertEqual(observation.scheduler_bsr_bytes_total, 200)


if __name__ == "__main__":
    unittest.main()
