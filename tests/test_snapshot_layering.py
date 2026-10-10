import unittest
from types import SimpleNamespace

from core.state_updater import StateUpdater
from models.action import ControlMetadata
from models.observation import Observation
from state.antenna_state import AntennaState
from state.beamforming_state import BeamformingState
from state.connection_state import ConnectionState
from state.link_state import LinkState
from state.logical_channel_state import LogicalChannelState
from state.measurement_state import MeasurementState, UeMeasurement
from state.resource_grid_state import ResourceAllocationState, ResourceGridState


class TestSnapshotLayering(unittest.TestCase):
    def test_record_snapshot_keeps_true_observable_and_per_ue_measurement_views(self) -> None:
        state = SimpleNamespace(
            slot_id=8,
            sim_time_ms=8.0,
            total_tb_size_bytes=180,
            ues=[
                SimpleNamespace(
                    ue_id="ue_0",
                    queue_bytes=220,
                    throughput_mbps=1.1,
                    sinr_db=9.2,
                    radio_quality_db=8.5,
                    bler=0.06,
                    serving_beam_id=0,
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=80.0,
                    priority=1,
                    allocated_prbs=6,
                    allocated_symbols=4,
                    is_retx=False,
                    mcs_index=5,
                    grant_target_mcs=5,
                    grant_bytes_budget=180,
                    scheduled_bytes=180,
                    delivered_bytes=180,
                    new_tx_bytes=180,
                    retx_tx_bytes=0,
                    delivered_new_bytes=180,
                    delivered_retx_bytes=0,
                    is_scheduled=True,
                    ack=True,
                    logical_channels=[
                        LogicalChannelState(
                            logical_channel_id="lc_ctrl",
                            drb_id=2,
                            lcg_id=0,
                            queue_bytes=40,
                            priority=4,
                            packet_delay_budget_ms=20.0,
                            slice_id="urllc",
                            traffic_type="control",
                        ),
                        LogicalChannelState(
                            logical_channel_id="lc_video",
                            drb_id=1,
                            lcg_id=1,
                            queue_bytes=180,
                            priority=1,
                            packet_delay_budget_ms=80.0,
                            slice_id="embb",
                            traffic_type="video",
                            prioritized_bit_rate_bps=500000,
                            pbr_credit_bytes=24.0,
                            scheduled_bytes=180,
                            prioritized_scheduled_bytes=24,
                            delivered_bytes=180,
                        ),
                    ],
                    scheduled_logical_channel_ids=["lc_video"],
                    scheduled_drb_ids=[1],
                    scheduled_lcg_ids=[1],
                    harq=SimpleNamespace(pending_retx_bytes=0),
                )
            ],
            cell=SimpleNamespace(
                tx_power_dbm=30.0,
                current_codebook_id=1,
                current_beam_profile_id=0,
            ),
            antenna=AntennaState(
                gnb_num_tx_ant=4,
                ue_num_rx_ant_by_ue={"ue_0": 2},
                carrier_frequency_hz=3.5e9,
                subcarrier_spacing_hz=30e3,
                gnb_panel_layout="2x2",
                ue_panel_layout_by_ue={"ue_0": "1x2"},
                polarization="single",
            ),
            traffic=SimpleNamespace(
                arrival_bytes_total=80,
                arrival_bytes_by_ue={"ue_0": 80},
                arrival_bytes_by_logical_channel={"ue_0:lc_ctrl": 20, "ue_0:lc_video": 60},
                arrival_bytes_by_logical_channel_group={"ue_0:lcg_0": 20, "ue_0:lcg_1": 60},
            ),
            measurement=MeasurementState(
                produced_at_slot=8,
                delivered_at_slot=None,
                age_slots=0,
                queue_delay_slots=1,
                sinr_window_slots=8,
                sinr_noise_std_db_applied=2.0,
                bler_window_tbs=20,
                throughput_granularity_slots=10,
                ue_measurements={
                    "ue_0": UeMeasurement(
                        ue_id="ue_0",
                        queue_bytes=260,
                        throughput_mbps=0.9,
                        sinr_db=7.8,
                        bler=0.12,
                        is_scheduled=True,
                        ack=True,
                        tb_size_bytes=180,
                        produced_at_slot=8,
                        age_slots=0,
                        queue_age_slots=1,
                        queue_delay_slots_applied=1,
                        queue_produced_at_slot=7,
                        sinr_window_len=8,
                        sinr_window_slots=8,
                        sinr_effective_sample_count=4,
                        sinr_noise_std_db_applied=2.0,
                        bler_window_tbs=20,
                        bler_window_sample_count=3,
                        bler_sample_count=3,
                        throughput_window_slots=5,
                        throughput_granularity_slots=10,
                        last_measurement_update_slot=8,
                    )
                },
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="latency_aware",
                scheduled_ue_ids=["ue_0"],
                grants=[],
                last_decision=SimpleNamespace(total_prbs=12, total_symbols=14),
            ),
            control=SimpleNamespace(
                decision_metadata=ControlMetadata(
                    source="controller_manager",
                    issued_at_slot=8,
                    reason="merged_control_decision",
                    decision_id="decision:8",
                ),
                actuation_outcomes=[{"status": "applied"}],
                input_observation_trace={
                    "observation_slot_id": 7,
                    "observation_produced_at_slot": 7,
                    "observation_delivered_at_slot": 8,
                    "measurement_produced_at_slot": 7,
                    "measurement_delivered_at_slot": 8,
                    "measurement_age_slots": 1,
                    "measurement_queue_delay_slots": 1,
                    "measurement_sinr_window_slots": 8,
                    "measurement_sinr_noise_std_db_applied": 2.0,
                    "measurement_bler_window_tbs": 20,
                    "measurement_throughput_granularity_slots": 10,
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
                    active_harq_processes=[],
                )
            },
            links={
                "ue_0": LinkState(
                    ue_id="ue_0",
                    serving_cell_id="cell_0",
                    pathloss_db=108.0,
                    shadowing_db=2.0,
                    doppler_hz=20.0,
                    delay_spread_ns=200.0,
                    channel_model_type="tdl_c",
                    fast_fading_seed=101,
                    last_post_eq_sinr_db=9.2,
                    cqi=5,
                    ri=1,
                    pmi=0,
                )
            },
            beamforming={
                "ue_0": BeamformingState(
                    ue_id="ue_0",
                    codebook_id=1,
                    beam_profile_id=0,
                    serving_beam_id=0,
                    precoder_id=1,
                    combiner_id=0,
                    rank=1,
                    layer_count=2,
                    beam_gain_db=2.4,
                    beam_direction_deg=0.0,
                    precoding_matrix_label="cb1-r1-b0",
                )
            },
            resource_grid=ResourceGridState(
                total_prbs=12,
                total_symbols=14,
                subcarriers_per_prb=12,
                total_resource_elements=2016,
                allocated_resource_elements=288,
                utilization=288.0 / 2016.0,
                overlap_fraction_mean=0.0,
                interference_penalty_db_mean=0.1,
                scheduled_ue_count=1,
                scheduled_layer_count_total=2,
                mu_mimo_overlap_pair_count=0,
                allocations=[
                    ResourceAllocationState(
                        ue_id="ue_0",
                        start_prb=0,
                        allocated_prbs=6,
                        start_symbol=0,
                        allocated_symbols=4,
                        layer_count=2,
                        resource_elements=288,
                        overlap_fraction=0.0,
                        interference_penalty_db=0.1,
                    )
                ],
            ),
            snapshots=[],
        )
        observation = Observation(
            slot_id=8,
            avg_sinr_db=7.8,
            avg_bler=0.12,
            total_queue_bytes=260,
            total_throughput_mbps=0.9,
            cell_id="cell_0",
            current_codebook_id=1,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="latency_aware",
            scheduled_ue_ids=["ue_0"],
            num_scheduled_ues=1,
            ue_observations=[],
            produced_at_slot=8,
            measurement_produced_at_slot=8,
            measurement_queue_delay_slots=1,
            measurement_sinr_window_slots=8,
            measurement_sinr_noise_std_db_applied=2.0,
            measurement_bler_window_tbs=20,
            measurement_throughput_granularity_slots=10,
            measurement_queue_age_slots_mean=1.0,
            measurement_sinr_effective_sample_count_mean=4.0,
            measurement_bler_sample_count_mean=3.0,
            measurement_throughput_sample_count_mean=5.0,
        )

        updater = StateUpdater()
        updater.record_snapshot(state, observation=observation)

        self.assertEqual(len(state.snapshots), 1)
        snapshot = state.snapshots[0]
        self.assertIn("true_metrics", snapshot)
        self.assertIn("observable_metrics", snapshot)
        self.assertIn("measurement_trace", snapshot)
        self.assertIn("control_trace", snapshot)
        self.assertEqual(snapshot["antenna"]["gnb_num_tx_ant"], 4)
        self.assertEqual(snapshot["resource_grid"]["allocated_resource_elements"], 288)
        self.assertAlmostEqual(snapshot["resource_grid"]["utilization"], 288.0 / 2016.0)
        self.assertEqual(snapshot["true_metrics"]["queue_bytes"], 220)
        self.assertEqual(snapshot["observable_metrics"]["queue_bytes"], 260)
        self.assertEqual(snapshot["measurement_trace"]["sinr_window_slots"], 8)
        self.assertEqual(snapshot["control_trace"]["input_measurement_age_slots"], 1)
        self.assertEqual(len(snapshot["ues"]), 1)
        ue_snapshot = snapshot["ues"][0]
        self.assertEqual(ue_snapshot["true_state"]["queue_bytes"], 220)
        self.assertEqual(ue_snapshot["observable_state"]["queue_bytes"], 260)
        self.assertEqual(ue_snapshot["measurement_trace"]["queue_delay_slots_applied"], 1)
        self.assertEqual(ue_snapshot["measurement_trace"]["sinr_effective_sample_count"], 4)
        self.assertEqual(ue_snapshot["connection"]["connection_state"], "connected")
        self.assertEqual(ue_snapshot["connection"]["c_rnti"], 4100)
        self.assertEqual(ue_snapshot["link"]["pathloss_db"], 108.0)
        self.assertEqual(ue_snapshot["link"]["fast_fading_seed"], 101)
        self.assertEqual(ue_snapshot["link"]["large_scale"]["pathloss_db"], 108.0)
        self.assertEqual(ue_snapshot["link"]["small_scale"]["fast_fading_seed"], 101)
        self.assertEqual(ue_snapshot["beamforming"]["layer_count"], 2)
        self.assertEqual(ue_snapshot["beamforming"]["precoder_id"], 1)
        self.assertEqual(len(ue_snapshot["logical_channels"]), 2)
        self.assertEqual(len(ue_snapshot["logical_channel_groups"]), 2)
        self.assertEqual(ue_snapshot["logical_channel_groups"][1]["lcg_id"], 1)
        self.assertEqual(ue_snapshot["logical_channel_groups"][1]["bsr_reported_bytes"], 180)
        self.assertEqual(ue_snapshot["logical_channel_groups"][1]["last_bsr_report_slot"], 8)
        self.assertEqual(ue_snapshot["logical_channel_groups"][1]["bsr_trigger_reason"], "initial")
        self.assertEqual(ue_snapshot["true_state"]["logical_channel_group_count"], 2)
        self.assertEqual(ue_snapshot["true_state"]["scheduled_lcg_ids"], [1])


if __name__ == "__main__":
    unittest.main()
