from __future__ import annotations

import unittest

from core.metrics import build_observation
from core.mobility_manager import MobilityManager
from state.cell_state import CellState
from state.connection_state import ConnectionState
from state.global_state import GlobalState
from state.link_state import LinkState
from state.resource_grid_state import ResourceGridState
from state.scheduler_state import SchedulerState
from state.ue_state import UeState
from state.channel_state import ChannelState
from state.beam_state import BeamState
from state.antenna_state import AntennaState
from state.control_state import ControlState


class MobilityManagerTest(unittest.TestCase):
    def _build_state(self) -> GlobalState:
        cell = CellState(
            cell_id="cell_0",
            managed_cell_ids=["cell_0", "cell_1"],
            neighbor_cell_positions={
                "cell_0": {"x_m": 0.0, "y_m": 0.0, "z_m": 25.0},
                "cell_1": {"x_m": 80.0, "y_m": 0.0, "z_m": 25.0},
            },
            cell_radio_profiles={
                "cell_0": {
                    "cell_type": "macro",
                    "carrier_frequency_hz": 3.5e9,
                    "bandwidth_hz": 20e6,
                    "subcarrier_spacing_hz": 30e3,
                    "tx_power_dbm": 30.0,
                    "gnb_num_tx_ant": 4,
                    "scenario_type": "uma",
                },
                "cell_1": {
                    "cell_type": "small_cell",
                    "carrier_frequency_hz": 3.5e9,
                    "bandwidth_hz": 20e6,
                    "subcarrier_spacing_hz": 30e3,
                    "tx_power_dbm": 30.0,
                    "gnb_num_tx_ant": 4,
                    "scenario_type": "uma",
                },
            },
            handover_margin_db=2.0,
            handover_delay_slots=1,
            traffic_steering_preference="edge_offload",
            mobility_bias_db=8.0,
        )
        ue = UeState(
            ue_id="ue_0",
            queue_bytes=1000,
            throughput_mbps=0.0,
            sinr_db=10.0,
            radio_quality_db=10.0,
            bler=0.1,
            serving_beam_id=0,
            slice_id="embb",
            traffic_type="video",
            latency_budget_ms=100.0,
            priority=1,
        )
        link = LinkState(
            ue_id="ue_0",
            serving_cell_id="cell_0",
            scenario_type="uma",
            los_state="los",
            x_m=65.0,
            y_m=0.0,
            z_m=1.5,
            distance_2d_m=65.0,
            distance_3d_m=69.0,
            velocity_mps=0.0,
            heading_deg=0.0,
            pathloss_db=120.0,
            shadowing_db=1.0,
            doppler_hz=0.0,
            delay_spread_ns=100.0,
            fast_fading_seed=1000,
            spatial_corr_seed=2000,
        )
        connection = ConnectionState(
            ue_id="ue_0",
            serving_cell_id="cell_0",
            serving_beam_id=0,
        )
        return GlobalState(
            slot_id=0,
            sim_time_ms=0.0,
            slot_duration_ms=1.0,
            cell=cell,
            scheduler=SchedulerState(),
            channel=ChannelState(),
            beam=BeamState(),
            antenna=AntennaState(),
            beamforming={},
            ues=[ue],
            control=ControlState(),
            connections={"ue_0": connection},
            links={"ue_0": link},
            resource_grid=ResourceGridState(),
        )

    def test_mobility_manager_triggers_and_completes_handover(self) -> None:
        state = self._build_state()
        manager = MobilityManager(slot_duration_ms=1.0, rng_seed=1)

        manager.update(state)
        connection = state.connections["ue_0"]
        self.assertEqual(connection.connection_state, "handover_pending")
        self.assertEqual(connection.handover_pending_target_cell_id, "cell_1")

        state.slot_id = 1
        manager.update(state)
        self.assertEqual(connection.connection_state, "connected")
        self.assertEqual(connection.serving_cell_id, "cell_1")
        self.assertEqual(connection.handover_count, 1)

    def test_observation_exposes_candidate_cells_and_handover_counts(self) -> None:
        state = self._build_state()
        manager = MobilityManager(slot_duration_ms=1.0, rng_seed=1)

        manager.update(state)
        observation = build_observation(state)

        self.assertEqual(observation.handover_pending_ue_count, 1)
        self.assertEqual(len(observation.ue_observations[0].candidate_cell_observations), 2)
        self.assertEqual(
            observation.ue_observations[0].handover_pending_target_cell_id,
            "cell_1",
        )
        top_candidate = observation.ue_observations[0].candidate_cell_observations[0]
        self.assertGreaterEqual(top_candidate.estimated_load, 0.0)
        self.assertLessEqual(top_candidate.estimated_interference_db, top_candidate.rsrp_dbm)
        self.assertNotEqual(top_candidate.steering_score_db, 0.0)
        self.assertIn(top_candidate.rf_band, {"sub6", "mmwave", "midband_high"})
        self.assertGreater(top_candidate.adjacent_channel_leakage_ratio_db, 0.0)
        self.assertGreater(top_candidate.adjacent_channel_selectivity_db, 0.0)
        self.assertGreater(top_candidate.out_of_band_isolation_db, 0.0)

    def test_load_aware_steering_can_avoid_overloaded_target_cell(self) -> None:
        state = self._build_state()
        state.cell.traffic_steering_preference = "balanced"
        state.cell.mobility_bias_db = 0.0
        state.links["ue_0"].x_m = 45.0
        state.links["ue_0"].distance_2d_m = 45.0
        state.links["ue_0"].distance_3d_m = 51.0

        for index in range(1, 5):
            ue_id = f"ue_{index}"
            state.ues.append(
                UeState(
                    ue_id=ue_id,
                    queue_bytes=120000,
                    throughput_mbps=0.0,
                    sinr_db=10.0,
                    radio_quality_db=10.0,
                    bler=0.1,
                    serving_beam_id=0,
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=100.0,
                    priority=1,
                )
            )
            state.connections[ue_id] = ConnectionState(
                ue_id=ue_id,
                serving_cell_id="cell_1",
                serving_beam_id=0,
            )
            state.links[ue_id] = LinkState(
                ue_id=ue_id,
                serving_cell_id="cell_1",
                scenario_type="uma",
                los_state="los",
                x_m=74.0 + float(index),
                y_m=4.0,
                z_m=1.5,
                distance_2d_m=8.0,
                distance_3d_m=24.0,
                velocity_mps=0.0,
                heading_deg=0.0,
                pathloss_db=100.0,
                shadowing_db=1.0,
                doppler_hz=0.0,
                delay_spread_ns=100.0,
                fast_fading_seed=1100 + index,
                spatial_corr_seed=2100 + index,
            )

        manager = MobilityManager(slot_duration_ms=1.0, rng_seed=1)
        manager.update(state)

        connection = state.connections["ue_0"]
        self.assertEqual(connection.serving_cell_id, "cell_0")
        self.assertEqual(connection.connection_state, "connected")
        candidate_scores = {
            candidate.cell_id: candidate.steering_score_db for candidate in connection.candidate_cells
        }
        self.assertGreater(candidate_scores["cell_0"], candidate_scores["cell_1"])

    def test_hysteresis_and_handover_cost_can_block_ping_pong_handover(self) -> None:
        state = self._build_state()
        state.cell.steering_hysteresis_db = 15.0
        state.cell.handover_cost_db = 12.0
        manager = MobilityManager(slot_duration_ms=1.0, rng_seed=1)

        manager.update(state)

        connection = state.connections["ue_0"]
        self.assertEqual(connection.connection_state, "connected")
        self.assertIsNone(connection.handover_pending_target_cell_id)
        self.assertEqual(connection.last_trigger_reason, "trigger_met_but_below_handover_threshold")
        candidate_by_id = {
            candidate.cell_id: candidate for candidate in connection.candidate_cells
        }
        self.assertGreater(candidate_by_id["cell_1"].trigger_margin_db, 0.0)
        self.assertTrue(candidate_by_id["cell_1"].measurement_event_met)
        self.assertLess(
            candidate_by_id["cell_1"].hysteresis_adjusted_score_db,
            candidate_by_id["cell_1"].steering_score_db,
        )

    def test_handover_interruption_state_is_exposed_and_cleared(self) -> None:
        state = self._build_state()
        state.cell.handover_interruption_slots = 2
        manager = MobilityManager(slot_duration_ms=1.0, rng_seed=1)

        manager.update(state)
        connection = state.connections["ue_0"]
        self.assertEqual(connection.connection_state, "handover_pending")

        state.slot_id = 1
        manager.update(state)
        self.assertEqual(connection.connection_state, "handover_interruption")
        self.assertEqual(connection.serving_cell_id, "cell_1")
        self.assertEqual(connection.interruption_remaining_slots, 2)

        observation = build_observation(state)
        self.assertEqual(observation.handover_interruption_ue_count, 1)
        ue_observation = observation.ue_observations[0]
        self.assertEqual(ue_observation.connection_state, "handover_interruption")
        self.assertEqual(ue_observation.handover_interruption_slots, 2)
        self.assertEqual(ue_observation.interruption_remaining_slots, 2)

        state.slot_id = 2
        manager.update(state)
        self.assertEqual(connection.connection_state, "handover_interruption")
        self.assertEqual(connection.interruption_remaining_slots, 1)

        state.slot_id = 3
        manager.update(state)
        self.assertEqual(connection.connection_state, "connected")
        self.assertEqual(connection.interruption_remaining_slots, 0)

    def test_shadowing_resampling_uses_stable_sigma_not_previous_realization(self) -> None:
        state = self._build_state()
        state.links["ue_0"].shadowing_db = 48.0
        state.links["ue_0"].shadowing_sigma_db = 2.0
        state.links["ue_0"].refresh_propagation_views()
        manager = MobilityManager(slot_duration_ms=1.0, rng_seed=7)

        observed_shadowing: list[float] = []
        for slot_id in range(12):
            state.slot_id = slot_id
            manager.update(state)
            observed_shadowing.append(abs(float(state.links["ue_0"].shadowing_db)))

        self.assertLess(max(observed_shadowing), 8.0)


if __name__ == "__main__":
    unittest.main()
