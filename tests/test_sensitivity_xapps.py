import unittest
from pathlib import Path

from core.config_loader import load_yaml_file
from core.factory import build_controller_registry
from models.observation import (
    CandidateCellObservation,
    CellObservation,
    Observation,
    UeObservation,
)
from models.policy import PolicySnapshot
from ric.near_rt.massive_mimo_beamformer_xapp import MassiveMIMOBeamformerXApp
from ric.near_rt.mobility_predictor_xapp import MobilityPredictorXApp
from ric.near_rt.power_saving_xapp import PowerSavingXApp
from ric.near_rt.ran_slicing_manager_xapp import RANSlicingManagerXApp
from ric.near_rt.spectrum_optimizer_xapp import SpectrumOptimizerXApp
from ric.near_rt.traffic_steering_xapp import TrafficSteeringXApp
from ric.near_rt.urllc_guard_xapp import URLLCGuardXApp
from ric.near_rt.latency_scheduler_xapp import LatencySchedulerXApp


def _candidate(
    cell_id: str,
    *,
    steering_score_db: float,
    is_serving: bool = False,
) -> CandidateCellObservation:
    return CandidateCellObservation(
        cell_id=cell_id,
        steering_score_db=steering_score_db,
        signal_margin_db=steering_score_db,
        rsrp_dbm=-85.0 + steering_score_db,
        estimated_interference_db=-100.0,
        is_serving=is_serving,
    )


def _build_observation() -> Observation:
    return Observation(
        slot_id=8,
        avg_sinr_db=6.5,
        avg_bler=0.12,
        total_queue_bytes=11800,
        total_throughput_mbps=18.0,
        cell_id="cell_0",
        current_codebook_id=0,
        current_beam_profile_id=0,
        tx_power_dbm=30.0,
        scheduler_mode="round_robin",
        scheduled_ue_ids=["ue_urlcc", "ue_embb"],
        num_scheduled_ues=2,
        cell_observations=[
            CellObservation(
                cell_id="cell_0",
                resource_grid_utilization=0.86,
                total_queue_bytes=9600,
                connected_ue_count=5,
                inter_cell_interference_penalty_db_mean=3.2,
                scheduler_mode="round_robin",
            ),
            CellObservation(
                cell_id="cell_1",
                resource_grid_utilization=0.22,
                total_queue_bytes=800,
                connected_ue_count=1,
                inter_cell_interference_penalty_db_mean=0.7,
                scheduler_mode="round_robin",
            ),
            CellObservation(
                cell_id="cell_2",
                resource_grid_utilization=0.54,
                total_queue_bytes=1400,
                connected_ue_count=2,
                inter_cell_interference_penalty_db_mean=1.4,
                scheduler_mode="round_robin",
            ),
        ],
        ue_observations=[
            UeObservation(
                ue_id="ue_urlcc",
                queue_bytes=2200,
                throughput_mbps=2.0,
                sinr_db=3.5,
                bler=0.18,
                serving_beam_id=0,
                is_scheduled=True,
                slice_id="urllc",
                traffic_type="control",
                latency_budget_ms=10.0,
                priority=5,
                serving_cell_id="cell_0",
                demand_bytes=2300,
                demand_hol_delay_ms=14.0,
                demand_harq_urgency_score=0.9,
                demand_channel_quality_confidence=0.35,
                measurement_age_slots=4,
                grant_cqi_validity_age_slots=4,
                grant_channel_quality_confidence=0.32,
                beamforming_gain_db=2.5,
                post_eq_sinr_db=4.0,
                precoder_match_score=0.42,
                cqi=5,
                inter_cell_interference_penalty_db=2.5,
                candidate_cell_observations=[
                    _candidate("cell_0", steering_score_db=2.0, is_serving=True),
                    _candidate("cell_1", steering_score_db=8.5),
                    _candidate("cell_2", steering_score_db=4.0),
                ],
            ),
            UeObservation(
                ue_id="ue_embb",
                queue_bytes=7200,
                throughput_mbps=9.0,
                sinr_db=11.5,
                bler=0.05,
                serving_beam_id=1,
                is_scheduled=True,
                slice_id="embb",
                traffic_type="video",
                latency_budget_ms=80.0,
                priority=2,
                serving_cell_id="cell_0",
                demand_bytes=7600,
                demand_hol_delay_ms=35.0,
                demand_harq_urgency_score=0.2,
                demand_channel_quality_confidence=0.82,
                measurement_age_slots=1,
                grant_cqi_validity_age_slots=1,
                grant_channel_quality_confidence=0.84,
                beamforming_gain_db=8.5,
                post_eq_sinr_db=12.5,
                precoder_match_score=0.88,
                cqi=11,
                inter_cell_interference_penalty_db=1.0,
                candidate_cell_observations=[
                    _candidate("cell_0", steering_score_db=7.0, is_serving=True),
                    _candidate("cell_1", steering_score_db=5.0),
                ],
            ),
            UeObservation(
                ue_id="ue_iot",
                queue_bytes=2400,
                throughput_mbps=1.2,
                sinr_db=8.0,
                bler=0.08,
                serving_beam_id=2,
                is_scheduled=False,
                slice_id="mmtc",
                traffic_type="iot",
                latency_budget_ms=200.0,
                priority=1,
                serving_cell_id="cell_2",
                demand_bytes=1900,
                demand_hol_delay_ms=25.0,
                demand_harq_urgency_score=0.1,
                demand_channel_quality_confidence=0.60,
                measurement_age_slots=2,
                grant_cqi_validity_age_slots=2,
                grant_channel_quality_confidence=0.62,
                beamforming_gain_db=5.5,
                post_eq_sinr_db=8.2,
                precoder_match_score=0.64,
                cqi=8,
                inter_cell_interference_penalty_db=0.8,
                candidate_cell_observations=[
                    _candidate("cell_2", steering_score_db=4.2, is_serving=True),
                    _candidate("cell_1", steering_score_db=3.5),
                ],
            ),
        ],
    )


class TestSensitivityXApps(unittest.TestCase):
    def test_factory_builds_sensitivity_bundle(self) -> None:
        ric_config = load_yaml_file(str(Path(__file__).resolve().parents[1]/"configs/ric_sensitivity.yaml"))
        registry = build_controller_registry(ric_config)

        self.assertEqual(len(registry.xapps), 8)
        self.assertEqual(
            sorted(xapp.name for xapp in registry.xapps),
            sorted(
                [
                    "mobility_predictor_xapp",
                    "traffic_steering_xapp",
                    "power_saving_xapp",
                    "spectrum_optimizer_xapp",
                    "latency_scheduler_xapp",
                    "massive_mimo_beamformer_xapp",
                    "ran_slicing_manager_xapp",
                    "urllc_guard_xapp",
                ]
            ),
        )

    def test_sensitivity_xapps_emit_control_surfaces(self) -> None:
        observation = _build_observation()
        policy_snapshot = PolicySnapshot(
            preferred_power_cap_dbm=30.0,
            preferred_beam_candidate_preference="balanced",
            preferred_scheduler_policy_hint="slice_bias",
            preferred_latency_bias=1.0,
            preferred_slice_weight_adjustment={"urllc": 1.0, "embb": 1.0, "mmtc": 1.0},
            preferred_mobility_bias_db=1.5,
            preferred_measurement_event_offset_db=2.0,
            preferred_traffic_steering_preference="balanced",
        )

        mobility = MobilityPredictorXApp(risk_threshold=0.30).run(observation, policy_snapshot)
        steering = TrafficSteeringXApp().run(observation, policy_snapshot)
        power = PowerSavingXApp().run(observation, policy_snapshot)
        spectrum = SpectrumOptimizerXApp().run(observation, policy_snapshot)
        latency = LatencySchedulerXApp().run(observation, policy_snapshot)
        mimo = MassiveMIMOBeamformerXApp().run(observation, policy_snapshot)
        slicing = RANSlicingManagerXApp().run(observation, policy_snapshot)
        urllc = URLLCGuardXApp().run(observation, policy_snapshot)

        self.assertIsNotNone(mobility.config_action.mobility_bias_db)
        self.assertEqual(mobility.config_action.apply_scope, "cluster")
        self.assertTrue(len(mobility.policy_constraints) > 0)

        self.assertIn(
            steering.config_action.traffic_steering_preference,
            {"edge_offload", "stay_put", "balanced"},
        )
        self.assertTrue(len(steering.policy_constraints) > 0)

        self.assertEqual(power.config_action.target_cell_id, "cell_1")
        self.assertIsNotNone(power.config_action.power_cap_dbm)

        self.assertEqual(spectrum.config_action.target_cell_id, "cell_0")
        self.assertIsNotNone(spectrum.config_action.scheduler_policy_hint)

        self.assertEqual(latency.config_action.scheduler_policy_hint, "slice_bias")
        self.assertTrue(len(latency.policy_constraints) > 0)

        self.assertEqual(mimo.config_action.target_cell_id, "cell_0")
        self.assertIn(mimo.config_action.beam_candidate_preference, {"coverage", "balanced", "capacity"})

        self.assertEqual(slicing.config_action.scheduler_policy_hint, "slice_bias")
        self.assertTrue("embb" in slicing.config_action.slice_weight_adjustment)

        self.assertEqual(urllc.config_action.target_cell_id, "cell_0")
        self.assertEqual(urllc.config_action.scheduler_policy_hint, "latency_bias")
        self.assertIsNotNone(urllc.config_action.latency_bias)
        self.assertTrue(len(urllc.policy_constraints) > 0)

    def test_power_saving_emits_mild_action_under_moderate_headroom(self) -> None:
        observation = Observation(
            slot_id=12,
            avg_sinr_db=7.0,
            avg_bler=0.22,
            total_queue_bytes=22000,
            total_throughput_mbps=12.0,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            cell_observations=[
                CellObservation(
                    cell_id="cell_0",
                    resource_grid_utilization=0.91,
                    total_queue_bytes=18000,
                    connected_ue_count=8,
                ),
                CellObservation(
                    cell_id="cell_1",
                    resource_grid_utilization=0.58,
                    total_queue_bytes=9000,
                    connected_ue_count=3,
                ),
            ],
        )
        policy_snapshot = PolicySnapshot(
            preferred_power_cap_dbm=29.5,
            preferred_beam_candidate_preference="capacity",
        )

        decision = PowerSavingXApp().run(observation, policy_snapshot)

        self.assertEqual(decision.config_action.target_cell_id, "cell_1")
        self.assertIsNotNone(decision.config_action.power_cap_dbm)
        power_cap_dbm = decision.config_action.power_cap_dbm
        assert power_cap_dbm is not None
        self.assertLess(float(power_cap_dbm), 29.5)
        self.assertEqual(decision.config_action.beam_candidate_preference, "balanced")
        self.assertEqual(
            decision.policy_constraints[0].params.get("energy_mode"),
            "mild_saving",
        )

    def test_spectrum_optimizer_emits_moderate_action_before_full_congestion(self) -> None:
        observation = Observation(
            slot_id=14,
            avg_sinr_db=9.0,
            avg_bler=0.18,
            total_queue_bytes=26000,
            total_throughput_mbps=14.0,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            cell_observations=[
                CellObservation(
                    cell_id="cell_0",
                    resource_grid_utilization=0.52,
                    total_queue_bytes=16000,
                    connected_ue_count=6,
                    inter_cell_interference_penalty_db_mean=1.8,
                ),
                CellObservation(
                    cell_id="cell_1",
                    resource_grid_utilization=0.35,
                    total_queue_bytes=5000,
                    connected_ue_count=2,
                    inter_cell_interference_penalty_db_mean=0.7,
                ),
            ],
            ue_observations=[
                UeObservation(
                    ue_id="ue_embb_0",
                    queue_bytes=9000,
                    throughput_mbps=8.0,
                    sinr_db=8.0,
                    bler=0.20,
                    serving_beam_id=0,
                    is_scheduled=True,
                    serving_cell_id="cell_0",
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=90.0,
                    priority=2,
                    demand_bytes=9200,
                ),
                UeObservation(
                    ue_id="ue_embb_1",
                    queue_bytes=6000,
                    throughput_mbps=6.0,
                    sinr_db=9.5,
                    bler=0.14,
                    serving_beam_id=1,
                    is_scheduled=True,
                    serving_cell_id="cell_0",
                    slice_id="embb",
                    traffic_type="broadband",
                    latency_budget_ms=110.0,
                    priority=2,
                    demand_bytes=6500,
                ),
                UeObservation(
                    ue_id="ue_urlcc",
                    queue_bytes=1000,
                    throughput_mbps=1.5,
                    sinr_db=6.0,
                    bler=0.12,
                    serving_beam_id=2,
                    is_scheduled=False,
                    serving_cell_id="cell_0",
                    slice_id="urllc",
                    traffic_type="control",
                    latency_budget_ms=12.0,
                    priority=5,
                    demand_bytes=1200,
                ),
            ],
        )
        policy_snapshot = PolicySnapshot(
            preferred_scheduler_policy_hint="slice_bias",
            preferred_slice_weight_adjustment={"urllc": 1.0, "embb": 1.0, "mmtc": 1.0},
        )

        decision = SpectrumOptimizerXApp().run(observation, policy_snapshot)

        self.assertEqual(decision.config_action.target_cell_id, "cell_0")
        self.assertIsNotNone(decision.config_action.scheduler_policy_hint)
        self.assertEqual(
            decision.policy_constraints[0].params.get("optimizer_mode"),
            "moderate_pressure",
        )


if __name__ == "__main__":
    unittest.main()
