import unittest
from types import SimpleNamespace
from typing import Any, cast

from core.simulator import Simulator
from models.policy import PolicySnapshot
from state.measurement_state import MeasurementState, UeMeasurement


class TestFinalSummaryMeasurementStats(unittest.TestCase):
    def test_build_final_summary_reports_trace_stats_and_final_per_ue_snapshot(self) -> None:
        simulator = cast(Any, Simulator.__new__(Simulator))
        simulator.config = SimpleNamespace(total_slots=3, slot_duration_ms=1.0)
        simulator.slot_records = [
            {
                "slot_id": 1,
                "sim_time_ms": 1.0,
                "true_metrics": {"queue_bytes": 500},
                "observable_metrics": {"queue_bytes": 520},
                "measurement_trace": {
                    "age_slots": 0,
                    "queue_age_slots_mean": 1.0,
                    "sinr_effective_sample_count_mean": 2.0,
                    "bler_sample_count_mean": 1.0,
                    "throughput_sample_count_mean": 2.0,
                },
                "control_trace": {
                    "input_measurement_age_slots": 1,
                    "input_queue_age_slots_mean": 1.0,
                    "input_sinr_effective_sample_count_mean": 2.0,
                    "input_bler_sample_count_mean": 1.0,
                    "input_throughput_sample_count_mean": 2.0,
                },
                "phy_comparison_trace": {
                    "compared_ue_count": 1,
                    "ack_match_rate": 1.0,
                    "mean_abs_sinr_gap_db": 0.8,
                    "mean_abs_post_eq_sinr_gap_db": 0.7,
                    "max_abs_post_eq_sinr_gap_db": 0.7,
                    "mean_abs_bler_gap": 0.05,
                    "mean_abs_throughput_gap_mbps": 0.2,
                    "mean_abs_tb_size_gap_bytes": 10.0,
                    "mean_abs_spatial_multiplexing_gain_gap_db": 0.3,
                    "spatial_mode_match_rate": 1.0,
                    "per_ue_diffs": [{"ue_id": "ue_0", "post_eq_sinr_gap_db": 0.7}],
                    "spatial_group_summaries": [{"spatial_group_id": 2, "member_count": 1}],
                },
            },
            {
                "slot_id": 2,
                "sim_time_ms": 2.0,
                "true_metrics": {"queue_bytes": 420},
                "observable_metrics": {"queue_bytes": 470},
                "measurement_trace": {
                    "age_slots": 2,
                    "queue_age_slots_mean": 3.0,
                    "sinr_effective_sample_count_mean": 4.0,
                    "bler_sample_count_mean": 3.0,
                    "throughput_sample_count_mean": 5.0,
                },
                "control_trace": {
                    "input_measurement_age_slots": 3,
                    "input_queue_age_slots_mean": 2.0,
                    "input_sinr_effective_sample_count_mean": 3.0,
                    "input_bler_sample_count_mean": 2.0,
                    "input_throughput_sample_count_mean": 4.0,
                },
                "phy_comparison_trace": {
                    "compared_ue_count": 1,
                    "ack_match_rate": 0.0,
                    "mean_abs_sinr_gap_db": 1.2,
                    "mean_abs_post_eq_sinr_gap_db": 1.0,
                    "max_abs_post_eq_sinr_gap_db": 1.0,
                    "mean_abs_bler_gap": 0.1,
                    "mean_abs_throughput_gap_mbps": 0.5,
                    "mean_abs_tb_size_gap_bytes": 30.0,
                    "mean_abs_spatial_multiplexing_gain_gap_db": 0.6,
                    "spatial_mode_match_rate": 0.0,
                    "per_ue_diffs": [{"ue_id": "ue_0", "post_eq_sinr_gap_db": 1.0}],
                    "spatial_group_summaries": [{"spatial_group_id": 2, "member_count": 1}],
                },
            },
        ]
        simulator.state = SimpleNamespace(
            cell=SimpleNamespace(
                cell_id="cell_0",
                tx_power_dbm=30.0,
                current_codebook_id=0,
                current_beam_profile_id=0,
            ),
            scheduler=SimpleNamespace(
                scheduler_mode="round_robin",
                scheduled_ue_ids=["ue_0"],
            ),
            ues=[
                SimpleNamespace(
                    ue_id="ue_0",
                    queue_bytes=420,
                    throughput_mbps=1.0,
                    sinr_db=5.0,
                    bler=0.1,
                    serving_beam_id=0,
                    slice_id="embb",
                    traffic_type="video",
                    latency_budget_ms=50.0,
                    priority=1,
                    radio_quality_db=4.5,
                    harq=SimpleNamespace(pending_retx_bytes=0, last_decision="ACK"),
                )
            ],
            measurement=MeasurementState(
                produced_at_slot=2,
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
                        queue_bytes=470,
                        throughput_mbps=0.8,
                        sinr_db=4.0,
                        bler=0.2,
                        is_scheduled=True,
                        produced_at_slot=2,
                        queue_age_slots=2,
                        sinr_window_len=8,
                        sinr_window_slots=8,
                        sinr_effective_sample_count=4,
                        bler_window_tbs=20,
                        bler_window_sample_count=3,
                        bler_sample_count=3,
                        throughput_window_slots=5,
                        throughput_granularity_slots=10,
                    )
                },
            ),
            snapshots=[
                {
                    "slot_id": 2,
                    "ues": [
                        {
                            "ue_id": "ue_0",
                            "true_state": {"queue_bytes": 420},
                            "observable_state": {"queue_bytes": 470},
                            "measurement_trace": {"queue_age_slots": 2},
                        }
                    ],
                }
            ],
        )
        simulator.controller_manager = SimpleNamespace(
            get_orchestration_epoch_history=lambda: [
                {
                    "epoch_id": 1,
                    "slot_id": 1,
                    "active_intent_id": "intent:test",
                    "intent_changed_this_epoch": True,
                    "translation_refreshed_this_epoch": True,
                    "selected_xapps": ["xapp_a"],
                    "guidance_id": "guidance:test",
                    "validator_enabled": True,
                    "rl_guidance_enabled": True,
                    "safety_shield_enabled": True,
                    "static_xapp_selection_enabled": False,
                    "intent_loss_raw": 2.0,
                    "intent_loss_normalized": 0.1,
                    "intent_loss_definition_version": "v2_paper_intent_loss",
                    "exploration_health_gate": {"max_avg_bler": 0.1},
                }
            ],
            policy_store=SimpleNamespace(get_snapshot=lambda: PolicySnapshot()),
        )

        summary = simulator._build_final_summary()

        self.assertIn("measurement_trace_stats", summary)
        self.assertIn("control_input_measurement_trace_stats", summary)
        self.assertIn("phy_comparison_trace_stats", summary)
        self.assertEqual(summary["measurement_trace_stats"]["age_slots"]["mean"], 1.0)
        self.assertEqual(
            summary["control_input_measurement_trace_stats"]["input_measurement_age_slots"]["max"],
            3.0,
        )
        self.assertEqual(summary["phy_comparison_trace_stats"]["mean_abs_sinr_gap_db"]["mean"], 1.0)
        self.assertEqual(summary["phy_comparison_trace_stats"]["max_abs_post_eq_sinr_gap_db"]["max"], 1.0)
        self.assertAlmostEqual(
            summary["phy_comparison_trace_stats"]["mean_abs_spatial_multiplexing_gain_gap_db"]["mean"],
            0.45,
        )
        self.assertEqual(summary["final_phy_comparison_trace"]["per_ue_diffs"][0]["ue_id"], "ue_0")
        self.assertEqual(len(summary["final_per_ue_snapshot"]), 1)
        self.assertEqual(summary["final_per_ue_snapshot"][0]["ue_id"], "ue_0")
        self.assertIn("experimental_metrics", summary)
        self.assertEqual(summary["experimental_metrics"]["intent_loss_raw"], 2.0)
        self.assertIn("method_config", summary)
        self.assertIn("orchestration_epoch_history_summary", summary)


if __name__ == "__main__":
    unittest.main()
