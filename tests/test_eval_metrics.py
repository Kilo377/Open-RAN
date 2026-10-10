import unittest

from experiments.eval_metrics import (
    INTENT_LOSS_DEFINITION_VERSION,
    STABILITY_CONTROL_FIELDS,
    build_run_metric_summary,
    compute_cross_scenario_generalization_gap,
    compute_intent_loss,
    compute_stability_penalty,
)
from models.intent_orchestration import (
    KpiNormalizationMetadata,
    ValidatedIntentTemplate,
    ValidatedKpiTemplateEntry,
)
from models.observation import Observation


class TestEvalMetrics(unittest.TestCase):
    def test_compute_intent_loss_exports_raw_and_normalized_fields(self) -> None:
        template = ValidatedIntentTemplate(
            template_id="template:test",
            goal_id="goal:test",
            intent_id="intent:test",
            kpi_template=[
                ValidatedKpiTemplateEntry(
                    kpi_name="total_throughput_mbps",
                    semantic_type="max",
                    threshold_value=50.0,
                    weight=2.0,
                    normalization=KpiNormalizationMetadata(unit="mbps"),
                ),
                ValidatedKpiTemplateEntry(
                    kpi_name="avg_bler",
                    semantic_type="min",
                    threshold_value=0.05,
                    weight=1.0,
                    normalization=KpiNormalizationMetadata(unit="ratio"),
                ),
            ],
        )
        observation = Observation(
            slot_id=1,
            total_throughput_mbps=40.0,
            avg_bler=0.07,
            avg_sinr_db=10.0,
            total_queue_bytes=1000,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            connected_ue_count=10,
            handover_pending_ue_count=1,
        )

        result = compute_intent_loss(
            observation=observation,
            validated_intent_template=template,
        )

        self.assertIn("intent_loss_raw", result)
        self.assertIn("intent_loss_normalized", result)
        self.assertEqual(result["intent_loss_definition_version"], INTENT_LOSS_DEFINITION_VERSION)
        self.assertEqual(result["intent_loss_component_count"], 2)
        self.assertGreater(result["intent_loss_raw"], 0.0)
        self.assertGreater(result["intent_loss_normalized"], 0.0)

    def test_compute_stability_penalty_reports_control_fields(self) -> None:
        slot_records = [
            {
                "control_trace": {
                    "effective_control_profile": {
                        "cells": {
                            "cell_0": {
                                "scheduler_policy_hint": "balanced",
                                "latency_bias": 1.0,
                                "slice_weight_adjustment": {"embb": 1.0},
                                "beam_candidate_preference": "coverage",
                                "power_cap_dbm": 30.0,
                                "traffic_steering_preference": "edge_offload",
                                "mobility_bias_db": 5.0,
                                "measurement_event_offset_db": 0.8,
                            }
                        }
                    }
                }
            },
            {
                "control_trace": {
                    "effective_control_profile": {
                        "cells": {
                            "cell_0": {
                                "scheduler_policy_hint": "latency_bias",
                                "latency_bias": 1.2,
                                "slice_weight_adjustment": {"embb": 1.1},
                                "beam_candidate_preference": "coverage",
                                "power_cap_dbm": 30.5,
                                "traffic_steering_preference": "edge_offload",
                                "mobility_bias_db": 6.0,
                                "measurement_event_offset_db": 0.9,
                            }
                        }
                    }
                }
            },
        ]

        result = compute_stability_penalty(slot_records)

        self.assertIn("stability_penalty_total", result)
        self.assertEqual(result["stability_control_fields"], STABILITY_CONTROL_FIELDS)
        self.assertGreater(result["stability_penalty_total"], 0.0)

    def test_build_run_metric_summary_includes_core_fields(self) -> None:
        slot_records = [
            {
                "slot_id": 1,
                "observable_metrics": {
                    "throughput_mbps": 40.0,
                    "bler": 0.08,
                    "queue_bytes": 12000,
                    "sinr_db": 10.0,
                    "handover_pending_ue_count": 1,
                },
                "scheduler_trace": {"demand_hol_delay_ms_mean": 5.0},
                "control_trace": {
                    "effective_control_profile": {
                        "cells": {
                            "cell_0": {
                                "scheduler_policy_hint": "balanced",
                                "latency_bias": 1.0,
                                "slice_weight_adjustment": {"embb": 1.0},
                                "beam_candidate_preference": "coverage",
                                "power_cap_dbm": 30.0,
                                "traffic_steering_preference": "edge_offload",
                                "mobility_bias_db": 5.0,
                                "measurement_event_offset_db": 0.8,
                            }
                        }
                    },
                    "rl_health_gate_block_reasons": [],
                    "rl_safe_baseline_fallback": False,
                },
            }
        ]
        epoch_history = [
            {
                "slot_id": 1,
                "intent_loss_raw": 2.0,
                "intent_loss_normalized": 0.1,
                "selected_xapps": ["xapp_a"],
                "exploration_health_gate": {"max_avg_bler": 0.1},
            }
        ]

        summary = build_run_metric_summary(
            slot_records=slot_records,
            orchestration_epoch_history=epoch_history,
        )

        self.assertEqual(summary["intent_loss_normalized"], 0.1)
        self.assertIn("xapp_switching_cost_total", summary)
        self.assertIn("stability_penalty_total", summary)
        self.assertIn("safety_violation_rate", summary)
        self.assertIn("kpi_summary", summary)

    def test_compute_cross_scenario_generalization_gap_uses_fixed_reference(self) -> None:
        case_records = [
            {
                "scenario_name": "benchmark_nominal_cluster",
                "summary_experimental_metrics_intent_loss_mean_normalized": 0.10,
            },
            {
                "scenario_name": "benchmark_edge_coverage",
                "summary_experimental_metrics_intent_loss_mean_normalized": 0.25,
            },
        ]

        gap = compute_cross_scenario_generalization_gap(case_records)

        self.assertEqual(gap["reference_scenario_name"], "benchmark_nominal_cluster")
        self.assertAlmostEqual(gap["mean_gap"], 0.15)
        self.assertAlmostEqual(
            gap["per_scenario_gap"]["benchmark_edge_coverage"],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
