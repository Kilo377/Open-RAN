from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.postprocess_generalization import (
    _parse_case_dir_name,
    postprocess_generalization_artifacts,
)


class PostprocessGeneralizationTest(unittest.TestCase):
    def test_case_dir_alias_maps_back_to_experiment_variant(self) -> None:
        self.assertEqual(
            _parse_case_dir_name(
                "transferr1__oran_medium_high_mobility_interference__"
                "intent-interference-urllc-coexistence__orch__seed1"
            ),
            (
                "transferr1__oran_medium_high_mobility_interference",
                "interference-urllc-coexistence",
                "orchestran_like_resource_aware",
                1,
            ),
        )

    def test_postprocess_builds_aggregates_and_behavior_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiment_dir = Path(tmp_dir)
            manifest_path = experiment_dir / "scenario_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "scenario_id",
                        "split",
                        "topology_class",
                        "load_class",
                        "user_distribution_class",
                        "xapp_pool_class",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario_id": "benchmark_nominal_cluster",
                        "split": "seen",
                        "topology_class": "macro_mmwave_cluster",
                        "load_class": "nominal_load",
                        "user_distribution_class": "mixed_service_cluster",
                        "xapp_pool_class": "full_portfolio",
                    }
                )
                writer.writerow(
                    {
                        "scenario_id": "benchmark_high_mobility",
                        "split": "unseen",
                        "topology_class": "macro_mmwave_cluster",
                        "load_class": "nominal_load",
                        "user_distribution_class": "mobility_shifted_cluster",
                        "xapp_pool_class": "mobility_safe_portfolio",
                    }
                )

            case_payloads = {
                "benchmark_nominal_cluster__full_method__seed1": 0.10,
                "benchmark_high_mobility__full_method__seed1": 0.25,
            }
            for case_dir_name, intent_loss_value in case_payloads.items():
                case_dir = experiment_dir / "cases" / case_dir_name
                case_dir.mkdir(parents=True, exist_ok=True)
                with (case_dir / "final_summary.json").open("w", encoding="utf-8") as json_file:
                    json.dump(
                        {
                            "method_config": {
                                "variant_name": "full_method",
                                "validator_enabled": True,
                                "rl_guidance_enabled": True,
                                "safety_shield_enabled": True,
                                "static_xapp_selection_enabled": False,
                                "reflection_enabled": True,
                            },
                            "final_policy_snapshot_summary": {
                                "active_intent_id": "intent:test",
                            },
                            "experimental_metrics": {
                                "intent_loss_mean_raw": intent_loss_value,
                                "intent_loss_mean_normalized": intent_loss_value,
                                "safety_violation_rate": 0.02,
                                "stability_penalty_mean": 0.10,
                                "xapp_switching_cost_mean": 1.0,
                                "kpi_summary": {
                                    "mean_observable_throughput_mbps": 42.0,
                                    "final_observable_bler": 0.08,
                                },
                            },
                        },
                        json_file,
                        indent=2,
                        ensure_ascii=False,
                    )
                with (case_dir / "orchestration_epoch_history.json").open("w", encoding="utf-8") as json_file:
                    json.dump(
                        [
                            {
                                "epoch_id": 1,
                                "slot_id": 10,
                                "intent_changed_this_epoch": True,
                                "translation_refreshed_this_epoch": True,
                                "translation_query_id": "qctx:test:abc123",
                                "translation_query_encoding_version": "translation_context_v2_embedding",
                                "translation_query_retriever_backend_name": "hashed_feature_embedding_v1",
                                "translation_query_embedding_dim": 48,
                                "translation_context_signature": {"queue_class": "medium"},
                                "translation_retrieved_context_count": 1,
                                "translation_retrieved_context_source_ids": ["kpi:throughput"],
                                "translation_retrieved_context_summary": {
                                    "retrieved_item_count": 1,
                                    "recommended_kpis": ["total_throughput_mbps"],
                                },
                                "selected_xapps": ["massive_mimo_beamformer_xapp_capacity_push"],
                                "selection_query_id": "qsel:goal:test:throughput:slot10",
                                "selection_query_slot_id": 10,
                                "selection_query_encoding_version": "structured_retrieval_v2_embedding",
                                "selection_query_retriever_backend_name": "hashed_feature_embedding_v1",
                                "selection_query_embedding_dim": 48,
                                "selection_query_kpis": ["total_throughput_mbps"],
                                "selection_query_target_profile": "throughput",
                                "selection_query_previous_xapp_categories": ["traffic_steering_xapp"],
                                "guidance_id": "guidance:test",
                                "guidance_source_epoch": 1,
                                "guidance_fallback_used": False,
                                "selected_rule_ids": ["rule:hard_safety_envelope"],
                                "hard_safety_envelope_count": 2,
                                "hard_safety_envelope_fields": ["power_cap_dbm"],
                                "guided_action_region_count": 2,
                                "guided_action_region_fields": ["power_cap_dbm"],
                                "forbidden_combination_count": 1,
                                "exploration_health_gate": {"max_avg_bler": 0.12},
                                "validator_enabled": True,
                                "rl_guidance_enabled": True,
                                "safety_shield_enabled": True,
                                "static_xapp_selection_enabled": False,
                                "selection_plan_id": "selection:test",
                                "candidate_plan_count": 1,
                                "selected_candidate_id": "candidate:beamformer",
                                "selected_candidate_score_version": "j_sel_surrogate_v3_predictor_objective",
                                "selected_candidate_score_direction": "minimize",
                    "selected_candidate_predictor_name": "memory_calibrated_surrogate_predictor_v1",
                                "selected_candidate_total_score": 0.12,
                                "selected_candidate_predicted_intent_loss": 0.05,
                                "selected_candidate_conflict_cost": 0.0,
                                "selected_candidate_stability_risk": 0.0,
                                "selected_candidate_resource_pressure": 0.02,
                                "selected_candidate_eta_profile_name": "eta:throughput_bias",
                                "selected_candidate_chi_profile_name": "chi:capacity_weighted_merge",
                                "selected_candidate_parameter_bound_count": 3,
                                "selected_candidate_feasible": True,
                                "selected_candidate_projection_applied": True,
                                "selected_candidate_total_cpu_cost": 2.2,
                                "selected_candidate_total_mem_cost": 1.8,
                                "selected_candidate_rationale": "intent_loss=0.050, alignment=0.950, memory_support=0.500",
                                "eta_profile_name": "eta:throughput_bias",
                                "eta_goal_focus": "throughput",
                                "eta_payload": {"profile_name": "eta:throughput_bias"},
                                "composition_profile_name": "chi:capacity_weighted_merge",
                                "composition_mode": "parameter_tuning",
                                "arbitration_mode": "weighted_merge",
                                "override_order": [],
                                "chi_defaulted": False,
                                "chi_notes": ["capacity-centric merge profile"],
                                "feasible": True,
                                "projection_applied": True,
                                "total_cpu_cost": 2.2,
                                "total_mem_cost": 1.8,
                                "max_cpu_cost": 6.0,
                                "max_mem_cost": 5.5,
                                "dependency_pairs": ["urllc_guard_xapp->latency_scheduler_xapp"],
                                "mutex_pairs": ["power_saving_xapp<->massive_mimo_beamformer_xapp"],
                                "projection_notes": ["added dependency latency_scheduler_xapp_balanced"],
                                "last_retrieved_success_case_ids": ["success:1"],
                                "last_retrieved_failure_case_ids": [],
                                "last_retrieved_success_traces": [
                                    {
                                        "case_id": "success:1",
                                        "retrieval_rank": 1,
                                        "feature_similarity": 0.72,
                                        "retrieval_score": 0.75,
                                        "retrieval_confidence": 0.66,
                                        "memory_retention_score": 0.71,
                                    }
                                ],
                                "last_retrieved_failure_traces": [],
                                "goal_id": "goal:test",
                                "intent_loss_raw": intent_loss_value,
                                "intent_loss_normalized": intent_loss_value,
                            }
                        ],
                        json_file,
                        indent=2,
                        ensure_ascii=False,
                    )

            artifacts = postprocess_generalization_artifacts(
                experiment_dir=experiment_dir,
                manifest_path=manifest_path,
                reference_scenario_id="benchmark_nominal_cluster",
                variant_names=["full_method"],
            )

            self.assertEqual(len(artifacts["case_records"]), 2)
            self.assertEqual(len(artifacts["behavior_rows"]), 2)
            self.assertAlmostEqual(
                artifacts["aggregated_rows"][0]["transfer_gap_intent_loss_normalized"],
                0.15,
            )
            self.assertEqual(
                artifacts["behavior_rows"][0]["selection_query_id"],
                "qsel:goal:test:throughput:slot10",
            )
            self.assertEqual(artifacts["behavior_rows"][0]["selection_query_slot_id"], 10)
            self.assertEqual(
                artifacts["behavior_rows"][0]["selection_query_encoding_version"],
                "structured_retrieval_v2_embedding",
            )
            self.assertEqual(
                artifacts["behavior_rows"][0]["selection_query_retriever_backend_name"],
                "hashed_feature_embedding_v1",
            )
            self.assertEqual(artifacts["behavior_rows"][0]["selection_query_embedding_dim"], 48)
            self.assertEqual(
                artifacts["behavior_rows"][0]["translation_query_id"],
                "qctx:test:abc123",
            )
            self.assertEqual(
                artifacts["behavior_rows"][0]["translation_query_encoding_version"],
                "translation_context_v2_embedding",
            )
            self.assertEqual(
                artifacts["behavior_rows"][0]["translation_query_retriever_backend_name"],
                "hashed_feature_embedding_v1",
            )
            self.assertEqual(artifacts["behavior_rows"][0]["translation_query_embedding_dim"], 48)
            self.assertEqual(artifacts["behavior_rows"][0]["candidate_plan_count"], 1)
            self.assertEqual(
                artifacts["behavior_rows"][0]["selected_candidate_score_direction"],
                "minimize",
            )
            self.assertEqual(
                artifacts["behavior_rows"][0]["selected_candidate_predictor_name"],
            "memory_calibrated_surrogate_predictor_v1",
            )
            self.assertEqual(
                artifacts["behavior_rows"][0]["selected_candidate_eta_profile_name"],
                "eta:throughput_bias",
            )
            self.assertEqual(
                artifacts["behavior_rows"][0]["composition_profile_name"],
                "chi:capacity_weighted_merge",
            )
            self.assertEqual(
                artifacts["behavior_rows"][0]["projection_applied"],
                True,
            )
            self.assertIn("success:1", artifacts["behavior_rows"][0]["last_retrieved_success_case_ids"])
            self.assertTrue((experiment_dir / "aggregated_results.csv").exists())
            self.assertTrue((experiment_dir / "orchestration_behavior.csv").exists())


if __name__ == "__main__":
    unittest.main()
