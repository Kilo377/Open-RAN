import unittest

from models.observation import Observation
from models.policy import PolicySnapshot
from ric.non_rt.intent_goal_translator_rapp import IntentGoalTranslatorRApp


class TestIntentGoalTranslatorRApp(unittest.TestCase):
    def _make_observation(
        self,
        *,
        slot_id: int,
        throughput_mbps: float,
        avg_bler: float = 0.05,
        queue_bytes: int = 5000,
    ) -> Observation:
        return Observation(
            slot_id=slot_id,
            avg_sinr_db=15.0,
            avg_bler=avg_bler,
            total_queue_bytes=queue_bytes,
            total_throughput_mbps=throughput_mbps,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            scheduled_ue_ids=[],
            num_scheduled_ues=0,
            ue_observations=[],
        )

    def _make_snapshot(self) -> PolicySnapshot:
        return PolicySnapshot(
            active_intent_text="increase throughput by 10 percent while keeping reliability stable",
            active_intent_id="intent:test_refresh",
            intent_changed_this_epoch=False,
        )

    def test_translation_refreshes_even_when_active_intent_is_unchanged(self) -> None:
        app = IntentGoalTranslatorRApp(intent_text="", llm_enabled=True)

        def _fake_generate_raw_candidate(
            observation: Observation,
            intent_text: str,
        ) -> dict[str, object]:
            return {
                "goal_id": "goal:test_refresh",
                "summary": f"refresh_from_slot_{observation.slot_id}",
                "horizon_slots": 12,
                "risk_posture": "balanced",
                "target_entities": ["cluster"],
                "kpi_targets": [
                    {
                        "kpi_name": "total_throughput_mbps",
                        "objective": "maximize",
                        "target_value": float(observation.total_throughput_mbps) + 5.0,
                        "weight": 1.2,
                        "unit": "mbps",
                        "priority": 3,
                    }
                ],
            }

        app._generate_raw_candidate = _fake_generate_raw_candidate  # type: ignore[method-assign]

        first_update = app.run(
            observation=self._make_observation(slot_id=10, throughput_mbps=20.0),
            policy_snapshot=self._make_snapshot(),
        )
        second_update = app.run(
            observation=self._make_observation(slot_id=20, throughput_mbps=30.0),
            policy_snapshot=self._make_snapshot(),
        )

        self.assertTrue(bool(first_update.translation_refreshed_this_epoch))
        self.assertTrue(bool(second_update.translation_refreshed_this_epoch))
        self.assertEqual(first_update.active_intent_id, "intent:test_refresh")
        self.assertEqual(second_update.active_intent_id, "intent:test_refresh")
        self.assertEqual(first_update.structured_goal.summary, "refresh_from_slot_10")
        self.assertEqual(second_update.structured_goal.summary, "refresh_from_slot_20")
        self.assertNotEqual(
            first_update.structured_goal.kpi_targets[0].target_value,
            second_update.structured_goal.kpi_targets[0].target_value,
        )
        self.assertIsNotNone(first_update.validated_intent_template)
        self.assertTrue(first_update.validated_intent_template.validation_report.schema_ok)
        self.assertIsNotNone(first_update.validated_intent_template.translation_query_key)
        self.assertEqual(
            first_update.validated_intent_template.translation_query_key.encoding_version,
            "translation_context_v2_embedding",
        )
        self.assertGreater(
            len(first_update.validated_intent_template.translation_query_key.embedding_vector),
            0,
        )
        self.assertGreater(
            int(
                first_update.validated_intent_template.retrieved_context_summary.get(
                    "retrieved_item_count",
                    0,
                )
                or 0
            ),
            0,
        )
        self.assertGreater(
            len(first_update.validated_intent_template.retrieved_context_traces),
            0,
        )
        self.assertGreater(
            len(first_update.validated_intent_template.translation_query_key.encoded_features),
            0,
        )
        self.assertGreaterEqual(
            float(
                getattr(
                    first_update.validated_intent_template.retrieved_context_traces[0],
                    "feature_similarity",
                    0.0,
                )
                or 0.0
            ),
            0.0,
        )
        self.assertGreater(
            float(
                getattr(
                    first_update.validated_intent_template.retrieved_context_traces[0],
                    "retrieval_confidence",
                    0.0,
                )
                or 0.0
            ),
            0.0,
        )
        self.assertGreater(
            int(
                first_update.validated_intent_template.retrieved_context_summary.get(
                    "embedding_dimension",
                    0,
                )
                or 0
            ),
            0,
        )
        self.assertEqual(
            first_update.validated_intent_template.kpi_template[0].threshold_value,
            25.0,
        )
        self.assertEqual(
            second_update.validated_intent_template.kpi_template[0].threshold_value,
            35.0,
        )

    def test_empty_template_triggers_explicit_fallback(self) -> None:
        app = IntentGoalTranslatorRApp(intent_text="", llm_enabled=True)
        app._generate_raw_candidate = lambda observation, intent_text: {  # type: ignore[method-assign]
            "goal_id": "goal:empty_candidate",
            "summary": "empty",
            "kpi_targets": [],
        }

        update = app.run(
            observation=self._make_observation(slot_id=10, throughput_mbps=20.0),
            policy_snapshot=self._make_snapshot(),
        )

        self.assertIsNotNone(update.validated_intent_template)
        self.assertTrue(update.validated_intent_template.fallback_report.fallback_used)
        self.assertEqual(
            update.validated_intent_template.fallback_report.fallback_reason,
            "empty candidate",
        )
        self.assertEqual(
            update.validated_intent_template.fallback_output["fallback_reason"],
            "empty candidate",
        )

    def test_illegal_kpi_candidate_records_rejection_and_fallback(self) -> None:
        app = IntentGoalTranslatorRApp(intent_text="", llm_enabled=True)
        app._generate_raw_candidate = lambda observation, intent_text: {  # type: ignore[method-assign]
            "goal_id": "goal:illegal_kpi",
            "summary": "illegal",
            "kpi_targets": [
                {
                    "kpi_name": "foobar_metric",
                    "objective": "maximize",
                    "target_value": 10.0,
                    "weight": 1.0,
                    "unit": "n/a",
                }
            ],
        }

        update = app.run(
            observation=self._make_observation(slot_id=11, throughput_mbps=18.0),
            policy_snapshot=self._make_snapshot(),
        )

        template = update.validated_intent_template
        self.assertIsNotNone(template)
        self.assertIn("foobar_metric", template.validation_report.rejected_kpis)
        self.assertFalse(template.validation_report.admissibility_ok)
        self.assertTrue(template.fallback_report.fallback_used)
        self.assertEqual(template.fallback_report.fallback_reason, "all KPIs inadmissible")

    def test_out_of_range_target_is_projected_without_fallback(self) -> None:
        app = IntentGoalTranslatorRApp(intent_text="", llm_enabled=True)
        app._generate_raw_candidate = lambda observation, intent_text: {  # type: ignore[method-assign]
            "goal_id": "goal:range_projection",
            "summary": "range_projection",
            "horizon_slots": 12,
            "kpi_targets": [
                {
                    "kpi_name": "total_throughput_mbps",
                    "objective": "maximize",
                    "target_value": 1_000_000.0,
                    "weight": 1.0,
                    "unit": "mbps",
                }
            ],
        }

        observation = self._make_observation(slot_id=12, throughput_mbps=40.0)
        update = app.run(
            observation=observation,
            policy_snapshot=self._make_snapshot(),
        )

        template = update.validated_intent_template
        self.assertIsNotNone(template)
        self.assertFalse(template.fallback_report.fallback_used)
        self.assertTrue(template.projection_report.projected)
        self.assertFalse(template.validation_report.range_ok)
        self.assertIn("kpi_targets[0].target_value", template.projection_report.repaired_fields)
        self.assertLess(
            update.structured_goal.kpi_targets[0].target_value,
            1_000_000.0,
        )
        self.assertEqual(
            template.projector_output["repair_actions"][0]["action"],
            "clip_to_feasible_range",
        )

    def test_no_validator_variant_uses_minimal_parse_without_formal_repair_chain(self) -> None:
        app = IntentGoalTranslatorRApp(intent_text="", llm_enabled=True, validator_enabled=False)
        app._generate_raw_candidate = lambda observation, intent_text: {  # type: ignore[method-assign]
            "goal_id": "goal:minimal_parse",
            "summary": "minimal_parse",
            "horizon_slots": 10,
            "kpi_targets": [
                {
                    "kpi_name": "total_throughput_mbps",
                    "objective": "maximize",
                    "target_value": 44.0,
                    "weight": 1.0,
                }
            ],
        }

        update = app.run(
            observation=self._make_observation(slot_id=13, throughput_mbps=20.0),
            policy_snapshot=self._make_snapshot(),
        )

        template = update.validated_intent_template
        self.assertIsNotNone(template)
        self.assertFalse(bool(template.validator_outcome["validator_enabled"]))
        self.assertEqual(template.validation_report.validator_name, "disabled_minimal_parse")
        self.assertEqual(template.projection_report.projector_name, "disabled_passthrough")
        self.assertFalse(template.fallback_report.fallback_used)
        self.assertEqual(update.structured_goal.goal_id, "goal:minimal_parse")

    def test_weighted_translation_backend_is_reflected_in_query_metadata(self) -> None:
        app = IntentGoalTranslatorRApp(
            intent_text="",
            llm_enabled=True,
            translation_embedding_backend_name="weighted_hashed_embedding_v1",
        )
        app._generate_raw_candidate = lambda observation, intent_text: {  # type: ignore[method-assign]
            "goal_id": "goal:weighted_translation_backend",
            "summary": "weighted_translation_backend",
            "horizon_slots": 8,
            "kpi_targets": [
                {
                    "kpi_name": "avg_bler",
                    "objective": "minimize",
                    "target_value": 0.08,
                    "weight": 1.0,
                }
            ],
        }

        update = app.run(
            observation=self._make_observation(slot_id=30, throughput_mbps=18.0, avg_bler=0.12),
            policy_snapshot=self._make_snapshot(),
        )

        template = update.validated_intent_template
        self.assertIsNotNone(template)
        self.assertEqual(
            template.translation_query_key.retriever_backend_name,
            "weighted_hashed_embedding_v1",
        )


if __name__ == "__main__":
    unittest.main()
