import tempfile
import tempfile
import unittest
from pathlib import Path

from models.intent_orchestration import GoalKpiTarget, StructuredIntentGoal, XAppSelectionPlan
from models.observation import Observation
from models.policy import PolicySnapshot
from experiments.experiment_variants import build_minimal_paper_variants
from ric.non_rt.xapp_selection_rapp import XAppSelectionRApp


class TestXAppSelectionVariants(unittest.TestCase):
    def _make_goal(self) -> StructuredIntentGoal:
        return StructuredIntentGoal(
            goal_id="goal:selection",
            intent_id="intent:selection",
            intent_text="increase throughput by 10 percent",
            summary="Increase throughput while keeping BLER stable.",
            risk_posture="balanced",
            target_entities=["cluster"],
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=48.0,
                    weight=1.4,
                ),
                GoalKpiTarget(
                    kpi_name="avg_bler",
                    objective="minimize",
                    target_value=0.12,
                    weight=1.0,
                ),
            ],
        )

    def _make_observation(
        self,
        *,
        slot_id: int = 10,
        throughput_mbps: float = 32.0,
        avg_bler: float = 0.08,
    ) -> Observation:
        return Observation(
            slot_id=slot_id,
            avg_sinr_db=15.0,
            avg_bler=avg_bler,
            total_queue_bytes=64000,
            total_throughput_mbps=throughput_mbps,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            connected_ue_count=12,
            scheduled_ue_ids=[],
            num_scheduled_ues=0,
            ue_observations=[],
        )

    def test_disabled_llm_uses_heuristic_selection_fallback(self) -> None:
        app = XAppSelectionRApp(llm_enabled=False)
        update = app.run(
            observation=self._make_observation(),
            policy_snapshot=PolicySnapshot(structured_goal=self._make_goal()),
        )

        self.assertIsNotNone(update.xapp_selection_plan)
        self.assertGreater(len(update.xapp_selection_plan.selected_xapps), 0)
        self.assertIn("fallback", update.xapp_selection_plan.generated_by)
        self.assertIsNotNone(update.xapp_selection_plan.selection_query_key)
        self.assertGreater(len(update.xapp_selection_plan.candidate_plan_scores), 0)
        self.assertTrue(
            any(
                bool(getattr(candidate_score, "selected", False))
                for candidate_score in update.xapp_selection_plan.candidate_plan_scores
            )
        )
        self.assertGreater(len(update.xapp_selection_plan.candidate_plan_projections), 0)
        self.assertTrue(bool(update.xapp_selection_plan.eta))
        self.assertIsNotNone(update.xapp_selection_plan.chi)
        self.assertIsNotNone(update.xapp_selection_plan.feasibility_summary)
        self.assertEqual(
            update.xapp_selection_plan.selection_query_key.encoding_version,
            "structured_retrieval_v2_embedding",
        )
        self.assertGreater(
            len(update.xapp_selection_plan.selection_query_key.embedding_vector),
            0,
        )
        self.assertIn("goal_parameter_tuning_xapp", update.enabled_xapps)

    def test_static_xapp_selection_keeps_initial_portfolio_fixed(self) -> None:
        app = XAppSelectionRApp(
            llm_enabled=False,
            static_xapp_selection_enabled=True,
        )
        existing_plan = XAppSelectionPlan(
            plan_id="selection:goal:selection:g1",
            goal_id="goal:selection",
            selected_xapps=[
                "massive_mimo_beamformer_xapp_capacity_push",
                "urllc_guard_xapp_balanced",
            ],
            selection_generation=1,
            last_goal_gap_score=0.25,
            best_goal_gap_score=0.25,
            consecutive_gap_violation_count=1,
            stagnation_count=1,
        )
        update = app.run(
            observation=self._make_observation(throughput_mbps=18.0, avg_bler=0.16),
            policy_snapshot=PolicySnapshot(
                structured_goal=self._make_goal(),
                xapp_selection_plan=existing_plan,
            ),
        )

        self.assertIsNotNone(update.xapp_selection_plan)
        self.assertEqual(
            update.xapp_selection_plan.selected_xapps,
            existing_plan.selected_xapps,
        )
        self.assertEqual(update.xapp_selection_plan.plan_id, existing_plan.plan_id)
        self.assertTrue(bool(update.xapp_selection_plan.selection_locked))
        self.assertEqual(update.xapp_selection_plan.status, "active")
        self.assertEqual(
            update.policy_description,
            "Static xApp selection baseline keeps the initial portfolio fixed",
        )

    def test_available_xapp_names_filter_selection_catalog(self) -> None:
        app = XAppSelectionRApp(
            llm_enabled=False,
            available_xapp_names=[
                "massive_mimo_beamformer_xapp",
                "urllc_guard_xapp",
            ],
        )
        update = app.run(
            observation=self._make_observation(),
            policy_snapshot=PolicySnapshot(structured_goal=self._make_goal()),
        )

        self.assertIsNotNone(update.xapp_selection_plan)
        selected_xapps = list(update.xapp_selection_plan.selected_xapps)
        self.assertGreater(len(selected_xapps), 0)
        for xapp_name in selected_xapps:
            self.assertTrue(
                xapp_name.startswith("massive_mimo_beamformer_xapp")
                or xapp_name.startswith("urllc_guard_xapp")
            )

    def test_selection_updates_success_and_failure_memory_state(self) -> None:
        app = XAppSelectionRApp(
            llm_enabled=False,
            failure_gap_threshold=0.35,
            gap_patience_rounds=1,
            success_gap_threshold=0.12,
        )
        goal = self._make_goal()

        initial_update = app.run(
            observation=self._make_observation(slot_id=10, throughput_mbps=32.0, avg_bler=0.08),
            policy_snapshot=PolicySnapshot(structured_goal=goal),
        )
        self.assertIsNotNone(initial_update.memory_state)
        self.assertEqual(len(initial_update.memory_state.success_cases), 0)
        self.assertEqual(len(initial_update.memory_state.failure_cases), 0)

        good_update = app.run(
            observation=self._make_observation(slot_id=20, throughput_mbps=49.0, avg_bler=0.08),
            policy_snapshot=PolicySnapshot(
                structured_goal=goal,
                xapp_selection_plan=initial_update.xapp_selection_plan,
            ),
        )
        self.assertIsNotNone(good_update.memory_state)
        self.assertGreaterEqual(len(good_update.memory_state.success_cases), 1)
        self.assertEqual(len(good_update.memory_state.failure_cases), 0)

        bad_update = app.run(
            observation=self._make_observation(slot_id=30, throughput_mbps=8.0, avg_bler=0.18),
            policy_snapshot=PolicySnapshot(
                structured_goal=goal,
                xapp_selection_plan=good_update.xapp_selection_plan,
            ),
        )
        self.assertIsNotNone(bad_update.memory_state)
        self.assertGreaterEqual(len(bad_update.memory_state.failure_cases), 1)
        self.assertGreaterEqual(len(bad_update.memory_state.last_retrieved_success_case_ids), 1)
        self.assertGreaterEqual(len(bad_update.memory_state.last_retrieved_failure_case_ids), 0)
        self.assertGreaterEqual(len(bad_update.memory_state.last_retrieved_success_traces), 1)
        self.assertEqual(
            bad_update.xapp_selection_plan.last_failure_reason,
            "intent_gap_threshold_exceeded",
        )
        self.assertIsNotNone(bad_update.xapp_selection_plan.selection_query_key)
        self.assertGreater(len(bad_update.xapp_selection_plan.candidate_plan_scores), 0)
        selected_candidate_scores = [
            candidate_score
            for candidate_score in bad_update.xapp_selection_plan.candidate_plan_scores
            if bool(getattr(candidate_score, "selected", False))
        ]
        self.assertEqual(len(selected_candidate_scores), 1)
        self.assertEqual(
            getattr(selected_candidate_scores[0], "score_version", ""),
            "j_sel_surrogate_v3_predictor_objective",
        )
        self.assertEqual(
            getattr(selected_candidate_scores[0], "predictor_name", ""),
            "memory_calibrated_surrogate_predictor_v1",
        )
        self.assertGreaterEqual(
            float(getattr(selected_candidate_scores[0], "predicted_conflict_cost", 0.0) or 0.0),
            0.0,
        )
        self.assertNotEqual(
            str(getattr(selected_candidate_scores[0], "eta_profile_name", "") or ""),
            "",
        )
        self.assertNotEqual(
            str(getattr(selected_candidate_scores[0], "chi_profile_name", "") or ""),
            "",
        )
        self.assertGreater(
            int(getattr(selected_candidate_scores[0], "parameter_bound_count", 0) or 0),
            0,
        )
        self.assertGreaterEqual(
            float(getattr(selected_candidate_scores[0], "total_cpu_cost", 0.0) or 0.0),
            0.0,
        )
        self.assertGreaterEqual(
            float(getattr(selected_candidate_scores[0], "resource_pressure", 0.0) or 0.0),
            0.0,
        )
        self.assertEqual(
            bad_update.memory_state.last_selection_query_key.query_id,
            bad_update.xapp_selection_plan.selection_query_key.query_id,
        )
        self.assertGreater(
            int(getattr(bad_update.memory_state.success_cases[0], "retrieval_hits", 0) or 0),
            0,
        )
        self.assertEqual(
            getattr(bad_update.memory_state.success_cases[0], "last_retrieved_slot_id", None),
            30,
        )
        self.assertGreaterEqual(
            float(getattr(bad_update.memory_state.success_cases[0], "retention_score", 0.0) or 0.0),
            0.0,
        )
        self.assertGreater(
            float(
                getattr(
                    bad_update.memory_state.last_retrieved_success_traces[0],
                    "retrieval_score",
                    0.0,
                )
                or 0.0
            ),
            0.0,
        )
        self.assertGreater(
            len(
                getattr(
                    bad_update.memory_state.last_retrieved_success_traces[0],
                    "embedding_vector",
                    [],
                )
                or []
            ),
            0,
        )
        self.assertGreaterEqual(
            float(
                getattr(
                    bad_update.memory_state.last_retrieved_success_traces[0],
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
                    bad_update.memory_state.last_retrieved_success_traces[0],
                    "retrieval_confidence",
                    0.0,
                )
                or 0.0
            ),
            0.0,
        )
        self.assertGreaterEqual(
            float(
                getattr(
                    bad_update.memory_state.last_retrieved_success_traces[0],
                    "memory_retention_score",
                    0.0,
                )
                or 0.0
            ),
            0.0,
        )

    def test_persistent_memory_loads_across_rapp_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_store_path = str(Path(temp_dir) / "selection_memory.json")
            app = XAppSelectionRApp(
                llm_enabled=False,
                failure_gap_threshold=0.35,
                gap_patience_rounds=1,
                success_gap_threshold=0.12,
                persistent_memory_enabled=True,
                memory_store_path=memory_store_path,
            )
            goal = self._make_goal()

            initial_update = app.run(
                observation=self._make_observation(slot_id=10, throughput_mbps=32.0, avg_bler=0.08),
                policy_snapshot=PolicySnapshot(structured_goal=goal),
            )
            good_update = app.run(
                observation=self._make_observation(slot_id=20, throughput_mbps=49.0, avg_bler=0.08),
                policy_snapshot=PolicySnapshot(
                    structured_goal=goal,
                    xapp_selection_plan=initial_update.xapp_selection_plan,
                ),
            )
            app.run(
                observation=self._make_observation(slot_id=30, throughput_mbps=8.0, avg_bler=0.18),
                policy_snapshot=PolicySnapshot(
                    structured_goal=goal,
                    xapp_selection_plan=good_update.xapp_selection_plan,
                ),
            )

            reloaded_app = XAppSelectionRApp(
                llm_enabled=False,
                persistent_memory_enabled=True,
                memory_store_path=memory_store_path,
            )

            self.assertGreaterEqual(len(reloaded_app.success_memory), 1)
            self.assertGreaterEqual(len(reloaded_app.failure_memory), 1)
            self.assertGreaterEqual(len(reloaded_app.reflection_memory), 1)
            self.assertEqual(reloaded_app.success_memory[0].source_slot_id, 20)
            self.assertEqual(reloaded_app.failure_memory[0].source_slot_id, 30)
            self.assertEqual(
                reloaded_app.success_memory[0].encoding_version,
                "structured_retrieval_v2_embedding",
            )
            self.assertGreater(len(reloaded_app.success_memory[0].embedding_vector), 0)
            self.assertGreaterEqual(
                float(getattr(reloaded_app.success_memory[0], "retention_score", 0.0) or 0.0),
                0.0,
            )

    def test_feasibility_projector_applies_dependency_and_budget_constraints(self) -> None:
        app = XAppSelectionRApp(
            llm_enabled=False,
            max_selected_xapps=2,
            max_cpu_cost=2.3,
            max_mem_cost=2.0,
            dependency_category_pairs=[["urllc_guard_xapp", "latency_scheduler_xapp"]],
            mutex_category_pairs=[["power_saving_xapp", "massive_mimo_beamformer_xapp"]],
        )

        projected = app.feasibility_projector.project_candidate(
            candidate_id="candidate:test_constraints",
            requested_xapps=[
                "urllc_guard_xapp_balanced",
                "massive_mimo_beamformer_xapp_capacity_push",
                "power_saving_xapp_balanced",
            ],
            goal=self._make_goal(),
            observation=self._make_observation(slot_id=40, throughput_mbps=18.0, avg_bler=0.14),
        )

        self.assertGreater(len(projected.x), 0)
        self.assertLessEqual(len(projected.x), 2)
        self.assertNotIn("power_saving_xapp_balanced", projected.x)
        self.assertTrue(bool(projected.eta))
        self.assertIsNotNone(projected.chi)
        self.assertIsNotNone(projected.feasibility_summary)
        self.assertTrue(bool(projected.feasibility_summary.projected))
        self.assertIn(
            "urllc_guard_xapp->latency_scheduler_xapp",
            list(projected.feasibility_summary.dependency_pairs),
        )
        self.assertTrue(
            any(
                "dropped_for_missing_dependency:urllc_guard_xapp->latency_scheduler_xapp"
                in str(note)
                for note in list(projected.feasibility_summary.projection_notes)
            )
        )
        self.assertGreater(len(projected.B), 0)

    def test_no_memory_mode_suppresses_retrieval_and_storage(self) -> None:
        app = XAppSelectionRApp(
            llm_enabled=False,
            memory_mode="none",
            failure_gap_threshold=0.35,
            gap_patience_rounds=1,
            success_gap_threshold=0.12,
        )
        goal = self._make_goal()

        initial_update = app.run(
            observation=self._make_observation(slot_id=10, throughput_mbps=32.0, avg_bler=0.08),
            policy_snapshot=PolicySnapshot(structured_goal=goal),
        )
        good_update = app.run(
            observation=self._make_observation(slot_id=20, throughput_mbps=49.0, avg_bler=0.08),
            policy_snapshot=PolicySnapshot(
                structured_goal=goal,
                xapp_selection_plan=initial_update.xapp_selection_plan,
            ),
        )
        bad_update = app.run(
            observation=self._make_observation(slot_id=30, throughput_mbps=8.0, avg_bler=0.18),
            policy_snapshot=PolicySnapshot(
                structured_goal=goal,
                xapp_selection_plan=good_update.xapp_selection_plan,
            ),
        )

        self.assertIsNotNone(bad_update.memory_state)
        self.assertFalse(bad_update.memory_state.memory_enabled)
        self.assertEqual(bad_update.memory_state.memory_mode, "none")
        self.assertEqual(len(bad_update.memory_state.success_cases), 0)
        self.assertEqual(len(bad_update.memory_state.failure_cases), 0)
        self.assertEqual(bad_update.memory_state.last_retrieved_success_case_ids, [])
        self.assertEqual(bad_update.memory_state.last_retrieved_failure_case_ids, [])

    def test_success_only_memory_mode_suppresses_failed_case_storage(self) -> None:
        app = XAppSelectionRApp(
            llm_enabled=False,
            memory_mode="success_only",
            failure_gap_threshold=0.35,
            gap_patience_rounds=1,
            success_gap_threshold=0.12,
        )
        goal = self._make_goal()

        initial_update = app.run(
            observation=self._make_observation(slot_id=10, throughput_mbps=32.0, avg_bler=0.08),
            policy_snapshot=PolicySnapshot(structured_goal=goal),
        )
        good_update = app.run(
            observation=self._make_observation(slot_id=20, throughput_mbps=49.0, avg_bler=0.08),
            policy_snapshot=PolicySnapshot(
                structured_goal=goal,
                xapp_selection_plan=initial_update.xapp_selection_plan,
            ),
        )
        bad_update = app.run(
            observation=self._make_observation(slot_id=30, throughput_mbps=8.0, avg_bler=0.18),
            policy_snapshot=PolicySnapshot(
                structured_goal=goal,
                xapp_selection_plan=good_update.xapp_selection_plan,
            ),
        )

        self.assertIsNotNone(bad_update.memory_state)
        self.assertTrue(bad_update.memory_state.memory_enabled)
        self.assertEqual(bad_update.memory_state.memory_mode, "success_only")
        self.assertGreaterEqual(len(bad_update.memory_state.success_cases), 1)
        self.assertEqual(len(bad_update.memory_state.failure_cases), 0)
        self.assertGreaterEqual(len(bad_update.memory_state.last_retrieved_success_case_ids), 1)
        self.assertEqual(bad_update.memory_state.last_retrieved_failure_case_ids, [])

    def test_no_age_decay_keeps_old_memory_recency_flat(self) -> None:
        goal = self._make_goal()
        selected_xapps = [
            "massive_mimo_beamformer_xapp_capacity_push",
            "urllc_guard_xapp_balanced",
        ]
        plan = XAppSelectionPlan(
            plan_id="selection:old-memory",
            goal_id=goal.goal_id,
            selected_xapps=selected_xapps,
        )
        ranked_app = XAppSelectionRApp(llm_enabled=False)
        no_decay_app = XAppSelectionRApp(llm_enabled=False, memory_age_decay_enabled=False)
        memory_case = ranked_app._build_memory_case(
            memory_type="success",
            goal=goal,
            observation=self._make_observation(slot_id=10),
            plan=plan,
            reflection=None,
        )
        ranked_query = ranked_app._selection_query_key(
            goal=goal,
            observation=self._make_observation(slot_id=210),
            previous_selected_xapps=selected_xapps,
        )
        no_decay_query = no_decay_app._selection_query_key(
            goal=goal,
            observation=self._make_observation(slot_id=210),
            previous_selected_xapps=selected_xapps,
        )

        ranked_recency = ranked_app._memory_case_similarity(
            memory_case,
            selection_query_key=ranked_query,
        )[2]
        no_decay_recency = no_decay_app._memory_case_similarity(
            memory_case,
            selection_query_key=no_decay_query,
        )[2]

        self.assertLess(ranked_recency, no_decay_recency)
        self.assertEqual(no_decay_recency, 1.0)

    def test_exp1_b_variants_expose_memory_ablation_switches(self) -> None:
        variants = build_minimal_paper_variants()

        self.assertFalse(
            variants["no_transfer_guard"].ric_overrides()["reflective_transfer_guard_enabled"]
        )
        self.assertEqual(
            variants["shuffled_memory"].ric_overrides()["memory_retrieval_mode"],
            "shuffled",
        )
        self.assertFalse(variants["no_age_decay"].ric_overrides()["memory_age_decay_enabled"])

    def test_no_transfer_guard_disables_reflective_pareto_guard(self) -> None:
        mobility_goal = StructuredIntentGoal(
            goal_id="goal:mobility",
            intent_id="intent:mobility",
            intent_text="stabilize handover for a moving hotspot",
            summary="Keep mobility and handover risk stable.",
            risk_posture="balanced",
            target_entities=["cluster"],
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="handover_risk",
                    objective="minimize",
                    target_value=0.20,
                    weight=1.0,
                )
            ],
        )
        observation = self._make_observation(slot_id=40)
        observation.handover_pending_ue_count = 4
        guarded_app = XAppSelectionRApp(llm_enabled=False)
        no_guard_app = XAppSelectionRApp(
            llm_enabled=False,
            reflective_transfer_guard_enabled=False,
        )

        self.assertGreater(
            len(
                guarded_app._reflective_pareto_guard_portfolio(
                    goal=mobility_goal,
                    observation=observation,
                    retrieved_failure_cases=[],
                )
            ),
            0,
        )
        self.assertEqual(
            no_guard_app._reflective_pareto_guard_portfolio(
                goal=mobility_goal,
                observation=observation,
                retrieved_failure_cases=[],
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
