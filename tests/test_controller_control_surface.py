import unittest

from models.action import ConfigAction, ControlDecision
from models.intent_orchestration import (
    ArbitrationProfileRef,
    CandidatePlanScore,
    ExplorationHealthGate,
    FeasibilityConstraintSummary,
    GoalKpiTarget,
    OrchestrationPlanView,
    OrchestrationMemoryCase,
    OrchestrationMemoryState,
    ReflectionNote,
    RewardParameterRange,
    RetrievedMemoryCaseTrace,
    RlGuidanceContract,
    SafetyConstraintSet,
    SafetyRuleReference,
    SelectionQueryKey,
    StructuredIntentGoal,
    TranslationContextTrace,
    TranslationQueryKey,
    ValidatedIntentTemplate,
    ValidatedKpiTemplateEntry,
    XAppSelectionPlan,
)
from models.observation import Observation
from models.policy import PolicySnapshot, PolicyUpdate
import ric.controller_manager as controller_manager_module
from ric.controller_manager import ControllerManager
from ric.interfaces.a1_like import A1LikeInterface
from ric.interfaces.e2_like import E2LikeInterface
from ric.near_rt.beam_xapp import BeamXApp
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.latency_scheduler_xapp import LatencySchedulerXApp
from ric.non_rt.active_intent_manager import ActiveIntentManager
from ric.non_rt.codebook_rapp import CodebookRApp
from ric.non_rt.rapp_base import RAppBase
from ric.non_rt.slicing_rapp import SlicingRApp


class TestControllerControlSurface(unittest.TestCase):
    class _CountingRApp(RAppBase):
        def __init__(self, *, name: str) -> None:
            super().__init__(name=name)
            self.run_count = 0

        def run(
            self,
            observation: Observation,
            policy_snapshot: PolicySnapshot | None = None,
        ) -> PolicyUpdate:
            _ = observation
            _ = policy_snapshot
            self.run_count += 1
            return PolicyUpdate(source_name=self.name)

    class _ScopedDummyXApp(XAppBase):
        def __init__(
            self,
            *,
            name: str,
            priority: int,
            controlled_fields: list[str],
            target_cell_id: str,
            scheduler_policy_hint: str,
            latency_bias: float,
        ) -> None:
            super().__init__(name=name, priority=priority, controlled_fields=controlled_fields)
            self._target_cell_id = target_cell_id
            self._scheduler_policy_hint = scheduler_policy_hint
            self._latency_bias = latency_bias

        def run(self, observation: Observation, policy_snapshot: PolicySnapshot) -> ControlDecision:
            _ = observation
            _ = policy_snapshot
            return ControlDecision(
                config_action=ConfigAction(
                    target_cell_id=self._target_cell_id,
                    scheduler_policy_hint=self._scheduler_policy_hint,
                    latency_bias=self._latency_bias,
                    source_name=self.name,
                )
            )

    class _RefreshingTranslatorRApp(RAppBase):
        def __init__(self, *, name: str) -> None:
            super().__init__(name=name)
            self.snapshot_records: list[dict[str, object]] = []

        def run(
            self,
            observation: Observation,
            policy_snapshot: PolicySnapshot | None = None,
        ) -> PolicyUpdate:
            active_intent_text = str(getattr(policy_snapshot, "active_intent_text", "") or "")
            active_intent_id = str(getattr(policy_snapshot, "active_intent_id", "") or "")
            self.snapshot_records.append(
                {
                    "active_intent_id": active_intent_id,
                    "intent_changed_this_epoch": bool(
                        getattr(policy_snapshot, "intent_changed_this_epoch", False)
                    ),
                    "translation_refreshed_this_epoch": bool(
                        getattr(policy_snapshot, "translation_refreshed_this_epoch", False)
                    ),
                    "throughput_mbps": float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0),
                }
            )
            translated_goal = StructuredIntentGoal(
                goal_id="goal:test_refresh",
                intent_id=active_intent_id,
                intent_text=active_intent_text,
                summary=f"refresh_from_{float(getattr(observation, 'total_throughput_mbps', 0.0) or 0.0):.1f}",
                kpi_targets=[
                    GoalKpiTarget(
                        kpi_name="total_throughput_mbps",
                        objective="maximize",
                        target_value=float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0) + 1.0,
                    )
                ],
            )
            return PolicyUpdate(
                active_intent_text=active_intent_text,
                active_intent_id=active_intent_id,
                intent_changed_this_epoch=bool(
                    getattr(policy_snapshot, "intent_changed_this_epoch", False)
                ),
                translation_refreshed_this_epoch=True,
                structured_goal=translated_goal,
                source_name=self.name,
            )

    def test_controller_manager_merges_control_surface_fields(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[
                A1LikeInterface(CodebookRApp()),
                A1LikeInterface(SlicingRApp()),
            ],
            e2_interfaces=[
                E2LikeInterface(
                    BeamXApp(
                        name="beam_xapp",
                        priority=20,
                        sinr_threshold_db=12.0,
                        queue_threshold_bytes=2000,
                    )
                ),
                E2LikeInterface(
                    LatencySchedulerXApp(
                        name="latency_scheduler_xapp",
                        priority=10,
                        queue_threshold_bytes=1500,
                    )
                ),
            ],
            xapp_interval_slots=1,
            rapp_interval_slots=1,
        )

        observation = Observation(
            slot_id=1,
            avg_sinr_db=8.0,
            avg_bler=0.15,
            total_queue_bytes=6000,
            total_throughput_mbps=4.0,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            scheduled_ue_ids=[],
            num_scheduled_ues=1,
            ue_observations=[],
        )

        decision = manager.update(slot_id=1, observation=observation)
        config = decision.config_action

        self.assertEqual(config.scheduler_policy_hint, "slice_bias")
        self.assertEqual(
            config.slice_weight_adjustment,
            {"urllc": 1.35, "embb": 1.05, "mmtc": 0.85},
        )
        self.assertIsNone(config.latency_bias)
        self.assertEqual(config.beam_candidate_preference, "coverage")
        self.assertEqual(config.power_cap_dbm, 31.0)
        self.assertIn("scheduler_policy_hint", decision.trace.selected_config_action_by_field)
        self.assertIn("beam_candidate_preference", decision.trace.selected_config_action_by_field)

    def test_controller_manager_keeps_distinct_cell_scoped_config_actions(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[
                E2LikeInterface(
                    self._ScopedDummyXApp(
                        name="cell_1_latency_xapp",
                        priority=20,
                        controlled_fields=["scheduler_policy_hint", "latency_bias"],
                        target_cell_id="cell_1",
                        scheduler_policy_hint="latency_bias",
                        latency_bias=1.2,
                    )
                ),
                E2LikeInterface(
                    self._ScopedDummyXApp(
                        name="cell_2_latency_xapp",
                        priority=15,
                        controlled_fields=["scheduler_policy_hint", "latency_bias"],
                        target_cell_id="cell_2",
                        scheduler_policy_hint="latency_bias",
                        latency_bias=1.7,
                    )
                ),
            ],
            xapp_interval_slots=1,
            rapp_interval_slots=100000,
        )

        observation = Observation(
            slot_id=3,
            avg_sinr_db=10.0,
            avg_bler=0.05,
            total_queue_bytes=2000,
            total_throughput_mbps=8.0,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            scheduled_ue_ids=[],
            num_scheduled_ues=0,
            ue_observations=[],
        )

        decision = manager.update(slot_id=3, observation=observation)

        self.assertEqual(len(decision.config_actions), 3)
        scoped_actions = {
            action.target_cell_id: action
            for action in decision.config_actions
            if action.target_cell_id is not None
        }
        self.assertIn("cell_1", scoped_actions)
        self.assertIn("cell_2", scoped_actions)
        self.assertEqual(scoped_actions["cell_1"].latency_bias, 1.2)
        self.assertEqual(scoped_actions["cell_2"].latency_bias, 1.7)
        self.assertIn(
            "cell:cell_1.scheduler_policy_hint",
            decision.trace.selected_config_action_by_field,
        )
        self.assertIn(
            "cell:cell_2.scheduler_policy_hint",
            decision.trace.selected_config_action_by_field,
        )
        self.assertEqual(len(decision.trace.merged_config_actions), 3)

    def test_controller_manager_respects_orchestration_warmup_before_rapps_and_xapps(self) -> None:
        counting_rapp = self._CountingRApp(name="counting_rapp")
        counting_xapp = self._ScopedDummyXApp(
            name="counting_xapp",
            priority=20,
            controlled_fields=["scheduler_policy_hint", "latency_bias"],
            target_cell_id="cell_0",
            scheduler_policy_hint="latency_bias",
            latency_bias=1.2,
        )
        counting_xapp.run_count = 0
        original_run = counting_xapp.run

        def _wrapped_run(observation: Observation, policy_snapshot: PolicySnapshot) -> ControlDecision:
            counting_xapp.run_count += 1
            return original_run(observation, policy_snapshot)

        counting_xapp.run = _wrapped_run  # type: ignore[method-assign]

        manager = ControllerManager(
            a1_interfaces=[A1LikeInterface(counting_rapp)],
            e2_interfaces=[E2LikeInterface(counting_xapp)],
            xapp_interval_slots=5,
            rapp_interval_slots=100,
            orchestration_warmup_slots=5,
        )

        for slot_id in range(1, 11):
            manager.update(
                slot_id=slot_id,
                observation=Observation(
                    slot_id=slot_id,
                    avg_sinr_db=10.0,
                    avg_bler=0.05,
                    total_queue_bytes=1000,
                    total_throughput_mbps=5.0,
                    cell_id="cell_0",
                    current_codebook_id=0,
                    current_beam_profile_id=0,
                    tx_power_dbm=30.0,
                    scheduler_mode="round_robin",
                    scheduled_ue_ids=[],
                    num_scheduled_ues=0,
                    ue_observations=[],
                ),
            )

        self.assertEqual(counting_rapp.run_count, 1)
        self.assertEqual(counting_xapp.run_count, 1)

    def test_controller_manager_emits_selection_summary_only_when_plan_changes(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[],
            xapp_interval_slots=1,
            rapp_interval_slots=10,
        )
        emitted_messages: list[str] = []
        original_orchestration_print = controller_manager_module.orchestration_print

        def _capture(message: str) -> None:
            emitted_messages.append(message)

        controller_manager_module.orchestration_print = _capture
        try:
            manager._emit_rapp_summary(
                6,
                PolicyUpdate(
                    xapp_selection_plan=XAppSelectionPlan(
                        plan_id="selection:goal_a:g1",
                        goal_id="goal_a",
                        selected_xapps=["xapp_a", "xapp_b"],
                    ),
                    source_name="xapp_selection_rapp",
                ),
            )
            manager._emit_rapp_summary(
                10,
                PolicyUpdate(
                    xapp_selection_plan=XAppSelectionPlan(
                        plan_id="selection:goal_a:g1",
                        goal_id="goal_a",
                        selected_xapps=["xapp_a", "xapp_b"],
                    ),
                    source_name="xapp_selection_rapp",
                ),
            )
            manager._emit_rapp_summary(
                20,
                PolicyUpdate(
                    xapp_selection_plan=XAppSelectionPlan(
                        plan_id="selection:goal_a:g2",
                        goal_id="goal_a",
                        selected_xapps=["xapp_a", "xapp_c"],
                    ),
                    source_name="xapp_selection_rapp",
                ),
            )
        finally:
            controller_manager_module.orchestration_print = original_orchestration_print

        selection_messages = [
            message for message in emitted_messages if message.startswith("[rapp_selection]")
        ]
        self.assertEqual(len(selection_messages), 2)
        self.assertIn("selection:goal_a:g1", selection_messages[0])
        self.assertIn("selection:goal_a:g2", selection_messages[1])

    def test_controller_manager_emits_failure_reason_and_reflection_for_reselection(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[],
            xapp_interval_slots=1,
            rapp_interval_slots=10,
        )
        emitted_messages: list[str] = []
        original_orchestration_print = controller_manager_module.orchestration_print

        def _capture(message: str) -> None:
            emitted_messages.append(message)

        controller_manager_module.orchestration_print = _capture
        try:
            manager._emit_rapp_summary(
                20,
                PolicyUpdate(
                    xapp_selection_plan=XAppSelectionPlan(
                        plan_id="selection:goal_a:g2",
                        goal_id="goal_a",
                        selected_xapps=["xapp_c", "xapp_d"],
                        last_failure_reason="joint_tuning_not_converging",
                        last_failure_gap_score=0.24,
                        reflection=ReflectionNote(
                            reflection_round=1,
                            outcome_summary="Current tuning plateaued before meeting the throughput goal.",
                            self_critique="The active portfolio overused scheduler-centric actions.",
                            improvement_hypothesis="Replace one scheduler-dominant xApp with a steering-oriented xApp.",
                            recommended_changes=["swap xapp_b for xapp_d"],
                            generated_by="test_reflection",
                        ),
                    ),
                    source_name="xapp_selection_rapp",
                ),
            )
        finally:
            controller_manager_module.orchestration_print = original_orchestration_print

        selection_messages = [
            message for message in emitted_messages if message.startswith("[rapp_selection]")
        ]
        reflection_messages = [
            message for message in emitted_messages if message.startswith("[rapp_reflection]")
        ]
        self.assertEqual(len(selection_messages), 1)
        self.assertIn('"last_failure_reason": "joint_tuning_not_converging"', selection_messages[0])
        self.assertEqual(len(reflection_messages), 1)
        self.assertIn('"failure_reason": "joint_tuning_not_converging"', reflection_messages[0])
        self.assertIn('"swap xapp_b for xapp_d"', reflection_messages[0])

    def test_active_intent_persists_across_epochs_while_translation_can_refresh(self) -> None:
        translator_rapp = self._RefreshingTranslatorRApp(name="translator_rapp")
        active_intent_manager = ActiveIntentManager(
            initial_intent_text="提升10%吞吐",
            initial_epoch=1,
            initial_change_reason="test_initial_intent",
            source_name="test_operator",
        )
        manager = ControllerManager(
            a1_interfaces=[A1LikeInterface(translator_rapp)],
            e2_interfaces=[],
            xapp_interval_slots=1,
            rapp_interval_slots=10,
            active_intent_manager=active_intent_manager,
        )

        for slot_id, throughput in ((1, 10.0), (10, 20.0)):
            manager.update(
                slot_id=slot_id,
                observation=Observation(
                    slot_id=slot_id,
                    avg_sinr_db=10.0,
                    avg_bler=0.05,
                    total_queue_bytes=1000,
                    total_throughput_mbps=throughput,
                    cell_id="cell_0",
                    current_codebook_id=0,
                    current_beam_profile_id=0,
                    tx_power_dbm=30.0,
                    scheduler_mode="round_robin",
                    scheduled_ue_ids=[],
                    num_scheduled_ues=0,
                    ue_observations=[],
                ),
            )

        self.assertEqual(len(translator_rapp.snapshot_records), 2)
        first_epoch_record = translator_rapp.snapshot_records[0]
        second_epoch_record = translator_rapp.snapshot_records[1]
        self.assertNotEqual(first_epoch_record["active_intent_id"], "")
        self.assertEqual(first_epoch_record["active_intent_id"], second_epoch_record["active_intent_id"])
        self.assertTrue(bool(first_epoch_record["intent_changed_this_epoch"]))
        self.assertFalse(bool(second_epoch_record["intent_changed_this_epoch"]))
        self.assertFalse(bool(first_epoch_record["translation_refreshed_this_epoch"]))
        self.assertFalse(bool(second_epoch_record["translation_refreshed_this_epoch"]))
        self.assertNotEqual(first_epoch_record["throughput_mbps"], second_epoch_record["throughput_mbps"])

        final_snapshot = manager.policy_store.get_snapshot()
        self.assertEqual(final_snapshot.active_intent_id, first_epoch_record["active_intent_id"])
        self.assertFalse(bool(final_snapshot.intent_changed_this_epoch))
        self.assertTrue(bool(final_snapshot.translation_refreshed_this_epoch))
        self.assertIsNotNone(final_snapshot.structured_goal)
        self.assertAlmostEqual(final_snapshot.structured_goal.kpi_targets[0].target_value, 21.0)
        epoch_history = manager.get_orchestration_epoch_history()
        self.assertEqual(len(epoch_history), 2)
        self.assertEqual(epoch_history[0]["active_intent_id"], epoch_history[1]["active_intent_id"])
        self.assertTrue(bool(epoch_history[0]["intent_changed_this_epoch"]))
        self.assertFalse(bool(epoch_history[1]["intent_changed_this_epoch"]))
        self.assertTrue(bool(epoch_history[0]["translation_refreshed_this_epoch"]))
        self.assertTrue(bool(epoch_history[1]["translation_refreshed_this_epoch"]))
        self.assertIn("selected_xapps", epoch_history[0])
        self.assertIn("guidance_id", epoch_history[0])
        self.assertIn("selection_query_id", epoch_history[0])
        self.assertIn("selection_query_kpis", epoch_history[0])
        self.assertIn("selection_query_target_profile", epoch_history[0])
        self.assertIn("candidate_plan_count", epoch_history[0])
        self.assertIn("validator_enabled", epoch_history[0])
        self.assertIn("rl_guidance_enabled", epoch_history[0])
        self.assertIn("safety_shield_enabled", epoch_history[0])
        self.assertIn("static_xapp_selection_enabled", epoch_history[0])
        self.assertIn("intent_loss_raw", epoch_history[0])
        self.assertIn("intent_loss_normalized", epoch_history[0])
        self.assertEqual(
            epoch_history[0]["intent_loss_definition_version"],
            "v2_paper_intent_loss",
        )
        self.assertNotEqual(
            epoch_history[0]["intent_loss_normalized"],
            epoch_history[1]["intent_loss_normalized"],
        )

    def test_orchestration_epoch_history_exports_selection_query_and_candidate_scores(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[],
            xapp_interval_slots=1,
            rapp_interval_slots=10,
        )
        goal = StructuredIntentGoal(
            goal_id="goal:selection_trace",
            intent_id="intent:selection_trace",
            intent_text="提升吞吐并保持稳定",
            summary="Increase throughput while staying stable.",
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=42.0,
                    weight=1.2,
                )
            ],
        )
        selection_plan = XAppSelectionPlan(
            plan_id="selection:goal:selection_trace:g2",
            goal_id=goal.goal_id,
            selected_xapps=["massive_mimo_beamformer_xapp_capacity_push"],
            selection_query_key=SelectionQueryKey(
                query_id="qsel:goal:selection_trace:throughput:slot10",
                query_slot_id=10,
                goal_id=goal.goal_id,
                intent_id=goal.intent_id,
                kpi_names=["total_throughput_mbps"],
                context_signature={"target_profile": "throughput"},
                previous_xapp_categories=["traffic_steering_xapp"],
            ),
            candidate_plan_scores=[
                CandidatePlanScore(
                    candidate_id="candidate:beamformer",
                    score_version="j_sel_surrogate_v3_predictor_objective",
                    score_direction="minimize",
            predictor_name="memory_calibrated_surrogate_predictor_v1",
                    requested_categories=["massive_mimo_beamformer_xapp"],
                    selected_xapps=["massive_mimo_beamformer_xapp_capacity_push"],
                    predicted_intent_alignment=1.0,
                    predicted_intent_loss=0.0,
                    predicted_memory_support=0.5,
                    predicted_failure_risk=0.1,
                    predicted_conflict_cost=0.0,
                    switching_cost=0.2,
                    stability_risk=0.0,
                    resource_pressure=0.02,
                    total_score=0.005,
                    selected=True,
                    rationale="intent_loss=0.000, alignment=1.000, memory_support=0.500",
                )
            ],
            status="active",
            selection_locked=True,
            selection_generation=2,
        )
        manager.policy_store.apply_update(
            PolicyUpdate(
                policy_epoch_id=2,
                structured_goal=goal,
                active_intent_id=goal.intent_id,
                intent_changed_this_epoch=False,
                translation_refreshed_this_epoch=True,
                xapp_selection_plan=selection_plan,
                source_name="test_selection",
            )
        )
        entry = manager._build_orchestration_epoch_entry(
            slot_id=10,
            observation=Observation(
                slot_id=10,
                avg_sinr_db=12.0,
                avg_bler=0.05,
                total_queue_bytes=2000,
                total_throughput_mbps=41.0,
                cell_id="cell_0",
                current_codebook_id=0,
                current_beam_profile_id=0,
                tx_power_dbm=30.0,
                scheduler_mode="round_robin",
                scheduled_ue_ids=[],
                num_scheduled_ues=0,
                ue_observations=[],
            ),
        )

        self.assertEqual(entry["selection_query_id"], "qsel:goal:selection_trace:throughput:slot10")
        self.assertEqual(entry["selection_query_encoding_version"], "structured_retrieval_v2_embedding")
        self.assertEqual(entry["selection_query_retriever_backend_name"], "hashed_feature_embedding_v1")
        self.assertEqual(entry["selection_query_embedding_dim"], 0)
        self.assertEqual(entry["selection_query_kpis"], ["total_throughput_mbps"])
        self.assertEqual(entry["selection_query_target_profile"], "throughput")
        self.assertEqual(
            entry["selection_query_previous_xapp_categories"],
            ["traffic_steering_xapp"],
        )
        self.assertEqual(entry["candidate_plan_count"], 1)
        self.assertEqual(entry["selected_candidate_id"], "candidate:beamformer")
        self.assertEqual(entry["selected_candidate_score_version"], "j_sel_surrogate_v3_predictor_objective")
        self.assertEqual(entry["selected_candidate_score_direction"], "minimize")
        self.assertEqual(
            entry["selected_candidate_predictor_name"],
            "memory_calibrated_surrogate_predictor_v1",
        )
        self.assertAlmostEqual(entry["selected_candidate_total_score"], 0.005)
        self.assertAlmostEqual(entry["selected_candidate_predicted_intent_loss"], 0.0)
        self.assertAlmostEqual(entry["selected_candidate_conflict_cost"], 0.0)
        self.assertAlmostEqual(entry["selected_candidate_resource_pressure"], 0.02)
        self.assertEqual(
            entry["selected_candidate_rationale"],
            "intent_loss=0.000, alignment=1.000, memory_support=0.500",
        )
        self.assertEqual(entry["selection_query_slot_id"], 10)

    def test_orchestration_epoch_history_exports_translation_and_feasibility_fields(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[],
            xapp_interval_slots=1,
            rapp_interval_slots=10,
        )
        goal = StructuredIntentGoal(
            goal_id="goal:translation_trace",
            intent_id="intent:translation_trace",
            intent_text="increase throughput while protecting reliability",
            summary="Increase throughput while protecting reliability.",
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=42.0,
                    weight=1.2,
                )
            ],
        )
        validated_template = ValidatedIntentTemplate(
            template_id="template:translation_trace",
            goal_id=goal.goal_id,
            intent_id=goal.intent_id,
            intent_text=goal.intent_text,
            kpi_template=[
                ValidatedKpiTemplateEntry(
                    kpi_name="total_throughput_mbps",
                    semantic_type="max",
                    threshold_value=42.0,
                    weight=1.2,
                )
            ],
            translation_query_key=TranslationQueryKey(
                query_id="qctx:translation_trace:abc123",
                intent_id=goal.intent_id,
                encoding_version="translation_context_v2_embedding",
                retriever_backend_name="hashed_feature_embedding_v1",
                embedding_vector=[0.1, 0.2, 0.3],
                intent_tokens=["throughput", "reliability"],
                context_signature={"queue_class": "medium", "bler_class": "good"},
            ),
            retrieved_context_summary={
                "retrieved_item_count": 2,
                "recommended_kpis": ["total_throughput_mbps", "avg_bler"],
            },
            retrieved_context_traces=[
                TranslationContextTrace(
                    source_id="kpi:throughput",
                    source_type="kpi_dictionary",
                    relevance_score=0.9,
                    summary="Use total_throughput_mbps for capacity intents.",
                )
            ],
        )
        selection_plan = XAppSelectionPlan(
            plan_id="selection:goal:translation_trace:g1",
            goal_id=goal.goal_id,
            selected_xapps=["massive_mimo_beamformer_xapp_capacity_push"],
            candidate_plan_scores=[
                CandidatePlanScore(
                    candidate_id="candidate:translation_trace",
                    score_version="j_sel_surrogate_v3_predictor_objective",
                    score_direction="minimize",
            predictor_name="memory_calibrated_surrogate_predictor_v1",
                    selected_xapps=["massive_mimo_beamformer_xapp_capacity_push"],
                    predicted_intent_loss=0.04,
                    predicted_conflict_cost=0.02,
                    stability_risk=0.01,
                    resource_pressure=0.03,
                    eta_profile_name="eta:throughput_bias",
                    chi_profile_name="chi:capacity_weighted_merge",
                    parameter_bound_count=3,
                    feasible=True,
                    projection_applied=True,
                    total_cpu_cost=2.1,
                    total_mem_cost=1.7,
                    total_score=0.21,
                    selected=True,
                    rationale="intent_loss=0.040, failure_risk=0.080, projected=True",
                )
            ],
        )
        manager.policy_store.apply_update(
            PolicyUpdate(
                policy_epoch_id=4,
                structured_goal=goal,
                validated_intent_template=validated_template,
                active_intent_id=goal.intent_id,
                intent_changed_this_epoch=False,
                translation_refreshed_this_epoch=True,
                xapp_selection_plan=selection_plan,
                orchestration_plan_view=OrchestrationPlanView(
                    plan_id=selection_plan.plan_id,
                    goal_id=goal.goal_id,
                    x=list(selection_plan.selected_xapps),
                    B=[
                        RewardParameterRange(
                            field_name="power_cap_dbm",
                            target_xapp_name="massive_mimo_beamformer_xapp_capacity_push",
                            scope="cell",
                            lower_bound=29.0,
                            upper_bound=30.0,
                        )
                    ],
                    eta={
                        "profile_name": "eta:throughput_bias",
                        "goal_focus": "throughput",
                    },
                    chi=ArbitrationProfileRef(
                        profile_name="chi:capacity_weighted_merge",
                        composition_mode="parameter_tuning",
                        arbitration_mode="weighted_merge",
                        merge_scope="field",
                        defaulted=False,
                    ),
                    feasibility_summary=FeasibilityConstraintSummary(
                        feasible=True,
                        projected=True,
                        max_selected_xapps=4,
                        max_cpu_cost=6.0,
                        max_mem_cost=5.5,
                        total_cpu_cost=2.1,
                        total_mem_cost=1.7,
                        dependency_pairs=["urllc_guard_xapp->latency_scheduler_xapp"],
                        mutex_pairs=["power_saving_xapp<->massive_mimo_beamformer_xapp"],
                        projection_notes=["added dependency latency_scheduler_xapp_balanced"],
                    ),
                ),
                source_name="test_translation_trace",
            )
        )

        entry = manager._build_orchestration_epoch_entry(
            slot_id=12,
            observation=Observation(
                slot_id=12,
                avg_sinr_db=12.0,
                avg_bler=0.05,
                total_queue_bytes=2200,
                total_throughput_mbps=41.0,
                cell_id="cell_0",
                current_codebook_id=0,
                current_beam_profile_id=0,
                tx_power_dbm=30.0,
                scheduler_mode="round_robin",
                scheduled_ue_ids=[],
                num_scheduled_ues=0,
                ue_observations=[],
            ),
        )

        self.assertEqual(entry["translation_query_id"], "qctx:translation_trace:abc123")
        self.assertEqual(entry["translation_query_encoding_version"], "translation_context_v2_embedding")
        self.assertEqual(entry["translation_query_retriever_backend_name"], "hashed_feature_embedding_v1")
        self.assertEqual(entry["translation_query_embedding_dim"], 3)
        self.assertEqual(entry["translation_retrieved_context_count"], 1)
        self.assertIn("kpi:throughput", entry["translation_retrieved_context_source_ids"])
        self.assertEqual(entry["selected_candidate_score_version"], "j_sel_surrogate_v3_predictor_objective")
        self.assertEqual(
            entry["selected_candidate_predictor_name"],
            "memory_calibrated_surrogate_predictor_v1",
        )
        self.assertEqual(entry["selected_candidate_eta_profile_name"], "eta:throughput_bias")
        self.assertEqual(entry["selected_candidate_chi_profile_name"], "chi:capacity_weighted_merge")
        self.assertEqual(entry["selected_candidate_parameter_bound_count"], 3)
        self.assertTrue(bool(entry["selected_candidate_feasible"]))
        self.assertTrue(bool(entry["selected_candidate_projection_applied"]))
        self.assertAlmostEqual(entry["selected_candidate_total_cpu_cost"], 2.1)
        self.assertAlmostEqual(entry["selected_candidate_total_mem_cost"], 1.7)
        self.assertAlmostEqual(entry["selected_candidate_resource_pressure"], 0.03)
        self.assertEqual(entry["eta_profile_name"], "eta:throughput_bias")
        self.assertEqual(entry["eta_goal_focus"], "throughput")
        self.assertEqual(entry["composition_profile_name"], "chi:capacity_weighted_merge")
        self.assertTrue(bool(entry["feasible"]))
        self.assertTrue(bool(entry["projection_applied"]))
        self.assertIn("urllc_guard_xapp->latency_scheduler_xapp", entry["dependency_pairs"])
        self.assertIn("power_saving_xapp<->massive_mimo_beamformer_xapp", entry["mutex_pairs"])
        self.assertIn("added dependency latency_scheduler_xapp_balanced", entry["projection_notes"])

    def test_orchestration_epoch_history_exports_retrieval_traces(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[],
            xapp_interval_slots=1,
            rapp_interval_slots=10,
        )
        manager.policy_store.apply_update(
            PolicyUpdate(
                policy_epoch_id=3,
                active_intent_id="intent:memory_trace",
                memory_state=OrchestrationMemoryState(
                    memory_enabled=True,
                    last_retrieved_success_case_ids=["success:1"],
                    last_retrieved_failure_case_ids=["failure:1"],
                    last_retrieved_success_traces=[
                        RetrievedMemoryCaseTrace(
                            case_id="success:1",
                            memory_type="success",
                            retrieval_rank=1,
                            similarity_score=0.8,
                            feature_similarity=0.75,
                            recency_score=0.5,
                            retrieval_score=0.755,
                            retrieval_confidence=0.68,
                            source_slot_id=20,
                            memory_retention_score=0.74,
                            selected_xapps=["massive_mimo_beamformer_xapp_capacity_push"],
                            kpi_names=["total_throughput_mbps"],
                        )
                    ],
                    last_retrieved_failure_traces=[
                        RetrievedMemoryCaseTrace(
                            case_id="failure:1",
                            memory_type="failure",
                            retrieval_rank=1,
                            similarity_score=0.7,
                            feature_similarity=0.6,
                            recency_score=0.25,
                            retrieval_score=0.6325,
                            retrieval_confidence=0.52,
                            source_slot_id=10,
                            memory_retention_score=0.58,
                            selected_xapps=["urllc_guard_xapp_balanced"],
                            kpi_names=["avg_bler"],
                            failure_reason="intent_gap_threshold_exceeded",
                        )
                    ],
                ),
                source_name="test_memory_trace",
            )
        )
        entry = manager._build_orchestration_epoch_entry(
            slot_id=30,
            observation=Observation(
                slot_id=30,
                avg_sinr_db=12.0,
                avg_bler=0.08,
                total_queue_bytes=5000,
                total_throughput_mbps=35.0,
                cell_id="cell_0",
                current_codebook_id=0,
                current_beam_profile_id=0,
                tx_power_dbm=30.0,
                scheduler_mode="round_robin",
                scheduled_ue_ids=[],
                num_scheduled_ues=0,
                ue_observations=[],
            ),
        )

        self.assertEqual(entry["last_retrieved_success_case_ids"], ["success:1"])
        self.assertEqual(entry["last_retrieved_failure_case_ids"], ["failure:1"])
        self.assertEqual(len(entry["last_retrieved_success_traces"]), 1)
        self.assertEqual(len(entry["last_retrieved_failure_traces"]), 1)
        self.assertEqual(entry["last_retrieved_success_traces"][0]["case_id"], "success:1")
        self.assertAlmostEqual(
            entry["last_retrieved_failure_traces"][0]["retrieval_score"],
            0.6325,
        )
        self.assertAlmostEqual(
            entry["last_retrieved_success_traces"][0]["feature_similarity"],
            0.75,
        )
        self.assertAlmostEqual(
            entry["last_retrieved_failure_traces"][0]["retrieval_confidence"],
            0.52,
        )
        self.assertAlmostEqual(
            entry["last_retrieved_success_traces"][0]["memory_retention_score"],
            0.74,
        )

    def test_controller_manager_emits_guidance_summary_with_fallback_provenance(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[],
            xapp_interval_slots=1,
            rapp_interval_slots=10,
        )
        emitted_messages: list[str] = []
        original_orchestration_print = controller_manager_module.orchestration_print

        def _capture(message: str) -> None:
            emitted_messages.append(message)

        controller_manager_module.orchestration_print = _capture
        try:
            manager._emit_rapp_summary(
                20,
                PolicyUpdate(
                    rl_guidance=RlGuidanceContract(
                        guidance_id="guidance:goal_a",
                        goal_id="goal_a",
                        selection_plan_id="selection:goal_a:g1",
                        reward_coefficients={"throughput_gain": 1.4},
                        parameter_priority_vector={"latency_bias": 1.0},
                        selected_safety_rule_set=SafetyConstraintSet(
                            rule_set_id="guidance_safety:goal_a",
                            goal_id="goal_a",
                            selection_plan_id="selection:goal_a:g1",
                            selected_rule_refs=[
                                SafetyRuleReference(
                                    rule_id="rule:fallback_template_guard",
                                    config={"goal_id": "goal_a"},
                                )
                            ],
                            exploration_health_gate=ExplorationHealthGate(max_avg_bler=0.10),
                        ),
                        source_epoch=3,
                        derived_from_intent_id="intent:goal_a",
                        fallback_used=True,
                    ),
                    source_name="reward_shaping_rapp",
                ),
            )
        finally:
            controller_manager_module.orchestration_print = original_orchestration_print

        guidance_messages = [
            message for message in emitted_messages if message.startswith("[rapp_guidance]")
        ]
        self.assertEqual(len(guidance_messages), 1)
        self.assertIn('"fallback_used": true', guidance_messages[0])
        self.assertIn('"derived_from_intent_id": "intent:goal_a"', guidance_messages[0])
        self.assertIn('"source_epoch": 3', guidance_messages[0])
        self.assertIn('"selected_rule_ids": ["rule:fallback_template_guard"]', guidance_messages[0])

    def test_orchestration_epoch_history_captures_memory_summary(self) -> None:
        manager = ControllerManager(
            a1_interfaces=[],
            e2_interfaces=[],
            xapp_interval_slots=1,
            rapp_interval_slots=10,
        )
        manager.policy_store.snapshot.active_intent_id = "intent:test"
        manager.policy_store.snapshot.structured_goal = StructuredIntentGoal(
            goal_id="goal:test",
            intent_id="intent:test",
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=10.0,
                )
            ],
        )
        manager.policy_store.snapshot.xapp_selection_plan = XAppSelectionPlan(
            plan_id="selection:goal:test:g1",
            goal_id="goal:test",
            selected_xapps=["xapp_a"],
            selection_generation=1,
        )
        manager.policy_store.snapshot.orchestration_plan_view = OrchestrationPlanView(
            plan_id="selection:goal:test:g1",
            goal_id="goal:test",
            x=["xapp_a"],
            chi=ArbitrationProfileRef(
                profile_name="chi:test_memory",
                composition_mode="mixed",
                arbitration_mode="priority_override",
                merge_scope="field",
                override_order=["urllc_guard_xapp", "xapp_a"],
                defaulted=False,
            ),
        )
        manager.policy_store.snapshot.memory_state = OrchestrationMemoryState(
            memory_enabled=True,
            success_cases=[
                OrchestrationMemoryCase(case_id="success:1", selected_xapps=["xapp_a"])
            ],
            failure_cases=[
                OrchestrationMemoryCase(case_id="failure:1", selected_xapps=["xapp_b"])
            ],
            last_retrieved_success_case_ids=["success:1"],
            last_retrieved_failure_case_ids=["failure:1"],
        )

        entry = manager._build_orchestration_epoch_entry(
            slot_id=10,
            observation=Observation(
                slot_id=10,
                avg_sinr_db=10.0,
                avg_bler=0.05,
                total_queue_bytes=1000,
                total_throughput_mbps=8.0,
                cell_id="cell_0",
                current_codebook_id=0,
                current_beam_profile_id=0,
                tx_power_dbm=30.0,
                scheduler_mode="round_robin",
                scheduled_ue_ids=[],
                num_scheduled_ues=0,
                ue_observations=[],
            ),
        )

        self.assertTrue(bool(entry["memory_enabled"]))
        self.assertEqual(entry["success_memory_count"], 1)
        self.assertEqual(entry["failure_memory_count"], 1)
        self.assertEqual(entry["last_retrieved_success_case_ids"], ["success:1"])
        self.assertEqual(entry["last_retrieved_failure_case_ids"], ["failure:1"])
        self.assertEqual(entry["composition_profile_name"], "chi:test_memory")
        self.assertEqual(entry["composition_mode"], "mixed")
        self.assertEqual(entry["arbitration_mode"], "priority_override")


if __name__ == "__main__":
    unittest.main()
