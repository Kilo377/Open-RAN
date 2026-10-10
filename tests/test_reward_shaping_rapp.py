import unittest

from models.intent_orchestration import (
    ExplorationHealthGate,
    FallbackReport,
    GoalKpiTarget,
    RewardParameterRange,
    StructuredIntentGoal,
    ValidatedIntentTemplate,
    XAppSelectionPlan,
    validated_intent_template_from_goal,
)
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.non_rt.reward_shaping_rapp import RewardShapingRApp


class TestRewardShapingRApp(unittest.TestCase):
    def _make_goal(self) -> StructuredIntentGoal:
        return StructuredIntentGoal(
            goal_id="goal:guidance",
            intent_id="intent:guidance",
            intent_text="提升10%吞吐",
            summary="Increase throughput while keeping BLER stable.",
            risk_posture="balanced",
            target_entities=["cluster"],
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=48.0,
                    weight=1.4,
                    unit="mbps",
                    priority=3,
                ),
                GoalKpiTarget(
                    kpi_name="avg_bler",
                    objective="minimize",
                    target_value=0.12,
                    weight=1.0,
                    unit="ratio",
                    priority=2,
                ),
            ],
        )

    def _make_plan(self) -> XAppSelectionPlan:
        return XAppSelectionPlan(
            plan_id="selection:goal:guidance:g1",
            goal_id="goal:guidance",
            selected_xapps=[
                "urllc_guard_xapp_balanced",
                "massive_mimo_beamformer_xapp_capacity_push",
                "traffic_steering_xapp_balanced",
            ],
            hard_safety_envelope=[
                RewardParameterRange(
                    field_name="scheduler_policy_hint",
                    target_xapp_name="urllc_guard_xapp_balanced",
                    scope="cell",
                    discrete_values=["balanced", "latency_bias"],
                ),
                RewardParameterRange(
                    field_name="latency_bias",
                    target_xapp_name="urllc_guard_xapp_balanced",
                    scope="cell",
                    lower_bound=1.0,
                    upper_bound=1.7,
                    step=0.05,
                    max_delta=0.2,
                ),
                RewardParameterRange(
                    field_name="beam_candidate_preference",
                    target_xapp_name="massive_mimo_beamformer_xapp_capacity_push",
                    scope="cell",
                    discrete_values=["balanced", "coverage", "capacity"],
                ),
                RewardParameterRange(
                    field_name="power_cap_dbm",
                    target_xapp_name="massive_mimo_beamformer_xapp_capacity_push",
                    scope="cell",
                    lower_bound=28.0,
                    upper_bound=30.5,
                    step=0.25,
                    max_delta=0.75,
                ),
                RewardParameterRange(
                    field_name="traffic_steering_preference",
                    target_xapp_name="traffic_steering_xapp_balanced",
                    scope="cluster",
                    discrete_values=["balanced", "edge_offload", "stay_put"],
                ),
            ],
            exploration_health_gate=ExplorationHealthGate(
                max_avg_bler=0.12,
                max_total_queue_bytes=120000,
                min_total_throughput_mbps=26.0,
                max_handover_pending_ue_count=2,
                freeze_slots_on_violation=4,
            ),
        )

    def _make_observation(
        self,
        *,
        throughput_mbps: float,
        total_queue_bytes: int,
        avg_bler: float,
        handover_pending_ue_count: int = 0,
    ) -> Observation:
        return Observation(
            slot_id=10,
            avg_sinr_db=18.0,
            avg_bler=avg_bler,
            total_queue_bytes=total_queue_bytes,
            total_throughput_mbps=throughput_mbps,
            handover_pending_ue_count=handover_pending_ue_count,
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

    def _make_snapshot(
        self,
        *,
        fallback_used: bool = False,
        policy_epoch_id: int = 3,
    ) -> PolicySnapshot:
        goal = self._make_goal()
        validated_template = validated_intent_template_from_goal(
            goal,
            fallback_report=FallbackReport(
                fallback_used=fallback_used,
                fallback_strategy="rule_template" if fallback_used else "",
                fallback_reason="schema invalid" if fallback_used else "",
            ),
        )
        return PolicySnapshot(
            policy_epoch_id=policy_epoch_id,
            active_intent_id="intent:guidance",
            active_intent_text="提升10%吞吐",
            structured_goal=goal,
            validated_intent_template=validated_template,
            xapp_selection_plan=self._make_plan(),
        )

    def test_guidance_contract_is_generated_with_fixed_structure(self) -> None:
        app = RewardShapingRApp(llm_enabled=False)
        update = app.run(
            self._make_observation(
                throughput_mbps=39.0,
                total_queue_bytes=78000,
                avg_bler=0.07,
            ),
            policy_snapshot=self._make_snapshot(policy_epoch_id=5),
        )

        self.assertIsNotNone(update.reward_spec)
        self.assertIsNotNone(update.rl_guidance)
        self.assertEqual(update.rl_guidance.source_epoch, 5)
        self.assertEqual(update.rl_guidance.derived_from_intent_id, "intent:guidance")
        self.assertFalse(update.rl_guidance.fallback_used)
        self.assertSetEqual(
            set(update.rl_guidance.reward_coefficients.keys()),
            {
                "throughput_gain",
                "queue_reduction",
                "bler_reduction",
                "handover_risk_reduction",
                "power_cost_penalty",
            },
        )
        self.assertGreater(len(update.rl_guidance.guided_action_region), 0)
        self.assertGreater(len(update.rl_guidance.parameter_priority_vector), 0)
        self.assertAlmostEqual(
            sum(update.rl_guidance.parameter_priority_vector.values()),
            1.0,
            places=5,
        )
        self.assertIsNotNone(update.rl_guidance.selected_safety_rule_set)
        self.assertGreater(
            len(update.rl_guidance.selected_safety_rule_set.selected_rule_refs),
            0,
        )

    def test_guidance_refreshes_when_observation_changes_but_intent_is_constant(self) -> None:
        app = RewardShapingRApp(llm_enabled=False)
        policy_snapshot = self._make_snapshot(policy_epoch_id=6)

        first_update = app.run(
            self._make_observation(
                throughput_mbps=46.0,
                total_queue_bytes=42000,
                avg_bler=0.05,
            ),
            policy_snapshot=policy_snapshot,
        )
        second_update = app.run(
            self._make_observation(
                throughput_mbps=31.0,
                total_queue_bytes=112000,
                avg_bler=0.10,
                handover_pending_ue_count=2,
            ),
            policy_snapshot=policy_snapshot,
        )

        self.assertEqual(
            first_update.rl_guidance.derived_from_intent_id,
            second_update.rl_guidance.derived_from_intent_id,
        )
        self.assertNotEqual(
            first_update.rl_guidance.reward_coefficients,
            second_update.rl_guidance.reward_coefficients,
        )
        self.assertNotEqual(
            first_update.rl_guidance.parameter_priority_vector,
            second_update.rl_guidance.parameter_priority_vector,
        )
        first_rule_ids = [
            rule_ref.rule_id
            for rule_ref in first_update.rl_guidance.selected_safety_rule_set.selected_rule_refs
        ]
        second_rule_ids = [
            rule_ref.rule_id
            for rule_ref in second_update.rl_guidance.selected_safety_rule_set.selected_rule_refs
        ]
        self.assertNotEqual(first_rule_ids, second_rule_ids)
        self.assertIn("rule:degraded_network_freeze", second_rule_ids)

    def test_fallback_validated_template_still_generates_guidance_with_provenance(self) -> None:
        app = RewardShapingRApp(llm_enabled=False)
        update = app.run(
            self._make_observation(
                throughput_mbps=34.0,
                total_queue_bytes=93000,
                avg_bler=0.08,
            ),
            policy_snapshot=self._make_snapshot(fallback_used=True, policy_epoch_id=7),
        )

        self.assertIsNotNone(update.rl_guidance)
        self.assertTrue(update.rl_guidance.fallback_used)
        self.assertEqual(update.rl_guidance.source_epoch, 7)
        self.assertEqual(update.rl_guidance.derived_from_intent_id, "intent:guidance")
        selected_rule_ids = [
            rule_ref.rule_id
            for rule_ref in update.rl_guidance.selected_safety_rule_set.selected_rule_refs
        ]
        self.assertIn("rule:fallback_template_guard", selected_rule_ids)
        self.assertIsNotNone(update.safety_constraint_set)
        self.assertEqual(
            update.safety_constraint_set.rule_set_id,
            update.rl_guidance.selected_safety_rule_set.rule_set_id,
        )

    def test_no_rl_guidance_variant_keeps_reward_spec_and_safety_set(self) -> None:
        app = RewardShapingRApp(llm_enabled=False, rl_guidance_enabled=False)
        update = app.run(
            self._make_observation(
                throughput_mbps=36.0,
                total_queue_bytes=88000,
                avg_bler=0.09,
            ),
            policy_snapshot=self._make_snapshot(policy_epoch_id=8),
        )

        self.assertIsNotNone(update.reward_spec)
        self.assertIsNone(update.rl_guidance)
        self.assertIsNotNone(update.safety_constraint_set)
        self.assertEqual(
            update.policy_description,
            "Fixed reward-template instantiation with legacy Near-RT fallback",
        )


if __name__ == "__main__":
    unittest.main()
