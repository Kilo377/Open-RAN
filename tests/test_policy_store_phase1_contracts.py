import unittest

from models.intent_orchestration import (
    GoalKpiTarget,
    RewardParameterRange,
    RewardSpec,
    StructuredIntentGoal,
    XAppSelectionPlan,
)
from models.policy import PolicyUpdate
from ric.non_rt.policy_store import PolicyStore


class TestPolicyStorePhase1Contracts(unittest.TestCase):
    def _make_goal(self) -> StructuredIntentGoal:
        return StructuredIntentGoal(
            goal_id="goal:phase1",
            intent_id="intent:phase1",
            intent_text="increase throughput by 10 percent",
            summary="Increase throughput while maintaining BLER stability.",
            horizon_slots=12,
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
                    tolerance=0.03,
                    weight=1.0,
                    unit="ratio",
                    priority=2,
                ),
            ],
            constraints={"avoid_extreme_power_increase": True},
            generated_by="test_goal_translator",
        )

    def _make_plan(self) -> XAppSelectionPlan:
        return XAppSelectionPlan(
            plan_id="selection:goal:phase1:g1",
            goal_id="goal:phase1",
            selected_xapps=[
                "massive_mimo_beamformer_xapp_capacity_push",
                "urllc_guard_xapp_balanced",
            ],
            hard_safety_envelope=[
                RewardParameterRange(
                    field_name="power_cap_dbm",
                    target_xapp_name="massive_mimo_beamformer_xapp_capacity_push",
                    scope="cell",
                    lower_bound=28.0,
                    upper_bound=30.5,
                    step=0.25,
                    max_delta=0.75,
                )
            ],
            generated_by="test_selector",
        )

    def _make_reward_spec(self) -> RewardSpec:
        return RewardSpec(
            reward_id="reward:phase1",
            goal_id="goal:phase1",
            selection_plan_id="selection:goal:phase1:g1",
            reward_terms={"throughput_gain": 1.6, "bler_reduction": 1.1},
            parameter_ranges=[
                RewardParameterRange(
                    field_name="latency_bias",
                    target_xapp_name="urllc_guard_xapp_balanced",
                    scope="cell",
                    lower_bound=1.0,
                    upper_bound=1.7,
                    step=0.05,
                    max_delta=0.2,
                )
            ],
            black_box_xapps=["massive_mimo_beamformer_xapp_capacity_push"],
            generated_by="test_reward_shaper",
        )

    def test_policy_store_derives_phase1_contracts_from_legacy_objects(self) -> None:
        store = PolicyStore()
        store.apply_update(
            PolicyUpdate(
                active_intent_text="increase throughput by 10 percent",
                structured_goal=self._make_goal(),
                xapp_selection_plan=self._make_plan(),
                reward_spec=self._make_reward_spec(),
                source_name="phase1_test_rapp",
            ),
            slot_id=6,
        )

        snapshot = store.get_snapshot()

        self.assertIsNotNone(snapshot.active_intent)
        self.assertEqual(snapshot.active_intent.intent_text, "increase throughput by 10 percent")
        self.assertEqual(snapshot.active_intent.last_change_reason, "legacy_policy_projection")

        self.assertIsNotNone(snapshot.validated_intent_template)
        self.assertEqual(snapshot.validated_intent_template.intent_id, "intent:phase1")
        self.assertEqual(snapshot.validated_intent_template.kpi_template[0].semantic_type, "max")
        self.assertEqual(snapshot.validated_intent_template.kpi_template[1].semantic_type, "min")
        self.assertEqual(
            snapshot.validated_intent_template.kpi_template[0].normalization.unit,
            "mbps",
        )

        self.assertIsNotNone(snapshot.orchestration_plan_view)
        self.assertEqual(snapshot.orchestration_plan_view.x, self._make_plan().selected_xapps)
        self.assertIsNotNone(snapshot.orchestration_plan_view.chi)
        self.assertEqual(snapshot.orchestration_plan_view.chi.merge_scope, "field")
        self.assertIn(
            snapshot.orchestration_plan_view.chi.arbitration_mode,
            {"weighted_merge", "priority_override", "guardrail_first"},
        )

        self.assertIsNotNone(snapshot.safety_constraint_set)
        self.assertEqual(
            snapshot.safety_constraint_set.selection_plan_id,
            "selection:goal:phase1:g1",
        )

        self.assertIsNotNone(snapshot.rl_guidance)
        self.assertEqual(snapshot.rl_guidance.source_epoch, 0)
        self.assertEqual(snapshot.rl_guidance.derived_from_intent_id, "intent:phase1")
        self.assertFalse(snapshot.rl_guidance.fallback_used)
        self.assertIn("latency_bias", snapshot.rl_guidance.parameter_priority_vector)

    def test_policy_store_persists_phase1_contracts_across_epoch_reset(self) -> None:
        store = PolicyStore()
        store.apply_update(
            PolicyUpdate(
                active_intent_text="increase throughput by 10 percent",
                structured_goal=self._make_goal(),
                xapp_selection_plan=self._make_plan(),
                reward_spec=self._make_reward_spec(),
                source_name="phase1_test_rapp",
            ),
            slot_id=6,
        )

        store.start_new_epoch(slot_id=10)
        snapshot = store.get_snapshot()

        self.assertIsNotNone(snapshot.active_intent)
        self.assertEqual(snapshot.active_intent.intent_text, "increase throughput by 10 percent")
        self.assertIsNotNone(snapshot.validated_intent_template)
        self.assertEqual(snapshot.validated_intent_template.goal_id, "goal:phase1")
        self.assertIsNotNone(snapshot.orchestration_plan_view)
        self.assertEqual(snapshot.orchestration_plan_view.plan_id, "selection:goal:phase1:g1")
        self.assertIsNotNone(snapshot.rl_guidance)
        self.assertEqual(snapshot.rl_guidance.selection_plan_id, "selection:goal:phase1:g1")
        self.assertIsNotNone(snapshot.safety_constraint_set)
        self.assertEqual(
            snapshot.safety_constraint_set.rule_set_id,
            "safety:selection:goal:phase1:g1",
        )

    def test_policy_store_can_disable_implicit_rl_guidance_derivation(self) -> None:
        store = PolicyStore()
        store.derive_rl_guidance_when_missing = False
        store.apply_update(
            PolicyUpdate(
                active_intent_text="increase throughput by 10 percent",
                structured_goal=self._make_goal(),
                xapp_selection_plan=self._make_plan(),
                reward_spec=self._make_reward_spec(),
                source_name="phase1_test_rapp",
            ),
            slot_id=6,
        )

        snapshot = store.get_snapshot()

        self.assertIsNotNone(snapshot.reward_spec)
        self.assertIsNone(snapshot.rl_guidance)
        self.assertIsNotNone(snapshot.safety_constraint_set)


if __name__ == "__main__":
    unittest.main()
