import unittest

from models.action import ConfigAction
from models.intent_orchestration import (
    ArbitrationProfileRef,
    ExplorationHealthGate,
    ForbiddenFieldCombination,
    GoalKpiTarget,
    OrchestrationPlanView,
    RewardParameterRange,
    RewardSpec,
    RlGuidanceContract,
    SafetyConstraintSet,
    SafetyRuleReference,
    StructuredIntentGoal,
    XAppSelectionPlan,
)
from models.observation import CellObservation, Observation
from models.policy import PolicySnapshot
from ric.near_rt.goal_parameter_tuning_xapp import GoalParameterTuningXApp, _HeadOption


class TestSafeGoalParameterTuning(unittest.TestCase):
    def _make_goal(self) -> StructuredIntentGoal:
        return StructuredIntentGoal(
            goal_id="goal:safe_rl",
            intent_id="intent:safe_rl",
            intent_text="boost throughput while avoiding service collapse",
            summary="Constrained online tuning for throughput uplift.",
            risk_posture="balanced",
            target_entities=["cluster"],
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=40.0,
                    weight=1.0,
                ),
                GoalKpiTarget(
                    kpi_name="avg_bler",
                    objective="minimize",
                    target_value=0.10,
                    weight=0.7,
                ),
            ],
        )

    def _make_observation(self, *, avg_bler: float = 0.03) -> Observation:
        return Observation(
            slot_id=11,
            avg_sinr_db=16.5,
            avg_bler=avg_bler,
            total_queue_bytes=42000,
            total_throughput_mbps=30.0,
            cell_id="cell_0",
            current_codebook_id=0,
            current_beam_profile_id=0,
            tx_power_dbm=30.0,
            scheduler_mode="round_robin",
            cell_observations=[
                CellObservation(
                    cell_id="cell_0",
                    scheduler_policy_hint="balanced",
                    latency_bias=1.2,
                    beam_candidate_preference="coverage",
                    power_cap_dbm=30.0,
                    total_queue_bytes=42000,
                )
            ],
            connected_ue_count=10,
            handover_pending_ue_count=0,
        )

    def _make_plan(self) -> XAppSelectionPlan:
        return XAppSelectionPlan(
            plan_id="selection:goal:safe_rl:g1",
            goal_id="goal:safe_rl",
            selected_xapps=[
                "latency_scheduler_xapp_balanced",
                "massive_mimo_beamformer_xapp_capacity_push",
            ],
            hard_safety_envelope=[
                RewardParameterRange(
                    field_name="scheduler_policy_hint",
                    target_xapp_name="latency_scheduler_xapp_balanced",
                    scope="cell",
                    discrete_values=["balanced", "latency_bias"],
                ),
                RewardParameterRange(
                    field_name="latency_bias",
                    target_xapp_name="latency_scheduler_xapp_balanced",
                    scope="cell",
                    lower_bound=1.0,
                    upper_bound=1.4,
                    step=0.1,
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
                    lower_bound=29.0,
                    upper_bound=30.0,
                    step=0.5,
                    max_delta=0.5,
                ),
            ],
            forbidden_combinations=[
                ForbiddenFieldCombination(
                    field_values={
                        "scheduler_policy_hint": "latency_bias",
                        "beam_candidate_preference": "capacity",
                    },
                    scope="cell",
                    reason="no latency+capacity combo",
                )
            ],
            exploration_health_gate=ExplorationHealthGate(
                max_avg_bler=0.08,
                max_total_queue_bytes=100000,
                min_total_throughput_mbps=20.0,
                max_handover_pending_ue_count=1,
                freeze_slots_on_violation=3,
            ),
        )

    def _make_reward_spec(self) -> RewardSpec:
        return RewardSpec(
            reward_id="reward:safe_rl",
            goal_id="goal:safe_rl",
            selection_plan_id="selection:goal:safe_rl:g1",
            reward_terms={
                "throughput_gain": 1.0,
                "queue_reduction": 0.4,
                "bler_reduction": 0.8,
            },
            parameter_ranges=[
                RewardParameterRange(
                    field_name="scheduler_policy_hint",
                    target_xapp_name="latency_scheduler_xapp_balanced",
                    scope="cell",
                    discrete_values=["balanced", "latency_bias", "slice_bias"],
                ),
                RewardParameterRange(
                    field_name="latency_bias",
                    target_xapp_name="latency_scheduler_xapp_balanced",
                    scope="cell",
                    lower_bound=0.8,
                    upper_bound=2.0,
                    step=0.1,
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
                    lower_bound=24.0,
                    upper_bound=32.0,
                    step=0.5,
                ),
            ],
        )

    def _make_rl_guidance(self, *, strict_health_gate: bool = False) -> RlGuidanceContract:
        health_gate = ExplorationHealthGate(
            max_avg_bler=0.01 if strict_health_gate else 0.08,
            max_total_queue_bytes=100000,
            min_total_throughput_mbps=20.0,
            max_handover_pending_ue_count=1,
            freeze_slots_on_violation=3,
        )
        safety_rule_set = SafetyConstraintSet(
            rule_set_id="guidance_safety:goal:safe_rl",
            goal_id="goal:safe_rl",
            selection_plan_id="selection:goal:safe_rl:g1",
            selected_rule_refs=[
                SafetyRuleReference(
                    rule_id="rule:hard_safety_envelope",
                    config={"range_count": 4},
                ),
                SafetyRuleReference(
                    rule_id="rule:exploration_health_gate",
                    config={"max_avg_bler": health_gate.max_avg_bler},
                ),
            ],
            safety_envelope=list(self._make_plan().hard_safety_envelope),
            forbidden_combinations=list(self._make_plan().forbidden_combinations),
            exploration_health_gate=health_gate,
            generated_by="test_guidance",
        )
        return RlGuidanceContract(
            guidance_id="guidance:safe_rl",
            goal_id="goal:safe_rl",
            selection_plan_id="selection:goal:safe_rl:g1",
            guided_action_region=list(self._make_plan().hard_safety_envelope),
            reward_coefficients={
                "throughput_gain": 1.2,
                "queue_reduction": 0.6,
                "bler_reduction": 1.1,
                "handover_risk_reduction": 0.2,
                "power_cost_penalty": 0.1,
            },
            parameter_priority_vector={
                "latency_bias": 1.1,
                "beam_candidate_preference": 0.9,
                "power_cap_dbm": 0.8,
                "scheduler_policy_hint": 1.0,
            },
            selected_safety_rule_set=safety_rule_set,
            black_box_xapps=[],
            source_epoch=4,
            derived_from_intent_id="intent:safe_rl",
            fallback_used=False,
            generated_by="test_guidance",
        )

    def test_range_groups_intersect_hard_and_soft_ranges(self) -> None:
        app = GoalParameterTuningXApp()
        heads = app._range_groups(self._make_reward_spec(), self._make_plan())

        latency_head = next(head for head in heads if head.xapp_name == "latency_scheduler_xapp_balanced")
        latency_range = latency_head.field_ranges["latency_bias"]
        scheduler_range = latency_head.field_ranges["scheduler_policy_hint"]

        self.assertAlmostEqual(latency_range.lower_bound, 1.0)
        self.assertAlmostEqual(latency_range.upper_bound, 1.4)
        self.assertAlmostEqual(latency_range.max_delta, 0.2)
        self.assertListEqual(scheduler_range.discrete_values, ["balanced", "latency_bias"])

    def test_forbidden_combinations_are_filtered_from_joint_actions(self) -> None:
        app = GoalParameterTuningXApp(max_joint_actions=64, random_seed=7)
        policy_snapshot = PolicySnapshot(
            structured_goal=self._make_goal(),
            xapp_selection_plan=self._make_plan(),
            reward_spec=self._make_reward_spec(),
        )

        joint_actions = app._joint_action_library(
            observation=self._make_observation(),
            policy_snapshot=policy_snapshot,
            reward_spec=policy_snapshot.reward_spec,
            plan=policy_snapshot.xapp_selection_plan,
        )

        self.assertGreater(len(joint_actions), 0)
        for joint_action in joint_actions:
            for config_action in joint_action.config_actions:
                self.assertFalse(
                    getattr(config_action, "scheduler_policy_hint", None) == "latency_bias"
                    and getattr(config_action, "beam_candidate_preference", None) == "capacity"
                )

    def test_unhealthy_observation_uses_safe_baseline_fallback(self) -> None:
        app = GoalParameterTuningXApp(epsilon=1.0, min_epsilon=1.0, random_seed=11)
        policy_snapshot = PolicySnapshot(
            structured_goal=self._make_goal(),
            xapp_selection_plan=self._make_plan(),
            reward_spec=self._make_reward_spec(),
        )

        decision = app.run(
            observation=self._make_observation(avg_bler=0.20),
            policy_snapshot=policy_snapshot,
        )

        self.assertAlmostEqual(decision.config_action.latency_bias, 1.2)
        self.assertEqual(decision.config_action.scheduler_policy_hint, "balanced")
        analysis_constraint = decision.policy_constraints[0]
        self.assertFalse(bool(analysis_constraint.params["exploration_allowed"]))
        self.assertEqual(analysis_constraint.params["selection_mode"], "safe_baseline_fallback")
        self.assertEqual(analysis_constraint.params["guidance_mode"], "legacy_reward_spec")
        self.assertIn("avg_bler_exceeded", analysis_constraint.params["health_gate_block_reasons"])

    def test_guidance_contract_is_consumed_by_near_rt_xapp(self) -> None:
        app = GoalParameterTuningXApp(epsilon=1.0, min_epsilon=1.0, random_seed=13)
        policy_snapshot = PolicySnapshot(
            policy_epoch_id=4,
            structured_goal=self._make_goal(),
            xapp_selection_plan=self._make_plan(),
            orchestration_plan_view=OrchestrationPlanView(
                plan_id="selection:goal:safe_rl:g1",
                goal_id="goal:safe_rl",
                x=list(self._make_plan().selected_xapps),
                B=list(self._make_plan().hard_safety_envelope),
                chi=ArbitrationProfileRef(
                    profile_name="chi:test_guardrail",
                    composition_mode="parameter_tuning",
                    arbitration_mode="guardrail_first",
                    merge_scope="field",
                    override_order=["urllc_guard_xapp", "latency_scheduler_xapp"],
                    defaulted=False,
                ),
            ),
            reward_spec=self._make_reward_spec(),
            rl_guidance=self._make_rl_guidance(strict_health_gate=True),
        )

        decision = app.run(
            observation=self._make_observation(avg_bler=0.03),
            policy_snapshot=policy_snapshot,
        )

        analysis_constraint = decision.policy_constraints[0]
        self.assertEqual(analysis_constraint.params["guidance_mode"], "rl_guidance_contract")
        self.assertEqual(analysis_constraint.params["guidance_id"], "guidance:safe_rl")
        self.assertEqual(analysis_constraint.params["guidance_source_epoch"], 4)
        self.assertAlmostEqual(
            sum(analysis_constraint.params["parameter_priority_vector"].values()),
            1.0,
            places=5,
        )
        self.assertEqual(analysis_constraint.params["composition_profile_name"], "chi:test_guardrail")
        self.assertEqual(analysis_constraint.params["composition_mode"], "parameter_tuning")
        self.assertEqual(analysis_constraint.params["arbitration_mode"], "guardrail_first")
        self.assertEqual(
            analysis_constraint.params["override_order"],
            ["urllc_guard_xapp", "latency_scheduler_xapp"],
        )
        self.assertIn("arbitration_signals", analysis_constraint.params)
        self.assertIn("theta_t", analysis_constraint.params)
        self.assertIn("omega_t", analysis_constraint.params)
        self.assertIn("active_parameter_mask", analysis_constraint.params)
        self.assertIn("near_rt_runtime_state", analysis_constraint.params)
        self.assertGreaterEqual(
            int(analysis_constraint.params["active_parameter_count"]),
            1,
        )
        self.assertIn(
            "theta_state",
            analysis_constraint.params["near_rt_runtime_state"],
        )
        self.assertIn(
            "omega_state",
            analysis_constraint.params["near_rt_runtime_state"],
        )
        self.assertIn("qos_profile_id", analysis_constraint.params)
        self.assertGreaterEqual(
            len(list(analysis_constraint.params["arbitration_signals"] or [])),
            1,
        )
        self.assertIn(
            "rule:exploration_health_gate",
            analysis_constraint.params["selected_safety_rule_ids"],
        )
        self.assertEqual(analysis_constraint.params["selection_mode"], "safe_baseline_fallback")
        self.assertIn("avg_bler_exceeded", analysis_constraint.params["health_gate_block_reasons"])

    def test_priority_override_profile_controls_discrete_field_merge(self) -> None:
        app = GoalParameterTuningXApp()
        merged = app._merge_scope_group(
            [
                _HeadOption(
                    option_id="beam_capacity",
                    xapp_name="massive_mimo_beamformer_xapp_capacity_push",
                    category="massive_mimo_beamformer_xapp",
                    tuning_mode="rl_parameter",
                    merge_weight=0.4,
                    config_action=ConfigAction(
                        apply_scope="cell",
                        target_cell_id="cell_0",
                        beam_candidate_preference="capacity",
                        power_cap_dbm=29.0,
                        source_name="beam_head",
                    ),
                ),
                _HeadOption(
                    option_id="latency_guard",
                    xapp_name="latency_scheduler_xapp_balanced",
                    category="latency_scheduler_xapp",
                    tuning_mode="rl_parameter",
                    merge_weight=0.9,
                    config_action=ConfigAction(
                        apply_scope="cell",
                        target_cell_id="cell_0",
                        beam_candidate_preference="coverage",
                        power_cap_dbm=30.0,
                        source_name="latency_head",
                    ),
                ),
            ],
            arbitration_profile=ArbitrationProfileRef(
                profile_name="chi:test_priority_override",
                composition_mode="mixed",
                arbitration_mode="priority_override",
                merge_scope="field",
                override_order=["latency_scheduler_xapp", "massive_mimo_beamformer_xapp"],
                defaulted=False,
            ),
        )

        self.assertEqual(merged.beam_candidate_preference, "coverage")
        self.assertAlmostEqual(merged.power_cap_dbm, 30.0)

    def test_guardrail_first_prefers_guard_categories_when_fields_overlap(self) -> None:
        app = GoalParameterTuningXApp()
        merged = app._merge_scope_group(
            [
                _HeadOption(
                    option_id="beam_capacity",
                    xapp_name="massive_mimo_beamformer_xapp_capacity_push",
                    category="massive_mimo_beamformer_xapp",
                    tuning_mode="rl_parameter",
                    merge_weight=1.0,
                    config_action=ConfigAction(
                        apply_scope="cell",
                        target_cell_id="cell_0",
                        scheduler_policy_hint="slice_bias",
                        latency_bias=1.6,
                        source_name="beam_head",
                    ),
                ),
                _HeadOption(
                    option_id="urllc_guard",
                    xapp_name="urllc_guard_xapp_balanced",
                    category="urllc_guard_xapp",
                    tuning_mode="rl_parameter",
                    merge_weight=0.1,
                    config_action=ConfigAction(
                        apply_scope="cell",
                        target_cell_id="cell_0",
                        scheduler_policy_hint="latency_bias",
                        latency_bias=1.1,
                        source_name="guard_head",
                    ),
                ),
            ],
            arbitration_profile=ArbitrationProfileRef(
                profile_name="chi:test_guardrail_direct",
                composition_mode="mixed",
                arbitration_mode="guardrail_first",
                merge_scope="field",
                override_order=["urllc_guard_xapp", "massive_mimo_beamformer_xapp"],
                defaulted=False,
            ),
        )

        self.assertEqual(merged.scheduler_policy_hint, "latency_bias")
        self.assertAlmostEqual(merged.latency_bias, 1.1)

    def test_no_safety_shield_variant_ignores_health_gate_and_fallback(self) -> None:
        app = GoalParameterTuningXApp(
            epsilon=0.0,
            min_epsilon=0.0,
            random_seed=17,
            safety_shield_enabled=False,
        )
        policy_snapshot = PolicySnapshot(
            policy_epoch_id=4,
            structured_goal=self._make_goal(),
            xapp_selection_plan=self._make_plan(),
            reward_spec=self._make_reward_spec(),
            rl_guidance=self._make_rl_guidance(strict_health_gate=True),
        )

        decision = app.run(
            observation=self._make_observation(avg_bler=0.30),
            policy_snapshot=policy_snapshot,
        )

        analysis_constraint = decision.policy_constraints[0]
        self.assertTrue(bool(analysis_constraint.params["exploration_allowed"]))
        self.assertEqual(analysis_constraint.params["selection_mode"], "epsilon_greedy")
        self.assertEqual(analysis_constraint.params["health_gate_block_reasons"], [])
        self.assertFalse(bool(analysis_constraint.params["rl_safe_baseline_fallback"]) if "rl_safe_baseline_fallback" in analysis_constraint.params else False)
        self.assertEqual(analysis_constraint.params["hard_safety_envelope_count"], 0)
        self.assertEqual(analysis_constraint.params["forbidden_combination_count"], 0)
        self.assertFalse(bool(analysis_constraint.params["safety_shield_enabled"]))


if __name__ == "__main__":
    unittest.main()
