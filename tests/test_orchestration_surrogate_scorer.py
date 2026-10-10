import unittest

from models.intent_orchestration import (
    ArbitrationProfileRef,
    FeasibilityConstraintSummary,
    OrchestrationCandidatePlan,
    OrchestrationMemoryCase,
    RewardParameterRange,
)
from ric.non_rt.orchestration_surrogate_scorer import OrchestrationSurrogateScorer


class TestOrchestrationSurrogateScorer(unittest.TestCase):
    def test_surrogate_score_exposes_paper_aligned_components(self) -> None:
        scorer = OrchestrationSurrogateScorer(max_cpu_cost=6.0, max_mem_cost=5.5)
        candidate_plan = OrchestrationCandidatePlan(
            candidate_id="candidate:test",
            requested_xapps=[
                "massive_mimo_beamformer_xapp_capacity_push",
                "urllc_guard_xapp_balanced",
            ],
            x=[
                "massive_mimo_beamformer_xapp_capacity_push",
                "urllc_guard_xapp_balanced",
            ],
            B=[
                RewardParameterRange(
                    field_name="power_cap_dbm",
                    target_xapp_name="massive_mimo_beamformer_xapp_capacity_push",
                    scope="cell",
                    lower_bound=28.0,
                    upper_bound=30.5,
                )
            ],
            eta={"profile_name": "eta:throughput_bias"},
            chi=ArbitrationProfileRef(profile_name="chi:capacity_weighted_merge", defaulted=False),
            feasibility_summary=FeasibilityConstraintSummary(
                feasible=True,
                projected=True,
                max_selected_xapps=4,
                max_cpu_cost=6.0,
                max_mem_cost=5.5,
                total_cpu_cost=2.4,
                total_mem_cost=1.8,
                projection_notes=["kept requested portfolio"],
            ),
        )
        score = scorer.score_candidate_plan(
            candidate_plan=candidate_plan,
            requested_categories=["massive_mimo_beamformer_xapp", "urllc_guard_xapp"],
            base_categories=["massive_mimo_beamformer_xapp", "urllc_guard_xapp"],
            previous_selected_xapps=["traffic_steering_xapp_balanced"],
            retrieved_success_cases=[
                OrchestrationMemoryCase(
                    case_id="success:1",
                    selected_xapps=["massive_mimo_beamformer_xapp_capacity_push"],
                )
            ],
            retrieved_failure_cases=[
                OrchestrationMemoryCase(
                    case_id="failure:1",
                    selected_xapps=["urllc_guard_xapp_balanced"],
                )
            ],
        )

        self.assertEqual(score.score_version, "j_sel_surrogate_v3_predictor_objective")
        self.assertEqual(score.score_direction, "minimize")
        self.assertEqual(
            score.predictor_name,
            "heuristic_structured_surrogate_predictor_v1",
        )
        self.assertEqual(score.eta_profile_name, "eta:throughput_bias")
        self.assertEqual(score.chi_profile_name, "chi:capacity_weighted_merge")
        self.assertTrue(score.projection_applied)
        self.assertGreaterEqual(score.predicted_intent_loss, 0.0)
        self.assertGreaterEqual(score.predicted_conflict_cost, 0.0)
        self.assertGreaterEqual(score.stability_risk, 0.0)
        self.assertGreaterEqual(score.resource_pressure, 0.0)
        self.assertGreater(score.total_score, -1.0)


if __name__ == "__main__":
    unittest.main()
