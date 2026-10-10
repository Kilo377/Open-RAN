import unittest

from models.intent_orchestration import (
    FeasibilityConstraintSummary,
    OrchestrationCandidatePlan,
    OrchestrationMemoryCase,
)
from ric.non_rt.orchestration_surrogate_predictor import (
    HeuristicOrchestrationSurrogatePredictor,
    LinearStatefulSurrogatePredictor,
    MemoryCalibratedSurrogatePredictor,
)


class TestOrchestrationSurrogatePredictor(unittest.TestCase):
    def test_predictor_exposes_resource_pressure_term(self) -> None:
        predictor = HeuristicOrchestrationSurrogatePredictor(max_cpu_cost=6.0, max_mem_cost=5.5)
        prediction = predictor.predict(
            candidate_plan=OrchestrationCandidatePlan(
                candidate_id="candidate:test",
                x=[
                    "massive_mimo_beamformer_xapp_capacity_push",
                    "latency_scheduler_xapp_balanced",
                ],
                feasibility_summary=FeasibilityConstraintSummary(
                    feasible=True,
                    projected=False,
                    max_selected_xapps=4,
                    max_cpu_cost=6.0,
                    max_mem_cost=5.5,
                    total_cpu_cost=5.4,
                    total_mem_cost=4.8,
                ),
            ),
            requested_categories=["massive_mimo_beamformer_xapp", "latency_scheduler_xapp"],
            base_categories=["massive_mimo_beamformer_xapp", "latency_scheduler_xapp"],
            previous_selected_xapps=["traffic_steering_xapp_balanced"],
            retrieved_success_cases=[],
            retrieved_failure_cases=[],
        )

        self.assertEqual(
            prediction.predictor_name,
            "heuristic_structured_surrogate_predictor_v1",
        )
        self.assertGreaterEqual(prediction.resource_pressure, 0.0)
        self.assertGreaterEqual(prediction.predicted_intent_loss, 0.0)

    def test_memory_calibrated_predictor_uses_success_and_failure_cases(self) -> None:
        predictor = MemoryCalibratedSurrogatePredictor()
        prediction = predictor.predict(
            candidate_plan=OrchestrationCandidatePlan(
                candidate_id="candidate:test_memory",
                x=[
                    "massive_mimo_beamformer_xapp_capacity_push",
                    "latency_scheduler_xapp_balanced",
                ],
                feasibility_summary=FeasibilityConstraintSummary(
                    feasible=True,
                    projected=False,
                    max_selected_xapps=4,
                    max_cpu_cost=6.0,
                    max_mem_cost=5.5,
                    total_cpu_cost=4.8,
                    total_mem_cost=4.1,
                ),
            ),
            requested_categories=["massive_mimo_beamformer_xapp", "latency_scheduler_xapp"],
            base_categories=["massive_mimo_beamformer_xapp", "latency_scheduler_xapp"],
            previous_selected_xapps=["traffic_steering_xapp_balanced"],
            retrieved_success_cases=[
                OrchestrationMemoryCase(
                    case_id="success:1",
                    selected_xapps=[
                        "massive_mimo_beamformer_xapp_capacity_push",
                        "latency_scheduler_xapp_balanced",
                    ],
                    intent_loss_normalized=0.08,
                    retrieval_hits=2,
                    retention_score=0.6,
                )
            ],
            retrieved_failure_cases=[
                OrchestrationMemoryCase(
                    case_id="failure:1",
                    memory_type="failure",
                    selected_xapps=[
                        "massive_mimo_beamformer_xapp_capacity_push",
                        "latency_scheduler_xapp_balanced",
                    ],
                    intent_loss_normalized=0.45,
                    failure_reason="conflict oscillation under aggressive override",
                    retrieval_hits=1,
                    retention_score=0.4,
                )
            ],
        )

        self.assertEqual(
            prediction.predictor_name,
            "memory_calibrated_surrogate_predictor_v1",
        )
        self.assertGreaterEqual(prediction.predicted_conflict_cost, 0.0)
        self.assertGreaterEqual(prediction.stability_risk, 0.0)
        self.assertGreaterEqual(prediction.predicted_failure_risk, 0.0)

    def test_linear_stateful_predictor_overrides_base_outputs(self) -> None:
        predictor = LinearStatefulSurrogatePredictor(
            state={
                "predictor_name": "linear_stateful_surrogate_predictor_v1",
                "outputs": {
                    "predicted_intent_loss": {
                        "intercept": 0.1,
                        "predicted_intent_loss": 0.5,
                        "predicted_failure_risk": 0.2,
                    }
                },
            }
        )
        prediction = predictor.predict(
            candidate_plan=OrchestrationCandidatePlan(
                candidate_id="candidate:test_linear",
                x=["massive_mimo_beamformer_xapp_capacity_push"],
                feasibility_summary=FeasibilityConstraintSummary(
                    feasible=True,
                    projected=False,
                    max_selected_xapps=4,
                    max_cpu_cost=6.0,
                    max_mem_cost=5.5,
                    total_cpu_cost=2.0,
                    total_mem_cost=1.5,
                ),
            ),
            requested_categories=["massive_mimo_beamformer_xapp"],
            base_categories=["massive_mimo_beamformer_xapp"],
            previous_selected_xapps=[],
            retrieved_success_cases=[],
            retrieved_failure_cases=[],
        )

        self.assertEqual(
            prediction.predictor_name,
            "linear_stateful_surrogate_predictor_v1",
        )
        self.assertGreaterEqual(prediction.predicted_intent_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
