"""Structured surrogate scorer for full-form orchestration candidates."""

from __future__ import annotations

from models.intent_orchestration import (
    CandidatePlanScore,
    OrchestrationCandidatePlan,
    OrchestrationMemoryCase,
    a1_guidance_profile_payload,
)
from ric.non_rt.orchestration_surrogate_predictor import (
    HeuristicOrchestrationSurrogatePredictor,
    OrchestrationSurrogatePredictor,
)


class OrchestrationSurrogateScorer:
    """Scores projected orchestration candidates using paper-aligned surrogate terms."""

    def __init__(
        self,
        *,
        max_cpu_cost: float = 6.0,
        max_mem_cost: float = 5.5,
        predictor: OrchestrationSurrogatePredictor | None = None,
    ) -> None:
        self.max_cpu_cost = float(max_cpu_cost)
        self.max_mem_cost = float(max_mem_cost)
        self.predictor = predictor or HeuristicOrchestrationSurrogatePredictor(
            max_cpu_cost=self.max_cpu_cost,
            max_mem_cost=self.max_mem_cost,
        )

    def score_candidate_plan(
        self,
        *,
        candidate_plan: OrchestrationCandidatePlan,
        requested_categories: list[str],
        base_categories: list[str],
        previous_selected_xapps: list[str],
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> CandidatePlanScore:
        selected_names = list(getattr(candidate_plan, "x", []) or [])
        candidate_id = str(getattr(candidate_plan, "candidate_id", "") or "")
        feasibility_summary = getattr(candidate_plan, "feasibility_summary", None)
        prediction = self.predictor.predict(
            candidate_plan=candidate_plan,
            requested_categories=requested_categories,
            base_categories=base_categories,
            previous_selected_xapps=previous_selected_xapps,
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        parameter_bound_count = int(len(list(getattr(candidate_plan, "B", []) or [])))
        total_cpu_cost = float(getattr(feasibility_summary, "total_cpu_cost", 0.0) or 0.0)
        total_mem_cost = float(getattr(feasibility_summary, "total_mem_cost", 0.0) or 0.0)

        total_score = (
            1.00 * float(prediction.predicted_intent_loss)
            + 0.50 * float(prediction.predicted_failure_risk)
            + 0.28 * float(prediction.switching_cost)
            + 0.28 * float(prediction.predicted_conflict_cost)
            + 0.26 * float(prediction.stability_risk)
            + 0.18 * float(prediction.resource_pressure)
            - 0.35 * float(prediction.predicted_memory_support)
        )
        projection_notes = list(getattr(feasibility_summary, "projection_notes", []) or [])
        return CandidatePlanScore(
            candidate_id=str(candidate_id),
            score_version="j_sel_surrogate_v3_predictor_objective",
            score_direction="minimize",
            predictor_name=str(getattr(prediction, "predictor_name", "") or ""),
            requested_categories=list(requested_categories),
            selected_xapps=list(selected_names),
            predicted_intent_alignment=float(prediction.predicted_intent_alignment),
            predicted_intent_loss=float(prediction.predicted_intent_loss),
            predicted_memory_support=float(prediction.predicted_memory_support),
            predicted_failure_risk=float(prediction.predicted_failure_risk),
            predicted_conflict_cost=float(prediction.predicted_conflict_cost),
            switching_cost=float(prediction.switching_cost),
            stability_risk=float(prediction.stability_risk),
            resource_pressure=float(prediction.resource_pressure),
            eta_profile_name=str(
                a1_guidance_profile_payload(getattr(candidate_plan, "eta", None)).get("profile_name", "")
                or ""
            ),
            chi_profile_name=str(getattr(getattr(candidate_plan, "chi", None), "profile_name", "") or ""),
            parameter_bound_count=int(parameter_bound_count),
            feasible=bool(getattr(feasibility_summary, "feasible", True)),
            projection_applied=bool(getattr(feasibility_summary, "projected", False)),
            total_cpu_cost=float(total_cpu_cost),
            total_mem_cost=float(total_mem_cost),
            total_score=float(total_score),
            selected=False,
            rationale=(
                f"intent_loss={float(prediction.predicted_intent_loss):.3f}, "
                f"memory_support={float(prediction.predicted_memory_support):.3f}, "
                f"failure_risk={float(prediction.predicted_failure_risk):.3f}, "
                f"conflict_cost={float(prediction.predicted_conflict_cost):.3f}, "
                f"switching_cost={float(prediction.switching_cost):.3f}, "
                f"stability_risk={float(prediction.stability_risk):.3f}, "
                f"resource_pressure={float(prediction.resource_pressure):.3f}, "
                f"projected={bool(getattr(feasibility_summary, 'projected', False))}, "
                f"projection_notes={projection_notes[:2]}"
            ),
        )
