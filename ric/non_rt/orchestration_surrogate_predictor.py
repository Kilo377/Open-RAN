"""Predictor interface for surrogate orchestration scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from models.intent_orchestration import (
    OrchestrationCandidatePlan,
    OrchestrationMemoryCase,
    a1_guidance_profile_payload,
)
from ric.non_rt.orchestration_memory_attribution import (
    attributed_failure_categories,
    attributed_failure_category_set,
    memory_case_outcome_quality,
)
from ric.near_rt.xapp_variant_catalog import canonical_xapp_category


@dataclass
class SurrogatePrediction:
    predictor_name: str = "heuristic_structured_surrogate_predictor_v1"
    predicted_intent_alignment: float = 0.0
    predicted_intent_loss: float = 0.0
    predicted_memory_support: float = 0.0
    predicted_failure_risk: float = 0.0
    predicted_conflict_cost: float = 0.0
    switching_cost: float = 0.0
    stability_risk: float = 0.0
    resource_pressure: float = 0.0
    notes: list[str] = field(default_factory=list)


class OrchestrationSurrogatePredictor:
    predictor_name = "base_surrogate_predictor"

    def predict(
        self,
        *,
        candidate_plan: OrchestrationCandidatePlan,
        requested_categories: list[str],
        base_categories: list[str],
        previous_selected_xapps: list[str],
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> SurrogatePrediction:
        raise NotImplementedError


class HeuristicOrchestrationSurrogatePredictor(OrchestrationSurrogatePredictor):
    predictor_name = "heuristic_structured_surrogate_predictor_v1"

    def __init__(
        self,
        *,
        max_cpu_cost: float = 6.0,
        max_mem_cost: float = 5.5,
    ) -> None:
        self.max_cpu_cost = float(max_cpu_cost)
        self.max_mem_cost = float(max_mem_cost)

    def _xapp_categories(self, selected_xapps: list[str]) -> list[str]:
        categories: list[str] = []
        for xapp_name in list(selected_xapps or []):
            normalized = canonical_xapp_category(str(xapp_name).strip())
            if normalized != "" and normalized not in categories:
                categories.append(normalized)
        return categories

    def predict(
        self,
        *,
        candidate_plan: OrchestrationCandidatePlan,
        requested_categories: list[str],
        base_categories: list[str],
        previous_selected_xapps: list[str],
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> SurrogatePrediction:
        selected_names = list(getattr(candidate_plan, "x", []) or [])
        feasibility_summary = getattr(candidate_plan, "feasibility_summary", None)
        selected_category_set = set(self._xapp_categories(selected_names))
        base_category_set = set(base_categories)
        previous_category_set = set(self._xapp_categories(previous_selected_xapps))

        predicted_intent_alignment = (
            float(len(selected_category_set & base_category_set))
            / float(max(len(base_category_set), 1))
            if len(base_category_set) > 0
            else 0.0
        )
        predicted_intent_loss = max(0.0, 1.0 - float(predicted_intent_alignment))

        success_category_set = set()
        for memory_case in list(retrieved_success_cases or []):
            success_category_set |= set(
                self._xapp_categories(list(getattr(memory_case, "selected_xapps", []) or []))
            )
        failure_category_set = set()
        for memory_case in list(retrieved_failure_cases or []):
            failure_category_set |= set(
                self._xapp_categories(list(getattr(memory_case, "selected_xapps", []) or []))
            )
        attributed_failure_category_names = attributed_failure_category_set(retrieved_failure_cases)
        if len(attributed_failure_category_names) > 0:
            failure_category_set = set(attributed_failure_category_names)

        predicted_memory_support = (
            float(len(selected_category_set & success_category_set))
            / float(max(len(success_category_set), 1))
            if len(success_category_set) > 0
            else 0.0
        )
        predicted_failure_risk = (
            float(len(selected_category_set & failure_category_set))
            / float(max(len(failure_category_set), 1))
            if len(failure_category_set) > 0
            else 0.0
        )
        switching_cost = (
            float(len(selected_category_set ^ previous_category_set))
            / float(max(len(selected_category_set | previous_category_set), 1))
            if len(previous_category_set) > 0
            else 0.0
        )

        scheduler_dominant_categories = {
            "latency_scheduler_xapp",
            "spectrum_optimizer_xapp",
            "ran_slicing_manager_xapp",
            "urllc_guard_xapp",
        }
        scheduler_conflict_count = int(len(selected_category_set & scheduler_dominant_categories))
        forbidden_combination_count = int(
            len(list(getattr(candidate_plan, "forbidden_combinations", []) or []))
        )
        predicted_conflict_cost = float(max(0, scheduler_conflict_count - 2) * 0.35) + (
            0.08 * float(forbidden_combination_count)
        )

        parameter_bound_count = int(len(list(getattr(candidate_plan, "B", []) or [])))
        stability_risk = float(max(0, len(selected_names) - 3) * 0.12) + (
            0.02 * max(0, parameter_bound_count - 6)
        )

        total_cpu_cost = float(getattr(feasibility_summary, "total_cpu_cost", 0.0) or 0.0)
        total_mem_cost = float(getattr(feasibility_summary, "total_mem_cost", 0.0) or 0.0)
        max_cpu_cost = float(
            getattr(feasibility_summary, "max_cpu_cost", self.max_cpu_cost) or self.max_cpu_cost
        )
        max_mem_cost = float(
            getattr(feasibility_summary, "max_mem_cost", self.max_mem_cost) or self.max_mem_cost
        )
        resource_pressure = 0.0
        if max_cpu_cost > 0.0:
            resource_pressure += max(0.0, min(1.0, total_cpu_cost / max_cpu_cost) - 0.75)
        if max_mem_cost > 0.0:
            resource_pressure += max(0.0, min(1.0, total_mem_cost / max_mem_cost) - 0.75)
        if not bool(getattr(feasibility_summary, "feasible", True)):
            resource_pressure += 0.40

        notes = [
            f"eta_profile={str(a1_guidance_profile_payload(getattr(candidate_plan, 'eta', None)).get('profile_name', '') or '')}",
            f"requested_categories={list(requested_categories or [])}",
        ]
        return SurrogatePrediction(
            predictor_name=self.predictor_name,
            predicted_intent_alignment=float(predicted_intent_alignment),
            predicted_intent_loss=float(predicted_intent_loss),
            predicted_memory_support=float(predicted_memory_support),
            predicted_failure_risk=float(predicted_failure_risk),
            predicted_conflict_cost=float(predicted_conflict_cost),
            switching_cost=float(switching_cost),
            stability_risk=float(stability_risk),
            resource_pressure=float(resource_pressure),
            notes=notes,
        )


class MemoryCalibratedSurrogatePredictor(OrchestrationSurrogatePredictor):
    """Case-calibrated surrogate predictor that refines heuristic estimates with memory traces."""

    predictor_name = "memory_calibrated_surrogate_predictor_v1"

    def __init__(
        self,
        *,
        base_predictor: OrchestrationSurrogatePredictor | None = None,
    ) -> None:
        self.base_predictor = base_predictor or HeuristicOrchestrationSurrogatePredictor()

    def _xapp_categories(self, selected_xapps: list[str]) -> list[str]:
        categories: list[str] = []
        for xapp_name in list(selected_xapps or []):
            normalized = canonical_xapp_category(str(xapp_name).strip())
            if normalized != "" and normalized not in categories:
                categories.append(normalized)
        return categories

    def _category_similarity(
        self,
        selected_categories: set[str],
        memory_case: OrchestrationMemoryCase,
    ) -> float:
        case_categories = set(
            self._xapp_categories(list(getattr(memory_case, "selected_xapps", []) or []))
        )
        if len(selected_categories) == 0 or len(case_categories) == 0:
            return 0.0
        union_count = len(selected_categories | case_categories)
        if union_count <= 0:
            return 0.0
        overlap_score = float(len(selected_categories & case_categories)) / float(union_count)
        retrieval_hits = int(getattr(memory_case, "retrieval_hits", 0) or 0)
        retention_score = float(getattr(memory_case, "retention_score", 0.0) or 0.0)
        confidence_boost = min(0.20, 0.03 * float(retrieval_hits)) + min(0.10, 0.15 * retention_score)
        if str(getattr(memory_case, "memory_type", "success") or "success") == "success":
            confidence_boost += 0.20 * memory_case_outcome_quality(memory_case)
        return float(max(0.0, min(1.0, overlap_score + confidence_boost)))

    def _weighted_case_metric(
        self,
        memory_cases: list[OrchestrationMemoryCase] | None,
        *,
        selected_categories: set[str],
        metric_getter,
    ) -> tuple[float, float]:
        weighted_sum = 0.0
        weight_total = 0.0
        for memory_case in list(memory_cases or []):
            similarity = self._category_similarity(selected_categories, memory_case)
            if similarity <= 0.0:
                continue
            weighted_sum += similarity * float(metric_getter(memory_case))
            weight_total += similarity
        if weight_total <= 1e-9:
            return 0.0, 0.0
        return float(weighted_sum / weight_total), float(weight_total)

    def _failure_reason_signal(
        self,
        memory_cases: list[OrchestrationMemoryCase] | None,
        *,
        selected_categories: set[str],
        keywords: tuple[str, ...],
    ) -> float:
        def _signal(memory_case: OrchestrationMemoryCase) -> float:
            reason = str(getattr(memory_case, "failure_reason", "") or "").strip().lower()
            summary = str(getattr(memory_case, "summary", "") or "").strip().lower()
            reflection = getattr(memory_case, "reflection", None)
            reflection_text = " ".join(
                [
                    str(getattr(reflection, "outcome_summary", "") or ""),
                    str(getattr(reflection, "self_critique", "") or ""),
                    str(getattr(reflection, "improvement_hypothesis", "") or ""),
                ]
            ).strip().lower()
            combined = " ".join([reason, summary, reflection_text]).strip()
            if combined == "":
                return 0.0
            if any(keyword in combined for keyword in keywords):
                return 1.0
            return 0.0

        weighted_value, _ = self._weighted_case_metric(
            memory_cases,
            selected_categories=selected_categories,
            metric_getter=_signal,
        )
        return float(max(0.0, min(1.0, weighted_value)))

    def predict(
        self,
        *,
        candidate_plan: OrchestrationCandidatePlan,
        requested_categories: list[str],
        base_categories: list[str],
        previous_selected_xapps: list[str],
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> SurrogatePrediction:
        base_prediction = self.base_predictor.predict(
            candidate_plan=candidate_plan,
            requested_categories=requested_categories,
            base_categories=base_categories,
            previous_selected_xapps=previous_selected_xapps,
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        selected_categories = set(
            self._xapp_categories(list(getattr(candidate_plan, "x", []) or []))
        )
        success_loss, success_support = self._weighted_case_metric(
            retrieved_success_cases,
            selected_categories=selected_categories,
            metric_getter=lambda case: float(
                getattr(case, "intent_loss_normalized", getattr(case, "goal_gap_score", 0.0)) or 0.0
            )
            + 0.35
            * (
                1.0
                - memory_case_outcome_quality(case)
            ),
        )
        relevant_failure_cases = [
            memory_case
            for memory_case in list(retrieved_failure_cases or [])
            if len(selected_categories & attributed_failure_categories(memory_case)) > 0
        ]
        failure_loss, failure_support = self._weighted_case_metric(
            relevant_failure_cases,
            selected_categories=selected_categories,
            metric_getter=lambda case: float(
                getattr(case, "intent_loss_normalized", getattr(case, "goal_gap_score", 0.0)) or 0.0
            ),
        )
        conflict_failure_signal = self._failure_reason_signal(
            relevant_failure_cases,
            selected_categories=selected_categories,
            keywords=("conflict", "forbidden", "mutex", "override"),
        )
        stability_failure_signal = self._failure_reason_signal(
            relevant_failure_cases,
            selected_categories=selected_categories,
            keywords=("stability", "oscillat", "stagnat", "freeze"),
        )
        predicted_intent_loss = float(base_prediction.predicted_intent_loss)
        if success_support > 0.0:
            predicted_intent_loss = 0.52 * predicted_intent_loss + 0.48 * float(success_loss)
        if failure_support > 0.0:
            predicted_intent_loss = 0.55 * predicted_intent_loss + 0.45 * float(failure_loss)
        predicted_intent_loss = float(max(0.0, min(2.0, predicted_intent_loss)))

        predicted_memory_support = float(
            max(
                base_prediction.predicted_memory_support,
                min(1.0, 0.50 * float(base_prediction.predicted_memory_support) + 0.20 * success_support),
            )
        )
        predicted_failure_risk = float(
            max(
                0.0,
                min(
                    1.5,
                    0.50 * float(base_prediction.predicted_failure_risk)
                    + 0.35 * failure_support
                    + 0.15 * conflict_failure_signal,
                ),
            )
        )
        predicted_conflict_cost = float(
            max(
                0.0,
                min(
                    2.0,
                    float(base_prediction.predicted_conflict_cost) + 0.45 * conflict_failure_signal,
                ),
            )
        )
        stability_risk = float(
            max(
                0.0,
                min(
                    2.0,
                    float(base_prediction.stability_risk) + 0.35 * stability_failure_signal,
                ),
            )
        )
        notes = list(getattr(base_prediction, "notes", []) or [])
        notes.extend(
            [
                f"memory_success_support={success_support:.3f}",
                f"memory_failure_support={failure_support:.3f}",
                f"conflict_failure_signal={conflict_failure_signal:.3f}",
                f"stability_failure_signal={stability_failure_signal:.3f}",
            ]
        )
        return SurrogatePrediction(
            predictor_name=self.predictor_name,
            predicted_intent_alignment=max(0.0, 1.0 - predicted_intent_loss),
            predicted_intent_loss=predicted_intent_loss,
            predicted_memory_support=predicted_memory_support,
            predicted_failure_risk=predicted_failure_risk,
            predicted_conflict_cost=predicted_conflict_cost,
            switching_cost=float(base_prediction.switching_cost),
            stability_risk=stability_risk,
            resource_pressure=float(base_prediction.resource_pressure),
            notes=notes,
        )


class LinearStatefulSurrogatePredictor(OrchestrationSurrogatePredictor):
    """Parameterized linear surrogate predictor loaded from a local state file."""

    predictor_name = "linear_stateful_surrogate_predictor_v1"

    def __init__(
        self,
        *,
        state: dict[str, object] | None = None,
        base_predictor: OrchestrationSurrogatePredictor | None = None,
    ) -> None:
        self.base_predictor = base_predictor or MemoryCalibratedSurrogatePredictor()
        normalized_state = dict(state or {})
        self.state_name = str(
            normalized_state.get("predictor_name", self.predictor_name) or self.predictor_name
        )
        self.outputs = {
            str(output_name): {
                str(key): float(value)
                for key, value in dict(coefficients or {}).items()
                if str(key).strip() != ""
            }
            for output_name, coefficients in dict(normalized_state.get("outputs", {}) or {}).items()
            if str(output_name).strip() != ""
        }

    @classmethod
    def from_state_file(
        cls,
        state_path: str | None,
        *,
        base_predictor: OrchestrationSurrogatePredictor | None = None,
    ) -> "LinearStatefulSurrogatePredictor":
        payload: dict[str, object] = {}
        normalized_path = str(state_path or "").strip()
        if normalized_path != "":
            try:
                payload = json.loads(Path(normalized_path).read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        return cls(state=payload, base_predictor=base_predictor)

    def _feature_map(
        self,
        *,
        base_prediction: SurrogatePrediction,
        candidate_plan: OrchestrationCandidatePlan,
    ) -> dict[str, float]:
        feasibility_summary = getattr(candidate_plan, "feasibility_summary", None)
        return {
            "bias": 1.0,
            "predicted_intent_alignment": float(base_prediction.predicted_intent_alignment),
            "predicted_intent_loss": float(base_prediction.predicted_intent_loss),
            "predicted_memory_support": float(base_prediction.predicted_memory_support),
            "predicted_failure_risk": float(base_prediction.predicted_failure_risk),
            "predicted_conflict_cost": float(base_prediction.predicted_conflict_cost),
            "switching_cost": float(base_prediction.switching_cost),
            "stability_risk": float(base_prediction.stability_risk),
            "resource_pressure": float(base_prediction.resource_pressure),
            "parameter_bound_count": float(len(list(getattr(candidate_plan, "B", []) or []))),
            "selected_xapp_count": float(len(list(getattr(candidate_plan, "x", []) or []))),
            "feasible_flag": 1.0 if bool(getattr(feasibility_summary, "feasible", True)) else 0.0,
            "projected_flag": 1.0 if bool(getattr(feasibility_summary, "projected", False)) else 0.0,
        }

    def _linear_output(
        self,
        output_name: str,
        *,
        base_value: float,
        feature_map: dict[str, float],
    ) -> float:
        coefficients = dict(self.outputs.get(output_name, {}) or {})
        if len(coefficients) == 0:
            return float(base_value)
        value = float(coefficients.get("intercept", base_value))
        for feature_name, feature_value in feature_map.items():
            if feature_name == "bias":
                continue
            value += float(coefficients.get(feature_name, 0.0) or 0.0) * float(feature_value)
        return float(value)

    def predict(
        self,
        *,
        candidate_plan: OrchestrationCandidatePlan,
        requested_categories: list[str],
        base_categories: list[str],
        previous_selected_xapps: list[str],
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> SurrogatePrediction:
        base_prediction = self.base_predictor.predict(
            candidate_plan=candidate_plan,
            requested_categories=requested_categories,
            base_categories=base_categories,
            previous_selected_xapps=previous_selected_xapps,
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        feature_map = self._feature_map(
            base_prediction=base_prediction,
            candidate_plan=candidate_plan,
        )
        predicted_intent_loss = max(
            0.0,
            self._linear_output(
                "predicted_intent_loss",
                base_value=float(base_prediction.predicted_intent_loss),
                feature_map=feature_map,
            ),
        )
        predicted_memory_support = max(
            0.0,
            self._linear_output(
                "predicted_memory_support",
                base_value=float(base_prediction.predicted_memory_support),
                feature_map=feature_map,
            ),
        )
        predicted_failure_risk = max(
            0.0,
            self._linear_output(
                "predicted_failure_risk",
                base_value=float(base_prediction.predicted_failure_risk),
                feature_map=feature_map,
            ),
        )
        predicted_conflict_cost = max(
            0.0,
            self._linear_output(
                "predicted_conflict_cost",
                base_value=float(base_prediction.predicted_conflict_cost),
                feature_map=feature_map,
            ),
        )
        stability_risk = max(
            0.0,
            self._linear_output(
                "stability_risk",
                base_value=float(base_prediction.stability_risk),
                feature_map=feature_map,
            ),
        )
        resource_pressure = max(
            0.0,
            self._linear_output(
                "resource_pressure",
                base_value=float(base_prediction.resource_pressure),
                feature_map=feature_map,
            ),
        )
        notes = list(getattr(base_prediction, "notes", []) or [])
        notes.append(f"linear_state_outputs={sorted(self.outputs.keys())}")
        return SurrogatePrediction(
            predictor_name=self.state_name,
            predicted_intent_alignment=max(0.0, 1.0 - predicted_intent_loss),
            predicted_intent_loss=float(predicted_intent_loss),
            predicted_memory_support=float(predicted_memory_support),
            predicted_failure_risk=float(predicted_failure_risk),
            predicted_conflict_cost=float(predicted_conflict_cost),
            switching_cost=float(base_prediction.switching_cost),
            stability_risk=float(stability_risk),
            resource_pressure=float(resource_pressure),
            notes=notes,
        )


def build_surrogate_predictor(
    *,
    predictor_name: str = "memory_calibrated_surrogate_predictor_v1",
    predictor_state_path: str | None = None,
    max_cpu_cost: float = 6.0,
    max_mem_cost: float = 5.5,
) -> OrchestrationSurrogatePredictor:
    normalized_name = str(predictor_name or "memory_calibrated_surrogate_predictor_v1").strip().lower()
    heuristic_predictor = HeuristicOrchestrationSurrogatePredictor(
        max_cpu_cost=max_cpu_cost,
        max_mem_cost=max_mem_cost,
    )
    if normalized_name in {"heuristic_structured_surrogate_predictor_v1", "heuristic"}:
        return heuristic_predictor
    memory_predictor = MemoryCalibratedSurrogatePredictor(
        base_predictor=heuristic_predictor,
    )
    if normalized_name in {"linear_stateful_surrogate_predictor_v1", "linear_stateful", "learned_linear"}:
        return LinearStatefulSurrogatePredictor.from_state_file(
            predictor_state_path,
            base_predictor=memory_predictor,
        )
    return memory_predictor
