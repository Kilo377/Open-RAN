"""Fallback builder for intent translation when validation/projection fails."""

from __future__ import annotations

from models.intent_orchestration import (
    FallbackReport,
    GoalKpiTarget,
    StructuredIntentGoal,
)
from models.observation import Observation
from ric.non_rt.intent_orchestration_helpers import stable_intent_id


class IntentTranslationFallbackBuilder:
    """Builds deterministic fallback goals with explicit fallback reasons."""

    def __init__(self, *, name: str = "intent_translation_rule_fallback") -> None:
        self.name = str(name or "intent_translation_rule_fallback")

    def build(
        self,
        *,
        observation: Observation,
        intent_text: str,
        active_intent_id: str,
        fallback_reason: str,
    ) -> tuple[StructuredIntentGoal, FallbackReport, dict[str, object]]:
        normalized_intent = str(intent_text or "").strip().lower()
        current_throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        current_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        current_queue = float(getattr(observation, "total_queue_bytes", 0.0) or 0.0)

        kpi_targets: list[GoalKpiTarget] = []
        summary_focus = "balanced_service_stability"

        if any(token in normalized_intent for token in ("时延", "latency", "delay", "urllc")):
            summary_focus = "latency_reliability"
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name="queue_bytes",
                    objective="minimize",
                    target_value=max(0.0, current_queue * 0.9),
                    weight=1.2,
                    unit="bytes",
                    priority=3,
                )
            )
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name="avg_bler",
                    objective="minimize",
                    target_value=max(0.0, min(current_bler, 0.1)),
                    tolerance=0.03,
                    weight=1.0,
                    unit="ratio",
                    priority=2,
                )
            )
        else:
            summary_focus = "throughput_reliability"
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=max(1.0, current_throughput * 1.1),
                    weight=1.4,
                    unit="mbps",
                    priority=3,
                )
            )
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name="avg_bler",
                    objective="minimize",
                    target_value=max(0.0, min(current_bler, 0.12)),
                    tolerance=0.03,
                    weight=1.0,
                    unit="ratio",
                    priority=2,
                )
            )

        intent_id = str(active_intent_id or stable_intent_id(intent_text))
        goal = StructuredIntentGoal(
            goal_id=f"goal:{intent_id.split(':', 1)[-1]}",
            intent_id=intent_id,
            intent_text=str(intent_text or ""),
            summary=f"Rule fallback goal focused on {summary_focus}.",
            horizon_slots=12,
            risk_posture="balanced",
            target_entities=["cluster"],
            kpi_targets=kpi_targets,
            constraints={
                "fallback_reason": str(fallback_reason or ""),
                "fallback_strategy": self.name,
            },
            generated_by=self.name,
        )
        report = FallbackReport(
            fallback_used=True,
            fallback_strategy=self.name,
            fallback_reason=str(fallback_reason or ""),
            notes=[f"fallback_triggered:{fallback_reason}", f"summary_focus:{summary_focus}"],
        )
        fallback_output = {
            "fallback_reason": str(fallback_reason or ""),
            "fallback_strategy": self.name,
            "fallback_goal_preview": {
                "goal_id": goal.goal_id,
                "intent_id": goal.intent_id,
                "summary": goal.summary,
                "kpi_names": [target.kpi_name for target in goal.kpi_targets],
            },
        }
        return goal, report, fallback_output
