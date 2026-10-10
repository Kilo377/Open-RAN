"""Non-RT rApp that translates user intent into structured goals/KPIs."""

from __future__ import annotations

import re
from dataclasses import asdict

from models.intent_orchestration import (
    FallbackReport,
    GoalKpiTarget,
    ProjectionReport,
    StructuredIntentGoal,
    ValidationReport,
    validated_intent_template_from_goal,
)
from models.observation import Observation
from models.policy import PolicySnapshot, PolicyUpdate
from ric.non_rt.intent_orchestration_helpers import (
    llm_json_dumps,
    observation_summary_for_llm,
    stable_intent_id,
)
from ric.non_rt.intent_translation_fallbacks import IntentTranslationFallbackBuilder
from ric.non_rt.intent_translation_projector import IntentTemplateProjector
from ric.non_rt.intent_translation_validator import IntentTemplateValidator
from ric.non_rt.ollama_gptoss_client import StructuredJsonLlmClient
from ric.non_rt.rapp_base import RAppBase
from ric.non_rt.translation_context_retriever import TranslationContextRetriever


class IntentGoalTranslatorRApp(RAppBase):
    def __init__(
        self,
        *,
        name: str = "intent_goal_translator_rapp",
        intent_text: str = "",
        validator_enabled: bool = True,
        llm_enabled: bool = True,
        llm_provider: str = "ollama",
        llm_api_key: str | None = None,
        llm_api_key_env_var: str = "DASHSCOPE_API_KEY",
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_timeout_s: float = 120.0,
        llm_startup_timeout_s: float = 600.0,
        llm_max_output_tokens: int = 1024,
        llm_request_retries: int = 1,
        llm_temperature: float = 0.0,
        llm_enable_thinking: bool = False,
        llm_reasoning_effort: str = "",
        llm_extra_body: dict[str, object] | None = None,
        hard_kpi_targets: list[dict[str, object]] | None = None,
        translation_embedding_backend_name: str = "hashed_feature_embedding_v1",
        translation_embedding_backend_state_path: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.intent_text = str(intent_text).strip()
        self.validator_enabled = bool(validator_enabled)
        self.client = StructuredJsonLlmClient(
            enabled=llm_enabled,
            provider=llm_provider,
            api_key=llm_api_key,
            api_key_env_var=llm_api_key_env_var,
            base_url=llm_base_url,
            model_name=llm_model,
            temperature=float(llm_temperature),
            timeout_s=float(llm_timeout_s),
            startup_timeout_s=float(llm_startup_timeout_s),
            max_output_tokens=int(llm_max_output_tokens),
            request_retries=int(llm_request_retries),
            enable_thinking=bool(llm_enable_thinking),
            reasoning_effort=llm_reasoning_effort,
            extra_body=llm_extra_body,
        )
        self.hard_kpi_targets = [
            dict(item)
            for item in list(hard_kpi_targets or [])
            if isinstance(item, dict)
        ]
        self.validator = IntentTemplateValidator()
        self.projector = IntentTemplateProjector()
        self.fallback_builder = IntentTranslationFallbackBuilder()
        self.context_retriever = TranslationContextRetriever(
            embedding_backend_name=translation_embedding_backend_name,
            embedding_backend_state_path=translation_embedding_backend_state_path,
        )
        self._last_translation_query_key = None
        self._last_retrieved_context_summary: dict[str, object] = {}
        self._last_retrieved_context_traces = []

    def _resolve_intent_text(self, policy_snapshot: PolicySnapshot | None) -> str:
        if self.intent_text != "":
            return self.intent_text
        if policy_snapshot is not None:
            snapshot_intent = str(getattr(policy_snapshot, "active_intent_text", "") or "").strip()
            if snapshot_intent != "":
                return snapshot_intent
        return ""

    def _coerce_numeric(self, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if text == "":
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match is None:
            return None
        return float(match.group(0))

    def _normalize_kpi_name(self, raw_name: object) -> str:
        normalized = str(raw_name or "").strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        alias_map = {
            "throughput": "total_throughput_mbps",
            "total_throughput": "total_throughput_mbps",
            "throughput_mbps": "total_throughput_mbps",
            "total_throughput_mbps": "total_throughput_mbps",
            "bler": "avg_bler",
            "avg_bler": "avg_bler",
            "sinr": "sinr",
            "sinr_db": "sinr",
            "avg_sinr_db": "sinr",
            "queue": "queue_bytes",
            "queue_bytes": "queue_bytes",
            "total_queue": "total_queue_bytes",
            "total_queue_bytes": "total_queue_bytes",
            "handover": "handover_risk",
            "handover_risk": "handover_risk",
            "mobility_risk": "handover_risk",
            "power": "power_cost",
            "power_cost": "power_cost",
            "energy": "power_cost",
        }
        return alias_map.get(normalized, normalized)

    def _normalize_constraints(self, raw_constraints: object) -> dict[str, object]:
        if isinstance(raw_constraints, dict):
            return raw_constraints
        if isinstance(raw_constraints, list):
            return {
                f"constraint_{index + 1}": str(item)
                for index, item in enumerate(raw_constraints)
                if str(item).strip() != ""
            }
        if raw_constraints is None:
            return {}
        text = str(raw_constraints).strip()
        if text == "":
            return {}
        return {"constraint_1": text}

    def _observation_kpi_value(self, observation: Observation, kpi_name: str) -> float:
        normalized = self._normalize_kpi_name(kpi_name)
        if normalized == "total_throughput_mbps":
            return float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        if normalized == "avg_bler":
            return float(getattr(observation, "avg_bler", 0.0) or 0.0)
        if normalized in {"queue_bytes", "total_queue_bytes"}:
            return float(getattr(observation, "total_queue_bytes", 0.0) or 0.0)
        if normalized == "sinr":
            return float(getattr(observation, "avg_sinr_db", 0.0) or 0.0)
        if normalized == "handover_risk":
            pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
            return min(1.0, pending / 24.0)
        if normalized == "power_cost":
            return float(getattr(observation, "power_cost", 0.0) or 0.0)
        return 0.0

    def _target_from_hard_spec(
        self,
        *,
        observation: Observation,
        kpi_name: str,
        objective: str,
        spec: dict[str, object],
    ) -> float | None:
        if "target_value" in spec:
            return self._coerce_numeric(spec.get("target_value"))
        current_value = self._observation_kpi_value(observation, kpi_name)
        delta_percent = self._coerce_numeric(
            spec.get("target_delta_percent", spec.get("relative_delta_percent"))
        )
        if delta_percent is not None:
            return max(0.0, current_value * (1.0 + float(delta_percent) / 100.0))
        improvement_percent = self._coerce_numeric(spec.get("improvement_percent"))
        if improvement_percent is not None:
            normalized_objective = str(objective or "").strip().lower()
            direction = 1.0 if normalized_objective in {"maximize", "max"} else -1.0
            return max(0.0, current_value * (1.0 + direction * float(improvement_percent) / 100.0))
        delta_absolute = self._coerce_numeric(spec.get("target_delta_absolute"))
        if delta_absolute is not None:
            return max(0.0, current_value + float(delta_absolute))
        return None

    def _apply_hard_kpi_targets(
        self,
        *,
        goal: StructuredIntentGoal,
        observation: Observation,
    ) -> StructuredIntentGoal:
        if len(self.hard_kpi_targets) == 0:
            return goal
        existing_by_kpi = {
            str(getattr(target, "kpi_name", "") or "").strip().lower(): target
            for target in list(getattr(goal, "kpi_targets", []) or [])
        }
        applied_targets: list[GoalKpiTarget] = []
        applied_specs: list[dict[str, object]] = []
        for spec in self.hard_kpi_targets:
            kpi_name = self._normalize_kpi_name(spec.get("kpi_name", ""))
            if kpi_name == "":
                continue
            objective = str(spec.get("objective", "maximize") or "maximize")
            target_value = self._target_from_hard_spec(
                observation=observation,
                kpi_name=kpi_name,
                objective=objective,
                spec=spec,
            )
            if target_value is None:
                continue
            tolerance = self._coerce_numeric(spec.get("tolerance"))
            weight = self._coerce_numeric(spec.get("weight"))
            priority = self._coerce_numeric(spec.get("priority"))
            hard_target = GoalKpiTarget(
                kpi_name=kpi_name,
                objective=objective,
                target_value=float(target_value),
                tolerance=tolerance,
                weight=float(weight if weight is not None else 2.0),
                unit=str(spec.get("unit", "") or ""),
                priority=int(priority if priority is not None else 5),
                hard=bool(spec.get("hard", True)),
            )
            existing_by_kpi[kpi_name] = hard_target
            applied_targets.append(hard_target)
            applied_specs.append(dict(spec))
        if len(applied_targets) == 0:
            return goal
        ordered_targets = list(existing_by_kpi.values())
        constraints = dict(getattr(goal, "constraints", {}) or {})
        constraints["hard_kpi_targets_applied"] = True
        constraints["hard_kpi_target_count"] = int(len(applied_targets))
        constraints["hard_kpi_target_specs"] = applied_specs
        return StructuredIntentGoal(
            goal_id=str(getattr(goal, "goal_id", "") or ""),
            intent_id=str(getattr(goal, "intent_id", "") or ""),
            intent_text=str(getattr(goal, "intent_text", "") or ""),
            summary=str(getattr(goal, "summary", "") or "") + " Hard KPI targets injected from experiment config.",
            horizon_slots=int(getattr(goal, "horizon_slots", 10) or 10),
            risk_posture=str(getattr(goal, "risk_posture", "balanced") or "balanced"),
            target_entities=list(getattr(goal, "target_entities", []) or ["cluster"]),
            kpi_targets=ordered_targets,
            constraints=constraints,
            generated_by=str(getattr(goal, "generated_by", "heuristic") or "heuristic") + "+hard_kpi",
        )

    def _build_user_prompt(self, observation: Observation, intent_text: str) -> str:
        example_output = {
            "goal_id": "goal:throughput_plus_10pct",
            "summary": "Increase cluster throughput by about 10 percent while keeping reliability stable.",
            "horizon_slots": 12,
            "risk_posture": "balanced",
            "target_entities": ["cluster"],
            "kpi_targets": [
                {
                    "kpi_name": "total_throughput_mbps",
                    "objective": "maximize",
                    "target_value": 8.8,
                    "weight": 1.4,
                    "unit": "mbps",
                    "priority": 3,
                },
                {
                    "kpi_name": "avg_bler",
                    "objective": "minimize",
                    "target_value": 0.18,
                    "tolerance": 0.03,
                    "weight": 1.0,
                    "unit": "ratio",
                    "priority": 2,
                },
            ],
            "constraints": {
                "avoid_extreme_power_increase": True,
            },
        }
        retrieved_context_summary = dict(self._last_retrieved_context_summary or {})
        retrieved_context_traces = [
            {
                "source_id": str(getattr(trace, "source_id", "") or ""),
                "source_type": str(getattr(trace, "source_type", "") or ""),
                "relevance_score": float(getattr(trace, "relevance_score", 0.0) or 0.0),
                "summary": str(getattr(trace, "summary", "") or ""),
                "payload": dict(getattr(trace, "payload", {}) or {}),
            }
            for trace in list(self._last_retrieved_context_traces or [])
        ]
        return llm_json_dumps(
            {
                "intent": intent_text,
                "observation_summary": observation_summary_for_llm(observation),
                "retrieved_context": {
                    "query_id": str(getattr(self._last_translation_query_key, "query_id", "") or ""),
                    "summary": retrieved_context_summary,
                    "top_items": retrieved_context_traces,
                },
                "output_rules": [
                    "Return exactly one JSON object.",
                    "Use only these KPI names when applicable: total_throughput_mbps, avg_bler, queue_bytes, total_queue_bytes, sinr, handover_risk, power_cost.",
                    "horizon_slots must be an integer.",
                    "target_value, tolerance, weight, and priority should be numeric when present.",
                    "constraints should be a JSON object.",
                ],
                "example_output": example_output,
            }
        )

    def _refresh_translation_context(
        self,
        *,
        observation: Observation,
        intent_text: str,
        active_intent_id: str,
    ) -> None:
        (
            self._last_translation_query_key,
            self._last_retrieved_context_traces,
            self._last_retrieved_context_summary,
        ) = self.context_retriever.retrieve(
            intent_text=intent_text,
            intent_id=active_intent_id,
            observation=observation,
            top_k=5,
        )

    def _build_goal_from_payload_minimal(
        self,
        *,
        payload: dict[str, object],
        intent_text: str,
        active_intent_id: str,
    ) -> StructuredIntentGoal:
        raw_kpi_targets = payload.get("kpi_targets", [])
        if not isinstance(raw_kpi_targets, list) or len(raw_kpi_targets) == 0:
            raise RuntimeError("minimal_parse_missing_kpi_targets")

        kpi_targets: list[GoalKpiTarget] = []
        for entry in raw_kpi_targets:
            if not isinstance(entry, dict):
                continue
            kpi_name = self._normalize_kpi_name(entry.get("kpi_name", ""))
            target_value = self._coerce_numeric(entry.get("target_value"))
            if kpi_name == "" or target_value is None:
                continue
            tolerance = self._coerce_numeric(entry.get("tolerance"))
            weight = self._coerce_numeric(entry.get("weight"))
            priority = self._coerce_numeric(entry.get("priority"))
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name=kpi_name,
                    objective=str(entry.get("objective", "maximize") or "maximize"),
                    target_value=float(target_value),
                    tolerance=tolerance,
                    weight=(weight if weight is not None else 1.0),
                    unit=str(entry.get("unit", "") or ""),
                    priority=int(priority or 0),
                )
            )
        if len(kpi_targets) == 0:
            raise RuntimeError("minimal_parse_no_valid_kpis")

        goal_id = str(payload.get("goal_id", "") or f"goal:{active_intent_id.split(':', 1)[-1]}")
        raw_target_entities = payload.get("target_entities", ["cluster"])
        if isinstance(raw_target_entities, list) and len(raw_target_entities) > 0:
            target_entities = [str(item) for item in raw_target_entities if str(item).strip() != ""]
        elif str(raw_target_entities).strip() != "":
            target_entities = [str(raw_target_entities).strip()]
        else:
            target_entities = ["cluster"]
        horizon_slots = int(self._coerce_numeric(payload.get("horizon_slots", 12)) or 12)
        return StructuredIntentGoal(
            goal_id=goal_id,
            intent_id=active_intent_id,
            intent_text=intent_text,
            summary=str(payload.get("summary", "Minimally parsed structured RAN goal") or "Minimally parsed structured RAN goal"),
            horizon_slots=max(1, horizon_slots),
            risk_posture=str(payload.get("risk_posture", "balanced") or "balanced"),
            target_entities=target_entities,
            kpi_targets=kpi_targets,
            constraints=self._normalize_constraints(payload.get("constraints", {})),
            generated_by=f"{self.client.generated_by_tag}_minimal_parse",
        )

    def _minimal_default_goal(
        self,
        *,
        observation: Observation,
        intent_text: str,
        active_intent_id: str,
        reason: str,
    ) -> StructuredIntentGoal:
        current_throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        current_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        current_queue = float(getattr(observation, "total_queue_bytes", 0) or 0.0)
        return StructuredIntentGoal(
            goal_id=f"goal:{active_intent_id.split(':', 1)[-1]}",
            intent_id=active_intent_id,
            intent_text=intent_text,
            summary=f"Minimal default goal generated without formal validator chain ({reason}).",
            horizon_slots=12,
            risk_posture="balanced",
            target_entities=["cluster"],
            kpi_targets=[
                GoalKpiTarget(
                    kpi_name="total_throughput_mbps",
                    objective="maximize",
                    target_value=max(current_throughput * 1.05, current_throughput, 1.0),
                    weight=1.2,
                    unit="mbps",
                    priority=3,
                ),
                GoalKpiTarget(
                    kpi_name="avg_bler",
                    objective="minimize",
                    target_value=max(0.02, current_bler if current_bler > 0.0 else 0.10),
                    tolerance=0.03,
                    weight=1.0,
                    unit="ratio",
                    priority=2,
                ),
                GoalKpiTarget(
                    kpi_name="queue_bytes",
                    objective="minimize",
                    target_value=max(0.0, current_queue * 0.95),
                    weight=0.6,
                    unit="bytes",
                    priority=1,
                ),
            ],
            constraints={"source": "minimal_default_goal", "reason": str(reason)},
            generated_by=f"{self.name}_minimal_default",
        )

    def _generate_raw_candidate(self, observation: Observation, intent_text: str) -> dict[str, object]:
        if intent_text == "":
            raise ValueError("Intent text is empty and cannot be translated into a structured goal.")
        payload = self.client.generate_json_or_raise(
            system_prompt=(
                "You are a RAN intent planner. Convert the input intent and current RAN summary "
                "into one structured goal JSON object. Return JSON only."
            ),
            user_prompt=self._build_user_prompt(observation, intent_text),
        )
        if not isinstance(payload, dict):
            raise RuntimeError(f"{self.client.model_label} returned an invalid goal payload.")
        return payload

    def _build_validator_outcome(
        self,
        *,
        validation_report: ValidationReport,
        generation_error: str = "",
    ) -> dict[str, object]:
        validator_outcome = asdict(validation_report)
        validator_outcome["validator_enabled"] = bool(self.validator_enabled)
        if generation_error != "":
            validator_outcome["generation_error"] = generation_error
        return validator_outcome

    def _determine_fallback_reason(
        self,
        *,
        raw_candidate: object,
        validation_report: ValidationReport,
        projector_output: dict[str, object],
    ) -> str:
        if not isinstance(raw_candidate, dict) or len(raw_candidate) == 0:
            return "empty candidate"

        raw_kpis = raw_candidate.get("kpi_targets")
        if isinstance(raw_kpis, list) and len(raw_kpis) == 0:
            return "empty candidate"

        retained_kpis = list(projector_output.get("retained_kpis", []) or [])
        if not validation_report.schema_ok:
            return "schema invalid"
        if len(retained_kpis) == 0 and not validation_report.admissibility_ok:
            return "all KPIs inadmissible"
        if len(retained_kpis) == 0 and not validation_report.direction_ok:
            return "inconsistent semantics"
        if len(retained_kpis) == 0 and not validation_report.range_ok:
            return "target infeasible"
        if not validation_report.direction_ok:
            return "inconsistent semantics"
        if not validation_report.range_ok:
            return "target infeasible"
        return "schema invalid"

    def _build_compact_goal_from_sketch(
        self,
        *,
        observation: Observation,
        intent_text: str,
        sketch: dict[str, object],
    ) -> StructuredIntentGoal:
        current_throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        current_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        current_queue = float(getattr(observation, "total_queue_bytes", 0) or 0.0)
        delta_percent = self._coerce_numeric(sketch.get("throughput_delta_percent"))
        if delta_percent is None:
            delta_percent = 0.0
        target_throughput = current_throughput * (1.0 + float(delta_percent) / 100.0)
        if current_throughput <= 0.0 and delta_percent > 0.0:
            target_throughput = max(1.0, float(delta_percent) / 10.0)
        if target_throughput <= 0.0:
            target_throughput = max(current_throughput, 0.1)

        protect_bler = bool(sketch.get("protect_bler", False))
        protect_queue = bool(sketch.get("protect_queue", True))
        minimize_power_cost = bool(sketch.get("minimize_power_cost", False))
        mobility_sensitive = bool(sketch.get("mobility_sensitive", False))
        risk_posture = str(sketch.get("risk_posture", "balanced") or "balanced")
        intent_focus = str(sketch.get("intent_focus", "throughput") or "throughput").lower()
        target_entities = ["cluster"]
        if "cell" in intent_focus:
            target_entities = ["cell"]

        kpi_targets: list[GoalKpiTarget] = [
            GoalKpiTarget(
                kpi_name="total_throughput_mbps",
                objective="maximize",
                target_value=float(target_throughput),
                tolerance=0.0 if abs(delta_percent) < 1e-6 else None,
                weight=1.4,
                unit="mbps",
                priority=3,
            )
        ]
        if protect_bler or current_bler > 0.0:
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name="avg_bler",
                    objective="minimize",
                    target_value=max(0.0, current_bler),
                    tolerance=0.03,
                    weight=1.0,
                    unit="ratio",
                    priority=2,
                )
            )
        if protect_queue:
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name="queue_bytes",
                    objective="minimize",
                    target_value=max(0.0, current_queue * 0.95),
                    tolerance=None,
                    weight=0.8,
                    unit="bytes",
                    priority=1,
                )
            )
        if minimize_power_cost:
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name="power_cost",
                    objective="minimize",
                    target_value=28.0,
                    weight=0.6,
                    unit="dbm_like",
                    priority=1,
                )
            )
        if mobility_sensitive:
            kpi_targets.append(
                GoalKpiTarget(
                    kpi_name="handover_risk",
                    objective="minimize",
                    target_value=0.25,
                    weight=0.7,
                    unit="score",
                    priority=1,
                )
            )

        intent_id = stable_intent_id(intent_text)
        return StructuredIntentGoal(
            goal_id=f"goal:{intent_id.split(':', 1)[-1]}",
            intent_id=intent_id,
            intent_text=intent_text,
            summary=f"Compact LLM goal focused on {intent_focus}.",
            horizon_slots=12,
            risk_posture=risk_posture,
            target_entities=target_entities,
            kpi_targets=kpi_targets,
            constraints={"source": "compact_llm_goal_sketch"},
            generated_by=f"{self.client.generated_by_tag}_compact",
        )

    def _try_llm_goal_compact(self, observation: Observation, intent_text: str) -> StructuredIntentGoal:
        payload = self.client.generate_json_or_raise(
            system_prompt=(
                "Return exactly one JSON object with keys: intent_focus, throughput_delta_percent, "
                "protect_bler, protect_queue, minimize_power_cost, mobility_sensitive, risk_posture."
            ),
            user_prompt=(
                f"intent: {intent_text}\n"
                f"current_throughput_mbps: {float(getattr(observation, 'total_throughput_mbps', 0.0) or 0.0):.4f}\n"
                f"current_bler: {float(getattr(observation, 'avg_bler', 0.0) or 0.0):.4f}\n"
                f"current_queue_bytes: {int(getattr(observation, 'total_queue_bytes', 0) or 0)}\n"
                "Return exactly one valid JSON object."
            ),
        )
        if not isinstance(payload, dict) or len(payload) == 0:
            raise RuntimeError(f"{self.client.model_label} compact goal sketch is empty.")
        return self._build_compact_goal_from_sketch(
            observation=observation,
            intent_text=intent_text,
            sketch=payload,
        )

    def _try_llm_goal(self, observation: Observation, intent_text: str) -> StructuredIntentGoal:
        if intent_text == "":
            raise ValueError("Intent text is empty and cannot be translated into a structured goal.")
        try:
            payload = self.client.generate_json_or_raise(
                system_prompt=(
                    "You are a RAN intent planner. Convert the input intent and current RAN summary "
                    "into one structured goal JSON object. Return JSON only."
                ),
                user_prompt=self._build_user_prompt(observation, intent_text),
            )
            if not isinstance(payload, dict):
                raise RuntimeError(f"{self.client.model_label} returned an invalid goal payload.")

            raw_kpi_targets = payload.get("kpi_targets", [])
            if not isinstance(raw_kpi_targets, list) or len(raw_kpi_targets) == 0:
                raise RuntimeError(
                    f"{self.client.model_label} must return a non-empty 'kpi_targets' list."
                )

            kpi_targets: list[GoalKpiTarget] = []
            for entry in raw_kpi_targets:
                if not isinstance(entry, dict):
                    continue
                kpi_name = self._normalize_kpi_name(entry.get("kpi_name", ""))
                if kpi_name == "":
                    continue
                target_value = self._coerce_numeric(entry.get("target_value"))
                tolerance = self._coerce_numeric(entry.get("tolerance"))
                weight = self._coerce_numeric(entry.get("weight"))
                priority = self._coerce_numeric(entry.get("priority"))
                kpi_targets.append(
                    GoalKpiTarget(
                        kpi_name=kpi_name,
                        objective=str(entry.get("objective", "maximize") or "maximize"),
                        target_value=target_value,
                        tolerance=tolerance,
                        weight=(weight if weight is not None else 1.0),
                        unit=str(entry.get("unit", "") or ""),
                        priority=int(priority or 0),
                    )
                )
            if len(kpi_targets) == 0:
                raise RuntimeError(f"{self.client.model_label} did not return any valid KPI targets.")

            intent_id = stable_intent_id(intent_text)
            goal_id = str(payload.get("goal_id", "") or f"goal:{intent_id.split(':', 1)[-1]}")
            raw_target_entities = payload.get("target_entities", ["cluster"])
            if isinstance(raw_target_entities, list) and len(raw_target_entities) > 0:
                target_entities = [str(item) for item in raw_target_entities if str(item).strip() != ""]
            elif str(raw_target_entities).strip() != "":
                target_entities = [str(raw_target_entities).strip()]
            else:
                target_entities = ["cluster"]
            horizon_slots = int(self._coerce_numeric(payload.get("horizon_slots", 12)) or 12)
            constraints = self._normalize_constraints(payload.get("constraints", {}))
            return StructuredIntentGoal(
                goal_id=goal_id,
                intent_id=intent_id,
                intent_text=intent_text,
                summary=str(
                    payload.get("summary", "LLM-translated structured RAN goal")
                    or "LLM-translated structured RAN goal"
                ),
                horizon_slots=max(1, horizon_slots),
                risk_posture=str(payload.get("risk_posture", "balanced") or "balanced"),
                target_entities=target_entities,
                kpi_targets=kpi_targets,
                constraints=constraints,
                generated_by=self.client.generated_by_tag,
            )
        except RuntimeError:
            return self._try_llm_goal_compact(observation, intent_text)

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None = None,
    ) -> PolicyUpdate:
        intent_text = self._resolve_intent_text(policy_snapshot)
        if intent_text == "":
            return PolicyUpdate(source_name=self.name)
        active_intent_id = str(getattr(policy_snapshot, "active_intent_id", "") or "").strip()
        if active_intent_id == "":
            active_intent_id = stable_intent_id(intent_text)
        self._refresh_translation_context(
            observation=observation,
            intent_text=intent_text,
            active_intent_id=active_intent_id,
        )
        raw_candidate: dict[str, object] = {}
        generation_error = ""
        try:
            raw_candidate = self._generate_raw_candidate(observation, intent_text)
        except Exception as exc:
            generation_error = str(exc)
            raw_candidate = {}

        if self.validator_enabled:
            validation_report = self.validator.validate(raw_candidate, observation)
            projected_goal, projection_report, projector_output = self.projector.project(
                raw_candidate,
                observation=observation,
                intent_text=intent_text,
                active_intent_id=active_intent_id,
            )

            fallback_report = FallbackReport()
            fallback_output: dict[str, object] = {}
            final_goal = projected_goal
            if final_goal is None:
                fallback_reason = self._determine_fallback_reason(
                    raw_candidate=raw_candidate,
                    validation_report=validation_report,
                    projector_output=projector_output,
                )
                final_goal, fallback_report, fallback_output = self.fallback_builder.build(
                    observation=observation,
                    intent_text=intent_text,
                    active_intent_id=active_intent_id,
                    fallback_reason=fallback_reason,
                )
                if generation_error != "":
                    fallback_report.notes.append(f"generation_error:{generation_error}")
                    fallback_output["generation_error"] = generation_error
        else:
            validation_report = ValidationReport(
                schema_ok=isinstance(raw_candidate, dict) and len(raw_candidate) > 0,
                admissibility_ok=True,
                direction_ok=True,
                range_ok=True,
                validator_name="disabled_minimal_parse",
                warnings=["formal_validator_chain_disabled"],
            )
            projection_report = ProjectionReport(
                projected=False,
                projector_name="disabled_passthrough",
                projection_target="raw_candidate_passthrough",
                notes=["formal_projector_disabled"],
            )
            projector_output = {
                "projector_enabled": False,
                "retained_kpis": list(raw_candidate.get("kpi_targets", []) or [])
                if isinstance(raw_candidate, dict)
                else [],
            }
            fallback_report = FallbackReport()
            fallback_output: dict[str, object] = {}
            try:
                final_goal = self._build_goal_from_payload_minimal(
                    payload=raw_candidate,
                    intent_text=intent_text,
                    active_intent_id=active_intent_id,
                )
            except Exception as exc:
                fallback_reason = str(exc) or "minimal_parse_failed"
                final_goal = self._minimal_default_goal(
                    observation=observation,
                    intent_text=intent_text,
                    active_intent_id=active_intent_id,
                    reason=fallback_reason,
                )
                fallback_report = FallbackReport(
                    fallback_used=True,
                    fallback_strategy="minimal_default_goal",
                    fallback_reason=fallback_reason,
                    notes=(["formal_validator_chain_disabled"] + ([f"generation_error:{generation_error}"] if generation_error != "" else [])),
                )
                fallback_output = {
                    "fallback_path": "minimal_default_goal",
                    "fallback_reason": fallback_reason,
                }
                if generation_error != "":
                    fallback_output["generation_error"] = generation_error
        final_goal = self._apply_hard_kpi_targets(goal=final_goal, observation=observation)
        validated_template = validated_intent_template_from_goal(
            final_goal,
            translation_query_key=self._last_translation_query_key,
            retrieved_context_summary=self._last_retrieved_context_summary,
            retrieved_context_traces=list(self._last_retrieved_context_traces or []),
            raw_llm_candidate=raw_candidate,
            validator_outcome=self._build_validator_outcome(
                validation_report=validation_report,
                generation_error=generation_error,
            ),
            projector_output=projector_output,
            fallback_output=fallback_output,
            validation_report=validation_report,
            projection_report=projection_report,
            fallback_report=fallback_report,
        )
        return PolicyUpdate(
            active_intent_text=intent_text,
            active_intent_id=active_intent_id,
            intent_changed_this_epoch=(
                getattr(policy_snapshot, "intent_changed_this_epoch", None)
                if policy_snapshot is not None
                else False
            ),
            translation_refreshed_this_epoch=True,
            structured_goal=final_goal,
            validated_intent_template=validated_template,
            policy_type_name="intent_goal_translation",
            policy_description="Translate raw user intent into structured RAN goals/KPIs",
            source_name=self.name,
        )
