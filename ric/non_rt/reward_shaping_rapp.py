"""Non-RT rApp that generates reward specs and tuning ranges."""

from __future__ import annotations

from typing import Any

from models.intent_orchestration import (
    AmbientParameterDimension,
    AmbientParameterSpace,
    ArbitrationVariableSpec,
    ExplorationHealthGate,
    ForbiddenFieldCombination,
    RewardParameterRange,
    RewardSpec,
    RlGuidanceContract,
    SafetyConstraintSet,
    SafetyRuleReference,
    a1_guidance_profile_payload,
)
from models.observation import Observation
from models.policy import PolicySnapshot, PolicyUpdate
from ric.near_rt.xapp_variant_catalog import canonical_xapp_category
from ric.non_rt.intent_orchestration_helpers import (
    available_xapp_catalog_for_llm,
    default_reward_parameter_ranges,
    goal_gap_breakdown,
    llm_json_dumps,
    observation_summary_for_llm,
)
from ric.non_rt.ollama_gptoss_client import StructuredJsonLlmClient
from ric.non_rt.rapp_base import RAppBase


class RewardShapingRApp(RAppBase):
    _FIXED_REWARD_TEMPLATE_KEYS: tuple[str, ...] = (
        "throughput_gain",
        "queue_reduction",
        "bler_reduction",
        "handover_risk_reduction",
        "power_cost_penalty",
    )
    _FIELD_PRIORITY_HINTS: dict[str, tuple[str, ...]] = {
        "total_throughput_mbps": (
            "beam_candidate_preference",
            "power_cap_dbm",
            "slice_weight_adjustment",
            "scheduler_policy_hint",
        ),
        "avg_bler": (
            "latency_bias",
            "scheduler_policy_hint",
            "beam_candidate_preference",
            "power_cap_dbm",
        ),
        "queue_bytes": (
            "scheduler_policy_hint",
            "slice_weight_adjustment",
            "latency_bias",
            "traffic_steering_preference",
        ),
        "handover_risk": (
            "mobility_bias_db",
            "measurement_event_offset_db",
            "steering_hysteresis_db",
            "traffic_steering_preference",
        ),
        "power_cost": (
            "power_cap_dbm",
            "beam_candidate_preference",
            "cell_sleep_mode",
        ),
    }
    _SAFETY_RULE_LIBRARY: dict[str, str] = {
        "rule:hard_safety_envelope": "Apply the Non-RT hard safety envelope as the primary action bound.",
        "rule:forbidden_combinations": "Filter disallowed field combinations before near-RT execution.",
        "rule:exploration_health_gate": "Disable exploration and freeze when health thresholds are violated.",
        "rule:degraded_network_freeze": "Enter conservative online tuning when the network approaches degradation thresholds.",
        "rule:fallback_template_guard": "Use conservative guardrails when the validated intent template came from fallback.",
    }
    _REWARD_TERM_ALIASES: dict[str, tuple[str, ...]] = {
        "throughput_gain": (
            "throughput_gain",
            "throughput",
            "throughput_weight",
            "throughput_reward",
            "throughput_reward_weight",
            "total_throughput_mbps",
        ),
        "queue_reduction": (
            "queue_reduction",
            "queue",
            "queue_weight",
            "queue_reward",
            "queue_reduction_weight",
            "queue_bytes",
            "total_queue_bytes",
            "backlog_reduction",
        ),
        "bler_reduction": (
            "bler_reduction",
            "bler",
            "bler_weight",
            "bler_reward",
            "reliability",
            "reliability_gain",
            "avg_bler",
        ),
        "handover_risk_reduction": (
            "handover_risk_reduction",
            "handover",
            "handover_weight",
            "handover_stability",
            "mobility_stability",
            "steering_stability",
            "handover_risk",
        ),
        "power_cost_penalty": (
            "power_cost_penalty",
            "power_penalty",
            "energy_penalty",
            "power_cost",
            "energy_cost",
            "power_weight",
        ),
    }
    _KPI_ALIASES: dict[str, tuple[str, ...]] = {
        "total_throughput_mbps": (
            "total_throughput_mbps",
            "throughput_mbps",
            "throughput",
        ),
        "avg_bler": (
            "avg_bler",
            "bler",
        ),
        "queue_bytes": (
            "queue_bytes",
            "total_queue_bytes",
            "queue",
            "backlog_bytes",
        ),
        "handover_risk": (
            "handover_risk",
            "handover",
            "handover_pending_ue_count",
        ),
        "power_cost": (
            "power_cost",
            "energy_cost",
            "power_cap_dbm",
        ),
        "avg_sinr_db": (
            "avg_sinr_db",
            "sinr",
            "sinr_db",
        ),
    }
    _WRAPPER_KEYS = ("reward_spec", "output", "result", "data")
    _REWARD_MAPPING_KEYS = ("reward_terms", "reward_weights", "weights", "reward")
    _KPI_MAPPING_KEYS = ("kpi_weights", "kpis", "kpi_priorities", "kpi_importance")
    _BLACK_BOX_KEYS = ("black_box_xapps", "black_box", "output_shaping_xapps", "locked_xapps")

    def __init__(
        self,
        *,
        name: str = "reward_shaping_rapp",
        rl_guidance_enabled: bool = True,
        llm_enabled: bool = True,
        llm_provider: str = "ollama",
        llm_api_key: str | None = None,
        llm_api_key_env_var: str = "DASHSCOPE_API_KEY",
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_timeout_s: float = 120.0,
        llm_startup_timeout_s: float = 600.0,
        llm_max_output_tokens: int = 1536,
        llm_request_retries: int = 1,
        llm_temperature: float = 0.0,
        llm_enable_thinking: bool = False,
        llm_reasoning_effort: str = "",
        llm_extra_body: dict[str, object] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.rl_guidance_enabled = bool(rl_guidance_enabled)
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

    def _canonicalize_name(self, name: str) -> str:
        return str(name or "").strip().lower()

    def _canonical_kpi_name(self, kpi_name: str) -> str:
        normalized_name = self._canonicalize_name(kpi_name)
        for canonical_name, aliases in self._KPI_ALIASES.items():
            if normalized_name in aliases:
                return canonical_name
        return normalized_name

    def _goal_kpi_keys(self, goal) -> list[str]:
        goal_kpi_keys: list[str] = []
        for target in list(getattr(goal, "kpi_targets", []) or []):
            canonical_name = self._canonical_kpi_name(str(getattr(target, "kpi_name", "") or ""))
            if canonical_name == "" or canonical_name in goal_kpi_keys:
                continue
            goal_kpi_keys.append(canonical_name)
        if len(goal_kpi_keys) == 0:
            goal_kpi_keys = ["total_throughput_mbps", "avg_bler", "queue_bytes"]
        return goal_kpi_keys

    def _selected_xapp_context(self, plan) -> list[dict[str, object]]:
        selected_xapps = [
            str(item)
            for item in list(getattr(plan, "selected_xapps", []) or [])
            if str(item).strip() != ""
        ]
        catalog = {
            str(item.get("xapp_name", "")): item
            for item in available_xapp_catalog_for_llm()
            if str(item.get("xapp_name", "")).strip() != ""
        }
        selected_context: list[dict[str, object]] = []
        for xapp_name in selected_xapps:
            spec = catalog.get(xapp_name, {})
            kpis = spec.get("kpis", [])
            tunable_fields = spec.get("tunable_fields", [])
            selected_context.append(
                {
                    "xapp_name": xapp_name,
                    "xapp_category": canonical_xapp_category(xapp_name),
                    "kpis": list(kpis) if isinstance(kpis, list) else [],
                    "tunable_fields": list(tunable_fields) if isinstance(tunable_fields, list) else [],
                }
            )
        return selected_context

    def _compact_goal_payload(self, goal) -> dict[str, object]:
        return {
            "goal_id": str(getattr(goal, "goal_id", "") or ""),
            "summary": str(getattr(goal, "summary", "") or ""),
            "risk_posture": str(getattr(goal, "risk_posture", "balanced") or "balanced"),
            "kpi_targets": [
                {
                    "kpi_name": self._canonical_kpi_name(str(getattr(target, "kpi_name", "") or "")),
                    "objective": str(getattr(target, "objective", "maximize") or "maximize"),
                    "target_value": getattr(target, "target_value", None),
                    "weight": float(getattr(target, "weight", 1.0) or 1.0),
                }
                for target in list(getattr(goal, "kpi_targets", []) or [])
                if str(getattr(target, "kpi_name", "") or "").strip() != ""
            ],
        }

    def _compact_selection_payload(self, plan) -> dict[str, object]:
        selected_xapp_context = self._selected_xapp_context(plan)
        return {
            "plan_id": str(getattr(plan, "plan_id", "") or ""),
            "selected_xapps": [item["xapp_name"] for item in selected_xapp_context],
            "selected_categories": [item["xapp_category"] for item in selected_xapp_context],
            "selected_xapp_context": selected_xapp_context,
            "orchestration_mode": str(
                getattr(plan, "orchestration_mode", "parallel_with_guardrail") or "parallel_with_guardrail"
            ),
        }

    def _compact_observation_payload(self, observation: Observation) -> dict[str, object]:
        summary = observation_summary_for_llm(observation)
        return {
            "slot_id": int(summary.get("slot_id", -1) or -1),
            "avg_sinr_db": float(summary.get("avg_sinr_db", 0.0) or 0.0),
            "avg_bler": float(summary.get("avg_bler", 0.0) or 0.0),
            "total_queue_bytes": int(summary.get("total_queue_bytes", 0) or 0),
            "total_throughput_mbps": float(summary.get("total_throughput_mbps", 0.0) or 0.0),
            "handover_pending_ue_count": int(summary.get("handover_pending_ue_count", 0) or 0),
        }

    def _generation_user_payload(self, *, goal, plan, observation: Observation) -> dict[str, object]:
        example_reward_input = {
            "goal": {
                "goal_id": "goal:demo_throughput_balanced",
                "summary": "Increase throughput while keeping BLER stable.",
                "risk_posture": "balanced",
                "kpi_targets": [
                    {"kpi_name": "total_throughput_mbps", "objective": "maximize", "target_value": 8.8},
                    {"kpi_name": "avg_bler", "objective": "minimize", "target_value": 0.18},
                ],
            },
            "selection_plan": {
                "plan_id": "selection:goal:demo:g1",
                "selected_xapps": [
                    "massive_mimo_beamformer_xapp_capacity_push",
                    "traffic_steering_xapp_quality_first",
                    "urllc_guard_xapp_balanced",
                ],
                "selected_categories": [
                    "massive_mimo_beamformer_xapp",
                    "traffic_steering_xapp",
                    "urllc_guard_xapp",
                ],
            },
            "observation_summary": {
                "avg_sinr_db": 10.5,
                "avg_bler": 0.21,
                "total_queue_bytes": 24000,
                "total_throughput_mbps": 7.4,
            },
        }
        example_reward_output = {
            "reward_terms": {
                "throughput_gain": 1.4,
                "queue_reduction": 0.6,
                "bler_reduction": 1.2,
                "handover_risk_reduction": 0.3,
                "power_cost_penalty": 0.2,
            },
            "kpi_weights": {
                "total_throughput_mbps": 1.5,
                "avg_bler": 1.2,
                "queue_bytes": 0.6,
            },
            "black_box_xapps": [],
        }
        goal_kpi_keys = self._goal_kpi_keys(goal)
        return {
            "task": "Generate reward weights and black-box assignments for the selected xApp portfolio.",
            "output_rules": [
                "Return exactly one JSON object.",
                "Use top-level keys: reward_terms, kpi_weights, black_box_xapps.",
                "reward_terms must be a non-empty object with numeric values.",
                "Prefer these reward_terms keys: throughput_gain, queue_reduction, bler_reduction, handover_risk_reduction, power_cost_penalty.",
                "kpi_weights must be a non-empty object with numeric values.",
                "black_box_xapps must be a list and every name must come from selected_xapps.",
                "If a reward term is less important, give it a small positive weight instead of omitting it.",
                "Do not include markdown or prose.",
            ],
            "required_schema_hint": {
                "reward_terms": {
                    "throughput_gain": "float",
                    "queue_reduction": "float",
                    "bler_reduction": "float",
                    "handover_risk_reduction": "float",
                    "power_cost_penalty": "float",
                },
                "kpi_weights": {key: "float" for key in goal_kpi_keys},
                "black_box_xapps": ["subset_of_selected_xapps"],
            },
            "few_shot_example": {
                "input": example_reward_input,
                "output": example_reward_output,
            },
            "input": {
                "goal": self._compact_goal_payload(goal),
                "selection_plan": self._compact_selection_payload(plan),
                "observation_summary": self._compact_observation_payload(observation),
            },
        }

    def _repair_user_payload(
        self,
        *,
        goal,
        plan,
        observation: Observation,
        invalid_payload: dict[str, Any],
    ) -> dict[str, object]:
        goal_kpi_keys = self._goal_kpi_keys(goal)
        return {
            "task": "Repair the invalid reward-shaping JSON into the required schema.",
            "output_rules": [
                "Return exactly one JSON object.",
                "Use top-level keys: reward_terms, kpi_weights, black_box_xapps.",
                "reward_terms and kpi_weights must both be non-empty objects with numeric values.",
                "black_box_xapps must be a list and every name must come from selected_xapps.",
                "Do not include markdown or prose.",
            ],
            "required_schema_hint": {
                "reward_terms": {
                    "throughput_gain": "float",
                    "queue_reduction": "float",
                    "bler_reduction": "float",
                    "handover_risk_reduction": "float",
                    "power_cost_penalty": "float",
                },
                "kpi_weights": {key: "float" for key in goal_kpi_keys},
                "black_box_xapps": ["subset_of_selected_xapps"],
            },
            "context": {
                "goal": self._compact_goal_payload(goal),
                "selection_plan": self._compact_selection_payload(plan),
                "observation_summary": self._compact_observation_payload(observation),
            },
            "invalid_payload": invalid_payload,
        }

    def _coerce_float(self, value: object) -> float | None:
        candidate = value
        if isinstance(candidate, dict):
            for nested_key in ("weight", "value", "score"):
                if nested_key in candidate:
                    candidate = candidate[nested_key]
                    break
        if not isinstance(candidate, (int, float, str)):
            return None
        try:
            return float(candidate)
        except (TypeError, ValueError):
            return None

    def _to_mapping(self, value: object) -> dict[str, object]:
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        if not isinstance(value, list):
            return {}
        mapping: dict[str, object] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("term", item.get("kpi_name", item.get("key", None))))
            if name is None:
                continue
            mapping[str(name)] = item.get("weight", item.get("value", item.get("score", None)))
        return mapping

    def _payload_sections(self, payload: dict[str, Any]) -> list[dict[str, object]]:
        sections: list[dict[str, object]] = [payload]
        for wrapper_key in self._WRAPPER_KEYS:
            wrapped = payload.get(wrapper_key)
            if isinstance(wrapped, dict):
                sections.append(wrapped)
        return sections

    def _extract_mapping_candidates(
        self,
        *,
        payload: dict[str, Any],
        field_names: tuple[str, ...],
        alias_map: dict[str, tuple[str, ...]],
    ) -> list[dict[str, object]]:
        alias_names = {alias for aliases in alias_map.values() for alias in aliases}
        candidates: list[dict[str, object]] = []
        for section in self._payload_sections(payload):
            for field_name in field_names:
                candidate = self._to_mapping(section.get(field_name))
                if len(candidate) > 0:
                    candidates.append(candidate)
            section_keys = {self._canonicalize_name(key) for key in section.keys()}
            if len(section_keys.intersection(alias_names)) > 0:
                candidates.append(section)
        return candidates

    def _normalize_alias_mapping(
        self,
        raw_mapping: dict[str, object],
        alias_map: dict[str, tuple[str, ...]],
    ) -> dict[str, float]:
        normalized_mapping: dict[str, float] = {}
        flattened_mapping = {
            self._canonicalize_name(key): value
            for key, value in raw_mapping.items()
        }
        for canonical_name, aliases in alias_map.items():
            for alias in aliases:
                if alias not in flattened_mapping:
                    continue
                numeric_value = self._coerce_float(flattened_mapping[alias])
                if numeric_value is None:
                    continue
                normalized_mapping[canonical_name] = float(numeric_value)
                break
        return normalized_mapping

    def _normalize_black_box_xapps(
        self,
        raw_value: object,
        *,
        selected_xapps: list[str],
    ) -> list[str]:
        if not isinstance(raw_value, list):
            return []
        selected_lookup = {str(xapp_name): str(xapp_name) for xapp_name in selected_xapps}
        selected_by_category = {
            canonical_xapp_category(xapp_name): xapp_name
            for xapp_name in selected_xapps
        }
        normalized_black_box_xapps: list[str] = []
        seen: set[str] = set()
        for item in raw_value:
            normalized_name = str(item or "").strip()
            if normalized_name == "":
                continue
            mapped_name = selected_lookup.get(normalized_name)
            if mapped_name is None:
                mapped_name = selected_by_category.get(canonical_xapp_category(normalized_name))
            if mapped_name is None or mapped_name in seen:
                continue
            normalized_black_box_xapps.append(mapped_name)
            seen.add(mapped_name)
        return normalized_black_box_xapps

    def _truncate_payload_excerpt(self, payload: dict[str, Any], max_chars: int = 320) -> str:
        collapsed = " ".join(llm_json_dumps(payload).split())
        if len(collapsed) <= max_chars:
            return collapsed
        return collapsed[: max_chars - 3] + "..."

    def _normalize_reward_payload(
        self,
        *,
        payload: dict[str, Any],
        goal,
        plan,
    ) -> dict[str, object]:
        selected_xapps = [
            str(item)
            for item in list(getattr(plan, "selected_xapps", []) or [])
            if str(item).strip() != ""
        ]
        reward_terms: dict[str, float] = {}
        for candidate in self._extract_mapping_candidates(
            payload=payload,
            field_names=self._REWARD_MAPPING_KEYS,
            alias_map=self._REWARD_TERM_ALIASES,
        ):
            reward_terms = self._normalize_alias_mapping(candidate, self._REWARD_TERM_ALIASES)
            if len(reward_terms) > 0:
                break

        kpi_alias_map = dict(self._KPI_ALIASES)
        for goal_kpi_key in self._goal_kpi_keys(goal):
            aliases = list(kpi_alias_map.get(goal_kpi_key, ()))
            if goal_kpi_key not in aliases:
                aliases.append(goal_kpi_key)
            kpi_alias_map[goal_kpi_key] = tuple(aliases)
        kpi_weights: dict[str, float] = {}
        for candidate in self._extract_mapping_candidates(
            payload=payload,
            field_names=self._KPI_MAPPING_KEYS,
            alias_map=kpi_alias_map,
        ):
            kpi_weights = self._normalize_alias_mapping(candidate, kpi_alias_map)
            if len(kpi_weights) > 0:
                break

        black_box_xapps: list[str] = []
        for section in self._payload_sections(payload):
            for field_name in self._BLACK_BOX_KEYS:
                black_box_candidate = self._normalize_black_box_xapps(
                    section.get(field_name),
                    selected_xapps=selected_xapps,
                )
                if len(black_box_candidate) > 0:
                    black_box_xapps = black_box_candidate
                    break
            if len(black_box_xapps) > 0:
                break

        reward_id = ""
        for section in self._payload_sections(payload):
            candidate_reward_id = str(section.get("reward_id", "") or "").strip()
            if candidate_reward_id != "":
                reward_id = candidate_reward_id
                break

        return {
            "reward_id": reward_id,
            "reward_terms": reward_terms,
            "kpi_weights": kpi_weights,
            "black_box_xapps": black_box_xapps,
        }

    def _llm_reward_spec(self, *, goal, plan, observation: Observation) -> RewardSpec:
        payload = self.client.generate_json_or_raise(
            system_prompt=(
                "You are a reward designer for a Near-RT tuning xApp. Build reward "
                "weights and black-box assignments for the chosen xApp portfolio. "
                "Respond as JSON with top-level keys reward_terms, kpi_weights, and "
                "black_box_xapps. Return JSON only."
            ),
            user_prompt=llm_json_dumps(
                self._generation_user_payload(
                    goal=goal,
                    plan=plan,
                    observation=observation,
                )
            ),
        )
        if not isinstance(payload, dict):
            raise RuntimeError(f"{self.client.model_label} returned an invalid reward shaping payload.")

        normalized_payload = self._normalize_reward_payload(payload=payload, goal=goal, plan=plan)
        reward_terms = normalized_payload.get("reward_terms", {})
        kpi_weights = normalized_payload.get("kpi_weights", {})
        if (
            not isinstance(reward_terms, dict)
            or len(reward_terms) == 0
            or not isinstance(kpi_weights, dict)
            or len(kpi_weights) == 0
        ):
            repaired_payload = self.client.generate_json_or_raise(
                system_prompt=(
                    "You repair invalid reward-shaping JSON for a Near-RT tuning xApp. "
                    "Return JSON only with top-level keys reward_terms, kpi_weights, and "
                    "black_box_xapps."
                ),
                user_prompt=llm_json_dumps(
                    self._repair_user_payload(
                        goal=goal,
                        plan=plan,
                        observation=observation,
                        invalid_payload=payload,
                    )
                ),
            )
            if not isinstance(repaired_payload, dict):
                raise RuntimeError(
                    f"{self.client.model_label} returned an invalid reward shaping repair payload."
                )
            normalized_payload = self._normalize_reward_payload(
                payload=repaired_payload,
                goal=goal,
                plan=plan,
            )
            reward_terms = normalized_payload.get("reward_terms", {})
            kpi_weights = normalized_payload.get("kpi_weights", {})
            if (
                not isinstance(reward_terms, dict)
                or len(reward_terms) == 0
                or not isinstance(kpi_weights, dict)
                or len(kpi_weights) == 0
            ):
                raise RuntimeError(
                    f"{self.client.model_label} returned reward shaping JSON, but it did not provide usable "
                    "'reward_terms' and 'kpi_weights' even after schema repair. "
                    f"Initial payload excerpt: {self._truncate_payload_excerpt(payload)}. "
                    f"Repaired payload excerpt: {self._truncate_payload_excerpt(repaired_payload)}."
                )

        default_parameter_ranges = default_reward_parameter_ranges(
            list(getattr(plan, "selected_xapps", []) or [])
        )
        reward_spec = RewardSpec(
            reward_id=str(
                normalized_payload.get("reward_id", f"reward:{goal.goal_id}") or f"reward:{goal.goal_id}"
            ),
            goal_id=str(goal.goal_id),
            selection_plan_id=str(getattr(plan, "plan_id", "")),
            reward_terms={},
            kpi_weights={},
            parameter_ranges=default_parameter_ranges,
            black_box_xapps=[],
            generated_by=self.client.generated_by_tag,
        )

        reward_terms = normalized_payload.get("reward_terms", {})
        if isinstance(reward_terms, dict):
            reward_spec.reward_terms.update(
                {str(key): float(value) for key, value in reward_terms.items()}
            )
        kpi_weights = normalized_payload.get("kpi_weights", {})
        if isinstance(kpi_weights, dict):
            reward_spec.kpi_weights.update(
                {str(key): float(value) for key, value in kpi_weights.items()}
            )

        black_box_xapps = normalized_payload.get("black_box_xapps", [])
        if isinstance(black_box_xapps, list):
            reward_spec.black_box_xapps = [str(item) for item in black_box_xapps if str(item).strip() != ""]
        return reward_spec

    def _goal_kpi_weight_map(self, goal) -> dict[str, float]:
        weights: dict[str, float] = {}
        for target in list(getattr(goal, "kpi_targets", []) or []):
            canonical_name = self._canonical_kpi_name(str(getattr(target, "kpi_name", "") or ""))
            if canonical_name == "":
                continue
            weights[canonical_name] = max(0.0, float(getattr(target, "weight", 1.0) or 1.0))
        return weights

    def _goal_kpi_weights(self, goal) -> dict[str, float]:
        weights = self._goal_kpi_weight_map(goal)
        if len(weights) == 0:
            return {
                "total_throughput_mbps": 1.0,
                "avg_bler": 1.0,
                "queue_bytes": 0.6,
            }
        return weights

    def _resolve_guided_range(
        self,
        soft_range: RewardParameterRange,
        hard_range: RewardParameterRange | None,
    ) -> RewardParameterRange | None:
        field_name = str(getattr(soft_range, "field_name", "") or "").strip()
        target_xapp_name = str(getattr(soft_range, "target_xapp_name", "") or "").strip()
        if field_name == "" or target_xapp_name == "":
            return None

        lower_bound = getattr(soft_range, "lower_bound", None)
        upper_bound = getattr(soft_range, "upper_bound", None)
        if hard_range is not None and getattr(hard_range, "lower_bound", None) is not None:
            hard_lower = float(getattr(hard_range, "lower_bound"))
            lower_bound = hard_lower if lower_bound is None else max(float(lower_bound), hard_lower)
        if hard_range is not None and getattr(hard_range, "upper_bound", None) is not None:
            hard_upper = float(getattr(hard_range, "upper_bound"))
            upper_bound = hard_upper if upper_bound is None else min(float(upper_bound), hard_upper)
        if lower_bound is not None:
            lower_bound = float(lower_bound)
        if upper_bound is not None:
            upper_bound = float(upper_bound)
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            return None

        soft_discrete = list(getattr(soft_range, "discrete_values", []) or [])
        hard_discrete = list(getattr(hard_range, "discrete_values", []) or []) if hard_range is not None else []
        if len(soft_discrete) > 0 and len(hard_discrete) > 0:
            discrete_values = [item for item in soft_discrete if item in hard_discrete]
        elif len(soft_discrete) > 0:
            discrete_values = list(soft_discrete)
        else:
            discrete_values = list(hard_discrete)

        blocked_values: list[object] = []
        for item in list(getattr(soft_range, "blocked_discrete_values", []) or []):
            if item not in blocked_values:
                blocked_values.append(item)
        if hard_range is not None:
            for item in list(getattr(hard_range, "blocked_discrete_values", []) or []):
                if item not in blocked_values:
                    blocked_values.append(item)
        if len(discrete_values) > 0:
            discrete_values = [item for item in discrete_values if item not in blocked_values]
            if len(discrete_values) == 0:
                return None

        step_candidates = [
            float(value)
            for value in (
                getattr(soft_range, "step", None),
                getattr(hard_range, "step", None) if hard_range is not None else None,
            )
            if value not in (None, 0, 0.0)
        ]
        max_delta = getattr(hard_range, "max_delta", None) if hard_range is not None else None
        if max_delta is None:
            max_delta = getattr(soft_range, "max_delta", None)

        return RewardParameterRange(
            field_name=field_name,
            target_xapp_name=target_xapp_name,
            scope=str(
                getattr(
                    hard_range,
                    "scope",
                    getattr(soft_range, "scope", "cluster"),
                )
                or "cluster"
            ),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            step=min(step_candidates) if len(step_candidates) > 0 else None,
            discrete_values=discrete_values,
            blocked_discrete_values=blocked_values,
            max_delta=float(max_delta) if max_delta is not None else None,
            tuning_mode="guided_action_region",
        )

    def _guided_action_region(self, plan) -> list[RewardParameterRange]:
        selected_xapps = [
            str(item)
            for item in list(getattr(plan, "selected_xapps", []) or [])
            if str(item).strip() != ""
        ]
        soft_ranges = default_reward_parameter_ranges(selected_xapps)
        hard_ranges_by_key: dict[tuple[str, str], RewardParameterRange] = {}
        for hard_range in list(getattr(plan, "hard_safety_envelope", []) or []):
            key = (
                str(getattr(hard_range, "target_xapp_name", "") or "").strip(),
                str(getattr(hard_range, "field_name", "") or "").strip(),
            )
            if key[0] == "" or key[1] == "":
                continue
            hard_ranges_by_key[key] = hard_range

        guided_action_region: list[RewardParameterRange] = []
        for soft_range in soft_ranges:
            key = (
                str(getattr(soft_range, "target_xapp_name", "") or "").strip(),
                str(getattr(soft_range, "field_name", "") or "").strip(),
            )
            resolved_range = self._resolve_guided_range(soft_range, hard_ranges_by_key.get(key))
            if resolved_range is not None:
                guided_action_region.append(resolved_range)
        return guided_action_region

    def _eta_payload(self, plan) -> dict[str, object]:
        return a1_guidance_profile_payload(getattr(plan, "eta", None))

    def _ambient_parameter_space(
        self,
        *,
        guided_action_region: list[RewardParameterRange],
        guidance_id: str,
    ) -> AmbientParameterSpace:
        dimensions: list[AmbientParameterDimension] = []
        active_parameter_mask: dict[str, int] = {}
        for parameter_range in list(guided_action_region or []):
            field_name = str(getattr(parameter_range, "field_name", "") or "").strip()
            if field_name == "":
                continue
            dimensions.append(
                AmbientParameterDimension(
                    field_name=field_name,
                    target_xapp_name=str(
                        getattr(parameter_range, "target_xapp_name", "") or ""
                    ),
                    tuning_mode=str(
                        getattr(parameter_range, "tuning_mode", "guided_action_region")
                        or "guided_action_region"
                    ),
                    active=True,
                )
            )
            active_parameter_mask[field_name] = 1
        return AmbientParameterSpace(
            space_id=f"ambient:{guidance_id}",
            dimensions=dimensions,
            active_parameter_mask=active_parameter_mask,
            active_parameter_count=int(sum(active_parameter_mask.values())),
        )

    def _arbitration_variable_specs(self, plan) -> list[ArbitrationVariableSpec]:
        chi = getattr(plan, "chi", None)
        selected_xapps = [
            str(item)
            for item in list(getattr(plan, "selected_xapps", []) or [])
            if str(item).strip() != ""
        ]
        arbitration_mode = str(getattr(chi, "arbitration_mode", "weighted_merge") or "weighted_merge")
        merge_scope = str(getattr(chi, "merge_scope", "field") or "field")
        specs: list[ArbitrationVariableSpec] = []
        for field_name in (
            "scheduler_policy_hint",
            "latency_bias",
            "slice_weight_adjustment",
            "beam_candidate_preference",
            "power_cap_dbm",
            "traffic_steering_preference",
            "mobility_bias_db",
            "measurement_event_offset_db",
        ):
            specs.append(
                ArbitrationVariableSpec(
                    field_name=field_name,
                    arbitration_mode=arbitration_mode,
                    merge_scope=merge_scope,
                    candidate_sources=list(selected_xapps),
                    notes=[
                        f"profile_name={str(getattr(chi, 'profile_name', '') or '')}",
                    ],
                )
            )
        return specs

    def _reward_coefficients(self, *, goal, observation: Observation, plan) -> dict[str, float]:
        goal_weights = self._goal_kpi_weight_map(goal)
        gap_breakdown = goal_gap_breakdown(goal, observation)
        queue_pressure = min(
            1.5,
            float(getattr(observation, "total_queue_bytes", 0) or 0.0) / 120000.0,
        )
        bler_pressure = min(1.5, float(getattr(observation, "avg_bler", 0.0) or 0.0) / 0.20)
        handover_pressure = min(
            1.5,
            float(getattr(observation, "handover_pending_ue_count", 0) or 0.0) / 3.0,
        )
        throughput_gap = float(gap_breakdown.get("total_throughput_mbps", 0.0))
        queue_gap = float(gap_breakdown.get("queue_bytes", gap_breakdown.get("total_queue_bytes", 0.0)))
        bler_gap = float(gap_breakdown.get("avg_bler", 0.0))
        handover_gap = float(gap_breakdown.get("handover_risk", 0.0))
        power_gap = float(gap_breakdown.get("power_cost", 0.0))
        risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()

        reward_coefficients = {
            "throughput_gain": round(
                0.90 + 0.45 * float(goal_weights.get("total_throughput_mbps", 0.8)) + 0.70 * throughput_gap,
                4,
            ),
            "queue_reduction": round(
                0.30 + 0.35 * float(goal_weights.get("queue_bytes", 0.5)) + 0.70 * max(queue_pressure, queue_gap),
                4,
            ),
            "bler_reduction": round(
                0.35 + 0.40 * float(goal_weights.get("avg_bler", 0.8)) + 0.90 * max(bler_pressure, bler_gap),
                4,
            ),
            "handover_risk_reduction": round(
                0.15
                + 0.35 * float(goal_weights.get("handover_risk", 0.2))
                + 0.65 * max(handover_pressure, handover_gap),
                4,
            ),
            "power_cost_penalty": round(
                0.12 + 0.18 * float(goal_weights.get("power_cost", 0.2)) + 0.20 * power_gap,
                4,
            ),
        }
        if risk_posture == "reliability_first":
            reward_coefficients["bler_reduction"] = round(
                reward_coefficients["bler_reduction"] + 0.25,
                4,
            )
            reward_coefficients["throughput_gain"] = round(
                max(0.10, reward_coefficients["throughput_gain"] - 0.10),
                4,
            )
        eta_payload = self._eta_payload(plan)
        reward_term_bias = {
            str(term_name): max(0.05, float(bias))
            for term_name, bias in dict(eta_payload.get("reward_term_bias", {}) or {}).items()
            if str(term_name).strip() != ""
        }
        for term_name, bias in reward_term_bias.items():
            if term_name in reward_coefficients:
                reward_coefficients[term_name] = round(float(reward_coefficients[term_name]) * float(bias), 4)
        return reward_coefficients

    def _parameter_priority_vector(
        self,
        *,
        goal,
        observation: Observation,
        guided_action_region: list[RewardParameterRange],
        plan,
    ) -> dict[str, float]:
        active_fields = [
            str(getattr(parameter_range, "field_name", "") or "").strip()
            for parameter_range in guided_action_region
            if str(getattr(parameter_range, "field_name", "") or "").strip() != ""
        ]
        field_priority_vector = {field_name: 0.25 for field_name in active_fields}
        goal_weights = self._goal_kpi_weight_map(goal)
        for kpi_name, weight in goal_weights.items():
            for index, field_name in enumerate(self._FIELD_PRIORITY_HINTS.get(kpi_name, ())):
                if field_name not in field_priority_vector:
                    continue
                field_priority_vector[field_name] += float(weight) * max(0.12, 0.38 - 0.07 * index)

        total_queue_bytes = int(getattr(observation, "total_queue_bytes", 0) or 0)
        if total_queue_bytes > 80000:
            for field_name in ("scheduler_policy_hint", "slice_weight_adjustment", "latency_bias"):
                if field_name in field_priority_vector:
                    field_priority_vector[field_name] += 0.20
        if float(getattr(observation, "avg_bler", 0.0) or 0.0) > 0.08:
            for field_name in ("latency_bias", "beam_candidate_preference", "power_cap_dbm"):
                if field_name in field_priority_vector:
                    field_priority_vector[field_name] += 0.18
        if int(getattr(observation, "handover_pending_ue_count", 0) or 0) > 0:
            for field_name in (
                "mobility_bias_db",
                "measurement_event_offset_db",
                "steering_hysteresis_db",
                "traffic_steering_preference",
            ):
                if field_name in field_priority_vector:
                    field_priority_vector[field_name] += 0.16
        eta_payload = self._eta_payload(plan)
        field_priority_hints = {
            str(field_name): max(0.05, float(priority_hint))
            for field_name, priority_hint in dict(eta_payload.get("field_priority_hints", {}) or {}).items()
            if str(field_name).strip() != ""
        }
        for field_name, priority_hint in field_priority_hints.items():
            if field_name in field_priority_vector:
                field_priority_vector[field_name] *= float(priority_hint)

        positive_priorities = {
            field_name: max(0.0, float(priority))
            for field_name, priority in field_priority_vector.items()
        }
        total_priority = float(sum(positive_priorities.values()))
        if total_priority <= 1e-9 and len(positive_priorities) > 0:
            uniform_priority = 1.0 / float(len(positive_priorities))
            return {
                field_name: round(float(uniform_priority), 6)
                for field_name in sorted(positive_priorities.keys())
            }
        if total_priority <= 1e-9:
            return {}
        return {
            field_name: round(float(priority) / total_priority, 6)
            for field_name, priority in sorted(positive_priorities.items())
        }

    def _selected_safety_rule_set(
        self,
        *,
        goal,
        plan,
        observation: Observation,
        guided_action_region: list[RewardParameterRange],
        fallback_used: bool,
    ) -> SafetyConstraintSet:
        hard_safety_envelope = list(getattr(plan, "hard_safety_envelope", []) or [])
        forbidden_combinations = list(getattr(plan, "forbidden_combinations", []) or [])
        exploration_health_gate = getattr(plan, "exploration_health_gate", None)
        selected_rule_refs: list[SafetyRuleReference] = []
        if len(hard_safety_envelope) > 0:
            selected_rule_refs.append(
                SafetyRuleReference(
                    rule_id="rule:hard_safety_envelope",
                    config={
                        "range_count": int(len(guided_action_region)),
                        "bounded_fields": sorted(
                            {
                                str(getattr(item, "field_name", "") or "")
                                for item in guided_action_region
                                if str(getattr(item, "field_name", "") or "").strip() != ""
                            }
                        ),
                    },
                    description=self._SAFETY_RULE_LIBRARY["rule:hard_safety_envelope"],
                )
            )
        if len(forbidden_combinations) > 0:
            selected_rule_refs.append(
                SafetyRuleReference(
                    rule_id="rule:forbidden_combinations",
                    config={
                        "combination_count": int(len(forbidden_combinations)),
                    },
                    description=self._SAFETY_RULE_LIBRARY["rule:forbidden_combinations"],
                )
            )
        near_violation_reasons: list[str] = []
        if exploration_health_gate is not None:
            selected_rule_refs.append(
                SafetyRuleReference(
                    rule_id="rule:exploration_health_gate",
                    config={
                        "max_avg_bler": getattr(exploration_health_gate, "max_avg_bler", None),
                        "max_total_queue_bytes": getattr(
                            exploration_health_gate,
                            "max_total_queue_bytes",
                            None,
                        ),
                        "min_total_throughput_mbps": getattr(
                            exploration_health_gate,
                            "min_total_throughput_mbps",
                            None,
                        ),
                        "max_handover_pending_ue_count": getattr(
                            exploration_health_gate,
                            "max_handover_pending_ue_count",
                            None,
                        ),
                        "freeze_slots_on_violation": int(
                            getattr(exploration_health_gate, "freeze_slots_on_violation", 4) or 4
                        ),
                    },
                    description=self._SAFETY_RULE_LIBRARY["rule:exploration_health_gate"],
                )
            )
            max_avg_bler = getattr(exploration_health_gate, "max_avg_bler", None)
            if max_avg_bler is not None and float(getattr(observation, "avg_bler", 0.0) or 0.0) >= 0.75 * float(max_avg_bler):
                near_violation_reasons.append("bler_margin_tight")
            max_total_queue_bytes = getattr(exploration_health_gate, "max_total_queue_bytes", None)
            if max_total_queue_bytes is not None and int(getattr(observation, "total_queue_bytes", 0) or 0) >= 0.75 * int(max_total_queue_bytes):
                near_violation_reasons.append("queue_margin_tight")
            min_total_throughput_mbps = getattr(exploration_health_gate, "min_total_throughput_mbps", None)
            if min_total_throughput_mbps is not None and float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0) <= 1.10 * float(min_total_throughput_mbps):
                near_violation_reasons.append("throughput_margin_tight")
            max_handover_pending_ue_count = getattr(exploration_health_gate, "max_handover_pending_ue_count", None)
            if max_handover_pending_ue_count is not None and int(getattr(observation, "handover_pending_ue_count", 0) or 0) >= max(1, int(0.75 * int(max_handover_pending_ue_count))):
                near_violation_reasons.append("handover_margin_tight")
        if len(near_violation_reasons) > 0:
            selected_rule_refs.append(
                SafetyRuleReference(
                    rule_id="rule:degraded_network_freeze",
                    config={"trigger_reasons": list(near_violation_reasons)},
                    description=self._SAFETY_RULE_LIBRARY["rule:degraded_network_freeze"],
                )
            )
        if fallback_used:
            selected_rule_refs.append(
                SafetyRuleReference(
                    rule_id="rule:fallback_template_guard",
                    config={
                        "goal_id": str(getattr(goal, "goal_id", "") or ""),
                    },
                    description=self._SAFETY_RULE_LIBRARY["rule:fallback_template_guard"],
                )
            )
        return SafetyConstraintSet(
            rule_set_id=f"guidance_safety:{str(getattr(plan, 'plan_id', '') or '')}",
            goal_id=str(getattr(plan, "goal_id", "") or ""),
            selection_plan_id=str(getattr(plan, "plan_id", "") or ""),
            selected_rule_refs=selected_rule_refs,
            safety_envelope=hard_safety_envelope,
            forbidden_combinations=forbidden_combinations,
            exploration_health_gate=exploration_health_gate,
            generated_by=f"{self.name}:fixed_rule_library",
        )

    def _black_box_xapps(
        self,
        *,
        plan,
        guided_action_region: list[RewardParameterRange],
    ) -> list[str]:
        xapps_with_ranges = {
            str(getattr(parameter_range, "target_xapp_name", "") or "").strip()
            for parameter_range in guided_action_region
            if str(getattr(parameter_range, "target_xapp_name", "") or "").strip() != ""
        }
        return [
            str(xapp_name)
            for xapp_name in list(getattr(plan, "selected_xapps", []) or [])
            if str(xapp_name).strip() != "" and str(xapp_name) not in xapps_with_ranges
        ]

    def _build_contracts(
        self,
        *,
        goal,
        plan,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> tuple[RewardSpec, RlGuidanceContract]:
        guided_action_region = self._guided_action_region(plan)
        fallback_used = bool(
            getattr(
                getattr(getattr(policy_snapshot, "validated_intent_template", None), "fallback_report", None),
                "fallback_used",
                False,
            )
        )
        reward_coefficients = self._reward_coefficients(goal=goal, observation=observation, plan=plan)
        parameter_priority_vector = self._parameter_priority_vector(
            goal=goal,
            observation=observation,
            guided_action_region=guided_action_region,
            plan=plan,
        )
        selected_safety_rule_set = self._selected_safety_rule_set(
            goal=goal,
            plan=plan,
            observation=observation,
            guided_action_region=guided_action_region,
            fallback_used=fallback_used,
        )
        black_box_xapps = self._black_box_xapps(
            plan=plan,
            guided_action_region=guided_action_region,
        )
        kpi_weights = {
            str(kpi_name): float(weight)
            for kpi_name, weight in self._goal_kpi_weights(goal).items()
        }
        source_epoch = int(getattr(policy_snapshot, "policy_epoch_id", 0) or 0)
        derived_from_intent_id = str(
            getattr(policy_snapshot, "active_intent_id", "")
            or getattr(goal, "intent_id", "")
            or ""
        )
        reward_spec = RewardSpec(
            reward_id=f"reward:{str(getattr(goal, 'goal_id', '') or '')}",
            goal_id=str(getattr(goal, "goal_id", "") or ""),
            selection_plan_id=str(getattr(plan, "plan_id", "") or ""),
            reward_terms=dict(reward_coefficients),
            kpi_weights=kpi_weights,
            parameter_ranges=list(guided_action_region),
            black_box_xapps=black_box_xapps,
            generated_by=f"{self.name}:fixed_template",
        )
        rl_guidance = RlGuidanceContract(
            guidance_id=f"guidance:{str(getattr(goal, 'goal_id', '') or '')}",
            goal_id=str(getattr(goal, "goal_id", "") or ""),
            selection_plan_id=str(getattr(plan, "plan_id", "") or ""),
            guided_action_region=list(guided_action_region),
            reward_coefficients=dict(reward_coefficients),
            parameter_priority_vector=dict(parameter_priority_vector),
            ambient_parameter_space=self._ambient_parameter_space(
                guided_action_region=guided_action_region,
                guidance_id=f"guidance:{str(getattr(goal, 'goal_id', '') or '')}",
            ),
            arbitration_variable_specs=self._arbitration_variable_specs(plan),
            selected_safety_rule_set=selected_safety_rule_set,
            black_box_xapps=black_box_xapps,
            source_epoch=source_epoch,
            derived_from_intent_id=derived_from_intent_id,
            fallback_used=fallback_used,
            generated_by=f"{self.name}:fixed_template",
        )
        return reward_spec, rl_guidance

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None = None,
    ) -> PolicyUpdate:
        goal = getattr(policy_snapshot, "structured_goal", None) if policy_snapshot is not None else None
        plan = getattr(policy_snapshot, "xapp_selection_plan", None) if policy_snapshot is not None else None
        if goal is None or plan is None:
            return PolicyUpdate(source_name=self.name)
        reward_spec, rl_guidance = self._build_contracts(
            goal=goal,
            plan=plan,
            observation=observation,
            policy_snapshot=policy_snapshot,
        )
        return PolicyUpdate(
            policy_epoch_id=int(getattr(policy_snapshot, "policy_epoch_id", 0) or 0),
            reward_spec=reward_spec,
            rl_guidance=rl_guidance if self.rl_guidance_enabled else None,
            safety_constraint_set=rl_guidance.selected_safety_rule_set,
            orchestration_plan_view=getattr(policy_snapshot, "orchestration_plan_view", None),
            policy_type_name="reward_specification",
            policy_description=(
                "Fixed guidance-template instantiation for intent-aligned tuning"
                if self.rl_guidance_enabled
                else "Fixed reward-template instantiation with legacy Near-RT fallback"
            ),
            source_name=self.name,
        )
