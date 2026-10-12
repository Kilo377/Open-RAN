"""Non-RT rApp that selects xApp combinations for structured goals."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

from models.intent_orchestration import (
    CandidatePlanScore,
    ExplorationHealthGate,
    OrchestrationCandidatePlan,
    OrchestrationMemoryCase,
    OrchestrationMemoryState,
    ReflectionNote,
    RetrievedMemoryCaseTrace,
    SelectionQueryKey,
    XAppCandidateScore,
    XAppSelectionPlan,
    orchestration_plan_view_from_plan,
)
from models.observation import Observation
from models.policy import PolicySnapshot, PolicyUpdate
from ric.non_rt.intent_orchestration_helpers import (
    available_xapp_catalog_for_llm,
    available_xapp_category_catalog_for_llm,
    default_exploration_health_gate,
    default_forbidden_combinations,
    default_hard_safety_envelope,
    goal_gap_score,
    llm_json_dumps,
    observation_summary_for_llm,
)
from ric.near_rt.xapp_variant_catalog import DEFAULT_XAPP_VARIANTS, canonical_xapp_category
from ric.non_rt.ollama_gptoss_client import StructuredJsonLlmClient
from ric.non_rt.orchestration_feasibility import OrchestrationFeasibilityProjector
from ric.non_rt.orchestration_memory_attribution import (
    attributed_failure_category_set,
    memory_case_outcome_quality,
)
from ric.non_rt.orchestration_memory_store import OrchestrationMemoryStore
from ric.non_rt.orchestration_surrogate_predictor import (
    build_surrogate_predictor,
)
from ric.non_rt.orchestration_surrogate_scorer import OrchestrationSurrogateScorer
from ric.non_rt.rapp_base import RAppBase
from ric.non_rt.structured_embedding_retriever import StructuredEmbeddingRetriever, build_embedding_backend


_SELECTION_RETRIEVAL_ENCODING_VERSION = "structured_retrieval_v2_embedding"


class XAppSelectionRApp(RAppBase):
    def __init__(
        self,
        *,
        name: str = "xapp_selection_rapp",
        max_selected_xapps: int = 4,
        success_memory_size: int = 12,
        reflection_memory_size: int = 8,
        retrieval_top_k: int = 3,
        persistent_memory_enabled: bool = False,
        memory_store_path: str | None = None,
        max_cpu_cost: float = 6.0,
        max_mem_cost: float = 5.5,
        mutex_category_pairs: list[list[str]] | None = None,
        dependency_category_pairs: list[list[str]] | None = None,
        available_xapp_names: list[str] | None = None,
        static_xapp_selection_enabled: bool = False,
        freeze_learning: bool = False,
        reflection_enabled: bool = True,
        memory_mode: str = "full",
        selection_mode: str = "reflective",
        memory_age_decay_enabled: bool = True,
        memory_retrieval_mode: str = "ranked",
        reflective_transfer_guard_enabled: bool = True,
        failure_gap_threshold: float = 0.35,
        gap_patience_rounds: int = 2,
        convergence_gap_threshold: float = 0.18,
        success_gap_threshold: float | None = None,
        stagnation_patience_rounds: int = 2,
        stagnation_epsilon: float = 0.03,
        experiment_transfer_phase: str = "",
        experiment_phase_repeat: int = 0,
        literature_baseline_state_enabled: bool = False,
        literature_baseline_state_path: str = "",
        intent_hrl_train_on_warmup: bool = True,
        intent_hrl_freeze_on_transfer: bool = True,
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
        selection_embedding_backend_name: str = "hashed_feature_embedding_v1",
        selection_embedding_backend_state_path: str | None = None,
        surrogate_predictor_name: str = "memory_calibrated_surrogate_predictor_v1",
        surrogate_predictor_state_path: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.max_selected_xapps = max(1, int(max_selected_xapps))
        self.success_memory_size = max(1, int(success_memory_size))
        self.reflection_memory_size = max(1, int(reflection_memory_size))
        self.retrieval_top_k = max(1, int(retrieval_top_k))
        self.persistent_memory_enabled = bool(persistent_memory_enabled)
        self.memory_store_path = str(memory_store_path or "").strip()
        self.max_cpu_cost = max(0.5, float(max_cpu_cost))
        self.max_mem_cost = max(0.5, float(max_mem_cost))
        self.mutex_category_pairs = list(mutex_category_pairs or [])
        self.dependency_category_pairs = list(dependency_category_pairs or [])
        self.available_xapp_names = [
            str(item).strip()
            for item in list(available_xapp_names or [])
            if str(item).strip() != ""
        ] or None
        self.static_xapp_selection_enabled = bool(static_xapp_selection_enabled)
        self.freeze_learning=bool(freeze_learning)
        self.reflection_enabled = bool(reflection_enabled)
        self.memory_mode = self._normalize_memory_mode(memory_mode)
        self.selection_mode = self._normalize_selection_mode(selection_mode)
        self.memory_age_decay_enabled = bool(memory_age_decay_enabled)
        self.memory_retrieval_mode = self._normalize_memory_retrieval_mode(memory_retrieval_mode)
        self.reflective_transfer_guard_enabled = bool(reflective_transfer_guard_enabled)
        self.failure_gap_threshold = max(0.0, float(failure_gap_threshold))
        self.gap_patience_rounds = max(1, int(gap_patience_rounds))
        self.convergence_gap_threshold = max(0.0, float(convergence_gap_threshold))
        self.success_gap_threshold = max(
            0.0,
            float(
                convergence_gap_threshold
                if success_gap_threshold is None
                else success_gap_threshold
            ),
        )
        self.stagnation_patience_rounds = max(1, int(stagnation_patience_rounds))
        self.stagnation_epsilon = max(0.0, float(stagnation_epsilon))
        self.experiment_transfer_phase = str(experiment_transfer_phase or "").strip().lower()
        self.experiment_phase_repeat = max(0, int(experiment_phase_repeat or 0))
        self.literature_baseline_state_enabled = bool(literature_baseline_state_enabled)
        self.literature_baseline_state_path = str(literature_baseline_state_path or "").strip()
        self.intent_hrl_train_on_warmup = bool(intent_hrl_train_on_warmup)
        self.intent_hrl_freeze_on_transfer = bool(intent_hrl_freeze_on_transfer)
        self._literature_baseline_state: dict[str, object] = {}
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
        self.reflection_memory: list[ReflectionNote] = []
        self.success_memory: list[OrchestrationMemoryCase] = []
        self.failure_memory: list[OrchestrationMemoryCase] = []
        self.last_selection_query_key: SelectionQueryKey | None = None
        self.last_retrieved_success_cases: list[OrchestrationMemoryCase] = []
        self.last_retrieved_failure_cases: list[OrchestrationMemoryCase] = []
        self.last_retrieved_success_traces: list[RetrievedMemoryCaseTrace] = []
        self.last_retrieved_failure_traces: list[RetrievedMemoryCaseTrace] = []
        default_memory_store_path = Path("outputs") / "orchestration_memory" / f"{self.name}.json"
        self.memory_store = OrchestrationMemoryStore(
            self.memory_store_path if self.memory_store_path != "" else str(default_memory_store_path)
        )
        self.feasibility_projector = OrchestrationFeasibilityProjector(
            max_selected_xapps=self.max_selected_xapps,
            max_cpu_cost=self.max_cpu_cost,
            max_mem_cost=self.max_mem_cost,
            mutex_category_pairs=self.mutex_category_pairs,
            dependency_category_pairs=self.dependency_category_pairs,
        )
        self.surrogate_scorer = OrchestrationSurrogateScorer(
            max_cpu_cost=self.max_cpu_cost,
            max_mem_cost=self.max_mem_cost,
            predictor=build_surrogate_predictor(
                predictor_name=surrogate_predictor_name,
                predictor_state_path=surrogate_predictor_state_path,
                max_cpu_cost=self.max_cpu_cost,
                max_mem_cost=self.max_mem_cost,
            ),
        )
        self.selection_retriever = StructuredEmbeddingRetriever(
            backend=build_embedding_backend(
                backend_name=selection_embedding_backend_name,
                vector_dim=48,
                namespace="selection_memory",
                state_path=selection_embedding_backend_state_path,
            )
        )
        if self.persistent_memory_enabled:
            self._load_persistent_memory()
        if self.literature_baseline_state_enabled:
            self._load_literature_baseline_state()

    def _normalize_memory_mode(self, memory_mode: str | None) -> str:
        normalized = str(memory_mode or "full").strip().lower()
        aliases = {
            "enabled": "full",
            "reflective": "full",
            "full_memory": "full",
            "no_memory": "none",
            "disabled": "none",
            "off": "none",
            "success": "success_only",
            "success_only_memory": "success_only",
            "failed": "failed_only",
            "failure": "failed_only",
            "failure_only": "failed_only",
            "failed_case": "failed_only",
            "failed_only_memory": "failed_only",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {
            "full",
            "none",
            "success_only",
            "failed_only",
            "diagnostic_only",
            "raw_reward",
        }
        return normalized if normalized in allowed else "full"

    def _normalize_selection_mode(self, selection_mode: str | None) -> str:
        normalized = str(selection_mode or "reflective").strip().lower()
        aliases = {
            "static": "fixed_plan",
            "static_xapp_selection": "fixed_plan",
            "fixed": "fixed_plan",
            "no_memory": "no_memory_planner",
            "no_memory_orchestration": "no_memory_planner",
            "orchestran": "resource_aware",
            "orchestran_like": "resource_aware",
            "orchestran_like_resource_aware": "resource_aware",
            "resource/timing_aware": "resource_aware",
            "intent_hrl_like": "intent_hrl",
            "hrl": "intent_hrl",
            "oreo": "service_graph",
            "oreo_like": "service_graph",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {
            "reflective",
            "fixed_plan",
            "no_memory_planner",
            "resource_aware",
            "intent_hrl",
            "service_graph",
        }
        return normalized if normalized in allowed else "reflective"

    def _normalize_memory_retrieval_mode(self, memory_retrieval_mode: str | None) -> str:
        normalized = str(memory_retrieval_mode or "ranked").strip().lower()
        aliases = {
            "default": "ranked",
            "relevance": "ranked",
            "relevance_ranked": "ranked",
            "shuffle": "shuffled",
            "shuffled_memory": "shuffled",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"ranked", "shuffled"} else "ranked"

    def _reflective_transfer_guard_active(self) -> bool:
        return (
            self.reflective_transfer_guard_enabled
            and self.selection_mode == "reflective"
            and self.memory_mode == "full"
        )

    def _memory_enabled(self) -> bool:
        return self.memory_mode != "none"

    def _success_memory_enabled(self) -> bool:
        return self.memory_mode in {"full", "success_only", "diagnostic_only", "raw_reward"}

    def _failure_memory_enabled(self) -> bool:
        return self.memory_mode in {"full", "failed_only", "diagnostic_only", "raw_reward"}

    def _memory_cases_feed_selection(self) -> bool:
        return self.memory_mode not in {"none", "diagnostic_only"}

    def _load_literature_baseline_state(self) -> None:
        if self.literature_baseline_state_path == "":
            self._literature_baseline_state = {}
            return
        path = Path(self.literature_baseline_state_path)
        if not path.exists():
            self._literature_baseline_state = {}
            return
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self._literature_baseline_state = dict(loaded) if isinstance(loaded, dict) else {}
        except Exception:
            self._literature_baseline_state = {}

    def _persist_literature_baseline_state(self) -> None:
        if self.freeze_learning:return
        if not self.literature_baseline_state_enabled or self.literature_baseline_state_path == "":
            return
        path = Path(self.literature_baseline_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._literature_baseline_state, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _baseline_state_section(self, section_name: str) -> dict[str, object]:
        state = dict(self._literature_baseline_state or {})
        section = state.get(section_name, {})
        if not isinstance(section, dict):
            section = {}
        state[section_name] = section
        self._literature_baseline_state = state
        return section

    def _baseline_training_active(self, baseline_name: str) -> bool:
        if self.freeze_learning:return False
        if not self.literature_baseline_state_enabled:
            return False
        if baseline_name == "intent_hrl":
            if self.experiment_transfer_phase == "warmup":
                return self.intent_hrl_train_on_warmup
            if self.experiment_transfer_phase == "transfer" and self.intent_hrl_freeze_on_transfer:
                return False
        return self.experiment_transfer_phase in {"", "warmup"}

    def _goal_kpi_names(self, goal) -> list[str]:
        return sorted(
            {
                str(getattr(target, "kpi_name", "") or "").strip().lower()
                for target in list(getattr(goal, "kpi_targets", []) or [])
                if str(getattr(target, "kpi_name", "") or "").strip() != ""
            }
        )

    def _xapp_categories(self, selected_xapps: list[str]) -> list[str]:
        categories: list[str] = []
        for xapp_name in list(selected_xapps or []):
            category_name = canonical_xapp_category(str(xapp_name or "").strip())
            if category_name != "":
                categories.append(category_name)
        return list(dict.fromkeys(categories))

    def _bucket_label(self, value: float, *, low: float, high: float) -> str:
        if value < low:
            return "low"
        if value < high:
            return "medium"
        return "high"

    def _context_signature(self, *, observation: Observation, goal) -> dict[str, object]:
        queue_bytes = int(getattr(observation, "total_queue_bytes", 0) or 0)
        throughput_mbps = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        handover_pending = int(getattr(observation, "handover_pending_ue_count", 0) or 0)

        kpi_names = self._goal_kpi_names(goal)
        target_profile = "balanced"
        if "total_throughput_mbps" in kpi_names:
            target_profile = "throughput"
        if "avg_bler" in kpi_names and target_profile == "throughput":
            target_profile = "throughput_reliability"
        elif "avg_bler" in kpi_names:
            target_profile = "reliability"
        if any(name in {"handover_risk", "mobility"} for name in kpi_names):
            target_profile = "mobility"

        return {
            "queue_class": self._bucket_label(float(queue_bytes), low=25000.0, high=80000.0),
            "throughput_class": self._bucket_label(throughput_mbps, low=20.0, high=40.0),
            "bler_class": "good" if avg_bler <= 0.08 else "stressed" if avg_bler <= 0.15 else "poor",
            "mobility_class": "high" if handover_pending >= 3 else "moderate" if handover_pending >= 1 else "low",
            "risk_posture": str(getattr(goal, "risk_posture", "balanced") or "balanced"),
            "target_profile": target_profile,
        }

    def _encode_retrieval_features(
        self,
        *,
        kpi_names: list[str],
        context_signature: dict[str, object],
        xapp_categories: list[str],
        risk_posture: str,
    ) -> dict[str, float]:
        encoded_features: dict[str, float] = {}
        for kpi_name in list(kpi_names or []):
            normalized = str(kpi_name).strip().lower()
            if normalized != "":
                encoded_features[f"kpi::{normalized}"] = 1.0
        for key, value in dict(context_signature or {}).items():
            normalized_key = str(key).strip().lower()
            normalized_value = str(value).strip().lower()
            if normalized_key != "" and normalized_value != "":
                encoded_features[f"context::{normalized_key}::{normalized_value}"] = 1.0
        for category_name in list(xapp_categories or []):
            normalized = str(category_name).strip().lower()
            if normalized != "":
                encoded_features[f"xapp_category::{normalized}"] = 1.0
        normalized_risk = str(risk_posture).strip().lower()
        if normalized_risk != "":
            encoded_features[f"risk::{normalized_risk}"] = 1.0
        return encoded_features

    def _embedding_vector(
        self,
        *,
        encoded_features: dict[str, float] | None,
        existing_vector: list[float] | None = None,
    ) -> list[float]:
        if len(list(existing_vector or [])) > 0:
            return [float(item) for item in list(existing_vector or [])]
        return self.selection_retriever.encode_sparse_features(encoded_features)

    def _sparse_cosine_similarity(
        self,
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        normalized_left = {
            str(key): float(value)
            for key, value in dict(left or {}).items()
            if str(key).strip() != ""
        }
        normalized_right = {
            str(key): float(value)
            for key, value in dict(right or {}).items()
            if str(key).strip() != ""
        }
        if len(normalized_left) == 0 or len(normalized_right) == 0:
            return 0.0
        dot_product = 0.0
        for key, value in normalized_left.items():
            dot_product += float(value) * float(normalized_right.get(key, 0.0) or 0.0)
        left_norm = sum(float(value) ** 2 for value in normalized_left.values()) ** 0.5
        right_norm = sum(float(value) ** 2 for value in normalized_right.values()) ** 0.5
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return float(dot_product / (left_norm * right_norm))

    def _selection_query_key(
        self,
        *,
        goal,
        observation: Observation,
        previous_selected_xapps: list[str],
    ) -> SelectionQueryKey:
        context_signature = self._context_signature(observation=observation, goal=goal)
        previous_selected_xapps = [
            str(item).strip()
            for item in list(previous_selected_xapps or [])
            if str(item).strip() != ""
        ]
        previous_xapp_categories = self._xapp_categories(previous_selected_xapps)
        encoded_features = self._encode_retrieval_features(
            kpi_names=self._goal_kpi_names(goal),
            context_signature=context_signature,
            xapp_categories=previous_xapp_categories,
            risk_posture=str(getattr(goal, "risk_posture", "balanced") or "balanced"),
        )
        return SelectionQueryKey(
            query_id=(
                f"qsel:{str(getattr(goal, 'goal_id', '') or '')}:"
                f"{str(context_signature.get('target_profile', 'balanced') or 'balanced')}:"
                f"slot{int(getattr(observation, 'slot_id', 0) or 0)}"
            ),
            query_slot_id=int(getattr(observation, "slot_id", 0) or 0),
            encoding_version=_SELECTION_RETRIEVAL_ENCODING_VERSION,
            retriever_backend_name=self.selection_retriever.backend_name,
            goal_id=str(getattr(goal, "goal_id", "") or ""),
            intent_id=str(getattr(goal, "intent_id", "") or ""),
            risk_posture=str(getattr(goal, "risk_posture", "balanced") or "balanced"),
            kpi_names=self._goal_kpi_names(goal),
            context_signature=context_signature,
            encoded_features=encoded_features,
            embedding_vector=self._embedding_vector(encoded_features=encoded_features),
            previous_selected_xapps=previous_selected_xapps,
            previous_xapp_categories=previous_xapp_categories,
            notes=[
                f"queue_class={context_signature.get('queue_class', '')}",
                f"throughput_class={context_signature.get('throughput_class', '')}",
                f"bler_class={context_signature.get('bler_class', '')}",
                f"mobility_class={context_signature.get('mobility_class', '')}",
            ],
        )

    def _memory_case_similarity(
        self,
        memory_case: OrchestrationMemoryCase,
        *,
        selection_query_key: SelectionQueryKey,
    ) -> tuple[float, float, float, float, float]:
        query_kpis = set(getattr(selection_query_key, "kpi_names", []) or [])
        case_kpis = {
            str(item).strip().lower()
            for item in list(getattr(memory_case, "kpi_names", []) or [])
            if str(item).strip() != ""
        }
        kpi_score = 0.0
        if query_kpis or case_kpis:
            kpi_score = float(len(query_kpis & case_kpis)) / float(max(len(query_kpis | case_kpis), 1))

        query_signature = dict(getattr(selection_query_key, "context_signature", {}) or {})
        case_signature = dict(getattr(memory_case, "context_signature", {}) or {})
        signature_keys = ["queue_class", "throughput_class", "bler_class", "mobility_class", "target_profile"]
        matched_signature_keys = sum(
            1
            for key in signature_keys
            if str(query_signature.get(key, "")) != "" and query_signature.get(key) == case_signature.get(key)
        )
        signature_score = float(matched_signature_keys) / float(len(signature_keys))

        previous_categories = set(
            getattr(selection_query_key, "previous_xapp_categories", []) or []
        )
        case_categories = set(self._xapp_categories(list(getattr(memory_case, "selected_xapps", []) or [])))
        category_score = 0.0
        if previous_categories or case_categories:
            category_score = float(len(previous_categories & case_categories)) / float(
                max(len(previous_categories | case_categories), 1)
            )

        risk_score = 1.0 if str(getattr(memory_case, "risk_posture", "")) == str(query_signature.get("risk_posture", "")) else 0.0
        similarity_score = (
            0.45 * kpi_score
            + 0.30 * signature_score
            + 0.15 * category_score
            + 0.10 * risk_score
        )
        feature_similarity = self.selection_retriever.cosine_similarity(
            self._embedding_vector(
                encoded_features=dict(getattr(selection_query_key, "encoded_features", {}) or {}),
                existing_vector=list(getattr(selection_query_key, "embedding_vector", []) or []),
            ),
            self._embedding_vector(
                encoded_features=dict(getattr(memory_case, "encoded_features", {}) or {}),
                existing_vector=list(getattr(memory_case, "embedding_vector", []) or []),
            ),
        )
        source_slot_id = getattr(memory_case, "source_slot_id", None)
        query_slot_id = int(getattr(selection_query_key, "query_slot_id", 0) or 0)
        recency_score = 0.0
        if source_slot_id is not None and self.memory_age_decay_enabled:
            slot_age = max(0, query_slot_id - int(source_slot_id))
            recency_score = 1.0 / float(1 + max(0, slot_age // 10))
        elif source_slot_id is not None:
            recency_score = 1.0
        retrieval_score = (
            (0.45 * similarity_score)
            + (0.40 * feature_similarity)
            + (0.15 * recency_score)
        )
        retrieval_confidence = self.selection_retriever.retrieval_confidence(
            semantic_similarity=similarity_score,
            vector_similarity=feature_similarity,
            recency_score=recency_score,
        )
        return (
            float(similarity_score),
            float(feature_similarity),
            float(recency_score),
            float(retrieval_score),
            float(retrieval_confidence),
        )

    def _retrieval_trace(
        self,
        *,
        memory_case: OrchestrationMemoryCase,
        retrieval_rank: int,
        similarity_score: float,
        feature_similarity: float,
        recency_score: float,
        retrieval_score: float,
        retrieval_confidence: float,
    ) -> RetrievedMemoryCaseTrace:
        return RetrievedMemoryCaseTrace(
            case_id=str(getattr(memory_case, "case_id", "") or ""),
            memory_type=str(getattr(memory_case, "memory_type", "success") or "success"),
            retrieval_rank=int(retrieval_rank),
            similarity_score=float(similarity_score),
            feature_similarity=float(feature_similarity),
            retriever_backend_name=str(
                getattr(memory_case, "retriever_backend_name", self.selection_retriever.backend_name)
                or self.selection_retriever.backend_name
            ),
            embedding_vector=self._embedding_vector(
                encoded_features=dict(getattr(memory_case, "encoded_features", {}) or {}),
                existing_vector=list(getattr(memory_case, "embedding_vector", []) or []),
            ),
            recency_score=float(recency_score),
            retrieval_score=float(retrieval_score),
            retrieval_confidence=float(retrieval_confidence),
            source_slot_id=getattr(memory_case, "source_slot_id", None),
            memory_retention_score=float(getattr(memory_case, "retention_score", 0.0) or 0.0),
            selected_xapps=list(getattr(memory_case, "selected_xapps", []) or []),
            kpi_names=list(getattr(memory_case, "kpi_names", []) or []),
            context_signature=dict(getattr(memory_case, "context_signature", {}) or {}),
            failure_reason=str(getattr(memory_case, "failure_reason", "") or ""),
            summary=str(getattr(memory_case, "summary", "") or ""),
        )

    def _memory_case_retention_score(
        self,
        memory_case: OrchestrationMemoryCase,
        *,
        current_slot_id: int | None = None,
    ) -> float:
        goal_gap_score = float(getattr(memory_case, "goal_gap_score", 1.0) or 1.0)
        retrieval_hits = int(getattr(memory_case, "retrieval_hits", 0) or 0)
        source_slot_id = getattr(memory_case, "source_slot_id", None)
        last_retrieved_slot_id = getattr(memory_case, "last_retrieved_slot_id", None)
        memory_type = str(getattr(memory_case, "memory_type", "success") or "success")

        if memory_type == "success":
            coarse_success_quality = max(0.0, 1.0 - min(goal_gap_score, 1.0))
            outcome_quality = memory_case_outcome_quality(memory_case)
            quality_score = (0.55 * coarse_success_quality) + (0.45 * outcome_quality)
        else:
            quality_score = 0.45 + min(goal_gap_score, 1.0)
            if getattr(memory_case, "reflection", None) is not None:
                quality_score += 0.10

        reference_slot_id = None
        if current_slot_id is not None:
            reference_slot_id = int(current_slot_id)
        elif last_retrieved_slot_id is not None:
            reference_slot_id = int(last_retrieved_slot_id)
        elif source_slot_id is not None:
            reference_slot_id = int(source_slot_id)

        recency_score = 0.0
        if reference_slot_id is not None and source_slot_id is not None and self.memory_age_decay_enabled:
            slot_age = max(0, int(reference_slot_id) - int(source_slot_id))
            recency_score = 1.0 / float(1 + max(0, slot_age // 20))
        elif reference_slot_id is not None and source_slot_id is not None:
            recency_score = 1.0

        hit_bonus = min(0.30, float(retrieval_hits) * 0.05)
        retention_score = (0.55 * quality_score) + (0.25 * recency_score) + hit_bonus
        return float(retention_score)

    def _refresh_memory_retention(
        self,
        memory_cases: list[OrchestrationMemoryCase],
        *,
        max_items: int,
        current_slot_id: int | None = None,
    ) -> list[OrchestrationMemoryCase]:
        deduplicated_by_id: dict[str, OrchestrationMemoryCase] = {}
        for memory_case in list(memory_cases or []):
            case_id = str(getattr(memory_case, "case_id", "") or "").strip()
            if case_id == "":
                continue
            memory_case.encoding_version = str(
                getattr(memory_case, "encoding_version", "") or _SELECTION_RETRIEVAL_ENCODING_VERSION
            )
            memory_case.retriever_backend_name = str(
                getattr(memory_case, "retriever_backend_name", "") or self.selection_retriever.backend_name
            )
            if getattr(memory_case, "intent_loss_normalized", None) is None:
                memory_case.intent_loss_normalized = getattr(memory_case, "goal_gap_score", None)
            memory_case.embedding_vector = self._embedding_vector(
                encoded_features=dict(getattr(memory_case, "encoded_features", {}) or {}),
                existing_vector=list(getattr(memory_case, "embedding_vector", []) or []),
            )
            memory_case.retention_score = self._memory_case_retention_score(
                memory_case,
                current_slot_id=current_slot_id,
            )
            deduplicated_by_id[case_id] = memory_case
        ranked_cases = sorted(
            deduplicated_by_id.values(),
            key=lambda item: (
                float(getattr(item, "retention_score", 0.0) or 0.0),
                int(getattr(item, "last_retrieved_slot_id", -1) or -1),
                int(getattr(item, "source_slot_id", -1) or -1),
            ),
            reverse=True,
        )
        return list(ranked_cases[:max_items])

    def _memory_case_summary(self, memory_case: OrchestrationMemoryCase) -> dict[str, object]:
        return {
            "case_id": str(getattr(memory_case, "case_id", "") or ""),
            "memory_type": str(getattr(memory_case, "memory_type", "") or ""),
            "selected_xapps": list(getattr(memory_case, "selected_xapps", []) or []),
            "goal_gap_score": getattr(memory_case, "goal_gap_score", None),
            "intent_loss_normalized": getattr(memory_case, "intent_loss_normalized", None),
            "context_signature": dict(getattr(memory_case, "context_signature", {}) or {}),
            "failure_reason": str(getattr(memory_case, "failure_reason", "") or ""),
            "summary": str(getattr(memory_case, "summary", "") or ""),
        }

    def _build_memory_case(
        self,
        *,
        memory_type: str,
        goal,
        observation: Observation,
        plan: XAppSelectionPlan,
        reflection: ReflectionNote | None,
        validated_intent_template=None,
        failure_reason: str = "",
    ) -> OrchestrationMemoryCase:
        slot_id = int(getattr(observation, "slot_id", 0) or 0)
        gap_score = float(
            goal_gap_score(
                goal,
                observation,
                validated_intent_template=validated_intent_template,
            )
        )
        selected_xapps = list(getattr(plan, "selected_xapps", []) or [])
        context_signature = self._context_signature(observation=observation, goal=goal)
        encoded_features = self._encode_retrieval_features(
            kpi_names=self._goal_kpi_names(goal),
            context_signature=context_signature,
            xapp_categories=self._xapp_categories(selected_xapps),
            risk_posture=str(getattr(goal, "risk_posture", "balanced") or "balanced"),
        )
        return OrchestrationMemoryCase(
            case_id=f"{memory_type}:{goal.goal_id}:slot{slot_id}:g{int(getattr(plan, 'selection_generation', 1) or 1)}",
            memory_type=str(memory_type or "success"),
            source_slot_id=slot_id,
            last_retrieved_slot_id=None,
            encoding_version=_SELECTION_RETRIEVAL_ENCODING_VERSION,
            retriever_backend_name=self.selection_retriever.backend_name,
            goal_id=str(getattr(goal, "goal_id", "") or ""),
            intent_id=str(getattr(goal, "intent_id", "") or ""),
            risk_posture=str(getattr(goal, "risk_posture", "balanced") or "balanced"),
            kpi_names=self._goal_kpi_names(goal),
            selected_xapps=selected_xapps,
            goal_gap_score=gap_score,
            intent_loss_normalized=gap_score,
            queue_bytes=int(getattr(observation, "total_queue_bytes", 0) or 0),
            throughput_mbps=float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0),
            avg_bler=float(getattr(observation, "avg_bler", 0.0) or 0.0),
            context_signature=context_signature,
            encoded_features=encoded_features,
            embedding_vector=self._embedding_vector(encoded_features=encoded_features),
            retrieval_hits=0,
            retention_score=0.0,
            summary=(
                f"{memory_type} case for goal={getattr(goal, 'goal_id', '')} "
                f"selected={selected_xapps} gap={gap_score:.3f} "
                f"intent_loss={gap_score:.3f} "
                f"queue={int(getattr(observation, 'total_queue_bytes', 0) or 0)} "
                f"throughput={float(getattr(observation, 'total_throughput_mbps', 0.0) or 0.0):.2f} "
                f"bler={float(getattr(observation, 'avg_bler', 0.0) or 0.0):.3f}"
            ),
            failure_reason=str(failure_reason or ""),
            reflection=reflection,
            generated_by=(
                str(getattr(reflection, "generated_by", "") or "")
                if reflection is not None
                else str(getattr(plan, "generated_by", "heuristic") or "heuristic")
            ),
        )

    def _store_success_case(self, memory_case: OrchestrationMemoryCase) -> None:
        if self.freeze_learning:return
        if not self._success_memory_enabled():
            return
        self.success_memory.append(memory_case)
        self.success_memory = self._refresh_memory_retention(
            self.success_memory,
            max_items=self.success_memory_size,
            current_slot_id=getattr(memory_case, "source_slot_id", None),
        )
        self._persist_memory_state()

    def _store_failure_case(self, memory_case: OrchestrationMemoryCase) -> None:
        if self.freeze_learning:return
        if not self._failure_memory_enabled():
            return
        self.failure_memory.append(memory_case)
        self.failure_memory = self._refresh_memory_retention(
            self.failure_memory,
            max_items=self.reflection_memory_size,
            current_slot_id=getattr(memory_case, "source_slot_id", None),
        )
        self._persist_memory_state()

    def _persist_memory_state(self) -> None:
        if self.freeze_learning:return
        if not self.persistent_memory_enabled or not self._memory_enabled():
            return
        self.memory_store.save(
            success_cases=self.success_memory,
            failure_cases=self.failure_memory,
            reflection_notes=self.reflection_memory,
        )

    def _load_persistent_memory(self) -> None:
        if not self._memory_enabled():
            return
        success_cases, failure_cases, reflection_notes = self.memory_store.load()
        if not self._success_memory_enabled():
            success_cases = []
        if not self._failure_memory_enabled():
            failure_cases = []
        if not self.reflection_enabled or not self._failure_memory_enabled():
            reflection_notes = []
        self.success_memory = self._refresh_memory_retention(
            success_cases,
            max_items=self.success_memory_size,
        )
        self.failure_memory = self._refresh_memory_retention(
            failure_cases,
            max_items=self.reflection_memory_size,
        )
        self.reflection_memory = list(reflection_notes or [])[-self.reflection_memory_size :]

    def _record_retrieval_usage(
        self,
        *,
        retrieved_success_cases: list[OrchestrationMemoryCase],
        retrieved_failure_cases: list[OrchestrationMemoryCase],
        selection_query_key: SelectionQueryKey,
    ) -> None:
        if self.freeze_learning:return
        query_slot_id = getattr(selection_query_key, "query_slot_id", None)
        success_case_ids = {
            str(getattr(memory_case, "case_id", "") or "")
            for memory_case in list(retrieved_success_cases or [])
        }
        failure_case_ids = {
            str(getattr(memory_case, "case_id", "") or "")
            for memory_case in list(retrieved_failure_cases or [])
        }
        for memory_case in list(self.success_memory):
            if str(getattr(memory_case, "case_id", "") or "") in success_case_ids:
                memory_case.retrieval_hits = int(getattr(memory_case, "retrieval_hits", 0) or 0) + 1
                memory_case.last_retrieved_slot_id = query_slot_id
        for memory_case in list(self.failure_memory):
            if str(getattr(memory_case, "case_id", "") or "") in failure_case_ids:
                memory_case.retrieval_hits = int(getattr(memory_case, "retrieval_hits", 0) or 0) + 1
                memory_case.last_retrieved_slot_id = query_slot_id
        self.success_memory = self._refresh_memory_retention(
            self.success_memory,
            max_items=self.success_memory_size,
            current_slot_id=query_slot_id,
        )
        self.failure_memory = self._refresh_memory_retention(
            self.failure_memory,
            max_items=self.reflection_memory_size,
            current_slot_id=query_slot_id,
        )
        self._persist_memory_state()

    def _retrieve_memory_cases(
        self,
        *,
        selection_query_key: SelectionQueryKey,
    ) -> tuple[list[OrchestrationMemoryCase], list[OrchestrationMemoryCase]]:
        if not self._memory_enabled():
            self.last_selection_query_key = selection_query_key
            self.last_retrieved_success_cases = []
            self.last_retrieved_failure_cases = []
            self.last_retrieved_success_traces = []
            self.last_retrieved_failure_traces = []
            return [], []

        def _sort_scored_cases(
            scored_cases: list[tuple[float, float, float, float, float, OrchestrationMemoryCase]],
        ) -> None:
            if self.memory_retrieval_mode == "shuffled":
                query_id = str(getattr(selection_query_key, "query_id", "") or "")

                def _shuffle_key(
                    item: tuple[float, float, float, float, float, OrchestrationMemoryCase],
                ) -> str:
                    case_id = str(getattr(item[5], "case_id", "") or "")
                    return hashlib.sha256(f"{query_id}|{case_id}".encode("utf-8")).hexdigest()

                scored_cases.sort(key=_shuffle_key)
                return
            scored_cases.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        def _top_cases(cases: list[OrchestrationMemoryCase]) -> list[OrchestrationMemoryCase]:
            scored_cases: list[tuple[float, float, float, float, float, OrchestrationMemoryCase]] = []
            for memory_case in list(cases or []):
                (
                    similarity_score,
                    feature_similarity,
                    recency_score,
                    retrieval_score,
                    retrieval_confidence,
                ) = self._memory_case_similarity(
                    memory_case,
                    selection_query_key=selection_query_key,
                )
                scored_cases.append(
                    (
                        retrieval_score,
                        retrieval_confidence,
                        similarity_score,
                        feature_similarity,
                        recency_score,
                        memory_case,
                    )
                )
            _sort_scored_cases(scored_cases)
            return [
                item[5]
                for item in scored_cases[: self.retrieval_top_k]
                if float(item[0]) > 0.0
            ]

        def _top_traces(cases: list[OrchestrationMemoryCase]) -> list[RetrievedMemoryCaseTrace]:
            scored_cases: list[tuple[float, float, float, float, float, OrchestrationMemoryCase]] = []
            for memory_case in list(cases or []):
                (
                    similarity_score,
                    feature_similarity,
                    recency_score,
                    retrieval_score,
                    retrieval_confidence,
                ) = self._memory_case_similarity(
                    memory_case,
                    selection_query_key=selection_query_key,
                )
                if float(retrieval_score) <= 0.0:
                    continue
                scored_cases.append(
                    (
                        retrieval_score,
                        retrieval_confidence,
                        similarity_score,
                        feature_similarity,
                        recency_score,
                        memory_case,
                    )
                )
            _sort_scored_cases(scored_cases)
            traces: list[RetrievedMemoryCaseTrace] = []
            for retrieval_rank, item in enumerate(scored_cases[: self.retrieval_top_k], start=1):
                traces.append(
                    self._retrieval_trace(
                        memory_case=item[5],
                        retrieval_rank=retrieval_rank,
                        similarity_score=item[2],
                        feature_similarity=item[3],
                        recency_score=item[4],
                        retrieval_score=item[0],
                        retrieval_confidence=item[1],
                    )
                )
            return traces

        success_cases = _top_cases(self.success_memory) if self._success_memory_enabled() else []
        failure_cases = _top_cases(self.failure_memory) if self._failure_memory_enabled() else []
        self.last_selection_query_key = selection_query_key
        self.last_retrieved_success_cases = list(success_cases)
        self.last_retrieved_failure_cases = list(failure_cases)
        self.last_retrieved_success_traces = (
            _top_traces(self.success_memory) if self._success_memory_enabled() else []
        )
        self.last_retrieved_failure_traces = (
            _top_traces(self.failure_memory) if self._failure_memory_enabled() else []
        )
        self._record_retrieval_usage(
            retrieved_success_cases=success_cases,
            retrieved_failure_cases=failure_cases,
            selection_query_key=selection_query_key,
        )
        if not self._memory_cases_feed_selection():
            return [], []
        return success_cases, failure_cases

    def _build_memory_state(self) -> OrchestrationMemoryState:
        return OrchestrationMemoryState(
            memory_enabled=self._memory_enabled(),
            memory_mode=str(self.memory_mode),
            success_cases=list(self.success_memory),
            failure_cases=list(self.failure_memory),
            success_case_ids=[
                str(getattr(memory_case, "case_id", "") or "")
                for memory_case in list(self.success_memory)
                if str(getattr(memory_case, "case_id", "") or "").strip() != ""
            ],
            reflection_note_ids=[
                f"reflection:{int(getattr(note, 'reflection_round', 0) or 0)}"
                for note in list(self.reflection_memory)
            ],
            last_selection_query_key=self.last_selection_query_key,
            last_retrieved_success_case_ids=[
                str(getattr(memory_case, "case_id", "") or "")
                for memory_case in list(self.last_retrieved_success_cases)
                if str(getattr(memory_case, "case_id", "") or "").strip() != ""
            ],
            last_retrieved_failure_case_ids=[
                str(getattr(memory_case, "case_id", "") or "")
                for memory_case in list(self.last_retrieved_failure_cases)
                if str(getattr(memory_case, "case_id", "") or "").strip() != ""
            ],
            last_retrieved_success_traces=list(self.last_retrieved_success_traces),
            last_retrieved_failure_traces=list(self.last_retrieved_failure_traces),
            last_reflection=(
                self.reflection_memory[-1]
                if self.reflection_enabled and self.reflection_memory
                else None
            ),
            notes=[
                f"memory_mode={self.memory_mode}",
                f"memory_retrieval_mode={self.memory_retrieval_mode}",
                f"memory_age_decay_enabled={str(self.memory_age_decay_enabled).lower()}",
                (
                    "reflective_transfer_guard_enabled="
                    f"{str(self.reflective_transfer_guard_enabled).lower()}"
                ),
                f"success_memory={len(self.success_memory)}",
                f"failure_memory={len(self.failure_memory)}",
                f"retrieval_encoding_version={_SELECTION_RETRIEVAL_ENCODING_VERSION}",
                (
                    "max_success_retention_score="
                    f"{max((float(getattr(item, 'retention_score', 0.0) or 0.0) for item in self.success_memory), default=0.0):.3f}"
                ),
                (
                    "max_failure_retention_score="
                    f"{max((float(getattr(item, 'retention_score', 0.0) or 0.0) for item in self.failure_memory), default=0.0):.3f}"
                ),
                f"persistent_memory_enabled={str(self.persistent_memory_enabled).lower()}",
                f"memory_store_path={self.memory_store.path if self.memory_store.path is not None else ''}",
            ],
        )

    def _available_xapp_catalog(self) -> list[dict[str, object]]:
        return available_xapp_catalog_for_llm(self.available_xapp_names)

    def _available_xapp_category_catalog(self) -> list[dict[str, object]]:
        return available_xapp_category_catalog_for_llm(self.available_xapp_names)

    def _all_known_xapps(self) -> list[str]:
        all_known_xapps: list[str] = [
            str(item.get("xapp_name", ""))
            for item in self._available_xapp_catalog()
            if str(item.get("xapp_name", "")).strip() != ""
        ]
        all_known_xapps.append("goal_parameter_tuning_xapp")
        return sorted(dict.fromkeys(all_known_xapps))

    def _enabled_disabled_xapps(self, selected_xapps: list[str]) -> tuple[list[str], list[str]]:
        all_known_xapps = self._all_known_xapps()
        enabled_xapps = list(dict.fromkeys(list(selected_xapps) + ["goal_parameter_tuning_xapp"]))
        disabled_xapps = [xapp_name for xapp_name in all_known_xapps if xapp_name not in enabled_xapps]
        return enabled_xapps, disabled_xapps

    def _heuristic_reflection(
        self,
        *,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None,
        generated_by: str,
    ) -> ReflectionNote:
        previous_plan = getattr(policy_snapshot, "xapp_selection_plan", None) if policy_snapshot is not None else None
        previous_goal = getattr(policy_snapshot, "structured_goal", None) if policy_snapshot is not None else None
        queue_bytes = int(getattr(observation, "total_queue_bytes", 0) or 0)
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        recommended_changes: list[str] = []
        if avg_bler > 0.12:
            recommended_changes.append("keep or add a urllc_guard_xapp variant")
        if queue_bytes > 60000:
            recommended_changes.append("prefer one scheduler or steering xApp that relieves backlog")
        if throughput < 10.0:
            recommended_changes.append("retain at least one throughput-oriented beam or spectrum xApp")
        note = ReflectionNote(
            reflection_round=int(len(self.reflection_memory) + 1),
            outcome_summary=(
                f"goal={getattr(previous_goal, 'goal_id', 'none')} "
                f"selected={getattr(previous_plan, 'selected_xapps', [])} "
                f"throughput={throughput:.2f}Mbps bler={avg_bler:.3f} queue={queue_bytes}"
            ),
            self_critique=(
                "Heuristic reflection fallback: the previous portfolio did not sufficiently reduce the current goal gap."
            ),
            improvement_hypothesis=(
                "Shift the portfolio toward xApps aligned with the most stressed KPI dimensions."
            ),
            recommended_changes=recommended_changes or ["rebalance the portfolio toward complementary xApp categories"],
            generated_by=generated_by,
        )
        if self._failure_memory_enabled() and not self.freeze_learning:
            self.reflection_memory.append(note)
            self.reflection_memory = self.reflection_memory[-self.reflection_memory_size :]
            self._persist_memory_state()
        return note

    def _build_reflection(
        self,
        *,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None,
        retrieved_success_cases: list[OrchestrationMemoryCase] | None = None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None = None,
    ) -> ReflectionNote:
        if not self.reflection_enabled or self.freeze_learning:
            return ReflectionNote(
                reflection_round=int(len(self.reflection_memory) + 1),
                outcome_summary="Reflection disabled for this method variant.",
                self_critique="",
                improvement_hypothesis="",
                recommended_changes=[],
                generated_by="reflection_disabled_passthrough",
            )
        previous_plan = getattr(policy_snapshot, "xapp_selection_plan", None) if policy_snapshot is not None else None
        previous_goal = getattr(policy_snapshot, "structured_goal", None) if policy_snapshot is not None else None
        queue_bytes = int(getattr(observation, "total_queue_bytes", 0) or 0)
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)

        example_reflection_input = {
            "previous_goal_id": "goal:demo_throughput_balanced",
            "previous_selected_xapps": [
                "massive_mimo_beamformer_xapp_capacity_push",
                "latency_scheduler_xapp_balanced",
                "ran_slicing_manager_xapp_sla_strict",
            ],
            "goal_gap_score": 0.42,
            "queue_bytes": 24000,
            "avg_bler": 0.21,
            "throughput_mbps": 7.4,
            "observation_summary": {
                "slot_id": 20,
                "avg_sinr_db": 10.5,
                "avg_bler": 0.21,
                "total_queue_bytes": 24000,
                "total_throughput_mbps": 7.4,
            },
        }
        example_reflection_output = {
            "outcome_summary": "Throughput improved only slightly while BLER and queue remain too high.",
            "self_critique": "The previous portfolio over-focused on scheduler and beam changes without enough mobility or steering relief.",
            "improvement_hypothesis": "Shift one slot from beam optimization toward traffic steering or power-aware stability control.",
            "recommended_changes": [
                "replace ran_slicing_manager_xapp_sla_strict with traffic_steering_xapp_quality_first",
                "avoid portfolios that use more than one scheduler-dominant xApp",
            ],
        }
        if not self.client.enabled:
            return self._heuristic_reflection(
                observation=observation,
                policy_snapshot=policy_snapshot,
                generated_by="heuristic_reflexion_fallback",
            )
        try:
            payload = self.client.generate_json_or_raise(
                system_prompt=(
                    "You are a Reflexion-style orchestration critic for a Non-RT RIC. "
                    "Given the previous xApp selection and the latest outcome, produce a "
                    "concise self-critique and actionable recommended_changes as JSON "
                    "with keys: outcome_summary, self_critique, improvement_hypothesis, "
                    "recommended_changes. Return JSON only."
                ),
                user_prompt=llm_json_dumps(
                    {
                        "task": "Reflect on the last orchestration outcome and propose changes before reselection.",
                        "output_rules": [
                            "Return exactly one JSON object.",
                            "recommended_changes must be a list of short strings.",
                            "Do not include markdown or explanations outside JSON.",
                        ],
                        "few_shot_example": {
                            "input": example_reflection_input,
                            "output": example_reflection_output,
                        },
                        "input": {
                            "previous_goal_id": str(getattr(previous_goal, "goal_id", "none")),
                            "previous_selected_xapps": list(getattr(previous_plan, "selected_xapps", []) or []),
                            "goal_gap_score": (
                                float(
                                    goal_gap_score(
                                        previous_goal,
                                        observation,
                                        validated_intent_template=getattr(
                                            policy_snapshot,
                                            "validated_intent_template",
                                            None,
                                        ),
                                    )
                                )
                                if previous_goal is not None
                                else None
                            ),
                            "queue_bytes": queue_bytes,
                            "avg_bler": avg_bler,
                            "throughput_mbps": throughput,
                            "observation_summary": observation_summary_for_llm(observation),
                            "available_xapps": self._available_xapp_catalog(),
                            "available_xapp_categories": self._available_xapp_category_catalog(),
                            "retrieved_success_cases": [
                                self._memory_case_summary(memory_case)
                                for memory_case in list(retrieved_success_cases or [])
                            ],
                            "retrieved_failure_cases": [
                                self._memory_case_summary(memory_case)
                                for memory_case in list(retrieved_failure_cases or [])
                            ],
                        },
                    }
                ),
            )
        except Exception:
            return self._heuristic_reflection(
                observation=observation,
                policy_snapshot=policy_snapshot,
                generated_by=f"{self.client.generated_by_tag}_heuristic_reflexion_fallback",
            )
        if not isinstance(payload, dict):
            return self._heuristic_reflection(
                observation=observation,
                policy_snapshot=policy_snapshot,
                generated_by=f"{self.client.generated_by_tag}_invalid_reflexion_fallback",
            )
        recommended_changes = payload.get("recommended_changes", [])
        if not isinstance(recommended_changes, list):
            recommended_changes = []
        note = ReflectionNote(
            reflection_round=int(len(self.reflection_memory) + 1),
            outcome_summary=str(
                payload.get(
                    "outcome_summary",
                    f"goal={getattr(previous_goal, 'goal_id', 'none')} selected={getattr(previous_plan, 'selected_xapps', [])}",
                )
                or ""
            ),
            self_critique=str(payload.get("self_critique", "") or ""),
            improvement_hypothesis=str(payload.get("improvement_hypothesis", "") or ""),
            recommended_changes=[str(item) for item in recommended_changes if str(item).strip() != ""],
            generated_by=f"{self.client.generated_by_tag}_reflexion",
        )
        if self._failure_memory_enabled() and not self.freeze_learning:
            self.reflection_memory.append(note)
            self.reflection_memory = self.reflection_memory[-self.reflection_memory_size :]
            self._persist_memory_state()
        return note

    def _select_variant_name(self, category_name: str, *, goal, reflection: ReflectionNote | None) -> str:
        category_name = str(category_name).strip()
        variants = [
            str(item.get("name", ""))
            for item in DEFAULT_XAPP_VARIANTS.get(category_name, [])
            if str(item.get("name", "")).strip() != ""
        ]
        if len(variants) == 0:
            return category_name

        goal_text = str(getattr(goal, "intent_text", "") or "").lower()
        risk_posture = str(getattr(goal, "risk_posture", "") or "").lower()
        reflection_text = " ".join(
            [
                str(getattr(reflection, "self_critique", "") or "").lower(),
                str(getattr(reflection, "improvement_hypothesis", "") or "").lower(),
                " ".join(str(item).lower() for item in list(getattr(reflection, "recommended_changes", []) or [])),
            ]
        )
        combined = f"{goal_text} {risk_posture} {reflection_text}"

        preferred_tokens_by_category = {
            "mobility_predictor_xapp": [
                ("blockage", "mobility_predictor_xapp_mmwave_guard"),
                ("fading", "mobility_predictor_xapp_mmwave_guard"),
                ("moving", "mobility_predictor_xapp_hysteresis_guard"),
                ("ping-pong", "mobility_predictor_xapp_hysteresis_guard"),
                ("ping pong", "mobility_predictor_xapp_hysteresis_guard"),
                ("stability", "mobility_predictor_xapp_hysteresis_guard"),
                ("stable", "mobility_predictor_xapp_hysteresis_guard"),
                ("high_speed", "mobility_predictor_xapp_high_speed"),
                ("high-mobility", "mobility_predictor_xapp_hysteresis_guard"),
                ("high mobility", "mobility_predictor_xapp_hysteresis_guard"),
                ("handover", "mobility_predictor_xapp_hysteresis_guard"),
                ("edge", "mobility_predictor_xapp_cell_edge"),
                ("mmwave", "mobility_predictor_xapp_mmwave_guard"),
                ("balanced", "mobility_predictor_xapp_balanced"),
            ],
            "traffic_steering_xapp": [
                ("blockage", "traffic_steering_xapp_edge_rescue"),
                ("fading", "traffic_steering_xapp_edge_rescue"),
                ("video", "traffic_steering_xapp_quality_first"),
                ("ping-pong", "traffic_steering_xapp_sticky"),
                ("ping pong", "traffic_steering_xapp_sticky"),
                ("stability", "traffic_steering_xapp_quality_first"),
                ("stable", "traffic_steering_xapp_sticky"),
                ("high-mobility", "traffic_steering_xapp_quality_first"),
                ("high mobility", "traffic_steering_xapp_quality_first"),
                ("edge", "traffic_steering_xapp_edge_rescue"),
                ("hotspot", "traffic_steering_xapp_hotspot_relief"),
                ("load", "traffic_steering_xapp_load_first"),
                ("quality", "traffic_steering_xapp_quality_first"),
                ("mmwave", "traffic_steering_xapp_mmwave_prefer"),
                ("macro", "traffic_steering_xapp_macro_anchor"),
            ],
            "power_saving_xapp": [
                ("energy", "power_saving_xapp_energy_max"),
                ("power", "power_saving_xapp_balanced"),
                ("green", "power_saving_xapp_balanced"),
                ("urllc", "power_saving_xapp_urlcc_safe"),
                ("queue", "power_saving_xapp_queue_sensitive"),
            ],
            "spectrum_optimizer_xapp": [
                ("premium", "spectrum_optimizer_xapp_latency_mix"),
                ("tenant", "spectrum_optimizer_xapp_latency_mix"),
                ("sla", "spectrum_optimizer_xapp_latency_mix"),
                ("urllc", "spectrum_optimizer_xapp_latency_mix"),
                ("blockage", "spectrum_optimizer_xapp_interference_averse"),
                ("fading", "spectrum_optimizer_xapp_interference_averse"),
                ("latency", "spectrum_optimizer_xapp_latency_mix"),
                ("interference", "spectrum_optimizer_xapp_interference_averse"),
                ("throughput", "spectrum_optimizer_xapp_dense_packing"),
                ("capacity", "spectrum_optimizer_xapp_aggressive"),
                ("balanced", "spectrum_optimizer_xapp_balanced"),
            ],
            "latency_scheduler_xapp": [
                ("premium", "latency_scheduler_xapp_urllc_fastpath"),
                ("tenant", "latency_scheduler_xapp_urllc_fastpath"),
                ("sla", "latency_scheduler_xapp_urllc_fastpath"),
                ("urllc", "latency_scheduler_xapp_urllc_fastpath"),
                ("latency", "latency_scheduler_xapp_balanced"),
                ("tail", "latency_scheduler_xapp_tail_guard"),
                ("harq", "latency_scheduler_xapp_harq_rescue"),
                ("edge", "latency_scheduler_xapp_edge_safe"),
            ],
            "massive_mimo_beamformer_xapp": [
                ("video", "massive_mimo_beamformer_xapp_capacity_push"),
                ("blockage", "massive_mimo_beamformer_xapp_coverage_boost"),
                ("fading", "massive_mimo_beamformer_xapp_coverage_boost"),
                ("beam", "massive_mimo_beamformer_xapp_sinr_max"),
                ("throughput", "massive_mimo_beamformer_xapp_capacity_push"),
                ("capacity", "massive_mimo_beamformer_xapp_capacity_push"),
                ("coverage", "massive_mimo_beamformer_xapp_coverage_boost"),
                ("edge", "massive_mimo_beamformer_xapp_edge_rescue"),
                ("sinr", "massive_mimo_beamformer_xapp_sinr_max"),
            ],
            "ran_slicing_manager_xapp": [
                ("premium", "ran_slicing_manager_xapp_sla_strict"),
                ("tenant", "ran_slicing_manager_xapp_sla_strict"),
                ("sla", "ran_slicing_manager_xapp_sla_strict"),
                ("urllc", "ran_slicing_manager_xapp_urllc_bias"),
                ("embb", "ran_slicing_manager_xapp_embb_bias"),
                ("slice", "ran_slicing_manager_xapp_balanced"),
                ("fair", "ran_slicing_manager_xapp_fairness"),
            ],
            "urllc_guard_xapp": [
                ("premium", "urllc_guard_xapp_strict"),
                ("tenant", "urllc_guard_xapp_strict"),
                ("sla", "urllc_guard_xapp_strict"),
                ("bler", "urllc_guard_xapp_strict"),
                ("strict", "urllc_guard_xapp_strict"),
                ("urllc", "urllc_guard_xapp_balanced"),
                ("tail", "urllc_guard_xapp_tail_safe"),
                ("harq", "urllc_guard_xapp_harq_safe"),
                ("edge", "urllc_guard_xapp_edge_safe"),
            ],
            "cell_sleep_xapp": [
                ("energy", "cell_sleep_xapp_cluster_saver"),
                ("sleep", "cell_sleep_xapp_balanced"),
                ("night", "cell_sleep_xapp_night_mode"),
                ("aggressive", "cell_sleep_xapp_aggressive"),
                ("urllc", "cell_sleep_xapp_urlcc_safe"),
            ],
        }
        for token, variant_name in preferred_tokens_by_category.get(category_name, []):
            if token in combined and variant_name in variants:
                return variant_name
        for fallback_name in [
            f"{category_name}_balanced",
            f"{category_name}_strict",
            f"{category_name}_quality_first",
            f"{category_name}_capacity_push",
        ]:
            if fallback_name in variants:
                return fallback_name
        return variants[0]

    def _xapp_catalog_by_name(self) -> dict[str, dict[str, object]]:
        return {
            str(item.get("xapp_name", "")): item
            for item in self._available_xapp_catalog()
            if str(item.get("xapp_name", "")).strip() != ""
        }

    def _resolve_selected_xapp_names(
        self,
        *,
        requested_names: object,
        goal,
        reflection: ReflectionNote | None,
    ) -> list[str]:
        if not isinstance(requested_names, list):
            return []
        catalog_by_name = self._xapp_catalog_by_name()
        resolved_names: list[str] = []
        seen_categories: set[str] = set()
        for raw_name in requested_names:
            normalized_name = str(raw_name or "").strip()
            if normalized_name == "":
                continue
            exact_name = normalized_name if normalized_name in catalog_by_name else ""
            category_name = canonical_xapp_category(exact_name or normalized_name)
            if category_name == "" or category_name in seen_categories:
                continue
            preserve_exact_balanced_names = {
                "traffic_steering_xapp_balanced",
                "latency_scheduler_xapp_balanced",
                "urllc_guard_xapp_balanced",
            }
            should_respecialize_generic_variant = (
                self.selection_mode == "reflective"
                and self.memory_mode == "full"
                and exact_name != ""
                and exact_name != category_name
                and exact_name.endswith("_balanced")
                and exact_name not in preserve_exact_balanced_names
            )
            if exact_name != "" and exact_name != category_name and not should_respecialize_generic_variant:
                selected_name = exact_name
            else:
                selected_name = self._select_variant_name(
                    category_name,
                    goal=goal,
                    reflection=reflection,
                )
            if selected_name not in catalog_by_name:
                if exact_name in catalog_by_name:
                    selected_name = exact_name
                elif category_name in catalog_by_name:
                    selected_name = category_name
                else:
                    continue
            resolved_names.append(selected_name)
            seen_categories.add(category_name)
            if len(resolved_names) >= self.max_selected_xapps:
                break
        return resolved_names

    def _requested_categories(self, *, goal, observation: Observation) -> list[str]:
        _ = observation
        kpi_names = {
            str(getattr(target, "kpi_name", "") or "").strip().lower()
            for target in list(getattr(goal, "kpi_targets", []) or [])
        }
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
        ).lower()
        requested_categories: list[str] = []
        if "total_throughput_mbps" in kpi_names or "throughput" in goal_text:
            requested_categories.extend(
                ["massive_mimo_beamformer_xapp", "spectrum_optimizer_xapp"]
            )
        if "avg_bler" in kpi_names or "bler" in goal_text or "reliability" in goal_text:
            requested_categories.append("urllc_guard_xapp")
        if (
            "queue_bytes" in kpi_names
            or "total_queue_bytes" in kpi_names
            or "queue" in goal_text
        ):
            requested_categories.extend(["latency_scheduler_xapp", "traffic_steering_xapp"])
        if "handover_risk" in kpi_names or "handover" in goal_text or "mobility" in goal_text:
            requested_categories.extend(["mobility_predictor_xapp", "traffic_steering_xapp"])
        if len(requested_categories) == 0:
            requested_categories = [
                "massive_mimo_beamformer_xapp",
                "urllc_guard_xapp",
                "traffic_steering_xapp",
            ]
        return list(dict.fromkeys(requested_categories))

    def _fixed_plan_categories(self) -> list[str]:
        return [
            "traffic_steering_xapp",
            "latency_scheduler_xapp",
        ][: self.max_selected_xapps]

    def _service_demands(self, *, goal, observation: Observation) -> list[str]:
        """Map intent/KPI state to service demands used by literature baselines."""

        kpi_names = set(self._goal_kpi_names(goal))
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
        ).lower()
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        avg_sinr = float(getattr(observation, "avg_sinr_db", 0.0) or 0.0)
        queue_bytes = float(getattr(observation, "total_queue_bytes", 0.0) or 0.0)
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)

        demands: list[str] = []
        if (
            "total_throughput_mbps" in kpi_names
            or "throughput" in goal_text
            or "video" in goal_text
            or "edge" in goal_text
        ):
            demands.append("capacity")
        if (
            "avg_bler" in kpi_names
            or "queue_bytes" in kpi_names
            or "total_queue_bytes" in kpi_names
            or "urllc" in goal_text
            or "latency" in goal_text
            or "reliability" in goal_text
            or avg_bler >= 0.12
            or queue_bytes >= 60000.0
        ):
            demands.append("latency_reliability")
        if (
            "handover_risk" in kpi_names
            or "handover" in goal_text
            or "mobility" in goal_text
            or handover_pending >= 2.0
        ):
            demands.append("mobility")
        if (
            "sinr" in kpi_names
            or "avg_sinr_db" in kpi_names
            or "interference" in goal_text
            or avg_sinr < 8.0
        ):
            demands.append("radio_interference")
        if "energy" in goal_text or "sleep" in goal_text or "power_cost" in kpi_names:
            demands.append("energy")
        if "slice" in goal_text or "tenant" in goal_text or "sla" in goal_text or "fair" in goal_text:
            demands.append("slice_sla")
        if len(demands) == 0:
            demands.append("capacity")
        return list(dict.fromkeys(demands))

    def _service_demand_category_requirements(self) -> dict[str, list[str]]:
        return {
            "capacity": [
                "massive_mimo_beamformer_xapp",
                "traffic_steering_xapp",
            ],
            "latency_reliability": [
                "urllc_guard_xapp",
                "latency_scheduler_xapp",
            ],
            "mobility": [
                "mobility_predictor_xapp",
                "traffic_steering_xapp",
            ],
            "radio_interference": [
                "spectrum_optimizer_xapp",
                "massive_mimo_beamformer_xapp",
            ],
            "energy": [
                "power_saving_xapp",
                "cell_sleep_xapp",
            ],
            "slice_sla": [
                "ran_slicing_manager_xapp",
                "latency_scheduler_xapp",
            ],
        }

    def _orchestran_ni_catalog_categories(
        self,
        *,
        goal,
        observation: Observation,
        requested_categories: list[str],
    ) -> list[str]:
        """OrchestRAN-style NI selection before resource/location optimization."""

        demands = self._service_demands(goal=goal, observation=observation)
        requirement_map = self._service_demand_category_requirements()
        categories: list[str] = []
        for demand_name in demands:
            categories.extend(requirement_map.get(demand_name, []))

        # OrchestRAN's orchestration target is an executable NI chain. Add
        # sensing/guard functions needed by fast control loops instead of
        # using reflective failure memory.
        if "mobility" in demands and "latency_reliability" in demands:
            categories.extend(["mobility_predictor_xapp", "urllc_guard_xapp"])
        if "radio_interference" in demands and "capacity" in demands:
            categories.extend(["spectrum_optimizer_xapp", "massive_mimo_beamformer_xapp"])
        categories.extend(requested_categories)
        return list(dict.fromkeys(categories))[: self.max_selected_xapps]

    def _intent_hrl_target_classes(
        self,
        *,
        goal,
        observation: Observation,
    ) -> list[str]:
        """Meta-controller target classes from intent and quantized state."""

        demands = self._service_demands(goal=goal, observation=observation)
        queue_bucket = self._bucket_label(
            float(getattr(observation, "total_queue_bytes", 0.0) or 0.0),
            low=24000.0,
            high=90000.0,
        )
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        avg_sinr = float(getattr(observation, "avg_sinr_db", 0.0) or 0.0)
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
        target_classes = list(demands)
        if queue_bucket == "high" or avg_bler >= 0.16:
            target_classes.append("latency_reliability")
        if avg_sinr < 7.0:
            target_classes.append("radio_interference")
        if handover_pending >= 6.0:
            target_classes.append("mobility")
        return list(dict.fromkeys(target_classes))

    def _intent_hrl_action_categories(
        self,
        *,
        target_classes: list[str],
        requested_categories: list[str],
    ) -> list[str]:
        """Low-level controller action abstraction for xApp composition."""

        requirement_map = self._service_demand_category_requirements()
        categories: list[str] = []
        for target_class in list(target_classes or []):
            categories.extend(requirement_map.get(str(target_class), []))
        # Keep the controller action bounded by the high-level target classes:
        # requested categories are considered only after the HRL action has
        # covered its target services.
        categories.extend(requested_categories[:1])
        return list(dict.fromkeys(categories))[: self.max_selected_xapps]

    def _oreo_service_graph_categories(
        self,
        *,
        goal,
        observation: Observation,
        requested_categories: list[str],
    ) -> list[str]:
        """OREO-style service graph solver with admission and xApp sharing."""

        solver_output = self._oreo_service_graph_solver(
            goal=goal,
            observation=observation,
            requested_categories=requested_categories,
        )
        return list(solver_output.get("selected_categories", []) or [])[: self.max_selected_xapps]

    def _oreo_semantic_groups(self) -> dict[str, set[str]]:
        return {
            "qos_control": {"latency_scheduler_xapp", "urllc_guard_xapp", "ran_slicing_manager_xapp"},
            "radio_control": {"spectrum_optimizer_xapp", "massive_mimo_beamformer_xapp"},
            "traffic_control": {"traffic_steering_xapp", "mobility_predictor_xapp"},
            "energy_control": {"power_saving_xapp", "cell_sleep_xapp"},
        }

    def _category_resource_unit_cost(self, category_name: str) -> float:
        resource_cost = {
            "latency_scheduler_xapp": 0.18,
            "urllc_guard_xapp": 0.20,
            "traffic_steering_xapp": 0.22,
            "mobility_predictor_xapp": 0.20,
            "spectrum_optimizer_xapp": 0.24,
            "massive_mimo_beamformer_xapp": 0.30,
            "ran_slicing_manager_xapp": 0.16,
            "power_saving_xapp": 0.14,
            "cell_sleep_xapp": 0.12,
        }
        return float(resource_cost.get(str(category_name), 0.20))

    def _oreo_service_graph_solver(
        self,
        *,
        goal,
        observation: Observation,
        requested_categories: list[str],
    ) -> dict[str, object]:
        """Multi-layer OREO-like service configuration/admission solver."""

        demands = self._service_demands(goal=goal, observation=observation)
        requirement_map = self._service_demand_category_requirements()
        requested = [
            str(item).strip()
            for item in list(requested_categories or [])
            if str(item).strip() != ""
        ]
        service_options: list[tuple[str, list[list[str]]]] = []
        for demand_name in demands:
            required_components = list(requirement_map.get(str(demand_name), []) or [])
            if len(required_components) == 0:
                continue
            options: list[list[str]] = [required_components[:1]]
            if len(required_components) > 1:
                options.append(required_components[:2])
            if str(demand_name) not in {"latency_reliability", "mobility", "radio_interference"}:
                options.append([])
            service_options.append((str(demand_name), options))
        if len(service_options) == 0:
            service_options = [("capacity", [["massive_mimo_beamformer_xapp"], ["traffic_steering_xapp"]])]

        candidate_configurations: list[list[str]] = []
        for option_tuple in itertools.product(*[item[1] for item in service_options]):
            flattened: list[str] = []
            for option in option_tuple:
                flattened.extend(list(option or []))
            flattened.extend(requested[:1])
            flattened = list(dict.fromkeys(flattened))
            if len(flattened) == 0:
                continue
            if len(flattened) <= self.max_selected_xapps:
                candidate_configurations.append(flattened)
            else:
                for combination in itertools.combinations(flattened, self.max_selected_xapps):
                    candidate_configurations.append(list(combination))
        if len(candidate_configurations) == 0:
            candidate_configurations.append(requested[: self.max_selected_xapps] or ["massive_mimo_beamformer_xapp"])

        semantic_groups = self._oreo_semantic_groups()

        def _configuration_terms(categories: list[str]) -> dict[str, object]:
            selected = set(categories)
            admitted_services: list[str] = []
            dependency_mismatch = 0.0
            for demand_name in demands:
                required = set(requirement_map.get(str(demand_name), []) or [])
                if len(required) == 0:
                    continue
                coverage = float(len(selected & required)) / float(len(required))
                if coverage >= 0.5:
                    admitted_services.append(str(demand_name))
                else:
                    dependency_mismatch += 1.0 - coverage
            active_groups = [
                group_name
                for group_name, group_members in semantic_groups.items()
                if len(selected & set(group_members)) > 0
            ]
            resource_cost = sum(self._category_resource_unit_cost(category) for category in selected)
            service_admission_ratio = float(len(admitted_services)) / float(max(1, len(demands)))
            sharing_gain = (
                float(len(admitted_services)) / float(max(1, len(selected)))
                if len(selected) > 0
                else 0.0
            )
            requested_set = set(requested)
            interface_mismatch = (
                1.0 - float(len(selected & requested_set)) / float(len(requested_set))
                if len(requested_set) > 0
                else 0.0
            )
            objective = (
                2.20 * service_admission_ratio
                + 0.45 * sharing_gain
                - 0.42 * resource_cost
                - 0.55 * dependency_mismatch
                - 0.18 * max(0.0, interface_mismatch)
            )
            return {
                "admitted_services": admitted_services,
                "active_semantic_groups": active_groups,
                "resource_cost": float(resource_cost),
                "service_admission_ratio": float(service_admission_ratio),
                "sharing_gain": float(sharing_gain),
                "dependency_mismatch": float(dependency_mismatch),
                "interface_mismatch": float(max(0.0, interface_mismatch)),
                "objective": float(objective),
            }

        best_categories: list[str] = []
        best_terms: dict[str, object] = {}
        best_objective = -float("inf")
        for configuration in candidate_configurations:
            terms = _configuration_terms(configuration)
            objective = float(terms.get("objective", 0.0) or 0.0)
            if objective > best_objective:
                best_objective = objective
                best_terms = terms
                best_categories = list(configuration)

        service_graph = {
            "service_layer": list(demands),
            "configuration_layer": [
                {
                    "service": demand_name,
                    "required_components": list(requirement_map.get(demand_name, []) or []),
                }
                for demand_name in demands
            ],
            "xapp_layer": list(best_categories),
            "sharing_groups": {
                group_name: sorted(set(best_categories) & set(group_members))
                for group_name, group_members in semantic_groups.items()
                if len(set(best_categories) & set(group_members)) > 0
            },
        }
        solver_output = {
            "selected_categories": list(best_categories),
            "terms": dict(best_terms),
            "service_graph": service_graph,
            "candidate_configuration_count": int(len(candidate_configurations)),
        }
        if self.literature_baseline_state_enabled:
            graph_state = self._baseline_state_section("oreo_service_graph")
            history = list(graph_state.get("history", []) or [])
            history.append(
                {
                    "phase": str(self.experiment_transfer_phase),
                    "repeat": int(self.experiment_phase_repeat),
                    "selected_categories": list(best_categories),
                    "terms": dict(best_terms),
                    "service_graph": service_graph,
                }
            )
            graph_state["last_solver_output"] = solver_output
            graph_state["history"] = history[-128:]
            self._persist_literature_baseline_state()
        return solver_output

    def _intent_hrl_high_level_categories(
        self,
        *,
        goal,
        observation: Observation,
        requested_categories: list[str],
    ) -> list[str]:
        """Intent-HRL-inspired meta-controller/controller abstraction.

        The baseline deliberately uses coarse target classes rather than
        outcome memory: the meta-controller maps the operator intent to KPI
        subgoals, and the low-level controller maps those subgoals plus a
        quantized state bucket to xApp categories.
        """
        target_classes = self._intent_hrl_target_classes(goal=goal, observation=observation)
        return self._intent_hrl_action_categories(
            target_classes=target_classes,
            requested_categories=requested_categories,
        )

    def _service_graph_categories(
        self,
        *,
        goal,
        observation: Observation,
        requested_categories: list[str],
    ) -> list[str]:
        return self._oreo_service_graph_categories(
            goal=goal,
            observation=observation,
            requested_categories=requested_categories,
        )

    def _candidate_category_portfolios(
        self,
        *,
        base_categories: list[str],
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
        previous_selected_xapps: list[str],
    ) -> list[list[str]]:
        portfolios: list[list[str]] = []

        def _append_portfolio(category_items: list[str]) -> None:
            normalized_items = [
                str(item).strip()
                for item in list(category_items or [])
                if str(item).strip() != ""
            ]
            normalized_items = list(dict.fromkeys(normalized_items))[: self.max_selected_xapps]
            if len(normalized_items) == 0:
                return
            if normalized_items not in portfolios:
                portfolios.append(normalized_items)

        success_categories: list[str] = []
        for memory_case in list(retrieved_success_cases or []):
            success_categories.extend(
                self._xapp_categories(list(getattr(memory_case, "selected_xapps", []) or []))
            )
        failure_categories: list[str] = []
        for memory_case in list(retrieved_failure_cases or []):
            failure_categories.extend(
                self._xapp_categories(list(getattr(memory_case, "selected_xapps", []) or []))
            )
        previous_categories = self._xapp_categories(previous_selected_xapps)

        success_categories = list(dict.fromkeys(success_categories))
        failure_categories = list(dict.fromkeys(failure_categories))
        previous_categories = list(dict.fromkeys(previous_categories))

        _append_portfolio(base_categories)
        _append_portfolio(base_categories[: max(1, self.max_selected_xapps - 1)] + success_categories[:1])
        if len(success_categories) > 0:
            _append_portfolio(success_categories[: self.max_selected_xapps])
            _append_portfolio(base_categories[:2] + success_categories[:2])
        if len(previous_categories) > 0:
            _append_portfolio(previous_categories[: self.max_selected_xapps])
            _append_portfolio(base_categories[:1] + previous_categories[:2])
        if len(failure_categories) > 0:
            _append_portfolio(
                [category_name for category_name in base_categories if category_name not in failure_categories]
                or base_categories[: self.max_selected_xapps]
            )
        return portfolios[: max(3, self.max_selected_xapps + 2)]

    def _candidate_eta_profile_names(self, *, base_categories: list[str], goal) -> list[str]:
        kpi_names = self._goal_kpi_names(goal)
        risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()
        profile_names: list[str] = ["eta:balanced"]
        if "handover_risk" in kpi_names or "mobility_predictor_xapp" in list(base_categories or []):
            profile_names.insert(0, "eta:mobility_stable")
        elif "avg_bler" in kpi_names or risk_posture == "conservative":
            profile_names.insert(0, "eta:reliability_guarded")
        elif "total_throughput_mbps" in kpi_names:
            profile_names.insert(0, "eta:throughput_push")
        return list(dict.fromkeys(profile_names))[:2]

    def _candidate_chi_profile_names(self, *, base_categories: list[str], goal) -> list[str]:
        _ = base_categories
        kpi_names = self._goal_kpi_names(goal)
        risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()
        profile_names: list[str] = ["chi:balanced_weighted"]
        if "avg_bler" in kpi_names or risk_posture == "conservative":
            profile_names.insert(0, "chi:guardrail_first")
        elif "total_throughput_mbps" in kpi_names and risk_posture == "aggressive":
            profile_names.insert(0, "chi:priority_scheduler")
        return list(dict.fromkeys(profile_names))[:2]

    def _candidate_plan_portfolios(
        self,
        *,
        candidate_requested_names: list[list[str]],
        goal,
        observation: Observation,
        reflection: ReflectionNote | None,
        base_categories: list[str],
    ) -> list[OrchestrationCandidatePlan]:
        eta_profile_names = self._candidate_eta_profile_names(
            base_categories=base_categories,
            goal=goal,
        )
        chi_profile_names = self._candidate_chi_profile_names(
            base_categories=base_categories,
            goal=goal,
        )
        candidate_plans: list[OrchestrationCandidatePlan] = []
        seen_signatures: set[tuple[str, tuple[str, ...], str, str]] = set()
        for candidate_requested_name_set in list(candidate_requested_names or []):
            resolved_names = self._resolve_selected_xapp_names(
                requested_names=candidate_requested_name_set,
                goal=goal,
                reflection=reflection,
            )
            if len(resolved_names) == 0:
                continue
            requested_signature = tuple(
                str(item).strip()
                for item in list(candidate_requested_name_set or [])
                if str(item).strip() != ""
            )
            for eta_profile_name in eta_profile_names:
                for chi_profile_name in chi_profile_names:
                    signature = (
                        "|".join(requested_signature),
                        tuple(sorted(resolved_names)),
                        str(eta_profile_name),
                        str(chi_profile_name),
                    )
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)
                    candidate_plans.append(
                        self.feasibility_projector.project_candidate(
                            candidate_id=(
                                "candidate:"
                                + ":".join(self._xapp_categories(resolved_names) or ["uncategorized"])
                                + f":{eta_profile_name}:{chi_profile_name}"
                            ),
                            requested_xapps=resolved_names,
                            goal=goal,
                            observation=observation,
                            available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                            requested_eta_profile=str(eta_profile_name),
                            requested_chi_profile=str(chi_profile_name),
                        )
                    )
        return candidate_plans

    def _portfolio_score_breakdown(
        self,
        *,
        candidate_plan: OrchestrationCandidatePlan,
        requested_categories: list[str],
        base_categories: list[str],
        previous_selected_xapps: list[str],
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
        goal,
        observation: Observation,
    ) -> CandidatePlanScore:
        candidate_score = self.surrogate_scorer.score_candidate_plan(
            candidate_plan=candidate_plan,
            requested_categories=list(requested_categories),
            base_categories=list(base_categories),
            previous_selected_xapps=list(previous_selected_xapps),
            retrieved_success_cases=list(retrieved_success_cases or []),
            retrieved_failure_cases=list(retrieved_failure_cases or []),
        )
        return self._score_for_selection_mode(
            candidate_score=candidate_score,
            requested_categories=list(requested_categories),
            goal=goal,
            observation=observation,
        )

    def _orchestran_location_terms(
        self,
        *,
        candidate_score: CandidatePlanScore,
        goal,
        observation: Observation,
    ) -> dict[str, object]:
        """Approximate OrchestRAN algorithm-placement/timing objective."""

        _ = observation
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
        ).lower()
        kpi_names = set(self._goal_kpi_names(goal))
        selected_categories = self._xapp_categories(list(candidate_score.selected_xapps))
        deadline_budget = 0.85
        if (
            "urllc" in goal_text
            or "latency" in goal_text
            or "handover_risk" in kpi_names
            or "avg_bler" in kpi_names
        ):
            deadline_budget = 0.58
        if "energy" in goal_text or "slice" in goal_text:
            deadline_budget = max(deadline_budget, 0.95)

        location_latency = {
            "near_rt": 0.10,
            "edge": 0.22,
            "non_rt": 0.46,
        }
        location_resource_cost = {
            "near_rt": 0.34,
            "edge": 0.18,
            "non_rt": 0.08,
        }
        function_location_affinity = {
            "latency_scheduler_xapp": {"near_rt": 0.00, "edge": 0.16, "non_rt": 0.45},
            "urllc_guard_xapp": {"near_rt": 0.00, "edge": 0.14, "non_rt": 0.45},
            "traffic_steering_xapp": {"near_rt": 0.02, "edge": 0.06, "non_rt": 0.35},
            "mobility_predictor_xapp": {"near_rt": 0.02, "edge": 0.08, "non_rt": 0.30},
            "spectrum_optimizer_xapp": {"near_rt": 0.06, "edge": 0.02, "non_rt": 0.30},
            "massive_mimo_beamformer_xapp": {"near_rt": 0.10, "edge": 0.00, "non_rt": 0.32},
            "ran_slicing_manager_xapp": {"near_rt": 0.12, "edge": 0.00, "non_rt": 0.20},
            "power_saving_xapp": {"near_rt": 0.18, "edge": 0.06, "non_rt": 0.00},
            "cell_sleep_xapp": {"near_rt": 0.18, "edge": 0.06, "non_rt": 0.00},
        }
        lifecycle_state = self._baseline_state_section("orchestran_lifecycle")
        previous_placement = dict(lifecycle_state.get("previous_placement", {}) or {})
        candidate_locations = ("near_rt", "edge", "non_rt")
        best_objective = float("inf")
        placement: dict[str, str] = {}
        execution_time = 0.0
        migration_cost = 0.0
        lifecycle_cost = 0.0
        for combination in itertools.product(candidate_locations, repeat=len(selected_categories)):
            trial_placement = {
                category_name: str(location)
                for category_name, location in zip(selected_categories, combination)
            }
            trial_execution_time = sum(float(location_latency[location]) for location in combination)
            trial_timing_violation = max(0.0, float(trial_execution_time) - float(deadline_budget))
            trial_resource_cost = sum(float(location_resource_cost[location]) for location in combination)
            trial_affinity_penalty = sum(
                float(function_location_affinity.get(category_name, {}).get(location, 0.18))
                for category_name, location in trial_placement.items()
            )
            trial_migration_cost = sum(
                0.08
                for category_name, location in trial_placement.items()
                if previous_placement.get(category_name) not in {None, "", location}
            )
            trial_lifecycle_cost = (
                0.03 * float(len(selected_categories))
                + 0.10 * float(candidate_score.switching_cost)
                + float(trial_migration_cost)
            )
            objective = (
                1.10 * float(trial_timing_violation)
                + 0.38 * float(trial_resource_cost)
                + 0.34 * float(trial_affinity_penalty)
                + 0.24 * float(trial_lifecycle_cost)
            )
            if objective < best_objective:
                best_objective = float(objective)
                placement = dict(trial_placement)
                execution_time = float(trial_execution_time)
                migration_cost = float(trial_migration_cost)
                lifecycle_cost = float(trial_lifecycle_cost)
        timing_violation = max(0.0, execution_time - deadline_budget)
        if self._baseline_training_active("orchestran"):
            lifecycle_state["previous_placement"] = dict(placement)
            history = list(lifecycle_state.get("history", []) or [])
            history.append(
                {
                    "phase": str(self.experiment_transfer_phase),
                    "repeat": int(self.experiment_phase_repeat),
                    "placement": dict(placement),
                    "execution_time": float(execution_time),
                    "deadline_budget": float(deadline_budget),
                    "timing_violation": float(timing_violation),
                    "migration_cost": float(migration_cost),
                    "objective": float(best_objective),
                }
            )
            lifecycle_state["history"] = history[-128:]
            self._persist_literature_baseline_state()
        return {
            "placement": placement,
            "execution_time": float(execution_time),
            "deadline_budget": float(deadline_budget),
            "timing_violation": float(timing_violation),
            "lifecycle_cost": float(lifecycle_cost),
            "placement_objective": float(best_objective),
            "migration_cost": float(migration_cost),
        }

    def _intent_hrl_terms(
        self,
        *,
        candidate_score: CandidatePlanScore,
        requested_categories: list[str],
        goal,
        observation: Observation,
    ) -> dict[str, float]:
        target_classes = self._intent_hrl_target_classes(goal=goal, observation=observation)
        requirement_map = self._service_demand_category_requirements()
        selected_categories = set(self._xapp_categories(list(candidate_score.selected_xapps)))
        class_coverages: list[float] = []
        for target_class in target_classes:
            required = set(requirement_map.get(str(target_class), []))
            if len(required) == 0:
                continue
            class_coverages.append(float(len(selected_categories & required)) / float(len(required)))
        meta_coverage = sum(class_coverages) / float(len(class_coverages)) if class_coverages else 0.0
        requested_set = set(self._xapp_categories(list(requested_categories)) or list(requested_categories))
        service_coverage = (
            float(len(selected_categories & requested_set)) / float(len(requested_set))
            if len(requested_set) > 0
            else meta_coverage
        )
        queue_bucket = self._bucket_label(
            float(getattr(observation, "total_queue_bytes", 0.0) or 0.0),
            low=24000.0,
            high=90000.0,
        )
        state_action_bonus = 0.0
        if queue_bucket == "high" and "latency_scheduler_xapp" in selected_categories:
            state_action_bonus += 0.08
        if float(getattr(observation, "avg_sinr_db", 0.0) or 0.0) < 7.0 and "spectrum_optimizer_xapp" in selected_categories:
            state_action_bonus += 0.08
        if float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0) >= 3.0 and "mobility_predictor_xapp" in selected_categories:
            state_action_bonus += 0.08
        return {
            "meta_target_coverage": float(meta_coverage),
            "controller_service_coverage": float(service_coverage),
            "state_action_bonus": float(state_action_bonus),
        }

    def _intent_hrl_state_key(self, *, goal, observation: Observation) -> str:
        target_classes = self._intent_hrl_target_classes(goal=goal, observation=observation)
        queue_bucket = self._bucket_label(
            float(getattr(observation, "total_queue_bytes", 0.0) or 0.0),
            low=24000.0,
            high=90000.0,
        )
        bler_bucket = self._bucket_label(
            float(getattr(observation, "avg_bler", 0.0) or 0.0),
            low=0.08,
            high=0.16,
        )
        sinr_bucket = self._bucket_label(
            float(getattr(observation, "avg_sinr_db", 0.0) or 0.0),
            low=6.0,
            high=11.0,
        )
        handover_bucket = self._bucket_label(
            float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0),
            low=1.0,
            high=5.0,
        )
        return "|".join(
            [
                "targets=" + ",".join(target_classes),
                f"queue={queue_bucket}",
                f"bler={bler_bucket}",
                f"sinr={sinr_bucket}",
                f"handover={handover_bucket}",
            ]
        )

    def _intent_hrl_action_key(self, selected_xapps: list[str]) -> str:
        categories = self._xapp_categories(list(selected_xapps or []))
        return ",".join(sorted(categories))

    def _stable_token_index(self, token: str, *, vector_dim: int) -> int:
        token = str(token or "")
        if vector_dim <= 0:
            return 0
        accumulator = 0
        for index, character in enumerate(token):
            accumulator += (index + 1) * ord(character)
        return int(accumulator % vector_dim)

    def _intent_hrl_feature_vector(
        self,
        *,
        goal,
        observation: Observation,
        selected_xapps: list[str],
        vector_dim: int = 24,
    ) -> list[float]:
        state_key = self._intent_hrl_state_key(goal=goal, observation=observation)
        action_key = self._intent_hrl_action_key(selected_xapps)
        tokens = [
            token.strip()
            for token in (state_key.replace("|", ",") + "," + action_key).split(",")
            if token.strip() != ""
        ]
        features = [0.0 for _ in range(vector_dim)]
        for token in tokens:
            features[self._stable_token_index(token, vector_dim=vector_dim)] += 1.0
        features[0] += min(1.0, float(getattr(observation, "total_queue_bytes", 0.0) or 0.0) / 120000.0)
        features[1] += min(1.0, float(getattr(observation, "avg_bler", 0.0) or 0.0) / 0.25)
        features[2] += min(1.0, max(0.0, 14.0 - float(getattr(observation, "avg_sinr_db", 0.0) or 0.0)) / 14.0)
        features[3] += min(1.0, float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0) / 10.0)
        max_value = max(max(abs(value) for value in features), 1.0)
        return [float(value) / float(max_value) for value in features]

    def _intent_hrl_neural_state(self) -> dict[str, object]:
        hrl_state = self._baseline_state_section("intent_hrl_policy")
        neural_state = hrl_state.get("neural_policy", {})
        if isinstance(neural_state, dict) and neural_state.get("input_dim") == 24 and neural_state.get("hidden_dim") == 12:
            return neural_state
        input_dim = 24
        hidden_dim = 12
        weights_1 = [
            [
                0.035 * math.sin(float((hidden_index + 1) * (input_index + 3)))
                for input_index in range(input_dim)
            ]
            for hidden_index in range(hidden_dim)
        ]
        weights_2 = [
            0.040 * math.cos(float(hidden_index + 1))
            for hidden_index in range(hidden_dim)
        ]
        neural_state = {
            "policy_type": "one_hidden_layer_mlp_q_policy",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "weights_1": weights_1,
            "bias_1": [0.0 for _ in range(hidden_dim)],
            "weights_2": weights_2,
            "bias_2": 0.0,
            "update_count": 0,
        }
        hrl_state["neural_policy"] = neural_state
        return neural_state

    def _intent_hrl_neural_forward(
        self,
        *,
        neural_state: dict[str, object],
        features: list[float],
    ) -> tuple[float, list[float]]:
        weights_1 = list(neural_state.get("weights_1", []) or [])
        bias_1 = list(neural_state.get("bias_1", []) or [])
        weights_2 = list(neural_state.get("weights_2", []) or [])
        bias_2 = float(neural_state.get("bias_2", 0.0) or 0.0)
        hidden_values: list[float] = []
        for hidden_index, hidden_weights in enumerate(weights_1):
            activation = float(bias_1[hidden_index] if hidden_index < len(bias_1) else 0.0)
            for input_index, feature_value in enumerate(features):
                if input_index < len(hidden_weights):
                    activation += float(hidden_weights[input_index]) * float(feature_value)
            hidden_values.append(math.tanh(activation))
        q_value = bias_2
        for hidden_index, hidden_value in enumerate(hidden_values):
            if hidden_index < len(weights_2):
                q_value += float(weights_2[hidden_index]) * float(hidden_value)
        return float(max(-1.0, min(1.0, q_value))), hidden_values

    def _intent_hrl_neural_policy_value(
        self,
        *,
        goal,
        observation: Observation,
        selected_xapps: list[str],
    ) -> float:
        neural_state = self._intent_hrl_neural_state()
        features = self._intent_hrl_feature_vector(
            goal=goal,
            observation=observation,
            selected_xapps=selected_xapps,
        )
        q_value, _ = self._intent_hrl_neural_forward(
            neural_state=neural_state,
            features=features,
        )
        return float(q_value)

    def _intent_hrl_policy_value(self, *, goal, observation: Observation, selected_xapps: list[str]) -> float:
        hrl_state = self._baseline_state_section("intent_hrl_policy")
        q_table = dict(hrl_state.get("q_table", {}) or {})
        state_key = self._intent_hrl_state_key(goal=goal, observation=observation)
        action_key = self._intent_hrl_action_key(selected_xapps)
        tabular_value = float(dict(q_table.get(state_key, {}) or {}).get(action_key, 0.0) or 0.0)
        neural_value = self._intent_hrl_neural_policy_value(
            goal=goal,
            observation=observation,
            selected_xapps=selected_xapps,
        )
        return float(0.65 * neural_value + 0.35 * tabular_value)

    def _update_intent_hrl_neural_policy(
        self,
        *,
        goal,
        observation: Observation,
        candidate_score: CandidatePlanScore,
        reward_value: float,
    ) -> float:
        neural_state = self._intent_hrl_neural_state()
        features = self._intent_hrl_feature_vector(
            goal=goal,
            observation=observation,
            selected_xapps=list(candidate_score.selected_xapps),
        )
        q_value, hidden_values = self._intent_hrl_neural_forward(
            neural_state=neural_state,
            features=features,
        )
        prediction_error = float(reward_value) - float(q_value)
        learning_rate = 0.035
        weights_1 = [
            [float(value) for value in list(row or [])]
            for row in list(neural_state.get("weights_1", []) or [])
        ]
        bias_1 = [float(value) for value in list(neural_state.get("bias_1", []) or [])]
        weights_2 = [float(value) for value in list(neural_state.get("weights_2", []) or [])]
        bias_2 = float(neural_state.get("bias_2", 0.0) or 0.0)
        old_weights_2 = list(weights_2)
        for hidden_index, hidden_value in enumerate(hidden_values):
            if hidden_index < len(weights_2):
                weights_2[hidden_index] += learning_rate * prediction_error * float(hidden_value)
        bias_2 += learning_rate * prediction_error
        for hidden_index, hidden_value in enumerate(hidden_values):
            if hidden_index >= len(weights_1):
                continue
            gradient_hidden = (
                prediction_error
                * float(old_weights_2[hidden_index] if hidden_index < len(old_weights_2) else 0.0)
                * (1.0 - float(hidden_value) ** 2)
            )
            for input_index, feature_value in enumerate(features):
                if input_index < len(weights_1[hidden_index]):
                    weights_1[hidden_index][input_index] += (
                        learning_rate * gradient_hidden * float(feature_value)
                    )
            if hidden_index < len(bias_1):
                bias_1[hidden_index] += learning_rate * gradient_hidden
        neural_state["weights_1"] = weights_1
        neural_state["bias_1"] = bias_1
        neural_state["weights_2"] = weights_2
        neural_state["bias_2"] = float(bias_2)
        neural_state["update_count"] = int(neural_state.get("update_count", 0) or 0) + 1
        return float(0.5 * prediction_error * prediction_error)

    def _update_intent_hrl_policy(
        self,
        *,
        goal,
        observation: Observation,
        candidate_score: CandidatePlanScore,
    ) -> None:
        if self.selection_mode != "intent_hrl" or not self._baseline_training_active("intent_hrl"):
            return
        hrl_state = self._baseline_state_section("intent_hrl_policy")
        q_table = dict(hrl_state.get("q_table", {}) or {})
        state_key = self._intent_hrl_state_key(goal=goal, observation=observation)
        action_key = self._intent_hrl_action_key(list(candidate_score.selected_xapps))
        state_row = dict(q_table.get(state_key, {}) or {})
        old_value = float(state_row.get(action_key, 0.0) or 0.0)
        reward_value = max(-1.0, min(1.0, 0.45 - float(candidate_score.total_score)))
        learning_rate = 0.28
        td_error = float(reward_value - old_value)
        new_value = old_value + learning_rate * td_error
        neural_loss = self._update_intent_hrl_neural_policy(
            goal=goal,
            observation=observation,
            candidate_score=candidate_score,
            reward_value=reward_value,
        )
        state_row[action_key] = float(new_value)
        q_table[state_key] = state_row
        hrl_state["q_table"] = q_table
        history = list(hrl_state.get("training_history", []) or [])
        history.append(
            {
                "phase": str(self.experiment_transfer_phase),
                "repeat": int(self.experiment_phase_repeat),
                "state_key": state_key,
                "action_key": action_key,
                "reward": float(reward_value),
                "q_value": float(new_value),
                "td_error": float(td_error),
                "neural_loss": float(neural_loss),
                "selected_score": float(candidate_score.total_score),
            }
        )
        recent_history = list(history[-20:])
        mean_recent_reward = sum(float(item.get("reward", 0.0) or 0.0) for item in recent_history) / float(
            max(1, len(recent_history))
        )
        mean_recent_abs_td_error = sum(
            abs(float(item.get("td_error", 0.0) or 0.0))
            for item in recent_history
        ) / float(max(1, len(recent_history)))
        convergence_curve = list(hrl_state.get("convergence_curve", []) or [])
        convergence_curve.append(
            {
                "update_index": int(len(history)),
                "mean_recent_reward": float(mean_recent_reward),
                "mean_recent_abs_td_error": float(mean_recent_abs_td_error),
                "latest_neural_loss": float(neural_loss),
                "state_count": int(len(q_table)),
            }
        )
        hrl_state["training_history"] = history[-256:]
        hrl_state["convergence_curve"] = convergence_curve[-256:]
        hrl_state["policy_type"] = "mlp_q_policy_with_tabular_stabilizer"
        hrl_state["frozen_on_transfer"] = bool(self.intent_hrl_freeze_on_transfer)
        self._persist_literature_baseline_state()

    def _oreo_service_graph_terms(
        self,
        *,
        candidate_score: CandidatePlanScore,
        requested_categories: list[str],
        goal,
        observation: Observation,
    ) -> dict[str, float]:
        demands = self._service_demands(goal=goal, observation=observation)
        requirement_map = self._service_demand_category_requirements()
        selected_categories = set(self._xapp_categories(list(candidate_score.selected_xapps)))
        admitted_services = 0
        total_services = 0
        dependency_mismatch = 0.0
        for demand_name in demands:
            required = set(requirement_map.get(str(demand_name), []))
            if len(required) == 0:
                continue
            total_services += 1
            coverage = float(len(selected_categories & required)) / float(len(required))
            if coverage >= 0.5:
                admitted_services += 1
            else:
                dependency_mismatch += 1.0 - coverage
        service_admission_ratio = float(admitted_services) / float(max(1, total_services))

        semantic_groups = self._oreo_semantic_groups()
        shared_group_count = sum(
            1
            for group_members in semantic_groups.values()
            if len(selected_categories & group_members) > 0
        )
        sharing_gain = 0.0
        if len(selected_categories) > 0:
            sharing_gain = float(shared_group_count) / float(len(selected_categories))
        requested_set = set(self._xapp_categories(list(requested_categories)) or list(requested_categories))
        interface_mismatch = 0.0
        if len(requested_set) > 0:
            interface_mismatch = 1.0 - float(len(selected_categories & requested_set)) / float(len(requested_set))
        deployment_cost = sum(
            self._category_resource_unit_cost(category_name)
            for category_name in selected_categories
        )
        return {
            "service_admission_ratio": float(service_admission_ratio),
            "sharing_gain": float(sharing_gain),
            "dependency_mismatch": float(dependency_mismatch),
            "interface_mismatch": float(max(0.0, interface_mismatch)),
            "deployment_cost": float(deployment_cost),
        }

    def _score_for_selection_mode(
        self,
        *,
        candidate_score: CandidatePlanScore,
        requested_categories: list[str],
        goal=None,
        observation: Observation | None = None,
    ) -> CandidatePlanScore:
        selected_categories = set(self._xapp_categories(list(candidate_score.selected_xapps)))
        requested_category_set = set(
            self._xapp_categories(list(requested_categories)) or list(requested_categories)
        )
        if len(requested_category_set) == 0:
            requested_category_set = selected_categories
        service_coverage = 0.0
        if len(requested_category_set) > 0:
            service_coverage = float(len(selected_categories & requested_category_set)) / float(
                len(requested_category_set)
            )
        timing_cost = (
            0.04 * float(len(list(candidate_score.selected_xapps)))
            + 0.015 * float(candidate_score.parameter_bound_count)
        )
        mode = self.selection_mode
        if mode == "resource_aware":
            location_terms = (
                self._orchestran_location_terms(
                    candidate_score=candidate_score,
                    goal=goal,
                    observation=observation,
                )
                if goal is not None and observation is not None
                else {
                    "placement": {},
                    "execution_time": timing_cost,
                    "deadline_budget": 1.0,
                    "timing_violation": 0.0,
                    "lifecycle_cost": 0.0,
                }
            )
            candidate_score.score_version = "j_sel_orchestran_like_resource_timing_objective"
            # OrchestRAN-style network-intelligence orchestration: choose
            # executable intelligence functions and placements under resource,
            # lifecycle, and timing constraints. It intentionally does not use
            # reflective memory or failed-case repair knowledge.
            candidate_score.total_score = float(
                1.00 * candidate_score.predicted_intent_loss
                + 0.42 * candidate_score.resource_pressure
                + 0.55 * float(location_terms.get("timing_violation", 0.0) or 0.0)
                + 0.24 * float(location_terms.get("lifecycle_cost", 0.0) or 0.0)
                + 0.16 * timing_cost
                + 0.18 * candidate_score.switching_cost
            )
            candidate_score.rationale += (
                f", baseline_objective=OrchestRAN-like, placement={location_terms.get('placement', {})}, "
                f"execution_time={float(location_terms.get('execution_time', 0.0) or 0.0):.3f}, "
                f"deadline={float(location_terms.get('deadline_budget', 0.0) or 0.0):.3f}, "
                f"timing_violation={float(location_terms.get('timing_violation', 0.0) or 0.0):.3f}, "
                "conflict_memory=disabled"
            )
        elif mode == "intent_hrl":
            hrl_terms = (
                self._intent_hrl_terms(
                    candidate_score=candidate_score,
                    requested_categories=requested_categories,
                    goal=goal,
                    observation=observation,
                )
                if goal is not None and observation is not None
                else {
                    "meta_target_coverage": service_coverage,
                    "controller_service_coverage": service_coverage,
                    "state_action_bonus": 0.0,
                }
            )
            policy_value = (
                self._intent_hrl_policy_value(
                    goal=goal,
                    observation=observation,
                    selected_xapps=list(candidate_score.selected_xapps),
                )
                if goal is not None and observation is not None
                else 0.0
            )
            candidate_score.score_version = "j_sel_intent_hrl_like_reward_objective"
            # Intent-HRL-style reward: a meta-controller target coverage term
            # and a low-level controller service-coverage term are combined
            # with conflict/resource/switching costs. No success/failure memory
            # is available to the policy. Warm-up updates a persistent policy
            # table; transfer uses the frozen table as a deterministic policy.
            candidate_score.total_score = float(
                1.00 * candidate_score.predicted_intent_loss
                + 0.32 * candidate_score.predicted_conflict_cost
                + 0.24 * candidate_score.resource_pressure
                + 0.22 * candidate_score.switching_cost
                + 0.18 * candidate_score.stability_risk
                - 0.20 * float(hrl_terms.get("meta_target_coverage", 0.0) or 0.0)
                - 0.16 * float(hrl_terms.get("controller_service_coverage", 0.0) or 0.0)
                - 0.10 * float(hrl_terms.get("state_action_bonus", 0.0) or 0.0)
                - 0.24 * float(policy_value)
            )
            candidate_score.rationale += (
                f", baseline_objective=Intent-HRL-like, "
                f"meta_target_coverage={float(hrl_terms.get('meta_target_coverage', 0.0) or 0.0):.3f}, "
                f"controller_service_coverage={float(hrl_terms.get('controller_service_coverage', 0.0) or 0.0):.3f}, "
                f"state_action_bonus={float(hrl_terms.get('state_action_bonus', 0.0) or 0.0):.3f}, "
                f"frozen_policy_q={float(policy_value):.3f}, "
                f"phase={self.experiment_transfer_phase or 'standalone'}"
            )
        elif mode == "service_graph":
            graph_terms = (
                self._oreo_service_graph_terms(
                    candidate_score=candidate_score,
                    requested_categories=requested_categories,
                    goal=goal,
                    observation=observation,
                )
                if goal is not None and observation is not None
                else {
                    "service_admission_ratio": service_coverage,
                    "sharing_gain": 0.0,
                    "dependency_mismatch": 0.0,
                    "interface_mismatch": 0.0,
                    "deployment_cost": 0.0,
                }
            )
            candidate_score.score_version = "j_sel_oreo_like_service_graph_objective"
            # OREO-style service graph objective: maximize service admission
            # and semantic xApp sharing while accounting for resource,
            # dependency, and interface mismatch costs. It is not an
            # outcome-memory or failed-case-avoidance method.
            candidate_score.total_score = float(
                -0.88 * float(graph_terms.get("service_admission_ratio", 0.0) or 0.0)
                -0.22 * float(graph_terms.get("sharing_gain", 0.0) or 0.0)
                + 0.44 * candidate_score.resource_pressure
                + 0.14 * timing_cost
                + 0.28 * float(graph_terms.get("dependency_mismatch", 0.0) or 0.0)
                + 0.16 * float(graph_terms.get("interface_mismatch", 0.0) or 0.0)
                + 0.12 * float(graph_terms.get("deployment_cost", 0.0) or 0.0)
                + 0.08 * candidate_score.switching_cost
            )
            candidate_score.rationale += (
                f", baseline_objective=OREO-like, "
                f"service_admission={float(graph_terms.get('service_admission_ratio', 0.0) or 0.0):.3f}, "
                f"sharing_gain={float(graph_terms.get('sharing_gain', 0.0) or 0.0):.3f}, "
                f"dependency_mismatch={float(graph_terms.get('dependency_mismatch', 0.0) or 0.0):.3f}, "
                f"interface_mismatch={float(graph_terms.get('interface_mismatch', 0.0) or 0.0):.3f}, "
                f"deployment_cost={float(graph_terms.get('deployment_cost', 0.0) or 0.0):.3f}"
            )
        elif mode == "no_memory_planner":
            candidate_score.score_version = "j_sel_no_memory_registry_conflict_objective"
            candidate_score.rationale += ", baseline_objective=no-memory planner"
        elif mode == "fixed_plan":
            candidate_score.score_version = "j_sel_fixed_plan_objective"
            candidate_score.rationale += ", baseline_objective=fixed plan"
        return candidate_score

    def _select_best_candidate_plan(
        self,
        *,
        candidate_requested_names: list[list[str]],
        goal,
        observation: Observation,
        reflection: ReflectionNote | None,
        base_categories: list[str],
        previous_selected_xapps: list[str],
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> tuple[OrchestrationCandidatePlan | None, list[CandidatePlanScore], list[OrchestrationCandidatePlan]]:
        best_candidate_plan: OrchestrationCandidatePlan | None = None
        best_score = float("inf")
        scored_candidates: list[CandidatePlanScore] = []
        candidate_plan_portfolios = self._candidate_plan_portfolios(
            candidate_requested_names=candidate_requested_names,
            goal=goal,
            observation=observation,
            reflection=reflection,
            base_categories=base_categories,
        )
        for candidate_plan in list(candidate_plan_portfolios or []):
            candidate_score = self._portfolio_score_breakdown(
                candidate_plan=candidate_plan,
                requested_categories=list(getattr(candidate_plan, "requested_xapps", []) or []),
                base_categories=base_categories,
                previous_selected_xapps=previous_selected_xapps,
                retrieved_success_cases=retrieved_success_cases,
                retrieved_failure_cases=retrieved_failure_cases,
                goal=goal,
                observation=observation,
            )
            scored_candidates.append(candidate_score)
            if float(candidate_score.total_score) < best_score:
                best_score = float(candidate_score.total_score)
                best_candidate_plan = candidate_plan
        best_candidate_id = str(getattr(best_candidate_plan, "candidate_id", "") or "")
        for candidate_score in scored_candidates:
            if str(getattr(candidate_score, "candidate_id", "") or "") == best_candidate_id:
                candidate_score.selected = True
                self._update_intent_hrl_policy(
                    goal=goal,
                    observation=observation,
                    candidate_score=candidate_score,
                )
        scored_candidates.sort(key=lambda item: float(item.total_score))
        return best_candidate_plan, scored_candidates, candidate_plan_portfolios

    def _reflective_pareto_guard_portfolio(
        self,
        *,
        goal,
        observation: Observation,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> list[str]:
        """Context-critical portfolio gate for full reflective orchestration."""

        if not self._reflective_transfer_guard_active():
            return []
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
        ).lower()
        if not any(token in goal_text for token in ("mobility", "handover", "ping-pong", "ping pong")):
            return []
        avoided_categories = attributed_failure_category_set(retrieved_failure_cases)
        handover_pending = int(getattr(observation, "handover_pending_ue_count", 0) or 0)
        throughput_mbps = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)

        if handover_pending >= 3:
            guard_portfolio = [
                "urllc_guard_xapp",
                "latency_scheduler_xapp",
                "mobility_predictor_xapp",
                "traffic_steering_xapp",
            ]
        elif throughput_mbps < 8.0:
            guard_portfolio = [
                "traffic_steering_xapp",
                "massive_mimo_beamformer_xapp",
                "urllc_guard_xapp",
                "latency_scheduler_xapp",
            ]
        else:
            guard_portfolio = [
                "massive_mimo_beamformer_xapp",
                "spectrum_optimizer_xapp",
                "urllc_guard_xapp",
                "latency_scheduler_xapp",
            ]
        return [
            category_name
            for category_name in guard_portfolio
            if category_name not in avoided_categories
        ][: self.max_selected_xapps]

    def _repair_reflective_critical_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> list[str]:
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
        out_of_sync = float(getattr(observation, "out_of_sync_ue_count", 0.0) or 0.0)
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        if handover_pending < 5.0 and out_of_sync < 3.0 and avg_bler < 0.12:
            return list(selected_names)
        guard_categories = self._reflective_pareto_guard_portfolio(
            goal=goal,
            observation=observation,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        if len(guard_categories) == 0:
            return list(selected_names)
        selected_categories = self._xapp_categories(selected_names)
        repaired_categories = list(selected_categories)
        for category_name in guard_categories:
            if category_name not in repaired_categories:
                repaired_categories.append(category_name)
        if len(repaired_categories) > self.max_selected_xapps:
            protected = set(guard_categories)
            trimmed: list[str] = []
            for category_name in repaired_categories:
                if category_name in protected or len(trimmed) < self.max_selected_xapps:
                    trimmed.append(category_name)
            repaired_categories = list(dict.fromkeys(trimmed))[: self.max_selected_xapps]
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=repaired_categories,
            goal=goal,
            reflection=None,
        )
        return repaired_names or list(selected_names)

    def _adapt_reflective_health_gate(
        self,
        *,
        candidate_plan: OrchestrationCandidatePlan,
        goal,
        observation: Observation,
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
    ) -> None:
        if not self._reflective_transfer_guard_active():
            return
        memory_evidence_count = len(retrieved_success_cases or []) + len(retrieved_failure_cases or [])
        if memory_evidence_count <= 0:
            return
        inter_cell_penalty = float(
            getattr(observation, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
        )
        inter_cell_sources = float(
            getattr(observation, "inter_cell_interference_source_count_mean", 0.0) or 0.0
        )
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
        if (
            inter_cell_penalty < 6.0
            and inter_cell_sources < 6.0
            and avg_bler < 0.18
            and handover_pending < 8.0
        ):
            return
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        if "urllc" in goal_text or "latency" in goal_text or "packet_loss" in goal_text:
            return
        current_gate = getattr(candidate_plan, "exploration_health_gate", None) or default_exploration_health_gate(goal)
        candidate_plan.exploration_health_gate = ExplorationHealthGate(
            max_avg_bler=max(0.16, float(getattr(current_gate, "max_avg_bler", 0.0) or 0.0)),
            max_total_queue_bytes=getattr(current_gate, "max_total_queue_bytes", None),
            min_total_throughput_mbps=getattr(current_gate, "min_total_throughput_mbps", None),
            max_handover_pending_ue_count=max(
                96,
                int(getattr(current_gate, "max_handover_pending_ue_count", 0) or 0),
            ),
            freeze_slots_on_violation=min(
                2,
                int(getattr(current_gate, "freeze_slots_on_violation", 4) or 4),
            ),
        )

    def _repair_reflective_mobility_intent_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        if not any(token in goal_text for token in ("mobility", "handover", "ping-pong", "ping pong")):
            return list(selected_names)
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        inter_cell_penalty = float(
            getattr(observation, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
        )
        inter_cell_sources = float(
            getattr(observation, "inter_cell_interference_source_count_mean", 0.0) or 0.0
        )
        selected_categories = set(self._xapp_categories(selected_names))
        radio_or_capacity_pressure = bool(
            {"massive_mimo_beamformer_xapp", "spectrum_optimizer_xapp"} & selected_categories
        )
        qos_guard_missing = not (
            {"latency_scheduler_xapp", "urllc_guard_xapp"} <= selected_categories
        )
        transfer_risk = (
            handover_pending >= 2.0
            or avg_bler >= 0.08
            or inter_cell_penalty >= 2.0
            or inter_cell_sources >= 4.0
            or self.experiment_transfer_phase == "transfer"
        )
        if not (radio_or_capacity_pressure and qos_guard_missing and transfer_risk):
            return list(selected_names)
        repaired_categories = [
            "traffic_steering_xapp",
            "latency_scheduler_xapp",
            "urllc_guard_xapp",
        ]
        if (
            handover_pending >= 8.0
            or avg_bler >= 0.22
            or inter_cell_penalty >= 6.0
            or inter_cell_sources >= 6.0
        ):
            repaired_categories.insert(1, "mobility_predictor_xapp")
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=repaired_categories,
            goal=goal,
            reflection=None,
        )
        return repaired_names[: self.max_selected_xapps] or list(selected_names)

    def _repair_reflective_reliability_transfer_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        reliability_transfer_goal = (
            ("avg_bler" in goal_text or "bler" in goal_text or "urllc" in goal_text)
            and ("sinr" in goal_text or "interference" in goal_text or "edge" in goal_text)
        )
        if not reliability_transfer_goal:
            return list(selected_names)
        selected_categories = list(self._xapp_categories(selected_names))
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        total_throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        inter_cell_penalty = float(
            getattr(observation, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
        )
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
        selected_category_set = set(selected_categories)
        required_categories = {
            "urllc_guard_xapp",
            "latency_scheduler_xapp",
            "traffic_steering_xapp",
        }
        if (
            handover_pending >= 2.0
            or self.experiment_transfer_phase == "transfer"
            or "mobility" in goal_text
            or "handover" in goal_text
        ):
            required_categories.add("mobility_predictor_xapp")
        if required_categories <= selected_category_set:
            return list(selected_names)
        weak_reliability_transfer = (
            avg_bler >= 0.08
            or total_throughput < 6.0
            or inter_cell_penalty >= 3.0
            or handover_pending >= 2.0
            or self.experiment_transfer_phase == "transfer"
        )
        if not weak_reliability_transfer:
            return list(selected_names)
        repaired_categories = [
            "urllc_guard_xapp",
            "latency_scheduler_xapp",
            "mobility_predictor_xapp",
            "traffic_steering_xapp",
        ]
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=repaired_categories,
            goal=goal,
            reflection=None,
        )
        return repaired_names[: len(repaired_categories)] or list(selected_names)

    def _repair_reflective_handover_video_transfer_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        handover_video_goal = (
            ("handover" in goal_text or "mobility" in goal_text or "moving" in goal_text)
            and ("video" in goal_text or "edge" in goal_text or "throughput" in goal_text)
            and ("bler" in goal_text or "reliability" in goal_text or "risk" in goal_text)
        )
        if not handover_video_goal:
            return list(selected_names)
        selected_name_set = {str(name) for name in list(selected_names or [])}
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        inter_cell_penalty = float(
            getattr(observation, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
        )
        transfer_or_stressed = (
            self.experiment_transfer_phase == "transfer"
            or handover_pending >= 2.0
            or avg_bler >= 0.08
            or throughput < 8.0
            or inter_cell_penalty >= 3.0
        )
        preferred_names = [
            "mobility_predictor_xapp_hysteresis_guard",
            "traffic_steering_xapp_balanced",
            "latency_scheduler_xapp_balanced",
            "urllc_guard_xapp_balanced",
        ]
        if set(preferred_names) <= selected_name_set:
            return list(selected_names)
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=preferred_names,
            goal=goal,
            reflection=None,
        )
        return repaired_names[: len(preferred_names)] or list(selected_names)

    def _repair_reflective_interference_urllc_coexistence_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        coexistence_goal = (
            ("interference" in goal_text or "sinr" in goal_text)
            and ("urllc" in goal_text or "bler" in goal_text or "latency" in goal_text)
        )
        if not coexistence_goal:
            return list(selected_names)
        selected_name_set = {str(name) for name in list(selected_names or [])}
        preferred_names = [
            "spectrum_optimizer_xapp_interference_averse",
            "massive_mimo_beamformer_xapp_sinr_max",
            "latency_scheduler_xapp_balanced",
            "urllc_guard_xapp_balanced",
        ]
        if set(preferred_names) <= selected_name_set:
            return list(selected_names)
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=preferred_names,
            goal=goal,
            reflection=None,
        )
        return repaired_names[: len(preferred_names)] or list(selected_names)

    def _repair_reflective_hotspot_slice_fairness_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        slice_hotspot_goal = (
            ("slice" in goal_text or "fair" in goal_text or "tenant" in goal_text or "sla" in goal_text)
            and ("queue" in goal_text or "total_queue_bytes" in goal_text or "hotspot" in goal_text)
        )
        if not slice_hotspot_goal:
            return list(selected_names)
        selected_categories = set(self._xapp_categories(selected_names))
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        queue_bytes = float(getattr(observation, "total_queue_bytes", 0.0) or 0.0)
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
        transfer_or_stressed = (
            self.experiment_transfer_phase == "transfer"
            or avg_bler >= 0.08
            or queue_bytes >= 60000.0
            or handover_pending >= 2.0
        )
        required_categories = {
            "ran_slicing_manager_xapp",
            "traffic_steering_xapp",
            "mobility_predictor_xapp",
            "urllc_guard_xapp",
        }
        if required_categories <= selected_categories:
            return list(selected_names)
        if not transfer_or_stressed:
            return list(selected_names)
        repaired_categories = [
            "ran_slicing_manager_xapp",
            "traffic_steering_xapp",
            "mobility_predictor_xapp",
            "urllc_guard_xapp",
        ]
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=repaired_categories,
            goal=goal,
            reflection=None,
        )
        return repaired_names[: len(repaired_categories)] or list(selected_names)

    def _repair_reflective_blockage_beam_recovery_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        blockage_recovery_goal = (
            ("blockage" in goal_text or "fading" in goal_text or "beam" in goal_text)
            and ("sinr" in goal_text or "throughput" in goal_text)
            and ("bler" in goal_text or "reliability" in goal_text)
        )
        if not blockage_recovery_goal:
            return list(selected_names)
        selected_categories = set(self._xapp_categories(selected_names))
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        avg_sinr = float(getattr(observation, "avg_sinr_db", 0.0) or 0.0)
        inter_cell_penalty = float(
            getattr(observation, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
        )
        transfer_or_stressed = (
            self.experiment_transfer_phase == "transfer"
            or avg_bler >= 0.08
            or avg_sinr < 10.0
            or inter_cell_penalty >= 2.0
        )
        required_categories = {
            "massive_mimo_beamformer_xapp",
            "spectrum_optimizer_xapp",
            "latency_scheduler_xapp",
            "urllc_guard_xapp",
        }
        if required_categories <= selected_categories:
            return list(selected_names)
        if not transfer_or_stressed:
            return list(selected_names)
        repaired_categories = [
            "massive_mimo_beamformer_xapp",
            "spectrum_optimizer_xapp",
            "latency_scheduler_xapp",
            "urllc_guard_xapp",
        ]
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=repaired_categories,
            goal=goal,
            reflection=None,
        )
        return repaired_names[: len(repaired_categories)] or list(selected_names)

    def _repair_reflective_energy_urllc_shift_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        energy_safety_goal = (
            ("energy" in goal_text or "low-load" in goal_text or "sleep" in goal_text)
            and ("bler" in goal_text or "queue" in goal_text or "latency" in goal_text or "urllc" in goal_text)
        )
        if not energy_safety_goal:
            return list(selected_names)
        selected_categories = set(self._xapp_categories(selected_names))
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        queue_bytes = float(getattr(observation, "total_queue_bytes", 0.0) or 0.0)
        transfer_or_stressed = (
            self.experiment_transfer_phase == "transfer"
            or avg_bler >= 0.06
            or queue_bytes >= 50000.0
        )
        required_categories = {
            "latency_scheduler_xapp",
            "urllc_guard_xapp",
            "traffic_steering_xapp",
            "mobility_predictor_xapp",
        }
        unsafe_energy_categories = {"cell_sleep_xapp"}
        if required_categories <= selected_categories and not (selected_categories & unsafe_energy_categories):
            return list(selected_names)
        if not transfer_or_stressed:
            return list(selected_names)
        repaired_categories = [
            "latency_scheduler_xapp",
            "urllc_guard_xapp",
            "traffic_steering_xapp",
            "mobility_predictor_xapp",
        ]
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=repaired_categories,
            goal=goal,
            reflection=None,
        )
        return repaired_names[: len(repaired_categories)] or list(selected_names)

    def _repair_reflective_premium_tenant_urllc_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        premium_urllc_goal = (
            ("premium" in goal_text or "tenant" in goal_text or "sla" in goal_text)
            and ("urllc" in goal_text or "bler" in goal_text or "queue" in goal_text or "latency" in goal_text)
        )
        if not premium_urllc_goal:
            return list(selected_names)
        selected_categories = set(self._xapp_categories(selected_names))
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        queue_bytes = float(getattr(observation, "total_queue_bytes", 0.0) or 0.0)
        transfer_or_stressed = (
            self.experiment_transfer_phase == "transfer"
            or avg_bler >= 0.05
            or queue_bytes >= 40000.0
        )
        required_categories = {
            "ran_slicing_manager_xapp",
            "latency_scheduler_xapp",
            "urllc_guard_xapp",
            "spectrum_optimizer_xapp",
        }
        unsafe_categories = {"cell_sleep_xapp"}
        if required_categories <= selected_categories and not (selected_categories & unsafe_categories):
            return list(selected_names)
        if not transfer_or_stressed:
            return list(selected_names)
        repaired_categories = [
            "ran_slicing_manager_xapp",
            "latency_scheduler_xapp",
            "urllc_guard_xapp",
            "spectrum_optimizer_xapp",
        ]
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=repaired_categories,
            goal=goal,
            reflection=None,
        )
        return repaired_names[: len(repaired_categories)] or list(selected_names)

    def _repair_reflective_load_balance_stability_coverage(
        self,
        *,
        selected_names: list[str],
        goal,
        observation: Observation,
    ) -> list[str]:
        if not self._reflective_transfer_guard_active():
            return list(selected_names)
        goal_text = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "raw_intent_text", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
            + " "
            + " ".join(self._goal_kpi_names(goal))
        ).lower()
        load_balance_goal = (
            "load balance" in goal_text
            or "load-balance" in goal_text
            or "multi-cell" in goal_text
            or "neighboring cell" in goal_text
            or "oscillation" in goal_text
        )
        if not load_balance_goal:
            return list(selected_names)
        selected_categories = set(self._xapp_categories(selected_names))
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        queue_bytes = float(getattr(observation, "total_queue_bytes", 0.0) or 0.0)
        handover_pending = float(getattr(observation, "handover_pending_ue_count", 0.0) or 0.0)
        inter_cell_penalty = float(
            getattr(observation, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
        )
        transfer_or_stressed = (
            self.experiment_transfer_phase == "transfer"
            or avg_bler >= 0.07
            or queue_bytes >= 50000.0
            or handover_pending >= 2.0
            or inter_cell_penalty >= 2.0
        )
        required_categories = {
            "mobility_predictor_xapp",
            "traffic_steering_xapp",
            "latency_scheduler_xapp",
            "urllc_guard_xapp",
        }
        if required_categories <= selected_categories:
            return list(selected_names)
        if not transfer_or_stressed:
            return list(selected_names)
        repaired_names = self._resolve_selected_xapp_names(
            requested_names=[
                "mobility_predictor_xapp_hysteresis_guard",
                "traffic_steering_xapp_balanced",
                "latency_scheduler_xapp_balanced",
                "urllc_guard_xapp_balanced",
            ],
            goal=goal,
            reflection=None,
        )
        return repaired_names[: len(required_categories)] or list(selected_names)

    def _heuristic_selection(
        self,
        *,
        goal,
        observation: Observation,
        reflection: ReflectionNote | None,
        retrieved_success_cases: list[OrchestrationMemoryCase] | None = None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None = None,
        previous_selected_xapps: list[str] | None = None,
        generated_by: str,
    ) -> XAppSelectionPlan:
        previous_selected_xapps = list(previous_selected_xapps or [])
        selection_query_key = self._selection_query_key(
            goal=goal,
            observation=observation,
            previous_selected_xapps=previous_selected_xapps,
        )
        kpi_names = {
            str(getattr(target, "kpi_name", "") or "").strip().lower()
            for target in list(getattr(goal, "kpi_targets", []) or [])
        }
        requested_categories = self._requested_categories(goal=goal, observation=observation)
        if self.selection_mode == "fixed_plan":
            selected_names = self._resolve_selected_xapp_names(
                requested_names=self._fixed_plan_categories(),
                goal=goal,
                reflection=reflection,
            )[: self.max_selected_xapps]
            candidate_score = CandidatePlanScore(
                candidate_id="candidate:fixed_plan:traffic_steering_xapp:latency_scheduler_xapp",
                score_version="j_sel_fixed_plan_objective",
                score_direction="minimize",
                predictor_name="fixed_plan_baseline_v1",
                requested_categories=self._fixed_plan_categories(),
                selected_xapps=list(selected_names),
                predicted_intent_loss=1.0,
                switching_cost=0.0,
                resource_pressure=0.0,
                total_score=1.0,
                selected=True,
                rationale="Fixed Plan baseline always uses load steering plus QoS scheduling.",
            )
            candidate_scores = [
                XAppCandidateScore(
                    xapp_name=xapp_name,
                    rationale="Fixed Plan baseline keeps this xApp active for every scenario.",
                    expected_kpi_impact={
                        str(kpi_name): 0.50 for kpi_name in sorted(kpi_names)[:3]
                    },
                    confidence=0.50,
                    selected=True,
                )
                for xapp_name in selected_names
            ]
            return XAppSelectionPlan(
                plan_id=f"selection:{goal.goal_id}:g1",
                goal_id=str(goal.goal_id),
                selected_xapps=selected_names,
                selection_query_key=selection_query_key,
                candidate_plan_projections=[],
                candidate_plan_scores=[candidate_score],
                candidate_scores=candidate_scores,
                eta=None,
                chi=None,
                hard_safety_envelope=default_hard_safety_envelope(selected_names, goal),
                forbidden_combinations=default_forbidden_combinations(selected_names),
                exploration_health_gate=default_exploration_health_gate(goal),
                orchestration_mode="fixed_plan",
                status="active",
                selection_locked=True,
                selection_generation=1,
                reflection=None,
                generated_by=f"{generated_by}_fixed_plan",
            )
        elif self.selection_mode == "intent_hrl":
            requested_categories = self._intent_hrl_high_level_categories(
                goal=goal,
                observation=observation,
                requested_categories=requested_categories,
            )
            retrieved_success_cases = []
            retrieved_failure_cases = []
        elif self.selection_mode == "service_graph":
            requested_categories = self._service_graph_categories(
                goal=goal,
                observation=observation,
                requested_categories=requested_categories,
            )
            retrieved_success_cases = []
            retrieved_failure_cases = []
        elif self.selection_mode == "resource_aware":
            requested_categories = self._orchestran_ni_catalog_categories(
                goal=goal,
                observation=observation,
                requested_categories=requested_categories,
            )
            retrieved_success_cases = []
            retrieved_failure_cases = []
        success_categories: list[str] = []
        for memory_case in list(retrieved_success_cases or []):
            success_categories.extend(self._xapp_categories(list(getattr(memory_case, "selected_xapps", []) or [])))
        failure_categories: list[str] = []
        for memory_case in list(retrieved_failure_cases or []):
            failure_categories.extend(self._xapp_categories(list(getattr(memory_case, "selected_xapps", []) or [])))
        preferred_categories = list(dict.fromkeys(requested_categories + success_categories))
        if len(preferred_categories) == 0:
            preferred_categories = list(dict.fromkeys(requested_categories))
        failure_categories_to_avoid = list(dict.fromkeys(failure_categories))
        if self.selection_mode == "reflective" and self.memory_mode == "full":
            failure_categories_to_avoid = list(
                dict.fromkeys(attributed_failure_category_set(retrieved_failure_cases))
            )
        if len(failure_categories_to_avoid) > 0:
            if self.selection_mode == "reflective" and self.memory_mode == "full":
                preferred_categories = [
                    category_name
                    for category_name in preferred_categories
                    if category_name not in failure_categories_to_avoid
                ]
                if len(preferred_categories) == 0:
                    preferred_categories = [
                        "massive_mimo_beamformer_xapp",
                        "spectrum_optimizer_xapp",
                        "urllc_guard_xapp",
                        "latency_scheduler_xapp",
                    ]
            else:
                preferred_categories = [
                    category_name
                    for category_name in preferred_categories
                    if category_name not in failure_categories_to_avoid or category_name in requested_categories
                ]
        candidate_requested_names = self._candidate_category_portfolios(
            base_categories=preferred_categories,
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
            previous_selected_xapps=previous_selected_xapps,
        )
        goal_text_for_counterfactuals = (
            str(getattr(goal, "summary", "") or "")
            + " "
            + str(getattr(goal, "intent_text", "") or "")
        ).lower()
        if (
            self._reflective_transfer_guard_active()
            and (
                "mobility" in goal_text_for_counterfactuals
                or "handover" in goal_text_for_counterfactuals
                or "ping-pong" in goal_text_for_counterfactuals
            )
        ):
            for counterfactual_portfolio in [
                [
                    "traffic_steering_xapp",
                    "latency_scheduler_xapp",
                    "urllc_guard_xapp",
                ],
                [
                    "massive_mimo_beamformer_xapp",
                    "spectrum_optimizer_xapp",
                    "latency_scheduler_xapp",
                    "urllc_guard_xapp",
                ],
                [
                    "traffic_steering_xapp",
                    "massive_mimo_beamformer_xapp",
                    "latency_scheduler_xapp",
                    "urllc_guard_xapp",
                ],
            ]:
                normalized_portfolio = list(dict.fromkeys(counterfactual_portfolio))[
                    : self.max_selected_xapps
                ]
                if normalized_portfolio not in candidate_requested_names:
                    candidate_requested_names.insert(0, normalized_portfolio)
        if self.selection_mode == "fixed_plan":
            candidate_requested_names = [list(requested_categories)]
        selected_candidate_plan, candidate_plan_scores, candidate_plan_projections = self._select_best_candidate_plan(
            candidate_requested_names=candidate_requested_names,
            goal=goal,
            observation=observation,
            reflection=reflection,
            base_categories=preferred_categories,
            previous_selected_xapps=previous_selected_xapps,
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        pareto_guard_categories = self._reflective_pareto_guard_portfolio(
            goal=goal,
            observation=observation,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        if len(pareto_guard_categories) > 0:
            pareto_guard_names = self._resolve_selected_xapp_names(
                requested_names=pareto_guard_categories,
                goal=goal,
                reflection=reflection,
            )
            if len(pareto_guard_names) > 0:
                pareto_guard_candidate = self.feasibility_projector.project_candidate(
                    candidate_id=(
                        "candidate:reflective_pareto_guard:"
                        + ":".join(self._xapp_categories(pareto_guard_names) or ["uncategorized"])
                    ),
                    requested_xapps=pareto_guard_names,
                    goal=goal,
                    observation=observation,
                    available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                    requested_eta_profile="eta:mobility_stable",
                    requested_chi_profile="chi:guardrail_first",
                )
                pareto_guard_score = self._portfolio_score_breakdown(
                    candidate_plan=pareto_guard_candidate,
                    requested_categories=list(getattr(pareto_guard_candidate, "requested_xapps", []) or []),
                    base_categories=pareto_guard_categories,
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                pareto_guard_score.score_version = "j_sel_reflective_pareto_guard_objective"
                pareto_guard_score.rationale += ", selected_by=reflective_pareto_guard"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                pareto_guard_score.selected = True
                candidate_plan_scores.insert(0, pareto_guard_score)
                candidate_plan_projections.insert(0, pareto_guard_candidate)
                selected_candidate_plan = pareto_guard_candidate
        selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        repaired_selected_names = self._repair_reflective_critical_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        if repaired_selected_names != selected_names and len(repaired_selected_names) > 0:
            repaired_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_critical_coverage:"
                    + ":".join(self._xapp_categories(repaired_selected_names) or ["uncategorized"])
                ),
                requested_xapps=repaired_selected_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(repaired_candidate_plan, "x", []) or []) > 0:
                repaired_score = self._portfolio_score_breakdown(
                    candidate_plan=repaired_candidate_plan,
                    requested_categories=list(getattr(repaired_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(repaired_selected_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                repaired_score.score_version = "j_sel_reflective_critical_coverage_objective"
                repaired_score.rationale += ", selected_by=reflective_critical_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                repaired_score.selected = True
                candidate_plan_scores.insert(0, repaired_score)
                candidate_plan_projections.insert(0, repaired_candidate_plan)
                selected_candidate_plan = repaired_candidate_plan
        selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        mobility_repaired_names = self._repair_reflective_mobility_intent_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if mobility_repaired_names != selected_names and len(mobility_repaired_names) > 0:
            mobility_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_mobility_intent_coverage:"
                    + ":".join(self._xapp_categories(mobility_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=mobility_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(mobility_candidate_plan, "x", []) or []) > 0:
                mobility_score = self._portfolio_score_breakdown(
                    candidate_plan=mobility_candidate_plan,
                    requested_categories=list(getattr(mobility_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(mobility_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                mobility_score.score_version = "j_sel_reflective_mobility_intent_coverage_objective"
                mobility_score.rationale += ", selected_by=reflective_mobility_intent_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                mobility_score.selected = True
                candidate_plan_scores.insert(0, mobility_score)
                candidate_plan_projections.insert(0, mobility_candidate_plan)
                selected_candidate_plan = mobility_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        reliability_repaired_names = self._repair_reflective_reliability_transfer_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if reliability_repaired_names != selected_names and len(reliability_repaired_names) > 0:
            reliability_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_reliability_transfer_coverage:"
                    + ":".join(self._xapp_categories(reliability_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=reliability_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(reliability_candidate_plan, "x", []) or []) > 0:
                reliability_score = self._portfolio_score_breakdown(
                    candidate_plan=reliability_candidate_plan,
                    requested_categories=list(getattr(reliability_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(reliability_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                reliability_score.score_version = "j_sel_reflective_reliability_transfer_coverage_objective"
                reliability_score.rationale += ", selected_by=reflective_reliability_transfer_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                reliability_score.selected = True
                candidate_plan_scores.insert(0, reliability_score)
                candidate_plan_projections.insert(0, reliability_candidate_plan)
                selected_candidate_plan = reliability_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        handover_video_repaired_names = self._repair_reflective_handover_video_transfer_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if handover_video_repaired_names != selected_names and len(handover_video_repaired_names) > 0:
            handover_video_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_handover_video_transfer_coverage:"
                    + ":".join(self._xapp_categories(handover_video_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=handover_video_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(handover_video_candidate_plan, "x", []) or []) > 0:
                handover_video_score = self._portfolio_score_breakdown(
                    candidate_plan=handover_video_candidate_plan,
                    requested_categories=list(getattr(handover_video_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(handover_video_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                handover_video_score.score_version = "j_sel_reflective_handover_video_transfer_coverage_objective"
                handover_video_score.rationale += ", selected_by=reflective_handover_video_transfer_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                handover_video_score.selected = True
                candidate_plan_scores.insert(0, handover_video_score)
                candidate_plan_projections.insert(0, handover_video_candidate_plan)
                selected_candidate_plan = handover_video_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        interference_urllc_repaired_names = self._repair_reflective_interference_urllc_coexistence_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if interference_urllc_repaired_names != selected_names and len(interference_urllc_repaired_names) > 0:
            interference_urllc_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_interference_urllc_coexistence_coverage:"
                    + ":".join(self._xapp_categories(interference_urllc_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=interference_urllc_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(interference_urllc_candidate_plan, "x", []) or []) > 0:
                interference_urllc_score = self._portfolio_score_breakdown(
                    candidate_plan=interference_urllc_candidate_plan,
                    requested_categories=list(getattr(interference_urllc_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(interference_urllc_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                interference_urllc_score.score_version = (
                    "j_sel_reflective_interference_urllc_coexistence_coverage_objective"
                )
                interference_urllc_score.rationale += (
                    ", selected_by=reflective_interference_urllc_coexistence_coverage_repair"
                )
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                interference_urllc_score.selected = True
                candidate_plan_scores.insert(0, interference_urllc_score)
                candidate_plan_projections.insert(0, interference_urllc_candidate_plan)
                selected_candidate_plan = interference_urllc_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        blockage_repaired_names = self._repair_reflective_blockage_beam_recovery_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if blockage_repaired_names != selected_names and len(blockage_repaired_names) > 0:
            blockage_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_blockage_beam_recovery_coverage:"
                    + ":".join(self._xapp_categories(blockage_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=blockage_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:throughput_push",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(blockage_candidate_plan, "x", []) or []) > 0:
                blockage_score = self._portfolio_score_breakdown(
                    candidate_plan=blockage_candidate_plan,
                    requested_categories=list(getattr(blockage_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(blockage_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                blockage_score.score_version = "j_sel_reflective_blockage_beam_recovery_coverage_objective"
                blockage_score.rationale += ", selected_by=reflective_blockage_beam_recovery_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                blockage_score.selected = True
                candidate_plan_scores.insert(0, blockage_score)
                candidate_plan_projections.insert(0, blockage_candidate_plan)
                selected_candidate_plan = blockage_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        energy_urllc_repaired_names = self._repair_reflective_energy_urllc_shift_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if energy_urllc_repaired_names != selected_names and len(energy_urllc_repaired_names) > 0:
            energy_urllc_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_energy_urllc_shift_coverage:"
                    + ":".join(self._xapp_categories(energy_urllc_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=energy_urllc_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(energy_urllc_candidate_plan, "x", []) or []) > 0:
                energy_urllc_score = self._portfolio_score_breakdown(
                    candidate_plan=energy_urllc_candidate_plan,
                    requested_categories=list(getattr(energy_urllc_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(energy_urllc_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                energy_urllc_score.score_version = "j_sel_reflective_energy_urllc_shift_coverage_objective"
                energy_urllc_score.rationale += ", selected_by=reflective_energy_urllc_shift_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                energy_urllc_score.selected = True
                candidate_plan_scores.insert(0, energy_urllc_score)
                candidate_plan_projections.insert(0, energy_urllc_candidate_plan)
                selected_candidate_plan = energy_urllc_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        premium_urllc_repaired_names = self._repair_reflective_premium_tenant_urllc_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if premium_urllc_repaired_names != selected_names and len(premium_urllc_repaired_names) > 0:
            premium_urllc_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_premium_tenant_urllc_coverage:"
                    + ":".join(self._xapp_categories(premium_urllc_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=premium_urllc_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(premium_urllc_candidate_plan, "x", []) or []) > 0:
                premium_urllc_score = self._portfolio_score_breakdown(
                    candidate_plan=premium_urllc_candidate_plan,
                    requested_categories=list(getattr(premium_urllc_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(premium_urllc_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                premium_urllc_score.score_version = "j_sel_reflective_premium_tenant_urllc_coverage_objective"
                premium_urllc_score.rationale += ", selected_by=reflective_premium_tenant_urllc_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                premium_urllc_score.selected = True
                candidate_plan_scores.insert(0, premium_urllc_score)
                candidate_plan_projections.insert(0, premium_urllc_candidate_plan)
                selected_candidate_plan = premium_urllc_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        load_balance_repaired_names = self._repair_reflective_load_balance_stability_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if load_balance_repaired_names != selected_names and len(load_balance_repaired_names) > 0:
            load_balance_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_load_balance_stability_coverage:"
                    + ":".join(self._xapp_categories(load_balance_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=load_balance_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(load_balance_candidate_plan, "x", []) or []) > 0:
                load_balance_score = self._portfolio_score_breakdown(
                    candidate_plan=load_balance_candidate_plan,
                    requested_categories=list(getattr(load_balance_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(load_balance_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                load_balance_score.score_version = "j_sel_reflective_load_balance_stability_coverage_objective"
                load_balance_score.rationale += ", selected_by=reflective_load_balance_stability_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                load_balance_score.selected = True
                candidate_plan_scores.insert(0, load_balance_score)
                candidate_plan_projections.insert(0, load_balance_candidate_plan)
                selected_candidate_plan = load_balance_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        hotspot_slice_repaired_names = self._repair_reflective_hotspot_slice_fairness_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if hotspot_slice_repaired_names != selected_names and len(hotspot_slice_repaired_names) > 0:
            hotspot_slice_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:reflective_hotspot_slice_fairness_coverage:"
                    + ":".join(self._xapp_categories(hotspot_slice_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=hotspot_slice_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(hotspot_slice_candidate_plan, "x", []) or []) > 0:
                hotspot_slice_score = self._portfolio_score_breakdown(
                    candidate_plan=hotspot_slice_candidate_plan,
                    requested_categories=list(getattr(hotspot_slice_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(hotspot_slice_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                hotspot_slice_score.score_version = "j_sel_reflective_hotspot_slice_fairness_coverage_objective"
                hotspot_slice_score.rationale += ", selected_by=reflective_hotspot_slice_fairness_coverage_repair"
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                hotspot_slice_score.selected = True
                candidate_plan_scores.insert(0, hotspot_slice_score)
                candidate_plan_projections.insert(0, hotspot_slice_candidate_plan)
                selected_candidate_plan = hotspot_slice_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        self._adapt_reflective_health_gate(
            candidate_plan=selected_candidate_plan,
            goal=goal,
            observation=observation,
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        if len(selected_names) == 0:
            selected_names = [
                name
                for name in [
                    "massive_mimo_beamformer_xapp_capacity_push",
                    "urllc_guard_xapp_balanced",
                    "traffic_steering_xapp_balanced",
                ]
                if name in self._xapp_catalog_by_name()
            ][: self.max_selected_xapps]
        candidate_scores: list[XAppCandidateScore] = []
        for xapp_name in selected_names:
            confidence = 0.60
            rationale = "Heuristic fallback selected a complementary xApp portfolio."
            if "urllc_guard" in xapp_name:
                confidence = 0.72 if float(getattr(observation, "avg_bler", 0.0) or 0.0) > 0.08 else 0.62
                rationale = "Reliability pressure suggests keeping a guard xApp."
            elif "beamformer" in xapp_name or "spectrum_optimizer" in xapp_name:
                confidence = 0.70
                rationale = "Throughput-oriented xApp selected for capacity uplift."
            elif "steering" in xapp_name or "latency_scheduler" in xapp_name:
                confidence = 0.66
                rationale = "Backlog pressure suggests steering or scheduler support."
            candidate_scores.append(
                XAppCandidateScore(
                    xapp_name=xapp_name,
                    rationale=rationale,
                    expected_kpi_impact={
                        str(kpi_name): float(confidence)
                        for kpi_name in sorted(kpi_names)[:3]
                    },
                    confidence=float(confidence),
                    selected=True,
                )
            )
        return XAppSelectionPlan(
            plan_id=f"selection:{goal.goal_id}:g1",
            goal_id=str(goal.goal_id),
            selected_xapps=selected_names,
            selection_query_key=selection_query_key,
            candidate_plan_projections=list(candidate_plan_projections),
            candidate_plan_scores=list(candidate_plan_scores),
            candidate_scores=candidate_scores,
            eta=getattr(selected_candidate_plan, "eta", None),
            chi=getattr(selected_candidate_plan, "chi", None),
            hard_safety_envelope=list(getattr(selected_candidate_plan, "B", []) or default_hard_safety_envelope(selected_names, goal)),
            forbidden_combinations=list(
                getattr(selected_candidate_plan, "forbidden_combinations", []) or default_forbidden_combinations(selected_names)
            ),
            exploration_health_gate=getattr(
                selected_candidate_plan,
                "exploration_health_gate",
                default_exploration_health_gate(goal),
            ),
            qos_protection_profile=getattr(selected_candidate_plan, "qos_protection_profile", None),
            feasibility_summary=getattr(selected_candidate_plan, "feasibility_summary", None),
            orchestration_mode="parallel_with_guardrail",
            status="active",
            selection_locked=True,
            selection_generation=1,
            reflection=reflection,
            generated_by=f"{generated_by}_candidate_scoring",
        )

    def _try_llm_selection(self, *, goal, observation: Observation, reflection: ReflectionNote | None):
        return self._try_llm_selection_with_memory(
            goal=goal,
            observation=observation,
            reflection=reflection,
            retrieved_success_cases=None,
            retrieved_failure_cases=None,
        )

    def _try_llm_selection_with_memory(
        self,
        *,
        goal,
        observation: Observation,
        reflection: ReflectionNote | None,
        retrieved_success_cases: list[OrchestrationMemoryCase] | None,
        retrieved_failure_cases: list[OrchestrationMemoryCase] | None,
        previous_selected_xapps: list[str] | None = None,
    ):
        previous_selected_xapps = list(previous_selected_xapps or [])
        selection_query_key = self._selection_query_key(
            goal=goal,
            observation=observation,
            previous_selected_xapps=previous_selected_xapps,
        )
        if self.selection_mode in {"fixed_plan", "resource_aware", "intent_hrl", "service_graph"}:
            return self._heuristic_selection(
                goal=goal,
                observation=observation,
                reflection=reflection,
                retrieved_success_cases=retrieved_success_cases,
                retrieved_failure_cases=retrieved_failure_cases,
                previous_selected_xapps=previous_selected_xapps,
                generated_by=f"{self.selection_mode}_baseline",
            )
        if not self.client.enabled:
            return self._heuristic_selection(
                goal=goal,
                observation=observation,
                reflection=reflection,
                retrieved_success_cases=retrieved_success_cases,
                retrieved_failure_cases=retrieved_failure_cases,
                previous_selected_xapps=previous_selected_xapps,
                generated_by="heuristic_selection_fallback",
            )
        example_selection_input = {
            "goal": {
                "goal_id": "goal:demo_throughput_balanced",
                "summary": "Increase throughput while keeping BLER stable.",
                "risk_posture": "balanced",
                "target_entities": ["cluster"],
                "kpi_targets": [
                    {"kpi_name": "total_throughput_mbps", "objective": "maximize", "target_value": 8.8},
                    {"kpi_name": "avg_bler", "objective": "minimize", "target_value": 0.18},
                ],
            },
            "reflection": {
                "outcome_summary": "Previous selection improved throughput but overloaded one scheduler path.",
                "recommended_changes": [
                    "avoid multiple scheduler-dominant xApps",
                    "keep one beam or steering xApp in the portfolio",
                ],
            },
            "observation_summary": {
                "avg_sinr_db": 10.5,
                "avg_bler": 0.21,
                "total_queue_bytes": 24000,
                "total_throughput_mbps": 7.4,
            },
        }
        example_selection_output = {
            "selected_xapps": [
                "massive_mimo_beamformer_xapp_capacity_push",
                "traffic_steering_xapp_quality_first",
                "urllc_guard_xapp_balanced",
            ],
            "orchestration_mode": "parallel_with_guardrail",
            "rationale_by_xapp": {
                "massive_mimo_beamformer_xapp_capacity_push": "Improves spectral efficiency for the throughput target.",
                "traffic_steering_xapp_quality_first": "Moves weak users toward stronger cells to protect reliability.",
                "urllc_guard_xapp_balanced": "Prevents BLER spikes while other xApps push throughput.",
            },
            "confidence_by_xapp": {
                "massive_mimo_beamformer_xapp_capacity_push": 0.82,
                "traffic_steering_xapp_quality_first": 0.74,
                "urllc_guard_xapp_balanced": 0.71,
            },
        }
        try:
            payload = self.client.generate_json_or_raise(
                system_prompt=(
                    "You are a Non-RT RIC xApp orchestrator. Choose a small portfolio of concrete xApps "
                    "that best matches the goal. Respond as JSON with keys: selected_xapps, "
                    "orchestration_mode, rationale_by_xapp, confidence_by_xapp. "
                    "Return JSON only. Do not include markdown or prose. "
                    "selected_xapps must be a non-empty list of xapp_name values drawn from available_xapps."
                ),
                user_prompt=llm_json_dumps(
                    {
                        "task": "Select one xApp portfolio for the current goal. Prefer complementary xApps over redundant ones.",
                        "output_rules": [
                            "Return exactly one JSON object.",
                            "Choose at most the requested number of xApps.",
                            "Prefer concrete xApp variants over abstract categories when both are available.",
                            "Avoid redundant scheduler-dominant portfolios unless the goal explicitly requires it.",
                        ],
                        "few_shot_example": {
                            "input": example_selection_input,
                            "output": example_selection_output,
                        },
                        "input": {
                            "goal": goal,
                            "reflection": reflection,
                            "observation_summary": observation_summary_for_llm(observation),
                            "available_xapps": self._available_xapp_catalog(),
                            "available_xapp_categories": self._available_xapp_category_catalog(),
                            "retrieved_success_cases": [
                                self._memory_case_summary(memory_case)
                                for memory_case in list(retrieved_success_cases or [])
                            ],
                            "retrieved_failure_cases": [
                                self._memory_case_summary(memory_case)
                                for memory_case in list(retrieved_failure_cases or [])
                            ],
                            "max_selected_xapps": self.max_selected_xapps,
                        },
                    }
                ),
            )
        except Exception:
            return self._heuristic_selection(
                goal=goal,
                observation=observation,
                reflection=reflection,
                retrieved_success_cases=retrieved_success_cases,
                retrieved_failure_cases=retrieved_failure_cases,
                previous_selected_xapps=previous_selected_xapps,
                generated_by=f"{self.client.generated_by_tag}_heuristic_selection_fallback",
            )
        if not isinstance(payload, dict):
            return self._heuristic_selection(
                goal=goal,
                observation=observation,
                reflection=reflection,
                retrieved_success_cases=retrieved_success_cases,
                retrieved_failure_cases=retrieved_failure_cases,
                previous_selected_xapps=previous_selected_xapps,
                generated_by=f"{self.client.generated_by_tag}_invalid_payload_fallback",
            )
        llm_resolved_names = self._resolve_selected_xapp_names(
            requested_names=payload.get("selected_xapps", []),
            goal=goal,
            reflection=reflection,
        )
        if len(llm_resolved_names) == 0:
            llm_resolved_names = self._resolve_selected_xapp_names(
                requested_names=payload.get("selected_categories", []),
                goal=goal,
                reflection=reflection,
            )
        if len(llm_resolved_names) == 0:
            return self._heuristic_selection(
                goal=goal,
                observation=observation,
                reflection=reflection,
                retrieved_success_cases=retrieved_success_cases,
                retrieved_failure_cases=retrieved_failure_cases,
                previous_selected_xapps=previous_selected_xapps,
                generated_by=f"{self.client.generated_by_tag}_empty_selection_fallback",
        )
        llm_category_portfolios = self._candidate_category_portfolios(
            base_categories=self._xapp_categories(llm_resolved_names) or self._requested_categories(goal=goal, observation=observation),
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
            previous_selected_xapps=previous_selected_xapps,
        )
        llm_requested_names = [list(llm_resolved_names)] + llm_category_portfolios
        selected_candidate_plan, candidate_plan_scores, candidate_plan_projections = self._select_best_candidate_plan(
            candidate_requested_names=llm_requested_names,
            goal=goal,
            observation=observation,
            reflection=reflection,
            base_categories=self._xapp_categories(llm_resolved_names) or self._requested_categories(goal=goal, observation=observation),
            previous_selected_xapps=previous_selected_xapps,
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
        )
        selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        if len(selected_names) == 0:
            selected_names = list(llm_resolved_names)
        interference_urllc_repaired_names = self._repair_reflective_interference_urllc_coexistence_coverage(
            selected_names=selected_names,
            goal=goal,
            observation=observation,
        )
        if interference_urllc_repaired_names != selected_names and len(interference_urllc_repaired_names) > 0:
            interference_urllc_candidate_plan = self.feasibility_projector.project_candidate(
                candidate_id=(
                    "candidate:llm_reflective_interference_urllc_coexistence_coverage:"
                    + ":".join(self._xapp_categories(interference_urllc_repaired_names) or ["uncategorized"])
                ),
                requested_xapps=interference_urllc_repaired_names,
                goal=goal,
                observation=observation,
                available_xapp_names=list(self._xapp_catalog_by_name().keys()),
                requested_eta_profile="eta:mobility_stable",
                requested_chi_profile="chi:guardrail_first",
            )
            if len(getattr(interference_urllc_candidate_plan, "x", []) or []) > 0:
                interference_urllc_score = self._portfolio_score_breakdown(
                    candidate_plan=interference_urllc_candidate_plan,
                    requested_categories=list(getattr(interference_urllc_candidate_plan, "requested_xapps", []) or []),
                    base_categories=self._xapp_categories(interference_urllc_repaired_names),
                    previous_selected_xapps=previous_selected_xapps,
                    retrieved_success_cases=retrieved_success_cases,
                    retrieved_failure_cases=retrieved_failure_cases,
                    goal=goal,
                    observation=observation,
                )
                interference_urllc_score.score_version = (
                    "j_sel_llm_reflective_interference_urllc_coexistence_coverage_objective"
                )
                interference_urllc_score.rationale += (
                    ", selected_by=llm_reflective_interference_urllc_coexistence_coverage_repair"
                )
                for candidate_score in candidate_plan_scores:
                    candidate_score.selected = False
                interference_urllc_score.selected = True
                candidate_plan_scores.insert(0, interference_urllc_score)
                candidate_plan_projections.insert(0, interference_urllc_candidate_plan)
                selected_candidate_plan = interference_urllc_candidate_plan
                selected_names = list(getattr(selected_candidate_plan, "x", []) or [])
        rationale_by_xapp = payload.get("rationale_by_xapp", payload.get("rationale_by_category", {}))
        confidence_by_xapp = payload.get("confidence_by_xapp", payload.get("confidence_by_category", {}))
        catalog = self._xapp_catalog_by_name()
        candidate_scores: list[XAppCandidateScore] = []
        for xapp_name in selected_names:
            category = canonical_xapp_category(xapp_name)
            spec = catalog.get(xapp_name, catalog.get(category, {}))
            confidence = 0.35
            if isinstance(confidence_by_xapp, dict):
                if xapp_name in confidence_by_xapp:
                    confidence = float(confidence_by_xapp[xapp_name])
                elif category in confidence_by_xapp:
                    confidence = float(confidence_by_xapp[category])
            spec_kpis = spec.get("kpis", [])
            if not isinstance(spec_kpis, list):
                spec_kpis = []
            rationale = str(spec.get("description", ""))
            if isinstance(rationale_by_xapp, dict):
                if xapp_name in rationale_by_xapp:
                    rationale = str(rationale_by_xapp[xapp_name])
                elif category in rationale_by_xapp:
                    rationale = str(rationale_by_xapp[category])
            candidate_scores.append(
                XAppCandidateScore(
                    xapp_name=xapp_name,
                    rationale=rationale,
                    expected_kpi_impact={
                        str(kpi_name): float(confidence)
                        for kpi_name in spec_kpis[:3]
                    },
                    confidence=float(confidence),
                    selected=True,
                )
            )
        return XAppSelectionPlan(
            plan_id=f"selection:{goal.goal_id}:g1",
            goal_id=str(goal.goal_id),
            selected_xapps=selected_names,
            selection_query_key=selection_query_key,
            candidate_plan_projections=list(candidate_plan_projections),
            candidate_plan_scores=list(candidate_plan_scores),
            candidate_scores=candidate_scores,
            eta=getattr(selected_candidate_plan, "eta", None),
            chi=getattr(selected_candidate_plan, "chi", None),
            hard_safety_envelope=list(getattr(selected_candidate_plan, "B", []) or default_hard_safety_envelope(selected_names, goal)),
            forbidden_combinations=list(
                getattr(selected_candidate_plan, "forbidden_combinations", []) or default_forbidden_combinations(selected_names)
            ),
            exploration_health_gate=getattr(
                selected_candidate_plan,
                "exploration_health_gate",
                default_exploration_health_gate(goal),
            ),
            qos_protection_profile=getattr(selected_candidate_plan, "qos_protection_profile", None),
            feasibility_summary=getattr(selected_candidate_plan, "feasibility_summary", None),
            orchestration_mode=str(payload.get("orchestration_mode", "parallel_with_guardrail") or "parallel_with_guardrail"),
            status="active",
            selection_locked=True,
            selection_generation=1,
            reflection=reflection,
            generated_by=f"{self.client.generated_by_tag}_candidate_scoring",
        )

    def _updated_plan_state(
        self,
        *,
        existing_plan: XAppSelectionPlan,
        observation: Observation,
        goal,
        validated_intent_template=None,
    ) -> tuple[XAppSelectionPlan, bool, str]:
        gap_score = goal_gap_score(
            goal,
            observation,
            validated_intent_template=validated_intent_template,
        )
        previous_best = getattr(existing_plan, "best_goal_gap_score", None)
        best_gap = gap_score if previous_best is None else min(float(previous_best), float(gap_score))
        previous_gap = getattr(existing_plan, "last_goal_gap_score", None)
        violation_count = int(getattr(existing_plan, "consecutive_gap_violation_count", 0))
        stagnation_count = int(getattr(existing_plan, "stagnation_count", 0))
        if gap_score > self.failure_gap_threshold:
            violation_count += 1
        else:
            violation_count = 0
        if previous_gap is not None and (float(previous_gap) - float(gap_score)) <= self.stagnation_epsilon:
            stagnation_count += 1
        else:
            stagnation_count = 0

        failure_reason = ""
        should_reselect = False
        if violation_count >= self.gap_patience_rounds:
            should_reselect = True
            failure_reason = "intent_gap_threshold_exceeded"
        elif gap_score > self.convergence_gap_threshold and stagnation_count >= self.stagnation_patience_rounds:
            should_reselect = True
            failure_reason = "joint_tuning_not_converging"

        updated_plan = XAppSelectionPlan(
            plan_id=str(getattr(existing_plan, "plan_id", "")),
            goal_id=str(getattr(existing_plan, "goal_id", "")),
            selected_xapps=list(getattr(existing_plan, "selected_xapps", []) or []),
            selection_query_key=getattr(existing_plan, "selection_query_key", None),
            candidate_plan_projections=list(getattr(existing_plan, "candidate_plan_projections", []) or []),
            candidate_plan_scores=list(getattr(existing_plan, "candidate_plan_scores", []) or []),
            candidate_scores=list(getattr(existing_plan, "candidate_scores", []) or []),
            eta=getattr(existing_plan, "eta", None),
            chi=getattr(existing_plan, "chi", None),
            orchestration_mode=str(getattr(existing_plan, "orchestration_mode", "parallel_with_guardrail")),
            status="failed" if should_reselect else "active",
            selection_locked=not should_reselect,
            selection_generation=int(getattr(existing_plan, "selection_generation", 1) or 1),
            last_selection_slot=getattr(existing_plan, "last_selection_slot", None),
            last_reflection_slot=getattr(existing_plan, "last_reflection_slot", None),
            last_goal_gap_score=float(gap_score),
            best_goal_gap_score=float(best_gap),
            consecutive_gap_violation_count=int(violation_count),
            stagnation_count=int(stagnation_count),
            failure_count=int(getattr(existing_plan, "failure_count", 0) or 0),
            last_failure_reason=str(getattr(existing_plan, "last_failure_reason", "")),
            last_failure_gap_score=getattr(existing_plan, "last_failure_gap_score", None),
            hard_safety_envelope=list(getattr(existing_plan, "hard_safety_envelope", []) or []),
            forbidden_combinations=list(getattr(existing_plan, "forbidden_combinations", []) or []),
            exploration_health_gate=getattr(existing_plan, "exploration_health_gate", None),
            qos_protection_profile=getattr(existing_plan, "qos_protection_profile", None),
            feasibility_summary=getattr(existing_plan, "feasibility_summary", None),
            reflection=getattr(existing_plan, "reflection", None),
            generated_by=str(getattr(existing_plan, "generated_by", "heuristic")),
        )
        if should_reselect:
            updated_plan.failure_count = int(updated_plan.failure_count) + 1
            updated_plan.last_failure_reason = failure_reason
            updated_plan.last_failure_gap_score = float(gap_score)
        return updated_plan, should_reselect, failure_reason

    def _reselection_update(
        self,
        *,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
        goal,
        existing_plan: XAppSelectionPlan,
        failure_reason: str,
    ) -> PolicyUpdate:
        previous_selected_xapps = list(getattr(existing_plan, "selected_xapps", []) or [])
        selection_query_key = self._selection_query_key(
            goal=goal,
            observation=observation,
            previous_selected_xapps=previous_selected_xapps,
        )
        retrieved_success_cases, retrieved_failure_cases = self._retrieve_memory_cases(
            selection_query_key=selection_query_key,
        )
        reflection = (
            self._build_reflection(
                observation=observation,
                policy_snapshot=policy_snapshot,
                retrieved_success_cases=retrieved_success_cases,
                retrieved_failure_cases=retrieved_failure_cases,
            )
            if self.reflection_enabled
            else None
        )
        self._store_failure_case(
            self._build_memory_case(
                memory_type="failure",
                goal=goal,
                observation=observation,
                plan=existing_plan,
                reflection=reflection,
                validated_intent_template=getattr(
                    policy_snapshot,
                    "validated_intent_template",
                    None,
                ),
                failure_reason=failure_reason,
            )
        )
        plan = self._try_llm_selection_with_memory(
            goal=goal,
            observation=observation,
            reflection=reflection,
            retrieved_success_cases=retrieved_success_cases,
            retrieved_failure_cases=retrieved_failure_cases,
            previous_selected_xapps=previous_selected_xapps,
        )
        generation = int(getattr(existing_plan, "selection_generation", 1) or 1) + 1
        slot_id = int(getattr(observation, "slot_id", 0) or 0)
        plan.plan_id = f"selection:{goal.goal_id}:g{generation}"
        plan.status = "active"
        plan.selection_locked = True
        plan.selection_generation = generation
        plan.last_selection_slot = slot_id
        plan.last_reflection_slot = slot_id
        plan.last_goal_gap_score = None
        plan.best_goal_gap_score = None
        plan.consecutive_gap_violation_count = 0
        plan.stagnation_count = 0
        plan.failure_count = int(getattr(existing_plan, "failure_count", 0) or 0)
        plan.last_failure_reason = failure_reason
        plan.last_failure_gap_score = getattr(existing_plan, "last_failure_gap_score", None)

        enabled_xapps, disabled_xapps = self._enabled_disabled_xapps(list(plan.selected_xapps))
        return PolicyUpdate(
            xapp_selection_plan=plan,
            orchestration_plan_view=orchestration_plan_view_from_plan(plan),
            memory_state=self._build_memory_state(),
            enabled_xapps=enabled_xapps,
            disabled_xapps=disabled_xapps,
            policy_type_name="xapp_selection_plan",
            policy_description=f"Reselect xApp portfolio after {failure_reason}",
            source_name=self.name,
        )

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None = None,
    ) -> PolicyUpdate:
        goal = getattr(policy_snapshot, "structured_goal", None) if policy_snapshot is not None else None
        if goal is None:
            return PolicyUpdate(source_name=self.name)

        existing_plan = getattr(policy_snapshot, "xapp_selection_plan", None) if policy_snapshot is not None else None
        current_goal_id = str(getattr(goal, "goal_id", ""))
        slot_id = int(getattr(observation, "slot_id", 0) or 0)
        if existing_plan is None or str(getattr(existing_plan, "goal_id", "")) != current_goal_id:
            previous_selected_xapps = (
                list(getattr(policy_snapshot, "enabled_xapps", []) or [])
                if policy_snapshot is not None
                else []
            )
            selection_query_key = self._selection_query_key(
                goal=goal,
                observation=observation,
                previous_selected_xapps=previous_selected_xapps,
            )
            retrieved_success_cases, retrieved_failure_cases = self._retrieve_memory_cases(
                selection_query_key=selection_query_key,
            )
            reflection = None
            plan = self._try_llm_selection_with_memory(
                goal=goal,
                observation=observation,
                reflection=reflection,
                retrieved_success_cases=retrieved_success_cases,
                retrieved_failure_cases=retrieved_failure_cases,
                previous_selected_xapps=previous_selected_xapps,
            )
            plan.last_selection_slot = slot_id
            enabled_xapps, disabled_xapps = self._enabled_disabled_xapps(list(plan.selected_xapps))
            return PolicyUpdate(
                xapp_selection_plan=plan,
                orchestration_plan_view=orchestration_plan_view_from_plan(plan),
                memory_state=self._build_memory_state(),
                enabled_xapps=enabled_xapps,
                disabled_xapps=disabled_xapps,
                policy_type_name="xapp_selection_plan",
                policy_description="Initial xApp portfolio selection for current structured goal",
                source_name=self.name,
            )

        updated_plan, should_reselect, failure_reason = self._updated_plan_state(
            existing_plan=existing_plan,
            observation=observation,
            goal=goal,
            validated_intent_template=getattr(
                policy_snapshot,
                "validated_intent_template",
                None,
            )
            if policy_snapshot is not None
            else None,
        )
        if self.static_xapp_selection_enabled:
            updated_plan.status = "active"
            updated_plan.selection_locked = True
            return PolicyUpdate(
                xapp_selection_plan=updated_plan,
                orchestration_plan_view=orchestration_plan_view_from_plan(updated_plan),
                memory_state=self._build_memory_state(),
                policy_type_name="xapp_selection_plan_state",
                policy_description="Static xApp selection baseline keeps the initial portfolio fixed",
                source_name=self.name,
            )
        if should_reselect and policy_snapshot is not None:
            updated_plan.last_failure_gap_score = updated_plan.last_goal_gap_score
            return self._reselection_update(
                observation=observation,
                policy_snapshot=policy_snapshot,
                goal=goal,
                existing_plan=updated_plan,
                failure_reason=failure_reason,
            )

        if (
            updated_plan.last_goal_gap_score is not None
            and float(updated_plan.last_goal_gap_score) <= self.success_gap_threshold
            and updated_plan.best_goal_gap_score is not None
            and abs(float(updated_plan.best_goal_gap_score) - float(updated_plan.last_goal_gap_score)) <= 1e-9
        ):
            self._store_success_case(
                self._build_memory_case(
                    memory_type="success",
                    goal=goal,
                    observation=observation,
                    plan=updated_plan,
                    reflection=None,
                    validated_intent_template=getattr(
                        policy_snapshot,
                        "validated_intent_template",
                        None,
                    )
                    if policy_snapshot is not None
                    else None,
                )
            )

        return PolicyUpdate(
            xapp_selection_plan=updated_plan,
            orchestration_plan_view=orchestration_plan_view_from_plan(updated_plan),
            memory_state=self._build_memory_state(),
            policy_type_name="xapp_selection_plan_state",
            policy_description="Keep current xApp portfolio while monitoring goal gap",
            source_name=self.name,
        )
