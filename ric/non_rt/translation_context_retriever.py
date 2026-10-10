"""Structured retrieval context for validated intent translation."""

from __future__ import annotations

import hashlib
import re

from models.intent_orchestration import TranslationContextTrace, TranslationQueryKey
from models.observation import Observation
from ric.non_rt.structured_embedding_retriever import (
    StructuredEmbeddingRetriever,
    build_embedding_backend,
)


_KPI_CONTEXT_LIBRARY: tuple[dict[str, object], ...] = (
    {
        "source_id": "kpi:throughput",
        "source_type": "kpi_dictionary",
        "trigger_tokens": ("throughput", "capacity", "embb", "video"),
        "summary": "Use total_throughput_mbps for capacity-uplift intents.",
        "payload": {
            "recommended_kpi": "total_throughput_mbps",
            "semantic_type": "max",
            "unit": "mbps",
            "related_xapp_categories": [
                "massive_mimo_beamformer_xapp",
                "spectrum_optimizer_xapp",
                "traffic_steering_xapp",
            ],
        },
    },
    {
        "source_id": "kpi:bler",
        "source_type": "kpi_dictionary",
        "trigger_tokens": ("bler", "reliability", "robust", "stable"),
        "summary": "Use avg_bler for reliability-protection intents.",
        "payload": {
            "recommended_kpi": "avg_bler",
            "semantic_type": "min",
            "unit": "ratio",
            "related_xapp_categories": [
                "urllc_guard_xapp",
                "latency_scheduler_xapp",
                "massive_mimo_beamformer_xapp",
            ],
        },
    },
    {
        "source_id": "kpi:queue",
        "source_type": "kpi_dictionary",
        "trigger_tokens": ("queue", "backlog", "delay", "latency"),
        "summary": "Use total_queue_bytes or queue_bytes for backlog-sensitive intents.",
        "payload": {
            "recommended_kpis": ["queue_bytes", "total_queue_bytes"],
            "semantic_type": "min",
            "unit": "bytes",
            "related_xapp_categories": [
                "latency_scheduler_xapp",
                "traffic_steering_xapp",
                "ran_slicing_manager_xapp",
            ],
        },
    },
    {
        "source_id": "kpi:handover",
        "source_type": "kpi_dictionary",
        "trigger_tokens": ("handover", "mobility", "edge", "steering"),
        "summary": "Use handover_risk when mobility robustness matters.",
        "payload": {
            "recommended_kpi": "handover_risk",
            "semantic_type": "min",
            "unit": "ratio",
            "related_xapp_categories": [
                "mobility_predictor_xapp",
                "traffic_steering_xapp",
            ],
        },
    },
)

_POLICY_NOTE_LIBRARY: tuple[dict[str, object], ...] = (
    {
        "source_id": "policy:queue_guard",
        "source_type": "operator_policy_note",
        "summary": "When queue pressure is high, include at least one backlog-relief KPI or constraint.",
        "context_gate": ("queue_class", "high"),
        "payload": {
            "constraint_hints": ["limit queue growth during orchestration"],
            "recommended_kpis": ["total_queue_bytes"],
        },
    },
    {
        "source_id": "policy:bler_guard",
        "source_type": "operator_policy_note",
        "summary": "When BLER is stressed, reliability guardrails should remain explicit in the template.",
        "context_gate": ("bler_class", "stressed"),
        "payload": {
            "risk_posture_hint": "reliability_first",
            "constraint_hints": ["avoid aggressive capacity-only tuning"],
            "recommended_kpis": ["avg_bler"],
        },
    },
    {
        "source_id": "policy:mobility_guard",
        "source_type": "operator_policy_note",
        "summary": "When mobility pressure is high, prefer mobility-aware KPI and protection hints.",
        "context_gate": ("mobility_class", "high"),
        "payload": {
            "risk_posture_hint": "balanced",
            "recommended_kpis": ["handover_risk"],
            "constraint_hints": ["avoid fragile steering transitions"],
        },
    },
)

_ENCODING_VERSION = "translation_context_v2_embedding"


class TranslationContextRetriever:
    """Builds a lightweight retrieved context bundle R_k^{ctx} for translation."""

    def __init__(
        self,
        *,
        embedding_backend_name: str = "hashed_feature_embedding_v1",
        embedding_backend_state_path: str | None = None,
    ) -> None:
        self.embedding_retriever = StructuredEmbeddingRetriever(
            backend=build_embedding_backend(
                backend_name=embedding_backend_name,
                vector_dim=48,
                namespace="translation_context",
                state_path=embedding_backend_state_path,
            )
        )

    def _tokenize(self, text: str) -> list[str]:
        normalized = re.sub(r"[^a-zA-Z0-9_]+", " ", str(text or "").strip().lower())
        return [token for token in normalized.split() if token != ""]

    def _bucket_label(self, value: float, *, low: float, high: float) -> str:
        if value < low:
            return "low"
        if value < high:
            return "medium"
        return "high"

    def _context_signature(self, observation: Observation) -> dict[str, object]:
        queue_bytes = int(getattr(observation, "total_queue_bytes", 0) or 0)
        throughput_mbps = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
        avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
        handover_pending = int(getattr(observation, "handover_pending_ue_count", 0) or 0)
        return {
            "queue_class": self._bucket_label(float(queue_bytes), low=25000.0, high=80000.0),
            "throughput_class": self._bucket_label(throughput_mbps, low=20.0, high=40.0),
            "bler_class": "good" if avg_bler <= 0.08 else "stressed" if avg_bler <= 0.15 else "poor",
            "mobility_class": "high" if handover_pending >= 3 else "moderate" if handover_pending >= 1 else "low",
        }

    def _encode_features(
        self,
        *,
        intent_tokens: list[str],
        context_signature: dict[str, object],
        trigger_tokens: list[str] | tuple[str, ...] | None = None,
        context_gate: tuple[str, str] | None = None,
        recommended_kpis: list[str] | None = None,
    ) -> dict[str, float]:
        encoded: dict[str, float] = {}
        for token in list(intent_tokens or []):
            normalized = str(token).strip().lower()
            if normalized == "":
                continue
            encoded[f"intent_token:{normalized}"] = 1.0
        for trigger_token in list(trigger_tokens or []):
            normalized = str(trigger_token).strip().lower()
            if normalized == "":
                continue
            encoded[f"trigger_token:{normalized}"] = 1.0
        for key, value in dict(context_signature or {}).items():
            normalized_key = str(key).strip()
            normalized_value = str(value).strip()
            if normalized_key == "" or normalized_value == "":
                continue
            encoded[f"context:{normalized_key}={normalized_value}"] = 1.0
        if context_gate is not None and len(tuple(context_gate)) == 2:
            encoded[f"context_gate:{context_gate[0]}={context_gate[1]}"] = 1.0
        for kpi_name in list(recommended_kpis or []):
            normalized = str(kpi_name).strip()
            if normalized == "":
                continue
            encoded[f"recommended_kpi:{normalized}"] = 1.0
        return encoded

    def _query_key(self, *, intent_text: str, intent_id: str, observation: Observation) -> TranslationQueryKey:
        intent_tokens = self._tokenize(intent_text)
        context_signature = self._context_signature(observation)
        encoded_features = self._encode_features(
            intent_tokens=intent_tokens,
            context_signature=context_signature,
        )
        embedding_vector = self.embedding_retriever.encode_sparse_features(encoded_features)
        digest = hashlib.sha1(
            f"{intent_id}|{' '.join(intent_tokens)}|{context_signature}".encode("utf-8")
        ).hexdigest()[:10]
        return TranslationQueryKey(
            query_id=f"qctx:{intent_id.split(':', 1)[-1]}:{digest}",
            intent_id=str(intent_id or ""),
            encoding_version=_ENCODING_VERSION,
            retriever_backend_name=self.embedding_retriever.backend_name,
            intent_tokens=intent_tokens,
            context_signature=context_signature,
            encoded_features=encoded_features,
            embedding_vector=embedding_vector,
            notes=[
                f"queue_class={context_signature.get('queue_class', '')}",
                f"throughput_class={context_signature.get('throughput_class', '')}",
                f"bler_class={context_signature.get('bler_class', '')}",
                f"mobility_class={context_signature.get('mobility_class', '')}",
            ],
        )

    def _kpi_trace_score(
        self,
        library_item: dict[str, object],
        *,
        intent_tokens: list[str],
        context_signature: dict[str, object],
        query_vector: list[float],
    ) -> tuple[float, float, float, dict[str, float], list[float]]:
        trigger_tokens = {
            str(token).strip().lower()
            for token in list(library_item.get("trigger_tokens", []) or [])
            if str(token).strip() != ""
        }
        token_overlap = 0.0
        if len(trigger_tokens) > 0:
            token_overlap = float(len(trigger_tokens.intersection(intent_tokens))) / float(len(trigger_tokens))
        stress_bonus = 0.0
        recommended_kpi = str(
            library_item.get("payload", {}).get("recommended_kpi", "")
            if isinstance(library_item.get("payload", {}), dict)
            else ""
        ).strip()
        if recommended_kpi == "avg_bler" and str(context_signature.get("bler_class", "")) in {"stressed", "poor"}:
            stress_bonus = 0.25
        elif recommended_kpi == "total_throughput_mbps" and str(context_signature.get("throughput_class", "")) == "low":
            stress_bonus = 0.25
        elif recommended_kpi == "handover_risk" and str(context_signature.get("mobility_class", "")) == "high":
            stress_bonus = 0.25
        library_features = self._encode_features(
            intent_tokens=[],
            context_signature={},
            trigger_tokens=tuple(trigger_tokens),
            recommended_kpis=[
                str(item).strip()
                for item in list(library_item.get("payload", {}).get("recommended_kpis", []) or [])
                if isinstance(library_item.get("payload", {}), dict)
            ]
            + ([recommended_kpi] if recommended_kpi != "" else []),
        )
        library_embedding = self.embedding_retriever.encode_sparse_features(library_features)
        feature_similarity = self.embedding_retriever.cosine_similarity(
            query_vector,
            library_embedding,
        )
        relevance_score = float(
            min(
                1.0,
                0.45 * token_overlap + 0.40 * feature_similarity + stress_bonus,
            )
        )
        retrieval_confidence = float(
            self.embedding_retriever.retrieval_confidence(
                semantic_similarity=relevance_score,
                vector_similarity=feature_similarity,
            )
        )
        return relevance_score, feature_similarity, retrieval_confidence, library_features, library_embedding

    def retrieve(
        self,
        *,
        intent_text: str,
        intent_id: str,
        observation: Observation,
        top_k: int = 5,
    ) -> tuple[TranslationQueryKey, list[TranslationContextTrace], dict[str, object]]:
        query_key = self._query_key(
            intent_text=intent_text,
            intent_id=intent_id,
            observation=observation,
        )
        intent_tokens = list(getattr(query_key, "intent_tokens", []) or [])
        context_signature = dict(getattr(query_key, "context_signature", {}) or {})
        query_vector = list(getattr(query_key, "embedding_vector", []) or [])

        traces: list[TranslationContextTrace] = []
        for library_item in _KPI_CONTEXT_LIBRARY:
            (
                relevance_score,
                feature_similarity,
                retrieval_confidence,
                library_features,
                library_embedding,
            ) = self._kpi_trace_score(
                library_item,
                intent_tokens=intent_tokens,
                context_signature=context_signature,
                query_vector=query_vector,
            )
            if relevance_score <= 0.0:
                continue
            traces.append(
                TranslationContextTrace(
                    source_id=str(library_item.get("source_id", "") or ""),
                    source_type=str(library_item.get("source_type", "") or ""),
                    relevance_score=float(round(relevance_score, 4)),
                    encoding_version=_ENCODING_VERSION,
                    retriever_backend_name=self.embedding_retriever.backend_name,
                    encoded_features=library_features,
                    embedding_vector=library_embedding,
                    feature_similarity=float(round(feature_similarity, 4)),
                    retrieval_confidence=float(round(retrieval_confidence, 4)),
                    summary=str(library_item.get("summary", "") or ""),
                    payload=dict(library_item.get("payload", {}) or {}),
                )
            )

        for library_item in _POLICY_NOTE_LIBRARY:
            context_gate = tuple(library_item.get("context_gate", ()) or ())
            if len(context_gate) != 2:
                continue
            if str(context_signature.get(str(context_gate[0]), "")) != str(context_gate[1]):
                continue
            policy_features = self._encode_features(
                intent_tokens=[],
                context_signature={},
                context_gate=(str(context_gate[0]), str(context_gate[1])),
                recommended_kpis=[
                    str(item).strip()
                    for item in list(library_item.get("payload", {}).get("recommended_kpis", []) or [])
                    if isinstance(library_item.get("payload", {}), dict)
                ],
            )
            policy_embedding = self.embedding_retriever.encode_sparse_features(policy_features)
            feature_similarity = self.embedding_retriever.cosine_similarity(
                query_vector,
                policy_embedding,
            )
            relevance_score = float(min(1.0, 0.60 + 0.35 * feature_similarity))
            retrieval_confidence = float(
                self.embedding_retriever.retrieval_confidence(
                    semantic_similarity=relevance_score,
                    vector_similarity=feature_similarity,
                )
            )
            traces.append(
                TranslationContextTrace(
                    source_id=str(library_item.get("source_id", "") or ""),
                    source_type=str(library_item.get("source_type", "") or ""),
                    relevance_score=float(round(relevance_score, 4)),
                    encoding_version=_ENCODING_VERSION,
                    retriever_backend_name=self.embedding_retriever.backend_name,
                    encoded_features=policy_features,
                    embedding_vector=policy_embedding,
                    feature_similarity=float(round(feature_similarity, 4)),
                    retrieval_confidence=float(round(retrieval_confidence, 4)),
                    summary=str(library_item.get("summary", "") or ""),
                    payload=dict(library_item.get("payload", {}) or {}),
                )
            )

        traces.sort(
            key=lambda item: (
                float(getattr(item, "relevance_score", 0.0) or 0.0),
                float(getattr(item, "feature_similarity", 0.0) or 0.0),
                str(getattr(item, "source_type", "") or ""),
                str(getattr(item, "source_id", "") or ""),
            ),
            reverse=True,
        )
        top_traces = list(traces[: max(1, int(top_k))])
        for retrieval_rank, trace in enumerate(top_traces, start=1):
            trace.retrieval_rank = int(retrieval_rank)

        recommended_kpis: list[str] = []
        constraint_hints: list[str] = []
        related_categories: list[str] = []
        risk_posture_hint = ""
        for trace in top_traces:
            payload = dict(getattr(trace, "payload", {}) or {})
            for value in list(payload.get("recommended_kpis", []) or []):
                normalized = str(value).strip()
                if normalized != "" and normalized not in recommended_kpis:
                    recommended_kpis.append(normalized)
            single_kpi = str(payload.get("recommended_kpi", "") or "").strip()
            if single_kpi != "" and single_kpi not in recommended_kpis:
                recommended_kpis.append(single_kpi)
            for value in list(payload.get("constraint_hints", []) or []):
                normalized = str(value).strip()
                if normalized != "" and normalized not in constraint_hints:
                    constraint_hints.append(normalized)
            for value in list(payload.get("related_xapp_categories", []) or []):
                normalized = str(value).strip()
                if normalized != "" and normalized not in related_categories:
                    related_categories.append(normalized)
            if risk_posture_hint == "":
                risk_posture_hint = str(payload.get("risk_posture_hint", "") or "").strip()

        summary = {
            "query_id": str(getattr(query_key, "query_id", "") or ""),
            "retrieved_item_count": int(len(top_traces)),
            "recommended_kpis": recommended_kpis,
            "constraint_hints": constraint_hints,
            "related_xapp_categories": related_categories,
            "risk_posture_hint": risk_posture_hint,
            "context_signature": context_signature,
            "encoding_version": str(getattr(query_key, "encoding_version", "") or ""),
            "embedding_dimension": int(len(getattr(query_key, "embedding_vector", []) or [])),
            "max_relevance_score": max(
                [float(getattr(trace, "relevance_score", 0.0) or 0.0) for trace in top_traces] or [0.0]
            ),
        }
        return query_key, top_traces, summary
