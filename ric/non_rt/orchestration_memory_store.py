"""Persistence helpers for reflective Non-RT orchestration memory."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from models.intent_orchestration import (
    OrchestrationMemoryCase,
    ReflectionNote,
)


def _reflection_from_dict(payload: dict[str, object] | None) -> ReflectionNote | None:
    if not isinstance(payload, dict):
        return None
    return ReflectionNote(
        reflection_round=int(payload.get("reflection_round", 0) or 0),
        outcome_summary=str(payload.get("outcome_summary", "") or ""),
        self_critique=str(payload.get("self_critique", "") or ""),
        improvement_hypothesis=str(payload.get("improvement_hypothesis", "") or ""),
        recommended_changes=[
            str(item)
            for item in list(payload.get("recommended_changes", []) or [])
            if str(item).strip() != ""
        ],
        generated_by=str(payload.get("generated_by", "heuristic") or "heuristic"),
    )


def _memory_case_from_dict(payload: dict[str, object] | None) -> OrchestrationMemoryCase | None:
    if not isinstance(payload, dict):
        return None
    return OrchestrationMemoryCase(
        case_id=str(payload.get("case_id", "") or ""),
        memory_type=str(payload.get("memory_type", "success") or "success"),
        source_slot_id=(
            int(payload.get("source_slot_id"))
            if payload.get("source_slot_id", None) is not None
            else None
        ),
        last_retrieved_slot_id=(
            int(payload.get("last_retrieved_slot_id"))
            if payload.get("last_retrieved_slot_id", None) is not None
            else None
        ),
        encoding_version=str(
            payload.get("encoding_version", "structured_retrieval_v2_embedding")
            or "structured_retrieval_v2_embedding"
        ),
        retriever_backend_name=str(
            payload.get("retriever_backend_name", "hashed_feature_embedding_v1")
            or "hashed_feature_embedding_v1"
        ),
        goal_id=str(payload.get("goal_id", "") or ""),
        intent_id=str(payload.get("intent_id", "") or ""),
        risk_posture=str(payload.get("risk_posture", "balanced") or "balanced"),
        kpi_names=[
            str(item)
            for item in list(payload.get("kpi_names", []) or [])
            if str(item).strip() != ""
        ],
        selected_xapps=[
            str(item)
            for item in list(payload.get("selected_xapps", []) or [])
            if str(item).strip() != ""
        ],
        goal_gap_score=(
            float(payload.get("goal_gap_score"))
            if payload.get("goal_gap_score", None) is not None
            else None
        ),
        intent_loss_normalized=(
            float(payload.get("intent_loss_normalized"))
            if payload.get("intent_loss_normalized", None) is not None
            else None
        ),
        queue_bytes=(
            int(payload.get("queue_bytes"))
            if payload.get("queue_bytes", None) is not None
            else None
        ),
        throughput_mbps=(
            float(payload.get("throughput_mbps"))
            if payload.get("throughput_mbps", None) is not None
            else None
        ),
        avg_bler=(
            float(payload.get("avg_bler"))
            if payload.get("avg_bler", None) is not None
            else None
        ),
        context_signature=dict(payload.get("context_signature", {}) or {}),
        encoded_features={
            str(key): float(value)
            for key, value in dict(payload.get("encoded_features", {}) or {}).items()
            if str(key).strip() != ""
        },
        embedding_vector=[
            float(item)
            for item in list(payload.get("embedding_vector", []) or [])
        ],
        retrieval_hits=int(payload.get("retrieval_hits", 0) or 0),
        retention_score=float(payload.get("retention_score", 0.0) or 0.0),
        summary=str(payload.get("summary", "") or ""),
        failure_reason=str(payload.get("failure_reason", "") or ""),
        reflection=_reflection_from_dict(payload.get("reflection", None)),
        generated_by=str(payload.get("generated_by", "heuristic") or "heuristic"),
    )


class OrchestrationMemoryStore:
    """Simple JSON-backed store for success/failure orchestration cases."""

    def __init__(self, path: str | None) -> None:
        normalized_path = str(path or "").strip()
        self.path = Path(normalized_path) if normalized_path != "" else None

    def load(
        self,
    ) -> tuple[list[OrchestrationMemoryCase], list[OrchestrationMemoryCase], list[ReflectionNote]]:
        if self.path is None or not self.path.exists():
            return [], [], []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return [], [], []
        success_cases = [
            memory_case
            for memory_case in (
                _memory_case_from_dict(item)
                for item in list(payload.get("success_cases", []) or [])
            )
            if memory_case is not None
        ]
        failure_cases = [
            memory_case
            for memory_case in (
                _memory_case_from_dict(item)
                for item in list(payload.get("failure_cases", []) or [])
            )
            if memory_case is not None
        ]
        reflection_notes = [
            note
            for note in (
                _reflection_from_dict(item)
                for item in list(payload.get("reflection_notes", []) or [])
            )
            if note is not None
        ]
        return success_cases, failure_cases, reflection_notes

    def save(
        self,
        *,
        success_cases: list[OrchestrationMemoryCase],
        failure_cases: list[OrchestrationMemoryCase],
        reflection_notes: list[ReflectionNote],
    ) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "success_cases": [asdict(memory_case) for memory_case in list(success_cases or [])],
            "failure_cases": [asdict(memory_case) for memory_case in list(failure_cases or [])],
            "reflection_notes": [asdict(note) for note in list(reflection_notes or [])],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
