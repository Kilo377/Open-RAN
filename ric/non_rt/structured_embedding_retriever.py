"""Shared structured embedding utilities for Non-RT retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class StructuredEmbeddingBackend:
    """Abstract backend for mapping sparse symbolic features into dense vectors."""

    backend_name = "base_structured_embedding_backend"

    def encode_sparse_features(self, features: dict[str, float] | None) -> list[float]:
        raise NotImplementedError


class HashedFeatureEmbeddingBackend(StructuredEmbeddingBackend):
    """Deterministic hashed backend used as the default local embedding model."""

    backend_name = "hashed_feature_embedding_v1"

    def __init__(
        self,
        *,
        vector_dim: int = 48,
        namespace: str = "default",
    ) -> None:
        self.vector_dim = max(8, int(vector_dim))
        self.namespace = str(namespace or "default").strip() or "default"

    def encode_sparse_features(self, features: dict[str, float] | None) -> list[float]:
        vector = [0.0] * self.vector_dim
        for raw_key, raw_value in dict(features or {}).items():
            key = str(raw_key).strip()
            if key == "":
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            digest = hashlib.sha1(
                f"{self.namespace}|{key}".encode("utf-8")
            ).digest()
            index = int.from_bytes(digest[:4], byteorder="big", signed=False) % self.vector_dim
            sign = 1.0 if int(digest[4]) % 2 == 0 else -1.0
            vector[index] += sign * value
        return self._normalize(vector)

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-9:
            return [0.0] * len(vector)
        return [float(value / norm) for value in vector]


class WeightedHashedEmbeddingBackend(StructuredEmbeddingBackend):
    """Hashed backend with externally loaded feature weights."""

    backend_name = "weighted_hashed_embedding_v1"

    def __init__(
        self,
        *,
        vector_dim: int = 48,
        namespace: str = "default",
        feature_weights: dict[str, float] | None = None,
        default_weight: float = 1.0,
    ) -> None:
        self.vector_dim = max(8, int(vector_dim))
        self.namespace = str(namespace or "default").strip() or "default"
        self.feature_weights = {
            str(key): float(value)
            for key, value in dict(feature_weights or {}).items()
            if str(key).strip() != ""
        }
        self.default_weight = max(0.0, float(default_weight))

    @classmethod
    def from_state_file(
        cls,
        state_path: str | None,
        *,
        vector_dim: int = 48,
        namespace: str = "default",
    ) -> "WeightedHashedEmbeddingBackend":
        payload: dict[str, object] = {}
        normalized_path = str(state_path or "").strip()
        if normalized_path != "":
            try:
                payload = json.loads(Path(normalized_path).read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        return cls(
            vector_dim=int(payload.get("vector_dim", vector_dim) or vector_dim),
            namespace=str(payload.get("namespace", namespace) or namespace),
            feature_weights=dict(payload.get("feature_weights", {}) or {}),
            default_weight=float(payload.get("default_weight", 1.0) or 1.0),
        )

    def encode_sparse_features(self, features: dict[str, float] | None) -> list[float]:
        vector = [0.0] * self.vector_dim
        for raw_key, raw_value in dict(features or {}).items():
            key = str(raw_key).strip()
            if key == "":
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            feature_weight = float(self.feature_weights.get(key, self.default_weight))
            if feature_weight <= 0.0:
                continue
            digest = hashlib.sha1(
                f"{self.namespace}|{key}".encode("utf-8")
            ).digest()
            index = int.from_bytes(digest[:4], byteorder="big", signed=False) % self.vector_dim
            sign = 1.0 if int(digest[4]) % 2 == 0 else -1.0
            vector[index] += sign * value * feature_weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-9:
            return [0.0] * len(vector)
        return [float(value / norm) for value in vector]


class StructuredEmbeddingRetriever:
    """Retriever facade with a pluggable structured embedding backend."""

    def __init__(
        self,
        *,
        vector_dim: int = 48,
        namespace: str = "default",
        backend: StructuredEmbeddingBackend | None = None,
    ) -> None:
        self.backend = backend or HashedFeatureEmbeddingBackend(
            vector_dim=vector_dim,
            namespace=namespace,
        )

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend, "backend_name", "unknown_backend") or "unknown_backend")

    def encode_sparse_features(self, features: dict[str, float] | None) -> list[float]:
        return list(self.backend.encode_sparse_features(features))

    def cosine_similarity(
        self,
        left_vector: list[float] | tuple[float, ...] | None,
        right_vector: list[float] | tuple[float, ...] | None,
    ) -> float:
        left = [float(item) for item in list(left_vector or [])]
        right = [float(item) for item in list(right_vector or [])]
        if len(left) == 0 or len(right) == 0:
            return 0.0
        vector_dim = min(len(left), len(right))
        if vector_dim <= 0:
            return 0.0
        numerator = sum(left[index] * right[index] for index in range(vector_dim))
        left_norm = math.sqrt(sum(value * value for value in left[:vector_dim]))
        right_norm = math.sqrt(sum(value * value for value in right[:vector_dim]))
        if left_norm <= 1e-9 or right_norm <= 1e-9:
            return 0.0
        similarity = numerator / (left_norm * right_norm)
        return float(max(0.0, min(1.0, similarity)))

    def retrieval_confidence(
        self,
        *,
        semantic_similarity: float = 0.0,
        vector_similarity: float = 0.0,
        recency_score: float = 0.0,
    ) -> float:
        confidence = (
            0.45 * max(0.0, float(semantic_similarity))
            + 0.40 * max(0.0, float(vector_similarity))
            + 0.15 * max(0.0, float(recency_score))
        )
        return float(max(0.0, min(1.0, confidence)))


def build_embedding_backend(
    *,
    backend_name: str = "hashed_feature_embedding_v1",
    vector_dim: int = 48,
    namespace: str = "default",
    state_path: str | None = None,
) -> StructuredEmbeddingBackend:
    normalized_name = str(backend_name or "hashed_feature_embedding_v1").strip().lower()
    if normalized_name in {"weighted_hashed_embedding_v1", "weighted_hashed", "trainable_weighted"}:
        return WeightedHashedEmbeddingBackend.from_state_file(
            state_path,
            vector_dim=vector_dim,
            namespace=namespace,
        )
    return HashedFeatureEmbeddingBackend(
        vector_dim=vector_dim,
        namespace=namespace,
    )
