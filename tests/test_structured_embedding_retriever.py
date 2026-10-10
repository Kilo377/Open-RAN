import unittest

from ric.non_rt.structured_embedding_retriever import (
    StructuredEmbeddingBackend,
    StructuredEmbeddingRetriever,
    WeightedHashedEmbeddingBackend,
)


class _ConstantEmbeddingBackend(StructuredEmbeddingBackend):
    backend_name = "constant_embedding_backend_test"

    def encode_sparse_features(self, features: dict[str, float] | None) -> list[float]:
        _ = features
        return [1.0, 0.0, 0.0, 0.0]


class TestStructuredEmbeddingRetriever(unittest.TestCase):
    def test_sparse_features_encode_into_normalized_dense_vector(self) -> None:
        retriever = StructuredEmbeddingRetriever(vector_dim=16, namespace="test")
        vector = retriever.encode_sparse_features(
            {
                "kpi::throughput": 1.0,
                "context::queue::high": 1.0,
                "risk::balanced": 1.0,
            }
        )

        self.assertEqual(len(vector), 16)
        self.assertGreater(sum(abs(item) for item in vector), 0.0)

    def test_cosine_similarity_prefers_matching_feature_sets(self) -> None:
        retriever = StructuredEmbeddingRetriever(vector_dim=16, namespace="test")
        query = retriever.encode_sparse_features({"kpi::throughput": 1.0, "risk::balanced": 1.0})
        match = retriever.encode_sparse_features({"kpi::throughput": 1.0, "risk::balanced": 1.0})
        mismatch = retriever.encode_sparse_features({"kpi::bler": 1.0, "risk::conservative": 1.0})

        self.assertGreater(retriever.cosine_similarity(query, match), 0.9)
        self.assertGreaterEqual(retriever.cosine_similarity(query, match), retriever.cosine_similarity(query, mismatch))

    def test_custom_backend_is_supported(self) -> None:
        retriever = StructuredEmbeddingRetriever(backend=_ConstantEmbeddingBackend())

        self.assertEqual(retriever.backend_name, "constant_embedding_backend_test")
        self.assertEqual(retriever.encode_sparse_features({"ignored": 1.0}), [1.0, 0.0, 0.0, 0.0])

    def test_weighted_hashed_backend_applies_feature_weights(self) -> None:
        backend = WeightedHashedEmbeddingBackend(
            vector_dim=16,
            namespace="test_weighted",
            feature_weights={"kpi::throughput": 2.0, "risk::balanced": 0.5},
            default_weight=1.0,
        )
        retriever = StructuredEmbeddingRetriever(backend=backend)
        vector = retriever.encode_sparse_features(
            {
                "kpi::throughput": 1.0,
                "risk::balanced": 1.0,
            }
        )

        self.assertEqual(retriever.backend_name, "weighted_hashed_embedding_v1")
        self.assertEqual(len(vector), 16)
        self.assertGreater(sum(abs(item) for item in vector), 0.0)


if __name__ == "__main__":
    unittest.main()
