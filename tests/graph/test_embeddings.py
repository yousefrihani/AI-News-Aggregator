"""Unit tests for deterministic and fastembed embedding providers."""

import math
import unittest
from unittest.mock import patch

from Articles.graph.embeddings import (
    DeterministicEmbedding,
    FastEmbedEmbedding,
    _import_text_embedding,
    _normalize,
)
from Articles.graph.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    GraphConfigError,
)


def l2_norm(vector: list[float]) -> float:
    """Return the Euclidean norm of a vector."""
    return math.sqrt(sum(value * value for value in vector))


class FakeTextEmbedding:
    """Small stand-in for ``fastembed.TextEmbedding``; never loads a model."""

    instances: list["FakeTextEmbedding"] = []
    vector: list[float] = [1.0, 2.0, 2.0, 0.0]
    fail_length: int | None = None
    non_finite: bool = False

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embed_calls: list[list[str]] = []
        FakeTextEmbedding.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return configured vectors for every input text."""
        self.embed_calls.append(list(texts))
        length = self.fail_length or len(self.vector)
        vector = self.vector[:length]
        if self.non_finite:
            vector = [float("nan")] * length
        return [list(vector) for _ in texts]


class TestDeterministicEmbedding(unittest.IsolatedAsyncioTestCase):
    """Deterministic vectors, dimensions, normalization, and lifecycle."""

    async def test_default_dimension_and_stability(self) -> None:
        provider = DeterministicEmbedding()
        vectors = await provider.embed(["alpha", "beta"])
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 384 for vector in vectors))
        self.assertEqual(vectors, await provider.embed(["alpha", "beta"]))
        self.assertEqual(vectors, await DeterministicEmbedding().embed(["alpha", "beta"]))
        self.assertNotEqual(vectors[0], vectors[1])

    async def test_vectors_are_normalized(self) -> None:
        provider = DeterministicEmbedding(dim=16)
        for vector in await provider.embed(["one", "two", "three"]):
            self.assertAlmostEqual(l2_norm(vector), 1.0)

    async def test_empty_input(self) -> None:
        provider = DeterministicEmbedding(dim=8)
        self.assertEqual(await provider.embed([]), [])

    async def test_dimension_one(self) -> None:
        vector = (await DeterministicEmbedding(dim=1).embed(["x"]))[0]
        self.assertEqual(len(vector), 1)
        self.assertAlmostEqual(abs(vector[0]), 1.0)

    async def test_invalid_dimensions(self) -> None:
        for dim in (0, -1, 1.5, True):
            with self.subTest(dim=dim), self.assertRaises(GraphConfigError):
                DeterministicEmbedding(dim=dim)  # type: ignore[arg-type]

    async def test_aclose_is_a_noop(self) -> None:
        provider = DeterministicEmbedding(dim=4)
        await provider.aclose()
        self.assertEqual(len(await provider.embed(["still works"])), 1)


class TestFastEmbedding(unittest.IsolatedAsyncioTestCase):
    """FastEmbed adapter behaviour with a fake text embedder."""

    def setUp(self) -> None:
        FakeTextEmbedding.instances = []
        FakeTextEmbedding.vector = [1.0, 2.0, 2.0, 0.0]
        FakeTextEmbedding.fail_length = None
        FakeTextEmbedding.non_finite = False

    async def test_missing_package_raises_install_hint(self) -> None:
        with (
            patch(
                "Articles.graph.embeddings.importlib.import_module",
                side_effect=ImportError("missing"),
            ),
            self.assertRaises(GraphConfigError) as raised,
        ):
            _import_text_embedding()
        self.assertIn("pip install ai-news-aggregator[embeddings-local]", str(raised.exception))
        with (
            patch(
                "Articles.graph.embeddings.importlib.import_module",
                side_effect=ImportError("missing"),
            ),
            self.assertRaises(GraphConfigError),
        ):
            FastEmbedEmbedding()

    async def test_model_is_created_lazily(self) -> None:
        with patch(
            "Articles.graph.embeddings._import_text_embedding",
            return_value=FakeTextEmbedding,
        ):
            provider = FastEmbedEmbedding(model_name="fake/model", dim=4, cache_dir="/tmp/cache")
            self.assertEqual(FakeTextEmbedding.instances, [])
            vectors = await provider.embed(["hello"])
        self.assertEqual(len(FakeTextEmbedding.instances), 1)
        instance = FakeTextEmbedding.instances[0]
        self.assertEqual(instance.model_name, "fake/model")
        self.assertEqual(instance.cache_dir, "/tmp/cache")
        self.assertEqual(instance.embed_calls, [["hello"]])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 4)
        self.assertAlmostEqual(l2_norm(vectors[0]), 1.0)

    async def test_empty_input_never_creates_model(self) -> None:
        with patch(
            "Articles.graph.embeddings._import_text_embedding",
            return_value=FakeTextEmbedding,
        ):
            provider = FastEmbedEmbedding(dim=4)
            self.assertEqual(await provider.embed([]), [])
        self.assertEqual(FakeTextEmbedding.instances, [])

    async def test_dimension_mismatch_raises(self) -> None:
        FakeTextEmbedding.fail_length = 3
        with patch(
            "Articles.graph.embeddings._import_text_embedding",
            return_value=FakeTextEmbedding,
        ):
            provider = FastEmbedEmbedding(dim=4)
            with self.assertRaises(EmbeddingDimensionError):
                await provider.embed(["hello"])

    async def test_non_finite_vector_raises(self) -> None:
        FakeTextEmbedding.non_finite = True
        with patch(
            "Articles.graph.embeddings._import_text_embedding",
            return_value=FakeTextEmbedding,
        ):
            provider = FastEmbedEmbedding(dim=4)
            with self.assertRaises(EmbeddingError):
                await provider.embed(["hello"])

    async def test_invalid_constructor_values(self) -> None:
        with patch(
            "Articles.graph.embeddings._import_text_embedding",
            return_value=FakeTextEmbedding,
        ):
            for dim in (0, -1, 2.5, True):
                with self.subTest(dim=dim), self.assertRaises(GraphConfigError):
                    FastEmbedEmbedding(dim=dim)  # type: ignore[arg-type]
            with self.assertRaises(GraphConfigError):
                FastEmbedEmbedding(model_name=" ")

    async def test_aclose_is_a_noop(self) -> None:
        with patch(
            "Articles.graph.embeddings._import_text_embedding",
            return_value=FakeTextEmbedding,
        ):
            provider = FastEmbedEmbedding(dim=4)
            await provider.aclose()
            self.assertEqual(len(await provider.embed(["x"])), 1)

    def test_normalize_zero_vector(self) -> None:
        self.assertEqual(_normalize([0.0, 0.0]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
