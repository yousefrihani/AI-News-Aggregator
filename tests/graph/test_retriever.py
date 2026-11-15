"""Offline tests for the Graph RAG retriever."""

import unittest
from collections.abc import Sequence

from Articles.graph.base import Chunk, Entity, Mention, Relation
from Articles.graph.errors import GraphConfigError
from Articles.graph.memory import InMemoryGraphStore
from Articles.graph.retriever import (
    Citation,
    GraphRAGRetriever,
    RetrievedContext,
)

from tests.graph.test_memory import make_article


class FixedEmbedding:
    """Embedder returning one fixed vector for every input."""

    def __init__(self, vector: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> None:
        self.vector = list(vector)
        self.dim = len(self.vector)
        self.calls: list[Sequence[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return the fixed vector once per input text."""
        self.calls.append(list(texts))
        return [list(self.vector) for _ in texts]

    async def aclose(self) -> None:
        """Release resources; none are owned."""


async def make_store() -> InMemoryGraphStore:
    """Build a two-article store with entities, relations, and mentions."""
    store = InMemoryGraphStore()
    await store.upsert_articles(
        [make_article("a1"), make_article("a2")],
        [
            Chunk("c1", "a1", 0, "OpenAI released GPT-5.", (1.0, 0.0, 0.0, 0.0)),
            Chunk("c2", "a1", 1, "GPT-5 improves reasoning.", (0.0, 1.0, 0.0, 0.0)),
            Chunk("c3", "a2", 0, "A different story.", (0.0, 0.0, 1.0, 0.0)),
        ],
    )
    await store.upsert_entities(
        [
            Entity("e1", "OpenAI", "organization"),
            Entity("e2", "GPT-5", "product"),
        ],
        [Relation("e1", "e2", "released", confidence=0.9, evidence_chunk_id="c1")],
        [
            Mention("c1", "e1"),
            Mention("c1", "e2"),
            Mention("c2", "e2"),
        ],
    )
    return store


class TestRetrieverConfiguration(unittest.TestCase):
    """Constructor and query validation."""

    def test_invalid_settings_rejected(self) -> None:
        store = InMemoryGraphStore()
        embedder = FixedEmbedding()
        cases = [
            {"k": 0},
            {"k": True},
            {"depth": 0},
            {"depth": 3},
            {"limit": -1},
            {"max_context_chars": 0},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(GraphConfigError):
                GraphRAGRetriever(store, embedder, **kwargs)  # type: ignore[arg-type]

    def test_retrieved_context_validates(self) -> None:
        with self.assertRaises(ValueError):
            RetrievedContext(query=" ")
        with self.assertRaises(ValueError):
            Citation("")


class TestRetrieval(unittest.IsolatedAsyncioTestCase):
    """Vector seeds, expansion, citations, and context rendering."""

    async def asyncSetUp(self) -> None:
        self.store = await make_store()
        self.retriever = GraphRAGRetriever(self.store, FixedEmbedding(), k=1, depth=1, limit=10)

    async def test_blank_query_rejected(self) -> None:
        retriever = GraphRAGRetriever(InMemoryGraphStore(), FixedEmbedding())
        with self.assertRaises(ValueError):
            await retriever.retrieve("   ")

    async def test_embedder_receives_trimmed_query(self) -> None:
        store = InMemoryGraphStore()
        embedder = FixedEmbedding()
        retriever = GraphRAGRetriever(store, embedder)
        await retriever.retrieve("  question  ")
        self.assertEqual(embedder.calls, [["question"]])

    async def test_seed_and_expanded_chunks_are_returned(self) -> None:
        context = await self.retriever.retrieve("OpenAI")
        self.assertEqual([chunk.id for chunk in context.chunks], ["c1", "c2"])
        self.assertEqual(context.article_ids, ("a1",))
        self.assertEqual(context.citations[0].chunk_ids, ("c1", "c2"))
        self.assertEqual(context.scores, {"c1": 1.0})

    async def test_context_text_includes_article_and_chunks(self) -> None:
        context = await self.retriever.retrieve("OpenAI")
        self.assertIn("[a1]", context.context_text)
        self.assertIn("c1: OpenAI released GPT-5.", context.context_text)
        self.assertIn("c2: GPT-5 improves reasoning.", context.context_text)
        self.assertIn("OpenAI -[released]-> GPT-5", context.context_text)
        self.assertIn("Entities:", context.context_text)

    async def test_citations_are_grouped_in_first_use_order(self) -> None:
        context = await self.retriever.retrieve("OpenAI")
        self.assertEqual(context.citations, (Citation("a1", ("c1", "c2")),))

    async def test_k_limits_seed_hits(self) -> None:
        retriever = GraphRAGRetriever(self.store, FixedEmbedding((0.0, 1.0, 0.0, 0.0)), k=1)
        context = await retriever.retrieve("reasoning")
        self.assertEqual(context.chunks[0].id, "c2")

    async def test_depth_two_reaches_more_chunks(self) -> None:
        store = InMemoryGraphStore()
        await store.upsert_articles(
            [make_article("a1")],
            [
                Chunk("c1", "a1", 0, "Seed.", (1.0, 0.0)),
                Chunk("c2", "a1", 1, "Bridge.", (0.0, 1.0)),
                Chunk("c4", "a1", 2, "Far away.", (0.0, 1.0)),
            ],
        )
        await store.upsert_entities(
            [
                Entity("e1", "Seed entity", "concept"),
                Entity("e2", "Bridge entity", "concept"),
            ],
            [],
            [Mention("c1", "e1"), Mention("c2", "e1"), Mention("c2", "e2"), Mention("c4", "e2")],
        )

        depth_one = await GraphRAGRetriever(
            store, FixedEmbedding((1.0, 0.0)), k=1, depth=1
        ).retrieve("q")
        depth_two = await GraphRAGRetriever(
            store, FixedEmbedding((1.0, 0.0)), k=1, depth=2
        ).retrieve("q")

        self.assertEqual([chunk.id for chunk in depth_one.chunks], ["c1", "c2"])
        self.assertEqual([chunk.id for chunk in depth_two.chunks], ["c1", "c2", "c4"])

    async def test_limit_bounds_expansion(self) -> None:
        retriever = GraphRAGRetriever(self.store, FixedEmbedding(), k=1, depth=1, limit=1)
        context = await retriever.retrieve("OpenAI")
        self.assertEqual([chunk.id for chunk in context.chunks], ["c1"])

    async def test_empty_store_returns_empty_context(self) -> None:
        retriever = GraphRAGRetriever(InMemoryGraphStore(), FixedEmbedding())
        context = await retriever.retrieve("anything")
        self.assertTrue(context.is_empty)
        self.assertEqual(context.citations, ())
        self.assertEqual(context.context_text, "")

    async def test_context_cap_drops_unrendered_chunks(self) -> None:
        retriever = GraphRAGRetriever(
            self.store,
            FixedEmbedding(),
            k=1,
            depth=1,
            limit=10,
            max_context_chars=40,
        )
        context = await retriever.retrieve("OpenAI")
        self.assertEqual([chunk.id for chunk in context.chunks], ["c1"])
        self.assertEqual(context.citations[0].chunk_ids, ("c1",))
        self.assertNotIn("c2", context.context_text)

    async def test_duplicate_chunks_are_removed(self) -> None:
        retriever = GraphRAGRetriever(self.store, FixedEmbedding(), k=3, depth=1, limit=10)
        context = await retriever.retrieve("OpenAI")
        self.assertEqual(len({chunk.id for chunk in context.chunks}), len(context.chunks))
