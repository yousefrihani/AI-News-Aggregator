"""Offline tests for the enrichment pipeline.

The pipeline is exercised end to end against a scripted route router, the
deterministic embedder, and the in-memory graph store. No socket is opened and
no real embedding model is loaded.
"""

from __future__ import annotations

import json
import unittest
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from Articles.graph.base import chunk_id_for
from Articles.graph.embeddings import DeterministicEmbedding
from Articles.graph.memory import InMemoryGraphStore
from Articles.llm import CompletionResponse, Message
from Articles.llm.patterns import SummarizePattern
from Articles.Models import Article
from Articles.pipeline import (
    EnrichmentPipeline,
    PipelineReport,
    article_document,
    chunk_text,
)

from tests.llm.support import make_response

PUBLISHED = datetime(2026, 4, 5, 6, 7, 8, tzinfo=UTC)


def make_article(
    article_id: str = "a1",
    *,
    title: str = "Article title",
    summary: str = "A short summary about OpenAI and GPT-5.",
    topics: tuple[str, ...] = ("technology",),
    raw: dict[str, Any] | None = None,
) -> Article:
    """Build an article for pipeline tests."""
    return Article(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        source="NewsAPI",
        published_at=PUBLISHED,
        summary=summary,
        topics=topics,
        raw=raw or {},
    )


class RouteRouter:
    """Route-aware scripted router.

    Each route keeps its own queue so concurrently started summarization and
    classification calls cannot race for a shared response list.
    """

    def __init__(self, responses: dict[str, Sequence[str]]) -> None:
        self.responses = {route: list(texts) for route, texts in responses.items()}
        self.calls: list[tuple[str, tuple[Message, ...]]] = []

    async def complete(
        self,
        route: str,
        messages: Sequence[Message],
        *,
        provider: str | None = None,
        model: str | None = None,
        allow_fallback: bool | None = None,
        **overrides: Any,
    ) -> CompletionResponse:
        """Record a call and return the next scripted text for the route."""
        self.calls.append((route, tuple(messages)))
        queue = self.responses.get(route)
        if not queue:
            raise AssertionError(f"no scripted response left for route {route!r}")
        return make_response(text=queue.pop(0))

    def count(self, route: str) -> int:
        """Return the number of calls made to a route."""
        return sum(1 for name, _ in self.calls if name == route)


class CountingEmbedding:
    """Deterministic embedder that records batch sizes."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self._inner = DeterministicEmbedding(dim)
        self.batches: list[int] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch and record its size."""
        self.batches.append(len(texts))
        return await self._inner.embed(texts)

    async def aclose(self) -> None:
        """Release resources; none are owned."""


class WrongCountEmbedding(CountingEmbedding):
    """Embedder returning no vectors."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return an empty vector list regardless of input size."""
        self.batches.append(len(texts))
        return []


class WrongDimEmbedding(CountingEmbedding):
    """Embedder declaring one dimension and returning another."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return vectors shorter than the declared dimension."""
        self.batches.append(len(texts))
        vectors = await self._inner.embed(texts)
        return [vector[:-1] for vector in vectors]


class FailingSummarizer(SummarizePattern):
    """Summarizer that fails for one article title."""

    def __init__(self, router: RouteRouter, failing_title: str) -> None:
        super().__init__(router)
        self.failing_title = failing_title

    async def run(
        self,
        title: str,
        text: str,
        *,
        max_words: int | None = None,
        **overrides: Any,
    ) -> str:
        """Fail for the configured title, otherwise summarize normally."""
        if title == self.failing_title:
            raise RuntimeError("provider exploded")
        return await super().run(title, text, max_words=max_words, **overrides)


class FailingStore(InMemoryGraphStore):
    """Store that fails every article write."""

    async def upsert_articles(self, articles: Sequence[Article], chunks: Sequence[Any]) -> None:
        """Raise instead of writing."""
        raise RuntimeError("database offline")


def extraction_json(
    entities: Sequence[dict[str, Any]] = (),
    relations: Sequence[dict[str, Any]] = (),
    mentions: Sequence[dict[str, Any]] = (),
) -> str:
    """Serialize a graph extraction payload."""
    return json.dumps(
        {"entities": list(entities), "relations": list(relations), "mentions": list(mentions)}
    )


def scripted_responses(
    *,
    summary: str = "Summary text.",
    classification: str = '{"labels": ["technology"], "primary": "technology"}',
    extraction: str | None = None,
    count: int = 1,
) -> dict[str, list[str]]:
    """Build one response per default route."""
    payload = (
        extraction
        if extraction is not None
        else extraction_json(
            entities=[
                {"name": "OpenAI", "type": "organization"},
                {"name": "GPT-5", "type": "product"},
            ],
            relations=[
                {"source": "OpenAI", "target": "GPT-5", "type": "released", "confidence": 0.9}
            ],
            mentions=[{"chunk_id": chunk_id_for("a1", 0, "placeholder"), "entity": "OpenAI"}],
        )
    )
    return {
        "summarize": [summary] * count,
        "classify": [classification] * count,
        "extract_graph": [payload] * count,
    }


def make_pipeline(
    router: RouteRouter,
    *,
    store: Any = None,
    embedder: CountingEmbedding | None = None,
    summarizer: SummarizePattern | None = None,
    **kwargs: Any,
) -> tuple[EnrichmentPipeline, Any, CountingEmbedding]:
    """Build a pipeline with deterministic collaborators."""
    active_store = store if store is not None else InMemoryGraphStore()
    active_embedder = embedder if embedder is not None else CountingEmbedding()
    pipeline = EnrichmentPipeline(
        router,
        active_store,
        active_embedder,
        summarizer=summarizer,
        **kwargs,
    )
    return pipeline, active_store, active_embedder


class TestArticleDocument(unittest.TestCase):
    """Document selection from normalized and raw fields."""

    def test_prefers_raw_content(self) -> None:
        article = make_article(raw={"content": "Raw body"})
        self.assertEqual(article_document(article), "Article title\n\nRaw body")

    def test_falls_back_to_summary(self) -> None:
        self.assertEqual(
            article_document(make_article(summary="Summary body")),
            "Article title\n\nSummary body",
        )

    def test_title_only_when_no_body(self) -> None:
        self.assertEqual(article_document(make_article(summary=None)), "Article title")

    def test_blank_raw_field_is_skipped(self) -> None:
        article = make_article(summary="Summary body", raw={"content": "   "})
        self.assertEqual(article_document(article), "Article title\n\nSummary body")


class TestChunkText(unittest.TestCase):
    """Deterministic chunking, overlap, and validation."""

    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(chunk_text("   "), [])

    def test_short_text_is_one_chunk(self) -> None:
        self.assertEqual(
            chunk_text("hello world", chunk_tokens=10, overlap_tokens=0),
            ["hello world"],
        )

    def test_long_text_splits_with_overlap(self) -> None:
        words = " ".join(f"word{index}" for index in range(60))
        chunks = chunk_text(words, chunk_tokens=10, overlap_tokens=3, chars_per_token=4)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].split()[0], "word0")
        self.assertTrue(any("word" in chunk for chunk in chunks[1:]))

    def test_chunk_ids_are_deterministic(self) -> None:
        text = "deterministic text"
        first = chunk_text(text, chunk_tokens=10, overlap_tokens=0)
        second = chunk_text(text, chunk_tokens=10, overlap_tokens=0)
        self.assertEqual(first, second)
        self.assertEqual(
            chunk_id_for("a1", 0, first[0]),
            chunk_id_for("a1", 0, second[0]),
        )

    def test_invalid_sizes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_tokens=0)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_tokens=5, overlap_tokens=5)
        with self.assertRaises(ValueError):
            chunk_text("text", overlap_tokens=-1)


class TestPipelineSuccess(unittest.IsolatedAsyncioTestCase):
    """Happy-path ingestion and persisted graph state."""

    async def test_process_persists_article_chunks_and_graph(self) -> None:
        article = make_article()
        router = RouteRouter(
            scripted_responses(
                extraction=extraction_json(
                    entities=[
                        {"name": "OpenAI", "type": "organization"},
                        {"name": "GPT-5", "type": "product"},
                    ],
                    relations=[
                        {
                            "source": "OpenAI",
                            "target": "GPT-5",
                            "type": "released",
                            "confidence": 2.0,
                        }
                    ],
                    mentions=[{"chunk_id": "unknown", "entity": "OpenAI"}],
                )
            )
        )
        pipeline, store, embedder = make_pipeline(router)

        report = await pipeline.process([article])

        self.assertEqual(report.succeeded_count, 1)
        self.assertEqual(report.failed_count, 0)
        self.assertEqual(report.skipped_count, 0)
        enrichment = report.enrichments[0]
        self.assertEqual(enrichment.article_id, "a1")
        self.assertEqual(enrichment.summary, "Summary text.")
        self.assertEqual(enrichment.classification.primary, "technology")
        self.assertEqual(enrichment.entity_count, 2)
        self.assertEqual(enrichment.relation_count, 1)
        self.assertEqual(len(enrichment.chunk_ids), 1)
        self.assertEqual(embedder.batches, [1])

        query_vector = (await embedder.embed([article_document(article)]))[0]
        results = await store.vector_search(query_vector, 5)
        self.assertEqual([chunk.id for chunk, _ in results], list(enrichment.chunk_ids))
        context = await store.expand([enrichment.chunk_ids[0]])
        self.assertEqual({entity.name for entity in context.entities}, {"OpenAI", "GPT-5"})
        self.assertEqual(context.relations[0].confidence, 1.0)

    async def test_empty_input_returns_empty_report(self) -> None:
        router = RouteRouter({})
        pipeline, _, embedder = make_pipeline(router)
        self.assertEqual(await pipeline.process([]), PipelineReport())
        self.assertEqual(embedder.batches, [])

    async def test_duplicate_identifiers_keep_first(self) -> None:
        router = RouteRouter(scripted_responses())
        pipeline, _, _ = make_pipeline(router)
        report = await pipeline.process([make_article(title="First"), make_article(title="Second")])
        self.assertEqual(report.total_count, 1)
        self.assertEqual(report.enrichments[0].article_id, "a1")


class TestPipelineIdempotency(unittest.IsolatedAsyncioTestCase):
    """Content-hash skipping and changed-article reprocessing."""

    async def test_rerun_writes_nothing_new(self) -> None:
        router = RouteRouter(scripted_responses(count=2))
        pipeline, store, embedder = make_pipeline(router)
        articles = [make_article("a1"), make_article("a2")]

        first = await pipeline.process(articles)
        calls_after_first = len(router.calls)
        batches_after_first = list(embedder.batches)
        chunks_after_first = [chunk.id for chunk, _ in await store.vector_search((1.0,) * 8, 10)]

        second = await pipeline.process(articles)

        self.assertEqual(first.succeeded_count, 2)
        self.assertEqual(second.skipped_count, 2)
        self.assertEqual(second.succeeded_count, 0)
        self.assertEqual(len(router.calls), calls_after_first)
        self.assertEqual(embedder.batches, batches_after_first)
        self.assertEqual(
            [chunk.id for chunk, _ in await store.vector_search((1.0,) * 8, 10)],
            chunks_after_first,
        )

    async def test_changed_article_is_reprocessed(self) -> None:
        router = RouteRouter(
            {
                "summarize": ["one", "two"],
                "classify": [
                    '{"labels": ["technology"]}',
                    '{"labels": ["policy"], "primary": "policy"}',
                ],
                "extract_graph": ['{"entities": [], "relations": []}'] * 2,
            }
        )
        pipeline, _, embedder = make_pipeline(router)
        original = make_article(title="Original")
        await pipeline.process([original])

        report = await pipeline.process([make_article(title="Changed")])

        self.assertEqual(report.succeeded_count, 1)
        self.assertEqual(report.enrichments[0].classification.primary, "policy")
        self.assertEqual(len(router.calls), 6)
        self.assertEqual(embedder.batches, [1, 1])


class TestPipelineFailures(unittest.IsolatedAsyncioTestCase):
    """Per-article isolation and stage reporting."""

    async def test_one_article_failure_does_not_block_others(self) -> None:
        router = RouteRouter(
            {
                "summarize": ["good summary", "bad summary"],
                "classify": [
                    '{"labels": ["technology"]}',
                    '{"labels": ["technology"]}',
                ],
                "extract_graph": [
                    '{"entities": [], "relations": []}',
                    '{"entities": [], "relations": []}',
                ],
            }
        )
        summarizer = FailingSummarizer(router, failing_title="Broken")
        pipeline, store, _ = make_pipeline(router, summarizer=summarizer)

        report = await pipeline.process([make_article("bad", title="Broken"), make_article("good")])

        self.assertEqual([item.article_id for item in report.enrichments], ["good"])
        self.assertEqual(tuple(item.article_id for item in report.failures), ("bad",))
        failure = report.failures[0]
        self.assertEqual(failure.stage, "summarize")
        self.assertIn("provider exploded", failure.error)
        self.assertEqual(await store.article_hashes(["bad"]), {})

    async def test_store_failure_is_isolated(self) -> None:
        router = RouteRouter(scripted_responses())
        pipeline, _, _ = make_pipeline(router, store=FailingStore())

        report = await pipeline.process([make_article()])

        self.assertEqual(report.succeeded_count, 0)
        self.assertEqual(report.failures[0].stage, "store")
        self.assertIn("database offline", report.failures[0].error)

    async def test_wrong_embedding_count_fails(self) -> None:
        router = RouteRouter(scripted_responses())
        pipeline, _, _ = make_pipeline(router, embedder=WrongCountEmbedding())

        report = await pipeline.process([make_article()])

        self.assertEqual(report.failures[0].stage, "embed")
        self.assertIn("vectors", report.failures[0].error)

    async def test_wrong_embedding_dimension_fails(self) -> None:
        router = RouteRouter(scripted_responses())
        pipeline, _, _ = make_pipeline(router, embedder=WrongDimEmbedding())

        report = await pipeline.process([make_article()])

        self.assertEqual(report.failures[0].stage, "embed")
        self.assertIn("dimension", report.failures[0].error)

    async def test_invalid_construction(self) -> None:
        router = RouteRouter({})
        store = InMemoryGraphStore()
        with self.assertRaises(ValueError):
            EnrichmentPipeline(router, store, CountingEmbedding(), chunk_tokens=0)
        with self.assertRaises(ValueError):
            EnrichmentPipeline(
                router, store, CountingEmbedding(), chunk_tokens=10, chunk_overlap=10
            )
        with self.assertRaises(ValueError):
            EnrichmentPipeline(router, store, CountingEmbedding(), embedding_batch_size=0)
        with self.assertRaises(ValueError):
            EnrichmentPipeline(router, store, CountingEmbedding(), labels=())


class TestPipelineBatching(unittest.IsolatedAsyncioTestCase):
    """Embedding and write batches stay inside their configured limits."""

    async def test_embedding_batches_are_bounded(self) -> None:
        long_body = " ".join(f"token{index}" for index in range(400))
        article = make_article(raw={"content": long_body})
        router = RouteRouter(
            {
                "summarize": ["summary"],
                "classify": ['{"labels": ["technology"]}'],
                "extract_graph": ['{"entities": [], "relations": []}'],
            }
        )
        pipeline, _, embedder = make_pipeline(
            router,
            chunk_tokens=20,
            chunk_overlap=5,
            embedding_batch_size=2,
        )

        report = await pipeline.process([article])

        self.assertEqual(report.succeeded_count, 1)
        self.assertGreater(len(embedder.batches), 1)
        self.assertTrue(all(size <= 2 for size in embedder.batches))
        self.assertEqual(sum(embedder.batches), len(report.enrichments[0].chunk_ids))

    async def test_concurrency_override_is_honoured(self) -> None:
        router = RouteRouter(
            {
                "summarize": ["s"] * 4,
                "classify": ['{"labels": ["technology"]}'] * 4,
                "extract_graph": ['{"entities": [], "relations": []}'] * 4,
            }
        )
        pipeline, _, _ = make_pipeline(router, max_concurrency=1)
        report = await pipeline.process(
            [make_article(f"a{index}") for index in range(4)],
            concurrency=2,
        )
        self.assertEqual(report.succeeded_count, 4)


if __name__ == "__main__":
    unittest.main()
