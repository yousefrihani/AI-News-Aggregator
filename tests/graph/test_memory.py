"""Unit tests for the in-memory graph store."""

import unittest
from datetime import UTC, datetime

from Articles.graph.base import Chunk, Entity, GraphContext, Mention, Relation
from Articles.graph.errors import GraphConfigError
from Articles.graph.memory import InMemoryGraphStore
from Articles.Models import Article

PUBLISHED = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)


def make_article(
    article_id: str = "a1",
    *,
    source: str = "NewsAPI",
    topics: tuple[str, ...] = ("technology",),
    title: str = "Title",
) -> Article:
    """Build an article for graph tests."""
    return Article(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        source=source,
        published_at=PUBLISHED,
        summary="Summary",
        topics=topics,
    )


def make_chunk(
    chunk_id: str,
    *,
    article_id: str = "a1",
    ordinal: int = 0,
    text: str | None = None,
    embedding: tuple[float, ...] | None = None,
) -> Chunk:
    """Build a chunk for graph tests."""
    return Chunk(
        id=chunk_id,
        article_id=article_id,
        ordinal=ordinal,
        text=text or f"text-{chunk_id}",
        embedding=embedding,
    )


def ids(context: GraphContext) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """Flatten a context into comparable identifier lists."""
    return (
        [chunk.id for chunk in context.chunks],
        [entity.id for entity in context.entities],
        [(relation.source_id, relation.target_id, relation.type) for relation in context.relations],
    )


class TestArticleUpsert(unittest.IsolatedAsyncioTestCase):
    """Article merging, exact chunk replacement, and idempotency."""

    async def test_upsert_is_idempotent(self) -> None:
        store = InMemoryGraphStore()
        article = make_article()
        chunks = [
            make_chunk("c1", embedding=(1.0, 0.0)),
            make_chunk("c2", ordinal=1, embedding=(0.0, 1.0)),
        ]
        await store.upsert_articles([article], chunks)
        first = await store.vector_search((1.0, 0.0), 10)
        await store.upsert_articles([article], chunks)
        second = await store.vector_search((1.0, 0.0), 10)
        self.assertEqual(first, second)
        self.assertEqual([chunk.id for chunk, _ in second], ["c1", "c2"])

    async def test_reingest_removes_stale_chunks_and_mentions(self) -> None:
        store = InMemoryGraphStore()
        await store.upsert_articles(
            [make_article(source="old")],
            [
                make_chunk("c1", embedding=(1.0, 0.0)),
                make_chunk("c2", ordinal=1, embedding=(0.0, 1.0)),
            ],
        )
        entity = Entity(id="e1", name="Ada", type="person")
        await store.upsert_entities([entity], [], [Mention(chunk_id="c2", entity_id="e1")])

        await store.upsert_articles(
            [make_article(source="new")],
            [make_chunk("c1", embedding=(1.0, 0.0))],
        )

        results = await store.vector_search((1.0, 0.0), 10)
        self.assertEqual([chunk.id for chunk, _ in results], ["c1"])
        self.assertEqual((await store.expand(["c2"])).chunks, ())
        self.assertEqual((await store.expand(["c1"])).entities, ())
        self.assertEqual(
            [
                chunk.id
                for chunk, _ in await store.vector_search((1.0, 0.0), 10, filters={"source": "new"})
            ],
            ["c1"],
        )
        self.assertEqual(
            await store.vector_search((1.0, 0.0), 10, filters={"source": "old"}),
            [],
        )

    async def test_reingesting_one_article_leaves_others_untouched(self) -> None:
        store = InMemoryGraphStore()
        await store.upsert_articles(
            [make_article("a1"), make_article("a2")],
            [make_chunk("c1", article_id="a1"), make_chunk("c2", article_id="a2")],
        )
        await store.upsert_articles([make_article("a1", title="Updated")], [])
        self.assertEqual((await store.expand(["c2"])).chunks[0].id, "c2")
        self.assertEqual((await store.expand(["c1"])).chunks, ())

    async def test_article_fields_merge(self) -> None:
        store = InMemoryGraphStore()
        chunk = make_chunk("c1", embedding=(1.0, 0.0))
        await store.upsert_articles([make_article(source="old")], [chunk])
        await store.upsert_articles([make_article(source="new")], [chunk])
        self.assertEqual(
            len(await store.vector_search((1.0, 0.0), 1, filters={"source": "new"})),
            1,
        )
        self.assertEqual(
            len(await store.vector_search((1.0, 0.0), 1, filters={"source": "old"})),
            0,
        )


class TestEntityUpsert(unittest.IsolatedAsyncioTestCase):
    """Entity merge, latest-relation selection, and mention union."""

    async def test_entities_merge_by_id(self) -> None:
        store = InMemoryGraphStore()
        chunk = make_chunk("c1")
        await store.upsert_articles([make_article()], [chunk])
        await store.upsert_entities([Entity(id="e1", name="Old", type="person")], [])
        await store.upsert_entities(
            [Entity(id="e1", name="New", type="person")],
            [],
            [Mention("c1", "e1")],
        )
        context = await store.expand(["c1"])
        self.assertEqual([entity.name for entity in context.entities], ["New"])

    async def test_relations_keep_latest_confidence_and_evidence(self) -> None:
        store = InMemoryGraphStore()
        article = make_article()
        await store.upsert_articles([article], [make_chunk("c1"), make_chunk("c2")])
        entities = [
            Entity(id="e1", name="Ada", type="person"),
            Entity(id="e2", name="Analytical Engine", type="technology"),
        ]
        await store.upsert_entities(
            entities,
            [Relation("e1", "e2", "invented", confidence=0.4, evidence_chunk_id="c1")],
            [Mention("c1", "e1")],
        )
        await store.upsert_entities(
            [],
            [Relation("e1", "e2", "invented", confidence=0.9, evidence_chunk_id="c2")],
            [Mention("c1", "e1"), Mention("c1", "e2")],
        )
        context = await store.expand(["c1"])
        self.assertEqual(len(context.relations), 1)
        self.assertEqual(context.relations[0].confidence, 0.9)
        self.assertEqual(context.relations[0].evidence_chunk_id, "c2")
        self.assertEqual({entity.id for entity in context.entities}, {"e1", "e2"})


class TestVectorSearch(unittest.IsolatedAsyncioTestCase):
    """Cosine ranking, filtering, and malformed-vector handling."""

    async def asyncSetUp(self) -> None:
        self.store = InMemoryGraphStore()
        await self.store.upsert_articles(
            [
                make_article("a1", source="NewsAPI", topics=("technology", "ai")),
                make_article("a2", source="Other", topics=("science",)),
            ],
            [
                make_chunk("c1", article_id="a1", embedding=(1.0, 0.0)),
                make_chunk("c2", article_id="a1", embedding=(0.0, 1.0)),
                make_chunk("c3", article_id="a1", embedding=(1.0, 1.0)),
                make_chunk("c4", article_id="a2", embedding=(1.0, 0.0)),
            ],
        )

    async def test_ranking_is_by_descending_score(self) -> None:
        results = await self.store.vector_search((1.0, 0.0), 10)
        self.assertEqual([chunk.id for chunk, _ in results], ["c1", "c4", "c3", "c2"])
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertAlmostEqual(results[1][1], 1.0)
        self.assertAlmostEqual(results[2][1], 2**-0.5)
        self.assertAlmostEqual(results[3][1], 0.0)

    async def test_ties_are_broken_by_chunk_id(self) -> None:
        store = InMemoryGraphStore()
        await store.upsert_articles(
            [make_article()],
            [
                make_chunk("b", embedding=(1.0, 0.0)),
                make_chunk("a", embedding=(1.0, 0.0)),
            ],
        )
        results = await store.vector_search((1.0, 0.0), 10)
        self.assertEqual([chunk.id for chunk, _ in results], ["a", "b"])

    async def test_filters(self) -> None:
        by_article = await self.store.vector_search((1.0, 0.0), 10, filters={"article_id": "a1"})
        self.assertEqual({chunk.id for chunk, _ in by_article}, {"c1", "c2", "c3"})
        by_source = await self.store.vector_search((1.0, 0.0), 10, filters={"source": "Other"})
        self.assertEqual([chunk.id for chunk, _ in by_source], ["c4"])
        by_topic = await self.store.vector_search((1.0, 0.0), 10, filters={"topic": "science"})
        self.assertEqual([chunk.id for chunk, _ in by_topic], ["c4"])
        combined = await self.store.vector_search(
            (1.0, 0.0),
            10,
            filters={"source": "NewsAPI", "topic": "science"},
        )
        self.assertEqual(combined, [])

    async def test_unknown_filter_rejected(self) -> None:
        with self.assertRaises(GraphConfigError):
            await self.store.vector_search((1.0, 0.0), 1, filters={"unknown": "x"})

    async def test_k_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            await self.store.vector_search((1.0, 0.0), 0)
        with self.assertRaises(ValueError):
            await self.store.vector_search((1.0, 0.0), -3)

    async def test_k_limits_results(self) -> None:
        self.assertEqual(len(await self.store.vector_search((1.0, 0.0), 2)), 2)

    async def test_skips_missing_and_mismatched_vectors(self) -> None:
        store = InMemoryGraphStore()
        await store.upsert_articles(
            [make_article()],
            [
                make_chunk("none"),
                make_chunk("short", embedding=(1.0,)),
                make_chunk("zero", embedding=(0.0, 0.0)),
                make_chunk("ok", embedding=(1.0, 0.0)),
            ],
        )
        results = await store.vector_search((1.0, 0.0), 10)
        self.assertEqual([chunk.id for chunk, _ in results], ["ok", "zero"])
        self.assertEqual(results[1][1], 0.0)

    async def test_zero_query_vector_scores_zero(self) -> None:
        results = await self.store.vector_search((0.0, 0.0), 10)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(score == 0.0 for _, score in results))
        self.assertEqual(
            [chunk.id for chunk, _ in results],
            sorted(chunk.id for chunk, _ in results),
        )


class TestExpand(unittest.IsolatedAsyncioTestCase):
    """Bounded, deterministic graph expansion."""

    async def asyncSetUp(self) -> None:
        self.store = InMemoryGraphStore()
        await self.store.upsert_articles(
            [make_article()],
            [
                make_chunk("c1"),
                make_chunk("c2", ordinal=1),
                make_chunk("c3", ordinal=2),
            ],
        )
        await self.store.upsert_entities(
            [
                Entity(id="e1", name="Ada", type="person"),
                Entity(id="e2", name="Engine", type="technology"),
            ],
            [Relation("e1", "e2", "invented", confidence=0.8, evidence_chunk_id="c1")],
            [Mention("c1", "e1"), Mention("c2", "e1"), Mention("c2", "e2"), Mention("c3", "e2")],
        )

    async def test_depth_one(self) -> None:
        context = await self.store.expand(["c1"], depth=1)
        self.assertEqual(
            ids(context),
            (["c1", "c2"], ["e1"], [("e1", "e2", "invented")]),
        )

    async def test_depth_two_reaches_further_chunks(self) -> None:
        context = await self.store.expand(["c1"], depth=2)
        self.assertEqual(
            ids(context),
            (
                ["c1", "c2", "c3"],
                ["e1", "e2"],
                [("e1", "e2", "invented")],
            ),
        )

    async def test_limit_bounds_every_collection(self) -> None:
        context = await self.store.expand(["c1"], depth=2, limit=1)
        self.assertEqual(
            ids(context),
            (["c1"], ["e1"], [("e1", "e2", "invented")]),
        )
        context = await self.store.expand(["c1"], depth=2, limit=2)
        self.assertEqual(len(context.chunks), 2)
        self.assertEqual(len(context.entities), 2)
        context = await self.store.expand(["c1", "c2"], depth=2, limit=2)
        self.assertEqual(len(context.chunks), 2)
        self.assertEqual(len(context.entities), 2)

    async def test_seed_order_is_preserved_and_duplicates_removed(self) -> None:
        context = await self.store.expand(["c3", "c1", "c3"], depth=1)
        self.assertEqual([chunk.id for chunk in context.chunks[:2]], ["c3", "c1"])

    async def test_missing_and_empty_seeds(self) -> None:
        self.assertEqual(await self.store.expand([]), GraphContext())
        self.assertEqual(await self.store.expand(["missing"]), GraphContext())
        self.assertEqual(
            ids(await self.store.expand(["missing", "c1"], depth=1))[0],
            ["c1", "c2"],
        )

    async def test_expansion_is_deterministic(self) -> None:
        first = await self.store.expand(["c1"], depth=2)
        second = await self.store.expand(["c1"], depth=2)
        self.assertEqual(first, second)

    async def test_invalid_bounds(self) -> None:
        for depth in (0, -1):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                await self.store.expand(["c1"], depth=depth)
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                await self.store.expand(["c1"], limit=limit)

    async def test_healthcheck_and_aclose(self) -> None:
        self.assertTrue(await self.store.healthcheck())
        await self.store.aclose()
        self.assertTrue(await self.store.healthcheck())


if __name__ == "__main__":
    unittest.main()
