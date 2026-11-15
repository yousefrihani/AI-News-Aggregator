"""Unit tests for the Neo4j adapter, driven by a fake async driver.

No test in this module opens a socket. The integration gate lives in
``test_neo4j_integration.py`` and is skipped unless explicitly enabled.
"""

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from Articles.graph import neo4j as neo4j_module
from Articles.graph.base import Chunk, Entity, GraphContext, Mention, Relation
from Articles.graph.errors import (
    GraphConfigError,
    GraphConnectionError,
    GraphQueryError,
)
from Articles.graph.neo4j import (
    Neo4jGraphStore,
    article_row,
    batch_iterable,
    chunk_from_properties,
    chunk_row,
    chunk_score_from_record,
    entity_from_properties,
    entity_row,
    mention_row,
    relation_from_properties,
    relation_row,
)
from Articles.Models import Article

PUBLISHED = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
PASSWORD = "unit-test-password"


class FakeResult:
    """Async-iterable result used by the fake session."""

    def __init__(self, records: list[Any]) -> None:
        self._records = records

    async def __aiter__(self) -> Any:
        for record in self._records:
            yield record


class FakeSession:
    """Minimal async Neo4j session."""

    def __init__(self, driver: "FakeDriver") -> None:
        self.driver = driver

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def run(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> FakeResult:
        """Record the query and return the next configured response."""
        self.driver.calls.append((query, parameters or {}))
        if self.driver.error is not None:
            raise self.driver.error
        if self.driver.handler is not None:
            return FakeResult(self.driver.handler(query, parameters or {}))
        if self.driver.responses:
            return FakeResult(self.driver.responses.pop(0))
        return FakeResult([])


class FakeDriver:
    """Minimal async Neo4j driver with injectable responses or failures."""

    def __init__(
        self,
        responses: list[list[Any]] | None = None,
        *,
        handler: Any = None,
        error: BaseException | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.handler = handler
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.databases: list[str | None] = []
        self.closed = False

    def session(self, database: str | None = None) -> FakeSession:
        """Create a session bound to a database."""
        self.databases.append(database)
        return FakeSession(self)

    async def close(self) -> None:
        """Mark the driver as closed."""
        self.closed = True


def make_store(driver: Any = None, dim: int = 4) -> Neo4jGraphStore:
    """Build a store bound to a fake (or missing) driver."""
    return Neo4jGraphStore(
        "bolt://localhost:7687",
        "neo4j",
        PASSWORD,
        "neo4j",
        driver=driver,
        dim=dim,
    )


def make_article() -> Article:
    """Build an article for adapter tests."""
    return Article(
        id="a1",
        title="Graph title",
        url="https://example.com/a1",
        source="NewsAPI",
        published_at=PUBLISHED,
        summary="Summary",
        topics=("technology", "ai"),
    )


def chunk_props(chunk_id: str = "c1", embedding: list[float] | None = None) -> dict[str, Any]:
    """Build chunk properties as returned from Neo4j."""
    return {
        "id": chunk_id,
        "article_id": "a1",
        "ordinal": 0,
        "text": f"text-{chunk_id}",
        "embedding": embedding,
    }


class TestConstruction(unittest.IsolatedAsyncioTestCase):
    """Constructor validation and offline construction."""

    async def test_valid_construction_does_not_create_a_driver(self) -> None:
        with patch(
            "Articles.graph.neo4j._create_async_driver",
            side_effect=AssertionError("must not connect"),
        ) as create:
            store = make_store()
            create.assert_not_called()
        self.assertIsNone(store._driver)
        self.assertEqual(store.dim, 4)

    async def test_invalid_settings(self) -> None:
        cases = [
            ("", "neo4j", PASSWORD, "neo4j"),
            ("bolt://x", "", PASSWORD, "neo4j"),
            ("bolt://x", "neo4j", "", "neo4j"),
            ("bolt://x", "neo4j", PASSWORD, ""),
        ]
        for uri, user, password, database in cases:
            with (
                self.subTest(uri=uri, user=user, database=database),
                self.assertRaises(GraphConfigError),
            ):
                Neo4jGraphStore(uri, user, password, database)
        for dim in (0, -1, 2.5, True):
            with self.subTest(dim=dim), self.assertRaises(GraphConfigError):
                make_store(dim=dim)  # type: ignore[arg-type]

    async def test_driver_factory_import_failure(self) -> None:
        with (
            patch(
                "Articles.graph.neo4j.importlib.import_module",
                side_effect=ImportError("missing"),
            ),
            self.assertRaises(GraphConfigError) as raised,
        ):
            neo4j_module._create_async_driver("bolt://x", "neo4j", PASSWORD)
        self.assertIn("pip install ai-news-aggregator[graph]", str(raised.exception))

    async def test_driver_factory_without_expected_attribute(self) -> None:
        with (
            patch(
                "Articles.graph.neo4j.importlib.import_module",
                return_value=SimpleNamespace(),
            ),
            self.assertRaises(GraphConfigError),
        ):
            neo4j_module._create_async_driver("bolt://x", "neo4j", PASSWORD)

    async def test_driver_factory_maps_connection_errors(self) -> None:
        class ServiceUnavailableError(Exception):
            pass

        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise ServiceUnavailableError("down")

        module = SimpleNamespace(AsyncGraphDatabase=SimpleNamespace(driver=fail))
        with (
            patch(
                "Articles.graph.neo4j.importlib.import_module",
                return_value=module,
            ),
            self.assertRaises(GraphConnectionError) as raised,
        ):
            neo4j_module._create_async_driver("bolt://x", "neo4j", PASSWORD)
        self.assertNotIn(PASSWORD, str(raised.exception))

    async def test_driver_factory_success(self) -> None:
        marker = object()
        module = SimpleNamespace(
            AsyncGraphDatabase=SimpleNamespace(driver=lambda *_a, **_k: marker)
        )
        with patch(
            "Articles.graph.neo4j.importlib.import_module",
            return_value=module,
        ):
            self.assertIs(
                neo4j_module._create_async_driver("bolt://x", "neo4j", PASSWORD),
                marker,
            )


class TestLifecycle(unittest.IsolatedAsyncioTestCase):
    """Schema initialization, health checks, and shutdown."""

    async def test_initialize_applies_every_statement(self) -> None:
        driver = FakeDriver(responses=[[] for _ in range(10)])
        store = make_store(driver)
        await store.initialize()
        self.assertEqual(len(driver.calls), 7)
        self.assertTrue(all(query.startswith("CREATE ") for query, _ in driver.calls))
        self.assertEqual(driver.databases, ["neo4j"] * 7)

    async def test_healthcheck(self) -> None:
        driver = FakeDriver(responses=[[{"ok": 1}]])
        self.assertTrue(await make_store(driver).healthcheck())
        self.assertEqual(driver.calls[0][0], "RETURN 1 AS ok")

    async def test_healthcheck_returns_false_on_failure(self) -> None:
        class ServiceUnavailableError(Exception):
            pass

        driver = FakeDriver(error=ServiceUnavailableError("down"))
        self.assertFalse(await make_store(driver).healthcheck())

    async def test_aclose(self) -> None:
        driver = FakeDriver()
        store = make_store(driver)
        await store.aclose()
        self.assertTrue(driver.closed)
        self.assertIsNone(store._driver)
        await store.aclose()

    async def test_aclose_maps_failures(self) -> None:
        class ServiceUnavailableError(Exception):
            pass

        class CloseErrorDriver(FakeDriver):
            async def close(self) -> None:
                raise ServiceUnavailableError("close failed")

        store = make_store(CloseErrorDriver())
        with self.assertRaises(GraphConnectionError):
            await store.aclose()

    async def test_errors_are_mapped_and_redacted(self) -> None:
        class ServiceUnavailableError(Exception):
            pass

        class CypherSyntaxError(Exception):
            pass

        connection_store = make_store(FakeDriver(error=ServiceUnavailableError("down")))
        with self.assertRaises(GraphConnectionError) as raised:
            await connection_store.initialize()
        self.assertNotIn(PASSWORD, str(raised.exception))

        query_store = make_store(FakeDriver(error=CypherSyntaxError("bad syntax")))
        with self.assertRaises(GraphQueryError):
            await query_store.upsert_entities(
                [Entity(id="e1", name="Ada", type="person")],
                [],
            )


class TestUpserts(unittest.IsolatedAsyncioTestCase):
    """Batched article, chunk, entity, relation, and mention writes."""

    async def test_upsert_articles_batches_and_serializes(self) -> None:
        driver = FakeDriver(responses=[[] for _ in range(10)])
        store = make_store(driver)
        article = make_article()
        chunks = [
            Chunk(id="c1", article_id="a1", ordinal=0, text="one", embedding=(1.0, 0.0)),
            Chunk(id="c2", article_id="a1", ordinal=1, text="two", embedding=None),
        ]
        await store.upsert_articles([article], chunks)

        self.assertEqual(len(driver.calls), 5)
        queries = [query for query, _ in driver.calls]
        self.assertIn("MERGE (a:Article", queries[0])
        self.assertIn("UNWIND row.topics", queries[1])
        self.assertIn("Source", queries[2])
        self.assertIn("DETACH DELETE", queries[3])
        self.assertIn("MERGE (c:Chunk", queries[4])

        row = driver.calls[0][1]["rows"][0]
        self.assertEqual(row["id"], "a1")
        self.assertEqual(row["published_at"], PUBLISHED.isoformat())
        self.assertEqual(row["topics"], ["technology", "ai"])
        self.assertEqual(row["content_hash"], article_row(article)["content_hash"])

        stale = driver.calls[3][1]["rows"][0]
        self.assertEqual(stale, {"article_id": "a1", "chunk_ids": ["c1", "c2"]})
        self.assertEqual(driver.calls[4][1]["rows"][0]["embedding"], [1.0, 0.0])
        self.assertIsNone(driver.calls[4][1]["rows"][1]["embedding"])

    async def test_empty_upsert_does_nothing(self) -> None:
        driver = FakeDriver()
        await make_store(driver).upsert_articles([], [])
        self.assertEqual(driver.calls, [])

    async def test_chunks_without_articles(self) -> None:
        driver = FakeDriver(responses=[[]])
        await make_store(driver).upsert_articles(
            [],
            [Chunk(id="c1", article_id="a1", ordinal=0, text="one")],
        )
        self.assertEqual(len(driver.calls), 1)
        self.assertIn("MERGE (c:Chunk", driver.calls[0][0])

    async def test_upsert_entities(self) -> None:
        driver = FakeDriver(responses=[[] for _ in range(5)])
        store = make_store(driver)
        await store.upsert_entities(
            [Entity(id="e1", name="Ada", type="person", canonical="Ada")],
            [Relation("e1", "e2", "knows", confidence=0.5, evidence_chunk_id="c1")],
            [Mention("c1", "e1")],
        )
        self.assertEqual(len(driver.calls), 3)
        self.assertIn("MERGE (e:Entity", driver.calls[0][0])
        self.assertIn("RELATED_TO", driver.calls[1][0])
        self.assertIn("MENTIONS", driver.calls[2][0])
        self.assertEqual(driver.calls[1][1]["rows"][0]["confidence"], 0.5)
        self.assertEqual(driver.calls[2][1]["rows"][0], {"chunk_id": "c1", "entity_id": "e1"})

    async def test_empty_entity_upsert_does_nothing(self) -> None:
        driver = FakeDriver()
        await make_store(driver).upsert_entities([], [])
        self.assertEqual(driver.calls, [])


class TestVectorSearch(unittest.IsolatedAsyncioTestCase):
    """Vector-index query parameters, parsing, and validation."""

    async def test_search_orders_and_omits_embeddings(self) -> None:
        driver = FakeDriver(
            responses=[
                [
                    {"chunk": chunk_props("c2", [0.0, 1.0]), "score": 0.2},
                    {"chunk": chunk_props("c1", [1.0, 0.0]), "score": 0.9},
                ]
            ]
        )
        results = await make_store(driver).vector_search((1.0, 0.0), 5)
        self.assertEqual([chunk.id for chunk, _ in results], ["c1", "c2"])
        self.assertTrue(all(chunk.embedding is None for chunk, _ in results))
        self.assertAlmostEqual(results[0][1], 0.9)
        query, parameters = driver.calls[0]
        self.assertIn("db.index.vector.queryNodes", query)
        self.assertEqual(parameters["k"], 5)
        self.assertEqual(parameters["embedding"], [1.0, 0.0])
        self.assertIsNone(parameters["article_id"])

    async def test_filters_are_forwarded(self) -> None:
        driver = FakeDriver(responses=[[]])
        await make_store(driver).vector_search(
            (1.0, 0.0),
            1,
            filters={"article_id": "a1", "source": "NewsAPI", "topic": "ai"},
        )
        parameters = driver.calls[0][1]
        self.assertEqual(parameters["article_id"], "a1")
        self.assertEqual(parameters["source"], "NewsAPI")
        self.assertEqual(parameters["topic"], "ai")

    async def test_invalid_arguments(self) -> None:
        driver = FakeDriver()
        store = make_store(driver)
        with self.assertRaises(ValueError):
            await store.vector_search((1.0,), 0)
        with self.assertRaises(GraphConfigError):
            await store.vector_search((1.0,), 1, filters={"unsupported": "x"})
        self.assertEqual(driver.calls, [])

    async def test_malformed_result(self) -> None:
        driver = FakeDriver(responses=[[{"chunk": {"id": "c1"}}]])
        with self.assertRaises(GraphQueryError):
            await make_store(driver).vector_search((1.0,), 1)

    async def test_empty_result(self) -> None:
        driver = FakeDriver(responses=[[]])
        self.assertEqual(await make_store(driver).vector_search((1.0,), 3), [])


class TestExpand(unittest.IsolatedAsyncioTestCase):
    """Bounded expansion queries and record parsing."""

    async def test_depth_one(self) -> None:
        driver = FakeDriver(
            responses=[
                [{"chunk": chunk_props("c1")}],
                [{"entity": {"id": "e1", "name": "Ada", "type": "person"}}],
                [
                    {
                        "source_id": "e1",
                        "target_id": "e2",
                        "type": "knows",
                        "confidence": 0.7,
                        "evidence_chunk_id": "c1",
                    }
                ],
                [{"chunk": chunk_props("c1")}, {"chunk": chunk_props("c2")}],
            ]
        )
        context = await make_store(driver).expand(["c1"], depth=1, limit=10)
        self.assertEqual([chunk.id for chunk in context.chunks], ["c1", "c2"])
        self.assertEqual([entity.id for entity in context.entities], ["e1"])
        self.assertEqual(context.relations[0].confidence, 0.7)
        self.assertEqual(len(driver.calls), 4)
        self.assertEqual(driver.calls[1][1]["chunk_ids"], ["c1"])
        self.assertEqual(driver.calls[1][1]["limit"], 10)

    async def test_depth_two_performs_two_hops(self) -> None:
        driver = FakeDriver(
            responses=[
                [{"chunk": chunk_props("c1")}],
                [{"entity": {"id": "e1", "name": "Ada", "type": "person"}}],
                [],
                [{"chunk": chunk_props("c2")}],
                [{"entity": {"id": "e2", "name": "Engine", "type": "technology"}}],
                [],
                [{"chunk": chunk_props("c2")}, {"chunk": chunk_props("c3")}],
            ]
        )
        context = await make_store(driver).expand(["c1"], depth=2, limit=10)
        self.assertEqual([chunk.id for chunk in context.chunks], ["c1", "c2", "c3"])
        self.assertEqual([entity.id for entity in context.entities], ["e1", "e2"])
        self.assertEqual(len(driver.calls), 7)

    async def test_limit_truncates_initial_chunks(self) -> None:
        driver = FakeDriver(
            responses=[[{"chunk": chunk_props("c1")}, {"chunk": chunk_props("c2")}]]
        )
        context = await make_store(driver).expand(["c1", "c2"], depth=1, limit=1)
        self.assertEqual([chunk.id for chunk in context.chunks], ["c1"])
        self.assertEqual(len(driver.calls), 2)

    async def test_empty_seeds_skip_queries(self) -> None:
        driver = FakeDriver()
        self.assertEqual(await make_store(driver).expand([]), GraphContext())
        self.assertEqual(driver.calls, [])

    async def test_invalid_bounds(self) -> None:
        store = make_store(FakeDriver())
        for depth in (0, -1, 3):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                await store.expand(["c1"], depth=depth)
        with self.assertRaises(ValueError):
            await store.expand(["c1"], limit=0)

    async def test_malformed_records(self) -> None:
        driver = FakeDriver(responses=[[{"chunk": {"id": "c1"}}]])
        with self.assertRaises(GraphQueryError):
            await make_store(driver).expand(["c1"])


class TestPureHelpers(unittest.TestCase):
    """Batching, record parsing, and row serialization helpers."""

    def test_batch_iterable(self) -> None:
        self.assertEqual(list(batch_iterable([])), [])
        self.assertEqual(list(batch_iterable([1, 2, 3], size=2)), [[1, 2], [3]])
        self.assertEqual(list(batch_iterable([1, 2], size=2)), [[1, 2]])
        with self.assertRaises(ValueError):
            list(batch_iterable([1], size=0))

    def test_chunk_from_properties(self) -> None:
        chunk = chunk_from_properties(chunk_props("c1", [1.0, 2.0]))
        self.assertIsNone(chunk.embedding)
        chunk = chunk_from_properties(chunk_props("c1", [1.0, 2.0]), include_embedding=True)
        self.assertEqual(chunk.embedding, (1.0, 2.0))

        class Node:
            def __init__(self, props: dict[str, Any]) -> None:
                self._props = props

            def items(self) -> Any:
                return self._props.items()

        self.assertEqual(chunk_from_properties(Node(chunk_props("c9"))).id, "c9")

    def test_chunk_from_properties_errors(self) -> None:
        with self.assertRaises(GraphQueryError):
            chunk_from_properties({"id": "c1"})
        with self.assertRaises(GraphQueryError):
            chunk_from_properties({**chunk_props("c1"), "ordinal": "not-an-int"})
        with self.assertRaises(GraphQueryError):
            chunk_from_properties(None)

    def test_entity_from_properties(self) -> None:
        entity = entity_from_properties({"id": "e1", "name": "Ada", "type": "person"})
        self.assertIsNone(entity.canonical)
        entity = entity_from_properties(
            {"id": "e1", "name": "Ada", "type": "person", "canonical": "Ada Lovelace"}
        )
        self.assertEqual(entity.canonical, "Ada Lovelace")
        with self.assertRaises(GraphQueryError):
            entity_from_properties({"id": "e1"})

    def test_relation_from_properties(self) -> None:
        relation = relation_from_properties({"source_id": "e1", "target_id": "e2", "type": "knows"})
        self.assertEqual(relation.confidence, 1.0)
        self.assertIsNone(relation.evidence_chunk_id)
        with self.assertRaises(GraphQueryError):
            relation_from_properties({"source_id": "e1"})
        with self.assertRaises(GraphQueryError):
            relation_from_properties(
                {
                    "source_id": "e1",
                    "target_id": "e2",
                    "type": "knows",
                    "confidence": "nope",
                }
            )

    def test_chunk_score_from_record(self) -> None:
        chunk, score = chunk_score_from_record({"chunk": chunk_props("c1"), "score": 0.5})
        self.assertEqual(chunk.id, "c1")
        self.assertEqual(score, 0.5)
        with self.assertRaises(GraphQueryError):
            chunk_score_from_record({"chunk": chunk_props("c1")})

    def test_record_mapping_uses_data_method(self) -> None:
        class Record:
            def data(self) -> dict[str, Any]:
                return {"source_id": "e1", "target_id": "e2", "type": "knows"}

        relation = relation_from_properties(neo4j_module._record_mapping(Record()))
        self.assertEqual(relation.type, "knows")

    def test_row_builders(self) -> None:
        article = make_article()
        row = article_row(article)
        self.assertEqual(row["url"], article.url)
        self.assertEqual(row["topics"], ["technology", "ai"])

        chunk = Chunk(id="c1", article_id="a1", ordinal=0, text="one", embedding=(1.0,))
        self.assertEqual(chunk_row(chunk)["embedding"], [1.0])
        self.assertEqual(entity_row(Entity(id="e1", name="Ada", type="person"))["name"], "Ada")
        self.assertEqual(
            relation_row(Relation("e1", "e2", "knows"))["type"],
            "knows",
        )
        self.assertEqual(
            mention_row(Mention("c1", "e1")),
            {"chunk_id": "c1", "entity_id": "e1"},
        )

    def test_null_optional_row_fields(self) -> None:
        article = make_article()
        article = Article(
            id=article.id,
            title=article.title,
            url=article.url,
            source=article.source,
            published_at=None,
            summary=None,
            topics=(),
        )
        row = article_row(article)
        self.assertIsNone(row["published_at"])
        self.assertIsNone(row["summary"])
        self.assertEqual(row["topics"], [])


if __name__ == "__main__":
    unittest.main()
