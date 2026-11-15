"""Opt-in integration test for the Neo4j adapter.

The test runs only when ``RUN_GRAPH_TESTS=1`` and ``NEO4J_PASSWORD`` are set, so
the default offline suite never opens a socket. Start Neo4j with the Compose
``graph`` profile (or a local server) before enabling it.
"""

import contextlib
import os
import unittest
from uuid import uuid4

from Articles.graph.base import Chunk, Entity, Mention, Relation, chunk_id_for, entity_id_for
from Articles.graph.embeddings import DeterministicEmbedding
from Articles.graph.neo4j import Neo4jGraphStore
from Articles.Models import Article

RUN = os.getenv("RUN_GRAPH_TESTS") == "1"
PASSWORD = os.getenv("NEO4J_PASSWORD")
DIM = 384


@unittest.skipUnless(
    RUN and PASSWORD,
    "set RUN_GRAPH_TESTS=1 and NEO4J_PASSWORD to run Neo4j integration tests",
)
class Neo4jIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """End-to-end schema, upsert, search, and expansion against a live server."""

    async def asyncSetUp(self) -> None:
        self.run_id = f"g1-{uuid4().hex[:12]}"
        self.article_id = f"{self.run_id}-article"
        self.chunk_ids = [f"{self.run_id}-chunk-{index}" for index in range(2)]
        self.entity_ids = [f"{self.run_id}-entity-{index}" for index in range(2)]
        self.store = Neo4jGraphStore(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=PASSWORD or "",
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            dim=DIM,
        )
        await self.store.initialize()
        self.embedder = DeterministicEmbedding(dim=DIM)

    async def asyncTearDown(self) -> None:
        driver = getattr(self.store, "_driver", None)
        if driver is not None:
            identifiers = [
                self.article_id,
                *self.chunk_ids,
                *self.entity_ids,
            ]
            with contextlib.suppress(Exception):
                await driver.execute_query(
                    "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
                    {"ids": identifiers},
                    database_=os.getenv("NEO4J_DATABASE", "neo4j"),
                )
        await self.store.aclose()

    async def test_roundtrip_search_and_expand(self) -> None:
        """A live round-trip exercises batching, search, and bounded expansion."""
        article = Article(
            id=self.article_id,
            title="Live Neo4j graph article",
            url=f"https://example.com/{self.article_id}",
            source="Integration",
            published_at=None,
            summary="Integration summary",
            topics=("technology", "graph"),
        )
        texts = ["Alpha chunk about Ada", "Beta chunk about the engine"]
        vectors = await self.embedder.embed(texts)
        chunks = [
            Chunk(
                id=self.chunk_ids[index],
                article_id=self.article_id,
                ordinal=index,
                text=texts[index],
                embedding=tuple(vectors[index]),
            )
            for index in range(2)
        ]
        entities = [
            Entity(id=self.entity_ids[0], name=f"{self.run_id} Ada", type="person"),
            Entity(id=self.entity_ids[1], name=f"{self.run_id} Engine", type="technology"),
        ]
        relation = Relation(
            source_id=self.entity_ids[0],
            target_id=self.entity_ids[1],
            type="invented",
            confidence=0.9,
            evidence_chunk_id=self.chunk_ids[0],
        )
        mentions = [
            Mention(chunk_id=self.chunk_ids[0], entity_id=self.entity_ids[0]),
            Mention(chunk_id=self.chunk_ids[1], entity_id=self.entity_ids[1]),
        ]

        await self.store.upsert_articles([article], chunks)
        await self.store.upsert_entities(entities, [relation], mentions)
        self.assertTrue(await self.store.healthcheck())

        results = await self.store.vector_search(
            vectors[0],
            4,
            filters={"article_id": self.article_id, "source": "Integration"},
        )
        self.assertEqual(results[0][0].id, self.chunk_ids[0])
        self.assertIsNone(results[0][0].embedding)
        self.assertGreater(results[0][1], 0.99)

        context = await self.store.expand([self.chunk_ids[0]], depth=2, limit=10)
        self.assertIn(self.chunk_ids[0], [chunk.id for chunk in context.chunks])
        self.assertIn(self.entity_ids[0], [entity.id for entity in context.entities])
        self.assertTrue(
            any(
                relation.source_id == self.entity_ids[0]
                and relation.target_id == self.entity_ids[1]
                for relation in context.relations
            )
        )

        await self.store.upsert_articles([article], chunks[:1])
        stale = await self.store.expand([self.chunk_ids[1]], depth=1, limit=10)
        self.assertEqual(stale.chunks, ())

    async def test_reingest_is_idempotent(self) -> None:
        """Re-running the same writes does not duplicate graph data."""
        article = Article(
            id=self.article_id,
            title="Idempotent article",
            url=f"https://example.com/{self.article_id}",
            source="Integration",
            published_at=None,
            summary=None,
            topics=("graph",),
        )
        text = "Idempotent chunk"
        vector = (await self.embedder.embed([text]))[0]
        chunk = Chunk(
            id=chunk_id_for(self.article_id, 0, text),
            article_id=self.article_id,
            ordinal=0,
            text=text,
            embedding=tuple(vector),
        )
        entity = Entity(
            id=entity_id_for(f"{self.run_id} Ada", "person"),
            name=f"{self.run_id} Ada",
            type="person",
        )
        await self.store.upsert_articles([article], [chunk])
        await self.store.upsert_entities([entity], [], [Mention(chunk.id, entity.id)])
        await self.store.upsert_articles([article], [chunk])
        await self.store.upsert_entities([entity], [], [Mention(chunk.id, entity.id)])

        results = await self.store.vector_search(vector, 3, filters={"article_id": self.article_id})
        self.assertEqual([item.id for item, _ in results], [chunk.id])
        context = await self.store.expand([chunk.id], depth=1, limit=10)
        self.assertEqual([item.id for item in context.entities], [entity.id])


if __name__ == "__main__":
    unittest.main()
