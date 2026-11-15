"""Unit tests for graph domain models, validation, and deterministic helpers."""

import unittest
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

from Articles.graph.base import (
    Chunk,
    Entity,
    GraphContext,
    Mention,
    Relation,
    article_content_hash,
    chunk_id_for,
    entity_id_for,
    relation_key,
    validate_filters,
)
from Articles.graph.errors import GraphConfigError
from Articles.Models import Article

PUBLISHED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def make_article(**overrides: object) -> Article:
    """Build an article with sensible defaults for tests."""
    values: dict[str, object] = {
        "id": "article-1",
        "title": "Graph RAG arrives",
        "url": "https://example.com/graph",
        "source": "Example News",
        "published_at": PUBLISHED,
        "summary": "A summary.",
        "topics": ("technology", "ai"),
    }
    values.update(overrides)
    return Article(**values)  # type: ignore[arg-type]


class TestChunk(unittest.TestCase):
    """Chunk field validation and embedding normalization."""

    def test_valid_chunk(self) -> None:
        chunk = Chunk(
            id="c1",
            article_id="a1",
            ordinal=0,
            text="hello",
            embedding=(1, 0.0, -1),
        )
        self.assertEqual(chunk.embedding, (1.0, 0.0, -1.0))
        self.assertEqual(chunk.ordinal, 0)

    def test_required_strings_rejected(self) -> None:
        for field, value in (
            ("id", ""),
            ("article_id", " "),
            ("text", ""),
        ):
            values = {"id": "c1", "article_id": "a1", "ordinal": 0, "text": "t"}
            values[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                Chunk(**values)  # type: ignore[arg-type]

    def test_ordinal_validation(self) -> None:
        for ordinal in (-1, 1.5, "0", True):
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                Chunk(id="c1", article_id="a1", ordinal=ordinal, text="t")  # type: ignore[arg-type]

    def test_embedding_validation(self) -> None:
        for embedding in ((float("nan"),), (float("inf"),), ("x",)):
            with self.subTest(embedding=embedding), self.assertRaises(ValueError):
                Chunk(id="c1", article_id="a1", ordinal=0, text="t", embedding=embedding)  # type: ignore[arg-type]

    def test_embedding_none_is_allowed(self) -> None:
        self.assertIsNone(Chunk(id="c1", article_id="a1", ordinal=0, text="t").embedding)


class TestEntityRelationMention(unittest.TestCase):
    """Entity, relation, and mention validation."""

    def test_entity_valid_and_optional_canonical(self) -> None:
        entity = Entity(id="e1", name="Ada", type="person", canonical="Augusta Ada King")
        self.assertEqual(entity.canonical, "Augusta Ada King")
        self.assertIsNone(Entity(id="e1", name="Ada", type="person").canonical)

    def test_entity_required_strings(self) -> None:
        for field in ("id", "name", "type"):
            values = {"id": "e1", "name": "Ada", "type": "person"}
            values[field] = ""
            with self.subTest(field=field), self.assertRaises(ValueError):
                Entity(**values)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Entity(id="e1", name="Ada", type="person", canonical="")

    def test_relation_defaults_and_bounds(self) -> None:
        relation = Relation(source_id="e1", target_id="e2", type="knows")
        self.assertEqual(relation.confidence, 1.0)
        self.assertIsNone(relation.evidence_chunk_id)
        for confidence in (0.0, 0.5, 1.0):
            with self.subTest(confidence=confidence):
                self.assertEqual(
                    Relation(
                        source_id="e1",
                        target_id="e2",
                        type="knows",
                        confidence=confidence,
                    ).confidence,
                    confidence,
                )

    def test_relation_invalid(self) -> None:
        for confidence in (-0.1, 1.1, float("nan"), "0.5", True):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                Relation(
                    source_id="e1",
                    target_id="e2",
                    type="knows",
                    confidence=confidence,  # type: ignore[arg-type]
                )
        with self.assertRaises(ValueError):
            Relation(source_id="", target_id="e2", type="knows")
        with self.assertRaises(ValueError):
            Relation(source_id="e1", target_id="e2", type="knows", evidence_chunk_id="")

    def test_mention_validation(self) -> None:
        mention = Mention(chunk_id="c1", entity_id="e1")
        self.assertEqual((mention.chunk_id, mention.entity_id), ("c1", "e1"))
        for chunk_id, entity_id in (("", "e1"), ("c1", "")):
            with self.assertRaises(ValueError):
                Mention(chunk_id=chunk_id, entity_id=entity_id)

    def test_graph_context_defaults_and_frozen(self) -> None:
        context = GraphContext()
        self.assertEqual(context, GraphContext(chunks=(), entities=(), relations=()))
        with self.assertRaises(FrozenInstanceError):
            context.chunks = ()  # type: ignore[misc]


class TestDeterministicHelpers(unittest.TestCase):
    """Stable identifier and content-hash helpers."""

    def test_chunk_id_is_stable_and_short(self) -> None:
        first = chunk_id_for("a1", 0, "text")
        second = chunk_id_for("a1", 0, "text")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        int(first, 16)
        self.assertNotEqual(first, chunk_id_for("a1", 1, "text"))
        self.assertNotEqual(first, chunk_id_for("a2", 0, "text"))
        self.assertNotEqual(first, chunk_id_for("a1", 0, "other"))

    def test_entity_id_normalizes_case_and_whitespace(self) -> None:
        expected = entity_id_for(" Ada Lovelace ", "Person")
        self.assertEqual(expected, entity_id_for("ada lovelace", "person"))
        self.assertEqual(len(expected), 16)
        self.assertNotEqual(expected, entity_id_for("ada lovelace", "place"))

    def test_relation_key_is_a_stable_tuple(self) -> None:
        self.assertEqual(relation_key("e1", "e2", "knows"), ("e1", "e2", "knows"))

    def test_article_content_hash_is_stable(self) -> None:
        article = make_article()
        digest = article_content_hash(article)
        self.assertEqual(digest, article_content_hash(make_article()))
        self.assertEqual(len(digest), 64)
        int(digest, 16)

    def test_article_content_hash_ignores_raw_payload(self) -> None:
        first = make_article(raw={"provider": "one"})
        second = make_article(raw={"provider": "two", "noise": [1, 2, 3]})
        self.assertEqual(article_content_hash(first), article_content_hash(second))

    def test_article_content_hash_changes_with_content(self) -> None:
        base = article_content_hash(make_article())
        variants = (
            make_article(title="Different"),
            make_article(url="https://example.com/other"),
            make_article(source="Other"),
            make_article(summary=None),
            make_article(topics=("technology",)),
            make_article(published_at=None),
        )
        for article in variants:
            with self.subTest(article=article):
                self.assertNotEqual(base, article_content_hash(article))

    def test_validate_filters(self) -> None:
        self.assertEqual(validate_filters(None), {})
        self.assertEqual(
            validate_filters({"article_id": "a1", "source": None}),
            {"article_id": "a1"},
        )
        with self.assertRaises(GraphConfigError) as raised:
            validate_filters({"unknown": "x"})
        self.assertIn("unknown", str(raised.exception))
        with self.assertRaises(GraphConfigError):
            validate_filters([])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
