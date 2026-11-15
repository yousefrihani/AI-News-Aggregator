"""Unit tests for the versioned Neo4j schema definitions."""

import unittest
from pathlib import Path

from Articles.graph.schema import (
    CONSTRAINTS,
    INDEXES,
    SCHEMA_STATEMENTS,
    SCHEMA_VERSION,
    schema_statements,
    schema_version,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CYPHER_FILE = REPO_ROOT / "config" / "graph_schema.cypher"


class TestSchema(unittest.TestCase):
    """Schema version, statements, and the mirrored Cypher file."""

    def test_version(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "1")
        self.assertEqual(schema_version(), "1")

    def test_constraints_and_indexes(self) -> None:
        self.assertEqual(len(CONSTRAINTS), 4)
        self.assertEqual(len(INDEXES), 3)
        self.assertEqual(SCHEMA_STATEMENTS, (*CONSTRAINTS, *INDEXES))
        joined = "\n".join(SCHEMA_STATEMENTS)
        for name in ("article_id", "chunk_id", "entity_id", "topic_name"):
            self.assertIn(name, joined)
        for name in ("chunk_embedding", "chunk_text", "entity_name"):
            self.assertIn(name, joined)
        self.assertIn("vector.dimensions`: 384", joined)
        self.assertIn("cosine", joined)
        self.assertIn("FULLTEXT INDEX", joined)

    def test_all_statements_are_idempotent_creates(self) -> None:
        for statement in schema_statements():
            with self.subTest(statement=statement):
                self.assertTrue(statement.startswith("CREATE "))
                self.assertIn("IF NOT EXISTS", statement)

    def test_custom_dimension_is_interpolated(self) -> None:
        statements = schema_statements(768)
        vector = next(s for s in statements if "VECTOR INDEX" in s)
        self.assertIn("768", vector)
        self.assertNotIn("384", vector)
        self.assertEqual(statements, schema_statements(dim=768))

    def test_invalid_dimension_rejected(self) -> None:
        for dim in (0, -1, "384", 3.5, True, None):
            with self.subTest(dim=dim), self.assertRaises(ValueError):
                schema_statements(dim)  # type: ignore[arg-type]

    def test_mirrored_cypher_file(self) -> None:
        text = CYPHER_FILE.read_text(encoding="utf-8")
        self.assertIn("version 1", text)
        for name in (
            "article_id",
            "chunk_id",
            "entity_id",
            "topic_name",
            "chunk_text",
            "entity_name",
            "chunk_embedding",
            "cosine",
        ):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
