"""Unit tests for graph and embedding configuration."""

import os
import unittest
from unittest.mock import patch

from Articles.graph.config import EmbeddingSettings, GraphSettings
from Articles.graph.errors import GraphConfigError

CLEAR_ENV = {
    "GRAPH_BACKEND": "",
    "NEO4J_URI": "",
    "NEO4J_USER": "",
    "NEO4J_PASSWORD": "",
    "NEO4J_DATABASE": "",
    "EMBEDDING_PROVIDER": "",
    "EMBEDDING_MODEL": "",
    "EMBEDDING_DIM": "",
}


class TestGraphSettings(unittest.TestCase):
    """GraphSettings defaults, env overrides, and validation."""

    def test_defaults(self) -> None:
        settings = GraphSettings()
        self.assertEqual(settings.backend, "memory")
        self.assertEqual(settings.neo4j_uri, "bolt://localhost:7687")
        self.assertEqual(settings.neo4j_user, "neo4j")
        self.assertIsNone(settings.neo4j_password)
        self.assertEqual(settings.neo4j_database, "neo4j")

    def test_from_env_defaults(self) -> None:
        with patch.dict(os.environ, CLEAR_ENV, clear=True):
            settings = GraphSettings.from_env()
        self.assertEqual(settings, GraphSettings())

    def test_from_env_overrides(self) -> None:
        env = {
            "GRAPH_BACKEND": "NEO4J",
            "NEO4J_URI": "bolt://graph:7687",
            "NEO4J_USER": "graph-user",
            "NEO4J_PASSWORD": "graph-pass",
            "NEO4J_DATABASE": "graph-db",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = GraphSettings.from_env()
        self.assertEqual(settings.backend, "neo4j")
        self.assertEqual(settings.neo4j_uri, "bolt://graph:7687")
        self.assertEqual(settings.neo4j_user, "graph-user")
        self.assertEqual(settings.neo4j_password, "graph-pass")
        self.assertEqual(settings.neo4j_database, "graph-db")

    def test_unknown_backend_rejected(self) -> None:
        with self.assertRaises(GraphConfigError):
            GraphSettings(backend="falkordb")  # type: ignore[arg-type]
        with (
            patch.dict(os.environ, {"GRAPH_BACKEND": "falkordb"}, clear=True),
            self.assertRaises(GraphConfigError),
        ):
            GraphSettings.from_env()

    def test_neo4j_requires_password(self) -> None:
        for password in (None, "", "   "):
            with self.subTest(password=password), self.assertRaises(GraphConfigError):
                GraphSettings(backend="neo4j", neo4j_password=password)
        with (
            patch.dict(os.environ, {"GRAPH_BACKEND": "neo4j"}, clear=True),
            self.assertRaises(GraphConfigError) as raised,
        ):
            GraphSettings.from_env()
        self.assertIn("NEO4J_PASSWORD", str(raised.exception))

    def test_required_connection_fields_non_empty(self) -> None:
        for field in ("neo4j_uri", "neo4j_user", "neo4j_database"):
            with self.subTest(field=field), self.assertRaises(GraphConfigError):
                GraphSettings(**{field: " "})  # type: ignore[arg-type]

    def test_memory_backend_does_not_require_password(self) -> None:
        self.assertIsNone(GraphSettings(backend="memory", neo4j_password=None).neo4j_password)


class TestEmbeddingSettings(unittest.TestCase):
    """EmbeddingSettings defaults, env overrides, and validation."""

    def test_defaults(self) -> None:
        settings = EmbeddingSettings()
        self.assertEqual(settings.provider, "fastembed")
        self.assertEqual(settings.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(settings.dim, 384)

    def test_from_env_defaults(self) -> None:
        with patch.dict(os.environ, CLEAR_ENV, clear=True):
            settings = EmbeddingSettings.from_env()
        self.assertEqual(settings, EmbeddingSettings())

    def test_from_env_overrides(self) -> None:
        env = {
            "EMBEDDING_PROVIDER": "DETERMINISTIC",
            "EMBEDDING_MODEL": "custom/model",
            "EMBEDDING_DIM": "768",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = EmbeddingSettings.from_env()
        self.assertEqual(settings.provider, "deterministic")
        self.assertEqual(settings.model, "custom/model")
        self.assertEqual(settings.dim, 768)

    def test_invalid_provider(self) -> None:
        with self.assertRaises(GraphConfigError):
            EmbeddingSettings(provider="openai")  # type: ignore[arg-type]
        with (
            patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai"}, clear=True),
            self.assertRaises(GraphConfigError),
        ):
            EmbeddingSettings.from_env()

    def test_invalid_dimension(self) -> None:
        for dim in (0, -1, 1.5, True):
            with self.subTest(dim=dim), self.assertRaises(GraphConfigError):
                EmbeddingSettings(dim=dim)  # type: ignore[arg-type]
        with (
            patch.dict(os.environ, {"EMBEDDING_DIM": "not-a-number"}, clear=True),
            self.assertRaises(GraphConfigError),
        ):
            EmbeddingSettings.from_env()
        with (
            patch.dict(os.environ, {"EMBEDDING_DIM": "-4"}, clear=True),
            self.assertRaises(GraphConfigError),
        ):
            EmbeddingSettings.from_env()

    def test_model_must_not_be_blank(self) -> None:
        with self.assertRaises(GraphConfigError):
            EmbeddingSettings(model="  ")


if __name__ == "__main__":
    unittest.main()
