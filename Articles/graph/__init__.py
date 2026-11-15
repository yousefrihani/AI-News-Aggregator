"""Graph RAG building blocks: models, stores, embeddings, and schema.

Importing this package never opens a database connection and never imports an
optional driver. Concrete adapters import their SDK lazily.
"""

from Articles.graph.base import (
    Chunk,
    Entity,
    GraphContext,
    GraphStore,
    Mention,
    Relation,
    article_content_hash,
    chunk_id_for,
    entity_id_for,
    relation_key,
    validate_filters,
)
from Articles.graph.config import EmbeddingSettings, GraphSettings
from Articles.graph.embeddings import (
    DeterministicEmbedding,
    EmbeddingProvider,
    FastEmbedEmbedding,
)
from Articles.graph.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    GraphConfigError,
    GraphConnectionError,
    GraphError,
    GraphQueryError,
)
from Articles.graph.memory import InMemoryGraphStore
from Articles.graph.neo4j import Neo4jGraphStore
from Articles.graph.schema import (
    CONSTRAINTS,
    INDEXES,
    SCHEMA_STATEMENTS,
    SCHEMA_VERSION,
    schema_statements,
    schema_version,
)

__all__ = [
    "CONSTRAINTS",
    "INDEXES",
    "SCHEMA_STATEMENTS",
    "SCHEMA_VERSION",
    "Chunk",
    "DeterministicEmbedding",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingSettings",
    "Entity",
    "FastEmbedEmbedding",
    "GraphConfigError",
    "GraphConnectionError",
    "GraphContext",
    "GraphError",
    "GraphQueryError",
    "GraphSettings",
    "GraphStore",
    "InMemoryGraphStore",
    "Mention",
    "Neo4jGraphStore",
    "Relation",
    "article_content_hash",
    "chunk_id_for",
    "entity_id_for",
    "relation_key",
    "schema_statements",
    "schema_version",
    "validate_filters",
]
