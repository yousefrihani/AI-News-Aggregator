"""Versioned Neo4j schema definitions for the Graph RAG layer.

Every statement is idempotent so :meth:`Neo4jGraphStore.initialize` can run on
each startup. The same statements are mirrored in ``config/graph_schema.cypher``
for manual application; keep the two files in sync.
"""

from typing import Any

__all__ = [
    "CONSTRAINTS",
    "INDEXES",
    "SCHEMA_STATEMENTS",
    "SCHEMA_VERSION",
    "schema_statements",
    "schema_version",
]

SCHEMA_VERSION = "1"

CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
)

_FULLTEXT_INDEXES: tuple[str, ...] = (
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
)


def _vector_index(dim: int) -> str:
    """Build the vector-index statement for a dimension.

    Args:
        dim: Vector dimension.

    Returns:
        An idempotent ``CREATE VECTOR INDEX`` Cypher statement.
    """
    return (
        "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: "
        f"{dim}, `vector.similarity_function`: 'cosine'}}"
    )


INDEXES: tuple[str, ...] = (*_FULLTEXT_INDEXES, _vector_index(384))
SCHEMA_STATEMENTS: tuple[str, ...] = (*CONSTRAINTS, *INDEXES)


def _validate_dim(dim: Any) -> int:
    """Validate a vector dimension.

    Args:
        dim: Candidate value.

    Returns:
        The validated integer.

    Raises:
        ValueError: If ``dim`` is not a positive integer.
    """
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError("dim must be an integer")
    if dim <= 0:
        raise ValueError("dim must be greater than 0")
    return dim


def schema_statements(dim: int = 384) -> tuple[str, ...]:
    """Return all schema statements with the vector dimension interpolated.

    Args:
        dim: Vector dimension. Must be a positive integer.

    Returns:
        Idempotent Cypher statements.

    Raises:
        ValueError: If ``dim`` is not a positive integer.
    """
    validated = _validate_dim(dim)
    return (*CONSTRAINTS, *_FULLTEXT_INDEXES, _vector_index(validated))


def schema_version() -> str:
    """Return the current schema version.

    Returns:
        A short version string such as ``"1"``.
    """
    return SCHEMA_VERSION
