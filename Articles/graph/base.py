"""Frozen graph domain models, store protocol, and deterministic identifiers.

The graph layer stores article chunks, named entities, typed relations, and
mentions. Identifiers are derived from content so re-ingesting the same article
is idempotent. Everything in this module is standard-library only so the
default test suite never imports a graph driver.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from Articles.graph.errors import GraphConfigError
from Articles.Models import Article

__all__ = [
    "Chunk",
    "Entity",
    "GraphContext",
    "GraphStore",
    "Mention",
    "Relation",
    "article_content_hash",
    "chunk_id_for",
    "entity_id_for",
    "relation_key",
    "validate_filters",
]

_ALLOWED_FILTER_KEYS = frozenset({"article_id", "source", "topic"})


def _require_non_empty(name: str, value: str) -> None:
    """Validate that a required string field is present and non-blank.

    Args:
        name: Field name used in the error message.
        value: Value to validate.

    Raises:
        ValueError: If ``value`` is not a string or is empty/whitespace only.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _optional_non_empty(name: str, value: str | None) -> None:
    """Validate an optional string field.

    Args:
        name: Field name used in the error message.
        value: Value to validate, or ``None``.

    Raises:
        ValueError: If ``value`` is present but not a non-empty string.
    """
    if value is not None:
        _require_non_empty(name, value)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable slice of an article.

    Attributes:
        id: Deterministic chunk identifier.
        article_id: Identifier of the owning article.
        ordinal: Zero-based position of the chunk inside the article.
        text: Chunk text used for full-text and vector retrieval.
        embedding: Optional dense vector. Entries must be finite when present.
    """

    id: str
    article_id: str
    ordinal: int
    text: str
    embedding: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate identifiers, ordinal, and embedding values.

        Raises:
            ValueError: If a required value is missing or an embedding entry is
                not a finite number.
        """
        _require_non_empty("id", self.id)
        _require_non_empty("article_id", self.article_id)
        _require_non_empty("text", self.text)
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool):
            raise ValueError("ordinal must be an integer")
        if self.ordinal < 0:
            raise ValueError("ordinal must be greater than or equal to 0")
        if self.embedding is None:
            return
        try:
            values = tuple(float(value) for value in self.embedding)
        except (TypeError, ValueError) as err:
            raise ValueError("embedding must contain finite floats") from err
        if any(not math.isfinite(value) for value in values):
            raise ValueError("embedding must contain finite floats")
        object.__setattr__(self, "embedding", values)


@dataclass(frozen=True, slots=True)
class Entity:
    """A named graph entity extracted from an article.

    Attributes:
        id: Deterministic entity identifier.
        name: Display name.
        type: Entity type such as ``"person"`` or ``"organization"``.
        canonical: Optional canonical form used for disambiguation.
    """

    id: str
    name: str
    type: str
    canonical: str | None = None

    def __post_init__(self) -> None:
        """Validate required strings.

        Raises:
            ValueError: If an identifier, name, or type is empty, or if
                ``canonical`` is provided but blank.
        """
        _require_non_empty("id", self.id)
        _require_non_empty("name", self.name)
        _require_non_empty("type", self.type)
        _optional_non_empty("canonical", self.canonical)


@dataclass(frozen=True, slots=True)
class Relation:
    """A typed relation between two entities.

    Attributes:
        source_id: Identifier of the source entity.
        target_id: Identifier of the target entity.
        type: Relation type such as ``"works_for"``.
        confidence: Extraction confidence in the inclusive ``[0, 1]`` range.
        evidence_chunk_id: Optional chunk that supports the relation.
    """

    source_id: str
    target_id: str
    type: str
    confidence: float = 1.0
    evidence_chunk_id: str | None = None

    def __post_init__(self) -> None:
        """Validate identifiers and confidence.

        Raises:
            ValueError: If a required string is empty or ``confidence`` is not
                a finite number between 0 and 1.
        """
        _require_non_empty("source_id", self.source_id)
        _require_non_empty("target_id", self.target_id)
        _require_non_empty("type", self.type)
        _optional_non_empty("evidence_chunk_id", self.evidence_chunk_id)
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise ValueError("confidence must be a number")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Mention:
    """A link between a chunk and an entity mentioned in it.

    Attributes:
        chunk_id: Identifier of the mentioning chunk.
        entity_id: Identifier of the mentioned entity.
    """

    chunk_id: str
    entity_id: str

    def __post_init__(self) -> None:
        """Validate both identifiers.

        Raises:
            ValueError: If either identifier is empty.
        """
        _require_non_empty("chunk_id", self.chunk_id)
        _require_non_empty("entity_id", self.entity_id)


@dataclass(frozen=True, slots=True)
class GraphContext:
    """A bounded subgraph returned from expansion.

    Attributes:
        chunks: Chunks in stable retrieval order.
        entities: Entities mentioned by the selected chunks.
        relations: Relations touching the selected entities.
    """

    chunks: tuple[Chunk, ...] = ()
    entities: tuple[Entity, ...] = ()
    relations: tuple[Relation, ...] = ()


class GraphStore(Protocol):
    """Storage port implemented by in-memory and Neo4j adapters."""

    async def upsert_articles(
        self,
        articles: Sequence[Article],
        chunks: Sequence[Chunk],
    ) -> None:
        """Insert or update articles and replace their chunk sets.

        Args:
            articles: Articles to upsert.
            chunks: Chunks belonging to those articles.
        """

    async def upsert_entities(
        self,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        mentions: Sequence[Mention] = (),
    ) -> None:
        """Insert or update entities, relations, and mention links.

        Args:
            entities: Entities to upsert.
            relations: Relations to upsert.
            mentions: Chunk-to-entity links to add.
        """

    async def article_hashes(self, article_ids: Sequence[str]) -> dict[str, str]:
        """Return stored content hashes for known article identifiers.

        The pipeline uses this lookup to skip articles whose normalized content
        has not changed since the last successful ingest.

        Args:
            article_ids: Identifiers to look up. Unknown identifiers are
                omitted from the result.

        Returns:
            A mapping of article identifier to content hash.
        """

    async def vector_search(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return the ``k`` most similar chunks with cosine scores.

        Args:
            embedding: Query vector.
            k: Maximum number of results.
            filters: Optional ``article_id``, ``source``, or ``topic`` filters.

        Returns:
            Chunk/score pairs ordered by descending similarity.
        """

    async def expand(
        self,
        chunk_ids: Sequence[str],
        *,
        depth: int = 1,
        limit: int = 50,
    ) -> GraphContext:
        """Expand from chunks through mentions and relations.

        Args:
            chunk_ids: Seed chunk identifiers.
            depth: Number of expansion hops.
            limit: Maximum chunks, entities, and relations to return.

        Returns:
            A bounded, deterministically ordered subgraph.
        """

    async def healthcheck(self) -> bool:
        """Return whether the backing store is reachable."""

    async def aclose(self) -> None:
        """Release any resources owned by the store."""


def article_content_hash(article: Article) -> str:
    """Compute a stable SHA-256 content hash for an article.

    Only normalized content fields participate; provider-specific ``raw``
    payloads are ignored so cosmetic upstream changes do not cause re-ingestion.

    Args:
        article: Article to hash.

    Returns:
        Lowercase SHA-256 hex digest.
    """
    published_at = article.published_at.isoformat() if article.published_at is not None else ""
    parts = (
        article.id,
        article.title,
        article.url,
        article.source,
        published_at,
        article.summary or "",
        "\x1f".join(article.topics),
    )
    payload = "\x1e".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_id_for(article_id: str, ordinal: int, text: str) -> str:
    """Derive a deterministic 16-character chunk identifier.

    Args:
        article_id: Identifier of the owning article.
        ordinal: Zero-based chunk position.
        text: Chunk text.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    payload = f"{article_id}\x1f{ordinal}\x1f{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def entity_id_for(name: str, type: str) -> str:
    """Derive a deterministic entity identifier from a normalized name and type.

    Args:
        name: Entity display name.
        type: Entity type.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    normalized = f"{name.strip().lower()}\x1f{type.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def relation_key(source_id: str, target_id: str, type: str) -> tuple[str, str, str]:
    """Build the stable merge key for a relation.

    Args:
        source_id: Source entity identifier.
        target_id: Target entity identifier.
        type: Relation type.

    Returns:
        A ``(source_id, target_id, type)`` tuple.
    """
    return (source_id, target_id, type)


def validate_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate optional vector-search filters.

    Args:
        filters: Mapping containing only ``article_id``, ``source``, and
            ``topic`` keys. ``None`` values are treated as absent.

    Returns:
        A new dictionary without ``None`` values.

    Raises:
        GraphConfigError: If the mapping contains an unsupported key.
    """
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise GraphConfigError("filters must be a mapping")
    unknown = sorted(str(key) for key in filters if key not in _ALLOWED_FILTER_KEYS)
    if unknown:
        raise GraphConfigError("unsupported graph filter key(s): " + ", ".join(unknown))
    return {key: value for key, value in filters.items() if value is not None}
