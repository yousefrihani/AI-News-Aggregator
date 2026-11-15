"""In-memory :class:`~Articles.graph.base.GraphStore` implementation.

The store is deterministic, dependency-free, and intended for unit tests, local
development, and small offline demos. It mirrors the Neo4j adapter's semantics:
idempotent upserts, exact chunk replacement per article, cosine vector search,
and bounded graph expansion.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from Articles.graph.base import (
    Chunk,
    Entity,
    GraphContext,
    Mention,
    Relation,
    article_content_hash,
    relation_key,
    validate_filters,
)
from Articles.Models import Article

__all__ = ["InMemoryGraphStore"]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity, returning ``0.0`` for zero-norm vectors.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        Similarity in ``[-1, 1]``, or ``0.0`` when either vector is all zeros.
    """
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


class InMemoryGraphStore:
    """Dictionary-backed graph store with deterministic ordering.

    The store keeps articles, chunks, entities, relations, and mentions in
    dictionaries. Upserts are idempotent and never leave stale chunks behind
    when an article is re-ingested.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._articles: dict[str, Article] = {}
        self._chunks: dict[str, Chunk] = {}
        self._article_chunks: dict[str, set[str]] = {}
        self._entities: dict[str, Entity] = {}
        self._relations: dict[tuple[str, str, str], Relation] = {}
        self._mentions: dict[str, set[str]] = {}

    async def upsert_articles(
        self,
        articles: Sequence[Article],
        chunks: Sequence[Chunk],
    ) -> None:
        """Insert or update articles and replace their chunk sets exactly.

        Articles not present in ``articles`` are left untouched. Chunks whose
        article is being re-ingested but which are absent from ``chunks`` are
        removed together with their mentions.

        Args:
            articles: Articles to upsert.
            chunks: Chunks belonging to those articles.
        """
        for article in articles:
            self._articles[article.id] = article

        grouped: dict[str, dict[str, Chunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.article_id, {})[chunk.id] = chunk

        for article in articles:
            previous = self._article_chunks.get(article.id, set())
            incoming = set(grouped.get(article.id, {}))
            for chunk_id in previous - incoming:
                self._chunks.pop(chunk_id, None)
                self._mentions.pop(chunk_id, None)
            self._article_chunks[article.id] = incoming

        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._article_chunks.setdefault(chunk.article_id, set()).add(chunk.id)

    async def upsert_entities(
        self,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        mentions: Sequence[Mention] = (),
    ) -> None:
        """Insert or update entities, relations, and mentions.

        Relations merge by ``(source_id, target_id, type)`` so the latest
        confidence and evidence win. Mentions are unioned.

        Args:
            entities: Entities to upsert.
            relations: Relations to upsert.
            mentions: Chunk-to-entity links to add.
        """
        for entity in entities:
            self._entities[entity.id] = entity
        for relation in relations:
            self._relations[relation_key(relation.source_id, relation.target_id, relation.type)] = (
                relation
            )
        for mention in mentions:
            self._mentions.setdefault(mention.chunk_id, set()).add(mention.entity_id)

    def _matches(self, chunk: Chunk, filters: Mapping[str, Any]) -> bool:
        """Return whether a chunk satisfies validated filters.

        Args:
            chunk: Candidate chunk.
            filters: Validated filter mapping.

        Returns:
            ``True`` when the chunk matches every provided filter.
        """
        article = self._articles.get(chunk.article_id)
        if "article_id" in filters and chunk.article_id != filters["article_id"]:
            return False
        if "source" in filters and (article is None or article.source != filters["source"]):
            return False
        return not (
            "topic" in filters and (article is None or filters["topic"] not in article.topics)
        )

    async def article_hashes(self, article_ids: Sequence[str]) -> dict[str, str]:
        """Return stored content hashes for known articles.

        Args:
            article_ids: Identifiers to look up.

        Returns:
            A mapping of known article identifier to content hash.
        """
        hashes: dict[str, str] = {}
        for article_id in article_ids:
            article = self._articles.get(article_id)
            if article is not None:
                hashes[article_id] = article_content_hash(article)
        return hashes

    async def vector_search(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return the ``k`` most cosine-similar chunks.

        Chunks without embeddings, or whose vector length differs from the query,
        are skipped. Ties are broken by ascending chunk identifier.

        Args:
            embedding: Query vector.
            k: Maximum number of results. Must be at least 1.
            filters: Optional ``article_id``, ``source``, or ``topic`` filters.

        Returns:
            Chunk/score pairs ordered by descending score, then chunk id.

        Raises:
            ValueError: If ``k`` is less than 1.
            GraphConfigError: If a filter key is unsupported.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        active = validate_filters(filters)
        query = tuple(float(value) for value in embedding)
        scored: list[tuple[Chunk, float]] = []
        for chunk in self._chunks.values():
            if chunk.embedding is None or len(chunk.embedding) != len(query):
                continue
            if not self._matches(chunk, active):
                continue
            scored.append((chunk, _cosine(query, chunk.embedding)))
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:k]

    async def expand(
        self,
        chunk_ids: Sequence[str],
        *,
        depth: int = 1,
        limit: int = 50,
    ) -> GraphContext:
        """Expand from seed chunks through mentions and relations.

        Seeds are deduplicated in request order. Each hop collects entities
        mentioned by the frontier, relations touching any collected entity, and
        chunks mentioning any collected entity. Every collection is bounded by
        ``limit`` and uses stable ordering.

        Args:
            chunk_ids: Seed chunk identifiers. Missing identifiers are ignored.
            depth: Number of expansion hops. Must be at least 1.
            limit: Maximum chunks, entities, and relations to return.

        Returns:
            A bounded subgraph.

        Raises:
            ValueError: If ``depth`` or ``limit`` is less than 1.
        """
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        selected: dict[str, Chunk] = {}
        for chunk_id in chunk_ids:
            chunk = self._chunks.get(chunk_id)
            if chunk is not None and chunk_id not in selected:
                selected[chunk_id] = chunk
                if len(selected) >= limit:
                    break
        frontier = list(selected)
        entity_ids: list[str] = []
        entity_seen: set[str] = set()
        relations: list[Relation] = []
        relation_seen: set[tuple[str, str, str]] = set()

        for _ in range(depth):
            new_frontier: list[str] = []
            for chunk_id in frontier:
                for entity_id in sorted(self._mentions.get(chunk_id, ())):
                    if entity_id in entity_seen or entity_id not in self._entities:
                        continue
                    if len(entity_ids) >= limit:
                        break
                    entity_seen.add(entity_id)
                    entity_ids.append(entity_id)

            if entity_seen:
                for key in sorted(self._relations):
                    relation = self._relations[key]
                    if key in relation_seen:
                        continue
                    if (
                        relation.source_id not in entity_seen
                        and relation.target_id not in entity_seen
                    ):
                        continue
                    if len(relations) >= limit:
                        break
                    relation_seen.add(key)
                    relations.append(relation)

                for chunk_id in sorted(self._chunks):
                    if chunk_id in selected:
                        continue
                    if not entity_seen.intersection(self._mentions.get(chunk_id, ())):
                        continue
                    if len(selected) >= limit:
                        break
                    selected[chunk_id] = self._chunks[chunk_id]
                    new_frontier.append(chunk_id)

            frontier = new_frontier
            if not frontier:
                break

        return GraphContext(
            chunks=tuple(selected.values()),
            entities=tuple(self._entities[entity_id] for entity_id in entity_ids),
            relations=tuple(relations),
        )

    async def healthcheck(self) -> bool:
        """Return ``True``; the in-memory store is always available."""
        return True

    async def aclose(self) -> None:
        """Release resources; the in-memory store owns none."""
