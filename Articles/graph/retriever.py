"""Graph RAG retrieval: embed a question, search, expand, and cite.

The retriever returns a bounded, deterministic context block whose chunk
identifiers come from the graph store. Citations are article identifiers so the
answer pattern can require ``[article_id]`` markers and callers can verify that
every marker maps back to retrieved content.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from Articles.graph.base import Chunk, Entity, GraphStore, Relation
from Articles.graph.embeddings import EmbeddingProvider
from Articles.graph.errors import GraphConfigError

__all__ = [
    "Citation",
    "DEFAULT_CONTEXT_CHARS",
    "DEFAULT_DEPTH",
    "DEFAULT_K",
    "DEFAULT_LIMIT",
    "GraphRAGRetriever",
    "RetrievedContext",
]

DEFAULT_K = 8
DEFAULT_DEPTH = 1
DEFAULT_LIMIT = 20
DEFAULT_CONTEXT_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class Citation:
    """An article-level citation backed by one or more retrieved chunks.

    Args:
        article_id: Identifier rendered in the answer as ``[article_id]``.
        chunk_ids: Retrieved chunks from that article, in context order.
    """

    article_id: str
    chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate identifiers.

        Raises:
            ValueError: If the article identifier is blank or a chunk
                identifier is blank.
        """
        if not isinstance(self.article_id, str) or not self.article_id.strip():
            raise ValueError("citation article_id must be a non-empty string")
        for chunk_id in self.chunk_ids:
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("citation chunk_ids must be non-empty strings")


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """Bounded retrieval result ready for the answer pattern.

    Args:
        query: Original question.
        chunks: Context chunks in deterministic order (vector hits first).
        entities: Entities mentioned by the context chunks.
        relations: Relations touching the context entities.
        citations: Article citations derived from the included chunks.
        context_text: Rendered context block for the answer prompt.
        scores: Vector similarity for the seed chunks, keyed by chunk id.
    """

    query: str
    chunks: tuple[Chunk, ...] = ()
    entities: tuple[Entity, ...] = ()
    relations: tuple[Relation, ...] = ()
    citations: tuple[Citation, ...] = ()
    context_text: str = ""
    scores: Mapping[str, float] = MappingProxyType({})

    def __post_init__(self) -> None:
        """Freeze the score mapping.

        Raises:
            ValueError: If the query is blank.
        """
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("retrieval query must be a non-empty string")
        if not isinstance(self.scores, Mapping):
            raise ValueError("retrieval scores must be a mapping")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    @property
    def article_ids(self) -> tuple[str, ...]:
        """Return cited article identifiers in context order."""
        return tuple(citation.article_id for citation in self.citations)

    @property
    def is_empty(self) -> bool:
        """Return whether no chunk was retrieved."""
        return not self.chunks


class GraphRAGRetriever:
    """Retrieve graph-backed context for a question.

    Args:
        store: Graph store providing vector search and bounded expansion.
        embedder: Provider used to embed the question.
        k: Maximum number of vector hits.
        depth: Expansion hops, 1 or 2.
        limit: Maximum chunks, entities, and relations returned by expansion.
        max_context_chars: Hard cap on the rendered context block.

    Raises:
        GraphConfigError: If a numeric setting is out of range.
    """

    def __init__(
        self,
        store: GraphStore,
        embedder: EmbeddingProvider,
        *,
        k: int = DEFAULT_K,
        depth: int = DEFAULT_DEPTH,
        limit: int = DEFAULT_LIMIT,
        max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    ) -> None:
        self.k = _positive_int("k", k)
        self.depth = _positive_int("depth", depth)
        if self.depth not in (1, 2):
            raise GraphConfigError("depth must be 1 or 2")
        self.limit = _positive_int("limit", limit)
        self.max_context_chars = _positive_int("max_context_chars", max_context_chars)
        self._store = store
        self._embedder = embedder

    async def retrieve(self, query: str) -> RetrievedContext:
        """Retrieve context for a question.

        Args:
            query: Natural-language question.

        Returns:
            A bounded context with article citations.

        Raises:
            ValueError: If the query is blank.
            EmbeddingError: If the embedder fails.
            GraphError: If the store fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        normalized_query = query.strip()
        vector = (await self._embedder.embed([normalized_query]))[0]
        hits = await self._store.vector_search(vector, self.k)
        seed_chunks = [chunk for chunk, _ in hits]
        scores = {chunk.id: float(score) for chunk, score in hits}

        expanded = await self._store.expand(
            [chunk.id for chunk in seed_chunks],
            depth=self.depth,
            limit=self.limit,
        )
        chunks = _dedupe_chunks((*seed_chunks, *expanded.chunks))
        included, context_text = _render_context(
            chunks,
            expanded.entities,
            expanded.relations,
            max_chars=self.max_context_chars,
        )
        return RetrievedContext(
            query=normalized_query,
            chunks=included,
            entities=expanded.entities,
            relations=expanded.relations,
            citations=_citations_for(included),
            context_text=context_text,
            scores={chunk.id: scores[chunk.id] for chunk in included if chunk.id in scores},
        )


def _render_context(
    chunks: Sequence[Chunk],
    entities: Sequence[Entity],
    relations: Sequence[Relation],
    *,
    max_chars: int,
) -> tuple[tuple[Chunk, ...], str]:
    """Render chunks and graph facts into a bounded prompt block.

    Chunks are added whole; a chunk that would exceed the cap is skipped along
    with everything after it so the citation list always matches the text.

    Args:
        chunks: Candidate chunks in deterministic order.
        entities: Expanded entities.
        relations: Expanded relations.
        max_chars: Character budget for the whole block.

    Returns:
        The included chunks and the rendered context text.
    """
    included: list[Chunk] = []
    blocks: list[str] = []
    current_article: str | None = None
    buffer: list[str] = []
    used = 0

    def flush() -> None:
        nonlocal current_article, buffer
        if current_article is None:
            return
        blocks.append(f"[{current_article}]\n" + "\n".join(buffer))
        current_article = None
        buffer = []

    for chunk in chunks:
        line = f"  {chunk.id}: {chunk.text.strip()}"
        header = f"[{chunk.article_id}]\n" if chunk.article_id != current_article else ""
        addition = len(header) + len(line) + 1
        if used + addition > max_chars:
            break
        if chunk.article_id != current_article:
            flush()
            current_article = chunk.article_id
        buffer.append(line)
        used += addition
        included.append(chunk)
    flush()

    facts = _render_facts(entities, relations)
    if facts and used + len(facts) + 1 <= max_chars:
        blocks.append(facts)
    return tuple(included), "\n\n".join(blocks)


def _render_facts(entities: Sequence[Entity], relations: Sequence[Relation]) -> str:
    """Render entity and relation facts for the prompt.

    Args:
        entities: Expanded entities.
        relations: Expanded relations.

    Returns:
        A human-readable facts block, or an empty string.
    """
    by_id = {entity.id: entity.name for entity in entities}
    lines: list[str] = []
    if entities:
        lines.append(
            "Entities: " + ", ".join(f"{entity.name} ({entity.type})" for entity in entities)
        )
    for relation in relations:
        source = by_id.get(relation.source_id)
        target = by_id.get(relation.target_id)
        if source is None or target is None:
            continue
        lines.append(f"{source} -[{relation.type}]-> {target}")
    return "\n".join(lines)


def _citations_for(chunks: Iterable[Chunk]) -> tuple[Citation, ...]:
    """Group included chunks by article in first-appearance order.

    Args:
        chunks: Included chunks in context order.

    Returns:
        Article citations with their chunk identifiers.
    """
    order: list[str] = []
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.article_id not in grouped:
            order.append(chunk.article_id)
            grouped[chunk.article_id] = []
        grouped[chunk.article_id].append(chunk.id)
    return tuple(Citation(article_id, tuple(grouped[article_id])) for article_id in order)


def _dedupe_chunks(chunks: Iterable[Chunk]) -> tuple[Chunk, ...]:
    """Remove duplicate chunk identifiers, preserving first occurrence.

    Args:
        chunks: Candidate chunks.

    Returns:
        Unique chunks in input order.
    """
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        unique.append(chunk)
    return tuple(unique)


def _positive_int(name: str, value: Any) -> int:
    """Validate a positive integer setting.

    Args:
        name: Setting name.
        value: Candidate value.

    Returns:
        The validated integer.

    Raises:
        GraphConfigError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise GraphConfigError(f"{name} must be a positive integer")
    return value
