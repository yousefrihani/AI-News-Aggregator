"""Enrichment pipeline: chunk, summarize, classify, extract, embed, and store.

The pipeline is deliberately boring: each article is independent, failures are
isolated per article, and writes are batched. Re-running over unchanged
articles performs no LLM or embedding work because the graph store remembers
the normalized content hash of every fully ingested article.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from Articles.graph.base import (
    Chunk,
    Entity,
    GraphStore,
    Mention,
    Relation,
    article_content_hash,
    chunk_id_for,
    entity_id_for,
)
from Articles.graph.embeddings import EmbeddingProvider
from Articles.graph.errors import EmbeddingDimensionError, EmbeddingError
from Articles.llm.patterns import (
    Classification,
    ClassifyPattern,
    CompletionRouter,
    ExtractGraphPattern,
    GraphExtraction,
    SummarizePattern,
)
from Articles.Models import Article

__all__ = [
    "ArticleEnrichment",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_LABELS",
    "EnrichmentPipeline",
    "PipelineFailure",
    "PipelineReport",
    "article_document",
    "chunk_text",
]

DEFAULT_CHUNK_TOKENS = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_WRITE_BATCH_SIZE = 50
DEFAULT_CONCURRENCY = 4
DEFAULT_LABELS: tuple[str, ...] = (
    "technology",
    "business",
    "research",
    "policy",
    "product",
    "safety",
    "other",
)
_CHARS_PER_TOKEN = 4
_RAW_TEXT_KEYS = ("content", "description", "text")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ArticleEnrichment:
    """Successful per-article result.

    Args:
        article_id: Article identifier.
        summary: Model-generated summary.
        classification: Validated topic classification.
        chunk_ids: Stored chunk identifiers in article order.
        entity_count: Number of stored entities.
        relation_count: Number of stored relations.
    """

    article_id: str
    summary: str
    classification: Classification
    chunk_ids: tuple[str, ...]
    entity_count: int
    relation_count: int


@dataclass(frozen=True, slots=True)
class PipelineFailure:
    """One isolated article failure.

    Args:
        article_id: Article identifier.
        stage: Pipeline stage that failed.
        error: Redacted error description.
    """

    article_id: str
    stage: str
    error: str


@dataclass(frozen=True, slots=True)
class PipelineReport:
    """Result of one pipeline run.

    Args:
        enrichments: Successful articles in input order.
        skipped: Article identifiers skipped because their content hash matched.
        failures: Isolated failures in input order.
    """

    enrichments: tuple[ArticleEnrichment, ...] = ()
    skipped: tuple[str, ...] = ()
    failures: tuple[PipelineFailure, ...] = ()

    @property
    def succeeded_count(self) -> int:
        """Return the number of fully ingested articles."""
        return len(self.enrichments)

    @property
    def failed_count(self) -> int:
        """Return the number of isolated failures."""
        return len(self.failures)

    @property
    def skipped_count(self) -> int:
        """Return the number of unchanged articles."""
        return len(self.skipped)

    @property
    def total_count(self) -> int:
        """Return the total number of articles considered."""
        return self.succeeded_count + self.failed_count + self.skipped_count


@dataclass(slots=True)
class _PreparedArticle:
    """In-memory enrichment waiting for a batched write."""

    article: Article
    chunks: tuple[Chunk, ...]
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    mentions: tuple[Mention, ...]
    summary: str
    classification: Classification


class _StageError(Exception):
    """Internal marker carrying the stage that failed."""

    def __init__(self, stage: str, error: BaseException) -> None:
        super().__init__(f"{stage}: {error}")
        self.stage = stage
        self.error = error


class EnrichmentPipeline:
    """Orchestrate article enrichment into a graph store.

    Args:
        router: Router facade used by patterns.
        store: Graph store receiving enriched articles.
        embedder: Local or deterministic embedding provider.
        summarizer: Optional summary pattern override.
        classifier: Optional classification pattern override.
        extractor: Optional graph extraction pattern override.
        summarize_route: Route used by the default summarizer.
        classify_route: Route used by the default classifier.
        extract_route: Route used by the default extractor.
        labels: Candidate classification labels.
        chunk_tokens: Approximate chunk size in tokens.
        chunk_overlap: Approximate overlap in tokens.
        embedding_batch_size: Maximum texts per embedding call.
        write_batch_size: Maximum articles per store batch.
        max_concurrency: Default number of articles processed concurrently.
    """

    def __init__(
        self,
        router: CompletionRouter,
        store: GraphStore,
        embedder: EmbeddingProvider,
        *,
        summarizer: SummarizePattern | None = None,
        classifier: ClassifyPattern | None = None,
        extractor: ExtractGraphPattern | None = None,
        summarize_route: str = "summarize",
        classify_route: str = "classify",
        extract_route: str = "extract_graph",
        labels: Sequence[str] = DEFAULT_LABELS,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
        max_concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self.labels = _clean_labels(labels)
        self.chunk_tokens = _positive_int("chunk_tokens", chunk_tokens)
        self.chunk_overlap = _non_negative_int("chunk_overlap", chunk_overlap)
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_tokens")
        self.embedding_batch_size = _positive_int("embedding_batch_size", embedding_batch_size)
        self.write_batch_size = _positive_int("write_batch_size", write_batch_size)
        self.max_concurrency = _positive_int("max_concurrency", max_concurrency)
        self._store = store
        self._embedder = embedder
        self._summarizer = summarizer or SummarizePattern(router, summarize_route)
        self._classifier = classifier or ClassifyPattern(router, classify_route, labels=self.labels)
        self._extractor = extractor or ExtractGraphPattern(router, extract_route)

    async def process(
        self,
        articles: Sequence[Article],
        *,
        concurrency: int | None = None,
    ) -> PipelineReport:
        """Enrich every new or changed article.

        Args:
            articles: Candidate articles. Duplicate identifiers keep the first
                occurrence.
            concurrency: Optional per-run override for article concurrency.

        Returns:
            A report with successes, skipped identifiers, and isolated failures.
        """
        limit = (
            self.max_concurrency
            if concurrency is None
            else _positive_int("concurrency", concurrency)
        )
        unique = _dedupe_articles(articles)
        if not unique:
            return PipelineReport()

        order = {article.id: index for index, article in enumerate(unique)}
        hashes = {article.id: article_content_hash(article) for article in unique}
        stored = await self._store.article_hashes([article.id for article in unique])
        skipped = tuple(
            article.id for article in unique if stored.get(article.id) == hashes[article.id]
        )
        pending = [article for article in unique if stored.get(article.id) != hashes[article.id]]
        if not pending:
            return PipelineReport(skipped=skipped)

        semaphore = asyncio.Semaphore(limit)

        async def prepare(article: Article) -> _PreparedArticle:
            async with semaphore:
                return await self._prepare_article(article)

        results = await asyncio.gather(
            *(prepare(article) for article in pending), return_exceptions=True
        )
        prepared: list[_PreparedArticle] = []
        failures: list[PipelineFailure] = []
        for article, result in zip(pending, results, strict=True):
            if isinstance(result, _StageError):
                failures.append(_failure(article.id, result.stage, result.error))
            elif isinstance(result, BaseException):
                failures.append(_failure(article.id, "prepare", result))
            else:
                prepared.append(result)

        enrichments: list[ArticleEnrichment] = []
        for batch in _batches(prepared, self.write_batch_size):
            error = await self._write_batch(batch)
            if error is None:
                enrichments.extend(_enrichment(item) for item in batch)
            else:
                failures.extend(_failure(item.article.id, "store", error) for item in batch)

        enrichments.sort(key=lambda item: order[item.article_id])
        failures.sort(key=lambda item: order[item.article_id])
        return PipelineReport(
            enrichments=tuple(enrichments),
            skipped=skipped,
            failures=tuple(failures),
        )

    async def _prepare_article(self, article: Article) -> _PreparedArticle:
        """Run every per-article stage without touching the store.

        Args:
            article: Article to enrich.

        Returns:
            A prepared article ready for a batched write.

        Raises:
            _StageError: If any stage fails.
        """
        document = article_document(article)
        texts = chunk_text(
            document,
            chunk_tokens=self.chunk_tokens,
            overlap_tokens=self.chunk_overlap,
        )
        if not texts:
            raise _StageError("chunk", ValueError("article produced no chunks"))
        chunks = tuple(
            Chunk(
                id=chunk_id_for(article.id, ordinal, text),
                article_id=article.id,
                ordinal=ordinal,
                text=text,
            )
            for ordinal, text in enumerate(texts)
        )

        summary_result, classification_result = await asyncio.gather(
            _capture("summarize", self._summarizer.run(article.title, document)),
            _capture(
                "classify",
                self._classifier.run(article.title, document, labels=self.labels),
            ),
        )
        for result in (summary_result, classification_result):
            if isinstance(result, _StageError):
                raise result
        summary: str = summary_result
        classification: Classification = classification_result

        extraction_result = await _capture(
            "extract",
            self._extractor.run(
                article.title,
                document,
                chunk_ids=[chunk.id for chunk in chunks],
            ),
        )
        if isinstance(extraction_result, _StageError):
            raise extraction_result
        extraction: GraphExtraction = extraction_result
        entities, relations, mentions = self._graph_objects(chunks, extraction)

        embedded_result = await _capture("embed", self._embed_chunks(chunks))
        if isinstance(embedded_result, _StageError):
            raise embedded_result
        embedded: tuple[Chunk, ...] = embedded_result

        return _PreparedArticle(
            article=article,
            chunks=embedded,
            entities=entities,
            relations=relations,
            mentions=mentions,
            summary=summary,
            classification=classification,
        )

    async def _embed_chunks(self, chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
        """Embed chunks in bounded batches.

        Args:
            chunks: Chunks without embeddings.

        Returns:
            Chunks carrying their embedding.

        Raises:
            EmbeddingError: If the provider returns the wrong number of vectors.
            EmbeddingDimensionError: If a vector does not match the provider dim.
        """
        vectors: list[Sequence[float]] = []
        for batch in _batches(list(chunks), self.embedding_batch_size):
            batch_vectors = await self._embedder.embed([chunk.text for chunk in batch])
            if len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    f"embedding provider returned {len(batch_vectors)} vectors "
                    f"for {len(batch)} texts"
                )
            vectors.extend(batch_vectors)
        dim = getattr(self._embedder, "dim", None)
        embedded: list[Chunk] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            values = tuple(float(value) for value in vector)
            if dim is not None and len(values) != dim:
                raise EmbeddingDimensionError(
                    f"embedding dimension {len(values)} does not match provider dim {dim}"
                )
            embedded.append(replace(chunk, embedding=values))
        return tuple(embedded)

    def _graph_objects(
        self,
        chunks: Sequence[Chunk],
        extraction: GraphExtraction,
    ) -> tuple[tuple[Entity, ...], tuple[Relation, ...], tuple[Mention, ...]]:
        """Map extraction names onto deterministic graph identifiers.

        Args:
            chunks: Chunks belonging to the article.
            extraction: Validated extraction output.

        Returns:
            Entities, relations, and mentions ready for the store.
        """
        entities: list[Entity] = []
        entity_by_name: dict[str, str] = {}
        seen_entities: set[str] = set()
        for extracted in extraction.entities:
            entity_id = entity_id_for(extracted.name, extracted.type)
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            entity = Entity(
                id=entity_id,
                name=extracted.name,
                type=extracted.type,
                canonical=extracted.canonical,
            )
            entities.append(entity)
            entity_by_name.setdefault(extracted.name.casefold(), entity_id)
            if extracted.canonical:
                entity_by_name.setdefault(extracted.canonical.casefold(), entity_id)

        relations: list[Relation] = []
        seen_relations: set[tuple[str, str, str]] = set()
        chunk_ids = {chunk.id for chunk in chunks}
        for extracted_relation in extraction.relations:
            source_id = entity_by_name.get(extracted_relation.source.casefold())
            target_id = entity_by_name.get(extracted_relation.target.casefold())
            if source_id is None or target_id is None:
                continue
            relation_key = (source_id, target_id, extracted_relation.type)
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)
            relations.append(
                Relation(
                    source_id=source_id,
                    target_id=target_id,
                    type=extracted_relation.type,
                    confidence=extracted_relation.confidence,
                    evidence_chunk_id=extracted_relation.evidence_chunk_id,
                )
            )

        mentions: list[Mention] = []
        seen_mentions: set[tuple[str, str]] = set()
        for extracted_mention in extraction.mentions:
            if extracted_mention.chunk_id not in chunk_ids:
                continue
            mentioned_entity_id = entity_by_name.get(extracted_mention.entity.casefold())
            if mentioned_entity_id is None:
                continue
            mention_key = (extracted_mention.chunk_id, mentioned_entity_id)
            if mention_key in seen_mentions:
                continue
            seen_mentions.add(mention_key)
            mentions.append(
                Mention(chunk_id=extracted_mention.chunk_id, entity_id=mentioned_entity_id)
            )

        for entity in entities:
            needle = entity.name.casefold()
            for chunk in chunks:
                mention_key = (chunk.id, entity.id)
                if mention_key in seen_mentions or needle not in chunk.text.casefold():
                    continue
                seen_mentions.add(mention_key)
                mentions.append(Mention(chunk_id=chunk.id, entity_id=entity.id))

        return tuple(entities), tuple(relations), tuple(mentions)

    async def _write_batch(self, batch: Sequence[_PreparedArticle]) -> str | None:
        """Write one batch of prepared articles.

        Entities and mentions are written first, then articles and chunks. The
        article write is what records the content hash, so a failure before it
        leaves the article eligible for a clean retry.

        Args:
            batch: Prepared articles.

        Returns:
            ``None`` on success, otherwise a redacted error description.
        """
        entities = tuple(entity for item in batch for entity in item.entities)
        relations = tuple(relation for item in batch for relation in item.relations)
        mentions = tuple(mention for item in batch for mention in item.mentions)
        chunks = tuple(chunk for item in batch for chunk in item.chunks)
        articles = tuple(item.article for item in batch)
        try:
            if entities or relations or mentions:
                await self._store.upsert_entities(entities, relations, mentions)
            await self._store.upsert_articles(articles, chunks)
        except Exception as err:
            return _describe_error(err)
        return None


def article_document(article: Article) -> str:
    """Build the text the pipeline enriches.

    Provider content fields are preferred over the normalized summary so full
    article bodies are used when available. Raw payload changes do not affect
    the idempotency hash because :func:`article_content_hash` ignores ``raw``.

    Args:
        article: Article to render.

    Returns:
        A non-empty document starting with the title.
    """
    parts = [article.title.strip()]
    body: str | None = None
    for key in _RAW_TEXT_KEYS:
        value = article.raw.get(key)
        if isinstance(value, str) and value.strip():
            body = value.strip()
            break
    if body is None and isinstance(article.summary, str) and article.summary.strip():
        body = article.summary.strip()
    if body and body != parts[0]:
        parts.append(body)
    return "\n\n".join(parts)


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    chars_per_token: int = _CHARS_PER_TOKEN,
) -> list[str]:
    """Split text into deterministic, word-boundary-aligned chunks.

    Args:
        text: Input text.
        chunk_tokens: Approximate chunk size in tokens.
        overlap_tokens: Approximate overlap in tokens.
        chars_per_token: Conservative token-to-character ratio.

    Returns:
        Non-empty chunks in source order.

    Raises:
        ValueError: If the sizes are invalid or overlap is not smaller.
    """
    chunk_tokens = _positive_int("chunk_tokens", chunk_tokens)
    overlap_tokens = _non_negative_int("overlap_tokens", overlap_tokens)
    chars_per_token = _positive_int("chars_per_token", chars_per_token)
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped:
        return []

    max_chars = chunk_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token
    chunks: list[str] = []
    start = 0
    length = len(stripped)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            window = stripped[start:end]
            boundary = max(window.rfind("\n"), window.rfind(" "))
            if boundary >= max_chars // 2:
                end = start + boundary
        piece = stripped[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


async def _capture(stage: str, awaitable: Any) -> Any:
    """Capture one stage failure without cancelling sibling stages.

    Args:
        stage: Stage name.
        awaitable: Awaitable stage result.

    Returns:
        The stage result or a ``_StageError`` value.
    """
    try:
        return await awaitable
    except asyncio.CancelledError:
        raise
    except Exception as err:
        return _StageError(stage, err)


def _dedupe_articles(articles: Sequence[Article]) -> list[Article]:
    """Return articles with duplicate identifiers removed.

    Args:
        articles: Candidate articles.

    Returns:
        Unique articles preserving the first occurrence.
    """
    seen: set[str] = set()
    unique: list[Article] = []
    for article in articles:
        if article.id in seen:
            continue
        seen.add(article.id)
        unique.append(article)
    return unique


def _batches(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield bounded batches.

    Args:
        items: Items to batch.
        size: Maximum batch size.

    Yields:
        Consecutive batches.
    """
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _enrichment(item: _PreparedArticle) -> ArticleEnrichment:
    """Build a report entry.

    Args:
        item: Written prepared article.

    Returns:
        The public per-article result.
    """
    return ArticleEnrichment(
        article_id=item.article.id,
        summary=item.summary,
        classification=item.classification,
        chunk_ids=tuple(chunk.id for chunk in item.chunks),
        entity_count=len(item.entities),
        relation_count=len(item.relations),
    )


def _failure(article_id: str, stage: str, error: BaseException | str) -> PipelineFailure:
    """Build a redacted failure entry.

    Args:
        article_id: Article identifier.
        stage: Failing stage.
        error: Original exception or an already-redacted message.

    Returns:
        A public failure description.
    """
    message = error if isinstance(error, str) else _describe_error(error)
    return PipelineFailure(article_id=article_id, stage=stage, error=message)


def _describe_error(error: BaseException) -> str:
    """Return a bounded error description.

    Args:
        error: Original exception.

    Returns:
        A short message safe to include in reports and logs.
    """
    message = str(error).strip() or type(error).__name__
    return message[:500]


def _clean_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Normalize candidate labels.

    Args:
        labels: Raw labels.

    Returns:
        Unique non-empty labels.

    Raises:
        ValueError: If a label is not a non-empty string.
    """
    cleaned: list[str] = []
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("labels must be non-empty strings")
        value = label.strip()
        if value not in cleaned:
            cleaned.append(value)
    if not cleaned:
        raise ValueError("at least one classification label is required")
    return tuple(cleaned)


def _positive_int(name: str, value: int) -> int:
    """Validate a positive integer.

    Args:
        name: Field name.
        value: Value to validate.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: int) -> int:
    """Validate a non-negative integer.

    Args:
        name: Field name.
        value: Value to validate.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is not a non-negative integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
