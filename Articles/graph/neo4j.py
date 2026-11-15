"""Async Neo4j adapter for the :class:`~Articles.graph.base.GraphStore` port.

The module imports the official driver lazily, so importing it never opens a
connection and never requires the ``graph`` extra. Writes are batched with
``UNWIND``/``MERGE``; every query is parameterized. Server failures are mapped to
typed graph errors that never contain credentials.
"""

import importlib
import inspect
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, TypeVar

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
from Articles.graph.errors import (
    GraphConfigError,
    GraphConnectionError,
    GraphError,
    GraphQueryError,
)
from Articles.graph.schema import schema_statements
from Articles.Models import Article

__all__ = [
    "Neo4jGraphStore",
    "article_row",
    "batch_iterable",
    "chunk_from_properties",
    "chunk_row",
    "chunk_score_from_record",
    "entity_from_properties",
    "entity_row",
    "mention_row",
    "relation_from_properties",
    "relation_row",
]

T = TypeVar("T")

_DRIVER_INSTALL_HINT = (
    "neo4j driver is not installed; install graph support with "
    "`pip install ai-news-aggregator[graph]`"
)

_ARTICLE_QUERY = """
UNWIND $rows AS row
MERGE (a:Article {id: row.id})
SET a.title = row.title,
    a.url = row.url,
    a.source = row.source,
    a.published_at = row.published_at,
    a.summary = row.summary,
    a.topics = row.topics,
    a.content_hash = row.content_hash
"""

_ARTICLE_HASH_QUERY = """
UNWIND $article_ids AS article_id
MATCH (a:Article {id: article_id})
WHERE a.content_hash IS NOT NULL
RETURN a.id AS id, a.content_hash AS content_hash
"""

_TOPIC_QUERY = """
UNWIND $rows AS row
UNWIND row.topics AS topic
MERGE (t:Topic {name: topic})
WITH t, row
MATCH (a:Article {id: row.id})
MERGE (a)-[:ABOUT]->(t)
"""

_SOURCE_QUERY = """
UNWIND $rows AS row
MERGE (s:Source {name: row.source})
WITH s, row
MATCH (a:Article {id: row.id})
MERGE (a)-[:FROM]->(s)
"""

_STALE_CHUNK_QUERY = """
UNWIND $rows AS row
MATCH (a:Article {id: row.article_id})-[:HAS_CHUNK]->(c:Chunk)
WHERE NOT c.id IN row.chunk_ids
DETACH DELETE c
"""

_CHUNK_QUERY = """
UNWIND $rows AS row
MERGE (c:Chunk {id: row.id})
SET c.article_id = row.article_id,
    c.ordinal = row.ordinal,
    c.text = row.text,
    c.embedding = row.embedding
WITH c, row
MATCH (a:Article {id: row.article_id})
MERGE (a)-[:HAS_CHUNK]->(c)
"""

_ENTITY_QUERY = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.id})
SET e.name = row.name,
    e.type = row.type,
    e.canonical = row.canonical
"""

_RELATION_QUERY = """
UNWIND $rows AS row
MATCH (s:Entity {id: row.source_id})
MATCH (t:Entity {id: row.target_id})
MERGE (s)-[r:RELATED_TO {type: row.type}]->(t)
SET r.confidence = row.confidence,
    r.evidence_chunk_id = row.evidence_chunk_id
"""

_MENTION_QUERY = """
UNWIND $rows AS row
MERGE (c:Chunk {id: row.chunk_id})
MATCH (e:Entity {id: row.entity_id})
MERGE (c)-[:MENTIONS]->(e)
"""

_VECTOR_SEARCH_QUERY = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
YIELD node AS c, score
MATCH (a:Article)-[:HAS_CHUNK]->(c)
WHERE ($article_id IS NULL OR a.id = $article_id)
  AND ($source IS NULL OR a.source = $source)
  AND ($topic IS NULL OR $topic IN a.topics)
RETURN c {.id, .article_id, .ordinal, .text} AS chunk, score
ORDER BY score DESC, chunk.id ASC
LIMIT $k
"""

_INITIAL_CHUNKS_QUERY = """
MATCH (c:Chunk)
WHERE c.id IN $chunk_ids
RETURN c {.id, .article_id, .ordinal, .text} AS chunk
ORDER BY chunk.id ASC
LIMIT $limit
"""

_ENTITIES_QUERY = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE c.id IN $chunk_ids
RETURN DISTINCT e {.id, .name, .type, .canonical} AS entity
ORDER BY entity.id ASC
LIMIT $limit
"""

_RELATIONS_QUERY = """
MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity)
WHERE s.id IN $entity_ids OR t.id IN $entity_ids
RETURN s.id AS source_id,
       t.id AS target_id,
       r.type AS type,
       r.confidence AS confidence,
       r.evidence_chunk_id AS evidence_chunk_id
ORDER BY source_id ASC, target_id ASC, type ASC
LIMIT $limit
"""

_MENTIONED_CHUNKS_QUERY = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.id IN $entity_ids
RETURN DISTINCT c {.id, .article_id, .ordinal, .text} AS chunk
ORDER BY chunk.id ASC
LIMIT $limit
"""


def batch_iterable(items: Iterable[T], size: int = 500) -> Iterator[list[T]]:
    """Split an iterable into bounded lists.

    Args:
        items: Values to batch.
        size: Maximum batch length. Must be at least 1.

    Yields:
        Lists containing at most ``size`` items.

    Raises:
        ValueError: If ``size`` is less than 1.
    """
    if size < 1:
        raise ValueError("batch size must be at least 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert a Neo4j node/property mapping into a plain dictionary.

    Args:
        value: Mapping, Neo4j node, or ``None``.

    Returns:
        A plain dictionary.

    Raises:
        GraphQueryError: If the value cannot be interpreted as a mapping.
    """
    if isinstance(value, Mapping):
        return dict(value)
    items = getattr(value, "items", None)
    if callable(items):
        try:
            return dict(items())
        except (TypeError, ValueError):
            pass
    try:
        return dict(value)
    except (TypeError, ValueError) as err:
        raise GraphQueryError("graph record is not a mapping") from err


def _require_keys(data: Mapping[str, Any], keys: Sequence[str], kind: str) -> None:
    """Ensure a record contains every required key.

    Args:
        data: Record properties.
        keys: Required keys.
        kind: Human-readable record type for diagnostics.

    Raises:
        GraphQueryError: If a key is missing.
    """
    missing = [key for key in keys if key not in data]
    if missing:
        raise GraphQueryError(f"invalid {kind} record: missing {', '.join(missing)}")


def _record_value(record: Any, key: str) -> Any:
    """Read one key from a Neo4j record or dictionary.

    Args:
        record: Record-like object.
        key: Field name.

    Returns:
        The field value.

    Raises:
        GraphQueryError: If the field is absent.
    """
    try:
        return record[key]
    except (KeyError, TypeError) as err:
        raise GraphQueryError(f"graph record is missing {key!r}") from err


def chunk_from_properties(properties: Any, *, include_embedding: bool = False) -> Chunk:
    """Parse chunk properties into a :class:`Chunk`.

    Args:
        properties: Mapping or Neo4j node with chunk properties.
        include_embedding: Retain the embedding when present. Vector-search and
            expansion payloads intentionally omit it.

    Returns:
        A validated chunk.

    Raises:
        GraphQueryError: If required properties are missing or invalid.
    """
    data = _as_mapping(properties)
    _require_keys(data, ("id", "article_id", "ordinal", "text"), "chunk")
    raw_embedding = data.get("embedding") if include_embedding else None
    try:
        embedding = (
            None if raw_embedding is None else tuple(float(value) for value in raw_embedding)
        )
        return Chunk(
            id=str(data["id"]),
            article_id=str(data["article_id"]),
            ordinal=int(data["ordinal"]),
            text=str(data["text"]),
            embedding=embedding,
        )
    except (TypeError, ValueError) as err:
        raise GraphQueryError("invalid chunk record") from err


def entity_from_properties(properties: Any) -> Entity:
    """Parse entity properties into an :class:`Entity`.

    Args:
        properties: Mapping or Neo4j node with entity properties.

    Returns:
        A validated entity.

    Raises:
        GraphQueryError: If required properties are missing or invalid.
    """
    data = _as_mapping(properties)
    _require_keys(data, ("id", "name", "type"), "entity")
    try:
        canonical = data.get("canonical")
        return Entity(
            id=str(data["id"]),
            name=str(data["name"]),
            type=str(data["type"]),
            canonical=None if canonical is None else str(canonical),
        )
    except (TypeError, ValueError) as err:
        raise GraphQueryError("invalid entity record") from err


def relation_from_properties(properties: Any) -> Relation:
    """Parse a relation record into a :class:`Relation`.

    Args:
        properties: Mapping with ``source_id``, ``target_id``, ``type``, and
            optional ``confidence``/``evidence_chunk_id`` fields.

    Returns:
        A validated relation.

    Raises:
        GraphQueryError: If required values are missing or invalid.
    """
    data = _as_mapping(properties)
    _require_keys(data, ("source_id", "target_id", "type"), "relation")
    try:
        confidence = data.get("confidence")
        evidence = data.get("evidence_chunk_id")
        return Relation(
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            type=str(data["type"]),
            confidence=1.0 if confidence is None else float(confidence),
            evidence_chunk_id=None if evidence is None else str(evidence),
        )
    except (TypeError, ValueError) as err:
        raise GraphQueryError("invalid relation record") from err


def chunk_score_from_record(record: Any) -> tuple[Chunk, float]:
    """Parse one vector-search record.

    Args:
        record: Neo4j record containing ``chunk`` and ``score`` fields.

    Returns:
        The chunk and its cosine score.

    Raises:
        GraphQueryError: If the record is malformed.
    """
    chunk = chunk_from_properties(_record_value(record, "chunk"))
    try:
        score = float(_record_value(record, "score"))
    except (TypeError, ValueError) as err:
        raise GraphQueryError("invalid vector score") from err
    return chunk, score


def _record_mapping(record: Any) -> dict[str, Any]:
    """Convert a Neo4j record into a plain dictionary.

    Args:
        record: Record-like object.

    Returns:
        A plain dictionary.

    Raises:
        GraphQueryError: If the record cannot be converted.
    """
    data_method = getattr(record, "data", None)
    if callable(data_method):
        try:
            return dict(data_method())
        except (TypeError, ValueError):
            pass
    try:
        return dict(record)
    except (TypeError, ValueError) as err:
        raise GraphQueryError("invalid graph record") from err


def article_row(article: Article) -> dict[str, Any]:
    """Build the parameter row for an article upsert.

    Args:
        article: Article to serialize.

    Returns:
        JSON/Neo4j-friendly article properties.
    """
    return {
        "id": article.id,
        "title": article.title,
        "url": article.url,
        "source": article.source,
        "published_at": (
            article.published_at.isoformat() if article.published_at is not None else None
        ),
        "summary": article.summary,
        "topics": list(article.topics),
        "content_hash": article_content_hash(article),
    }


def chunk_row(chunk: Chunk) -> dict[str, Any]:
    """Build the parameter row for a chunk upsert.

    Args:
        chunk: Chunk to serialize.

    Returns:
        Neo4j-friendly chunk properties.
    """
    return {
        "id": chunk.id,
        "article_id": chunk.article_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "embedding": None if chunk.embedding is None else list(chunk.embedding),
    }


def entity_row(entity: Entity) -> dict[str, Any]:
    """Build the parameter row for an entity upsert.

    Args:
        entity: Entity to serialize.

    Returns:
        Neo4j-friendly entity properties.
    """
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type,
        "canonical": entity.canonical,
    }


def relation_row(relation: Relation) -> dict[str, Any]:
    """Build the parameter row for a relation upsert.

    Args:
        relation: Relation to serialize.

    Returns:
        Neo4j-friendly relation properties.
    """
    return {
        "source_id": relation.source_id,
        "target_id": relation.target_id,
        "type": relation.type,
        "confidence": relation.confidence,
        "evidence_chunk_id": relation.evidence_chunk_id,
    }


def mention_row(mention: Mention) -> dict[str, Any]:
    """Build the parameter row for a mention upsert.

    Args:
        mention: Mention to serialize.

    Returns:
        Neo4j-friendly mention identifiers.
    """
    return {"chunk_id": mention.chunk_id, "entity_id": mention.entity_id}


def _map_error(err: BaseException, action: str) -> GraphError:
    """Map a driver/server failure to a typed graph error.

    Args:
        err: Original exception.
        action: Short description of the failed operation.

    Returns:
        A :class:`GraphConnectionError` for availability/authentication failures
        and :class:`GraphQueryError` otherwise. The message never includes
        credentials.
    """
    name = type(err).__name__.lower()
    message = str(err).strip() or name
    markers = ("unavailable", "connection", "sessionexpired", "auth", "security")
    if any(marker in name for marker in markers):
        return GraphConnectionError(f"{action} failed: {message}")
    return GraphQueryError(f"{action} failed: {message}")


def _create_async_driver(uri: str, user: str, password: str) -> Any:
    """Import the Neo4j driver and construct an async driver.

    Args:
        uri: Connection URI.
        user: Database user.
        password: Database password. Never included in error messages.

    Returns:
        An async Neo4j driver.

    Raises:
        GraphConfigError: If the optional ``neo4j`` package is missing.
        GraphError: If driver construction fails.
    """
    try:
        module = importlib.import_module("neo4j")
    except ImportError as err:
        raise GraphConfigError(_DRIVER_INSTALL_HINT) from err
    factory = getattr(module, "AsyncGraphDatabase", None)
    if factory is None:
        raise GraphConfigError(_DRIVER_INSTALL_HINT)
    try:
        return factory.driver(uri, auth=(user, password))
    except Exception as err:
        raise _map_error(err, "Neo4j driver creation") from err


class Neo4jGraphStore:
    """Graph store backed by the official async Neo4j driver.

    Construction validates configuration but never connects and never imports
    the driver. Call :meth:`initialize` once to apply the schema, then use the
    ``GraphStore`` methods.

    Args:
        uri: Bolt/Neo4j connection URI.
        user: Database user.
        password: Database password.
        database: Target database name.
        driver: Optional pre-built async driver, primarily for tests. The store
            closes whichever driver is present when :meth:`aclose` runs.
        dim: Vector dimension; must match the ``chunk_embedding`` index.

    Raises:
        GraphConfigError: If a required setting is missing or ``dim`` is not a
            positive integer.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        *,
        driver: Any = None,
        dim: int = 384,
    ) -> None:
        if not uri.strip():
            raise GraphConfigError("Neo4j URI must not be empty")
        if not user.strip():
            raise GraphConfigError("Neo4j user must not be empty")
        if not (password or "").strip():
            raise GraphConfigError("Neo4j password must not be empty")
        if not database.strip():
            raise GraphConfigError("Neo4j database must not be empty")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise GraphConfigError("graph dimension must be a positive integer")
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = driver
        self.dim = dim

    def _ensure_driver(self) -> Any:
        """Return the driver, creating it on first use.

        Returns:
            An async Neo4j driver.

        Raises:
            GraphConfigError: If the optional driver package is missing.
            GraphError: If driver creation fails.
        """
        if self._driver is None:
            self._driver = _create_async_driver(self._uri, self._user, self._password)
        return self._driver

    async def _run(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """Execute one parameterized query and materialize its records.

        Args:
            query: Cypher statement.
            parameters: Query parameters.

        Returns:
            Result records.

        Raises:
            GraphConnectionError: On connection/authentication failures.
            GraphQueryError: On query failures.
        """
        driver = self._ensure_driver()
        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(query, dict(parameters or {}))
                return [record async for record in result]
        except GraphError:
            raise
        except Exception as err:
            raise _map_error(err, "graph query") from err

    async def initialize(self) -> None:
        """Apply all idempotent schema constraints and indexes."""
        for statement in schema_statements(self.dim):
            await self._run(statement)

    async def healthcheck(self) -> bool:
        """Return whether the database answers ``RETURN 1``.

        Returns:
            ``True`` on success, ``False`` on any connection or query failure.
        """
        try:
            records = await self._run("RETURN 1 AS ok")
        except Exception:
            return False
        return bool(records)

    async def aclose(self) -> None:
        """Close the driver if one has been created."""
        driver = self._driver
        self._driver = None
        if driver is None:
            return
        closer = getattr(driver, "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception as err:
            raise _map_error(err, "Neo4j driver close") from err

    async def article_hashes(self, article_ids: Sequence[str]) -> dict[str, str]:
        """Return stored content hashes for known articles.

        Args:
            article_ids: Identifiers to look up.

        Returns:
            A mapping of known article identifier to content hash.

        Raises:
            GraphQueryError: If a returned record is malformed.
        """
        hashes: dict[str, str] = {}
        for batch in batch_iterable(list(dict.fromkeys(article_ids))):
            records = await self._run(_ARTICLE_HASH_QUERY, {"article_ids": batch})
            for record in records:
                data = _record_mapping(record)
                article_id = data.get("id")
                content_hash = data.get("content_hash")
                if not isinstance(article_id, str) or not isinstance(content_hash, str):
                    raise GraphQueryError("invalid article hash record")
                hashes[article_id] = content_hash
        return hashes

    async def upsert_articles(
        self,
        articles: Sequence[Article],
        chunks: Sequence[Chunk],
    ) -> None:
        """Upsert articles and replace each article's chunk set exactly.

        Args:
            articles: Articles to upsert.
            chunks: Chunks belonging to those articles.
        """
        if not articles and not chunks:
            return

        rows = [article_row(article) for article in articles]
        for batch in batch_iterable(rows):
            await self._run(_ARTICLE_QUERY, {"rows": batch})
            await self._run(_TOPIC_QUERY, {"rows": batch})
            await self._run(_SOURCE_QUERY, {"rows": batch})

        chunk_ids_by_article: dict[str, set[str]] = {}
        for chunk in chunks:
            chunk_ids_by_article.setdefault(chunk.article_id, set()).add(chunk.id)
        stale_rows = [
            {
                "article_id": article.id,
                "chunk_ids": sorted(chunk_ids_by_article.get(article.id, set())),
            }
            for article in articles
        ]
        for batch in batch_iterable(stale_rows):
            await self._run(_STALE_CHUNK_QUERY, {"rows": batch})

        for batch in batch_iterable([chunk_row(chunk) for chunk in chunks]):
            await self._run(_CHUNK_QUERY, {"rows": batch})

    async def upsert_entities(
        self,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        mentions: Sequence[Mention] = (),
    ) -> None:
        """Upsert entities, relations, and mention links.

        Args:
            entities: Entities to upsert.
            relations: Relations to upsert.
            mentions: Chunk-to-entity links to add.
        """
        for batch in batch_iterable([entity_row(entity) for entity in entities]):
            await self._run(_ENTITY_QUERY, {"rows": batch})
        for batch in batch_iterable([relation_row(relation) for relation in relations]):
            await self._run(_RELATION_QUERY, {"rows": batch})
        for batch in batch_iterable([mention_row(mention) for mention in mentions]):
            await self._run(_MENTION_QUERY, {"rows": batch})

    async def vector_search(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search the ``chunk_embedding`` vector index.

        Args:
            embedding: Query vector.
            k: Maximum number of results. Must be at least 1.
            filters: Optional ``article_id``, ``source``, or ``topic`` filters.

        Returns:
            Chunk/score pairs ordered by descending score, then chunk id.
            Embeddings are omitted from returned chunks.

        Raises:
            ValueError: If ``k`` is less than 1.
            GraphConfigError: If a filter key is unsupported.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        active = validate_filters(filters)
        parameters: dict[str, Any] = {
            "k": k,
            "embedding": [float(value) for value in embedding],
            "article_id": active.get("article_id"),
            "source": active.get("source"),
            "topic": active.get("topic"),
        }
        records = await self._run(_VECTOR_SEARCH_QUERY, parameters)
        results = [chunk_score_from_record(record) for record in records]
        results.sort(key=lambda item: (-item[1], item[0].id))
        return results[:k]

    async def expand(
        self,
        chunk_ids: Sequence[str],
        *,
        depth: int = 1,
        limit: int = 50,
    ) -> GraphContext:
        """Expand from seed chunks through mentions and relations.

        Expansion is bounded to two hops and every collection is capped by
        ``limit``. The returned chunks, entities, and relations use stable
        ordering.

        Args:
            chunk_ids: Seed chunk identifiers. Missing identifiers are ignored.
            depth: Number of hops, 1 or 2.
            limit: Maximum chunks, entities, and relations to return.

        Returns:
            A bounded subgraph.

        Raises:
            ValueError: If ``depth`` is outside ``[1, 2]`` or ``limit`` is
                less than 1.
        """
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if depth > 2:
            raise ValueError("depth must be at most 2")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        seeds = list(dict.fromkeys(chunk_ids))
        if not seeds:
            return GraphContext()

        records = await self._run(
            _INITIAL_CHUNKS_QUERY,
            {"chunk_ids": seeds, "limit": limit},
        )
        selected: dict[str, Chunk] = {}
        for record in records:
            chunk = chunk_from_properties(_record_value(record, "chunk"))
            if chunk.id not in selected and len(selected) < limit:
                selected[chunk.id] = chunk
        frontier = list(selected)

        entities: dict[str, Entity] = {}
        relations: dict[tuple[str, str, str], Relation] = {}

        for _ in range(depth):
            if not frontier:
                break
            remaining = limit - len(entities)
            if remaining > 0:
                records = await self._run(
                    _ENTITIES_QUERY,
                    {"chunk_ids": frontier, "limit": remaining},
                )
                for record in records:
                    entity = entity_from_properties(_record_value(record, "entity"))
                    if entity.id in entities or len(entities) >= limit:
                        continue
                    entities[entity.id] = entity

            if not entities:
                break

            remaining_relations = limit - len(relations)
            if remaining_relations > 0:
                records = await self._run(
                    _RELATIONS_QUERY,
                    {"entity_ids": sorted(entities), "limit": remaining_relations},
                )
                for record in records:
                    relation = relation_from_properties(_record_mapping(record))
                    key = relation_key(
                        relation.source_id,
                        relation.target_id,
                        relation.type,
                    )
                    if key in relations or len(relations) >= limit:
                        continue
                    relations[key] = relation

            remaining_chunks = limit - len(selected)
            if remaining_chunks <= 0:
                break
            records = await self._run(
                _MENTIONED_CHUNKS_QUERY,
                {"entity_ids": sorted(entities), "limit": remaining_chunks},
            )
            new_frontier: list[str] = []
            for record in records:
                chunk = chunk_from_properties(_record_value(record, "chunk"))
                if chunk.id in selected or len(selected) >= limit:
                    continue
                selected[chunk.id] = chunk
                new_frontier.append(chunk.id)
            frontier = new_frontier

        return GraphContext(
            chunks=tuple(selected[chunk_id] for chunk_id in sorted(selected)),
            entities=tuple(entities[entity_id] for entity_id in sorted(entities)),
            relations=tuple(relations[key] for key in sorted(relations)),
        )
