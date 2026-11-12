"""FreeNewsApi.io news provider implementation for AI-News-Aggregator.

The provider is a thin adapter over the FreeNewsApi.io REST API. Asynchronous
methods are the primary interface; legacy synchronous aliases are retained for
backwards compatibility and share the same request plumbing as the async path.
Constructors perform no network I/O, and metadata such as the topic list is
loaded lazily on first use.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from Articles.BaseProvider import BaseProvider
from Articles.config import Settings
from Articles.Models import Article
from Articles.Vars import Category

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.freenewsapi.io/v1"

__all__ = ["DEFAULT_BASE_URL", "NewsAPI"]


class NewsAPI(BaseProvider):
    """News provider interfacing with FreeNewsApi.io over HTTPX."""

    def __init__(
        self,
        apiKey: str | None = None,  # noqa: N803 - public compatibility name
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
        max_concurrency: int = 5,
    ) -> None:
        """Initialize the FreeNewsApi client without performing network I/O.

        Args:
            apiKey: FreeNewsApi.io API key sent via the ``x-api-key`` header.
                Falls back to ``FREENEWSAPI_API_KEY``, ``NEWS_API_KEY``, or
                ``NEWSAPI_KEY`` when omitted.
            base_url: Base endpoint for FreeNewsApi.
            timeout: HTTP read/write timeout in seconds.
            client: Optional pre-configured asynchronous HTTP client.
            sync_client: Optional pre-configured synchronous HTTP client used
                by legacy aliases.
            max_concurrency: Maximum concurrent asynchronous requests.
        """
        resolved_key = (
            apiKey
            or os.getenv("FREENEWSAPI_API_KEY")
            or os.getenv("NEWS_API_KEY")
            or os.getenv("NEWSAPI_KEY")
        )
        super().__init__(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            client=client,
            sync_client=sync_client,
            max_concurrency=max_concurrency,
        )
        if self.api_key:
            auth_header = {"x-api-key": self.api_key}
            self.client.headers.update(auth_header)
            self.sync_client.headers.update(auth_header)

        self._sources_cache: dict[str, Any] | None = None
        self._sources_loaded = False

    # -------------------------------------------------------------------------
    # Lazy metadata
    # -------------------------------------------------------------------------

    @property
    def _sources(self) -> dict[str, Any] | None:
        """Return the topic catalogue, fetching it on first access.

        Returns:
            The cached ``/topics`` payload, or ``None`` when the request fails.
        """
        if not self._sources_loaded:
            self._sources_cache = self._getsource()
            self._sources_loaded = True
        return self._sources_cache

    # -------------------------------------------------------------------------
    # Core FreeNewsApi methods (synchronous, legacy)
    # -------------------------------------------------------------------------

    def get_topics(self) -> dict[str, Any] | None:
        """Retrieve available news topics from ``/v1/topics``.

        Returns:
            The parsed topics payload, or ``None`` on failure.
        """
        return self._make_request_sync("/topics")

    def get_publishers(self) -> dict[str, Any] | None:
        """Retrieve available publishers from ``/v1/publishers``.

        Returns:
            The parsed publishers payload, or ``None`` on failure.
        """
        return self._make_request_sync("/publishers")

    def get_news(
        self,
        in_title: str | None = None,
        topic: str | None = None,
        publisher_uuid: str | None = None,
        author_uuid: str | None = None,
        language: str | None = None,
        country: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        offset: int | None = None,
        cursor: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query news articles via ``/v1/news``.

        Args:
            in_title: Keyword filter applied to the article title.
            topic: Topic/category name, for example ``"technology"``.
            publisher_uuid: UUID of a specific publisher.
            author_uuid: UUID of a specific author.
            language: Language code, for example ``"en"``.
            country: Country code, for example ``"us"``.
            published_after: ISO timestamp filter (start).
            published_before: ISO timestamp filter (end).
            offset: Offset for pagination.
            cursor: Cursor for pagination.
            limit: Maximum number of items requested and returned.
            **kwargs: Extra parameters forwarded to the request.

        Returns:
            JSON response containing ``data`` and ``meta``, or ``None`` on
            failure. When ``limit`` is provided the ``data`` list is truncated
            client-side as a hard guarantee.
        """
        params = self._news_params(
            limit=limit,
            in_title=in_title,
            topic=topic,
            publisher_uuid=publisher_uuid,
            author_uuid=author_uuid,
            language=language,
            country=country,
            published_after=published_after,
            published_before=published_before,
            offset=offset,
            cursor=cursor,
            **kwargs,
        )
        payload = self._make_request_sync("/news", params=params)
        return self._truncate(payload, limit)

    def get_article_details(self, uuid: str) -> dict[str, Any] | None:
        """Fetch full article details by UUID from ``/v1/details``.

        Args:
            uuid: Unique identifier of the article.

        Returns:
            Detailed article JSON payload, or ``None`` on failure.
        """
        return self._make_request_sync("/details", params={"uuid": uuid})

    def get_news_by_keyword(
        self,
        keyword: str,
        topic: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query news by title keyword with optional topic refinement.

        Args:
            keyword: Keyword matched against the article title.
            topic: Optional topic/category filter.
            **kwargs: Extra parameters forwarded to :meth:`get_news`.

        Returns:
            Parsed ``/news`` response, or ``None`` on failure.
        """
        return self.get_news(in_title=keyword, topic=topic, **kwargs)

    def get_news_by_topic(
        self,
        topic: str = Category,
        in_title: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query news by topic category.

        Args:
            topic: Topic/category filter.
            in_title: Optional title keyword filter.
            **kwargs: Extra parameters forwarded to :meth:`get_news`.

        Returns:
            Parsed ``/news`` response, or ``None`` on failure.
        """
        return self.get_news(topic=topic, in_title=in_title, **kwargs)

    def get_news_by_publisher(
        self,
        publisher_uuid: str,
        in_title: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query news published by a specific publisher UUID.

        Args:
            publisher_uuid: UUID of the publisher.
            in_title: Optional title keyword filter.
            **kwargs: Extra parameters forwarded to :meth:`get_news`.

        Returns:
            Parsed ``/news`` response, or ``None`` on failure.
        """
        return self.get_news(publisher_uuid=publisher_uuid, in_title=in_title, **kwargs)

    # -------------------------------------------------------------------------
    # BaseProvider abstract method implementations
    # -------------------------------------------------------------------------

    async def fetch_recent(self, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch recent news items, defaulting to the configured category.

        Args:
            limit: Maximum number of raw items to return.
            **kwargs: Provider-specific filters, including an optional ``topic``.

        Returns:
            A list of raw provider item dictionaries.
        """
        topic = kwargs.pop("topic", Category)
        return await self._fetch_news_list(limit=limit, topic=topic, **kwargs)

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search news items by title keyword.

        Args:
            query: Keyword matched against the article title.
            limit: Maximum number of raw items to return.
            **kwargs: Provider-specific filters.

        Returns:
            A list of raw provider item dictionaries.
        """
        return await self._fetch_news_list(limit=limit, in_title=query, **kwargs)

    async def fetch_articles(self, limit: int = 10, **kwargs: Any) -> list[Article]:
        """Fetch normalized articles with bounded pagination and deduplication.

        Pagination continues by offset (or cursor) until ``limit`` is reached,
        ``max_pages`` is exhausted, ``max_items`` is hit, or an empty page is
        returned. Items are deduplicated by UUID and mapped to :class:`Article`.

        Args:
            limit: Maximum number of articles to return.
            **kwargs: Provider filters plus optional ``max_pages``,
                ``max_items``, ``page_size``, ``topic``, ``offset``, and
                ``cursor`` overrides.

        Returns:
            A list of normalized articles ordered as returned by the provider.
        """
        if limit <= 0:
            return []

        settings = Settings.from_env()
        max_pages = max(1, int(kwargs.pop("max_pages", settings.max_pages)))
        max_items = max(1, int(kwargs.pop("max_items", settings.max_items)))
        page_size = max(1, int(kwargs.pop("page_size", settings.page_size)))
        default_topic = settings.topics[0] if settings.topics else Category
        topic = kwargs.pop("topic", default_topic)
        offset = int(kwargs.pop("offset", 0) or 0)
        cursor = kwargs.pop("cursor", None)

        target = min(limit, max_items)
        seen: set[str] = set()
        articles: list[Article] = []
        pages = 0

        while pages < max_pages and len(articles) < target:
            remaining = target - len(articles)
            params = self._news_params(
                limit=min(page_size, remaining),
                topic=topic,
                offset=offset if cursor is None else None,
                cursor=cursor,
                **kwargs,
            )
            payload = await self._make_request("/news", params=params)
            items = self._extract_items(payload)
            if not items:
                break

            for item in items:
                article = self._to_article(item, topic)
                if article is None or article.id in seen:
                    continue
                seen.add(article.id)
                articles.append(article)
                if len(articles) >= target:
                    break

            pages += 1
            if cursor is not None:
                cursor = self._next_cursor(payload)
                if cursor is None:
                    break
            else:
                offset += len(items)

        return articles

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _news_params(
        *,
        limit: int | None = None,
        in_title: str | None = None,
        topic: str | None = None,
        publisher_uuid: str | None = None,
        author_uuid: str | None = None,
        language: str | None = None,
        country: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        offset: int | None = None,
        cursor: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build the query parameters for a ``/news`` request.

        Args:
            limit: Requested page size.
            in_title: Title keyword filter.
            topic: Topic/category filter.
            publisher_uuid: Publisher UUID filter.
            author_uuid: Author UUID filter.
            language: Language code filter.
            country: Country code filter.
            published_after: Lower publication bound.
            published_before: Upper publication bound.
            offset: Offset pagination token.
            cursor: Cursor pagination token.
            **extra: Additional provider parameters.

        Returns:
            A parameter dictionary; ``None`` values are removed by the HTTP
            layer before the request is sent.
        """
        return {
            "in_title": in_title,
            "topic": topic,
            "publisher_uuid": publisher_uuid,
            "author_uuid": author_uuid,
            "language": language,
            "country": country,
            "published_after": published_after,
            "published_before": published_before,
            "offset": offset,
            "cursor": cursor,
            "limit": limit,
            **extra,
        }

    @staticmethod
    def _extract_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Extract raw item dictionaries from a ``/news`` payload.

        Args:
            payload: Parsed response payload, or ``None``.

        Returns:
            Item dictionaries from the ``data`` (or ``articles``) field.
        """
        if not payload:
            return []
        data = payload.get("data")
        if data is None:
            data = payload.get("articles")
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _truncate(
        payload: dict[str, Any] | None,
        limit: int | None,
    ) -> dict[str, Any] | None:
        """Truncate a payload's ``data`` list to ``limit`` items.

        Args:
            payload: Parsed response payload, or ``None``.
            limit: Maximum number of items, or ``None`` for no truncation.

        Returns:
            The original payload, or a shallow copy with truncated ``data``.
        """
        if payload is None or limit is None:
            return payload
        data = payload.get("data")
        if isinstance(data, list) and len(data) > max(limit, 0):
            return {**payload, "data": data[: max(limit, 0)]}
        return payload

    async def _fetch_news_list(
        self,
        *,
        limit: int,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Fetch one page of raw items through the asynchronous path.

        Args:
            limit: Maximum number of items to return.
            **filters: Provider filters forwarded to ``/news``.

        Returns:
            A truncated list of raw item dictionaries.
        """
        if limit <= 0:
            return []
        params = self._news_params(limit=limit, **filters)
        payload = await self._make_request("/news", params=params)
        return self._extract_items(payload)[:limit]

    @staticmethod
    def _next_cursor(payload: dict[str, Any] | None) -> str | None:
        """Read the next cursor from a response payload.

        Args:
            payload: Parsed response payload.

        Returns:
            The next cursor as a string, or ``None`` when absent.
        """
        if not payload:
            return None
        meta = payload.get("meta")
        if isinstance(meta, dict):
            for key in ("next_cursor", "nextCursor", "cursor"):
                value = meta.get(key)
                if value:
                    return str(value)
        for key in ("next_cursor", "nextCursor"):
            value = payload.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _as_text(value: Any) -> str | None:
        """Normalize a scalar value to a non-empty string.

        Args:
            value: Candidate value from a provider payload.

        Returns:
            A stripped string, or ``None`` when the value is missing/blank.
        """
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, int):
            return str(value)
        return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        """Parse an ISO-8601 timestamp, handling a trailing ``Z``.

        Args:
            value: ISO string, epoch number, or ``None``.

        Returns:
            A timezone-aware datetime in UTC when input was naive, or ``None``
            when the value cannot be parsed.
        """
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value, tz=UTC)
            except (OverflowError, OSError, ValueError):
                return None
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    @classmethod
    def _to_article(cls, item: dict[str, Any], topic: str | None) -> Article | None:
        """Map a raw provider item to a normalized article.

        Args:
            item: Raw item dictionary.
            topic: Topic used for the article's topic tuple.

        Returns:
            A normalized :class:`Article`, or ``None`` when the item is missing
            a required identifier, title, or URL.
        """
        article_id = cls._as_text(item.get("uuid")) or cls._as_text(item.get("id"))
        title = cls._as_text(item.get("title"))
        url = cls._as_text(item.get("url")) or cls._as_text(item.get("link"))
        if article_id is None or title is None or url is None:
            logger.debug("Skipping malformed article missing id, title, or url")
            return None

        published_at = cls._parse_datetime(
            item.get("published_at")
            or item.get("publishedAt")
            or item.get("published")
            or item.get("date")
        )
        summary = (
            cls._as_text(item.get("description"))
            or cls._as_text(item.get("summary"))
            or cls._as_text(item.get("snippet"))
        )

        publisher = item.get("publisher")
        publisher_name: str | None = None
        publisher_uuid: str | None = None
        if isinstance(publisher, dict):
            publisher_name = cls._as_text(publisher.get("name"))
            publisher_uuid = cls._as_text(publisher.get("uuid"))
        elif isinstance(publisher, str):
            publisher_name = cls._as_text(publisher)
        publisher_name = publisher_name or cls._as_text(item.get("publisher_name"))
        publisher_uuid = publisher_uuid or cls._as_text(item.get("publisher_uuid"))
        source = publisher_name or publisher_uuid or "freenewsapi"

        normalized_topic = cls._as_text(topic)
        topics = (normalized_topic,) if normalized_topic else ()

        return Article(
            id=article_id,
            title=title,
            url=url,
            source=source,
            published_at=published_at,
            summary=summary,
            topics=topics,
            raw=item,
        )

    # -------------------------------------------------------------------------
    # Legacy & compatibility aliases (synchronous, thin)
    # -------------------------------------------------------------------------

    def _getsource(self) -> dict[str, Any] | None:
        """Return available topics for news.

        Returns:
            The ``/topics`` payload, or ``None`` on failure.
        """
        return self.get_topics()

    def _source(self) -> dict[str, Any] | None:
        """Legacy alias for :meth:`_getsource`.

        Returns:
            The ``/topics`` payload, or ``None`` on failure.
        """
        return self._getsource()

    @staticmethod
    def _combine_results(
        results: list[dict[str, Any] | None],
    ) -> list[dict[str, Any] | None] | None:
        """Return ``None`` when every legacy query failed.

        Args:
            results: Results from the composed legacy queries.

        Returns:
            The original list, or ``None`` when all entries are ``None``.
        """
        if all(result is None for result in results):
            return None
        return results

    def _NewsQueryByKeyword(  # noqa: N802 - legacy public name
        self,
        topic: str,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy keyword query returning title-search and topic results.

        Args:
            topic: Keyword used for the search.

        Returns:
            A two-element result list, or ``None`` when both queries fail.
        """
        return self._combine_results(
            [
                self.get_news_by_keyword(keyword=topic),
                self.get_news_by_topic(topic=Category, in_title=topic),
            ]
        )

    def _NewsQueryByCategory(  # noqa: N802 - legacy public name
        self,
        topic: str | None = None,
        category: str = Category,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy category query searching headlines by category and keyword.

        Args:
            topic: Optional title keyword.
            category: Topic/category filter.

        Returns:
            A two-element result list, or ``None`` when both queries fail.
        """
        return self._combine_results(
            [
                self.get_news_by_topic(topic=category, in_title=topic),
                self.get_news(in_title=topic),
            ]
        )

    def _NewsQueryBySource(  # noqa: N802 - legacy public name
        self,
        source: str,
        topic: str | None = None,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy source query filtering by publisher UUID and topic keyword.

        Args:
            source: Publisher UUID.
            topic: Optional title keyword.

        Returns:
            A two-element result list, or ``None`` when both queries fail.
        """
        return self._combine_results(
            [
                self.get_news_by_publisher(publisher_uuid=source, in_title=topic),
                self.get_news(in_title=topic),
            ]
        )

    def _NewsQueryEverything(  # noqa: N802 - legacy public name
        self,
        source: str | None = None,
        topic: str | None = None,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy query returning publisher and keyword result sets.

        Args:
            source: Optional publisher UUID.
            topic: Optional title keyword.

        Returns:
            A two-element result list, or ``None`` when both queries fail.
        """
        return self._combine_results(
            [
                self.get_news(publisher_uuid=source, in_title=topic),
                self.get_news(in_title=topic),
            ]
        )

    def _NewsQueryHeadlines(  # noqa: N802 - legacy public name
        self,
        source: str | None = None,
        topic: str | None = None,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy headlines query by source or topic.

        Args:
            source: Optional publisher UUID.
            topic: Optional title keyword.

        Returns:
            A two-element result list, or ``None`` when both queries fail.
        """
        return self._combine_results(
            [
                self.get_news(publisher_uuid=source, in_title=topic),
                self.get_news(in_title=topic),
            ]
        )
