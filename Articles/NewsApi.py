"""FreeNewsApi.io news provider implementation for AI-News-Aggregator.

Provides access to news articles, topics, and article details using
freenewsapi.io REST endpoints via the requests library.
"""

import os
from typing import Any

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from Articles.BaseProvider import BaseProvider
    from Articles.Vars import Category
except ImportError:
    from BaseProvider import BaseProvider
    from Vars import Category


DEFAULT_BASE_URL = "https://api.freenewsapi.io/v1"


class NewsAPI(BaseProvider):
    """News provider interfacing with FreeNewsApi.io using requests."""

    def __init__(
        self,
        apiKey: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 10,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the FreeNewsApi client.

        Args:
            apiKey: FreeNewsApi.io API key (sent via x-api-key header).
                    If None, resolved from FREENEWSAPI_API_KEY, NEWS_API_KEY, or NEWSAPI_KEY in .env/environment.
            base_url: Base endpoint for FreeNewsApi.
            timeout: HTTP request timeout in seconds.
            session: Optional pre-configured requests.Session instance.
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
            session=session,
        )
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key

        self._sources = self._getsource()

    # -------------------------------------------------------------------------
    # Core FreeNewsApi Methods
    # -------------------------------------------------------------------------

    def get_topics(self) -> dict[str, Any] | None:
        """Retrieve available news topics from /v1/topics."""
        return self._make_request("/topics")

    def get_publishers(self) -> dict[str, Any] | None:
        """Retrieve available publishers from /v1/publishers."""
        return self._make_request("/publishers")

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
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query news articles via /v1/news with supported FreeNewsApi filters.

        Args:
            in_title: Keyword filter applied to the article title.
            topic: Topic/category name (e.g. 'technology').
            publisher_uuid: UUID of specific publisher.
            author_uuid: UUID of specific author.
            language: Language code (e.g. 'en').
            country: Country code (e.g. 'us').
            published_after: ISO timestamp filter (start).
            published_before: ISO timestamp filter (end).
            offset: Offset for pagination.
            cursor: Cursor for pagination.
            **kwargs: Extra parameters passed to the request.

        Returns:
            JSON response dictionary containing 'data' and 'meta', or None on failure.
        """
        params = {
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
            **kwargs,
        }
        return self._make_request("/news", params=params)

    def get_article_details(self, uuid: str) -> dict[str, Any] | None:
        """Fetch full article details and body text by UUID from /v1/details.

        Args:
            uuid: Unique identifier of the article.

        Returns:
            Detailed article JSON payload, or None on failure.
        """
        return self._make_request("/details", params={"uuid": uuid})

    def get_news_by_keyword(
        self,
        keyword: str,
        topic: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query news by title keyword with optional topic refinement."""
        return self.get_news(in_title=keyword, topic=topic, **kwargs)

    def get_news_by_topic(
        self,
        topic: str = Category,
        in_title: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query news by topic category."""
        return self.get_news(topic=topic, in_title=in_title, **kwargs)

    def get_news_by_publisher(
        self,
        publisher_uuid: str,
        in_title: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query news published by a specific publisher UUID."""
        return self.get_news(publisher_uuid=publisher_uuid, in_title=in_title, **kwargs)

    # -------------------------------------------------------------------------
    # BaseProvider Abstract Method Implementations
    # -------------------------------------------------------------------------

    def fetch_recent(self, limit: int = 10, **kwargs: Any) -> dict[str, Any] | None:
        """Fetch recent news articles (defaulting to configured Category)."""
        topic = kwargs.pop("topic", Category)
        return self.get_news(topic=topic, **kwargs)

    def search(self, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any] | None:
        """Search news articles by title keyword."""
        return self.get_news(in_title=query, **kwargs)

    # -------------------------------------------------------------------------
    # Legacy & Compatibility Aliases
    # -------------------------------------------------------------------------

    def _getsource(self) -> dict[str, Any] | None:
        """Return available topics for news."""
        return self.get_topics()

    def _source(self) -> dict[str, Any] | None:
        """Alias for _getsource."""
        return self._getsource()

    def _NewsQueryByKeyword(self, topic: str) -> list[dict[str, Any] | None] | None:
        """Legacy keyword query returning title search and topic headline results."""
        self._getsource()
        res1 = self.get_news_by_keyword(keyword=topic)
        res2 = self.get_news_by_topic(topic=Category, in_title=topic)
        if res1 is None and res2 is None:
            return None
        return [res1, res2]

    def _NewsQueryByCategory(
        self,
        topic: str | None = None,
        category: str = Category,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy category query searching headlines by category and keyword."""
        self._getsource()
        res1 = self.get_news_by_topic(topic=category, in_title=topic)
        res2 = self.get_news(in_title=topic)
        if res1 is None and res2 is None:
            return None
        return [res1, res2]

    def _NewsQueryBySource(
        self,
        source: str,
        topic: str | None = None,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy source query filtering by publisher_uuid and topic keyword."""
        self._getsource()
        res1 = self.get_news_by_publisher(publisher_uuid=source, in_title=topic)
        res2 = self.get_news(in_title=topic)
        if res1 is None and res2 is None:
            return None
        return [res1, res2]

    def _NewsQueryEverything(
        self,
        source: str | None = None,
        topic: str | None = None,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy query returning filtered results by publisher/keyword."""
        self._getsource()
        res1 = self.get_news(publisher_uuid=source, in_title=topic)
        res2 = self.get_news(in_title=topic)
        if res1 is None and res2 is None:
            return None
        return [res1, res2]

    def _NewsQueryHeadlines(
        self,
        source: str | None = None,
        topic: str | None = None,
    ) -> list[dict[str, Any] | None] | None:
        """Legacy headlines query by source or topic."""
        self._getsource()
        res1 = self.get_news(publisher_uuid=source, in_title=topic)
        res2 = self.get_news(in_title=topic)
        if res1 is None and res2 is None:
            return None
        return [res1, res2]