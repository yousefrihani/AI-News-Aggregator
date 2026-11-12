"""Hermetic unit tests for :mod:`Articles.NewsApi` and :mod:`Articles.config`.

All HTTP traffic is served by ``httpx.MockTransport``. Tests cover lazy
metadata, limit forwarding, bounded pagination, deduplication, typed errors,
legacy aliases, and environment-backed settings.
"""

import unittest
from datetime import UTC
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from Articles.BaseProvider import BaseProvider
from Articles.config import Settings
from Articles.errors import ProviderAuthError, ProviderConfigError, ProviderHTTPError
from Articles.NewsApi import DEFAULT_BASE_URL, NewsAPI
from Articles.Vars import Category

BASE_URL = "https://api.example.test/v1"


def make_item(
    uuid: str,
    *,
    title: str | None = None,
    url: str | None = None,
    published_at: str | None = "2024-01-01T00:00:00Z",
    **extra: Any,
) -> dict[str, Any]:
    """Build a raw FreeNewsApi-style item for a mock response."""
    item: dict[str, Any] = {
        "uuid": uuid,
        "title": title if title is not None else f"Title {uuid}",
        "url": url if url is not None else f"https://news.example/{uuid}",
        "published_at": published_at,
        "description": f"Summary {uuid}",
    }
    item.update(extra)
    return item


def make_news_api(
    handler: Any,
    *,
    api_key: str | None = "test_key",
    **kwargs: Any,
) -> NewsAPI:
    """Build a ``NewsAPI`` whose async and sync clients share a mock transport."""
    transport = httpx.MockTransport(handler)
    return NewsAPI(
        apiKey=api_key,
        base_url=BASE_URL,
        client=httpx.AsyncClient(transport=transport),
        sync_client=httpx.Client(transport=transport),
        **kwargs,
    )


class TestNewsAPIInitialization(unittest.IsolatedAsyncioTestCase):
    """Construction is offline and credentials resolve in the documented order."""

    async def test_inherits_from_base_provider(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        api = NewsAPI(
            apiKey="key",
            client=httpx.AsyncClient(transport=transport),
            sync_client=httpx.Client(transport=transport),
        )
        try:
            self.assertIsInstance(api, BaseProvider)
            self.assertEqual(api.base_url, DEFAULT_BASE_URL)
        finally:
            await api.aclose()

    async def test_api_key_header_on_both_clients(self) -> None:
        api = make_news_api(
            lambda request: httpx.Response(200, json={}),
            api_key="secret",
        )
        try:
            self.assertEqual(api.session.headers.get("x-api-key"), "secret")
            self.assertEqual(api.client.headers.get("x-api-key"), "secret")
        finally:
            await api.aclose()

    async def test_constructor_performs_no_network_io(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, json={"data": [{"topic": "technology"}]})

        api = make_news_api(handler)
        try:
            self.assertEqual(calls, [])
            self.assertFalse(api._sources_loaded)
        finally:
            await api.aclose()

    async def test_sources_is_lazy_and_cached(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, json={"data": [{"topic": "technology"}]})

        api = make_news_api(handler)
        try:
            self.assertEqual(calls, [])
            first = api._sources
            second = api._sources
        finally:
            await api.aclose()

        self.assertEqual(first, {"data": [{"topic": "technology"}]})
        self.assertIs(first, second)
        self.assertEqual(calls, ["/v1/topics"])

    @patch.dict("os.environ", {"FREENEWSAPI_API_KEY": "env_freenews"}, clear=True)
    async def test_resolves_primary_env_var(self) -> None:
        api = make_news_api(lambda request: httpx.Response(200, json={}), api_key=None)
        try:
            self.assertEqual(api.api_key, "env_freenews")
        finally:
            await api.aclose()

    @patch.dict("os.environ", {"NEWS_API_KEY": "env_news"}, clear=True)
    async def test_resolves_general_env_var(self) -> None:
        api = make_news_api(lambda request: httpx.Response(200, json={}), api_key=None)
        try:
            self.assertEqual(api.api_key, "env_news")
        finally:
            await api.aclose()

    @patch.dict("os.environ", {"FREENEWSAPI_API_KEY": "env_key"}, clear=True)
    async def test_explicit_key_overrides_env(self) -> None:
        api = make_news_api(
            lambda request: httpx.Response(200, json={}),
            api_key="explicit",
        )
        try:
            self.assertEqual(api.api_key, "explicit")
        finally:
            await api.aclose()


class TestNewsAPISyncMethods(unittest.IsolatedAsyncioTestCase):
    """Synchronous legacy methods still forward filters and honour limits."""

    async def test_get_news_forwards_limit_offset_and_cursor(self) -> None:
        payload = {"data": [make_item("u1"), make_item("u2"), make_item("u3")]}
        api = make_news_api(lambda request: httpx.Response(200, json=payload))
        try:
            with patch.object(api, "_make_request_sync", return_value=payload) as mocked:
                result = api.get_news(topic="technology", offset=2, cursor="abc", limit=2)
        finally:
            await api.aclose()

        params = mocked.call_args.kwargs["params"]
        self.assertEqual(params["limit"], 2)
        self.assertEqual(params["offset"], 2)
        self.assertEqual(params["cursor"], "abc")
        self.assertEqual(params["topic"], "technology")
        self.assertEqual(len(result["data"]), 2)
        self.assertEqual([item["uuid"] for item in result["data"]], ["u1", "u2"])

    async def test_get_news_without_limit_does_not_truncate(self) -> None:
        payload = {"data": [make_item("u1"), make_item("u2"), make_item("u3")]}
        api = make_news_api(lambda request: httpx.Response(200, json=payload))
        try:
            with patch.object(api, "_make_request_sync", return_value=payload) as mocked:
                result = api.get_news(in_title="ai")
        finally:
            await api.aclose()

        self.assertIs(result, payload)
        self.assertIsNone(mocked.call_args.kwargs["params"]["limit"])

    async def test_get_topics_and_details_use_sync_path(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/topics"):
                return httpx.Response(200, json={"data": ["technology"]})
            return httpx.Response(200, json={"data": {"uuid": "u1"}})

        api = make_news_api(handler)
        try:
            topics = api.get_topics()
            details = api.get_article_details("u1")
        finally:
            await api.aclose()

        self.assertEqual(topics, {"data": ["technology"]})
        self.assertEqual(details, {"data": {"uuid": "u1"}})


class TestNewsAPIAsyncMethods(unittest.IsolatedAsyncioTestCase):
    """Async retrieval, pagination, mapping, and error propagation."""

    async def test_fetch_recent_returns_items_and_forwards_limit(self) -> None:
        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(dict(request.url.params))
            return httpx.Response(
                200, json={"data": [make_item("u1"), make_item("u2"), make_item("u3")]}
            )

        api = make_news_api(handler)
        try:
            items = await api.fetch_recent(limit=2)
        finally:
            await api.aclose()

        self.assertEqual([item["uuid"] for item in items], ["u1", "u2"])
        self.assertEqual(seen[0]["limit"], "2")
        self.assertEqual(seen[0]["topic"], Category)

    async def test_search_returns_items_and_forwards_query(self) -> None:
        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(dict(request.url.params))
            return httpx.Response(200, json={"data": [make_item("u1")]})

        api = make_news_api(handler)
        try:
            items = await api.search("robotics", limit=1)
        finally:
            await api.aclose()

        self.assertEqual([item["uuid"] for item in items], ["u1"])
        self.assertEqual(seen[0]["in_title"], "robotics")
        self.assertEqual(seen[0]["limit"], "1")

    @patch.dict("os.environ", {}, clear=True)
    async def test_fetch_articles_paginates_and_deduplicates(self) -> None:
        pages = {
            "0": [make_item("u1"), make_item("u2")],
            "2": [make_item("u2"), make_item("u3")],
            "4": [make_item("u4"), make_item("u5")],
        }
        offsets: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            offset = request.url.params.get("offset", "")
            offsets.append(offset)
            return httpx.Response(200, json={"data": pages.get(offset, [])})

        api = make_news_api(handler)
        try:
            articles = await api.fetch_articles(
                limit=4,
                topic="technology",
                page_size=2,
                max_pages=10,
                max_items=100,
            )
        finally:
            await api.aclose()

        self.assertEqual([article.id for article in articles], ["u1", "u2", "u3", "u4"])
        self.assertEqual(offsets, ["0", "2", "4"])
        self.assertTrue(all(article.topics == ("technology",) for article in articles))

    @patch.dict("os.environ", {}, clear=True)
    async def test_fetch_articles_stops_on_empty_page(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, json={"data": [make_item("u1")]})
            return httpx.Response(200, json={"data": []})

        api = make_news_api(handler)
        try:
            articles = await api.fetch_articles(limit=10, page_size=1, max_pages=10)
        finally:
            await api.aclose()

        self.assertEqual([article.id for article in articles], ["u1"])
        self.assertEqual(calls, 2)

    @patch.dict("os.environ", {}, clear=True)
    async def test_fetch_articles_honours_max_pages(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"data": [make_item(f"u{calls}")]})

        api = make_news_api(handler)
        try:
            articles = await api.fetch_articles(limit=100, page_size=1, max_pages=2, max_items=100)
        finally:
            await api.aclose()

        self.assertEqual(calls, 2)
        self.assertEqual(len(articles), 2)

    @patch.dict("os.environ", {}, clear=True)
    async def test_fetch_articles_maps_fields_and_skips_malformed_items(self) -> None:
        items = [
            make_item(
                "u1",
                published_at="2024-01-01T00:00:00Z",
                publisher={"name": "Reuters"},
            ),
            make_item(
                "u2",
                published_at="2024-01-02T05:00:00",
                publisher={"uuid": "pub-2"},
            ),
            make_item("u3", published_at=None),
            {"uuid": "bad-1", "url": "https://news.example/bad1"},
            {"uuid": "bad-2", "title": "Missing URL"},
            {"title": "Missing id", "url": "https://news.example/bad3"},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            offset = request.url.params.get("offset", "")
            return httpx.Response(200, json={"data": items if offset == "0" else []})

        api = make_news_api(handler)
        try:
            articles = await api.fetch_articles(limit=10, topic="technology", page_size=10)
        finally:
            await api.aclose()

        self.assertEqual([article.id for article in articles], ["u1", "u2", "u3"])
        self.assertEqual(articles[0].source, "Reuters")
        self.assertEqual(articles[1].source, "pub-2")
        self.assertEqual(articles[2].source, "freenewsapi")
        self.assertEqual(articles[0].published_at.tzinfo, UTC)
        self.assertEqual(articles[1].published_at.tzinfo, UTC)
        self.assertIsNone(articles[2].published_at)
        self.assertEqual(articles[0].summary, "Summary u1")
        self.assertEqual(articles[0].raw["uuid"], "u1")

    @patch.dict("os.environ", {}, clear=True)
    async def test_fetch_articles_auth_error_is_not_retried(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, json={"error": "invalid key"})

        api = make_news_api(handler)
        try:
            with self.assertRaises(ProviderAuthError):
                await api.fetch_articles(limit=5)
        finally:
            await api.aclose()
        self.assertEqual(calls, 1)

    @patch.dict("os.environ", {}, clear=True)
    async def test_fetch_articles_server_error_retries_then_raises(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500, text="boom")

        api = make_news_api(handler)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            try:
                with self.assertRaises(ProviderHTTPError) as ctx:
                    await api.fetch_articles(limit=5)
            finally:
                await api.aclose()

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(calls, 3)


class TestLegacyAliases(unittest.IsolatedAsyncioTestCase):
    """Legacy ``_NewsQuery*`` methods remain thin wrappers over the sync path."""

    async def test_legacy_queries_use_sync_http(self) -> None:
        calls: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(dict(request.url.params))
            return httpx.Response(200, json={"data": []})

        api = make_news_api(handler)
        try:
            self.assertEqual(len(api._NewsQueryByKeyword("ai")), 2)
            self.assertEqual(len(api._NewsQueryByCategory("chips", "science")), 2)
            self.assertEqual(len(api._NewsQueryBySource("bbc", "ai")), 2)
            self.assertEqual(len(api._NewsQueryEverything("pub", "test")), 2)
            self.assertEqual(len(api._NewsQueryHeadlines("pub", "test")), 2)
        finally:
            await api.aclose()

        self.assertEqual(len(calls), 10)

    async def test_legacy_queries_return_none_when_all_fail(self) -> None:
        api = make_news_api(lambda request: httpx.Response(200, json={}))
        try:
            with (
                patch.object(api, "get_news_by_keyword", return_value=None),
                patch.object(api, "get_news_by_topic", return_value=None),
            ):
                self.assertIsNone(api._NewsQueryByKeyword("error"))
        finally:
            await api.aclose()


class TestSettingsFromEnv(unittest.TestCase):
    """Settings parsing keeps safe defaults and rejects malformed values."""

    @patch.dict("os.environ", {}, clear=True)
    def test_defaults(self) -> None:
        self.assertEqual(Settings.from_env(), Settings())

    @patch.dict(
        "os.environ",
        {
            "AI_NEWS_HTTP_TIMEOUT": "2.5",
            "AI_NEWS_MAX_CONCURRENCY": "8",
            "AI_NEWS_MAX_PAGES": "3",
            "AI_NEWS_MAX_ITEMS": "120",
            "AI_NEWS_TOPICS": "ai, ml ,,security",
            "AI_NEWS_PAGE_SIZE": "25",
        },
        clear=True,
    )
    def test_custom_values(self) -> None:
        settings = Settings.from_env()
        self.assertEqual(settings.http_timeout, 2.5)
        self.assertEqual(settings.max_concurrency, 8)
        self.assertEqual(settings.max_pages, 3)
        self.assertEqual(settings.max_items, 120)
        self.assertEqual(settings.topics, ("ai", "ml", "security"))
        self.assertEqual(settings.page_size, 25)

    @patch.dict("os.environ", {"AI_NEWS_MAX_CONCURRENCY": "0"}, clear=True)
    def test_invalid_integer_is_rejected(self) -> None:
        with self.assertRaises(ProviderConfigError):
            Settings.from_env()

    @patch.dict("os.environ", {"AI_NEWS_HTTP_TIMEOUT": "fast"}, clear=True)
    def test_invalid_float_is_rejected(self) -> None:
        with self.assertRaises(ProviderConfigError):
            Settings.from_env()

    @patch.dict("os.environ", {"AI_NEWS_TOPICS": " , "}, clear=True)
    def test_empty_topics_are_rejected(self) -> None:
        with self.assertRaises(ProviderConfigError):
            Settings.from_env()


if __name__ == "__main__":
    unittest.main()
