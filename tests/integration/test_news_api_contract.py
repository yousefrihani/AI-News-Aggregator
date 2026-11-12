"""Hermetic contract tests for the FreeNewsApi.io adapter.

Every HTTP exchange is served by :class:`httpx.MockTransport`, so these tests
exercise URL construction, headers, response mapping, retries, and pagination
without opening a socket. No live network access is required.

The adapter keeps its legacy ``get_*`` helpers synchronous (they share the
``sync_client`` path), while ``fetch_recent``/``search``/``fetch_articles`` are
asynchronous. Both clients are therefore backed by the same mock handler.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
from Articles.errors import ProviderAuthError, ProviderHTTPError
from Articles.Models import Article
from Articles.NewsApi import DEFAULT_BASE_URL, NewsAPI

FIXTURES = Path(__file__).with_name("fixtures")

Handler = Callable[[httpx.Request], httpx.Response]


def _fixture(name: str) -> dict[str, Any]:
    """Load one JSON fixture from the integration fixtures directory."""
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _json_response(payload: Any, status_code: int = 200) -> httpx.Response:
    """Build a JSON ``httpx`` response for the mock transport."""
    return httpx.Response(
        status_code,
        json=payload,
        headers={"content-type": "application/json"},
    )


async def _resolve(value: Any) -> Any:
    """Await ``value`` when the provider exposes an async method."""
    if inspect.isawaitable(value):
        return await value
    return value


def _unique_page(offset: int, count: int = 2) -> list[dict[str, Any]]:
    """Build a page of unique news items starting at ``offset``."""
    template = _fixture("news.json")["data"]
    return [
        {
            **template[index % len(template)],
            "uuid": f"news-{offset + index}",
            "title": f"Synthetic article {offset + index}",
            "url": f"https://example.com/news/{offset + index}",
        }
        for index in range(count)
    ]


@contextlib.contextmanager
def _no_sleep() -> Iterator[None]:
    """Replace async sleeps so retry tests do not wait for real backoff."""

    async def _fast_sleep(*_args: Any, **_kwargs: Any) -> None:
        return None

    patchers = [mock.patch("asyncio.sleep", new=_fast_sleep)]
    base_provider_module = importlib.import_module("Articles.BaseProvider")
    if hasattr(base_provider_module, "sleep"):
        patchers.append(mock.patch.object(base_provider_module, "sleep", new=_fast_sleep))

    with contextlib.ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield


class NewsAPIContractTest(unittest.IsolatedAsyncioTestCase):
    """Contract tests driven entirely by ``httpx.MockTransport``."""

    def make_provider(
        self,
        handler: Handler,
        *,
        api_key: str = "contract-test-key",
    ) -> NewsAPI:
        """Create a NewsAPI bound to mock async and sync clients."""
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sync_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(async_client.aclose)
        self.addCleanup(sync_client.close)
        return NewsAPI(
            apiKey=api_key,
            base_url=DEFAULT_BASE_URL,
            timeout=5.0,
            client=async_client,
            sync_client=sync_client,
        )

    async def test_x_api_key_header_is_sent(self) -> None:
        """The API key is sent as ``x-api-key`` on asynchronous requests."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _json_response(_fixture("news.json"))

        provider = self.make_provider(handler, api_key="secret-contract-key")
        await provider.fetch_recent(limit=1)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers.get("x-api-key"), "secret-contract-key")

    async def test_topics_endpoint_mapping(self) -> None:
        """``/topics`` is requested and its JSON payload is returned unchanged."""
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _json_response(_fixture("topics.json"))

        provider = self.make_provider(handler)
        payload = await _resolve(provider.get_topics())

        self.assertEqual(seen[0].method, "GET")
        self.assertTrue(seen[0].url.path.endswith("/topics"))
        self.assertEqual(payload, _fixture("topics.json"))

    async def test_publishers_endpoint_mapping(self) -> None:
        """``/publishers`` is requested and its JSON payload is returned unchanged."""
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _json_response(_fixture("publishers.json"))

        provider = self.make_provider(handler)
        payload = await _resolve(provider.get_publishers())

        self.assertTrue(seen[0].url.path.endswith("/publishers"))
        self.assertEqual(payload, _fixture("publishers.json"))

    async def test_news_endpoint_forwards_supported_filters(self) -> None:
        """``/news`` forwards keyword, topic, paging, language, and country."""
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _json_response(_fixture("news.json"))

        provider = self.make_provider(handler)
        payload = await _resolve(
            provider.get_news(
                in_title="autonomous",
                topic="technology",
                limit=5,
                offset=20,
                language="en",
                country="us",
            )
        )

        self.assertEqual(len(seen), 1)
        params = seen[0].url.params
        self.assertTrue(seen[0].url.path.endswith("/news"))
        self.assertEqual(params.get("in_title"), "autonomous")
        self.assertEqual(params.get("topic"), "technology")
        self.assertEqual(params.get("limit"), "5")
        self.assertEqual(params.get("offset"), "20")
        self.assertEqual(params.get("language"), "en")
        self.assertEqual(params.get("country"), "us")
        # ``None`` filters must not leak onto the wire.
        self.assertNotIn("publisher_uuid", params)
        self.assertEqual(payload, _fixture("news.json"))

    async def test_details_endpoint_mapping(self) -> None:
        """``/details`` sends the uuid and returns the detail payload."""
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _json_response(_fixture("details.json"))

        provider = self.make_provider(handler)
        payload = await _resolve(provider.get_article_details(uuid="news-1"))

        self.assertTrue(seen[0].url.path.endswith("/details"))
        self.assertEqual(seen[0].url.params.get("uuid"), "news-1")
        self.assertEqual(payload, _fixture("details.json"))

    async def test_401_raises_provider_auth_error_without_retry(self) -> None:
        """A 401 fails fast with ``ProviderAuthError`` and is not retried."""
        calls = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _json_response({"error": "unauthorized"}, status_code=401)

        provider = self.make_provider(handler, api_key="bad-key")

        with self.assertRaises(ProviderAuthError):
            await provider.fetch_recent(limit=1)

        self.assertEqual(calls, 1)

    async def test_500_is_retried_then_raises_provider_http_error(self) -> None:
        """A persistent 500 is retried up to three attempts, then surfaced."""
        calls = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _json_response({"error": "server error"}, status_code=500)

        provider = self.make_provider(handler)

        with _no_sleep(), self.assertRaises(ProviderHTTPError) as raised:
            await provider.fetch_recent(limit=1)

        self.assertEqual(calls, 3)
        self.assertEqual(raised.exception.status_code, 500)

    async def test_pagination_stops_on_empty_page(self) -> None:
        """Paging follows offsets and stops as soon as a page is empty."""
        data = _fixture("news.json")["data"]
        offsets: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            offset = int(request.url.params.get("offset", "0"))
            offsets.append(offset)
            page = data if offset == 0 else []
            return _json_response({"data": page, "meta": {"offset": offset}})

        provider = self.make_provider(handler)
        articles = await provider.fetch_articles(limit=100, max_pages=10, max_items=500)

        self.assertEqual(offsets, [0, len(data)])
        self.assertEqual(len(articles), len(data))

    async def test_pagination_respects_max_pages(self) -> None:
        """An endless feed is bounded by ``max_pages``."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            offset = int(request.url.params.get("offset", "0"))
            return _json_response({"data": _unique_page(offset)})

        provider = self.make_provider(handler)
        articles = await provider.fetch_articles(
            limit=100,
            max_pages=2,
            max_items=100,
            page_size=2,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(len(articles), 4)

    async def test_pagination_respects_max_items(self) -> None:
        """An endless feed is bounded by ``max_items``."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            offset = int(request.url.params.get("offset", "0"))
            return _json_response({"data": _unique_page(offset)})

        provider = self.make_provider(handler)
        articles = await provider.fetch_articles(
            limit=100,
            max_pages=10,
            max_items=3,
            page_size=2,
        )

        self.assertEqual(len(articles), 3)
        self.assertLessEqual(calls, 2)

    async def test_fetch_articles_maps_payload_to_models(self) -> None:
        """``fetch_articles`` returns normalized, timezone-aware ``Article`` objects."""
        payload = _fixture("news.json")
        data = payload["data"]

        def handler(request: httpx.Request) -> httpx.Response:
            offset = int(request.url.params.get("offset", "0"))
            return _json_response({"data": data if offset == 0 else []})

        provider = self.make_provider(handler)
        articles = await provider.fetch_articles(limit=2, page_size=2)

        self.assertEqual(len(articles), 2)
        self.assertTrue(all(isinstance(article, Article) for article in articles))
        self.assertEqual(articles[0].title, data[0]["title"])
        self.assertEqual(articles[0].url, data[0]["url"])
        self.assertIsNotNone(articles[0].published_at)
        assert articles[0].published_at is not None
        self.assertIsNotNone(articles[0].published_at.tzinfo)


if __name__ == "__main__":
    unittest.main()
