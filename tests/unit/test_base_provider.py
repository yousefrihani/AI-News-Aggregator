"""Hermetic unit tests for :mod:`Articles.BaseProvider`.

Every request is served by :class:`httpx.MockTransport`, so the suite never
opens a socket. Async tests use ``IsolatedAsyncioTestCase`` and sync coverage
exercises the legacy request path.
"""

import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from Articles.BaseProvider import BaseProvider
from Articles.errors import (
    ProviderAuthError,
    ProviderHTTPError,
    ProviderTransportError,
)


class DummyProvider(BaseProvider):
    """Concrete provider used to exercise the base HTTP plumbing."""

    async def fetch_recent(self, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        payload = await self._make_request("/recent", params={"limit": limit, **kwargs})
        return [item for item in payload.get("data", []) if isinstance(item, dict)][:limit]

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        payload = await self._make_request("/search", params={"q": query, "limit": limit, **kwargs})
        return [item for item in payload.get("data", []) if isinstance(item, dict)][:limit]

    async def fetch_articles(self, limit: int = 10, **kwargs: Any) -> list[Any]:
        await self._make_request("/articles", params={"limit": limit, **kwargs})
        return []


class IncompleteProvider(BaseProvider):
    """Provider missing ``fetch_articles`` to prove the ABC contract."""

    async def fetch_recent(self, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        return []


def make_provider(
    handler: Any,
    **kwargs: Any,
) -> DummyProvider:
    """Build a provider whose clients share a mock transport.

    Args:
        handler: ``httpx.MockTransport`` handler function.
        **kwargs: Extra constructor arguments.

    Returns:
        A configured :class:`DummyProvider`.
    """
    transport = httpx.MockTransport(handler)
    return DummyProvider(
        base_url="https://api.example.com",
        client=httpx.AsyncClient(transport=transport),
        sync_client=httpx.Client(transport=transport),
        **kwargs,
    )


class TestBaseProviderConstruction(unittest.IsolatedAsyncioTestCase):
    """Constructor, abstract contract, and lifecycle behaviour."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        with self.assertRaises(TypeError):
            BaseProvider(api_key="test_key")  # type: ignore[abstract]

    def test_fetch_articles_is_required(self) -> None:
        with self.assertRaises(TypeError):
            IncompleteProvider(base_url="https://api.example.com")  # type: ignore[abstract]

    def test_concrete_subclass_initialization(self) -> None:
        provider = DummyProvider(
            api_key="abc123key",
            base_url="https://api.example.com/v1/",
            timeout=15,
            max_concurrency=7,
        )
        try:
            self.assertEqual(provider.api_key, "abc123key")
            self.assertEqual(provider.base_url, "https://api.example.com/v1")
            self.assertEqual(provider.timeout, 15.0)
            self.assertEqual(provider.max_concurrency, 7)
            self.assertIsInstance(provider.client, httpx.AsyncClient)
            self.assertIsInstance(provider.sync_client, httpx.Client)
            self.assertIs(provider.session, provider.sync_client)
        finally:
            provider.close()

    @patch.dict("os.environ", {"API_KEY": "env_secret_key"}, clear=True)
    def test_initialization_falls_back_to_env_var(self) -> None:
        provider = DummyProvider(base_url="https://api.example.com")
        try:
            self.assertEqual(provider.api_key, "env_secret_key")
        finally:
            provider.close()

    @patch.dict("os.environ", {"API_KEY": "env_secret_key"}, clear=True)
    def test_explicit_api_key_overrides_env_var(self) -> None:
        provider = DummyProvider(api_key="explicit_key", base_url="https://api.example.com")
        try:
            self.assertEqual(provider.api_key, "explicit_key")
        finally:
            provider.close()

    def test_invalid_constructor_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DummyProvider(max_concurrency=0)
        with self.assertRaises(ValueError):
            DummyProvider(timeout=0)

    async def test_aclose_closes_both_clients(self) -> None:
        provider = make_provider(lambda request: httpx.Response(200, json={}))
        await provider.aclose()
        self.assertTrue(provider.client.is_closed)
        self.assertTrue(provider.sync_client.is_closed)

    async def test_async_context_manager_closes_clients(self) -> None:
        provider = make_provider(lambda request: httpx.Response(200, json={}))
        async with provider as entered:
            self.assertIs(entered, provider)
        self.assertTrue(provider.client.is_closed)
        self.assertTrue(provider.sync_client.is_closed)

    async def test_sync_context_manager_closes_sync_client(self) -> None:
        provider = make_provider(lambda request: httpx.Response(200, json={}))
        with provider as entered:
            self.assertIs(entered, provider)
        self.assertTrue(provider.sync_client.is_closed)
        await provider.client.aclose()


class TestAsyncRequests(unittest.IsolatedAsyncioTestCase):
    """Retry, status-mapping, and request-shaping for the async path."""

    async def test_make_request_success_filters_none_params(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url.copy_with(query=None))
            seen["params"] = dict(request.url.params)
            seen["method"] = request.method
            return httpx.Response(200, json={"status": "ok", "items": [1, 2, 3]})

        provider = make_provider(handler, api_key="key")
        try:
            result = await provider._make_request(
                "/search", params={"q": "machine learning", "empty": None}
            )
        finally:
            await provider.aclose()

        self.assertEqual(result, {"status": "ok", "items": [1, 2, 3]})
        self.assertEqual(seen["url"], "https://api.example.com/search")
        self.assertEqual(seen["params"], {"q": "machine learning"})
        self.assertEqual(seen["method"], "GET")

    async def test_auth_errors_are_not_retried(self) -> None:
        for status in (401, 403):
            calls = 0

            def handler(request: httpx.Request, code: int = status) -> httpx.Response:
                nonlocal calls
                calls += 1
                return httpx.Response(code, json={"error": "denied"})

            provider = make_provider(handler)
            try:
                with self.assertRaises(ProviderAuthError) as ctx:
                    await provider._make_request("/secure")
            finally:
                await provider.aclose()

            self.assertEqual(ctx.exception.status_code, status)
            self.assertEqual(calls, 1)

    async def test_client_errors_are_not_retried(self) -> None:
        for status in (400, 404, 422):
            calls = 0

            def handler(request: httpx.Request, code: int = status) -> httpx.Response:
                nonlocal calls
                calls += 1
                return httpx.Response(code, text="bad request")

            provider = make_provider(handler)
            try:
                with self.assertRaises(ProviderHTTPError) as ctx:
                    await provider._make_request("/invalid")
            finally:
                await provider.aclose()

            self.assertEqual(ctx.exception.status_code, status)
            self.assertEqual(ctx.exception.body, "bad request")
            self.assertEqual(calls, 1)

    async def test_server_error_retries_then_raises(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500, text="boom")

        provider = make_provider(handler)
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            try:
                with self.assertRaises(ProviderHTTPError) as ctx:
                    await provider._make_request("/broken")
            finally:
                await provider.aclose()

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(calls, 3)
        self.assertEqual(mock_sleep.await_count, 2)
        first_delay = mock_sleep.await_args_list[0].args[0]
        second_delay = mock_sleep.await_args_list[1].args[0]
        self.assertGreaterEqual(first_delay, 0.5)
        self.assertGreaterEqual(second_delay, 1.0)

    async def test_retry_after_header_controls_delay(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "1.5"})
            return httpx.Response(200, json={"ok": True})

        provider = make_provider(handler)
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            try:
                result = await provider._make_request("/limited")
            finally:
                await provider.aclose()

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls, 2)
        mock_sleep.assert_awaited_once_with(1.5)

    async def test_transport_failure_retries_then_raises(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("network down", request=request)

        provider = make_provider(handler)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            try:
                with self.assertRaises(ProviderTransportError):
                    await provider._make_request("/down")
            finally:
                await provider.aclose()

        self.assertEqual(calls, 3)

    async def test_retry_then_success(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": "1"}]})

        provider = make_provider(handler)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            try:
                result = await provider._make_request("/flaky")
            finally:
                await provider.aclose()

        self.assertEqual(result, {"data": [{"id": "1"}]})
        self.assertEqual(calls, 3)

    async def test_invalid_json_maps_to_http_error(self) -> None:
        provider = make_provider(lambda request: httpx.Response(200, text="not-json"))
        try:
            with self.assertRaises(ProviderHTTPError) as ctx:
                await provider._make_request("/bad-json")
        finally:
            await provider.aclose()
        self.assertEqual(ctx.exception.status_code, 200)

    async def test_fetch_recent_and_search_use_async_path(self) -> None:
        requests: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(dict(request.url.params))
            return httpx.Response(200, json={"data": [{"id": "1"}, {"id": "2"}]})

        provider = make_provider(handler)
        try:
            recent = await provider.fetch_recent(limit=1)
            found = await provider.search("robots", limit=2)
        finally:
            await provider.aclose()

        self.assertEqual(len(recent), 1)
        self.assertEqual(len(found), 2)
        self.assertEqual(requests[0], {"limit": "1"})
        self.assertEqual(requests[1], {"q": "robots", "limit": "2"})


class TestSyncRequests(unittest.IsolatedAsyncioTestCase):
    """Legacy synchronous path shares the same retry/status policy."""

    async def test_make_request_sync_success(self) -> None:
        provider = make_provider(lambda request: httpx.Response(200, json={"data": [{"id": "1"}]}))
        try:
            result = provider._make_request_sync("/legacy", params={"page": 1})
        finally:
            await provider.aclose()
        self.assertEqual(result, {"data": [{"id": "1"}]})

    async def test_make_request_sync_retries_server_errors(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"ok": True})

        provider = make_provider(handler)
        with patch("time.sleep") as mock_sleep:
            try:
                result = provider._make_request_sync("/legacy-flaky")
            finally:
                await provider.aclose()

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls, 2)
        mock_sleep.assert_called_once()

    async def test_make_request_sync_auth_error_not_retried(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, json={"error": "nope"})

        provider = make_provider(handler)
        try:
            with self.assertRaises(ProviderAuthError):
                provider._make_request_sync("/legacy-secure")
        finally:
            await provider.aclose()
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
