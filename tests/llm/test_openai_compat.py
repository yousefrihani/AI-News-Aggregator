"""Contract tests for the shared OpenAI-compatible transport.

Every exchange uses ``httpx.MockTransport``; no socket is opened and no key is
real. Groq and OpenRouter are exercised only for their default endpoint and
attribution headers.
"""

import json
import unittest
from collections.abc import Callable
from typing import Any

import httpx
from Articles.llm.base import CompletionRequest, Message
from Articles.llm.errors import (
    LLMAuthError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTransportError,
)
from Articles.llm.groq import GroqProvider
from Articles.llm.openai_compat import OpenAICompatibleProvider
from Articles.llm.openrouter import OpenRouterProvider

Handler = Callable[[httpx.Request], httpx.Response]


def make_request(**overrides: Any) -> CompletionRequest:
    """Build a completion request with sane defaults."""
    values: dict[str, Any] = {
        "messages": (Message(role="user", content="hello"),),
        "model": "model-1",
    }
    values.update(overrides)
    return CompletionRequest(**values)


def chat_payload(
    content: Any = "answer",
    *,
    usage: Any = None,
    finish_reason: Any = "stop",
) -> dict[str, Any]:
    """Build a minimal chat-completions payload."""
    choice: dict[str, Any] = {"message": {"role": "assistant", "content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    payload: dict[str, Any] = {"choices": [choice]}
    if usage is not None:
        payload["usage"] = usage
    return payload


class JSONResponse:
    """Mock handler that records requests and returns one JSON payload."""

    def __init__(self, payload: Any, status_code: int = 200, headers: Any = None) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self.status_code,
            json=self.payload,
            headers={"content-type": "application/json", **self.headers},
        )


class ProviderTestCase(unittest.IsolatedAsyncioTestCase):
    """Build a provider bound to a mock async client."""

    def make_provider(
        self,
        handler: Handler,
        *,
        provider_class: type[OpenAICompatibleProvider] = OpenAICompatibleProvider,
        **kwargs: Any,
    ) -> OpenAICompatibleProvider:
        """Create a provider sharing an injected mock client."""
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        options: dict[str, Any] = {
            "name": "test",
            "base_url": "https://api.example.test/v1",
            "api_key": "secret-key",
            "client": client,
        }
        options.update(kwargs)
        return provider_class(**options)


class TestRequestContract(ProviderTestCase):
    """URL, headers, and body shaping."""

    async def test_posts_chat_completion_with_bearer_auth(self) -> None:
        handler = JSONResponse(
            chat_payload("answer", usage={"prompt_tokens": 11, "completion_tokens": 7})
        )
        provider = self.make_provider(handler)
        response = await provider.complete(make_request())

        self.assertEqual(len(handler.requests), 1)
        request = handler.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "https://api.example.test/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer secret-key")
        self.assertEqual(request.headers["content-type"], "application/json")

        body = json.loads(request.content)
        self.assertEqual(body["model"], "model-1")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(body["temperature"], 0.2)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("response_format", body)

        self.assertEqual(response.text, "answer")
        self.assertEqual(response.provider, "test")
        self.assertEqual(response.model, "model-1")
        self.assertEqual(response.usage.prompt_tokens, 11)
        self.assertEqual(response.usage.completion_tokens, 7)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(
            response.raw,
            chat_payload("answer", usage={"prompt_tokens": 11, "completion_tokens": 7}),
        )

    async def test_temperature_and_max_tokens_forwarded(self) -> None:
        handler = JSONResponse(chat_payload())
        provider = self.make_provider(handler)
        await provider.complete(make_request(temperature=0.7, max_tokens=64))

        body = json.loads(handler.requests[0].content)
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["max_tokens"], 64)

    async def test_no_auth_header_without_key(self) -> None:
        for key in (None, ""):
            handler = JSONResponse(chat_payload())
            provider = self.make_provider(handler, api_key=key)
            await provider.complete(make_request())
            self.assertNotIn("authorization", handler.requests[0].headers)

    async def test_config_headers_cannot_mask_authorization(self) -> None:
        handler = JSONResponse(chat_payload())
        provider = self.make_provider(
            handler,
            headers={"Authorization": "Bearer replaced", "X-Custom": "yes"},
        )
        await provider.complete(make_request())
        self.assertEqual(handler.requests[0].headers["authorization"], "Bearer secret-key")
        self.assertEqual(handler.requests[0].headers["x-custom"], "yes")

    async def test_json_object_response_format(self) -> None:
        handler = JSONResponse(chat_payload('{"ok": true}'))
        provider = self.make_provider(handler)
        await provider.complete(make_request(response_format="json_object"))
        body = json.loads(handler.requests[0].content)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    async def test_json_schema_wraps_plain_mapping(self) -> None:
        handler = JSONResponse(chat_payload('{"ok": true}'))
        provider = self.make_provider(handler)
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        await provider.complete(make_request(response_format="json_schema", json_schema=schema))
        body = json.loads(handler.requests[0].content)
        self.assertEqual(
            body["response_format"],
            {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}},
        )

    async def test_json_schema_passes_named_mapping_through(self) -> None:
        handler = JSONResponse(chat_payload("{}"))
        provider = self.make_provider(handler)
        named = {"name": "graph_extraction.v1", "schema": {"type": "object"}}
        await provider.complete(make_request(response_format="json_schema", json_schema=named))
        body = json.loads(handler.requests[0].content)
        self.assertEqual(
            body["response_format"],
            {"type": "json_schema", "json_schema": named},
        )

    async def test_json_schema_requires_schema(self) -> None:
        handler = JSONResponse(chat_payload())
        self.make_provider(handler)
        with self.assertRaises(ValueError):
            make_request(response_format="json_schema")
        self.assertEqual(handler.requests, [])


class TestResponseParsing(ProviderTestCase):
    """Usage, finish reason, and malformed payload mapping."""

    async def test_missing_usage_defaults_to_zero(self) -> None:
        handler = JSONResponse(chat_payload())
        provider = self.make_provider(handler)
        response = await provider.complete(make_request())
        self.assertEqual(response.usage.total_tokens, 0)

    async def test_malformed_usage_defaults_to_zero(self) -> None:
        handler = JSONResponse(
            chat_payload(usage={"prompt_tokens": "many", "completion_tokens": -1})
        )
        provider = self.make_provider(handler)
        response = await provider.complete(make_request())
        self.assertEqual(response.usage.total_tokens, 0)

    async def test_non_string_finish_reason_becomes_none(self) -> None:
        handler = JSONResponse(chat_payload(finish_reason=42))
        provider = self.make_provider(handler)
        response = await provider.complete(make_request())
        self.assertIsNone(response.finish_reason)

    async def test_invalid_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not-json")

        provider = self.make_provider(handler)
        with self.assertRaises(LLMResponseError):
            await provider.complete(make_request())

    async def test_non_object_payload(self) -> None:
        handler = JSONResponse(["not", "an", "object"])
        provider = self.make_provider(handler)
        with self.assertRaises(LLMResponseError):
            await provider.complete(make_request())

    async def test_missing_or_empty_choices(self) -> None:
        for payload in ({}, {"choices": []}, {"choices": "nope"}, {"choices": [1]}):
            handler = JSONResponse(payload)
            provider = self.make_provider(handler)
            with self.assertRaises(LLMResponseError):
                await provider.complete(make_request())

    async def test_empty_completion_text(self) -> None:
        for content in ("", "   ", None, 42):
            handler = JSONResponse(chat_payload(content))
            provider = self.make_provider(handler)
            with self.assertRaises(LLMResponseError):
                await provider.complete(make_request())

    async def test_missing_message(self) -> None:
        handler = JSONResponse({"choices": [{"finish_reason": "stop"}]})
        provider = self.make_provider(handler)
        with self.assertRaises(LLMResponseError):
            await provider.complete(make_request())


class TestStatusMapping(ProviderTestCase):
    """HTTP statuses map to typed errors with the right retryability."""

    async def test_auth_statuses(self) -> None:
        for status in (401, 403):
            handler = JSONResponse({"error": "denied"}, status_code=status)
            provider = self.make_provider(handler)
            with self.assertRaises(LLMAuthError) as raised:
                await provider.complete(make_request())
            self.assertEqual(raised.exception.status_code, status)
            self.assertFalse(raised.exception.retryable)

    async def test_rate_limit_with_retry_after(self) -> None:
        handler = JSONResponse({}, status_code=429, headers={"Retry-After": "2.5"})
        provider = self.make_provider(handler)
        with self.assertRaises(LLMRateLimitError) as raised:
            await provider.complete(make_request())
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(raised.exception.retry_after, 2.5)

    async def test_rate_limit_without_retry_after(self) -> None:
        handler = JSONResponse({}, status_code=429)
        provider = self.make_provider(handler)
        with self.assertRaises(LLMRateLimitError) as raised:
            await provider.complete(make_request())
        self.assertIsNone(raised.exception.retry_after)

    async def test_server_and_client_errors(self) -> None:
        server = JSONResponse({"error": "boom"}, status_code=503)
        provider = self.make_provider(server)
        with self.assertRaises(LLMHTTPError) as raised:
            await provider.complete(make_request())
        self.assertEqual(raised.exception.status_code, 503)
        self.assertTrue(raised.exception.retryable)

        client_error = JSONResponse({"error": "bad"}, status_code=422)
        provider = self.make_provider(client_error)
        with self.assertRaises(LLMHTTPError) as raised:
            await provider.complete(make_request())
        self.assertEqual(raised.exception.status_code, 422)
        self.assertFalse(raised.exception.retryable)

    async def test_redirect_is_an_http_error(self) -> None:
        handler = JSONResponse({}, status_code=302)
        provider = self.make_provider(handler)
        with self.assertRaises(LLMHTTPError):
            await provider.complete(make_request())

    async def test_timeout_maps_to_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        provider = self.make_provider(handler)
        with self.assertRaises(LLMTransportError) as raised:
            await provider.complete(make_request())
        self.assertTrue(raised.exception.retryable)

    async def test_connection_error_maps_to_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        provider = self.make_provider(handler)
        with self.assertRaises(LLMTransportError):
            await provider.complete(make_request())


class TestThinAdapters(ProviderTestCase):
    """Groq and OpenRouter only add defaults."""

    async def test_groq_default_endpoint(self) -> None:
        handler = JSONResponse(chat_payload())
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        provider = GroqProvider(api_key="groq-key", client=client)
        response = await provider.complete(make_request())
        self.assertEqual(provider.name, "groq")
        self.assertEqual(
            str(handler.requests[0].url), "https://api.groq.com/openai/v1/chat/completions"
        )
        self.assertEqual(response.provider, "groq")

    async def test_openrouter_attribution_headers(self) -> None:
        handler = JSONResponse(chat_payload())
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        provider = OpenRouterProvider(api_key="or-key", client=client)
        await provider.complete(make_request())
        request = handler.requests[0]
        self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(
            request.headers["http-referer"],
            "https://github.com/yousefrihani/AI-News-Aggregator",
        )
        self.assertEqual(request.headers["x-title"], "AI-News-Aggregator")
        self.assertEqual(request.headers["authorization"], "Bearer or-key")

    async def test_openrouter_config_headers_override_defaults(self) -> None:
        handler = JSONResponse(chat_payload())
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        provider = OpenRouterProvider(
            api_key="or-key",
            headers={"X-Title": "Custom Title", "HTTP-Referer": "https://example.test"},
            client=client,
        )
        await provider.complete(make_request())
        request = handler.requests[0]
        self.assertEqual(request.headers["x-title"], "Custom Title")
        self.assertEqual(request.headers["http-referer"], "https://example.test")


class TestLifecycle(unittest.IsolatedAsyncioTestCase):
    """Ownership rules for the HTTP client."""

    async def test_owned_client_is_closed_idempotently(self) -> None:
        provider = OpenAICompatibleProvider(name="owned", base_url="https://example.test/v1")
        client = provider._get_client()  # noqa: SLF001 - ownership assertion
        await provider.aclose()
        await provider.aclose()
        self.assertTrue(client.is_closed)

    async def test_injected_client_is_not_closed(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        )
        self.addAsyncCleanup(client.aclose)
        provider = OpenAICompatibleProvider(
            name="shared",
            base_url="https://example.test/v1",
            client=client,
        )
        await provider.aclose()
        self.assertFalse(client.is_closed)


if __name__ == "__main__":
    unittest.main()
