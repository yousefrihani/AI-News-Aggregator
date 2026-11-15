"""Shared transport for OpenAI-compatible chat-completion APIs.

Groq, OpenRouter, and many self-hosted gateways expose the same
``POST /chat/completions`` shape. This module owns that wire format once; vendor
modules only supply defaults. Retries and fallback live in
:mod:`Articles.llm.router`, so every call here is a single attempt.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Self

import httpx

from Articles.llm.base import CompletionRequest, CompletionResponse, Usage
from Articles.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTransportError,
)

__all__ = ["OpenAICompatibleProvider"]

_MAX_ERROR_BODY = 2_000


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header into seconds.

    Args:
        value: Raw header value, either delay seconds or an HTTP date.

    Returns:
        Non-negative delay in seconds, or ``None`` when absent or invalid.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _as_token_count(value: Any) -> int:
    """Coerce a provider usage counter to a non-negative integer.

    Args:
        value: Raw counter.

    Returns:
        A non-negative integer, defaulting to zero for malformed values.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


class OpenAICompatibleProvider:
    """Single-attempt adapter for an OpenAI-compatible chat endpoint.

    Args:
        name: Provider name used in responses and diagnostics.
        base_url: API root, without a trailing slash.
        api_key: Bearer token. When empty, no ``Authorization`` header is sent.
        headers: Extra request headers, merged over adapter defaults.
        client: Optional pre-configured async client, mainly for tests.
        timeout: Default request timeout in seconds.

    Raises:
        LLMConfigError: If the name or base URL is blank.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise LLMConfigError("provider name must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise LLMConfigError(f"provider {name!r} requires a base_url")
        if timeout <= 0:
            raise LLMConfigError("timeout must be greater than 0")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.headers: dict[str, str] = dict(headers or {})
        self.timeout = float(timeout)
        self._client = client
        self._owns_client = client is None
        self._closed = False

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------

    def _request_headers(self) -> dict[str, str]:
        """Build request headers without leaking key material.

        Returns:
            Headers including ``Authorization`` only when a key is configured.
        """
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _response_format(request: CompletionRequest) -> dict[str, Any] | None:
        """Translate the request's response format to the wire shape.

        Args:
            request: Provider-independent request.

        Returns:
            The ``response_format`` payload, or ``None`` for plain text.
        """
        if request.response_format == "text":
            return None
        if request.response_format == "json_object":
            return {"type": "json_object"}
        schema = dict(request.json_schema or {})
        if "name" in schema and "schema" in schema:
            return {"type": "json_schema", "json_schema": schema}
        return {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        """Build the chat-completion JSON body.

        Args:
            request: Provider-independent request.

        Returns:
            JSON-serializable request body.
        """
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        response_format = self._response_format(request)
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the owned or injected HTTP client, creating it if needed.

        Returns:
            An async HTTP client.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    @staticmethod
    def _error_body(response: httpx.Response) -> str:
        """Return a bounded response body for diagnostics.

        Args:
            response: Failed HTTP response.

        Returns:
            Truncated response text.
        """
        return response.text[:_MAX_ERROR_BODY]

    def _raise_for_status(self, response: httpx.Response, request: CompletionRequest) -> None:
        """Map an unsuccessful response to a typed LLM error.

        Args:
            response: Completed HTTP response.
            request: Request that produced the response.

        Raises:
            LLMAuthError: For 401 and 403 responses.
            LLMRateLimitError: For 429 responses.
            LLMHTTPError: For any other 4xx or 5xx response.
        """
        status = response.status_code
        if status in (401, 403):
            raise LLMAuthError(
                f"Provider rejected credentials with HTTP {status}",
                status_code=status,
                provider=self.name,
                model=request.model,
            )
        if status == 429:
            raise LLMRateLimitError(
                "Provider rate-limited the request",
                retry_after=_parse_retry_after(response.headers.get("Retry-After")),
                provider=self.name,
                model=request.model,
            )
        raise LLMHTTPError(
            f"Provider returned HTTP {status}",
            status_code=status,
            body=self._error_body(response),
            provider=self.name,
            model=request.model,
        )

    def _parse_response(
        self, response: httpx.Response, request: CompletionRequest
    ) -> CompletionResponse:
        """Normalize a successful provider response.

        Args:
            response: Successful HTTP response.
            request: Request that produced the response.

        Returns:
            A normalized completion.

        Raises:
            LLMResponseError: If the payload is not a usable chat completion.
        """
        try:
            payload = response.json()
        except ValueError as err:
            raise LLMResponseError(
                "Provider returned invalid JSON",
                provider=self.name,
                model=request.model,
            ) from err
        if not isinstance(payload, Mapping):
            raise LLMResponseError(
                "Provider returned an unsupported JSON payload",
                provider=self.name,
                model=request.model,
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError(
                "Provider returned no choices",
                provider=self.name,
                model=request.model,
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise LLMResponseError(
                "Provider returned a malformed choice",
                provider=self.name,
                model=request.model,
            )
        message = choice.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError(
                "Provider returned empty assistant content",
                provider=self.name,
                model=request.model,
            )
        usage_raw = payload.get("usage")
        usage = Usage()
        if isinstance(usage_raw, Mapping):
            usage = Usage(
                prompt_tokens=_as_token_count(usage_raw.get("prompt_tokens")),
                completion_tokens=_as_token_count(usage_raw.get("completion_tokens")),
            )
        model = payload.get("model")
        finish_reason = choice.get("finish_reason")
        return CompletionResponse(
            text=content,
            provider=self.name,
            model=model if isinstance(model, str) and model else request.model,
            usage=usage,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            raw=payload,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute one chat completion.

        Args:
            request: Provider-independent request.

        Returns:
            A normalized completion.

        Raises:
            LLMConfigError: If the provider has already been closed.
            LLMAuthError: On authentication failure.
            LLMRateLimitError: On HTTP 429.
            LLMHTTPError: On other non-success statuses.
            LLMTransportError: On timeouts or connection failures.
            LLMResponseError: On an unusable response payload.
        """
        if self._closed:
            raise LLMConfigError(f"provider {self.name!r} is closed", provider=self.name)
        try:
            response = await self._get_client().post(
                f"{self.base_url}/chat/completions",
                json=self._payload(request),
                headers=self._request_headers(),
                timeout=self.timeout,
            )
        except httpx.TimeoutException as err:
            raise LLMTransportError(
                "Provider request timed out",
                provider=self.name,
                model=request.model,
            ) from err
        except httpx.RequestError as err:
            raise LLMTransportError(
                f"Provider request failed: {err}",
                provider=self.name,
                model=request.model,
            ) from err
        if not 200 <= response.status_code < 300:
            self._raise_for_status(response, request)
        return self._parse_response(response, request)

    async def aclose(self) -> None:
        """Close the owned HTTP client, if any."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Enter an asynchronous context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the provider when leaving an asynchronous context."""
        await self.aclose()
