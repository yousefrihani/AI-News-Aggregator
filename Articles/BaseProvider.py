"""Async-first base provider for data sources in AI-News-Aggregator.

Every ingestion provider shares one configured ``httpx.AsyncClient`` for its
asynchronous API and one lazily initialized ``httpx.Client`` for the legacy
synchronous aliases. Both request paths use the same retry policy and
HTTP-status mapping so a provider cannot drift between the two flavours.

Retry policy
------------
* At most :attr:`BaseProvider.MAX_ATTEMPTS` attempts per request.
* Only ``429``, ``5xx`` responses and transport failures are retried.
* Backoff is ``0.5 * 2 ** (attempt - 1)`` seconds plus jitter; a valid
  ``Retry-After`` header takes precedence.
* ``401``/``403`` raise :class:`ProviderAuthError` immediately.
* Other ``4xx`` responses raise :class:`ProviderHTTPError` immediately.

The module never calls :func:`asyncio.run`; callers own the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx

from Articles.errors import (
    ProviderAuthError,
    ProviderHTTPError,
    ProviderTransportError,
)
from Articles.Models import Article

if TYPE_CHECKING:
    from Articles.cache import ConditionalHTTPCache
    from Articles.quota import SQLiteQuotaGovernor

try:  # pragma: no cover - optional convenience for local development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is optional at runtime
    pass

logger = logging.getLogger(__name__)

__all__ = ["BaseProvider", "RawHTTPResponse"]


@dataclass(frozen=True, slots=True)
class RawHTTPResponse:
    """Normalized raw HTTP payload returned by :meth:`BaseProvider._request_bytes`.

    Args:
        status_code: HTTP status code (``200`` when revalidated via ``304``).
        body: Raw response bytes.
        headers: Immutable response headers mapping.
        from_cache: ``True`` when served from fresh cache or ``304`` revalidation.
    """

    status_code: int
    body: bytes
    headers: Mapping[str, str]
    from_cache: bool = False

    def __post_init__(self) -> None:
        """Freeze response headers into an immutable mapping."""
        if not isinstance(self.headers, MappingProxyType):
            object.__setattr__(
                self,
                "headers",
                MappingProxyType({str(k): str(v) for k, v in self.headers.items()}),
            )


class BaseProvider(ABC):
    """Abstract base class for all news and research paper ingestion providers.

    Args:
        api_key: Optional API key. Falls back to the ``API_KEY`` environment
            variable when omitted.
        base_url: Root URL of the provider API. A trailing slash is stripped.
        timeout: Read/write timeout in seconds. Connect and pool timeouts are
            fixed at three seconds.
        client: Optional pre-configured asynchronous HTTP client.
        sync_client: Optional pre-configured synchronous HTTP client used by
            legacy aliases. Lazily created on first access when omitted.
        max_concurrency: Maximum number of concurrent asynchronous requests
            issued by this provider instance.
        http_cache: Optional SQLite-backed conditional HTTP cache.
        quota_governor: Optional proactive rate-limit / quota governor.
        quota_provider: Provider identifier used when reserving quota leases.
    """

    MAX_ATTEMPTS = 3
    AUTH_STATUS_CODES = frozenset({401, 403})
    RETRY_STATUS_CODES = frozenset({429})

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "",
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
        max_concurrency: int = 5,
        *,
        http_cache: ConditionalHTTPCache | None = None,
        quota_governor: SQLiteQuotaGovernor | None = None,
        quota_provider: str | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_concurrency = max_concurrency

        self._timeout_config = httpx.Timeout(
            connect=3.0,
            read=self.timeout,
            write=self.timeout,
            pool=3.0,
        )
        self._limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        self._default_headers: dict[str, str] = {}
        self.client = client or httpx.AsyncClient(
            timeout=self._timeout_config, limits=self._limits
        )
        self._sync_client: httpx.Client | None = sync_client
        self.http_cache = http_cache
        self.quota_governor = quota_governor
        self.quota_provider = quota_provider or type(self).__name__.lower()

        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    # -------------------------------------------------------------------------
    # Abstract contract
    # -------------------------------------------------------------------------

    @abstractmethod
    async def fetch_recent(self, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch the most recent items from the provider."""

    @abstractmethod
    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """Search the provider for items matching a query string."""

    @abstractmethod
    async def fetch_articles(self, limit: int = 10, **kwargs: Any) -> list[Article]:
        """Fetch normalized articles from the provider."""

    # -------------------------------------------------------------------------
    # Shared HTTP helpers
    # -------------------------------------------------------------------------

    def _set_default_headers(self, headers: Mapping[str, str]) -> None:
        """Apply default headers to the async client and any future/existing sync client."""
        clean = {str(k): str(v) for k, v in headers.items()}
        self._default_headers.update(clean)
        self.client.headers.update(clean)
        if self._sync_client is not None:
            self._sync_client.headers.update(clean)

    @property
    def sync_client(self) -> httpx.Client:
        """Lazily initialized synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self._timeout_config,
                limits=self._limits,
                headers=self._default_headers,
            )
        return self._sync_client

    @sync_client.setter
    def sync_client(self, value: httpx.Client) -> None:
        """Replace the synchronous HTTP client."""
        self._sync_client = value

    @property
    def session(self) -> httpx.Client:
        """Legacy alias for :attr:`sync_client`."""
        return self.sync_client

    @session.setter
    def session(self, value: httpx.Client) -> None:
        """Replace the synchronous client via the legacy alias."""
        self.sync_client = value

    def _build_url(self, endpoint: str) -> str:
        """Join ``endpoint`` onto :attr:`base_url`."""
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        if not endpoint:
            return self.base_url
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    @staticmethod
    def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Drop parameters whose value is ``None``."""
        return {key: value for key, value in (params or {}).items() if value is not None}

    @classmethod
    def _is_retryable_status(cls, status_code: int) -> bool:
        """Return whether an HTTP status warrants another attempt."""
        return status_code in cls.RETRY_STATUS_CODES or 500 <= status_code < 600

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        """Parse a ``Retry-After`` header into seconds."""
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

    @classmethod
    def _backoff_delay(cls, attempt: int) -> float:
        """Compute exponential backoff with jitter for a failed attempt."""
        return 0.5 * 2 ** (attempt - 1) + random.uniform(0.0, 0.25)

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Choose the delay before retrying a failed HTTP response."""
        retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
        if retry_after is not None:
            return retry_after
        return self._backoff_delay(attempt)

    def _raise_for_error_status(
        self, status: int, body_text: str, url: str
    ) -> None:
        """Raise typed provider errors for 4xx/5xx responses."""
        provider = type(self).__name__
        if status in self.AUTH_STATUS_CODES:
            raise ProviderAuthError(
                f"Authentication failed with HTTP {status}",
                status_code=status,
                provider=provider,
                url=url,
            )
        if status >= 400:
            raise ProviderHTTPError(
                f"Request failed with HTTP {status}",
                status_code=status,
                body=body_text,
                provider=provider,
                url=url,
            )

    def _parse_response(self, response: httpx.Response, url: str) -> dict[str, Any]:
        """Map a response to a payload or a typed provider error."""
        status = response.status_code
        self._raise_for_error_status(status, response.text, url)
        provider = type(self).__name__
        try:
            payload = response.json()
        except ValueError as err:
            raise ProviderHTTPError(
                "Provider returned invalid JSON",
                status_code=status,
                body=response.text,
                provider=provider,
                url=url,
            ) from err
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"data": payload}
        raise ProviderHTTPError(
            "Provider returned an unsupported JSON payload",
            status_code=status,
            body=response.text,
            provider=provider,
            url=url,
        )

    def _transport_error(self, err: Exception, url: str) -> ProviderTransportError:
        """Build a typed transport error for a failed request."""
        return ProviderTransportError(
            f"Request failed: {err}",
            provider=type(self).__name__,
            url=url,
        )

    def _log_retry(self, url: str, attempt: int, reason: str, delay: float) -> None:
        """Log a retry decision without leaking credentials."""
        logger.info(
            "Retrying %s after attempt %d/%d (%s) in %.2fs",
            url,
            attempt,
            self.MAX_ATTEMPTS,
            reason,
            delay,
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return a semaphore bound to the running event loop."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    @staticmethod
    def _canonical_body_bytes(json_body: Any) -> bytes:
        """Serialize ``json_body`` deterministically for cache key generation."""
        if json_body is None:
            return b""
        return json.dumps(json_body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    # -------------------------------------------------------------------------
    # Asynchronous multi-format request path
    # -------------------------------------------------------------------------

    async def _dispatch_once(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None,
        json_body: Any,
    ) -> httpx.Response:
        """Dispatch a single HTTP attempt under optional quota reservation."""
        if self.quota_governor is None:
            return await self.client.request(
                method,
                url,
                params=params,
                headers=headers,
                json=json_body,
            )
        from Articles.quota import parse_rate_limit_headers

        async with self.quota_governor.reserve(self.quota_provider, tokens=0) as lease:
            lease.mark_dispatched()
            response = await self.client.request(
                method,
                url,
                params=params,
                headers=headers,
                json=json_body,
            )
            snapshot = parse_rate_limit_headers(response.headers)
            await lease.reconcile(0, snapshot)
            return response

    async def _request_bytes(
        self,
        endpoint: str,
        *,
        method: Literal["GET", "POST"] = "GET",
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        cache_ttl_s: float | None = None,
        allow_post_cache: bool = False,
    ) -> RawHTTPResponse:
        """Execute an asynchronous HTTP request returning raw bytes and metadata.

        Supports optional SQLite conditional HTTP caching (``ETag`` /
        ``If-Modified-Since`` on ``GET``, body-hash TTL caching on deterministic
        ``POST`` when ``allow_post_cache=True``) and proactive quota governance.
        """
        norm_method = method.upper()
        if norm_method not in ("GET", "POST"):
            raise ValueError(f"unsupported HTTP method: {method!r}")

        url = self._build_url(endpoint)
        clean_params = self._clean_params(params)
        req_headers: dict[str, str] = dict(headers) if headers is not None else {}
        accept_header = req_headers.get("Accept") or req_headers.get("accept") or ""
        body_bytes = self._canonical_body_bytes(json_body)

        cache_eligible = (
            self.http_cache is not None
            and cache_ttl_s is not None
            and cache_ttl_s > 0
            and (norm_method == "GET" or (norm_method == "POST" and allow_post_cache))
        )
        cache_key: str | None = None
        stale_entry: Any = None

        if cache_eligible and self.http_cache is not None:
            lookup = await self.http_cache.lookup(
                norm_method,
                url,
                params=clean_params,
                body_bytes=body_bytes,
                accept=accept_header,
            )
            cache_key = lookup.key
            if lookup.entry is not None and lookup.is_fresh:
                return RawHTTPResponse(
                    status_code=lookup.entry.status_code,
                    body=lookup.entry.body,
                    headers=lookup.entry.headers,
                    from_cache=True,
                )
            if lookup.entry is not None and norm_method == "GET":
                stale_entry = lookup.entry
                if stale_entry.etag and "If-None-Match" not in req_headers:
                    req_headers["If-None-Match"] = stale_entry.etag
                if stale_entry.last_modified and "If-Modified-Since" not in req_headers:
                    req_headers["If-Modified-Since"] = stale_entry.last_modified

        effective_headers = req_headers if req_headers else None
        last_error: ProviderTransportError | None = None

        async with self._get_semaphore():
            for attempt in range(1, self.MAX_ATTEMPTS + 1):
                try:
                    response = await self._dispatch_once(
                        norm_method,
                        url,
                        params=clean_params,
                        headers=effective_headers,
                        json_body=json_body,
                    )
                except httpx.RequestError as err:
                    last_error = self._transport_error(err, url)
                    if attempt >= self.MAX_ATTEMPTS:
                        raise last_error from err
                    delay = self._backoff_delay(attempt)
                    self._log_retry(url, attempt, str(err), delay)
                    await asyncio.sleep(delay)
                    continue

                if self._is_retryable_status(response.status_code) and attempt < self.MAX_ATTEMPTS:
                    delay = self._retry_delay(response, attempt)
                    self._log_retry(url, attempt, f"HTTP {response.status_code}", delay)
                    await asyncio.sleep(delay)
                    continue

                if (
                    response.status_code == 304
                    and norm_method == "GET"
                    and stale_entry is not None
                    and self.http_cache is not None
                    and cache_key is not None
                    and cache_ttl_s is not None
                ):
                    refreshed = await self.http_cache.touch(
                        cache_key,
                        ttl_s=cache_ttl_s,
                        response_headers=dict(response.headers),
                    )
                    entry = refreshed or stale_entry
                    return RawHTTPResponse(
                        status_code=entry.status_code,
                        body=entry.body,
                        headers=entry.headers,
                        from_cache=True,
                    )

                self._raise_for_error_status(response.status_code, response.text, url)
                resp_headers = {str(k): str(v) for k, v in response.headers.items()}

                if (
                    cache_eligible
                    and self.http_cache is not None
                    and cache_key is not None
                    and cache_ttl_s is not None
                    and 200 <= response.status_code < 300
                ):
                    await self.http_cache.store(
                        cache_key,
                        status_code=response.status_code,
                        body=response.content,
                        headers=resp_headers,
                        ttl_s=cache_ttl_s,
                    )

                return RawHTTPResponse(
                    status_code=response.status_code,
                    body=response.content,
                    headers=resp_headers,
                    from_cache=False,
                )

        raise last_error or ProviderTransportError(
            "Request failed without a response",
            provider=type(self).__name__,
            url=url,
        )

    async def _request_json_any(
        self,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any]:
        """Execute an asynchronous HTTP request and decode a JSON dict or list."""
        url = self._build_url(endpoint)
        raw = await self._request_bytes(endpoint, **kwargs)
        provider = type(self).__name__
        body_text = raw.body.decode("utf-8", errors="replace")
        try:
            payload = json.loads(body_text)
        except ValueError as err:
            raise ProviderHTTPError(
                "Provider returned invalid JSON",
                status_code=raw.status_code,
                body=body_text,
                provider=provider,
                url=url,
            ) from err
        if isinstance(payload, (dict, list)):
            return payload
        raise ProviderHTTPError(
            "Provider returned an unsupported JSON payload",
            status_code=raw.status_code,
            body=body_text,
            provider=provider,
            url=url,
        )

    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
        *,
        cache_ttl_s: float | None = None,
    ) -> dict[str, Any]:
        """Execute an asynchronous HTTP request with retries and typed errors."""
        norm_method = method.upper()
        if (
            norm_method in ("GET", "POST")
            and (self.http_cache is not None or self.quota_governor is not None)
        ):
            method_lit: Literal["GET", "POST"] = "POST" if norm_method == "POST" else "GET"
            payload = await self._request_json_any(
                endpoint,
                method=method_lit,
                params=params,
                headers=headers,
                json_body=json_data,
                cache_ttl_s=cache_ttl_s,
            )
            if isinstance(payload, list):
                return {"data": payload}
            return payload

        url = self._build_url(endpoint)
        clean_params = self._clean_params(params)
        last_error: ProviderTransportError | None = None

        async with self._get_semaphore():
            for attempt in range(1, self.MAX_ATTEMPTS + 1):
                try:
                    response = await self.client.request(
                        method,
                        url,
                        params=clean_params,
                        headers=headers,
                        json=json_data,
                    )
                except httpx.RequestError as err:
                    last_error = self._transport_error(err, url)
                    if attempt >= self.MAX_ATTEMPTS:
                        raise last_error from err
                    delay = self._backoff_delay(attempt)
                    self._log_retry(url, attempt, str(err), delay)
                    await asyncio.sleep(delay)
                    continue

                if self._is_retryable_status(response.status_code) and attempt < self.MAX_ATTEMPTS:
                    delay = self._retry_delay(response, attempt)
                    self._log_retry(url, attempt, f"HTTP {response.status_code}", delay)
                    await asyncio.sleep(delay)
                    continue

                return self._parse_response(response, url)

        raise last_error or ProviderTransportError(
            "Request failed without a response",
            provider=type(self).__name__,
            url=url,
        )

    # -------------------------------------------------------------------------
    # Synchronous request path (legacy compatibility)
    # -------------------------------------------------------------------------

    def _make_request_sync(
        self,
        endpoint: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a synchronous HTTP request with the shared retry policy."""
        url = self._build_url(endpoint)
        clean_params = self._clean_params(params)
        last_error: ProviderTransportError | None = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                response = self.sync_client.request(
                    method,
                    url,
                    params=clean_params,
                    headers=headers,
                    json=json_data,
                )
            except httpx.RequestError as err:
                last_error = self._transport_error(err, url)
                if attempt >= self.MAX_ATTEMPTS:
                    raise last_error from err
                delay = self._backoff_delay(attempt)
                self._log_retry(url, attempt, str(err), delay)
                time.sleep(delay)
                continue

            if self._is_retryable_status(response.status_code) and attempt < self.MAX_ATTEMPTS:
                delay = self._retry_delay(response, attempt)
                self._log_retry(url, attempt, f"HTTP {response.status_code}", delay)
                time.sleep(delay)
                continue

            return self._parse_response(response, url)

        raise last_error or ProviderTransportError(
            "Request failed without a response",
            provider=type(self).__name__,
            url=url,
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the asynchronous client and any initialized synchronous client."""
        await self.client.aclose()
        if self._sync_client is not None:
            self._sync_client.close()

    def close(self) -> None:
        """Close the synchronous HTTP client if initialized."""
        if self._sync_client is not None:
            self._sync_client.close()

    async def __aenter__(self) -> Self:
        """Enter an asynchronous context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close owned clients when leaving an asynchronous context."""
        await self.aclose()

    def __enter__(self) -> Self:
        """Enter a synchronous context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the synchronous client when leaving a synchronous context."""
        self.close()
