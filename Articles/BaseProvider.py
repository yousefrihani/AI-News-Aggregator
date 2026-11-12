"""Async-first base provider for data sources in AI-News-Aggregator.

Every ingestion provider shares one configured ``httpx.AsyncClient`` for its
asynchronous API and one ``httpx.Client`` for the legacy synchronous aliases.
Both request paths use the same retry policy and HTTP-status mapping so a
provider cannot drift between the two flavours.

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

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Self

import httpx

from Articles.errors import (
    ProviderAuthError,
    ProviderHTTPError,
    ProviderTransportError,
)
from Articles.Models import Article

try:  # pragma: no cover - optional convenience for local development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is optional at runtime
    pass

logger = logging.getLogger(__name__)

__all__ = ["BaseProvider"]


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
            legacy aliases.
        max_concurrency: Maximum number of concurrent asynchronous requests
            issued by this provider instance.
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
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_concurrency = max_concurrency

        timeout_config = httpx.Timeout(
            connect=3.0,
            read=self.timeout,
            write=self.timeout,
            pool=3.0,
        )
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        self.client = client or httpx.AsyncClient(timeout=timeout_config, limits=limits)
        self.sync_client = sync_client or httpx.Client(timeout=timeout_config, limits=limits)

        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    # -------------------------------------------------------------------------
    # Abstract contract
    # -------------------------------------------------------------------------

    @abstractmethod
    async def fetch_recent(self, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch the most recent items from the provider.

        Args:
            limit: Maximum number of items to retrieve.
            **kwargs: Provider-specific filtering parameters.

        Returns:
            A list of raw provider item dictionaries.
        """

    @abstractmethod
    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """Search the provider for items matching a query string.

        Args:
            query: The search term or topic.
            limit: Maximum number of items to retrieve.
            **kwargs: Provider-specific filtering parameters.

        Returns:
            A list of raw provider item dictionaries.
        """

    @abstractmethod
    async def fetch_articles(self, limit: int = 10, **kwargs: Any) -> list[Article]:
        """Fetch normalized articles from the provider.

        Args:
            limit: Maximum number of articles to return.
            **kwargs: Provider-specific filtering and pagination parameters.

        Returns:
            A list of normalized :class:`~Articles.Models.Article` instances.
        """

    # -------------------------------------------------------------------------
    # Shared HTTP helpers
    # -------------------------------------------------------------------------

    @property
    def session(self) -> httpx.Client:
        """Legacy alias for :attr:`sync_client`.

        Returns:
            The synchronous HTTP client owned by this provider.
        """
        return self.sync_client

    @session.setter
    def session(self, value: httpx.Client) -> None:
        """Replace the synchronous client via the legacy alias."""
        self.sync_client = value

    def _build_url(self, endpoint: str) -> str:
        """Join ``endpoint`` onto :attr:`base_url`.

        Args:
            endpoint: Relative API path such as ``"/news"``.

        Returns:
            The absolute request URL.
        """
        if not endpoint:
            return self.base_url
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    @staticmethod
    def _clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
        """Drop parameters whose value is ``None``.

        Args:
            params: Raw query parameters.

        Returns:
            A new dictionary without ``None`` values.
        """
        return {key: value for key, value in (params or {}).items() if value is not None}

    @classmethod
    def _is_retryable_status(cls, status_code: int) -> bool:
        """Return whether an HTTP status warrants another attempt.

        Args:
            status_code: Response status code.

        Returns:
            ``True`` for ``429`` and any ``5xx`` status.
        """
        return status_code in cls.RETRY_STATUS_CODES or 500 <= status_code < 600

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        """Parse a ``Retry-After`` header into seconds.

        Args:
            value: Raw header value, either delay seconds or an HTTP date.

        Returns:
            Non-negative delay in seconds, or ``None`` when absent/invalid.
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

    @classmethod
    def _backoff_delay(cls, attempt: int) -> float:
        """Compute exponential backoff with jitter for a failed attempt.

        Args:
            attempt: One-based attempt number that just failed.

        Returns:
            Delay in seconds.
        """
        return 0.5 * 2 ** (attempt - 1) + random.uniform(0.0, 0.25)

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Choose the delay before retrying a failed HTTP response.

        Args:
            response: Failed response, possibly carrying ``Retry-After``.
            attempt: One-based attempt number that just failed.

        Returns:
            Delay in seconds.
        """
        retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
        if retry_after is not None:
            return retry_after
        return self._backoff_delay(attempt)

    def _parse_response(self, response: httpx.Response, url: str) -> dict[str, Any]:
        """Map a response to a payload or a typed provider error.

        Args:
            response: Completed HTTP response.
            url: Request URL used for diagnostics.

        Returns:
            Parsed JSON object. Top-level JSON lists are wrapped under ``data``.

        Raises:
            ProviderAuthError: For ``401`` and ``403`` responses.
            ProviderHTTPError: For other unsuccessful or malformed responses.
        """
        status = response.status_code
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
                body=response.text,
                provider=provider,
                url=url,
            )
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

    def _transport_error(self, err: Exception, url: str) -> "ProviderTransportError":
        """Build a typed transport error for a failed request.

        Args:
            err: Lower-level ``httpx`` exception.
            url: Request URL used for diagnostics.

        Returns:
            A :class:`ProviderTransportError` instance.
        """
        return ProviderTransportError(
            f"Request failed: {err}",
            provider=type(self).__name__,
            url=url,
        )

    def _log_retry(self, url: str, attempt: int, reason: str, delay: float) -> None:
        """Log a retry decision without leaking credentials.

        Args:
            url: Request URL.
            attempt: Attempt that just failed.
            reason: Short failure description.
            delay: Delay before the next attempt in seconds.
        """
        logger.info(
            "Retrying %s after attempt %d/%d (%s) in %.2fs",
            url,
            attempt,
            self.MAX_ATTEMPTS,
            reason,
            delay,
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return a semaphore bound to the running event loop.

        Creates a new semaphore when the provider is reused across loops, which
        keeps ``IsolatedAsyncioTestCase`` and short-lived workers safe.

        Returns:
            Semaphore enforcing :attr:`max_concurrency`.
        """
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    # -------------------------------------------------------------------------
    # Asynchronous request path
    # -------------------------------------------------------------------------

    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an asynchronous HTTP request with retries and typed errors.

        Args:
            endpoint: API endpoint path, for example ``"/news"``.
            method: HTTP method.
            params: Optional query parameters; ``None`` values are dropped.
            headers: Optional per-request headers.
            json_data: Optional JSON body for write requests.

        Returns:
            Parsed JSON object.

        Raises:
            ProviderAuthError: For ``401``/``403`` responses.
            ProviderHTTPError: For non-retryable or exhausted HTTP failures.
            ProviderTransportError: When every transport attempt fails.
        """
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

        # Defensive: every loop iteration either returns or raises.
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
        """Execute a synchronous HTTP request with the shared retry policy.

        This path exists only for legacy aliases and must never be called from
        asynchronous code.

        Args:
            endpoint: API endpoint path, for example ``"/news"``.
            method: HTTP method.
            params: Optional query parameters; ``None`` values are dropped.
            headers: Optional per-request headers.
            json_data: Optional JSON body for write requests.

        Returns:
            Parsed JSON object.

        Raises:
            ProviderAuthError: For ``401``/``403`` responses.
            ProviderHTTPError: For non-retryable or exhausted HTTP failures.
            ProviderTransportError: When every transport attempt fails.
        """
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
        """Close both the asynchronous and synchronous HTTP clients."""
        await self.client.aclose()
        self.sync_client.close()

    def close(self) -> None:
        """Close the synchronous HTTP client.

        Asynchronous clients are closed with :meth:`aclose`; this method exists
        for the legacy synchronous context manager.
        """
        self.sync_client.close()

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
