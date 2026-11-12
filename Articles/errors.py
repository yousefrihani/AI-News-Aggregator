"""Typed exceptions raised by news and paper providers.

All provider failures derive from :class:`ProviderError`, allowing callers to
isolate a single failing source without catching unrelated exceptions. The
hierarchy distinguishes configuration, authentication, HTTP, and transport
failures so retry logic can make a safe decision.
"""

from typing import Any

__all__ = [
    "ProviderAuthError",
    "ProviderConfigError",
    "ProviderError",
    "ProviderHTTPError",
    "ProviderTransportError",
]


class ProviderError(Exception):
    """Base class for every provider failure.

    Args:
        message: Human-readable description of the failure.
        provider: Optional provider name for diagnostics.
        url: Optional request URL for diagnostics.

    Attributes:
        message: Human-readable description of the failure.
        provider: Provider name, when known.
        url: Request URL, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.url = url

    def _context_parts(self) -> list[str]:
        """Build diagnostic context fragments for the string representation.

        Returns:
            Context fragments such as ``provider='NewsAPI'`` or ``url='...'``.
        """
        parts: list[str] = []
        if self.provider is not None:
            parts.append(f"provider={self.provider!r}")
        if self.url is not None:
            parts.append(f"url={self.url!r}")
        return parts

    def __str__(self) -> str:
        """Return the message with available diagnostic context appended."""
        parts = self._context_parts()
        if not parts:
            return self.message
        return f"{self.message} [{', '.join(parts)}]"


class ProviderConfigError(ProviderError):
    """Raised when a provider is misconfigured, for example a missing API key."""


class ProviderAuthError(ProviderError):
    """Raised when a provider rejects the supplied credentials.

    Authentication failures are not retryable. The HTTP status code is kept so
    callers can distinguish 401 (invalid key) from 403 (insufficient access).

    Args:
        message: Human-readable description of the failure.
        status_code: HTTP status code returned by the provider, usually 401 or 403.
        provider: Optional provider name for diagnostics.
        url: Optional request URL for diagnostics.

    Attributes:
        status_code: HTTP status code returned by the provider.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        provider: str | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, url=url)
        self.status_code = status_code

    def _context_parts(self) -> list[str]:
        """Return base context plus the HTTP status code."""
        return [*super()._context_parts(), f"status_code={self.status_code}"]


class ProviderHTTPError(ProviderError):
    """Raised when a provider returns an unsuccessful HTTP response.

    Args:
        message: Human-readable description of the failure.
        status_code: HTTP status code returned by the provider.
        body: Optional response body retained for diagnostics.
        provider: Optional provider name for diagnostics.
        url: Optional request URL for diagnostics.

    Attributes:
        status_code: HTTP status code returned by the provider.
        body: Response body, when captured by the provider.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: Any = None,
        provider: str | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, url=url)
        self.status_code = status_code
        self.body = body

    def _context_parts(self) -> list[str]:
        """Return base context plus the HTTP status code."""
        return [*super()._context_parts(), f"status_code={self.status_code}"]


class ProviderTransportError(ProviderError):
    """Raised when a request fails before a response is received.

    Typical causes are connection errors, timeouts, and DNS failures. These
    failures are generally retryable, subject to the caller's backoff policy.
    """
