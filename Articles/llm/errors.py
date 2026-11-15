"""Typed exceptions raised by the LLM layer.

Every failure derives from :class:`LLMError`. The hierarchy carries two pieces
of routing information: whether the same candidate may be retried, and enough
context to decide whether another provider should be tried. No exception ever
contains an API key or an ``Authorization`` header.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LLMAuthError",
    "LLMConfigError",
    "LLMError",
    "LLMHTTPError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMTransportError",
    "LLMUnavailableError",
]


class LLMError(Exception):
    """Base class for every LLM-layer failure.

    Args:
        message: Human-readable description.
        provider: Provider name, when known.
        model: Concrete model id, when known.
        retryable: Whether the same candidate may be attempted again.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.retryable = retryable

    def _context_parts(self) -> list[str]:
        """Build diagnostic context fragments.

        Returns:
            Fragments such as ``provider='groq'`` or ``model='...'``.
        """
        parts: list[str] = []
        if self.provider is not None:
            parts.append(f"provider={self.provider!r}")
        if self.model is not None:
            parts.append(f"model={self.model!r}")
        return parts

    def __str__(self) -> str:
        """Return the message with available diagnostic context appended."""
        parts = self._context_parts()
        if not parts:
            return self.message
        return f"{self.message} [{', '.join(parts)}]"


class LLMConfigError(LLMError):
    """Raised when providers, models, routes, or overrides are invalid."""


class LLMAuthError(LLMError):
    """Raised when a provider rejects credentials.

    Args:
        message: Human-readable description.
        status_code: HTTP status, usually 401 or 403.
        provider: Provider name, when known.
        model: Concrete model id, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, retryable=False)
        self.status_code = status_code

    def _context_parts(self) -> list[str]:
        """Return base context plus the HTTP status code."""
        return [*super()._context_parts(), f"status_code={self.status_code}"]


class LLMRateLimitError(LLMError):
    """Raised when a provider rate-limits a request.

    Args:
        message: Human-readable description.
        retry_after: Delay requested by the provider, in seconds.
        provider: Provider name, when known.
        model: Concrete model id, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, retryable=True)
        self.retry_after = retry_after


class LLMHTTPError(LLMError):
    """Raised for a non-success HTTP response.

    Args:
        message: Human-readable description.
        status_code: HTTP status code.
        body: Response body retained for diagnostics.
        provider: Provider name, when known.
        model: Concrete model id, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: Any = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            retryable=500 <= status_code < 600,
        )
        self.status_code = status_code
        self.body = body

    def _context_parts(self) -> list[str]:
        """Return base context plus the HTTP status code."""
        return [*super()._context_parts(), f"status_code={self.status_code}"]


class LLMTransportError(LLMError):
    """Raised when a request fails before a response is received."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, retryable=True)


class LLMResponseError(LLMError):
    """Raised when a provider response cannot be normalized."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, retryable=True)


class LLMUnavailableError(LLMError):
    """Raised when no configured or healthy provider can serve a route."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, retryable=False)
