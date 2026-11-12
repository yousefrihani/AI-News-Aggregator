"""Runtime settings for the AI-News-Aggregator ingestion pipeline.

Settings are immutable and loaded from ``AI_NEWS_*`` environment variables with
conservative defaults. Invalid values raise :class:`ProviderConfigError` so a
misconfigured deployment fails at startup instead of silently using an unsafe
default.
"""

import logging
import os
from dataclasses import dataclass

from Articles.errors import ProviderConfigError

logger = logging.getLogger(__name__)

__all__ = ["Settings"]


def _parse_int(name: str, raw: str | None, default: int, minimum: int = 1) -> int:
    """Parse a positive integer environment variable.

    Args:
        name: Environment variable name, used in error messages.
        raw: Raw environment value, or ``None`` when unset.
        default: Value returned when ``raw`` is ``None`` or blank.
        minimum: Smallest accepted value.

    Returns:
        Parsed integer.

    Raises:
        ProviderConfigError: If the value is not an integer or is too small.
    """
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as err:
        raise ProviderConfigError(f"{name} must be an integer, got {raw!r}") from err
    if value < minimum:
        raise ProviderConfigError(f"{name} must be at least {minimum}, got {value}")
    return value


def _parse_float(name: str, raw: str | None, default: float) -> float:
    """Parse a positive float environment variable.

    Args:
        name: Environment variable name, used in error messages.
        raw: Raw environment value, or ``None`` when unset.
        default: Value returned when ``raw`` is ``None`` or blank.

    Returns:
        Parsed float.

    Raises:
        ProviderConfigError: If the value is not a float or is not positive.
    """
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as err:
        raise ProviderConfigError(f"{name} must be a number, got {raw!r}") from err
    if value <= 0:
        raise ProviderConfigError(f"{name} must be greater than 0, got {value}")
    return value


def _parse_topics(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated topic list.

    Args:
        raw: Raw environment value, or ``None`` when unset.
        default: Topics returned when ``raw`` is unset or blank.

    Returns:
        A non-empty tuple of stripped topic names.

    Raises:
        ProviderConfigError: If the value contains no usable topic names.
    """
    if raw is None or not raw.strip():
        return default
    topics = tuple(topic.strip() for topic in raw.split(",") if topic.strip())
    if not topics:
        raise ProviderConfigError("AI_NEWS_TOPICS must contain at least one topic")
    return topics


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration.

    Args:
        http_timeout: Read/write timeout in seconds for provider requests.
        max_concurrency: Maximum concurrent requests per provider.
        max_pages: Upper bound on pagination loops.
        max_items: Upper bound on items fetched per provider call.
        topics: Default topics used when a caller does not specify one.
        page_size: Items requested per pagination page.

    Raises:
        ProviderConfigError: If any value violates its documented bound.
    """

    http_timeout: float = 10.0
    max_concurrency: int = 5
    max_pages: int = 10
    max_items: int = 500
    topics: tuple[str, ...] = ("technology",)
    page_size: int = 50

    def __post_init__(self) -> None:
        """Validate configuration bounds.

        Raises:
            ProviderConfigError: If a field is out of range or empty.
        """
        if self.http_timeout <= 0:
            raise ProviderConfigError("http_timeout must be greater than 0")
        if self.max_concurrency < 1:
            raise ProviderConfigError("max_concurrency must be at least 1")
        if self.max_pages < 1:
            raise ProviderConfigError("max_pages must be at least 1")
        if self.max_items < 1:
            raise ProviderConfigError("max_items must be at least 1")
        if self.page_size < 1:
            raise ProviderConfigError("page_size must be at least 1")
        if not self.topics or any(not topic.strip() for topic in self.topics):
            raise ProviderConfigError("topics must contain at least one non-empty topic")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``AI_NEWS_*`` environment variables.

        Unset or blank variables fall back to the documented defaults.

        Returns:
            A validated :class:`Settings` instance.

        Raises:
            ProviderConfigError: If a configured value is malformed.
        """
        return cls(
            http_timeout=_parse_float(
                "AI_NEWS_HTTP_TIMEOUT", os.getenv("AI_NEWS_HTTP_TIMEOUT"), 10.0
            ),
            max_concurrency=_parse_int(
                "AI_NEWS_MAX_CONCURRENCY", os.getenv("AI_NEWS_MAX_CONCURRENCY"), 5
            ),
            max_pages=_parse_int("AI_NEWS_MAX_PAGES", os.getenv("AI_NEWS_MAX_PAGES"), 10),
            max_items=_parse_int("AI_NEWS_MAX_ITEMS", os.getenv("AI_NEWS_MAX_ITEMS"), 500),
            topics=_parse_topics(
                os.getenv("AI_NEWS_TOPICS"),
                ("technology",),
            ),
            page_size=_parse_int("AI_NEWS_PAGE_SIZE", os.getenv("AI_NEWS_PAGE_SIZE"), 50),
        )
