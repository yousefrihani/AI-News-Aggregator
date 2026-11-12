"""Normalized, immutable domain models shared by all ingestion providers.

The models use the standard library only and enforce the small set of
invariants relied upon by aggregators and consumers: required identifiers are
non-empty and publication timestamps are timezone-aware.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

__all__ = ["Article", "Paper"]


def _require_non_empty(name: str, value: str) -> None:
    """Validate that a required string field is present and non-blank.

    Args:
        name: Field name used in the error message.
        value: Value to validate.

    Raises:
        ValueError: If ``value`` is not a string or is empty/whitespace only.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_aware(name: str, value: datetime | None) -> None:
    """Validate that an optional timestamp is timezone-aware.

    Args:
        name: Field name used in the error message.
        value: Timestamp to validate, or ``None``.

    Raises:
        ValueError: If ``value`` is a naive datetime.
    """
    if value is None:
        return
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware when provided")


@dataclass(frozen=True, slots=True)
class Article:
    """A normalized news article.

    Attributes:
        id: Stable identifier assigned by the provider.
        title: Human-readable article title.
        url: Canonical article URL.
        source: Provider or publication name.
        published_at: Publication timestamp, timezone-aware, or ``None``.
        summary: Optional short summary or description.
        topics: Topic labels associated with the article.
        raw: Original provider payload. Excluded from equality and repr.
    """

    id: str
    title: str
    url: str
    source: str
    published_at: datetime | None
    summary: str | None
    topics: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate required fields and timestamp awareness.

        Raises:
            ValueError: If an identifier field is empty or ``published_at`` is
                a naive datetime.
        """
        _require_non_empty("id", self.id)
        _require_non_empty("title", self.title)
        _require_non_empty("url", self.url)
        _require_aware("published_at", self.published_at)


@dataclass(frozen=True, slots=True)
class Paper:
    """A normalized research paper.

    Attributes:
        id: Stable identifier assigned by the provider.
        title: Human-readable paper title.
        url: Canonical paper URL.
        source: Provider or publication name.
        published_at: Publication timestamp, timezone-aware, or ``None``.
        summary: Optional abstract or summary.
        authors: Author names in provider order.
        topics: Topic labels associated with the paper.
        raw: Original provider payload. Excluded from equality and repr.
    """

    id: str
    title: str
    url: str
    source: str
    published_at: datetime | None
    summary: str | None
    authors: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate required fields and timestamp awareness.

        Raises:
            ValueError: If an identifier field is empty or ``published_at`` is
                a naive datetime.
        """
        _require_non_empty("id", self.id)
        _require_non_empty("title", self.title)
        _require_non_empty("url", self.url)
        _require_aware("published_at", self.published_at)
