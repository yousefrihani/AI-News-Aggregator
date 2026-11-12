"""Concurrent fan-out, de-duplication, and ordering of provider articles.

The aggregator is deliberately decoupled from concrete providers. It depends
only on :class:`ProviderProtocol`, so it can be exercised with fakes and never
imports an HTTP or provider module. Failures are isolated per provider: one
broken source never aborts an aggregation run.

Data flow: ``Provider.fetch_articles()`` -> ``list[Article]`` -> fan-out ->
de-duplicate -> sort -> truncate.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from Articles.Models import Article

__all__ = ["Aggregator", "ProviderProtocol"]

logger = logging.getLogger(__name__)

#: Query-parameter prefixes dropped while canonicalizing a URL.
_DROPPED_QUERY_PREFIXES: tuple[str, ...] = ("utm_",)

#: Exact query-parameter names dropped while canonicalizing a URL.
_DROPPED_QUERY_KEYS: frozenset[str] = frozenset({"fbclid", "gclid"})

#: Sort sentinel that places articles without a timestamp last.
_MISSING_TIMESTAMP = datetime.min.replace(tzinfo=UTC)


@runtime_checkable
class ProviderProtocol(Protocol):
    """Structural contract every article provider must satisfy.

    The aggregator uses this protocol instead of the concrete provider base
    class so that it stays decoupled from the HTTP layer.
    """

    async def fetch_articles(self, limit: int = 10, **kwargs: Any) -> list[Article]:
        """Fetch a batch of normalized articles.

        Args:
            limit: Maximum number of articles to request from the provider.
            **kwargs: Provider-specific filters.

        Returns:
            Articles in provider order.
        """
        ...


def _canonical_url(url: str) -> str:
    """Return a canonical form of ``url`` used as a de-duplication key.

    The scheme and host are lower-cased, the fragment is removed, tracking
    parameters (``utm_*``, ``fbclid``, ``gclid``) are dropped, the remaining
    query parameters are sorted for a stable representation, and trailing
    slashes are stripped from the path. Path case is preserved because paths
    are case-sensitive.

    Args:
        url: Raw article URL.

    Returns:
        Canonical URL, or an empty string when ``url`` is blank or invalid.
    """
    candidate = url.strip()
    if not candidate:
        return ""
    try:
        parts = urlsplit(candidate)
    except ValueError:
        logger.warning("Could not parse article URL %r; using it verbatim", url)
        return candidate

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_DROPPED_QUERY_PREFIXES)
        and key.lower() not in _DROPPED_QUERY_KEYS
    ]
    query = urlencode(sorted(query_pairs))
    return urlunsplit((scheme, netloc, path, query, ""))


def _dedupe_key(article: Article) -> str:
    """Build the de-duplication key for ``article``.

    Articles are keyed by canonical URL. When a URL is empty (or blank), the
    provider-assigned ``id`` is used instead so the article is still kept.

    Args:
        article: Article to key.

    Returns:
        Namespaced key, either ``"url:<canonical>"`` or ``"id:<article id>"``.
    """
    canonical = _canonical_url(article.url)
    if canonical:
        return f"url:{canonical}"
    return f"id:{article.id}"


def _is_timezone_aware(value: datetime) -> bool:
    """Return whether ``value`` carries a usable timezone offset.

    Args:
        value: Datetime to inspect.

    Returns:
        ``True`` when the datetime is timezone-aware.
    """
    return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None


def _sort_key(article: Article) -> datetime:
    """Return the descending sort key for ``article``.

    Args:
        article: Article to sort.

    Returns:
        ``published_at`` when present, otherwise a minimum tz-aware sentinel so
        that missing timestamps sort last under ``reverse=True``.
    """
    if article.published_at is None:
        return _MISSING_TIMESTAMP
    return article.published_at


class Aggregator:
    """Fan out to providers concurrently, then de-duplicate and rank results.

    One shared :class:`asyncio.Semaphore` bounds the number of in-flight
    provider calls. Provider failures are logged and skipped. Duplicate
    canonical URLs keep their first occurrence in provider order. The merged
    list is sorted by ``published_at`` descending (``None`` last) and only then
    truncated to ``limit``.
    """

    def __init__(
        self,
        providers: Sequence[ProviderProtocol],
        max_concurrency: int = 10,
    ) -> None:
        """Initialize the aggregator.

        Args:
            providers: Providers to query. Order determines tie-breaking for
                articles with equal timestamps and de-duplication precedence.
            max_concurrency: Maximum number of provider calls in flight at
                once. Must be at least 1.

        Raises:
            ValueError: If ``max_concurrency`` is less than 1.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._providers = tuple(providers)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def aggregate(self, limit: int = 50, **filters: Any) -> list[Article]:
        """Aggregate articles from every provider.

        Providers run concurrently, so wall-clock time is bounded by the
        slowest provider rather than the sum of all provider latencies. A
        failing provider contributes no articles and does not raise.

        Args:
            limit: Maximum number of articles to return after de-duplication
                and sorting. Values below zero are treated as zero.
            **filters: Provider-specific filters forwarded verbatim.

        Returns:
            De-duplicated articles sorted by ``published_at`` descending, with
            articles lacking a timestamp last.
        """
        results = await asyncio.gather(
            *(self._run(provider, limit, filters) for provider in self._providers),
            return_exceptions=True,
        )

        merged: dict[str, Article] = {}
        for provider, result in zip(self._providers, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "Provider %r failed during aggregation: %r",
                    provider,
                    result,
                    exc_info=result,
                )
                continue
            for article in result:
                if article.published_at is not None and not _is_timezone_aware(
                    article.published_at
                ):
                    logger.warning(
                        "Dropping article %r from %r: published_at is naive",
                        article.id,
                        provider,
                    )
                    continue
                key = _dedupe_key(article)
                if key not in merged:
                    merged[key] = article

        ordered = sorted(merged.values(), key=_sort_key, reverse=True)
        return ordered[: max(limit, 0)]

    async def _run(
        self,
        provider: ProviderProtocol,
        limit: int,
        filters: Mapping[str, Any],
    ) -> list[Article]:
        """Call one provider under the shared concurrency bound.

        Args:
            provider: Provider to query.
            limit: Maximum number of articles to request.
            filters: Provider-specific filters forwarded verbatim.

        Returns:
            Articles returned by the provider.

        Raises:
            Exception: Whatever the provider raises, captured by ``gather``
                with ``return_exceptions=True`` in :meth:`aggregate`.
        """
        async with self._semaphore:
            return await provider.fetch_articles(limit=limit, **filters)
