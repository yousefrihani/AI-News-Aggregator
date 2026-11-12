"""Functional tests for :class:`Articles.Aggregator.Aggregator`.

The tests use in-memory fake providers only; no sockets, no network, no
dependence on the concrete provider implementations.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from Articles.Aggregator import Aggregator, ProviderProtocol, _dedupe_key
from Articles.Models import Article

_BASE_TIME = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def make_article(
    article_id: str,
    *,
    url: str | None = None,
    minutes_ago: int | None = 0,
    source: str = "fake",
    title: str | None = None,
) -> Article:
    """Build a normalized article for use in tests.

    Args:
        article_id: Stable identifier.
        url: Article URL; defaults to ``https://example.com/<id>``.
        minutes_ago: Publication offset from a fixed base time, or ``None``.
        source: Provider or publication name.
        title: Optional title override.

    Returns:
        A timezone-aware :class:`Article`.
    """
    published_at = None if minutes_ago is None else _BASE_TIME - timedelta(minutes=minutes_ago)
    return Article(
        id=article_id,
        title=title or f"Article {article_id}",
        url=url if url is not None else f"https://example.com/{article_id}",
        source=source,
        published_at=published_at,
        summary=None,
    )


class FakeProvider:
    """In-memory provider used to exercise the aggregator."""

    def __init__(
        self,
        articles: list[Article],
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        """Initialize the fake provider.

        Args:
            articles: Articles to return.
            delay: Seconds to sleep before returning.
            error: Exception to raise instead of returning articles.
        """
        self._articles = articles
        self._delay = delay
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def fetch_articles(self, limit: int = 10, **kwargs: Any) -> list[Article]:
        """Return the configured articles, optionally after a delay or error.

        Args:
            limit: Requested maximum; recorded but not enforced.
            **kwargs: Recorded filters.

        Returns:
            Configured articles.

        Raises:
            Exception: The configured ``error``, when set.
        """
        self.calls.append({"limit": limit, **kwargs})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return list(self._articles)


class AggregatorOrderingTests(unittest.IsolatedAsyncioTestCase):
    """Ordering, de-duplication, limiting, and failure-isolation behaviour."""

    async def test_sorts_by_published_at_descending_with_none_last(self) -> None:
        """Articles are ordered newest first and missing timestamps sort last."""
        providers = [
            FakeProvider([make_article("old", minutes_ago=60)]),
            FakeProvider(
                [
                    make_article("undated", minutes_ago=None),
                    make_article("new", minutes_ago=1),
                ]
            ),
            FakeProvider([make_article("middle", minutes_ago=10)]),
        ]

        articles = await Aggregator(providers).aggregate()

        self.assertEqual(
            [article.id for article in articles],
            ["new", "middle", "old", "undated"],
        )
        for provider in providers:
            self.assertEqual(provider.calls, [{"limit": 50}])

    async def test_dedupes_canonical_url_and_keeps_first_occurrence(self) -> None:
        """Equivalent URLs collapse to one article, keeping the first seen."""
        first = make_article(
            "first",
            url="HTTPS://Example.com/Story/?utm_source=newsletter&b=2&a=1#section",
            minutes_ago=5,
        )
        duplicate = make_article(
            "second",
            url="https://example.com/Story?a=1&b=2",
            minutes_ago=1,
            title="Duplicate with newer timestamp",
        )
        other = make_article("third", url="https://example.com/other", minutes_ago=30)

        articles = await Aggregator(
            [FakeProvider([first]), FakeProvider([duplicate, other])]
        ).aggregate()

        self.assertEqual([article.id for article in articles], ["first", "third"])

    async def test_dedupes_tracking_parameters_and_trailing_slash(self) -> None:
        """utm_*, fbclid, gclid, fragments, and trailing slashes are ignored."""
        urls = [
            "https://example.com/path?utm_campaign=x",
            "https://example.com/path/?fbclid=abc",
            "https://example.com/path#top",
            "https://example.com/path?gclid=def",
        ]
        providers = [
            FakeProvider([make_article(str(index), url=url)]) for index, url in enumerate(urls)
        ]

        articles = await Aggregator(providers).aggregate()

        self.assertEqual([article.id for article in articles], ["0"])

    async def test_falls_back_to_id_when_url_is_blank(self) -> None:
        """A blank URL keys the article by its identifier."""
        self.assertEqual(
            _dedupe_key(SimpleNamespace(id="abc", url="   ")),
            "id:abc",
        )
        self.assertEqual(
            _dedupe_key(SimpleNamespace(id="abc", url="")),
            "id:abc",
        )

    async def test_isolates_partial_provider_failure(self) -> None:
        """A failing provider is skipped and logged; others still contribute."""
        failing = FakeProvider([], error=RuntimeError("boom"))
        healthy = FakeProvider([make_article("ok", minutes_ago=1)])

        with self.assertLogs("Articles.Aggregator", level="WARNING") as captured:
            articles = await Aggregator([failing, healthy]).aggregate()

        self.assertEqual([article.id for article in articles], ["ok"])
        self.assertIn("failed during aggregation", "\n".join(captured.output))

    async def test_returns_empty_list_when_all_providers_are_empty(self) -> None:
        """No articles anywhere yields an empty list, not an error."""
        articles = await Aggregator([FakeProvider([]), FakeProvider([])]).aggregate()

        self.assertEqual(articles, [])

    async def test_applies_limit_after_sorting(self) -> None:
        """Only the newest articles survive truncation."""
        providers = [
            FakeProvider([make_article("a", minutes_ago=30)]),
            FakeProvider(
                [
                    make_article("b", minutes_ago=20),
                    make_article("c", minutes_ago=10),
                ]
            ),
        ]

        articles = await Aggregator(providers).aggregate(limit=2)

        self.assertEqual([article.id for article in articles], ["c", "b"])

    async def test_forwards_filters_to_every_provider(self) -> None:
        """Aggregation filters are passed through to each provider."""
        provider = FakeProvider([make_article("a")])

        await Aggregator([provider]).aggregate(limit=3, topic="technology")

        self.assertEqual(
            provider.calls,
            [{"limit": 3, "topic": "technology"}],
        )

    async def test_provider_calls_run_concurrently(self) -> None:
        """Wall clock is ~max provider latency, not the sum of latencies."""
        slow_one = FakeProvider([make_article("a")], delay=0.05)
        slow_two = FakeProvider([make_article("b")], delay=0.05)

        started = time.perf_counter()
        articles = await Aggregator([slow_one, slow_two]).aggregate()
        elapsed = time.perf_counter() - started

        self.assertEqual(len(articles), 2)
        self.assertLess(elapsed, 0.09)

    def test_rejects_non_positive_max_concurrency(self) -> None:
        """Invalid concurrency bounds fail fast."""
        with self.assertRaises(ValueError):
            Aggregator([FakeProvider([])], max_concurrency=0)

    def test_fake_provider_satisfies_protocol(self) -> None:
        """The runtime-checkable protocol accepts structural implementations."""
        self.assertIsInstance(FakeProvider([]), ProviderProtocol)


if __name__ == "__main__":
    unittest.main()
