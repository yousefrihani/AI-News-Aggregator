"""Unit tests for the ``Articles.cli`` command-line entry point."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from Articles.cli import build_parser, main, run, serialize
from Articles.errors import ProviderAuthError
from Articles.Models import Article


def make_article(article_id: str = "1") -> Article:
    """Build a representative article for CLI tests."""
    return Article(
        id=article_id,
        title="AI news",
        url=f"https://example.com/{article_id}",
        source="Example",
        published_at=datetime(2026, 11, 12, 12, 0, tzinfo=UTC),
        summary="Summary",
        topics=("technology",),
    )


class FakeProvider:
    """Minimal async context manager matching the provider surface."""

    def __init__(self, api_key: str | None = "key") -> None:
        self.api_key = api_key
        self.closed = False

    async def __aenter__(self) -> FakeProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self.closed = True


class FakeAggregator:
    """Aggregator stand-in with controllable output and failure."""

    articles: list[Article] = [make_article()]
    error: Exception | None = None

    def __init__(
        self,
        providers: object,
        max_concurrency: int = 10,
        *,
        raise_on_all_failed: bool = False,
    ) -> None:
        pass

    async def aggregate(self, limit: int = 50, **filters: object) -> list[Article]:
        if self.error is not None:
            raise self.error
        return self.articles[:limit]


class TestCli(unittest.TestCase):
    """CLI parsing, serialization, and exit-code behaviour."""

    def tearDown(self) -> None:
        FakeAggregator.error = None
        FakeAggregator.articles = [make_article()]

    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.limit, 20)
        self.assertIsNone(args.topic)
        self.assertIsNone(args.query)
        self.assertFalse(args.as_json)
        self.assertFalse(args.verbose)

    def test_parser_accepts_options(self) -> None:
        args = build_parser().parse_args(
            ["--limit", "5", "--topic", "ai", "--query", "agents", "--json", "-v"]
        )
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.topic, "ai")
        self.assertEqual(args.query, "agents")
        self.assertTrue(args.as_json)
        self.assertTrue(args.verbose)

    def test_serialize_uses_iso_timestamp_and_list_topics(self) -> None:
        payload = serialize(make_article("42"))
        self.assertEqual(payload["id"], "42")
        self.assertEqual(payload["published_at"], "2026-11-12T12:00:00+00:00")
        self.assertEqual(payload["topics"], ["technology"])

    def test_run_emits_json(self) -> None:
        args = build_parser().parse_args(["--json", "--limit", "5"])
        output = io.StringIO()
        with (
            patch("Articles.cli.NewsAPI", return_value=FakeProvider()),
            patch("Articles.cli.Aggregator", FakeAggregator),
            contextlib.redirect_stdout(output),
        ):
            code = asyncio.run(run(args))

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload[0]["id"], "1")

    def test_run_emits_text(self) -> None:
        args = build_parser().parse_args([])
        output = io.StringIO()
        with (
            patch("Articles.cli.NewsAPI", return_value=FakeProvider()),
            patch("Articles.cli.Aggregator", FakeAggregator),
            contextlib.redirect_stdout(output),
        ):
            code = asyncio.run(run(args))

        self.assertEqual(code, 0)
        self.assertIn("Example | AI news", output.getvalue())

    def test_run_without_api_key_returns_2(self) -> None:
        args = build_parser().parse_args([])
        error = io.StringIO()
        with (
            patch("Articles.cli.NewsAPI", return_value=FakeProvider(api_key=None)),
            contextlib.redirect_stderr(error),
        ):
            code = asyncio.run(run(args))

        self.assertEqual(code, 2)
        self.assertIn("no API key configured", error.getvalue())

    def test_run_provider_error_returns_2(self) -> None:
        FakeAggregator.error = ProviderAuthError("bad key", status_code=401)
        args = build_parser().parse_args([])
        error = io.StringIO()
        with (
            patch("Articles.cli.NewsAPI", return_value=FakeProvider()),
            patch("Articles.cli.Aggregator", FakeAggregator),
            contextlib.redirect_stderr(error),
        ):
            code = asyncio.run(run(args))

        self.assertEqual(code, 2)
        self.assertIn("bad key", error.getvalue())

    def test_main_delegates_to_run(self) -> None:
        with patch("Articles.cli.run", new=AsyncMock(return_value=0)) as mocked:
            code = main(["--limit", "3"])

        self.assertEqual(code, 0)
        mocked.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
