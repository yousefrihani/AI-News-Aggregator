"""Command-line entry point for AI-News-Aggregator.

Fetches recent articles through the configured provider, aggregates them, and
prints either human-readable text or JSON. Designed to run inside the container
image as ``python -m Articles``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from Articles.Aggregator import Aggregator
from Articles.errors import ProviderError
from Articles.Models import Article
from Articles.NewsApi import NewsAPI

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured parser for the ``ai-news-aggregator`` command.
    """
    parser = argparse.ArgumentParser(
        prog="ai-news-aggregator",
        description="Fetch recent AI/technology news and print them as text or JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="maximum number of articles to return (default: 20)",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="provider topic filter (default: AI_NEWS_TOPICS or technology)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="title keyword search instead of the latest feed",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit JSON instead of human-readable text",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def serialize(article: Article) -> dict[str, Any]:
    """Convert an :class:`Article` into a JSON-serializable dictionary.

    Args:
        article: Article to serialize.

    Returns:
        Dictionary with stable keys and ISO-8601 timestamps.
    """
    return {
        "id": article.id,
        "title": article.title,
        "url": article.url,
        "source": article.source,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "summary": article.summary,
        "topics": list(article.topics),
    }


def _render_text(article: Article) -> str:
    """Render one article as a two-line text block.

    Args:
        article: Article to render.

    Returns:
        Formatted text block.
    """
    published = article.published_at.isoformat() if article.published_at else "unknown"
    return f"{published} | {article.source} | {article.title}\n  {article.url}"


async def run(args: argparse.Namespace) -> int:
    """Execute one aggregation run.

    Args:
        args: Parsed command-line arguments.

    Returns:
        ``0`` on success (including an empty result), ``2`` on a provider or
        configuration error.
    """
    provider = NewsAPI()
    filters: dict[str, Any] = {}
    if args.topic:
        filters["topic"] = args.topic
    if args.query:
        filters["in_title"] = args.query

    if not provider.api_key:
        print(
            "error: no API key configured (set FREENEWSAPI_API_KEY, NEWS_API_KEY, or NEWSAPI_KEY)",
            file=sys.stderr,
        )
        await provider.aclose()
        return 2

    try:
        async with provider:
            articles = await Aggregator(
                [provider],
                raise_on_all_failed=True,
            ).aggregate(limit=max(args.limit, 0), **filters)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps([serialize(a) for a in articles], indent=2, ensure_ascii=False))
    else:
        for article in articles:
            print(_render_text(article))

    if not articles:
        print("warning: no articles returned", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the CLI.

    Args:
        argv: Optional argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
