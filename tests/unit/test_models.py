"""Unit tests for the normalized domain models in :mod:`Articles.Models`."""

import unittest
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from Articles.Models import Article, Paper

PUBLISHED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def make_article(**overrides: Any) -> Article:
    """Build a valid Article, overriding selected fields.

    Args:
        **overrides: Field values that replace the defaults.

    Returns:
        A valid Article instance.
    """
    values: dict[str, Any] = {
        "id": "article-1",
        "title": "A useful article",
        "url": "https://example.com/articles/1",
        "source": "Example",
        "published_at": PUBLISHED,
        "summary": "Short summary.",
    }
    values.update(overrides)
    return Article(**values)


def make_paper(**overrides: Any) -> Paper:
    """Build a valid Paper, overriding selected fields.

    Args:
        **overrides: Field values that replace the defaults.

    Returns:
        A valid Paper instance.
    """
    values: dict[str, Any] = {
        "id": "paper-1",
        "title": "A useful paper",
        "url": "https://example.com/papers/1",
        "source": "Example",
        "published_at": PUBLISHED,
        "summary": "Short abstract.",
        "authors": ("Ada Lovelace", "Alan Turing"),
        "topics": ("nlp",),
    }
    values.update(overrides)
    return Paper(**values)


class TestArticle(unittest.TestCase):
    """Tests for the Article dataclass."""

    def test_stores_values_and_defaults(self) -> None:
        """Article exposes supplied fields and empty collection defaults."""
        article = make_article()

        self.assertEqual(article.id, "article-1")
        self.assertEqual(article.title, "A useful article")
        self.assertEqual(article.url, "https://example.com/articles/1")
        self.assertEqual(article.source, "Example")
        self.assertEqual(article.published_at, PUBLISHED)
        self.assertEqual(article.summary, "Short summary.")
        self.assertEqual(article.topics, ())
        self.assertEqual(article.raw, {})

    def test_allows_none_published_at_and_summary(self) -> None:
        """Optional published_at and summary accept None."""
        article = make_article(published_at=None, summary=None)

        self.assertIsNone(article.published_at)
        self.assertIsNone(article.summary)

    def test_accepts_non_utc_aware_datetime(self) -> None:
        """Any timezone-aware datetime is accepted."""
        aware = datetime(2026, 1, 2, 5, 4, 5, tzinfo=timezone(timedelta(hours=2)))

        article = make_article(published_at=aware)

        self.assertEqual(article.published_at, aware)

    def test_rejects_empty_required_fields(self) -> None:
        """Blank id, title, and url raise ValueError."""
        for field_name in ("id", "title", "url"):
            for blank in ("", "   "):
                with (
                    self.subTest(field=field_name, value=repr(blank)),
                    self.assertRaisesRegex(ValueError, field_name),
                ):
                    make_article(**{field_name: blank})

    def test_rejects_naive_published_at(self) -> None:
        """Naive publication timestamps raise ValueError."""
        naive = datetime(2026, 1, 2, 3, 4, 5)

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            make_article(published_at=naive)

    def test_is_frozen(self) -> None:
        """Article instances are immutable."""
        article = make_article()

        with self.assertRaises(FrozenInstanceError):
            article.title = "changed"  # type: ignore[misc]

    def test_raw_is_excluded_from_equality_and_repr(self) -> None:
        """Raw payloads do not affect equality or the repr."""
        first = make_article(raw={"payload": 1})
        second = make_article(raw={"payload": 2})

        self.assertEqual(first, second)
        self.assertNotIn("payload", repr(first))

    def test_raw_default_is_not_shared(self) -> None:
        """Each instance receives its own default raw mapping."""
        first = make_article()
        second = make_article()

        self.assertEqual(first.raw, second.raw)
        self.assertIsNot(first.raw, second.raw)


class TestPaper(unittest.TestCase):
    """Tests for the Paper dataclass."""

    def test_stores_values_and_defaults(self) -> None:
        """Paper exposes supplied fields and collection defaults."""
        paper = make_paper()

        self.assertEqual(paper.id, "paper-1")
        self.assertEqual(paper.title, "A useful paper")
        self.assertEqual(paper.url, "https://example.com/papers/1")
        self.assertEqual(paper.source, "Example")
        self.assertEqual(paper.published_at, PUBLISHED)
        self.assertEqual(paper.summary, "Short abstract.")
        self.assertEqual(paper.authors, ("Ada Lovelace", "Alan Turing"))
        self.assertEqual(paper.topics, ("nlp",))
        self.assertEqual(paper.raw, {})

    def test_defaults_for_authors_topics_and_raw(self) -> None:
        """Paper provides empty tuple and mapping defaults."""
        paper = Paper(
            id="paper-2",
            title="Untitled",
            url="https://example.com/papers/2",
            source="Example",
            published_at=None,
            summary=None,
        )

        self.assertEqual(paper.authors, ())
        self.assertEqual(paper.topics, ())
        self.assertEqual(paper.raw, {})

    def test_rejects_empty_required_fields(self) -> None:
        """Blank id, title, and url raise ValueError."""
        for field_name in ("id", "title", "url"):
            with self.subTest(field=field_name), self.assertRaisesRegex(ValueError, field_name):
                make_paper(**{field_name: ""})

    def test_rejects_naive_published_at(self) -> None:
        """Naive publication timestamps raise ValueError."""
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            make_paper(published_at=datetime(2026, 1, 2, 3, 4, 5))

    def test_is_frozen(self) -> None:
        """Paper instances are immutable."""
        paper = make_paper()

        with self.assertRaises(FrozenInstanceError):
            paper.authors = ()  # type: ignore[misc]

    def test_raw_is_excluded_from_equality_and_repr(self) -> None:
        """Raw payloads do not affect equality or the repr."""
        first = make_paper(raw={"payload": 1})
        second = make_paper(raw={"payload": 2})

        self.assertEqual(first, second)
        self.assertNotIn("payload", repr(first))


if __name__ == "__main__":
    unittest.main()
