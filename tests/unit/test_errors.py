"""Unit tests for the provider exception hierarchy in :mod:`Articles.errors`."""

import unittest

from Articles.errors import (
    ProviderAuthError,
    ProviderConfigError,
    ProviderError,
    ProviderHTTPError,
    ProviderTransportError,
)


class TestProviderError(unittest.TestCase):
    """Tests for the base ProviderError class."""

    def test_stores_message_without_context(self) -> None:
        """A bare ProviderError exposes its message and no context."""
        error = ProviderError("something failed")

        self.assertEqual(error.message, "something failed")
        self.assertIsNone(error.provider)
        self.assertIsNone(error.url)
        self.assertEqual(str(error), "something failed")
        self.assertEqual(error.args, ("something failed",))

    def test_str_includes_provider_and_url(self) -> None:
        """Diagnostic context is appended to the string representation."""
        error = ProviderError(
            "request failed",
            provider="NewsAPI",
            url="https://example.com/news",
        )

        rendered = str(error)
        self.assertIn("request failed", rendered)
        self.assertIn("NewsAPI", rendered)
        self.assertIn("https://example.com/news", rendered)
        self.assertEqual(error.provider, "NewsAPI")
        self.assertEqual(error.url, "https://example.com/news")

    def test_is_an_exception(self) -> None:
        """ProviderError can be raised and caught as a normal exception."""
        with self.assertRaises(ProviderError):
            raise ProviderError("boom")


class TestProviderSubclasses(unittest.TestCase):
    """Tests for the concrete provider failures."""

    def test_all_subclasses_derive_from_provider_error(self) -> None:
        """Every provider failure is catchable as ProviderError."""
        errors = (
            ProviderConfigError("missing key"),
            ProviderAuthError("unauthorized", status_code=401),
            ProviderHTTPError("server error", status_code=500),
            ProviderTransportError("connection reset"),
        )

        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.assertIsInstance(error, ProviderError)

    def test_config_error_keeps_context(self) -> None:
        """ProviderConfigError accepts optional provider and url context."""
        error = ProviderConfigError(
            "missing API key",
            provider="NewsAPI",
            url="https://example.com",
        )

        self.assertIn("missing API key", str(error))
        self.assertEqual(error.provider, "NewsAPI")
        self.assertEqual(error.url, "https://example.com")

    def test_auth_error_stores_status_code(self) -> None:
        """ProviderAuthError records the HTTP status code."""
        error = ProviderAuthError("invalid key", status_code=403, provider="PaperApi")

        self.assertEqual(error.status_code, 403)
        self.assertIn("status_code=403", str(error))
        self.assertIn("PaperApi", str(error))

    def test_http_error_stores_status_and_body(self) -> None:
        """ProviderHTTPError records the response status and body."""
        error = ProviderHTTPError(
            "unexpected response",
            status_code=503,
            body='{"error": "unavailable"}',
            url="https://example.com/api",
        )

        self.assertEqual(error.status_code, 503)
        self.assertEqual(error.body, '{"error": "unavailable"}')
        self.assertIn("status_code=503", str(error))
        self.assertIn("https://example.com/api", str(error))

    def test_http_error_body_defaults_to_none(self) -> None:
        """The HTTP response body is optional."""
        error = ProviderHTTPError("not found", status_code=404)

        self.assertIsNone(error.body)

    def test_transport_error_keeps_context(self) -> None:
        """ProviderTransportError accepts optional provider and url context."""
        error = ProviderTransportError(
            "timeout",
            provider="NewsAPI",
            url="https://example.com/slow",
        )

        self.assertEqual(error.provider, "NewsAPI")
        self.assertEqual(error.url, "https://example.com/slow")
        self.assertIn("timeout", str(error))


if __name__ == "__main__":
    unittest.main()
