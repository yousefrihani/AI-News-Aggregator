"""Tests for the frozen LLM value objects and typed error hierarchy."""

import unittest
from dataclasses import FrozenInstanceError

from Articles.llm.base import (
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    Message,
    Usage,
)
from Articles.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTransportError,
    LLMUnavailableError,
)

from tests.llm import FakeProvider, make_response


class TestValueObjects(unittest.TestCase):
    """Immutability, defaults, and equality semantics."""

    def test_message_is_frozen_and_equatable(self) -> None:
        message = Message(role="user", content="hello")
        self.assertEqual(message, Message(role="user", content="hello"))
        self.assertNotEqual(message, Message(role="system", content="hello"))
        with self.assertRaises(FrozenInstanceError):
            message.content = "changed"  # type: ignore[misc]

    def test_completion_request_defaults_and_metadata_independence(self) -> None:
        messages = (Message(role="user", content="hi"),)
        first = CompletionRequest(messages=messages, model="m")
        second = CompletionRequest(messages=messages, model="m")
        self.assertEqual(first.temperature, 0.2)
        self.assertIsNone(first.max_tokens)
        self.assertEqual(first.response_format, "text")
        self.assertIsNone(first.json_schema)
        self.assertEqual(first.metadata, {})
        self.assertIsNot(first.metadata, second.metadata)

    def test_usage_total_tokens(self) -> None:
        usage = Usage(prompt_tokens=7, completion_tokens=5)
        self.assertEqual(usage.total_tokens, 12)
        self.assertEqual(Usage().total_tokens, 0)

    def test_response_raw_is_excluded_from_equality_and_repr(self) -> None:
        first = make_response(text="a", raw={"fake": True})
        second = CompletionResponse(
            text="a",
            provider=first.provider,
            model=first.model,
            usage=first.usage,
            finish_reason=first.finish_reason,
            raw={"different": True},
        )
        self.assertEqual(first, second)
        self.assertNotIn("fake", repr(first))
        self.assertEqual(first.raw, {"fake": True})

    def test_provider_protocol_matches_structural_fakes(self) -> None:
        self.assertIsInstance(FakeProvider(), LLMProvider)
        self.assertNotIsInstance(object(), LLMProvider)


class TestErrorHierarchy(unittest.TestCase):
    """Typed errors expose retryability and safe diagnostics."""

    def test_base_error_context_and_defaults(self) -> None:
        error = LLMError("boom")
        self.assertEqual(str(error), "boom")
        self.assertFalse(error.retryable)
        contextual = LLMError("boom", provider="groq", model="fast")
        self.assertEqual(str(contextual), "boom [provider='groq', model='fast']")

    def test_config_error_is_never_retryable(self) -> None:
        error = LLMConfigError("bad config", provider="p")
        self.assertFalse(error.retryable)
        self.assertIn("bad config", str(error))
        self.assertIn("provider='p'", str(error))

    def test_auth_error_reports_status_without_secrets(self) -> None:
        error = LLMAuthError(
            "Provider rejected credentials", status_code=401, provider="groq", model="fast"
        )
        self.assertFalse(error.retryable)
        self.assertEqual(error.status_code, 401)
        self.assertIn("401", str(error))
        self.assertNotIn("Bearer", str(error))

    def test_rate_limit_error_is_retryable(self) -> None:
        error = LLMRateLimitError("Rate limited the request", retry_after=2.5, provider="groq")
        self.assertTrue(error.retryable)
        self.assertEqual(error.retry_after, 2.5)
        self.assertIn("Rate limited", str(error))

    def test_http_error_retryability_depends_on_status(self) -> None:
        server = LLMHTTPError("server error", status_code=500, body="boom", provider="p")
        client = LLMHTTPError("client error", status_code=400, body="bad", provider="p")
        self.assertTrue(server.retryable)
        self.assertFalse(client.retryable)
        self.assertEqual(server.body, "boom")
        self.assertIn("status_code=500", str(server))
        self.assertNotIn("boom", str(server))

    def test_transport_and_response_errors_are_retryable(self) -> None:
        self.assertTrue(LLMTransportError("network").retryable)
        self.assertTrue(LLMResponseError("malformed").retryable)

    def test_unavailable_error_is_not_retryable(self) -> None:
        error = LLMUnavailableError("nothing available", provider="p")
        self.assertFalse(error.retryable)
        self.assertIn("nothing available", str(error))


if __name__ == "__main__":
    unittest.main()
