"""Unit tests for LLM request/response models and typed errors."""

from __future__ import annotations

import unittest

from Articles.llm.base import CompletionRequest, CompletionResponse, Message, Usage
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


class TestMessage(unittest.TestCase):
    """Message validation."""

    def test_valid_message(self) -> None:
        self.assertEqual(Message("user", "hello").content, "hello")

    def test_unknown_role_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Message("robot", "hello")  # type: ignore[arg-type]

    def test_non_string_content_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Message("user", 1)  # type: ignore[arg-type]


class TestCompletionRequest(unittest.TestCase):
    """Request validation and normalization."""

    def test_valid_request(self) -> None:
        request = CompletionRequest(messages=[Message("user", "hi")], model="m")  # type: ignore[arg-type]
        self.assertEqual(request.messages, (Message("user", "hi"),))

    def test_empty_messages_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompletionRequest(messages=(), model="m")

    def test_blank_model_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompletionRequest(messages=(Message("user", "hi"),), model=" ")

    def test_negative_temperature_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompletionRequest(messages=(Message("user", "hi"),), model="m", temperature=-1)

    def test_zero_max_tokens_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompletionRequest(messages=(Message("user", "hi"),), model="m", max_tokens=0)

    def test_unknown_format_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompletionRequest(
                messages=(Message("user", "hi"),),
                model="m",
                response_format="xml",  # type: ignore[arg-type]
            )

    def test_json_schema_requires_schema(self) -> None:
        with self.assertRaises(ValueError):
            CompletionRequest(
                messages=(Message("user", "hi"),),
                model="m",
                response_format="json_schema",
            )

    def test_json_schema_accepts_mapping(self) -> None:
        request = CompletionRequest(
            messages=(Message("user", "hi"),),
            model="m",
            response_format="json_schema",
            json_schema={"type": "object"},
        )
        self.assertEqual(request.json_schema, {"type": "object"})


class TestUsageAndResponse(unittest.TestCase):
    """Usage arithmetic and response validation."""

    def test_total_tokens(self) -> None:
        self.assertEqual(Usage(3, 4).total_tokens, 7)

    def test_negative_usage_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Usage(prompt_tokens=-1)

    def test_response_valid(self) -> None:
        response = CompletionResponse(
            text="hello",
            provider="p",
            model="m",
            usage=Usage(1, 2),
            raw={"id": "x"},
        )
        self.assertEqual(response.usage.total_tokens, 3)
        self.assertNotIn('{"id"', repr(response))

    def test_response_rejects_blank_provider(self) -> None:
        with self.assertRaises(ValueError):
            CompletionResponse(text="t", provider=" ", model="m")

    def test_response_rejects_blank_model(self) -> None:
        with self.assertRaises(ValueError):
            CompletionResponse(text="t", provider="p", model="")


class TestErrors(unittest.TestCase):
    """Error hierarchy and retry flags."""

    def test_base_error_context(self) -> None:
        error = LLMError("boom", provider="p", model="m")
        self.assertIn("provider='p'", str(error))
        self.assertIn("model='m'", str(error))
        self.assertFalse(error.retryable)

    def test_auth_is_not_retryable(self) -> None:
        error = LLMAuthError("bad key", status_code=401, provider="p", model="m")
        self.assertFalse(error.retryable)
        self.assertIn("status_code=401", str(error))

    def test_rate_limit_is_retryable(self) -> None:
        error = LLMRateLimitError("slow down", retry_after=2.0)
        self.assertTrue(error.retryable)
        self.assertEqual(error.retry_after, 2.0)

    def test_http_5xx_is_retryable(self) -> None:
        error = LLMHTTPError("bad", status_code=503, body="down")
        self.assertTrue(error.retryable)
        self.assertIn("status_code=503", str(error))

    def test_http_4xx_is_not_retryable(self) -> None:
        self.assertFalse(LLMHTTPError("bad", status_code=400).retryable)

    def test_transport_and_response_are_retryable(self) -> None:
        self.assertTrue(LLMTransportError("timeout").retryable)
        self.assertTrue(LLMResponseError("bad json").retryable)

    def test_config_and_unavailable_are_not_retryable(self) -> None:
        self.assertFalse(LLMConfigError("bad config").retryable)
        self.assertFalse(LLMUnavailableError("down").retryable)

    def test_errors_do_not_include_keys(self) -> None:
        error = LLMAuthError("bad key", status_code=401)
        self.assertNotIn("Bearer", str(error))


if __name__ == "__main__":
    unittest.main()
