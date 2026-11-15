"""Router tests: resolution, retries, fallback, breakers, and budgets.

Every test uses scripted in-process providers. No socket is opened, no API key
is real, and no test sleeps for real: the router accepts an injected clock and
sleep function.
"""

from __future__ import annotations

import unittest
from typing import Any

from Articles.llm.base import Message
from Articles.llm.config import RouteConfig
from Articles.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMTransportError,
    LLMUnavailableError,
)
from Articles.llm.router import LLMRouter, estimate_tokens

from tests.llm import (
    FakeProvider,
    make_config,
    make_response,
    make_router,
)

MESSAGES = (Message("user", "hello"),)
KEYS = {"P1_KEY": "k1", "P2_KEY": "k2"}


def ok(provider: str) -> Any:
    """Return a scripted success for one provider."""
    return make_response(provider, f"{provider}-model", text=f"{provider}-answer")


def scripted(**responses: list[Any]) -> dict[str, FakeProvider]:
    """Build providers from keyword scripts, for example ``p1=[error]``."""
    return {name: FakeProvider(name, items) for name, items in responses.items()}


class TestResolution(unittest.IsolatedAsyncioTestCase):
    """Route, provider, and model selection."""

    async def test_default_route_uses_first_candidate(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1")], p2=[ok("p2")]))
        response = await router.complete("", MESSAGES)
        self.assertEqual(response.provider, "p1")
        self.assertEqual(len(providers["p1"].calls), 1)
        self.assertEqual(providers["p2"].calls, [])

    async def test_unknown_route_is_rejected(self) -> None:
        router, _, _, _ = make_router()
        with self.assertRaises(LLMConfigError):
            await router.complete("missing", MESSAGES)

    async def test_empty_messages_are_rejected(self) -> None:
        router, _, _, _ = make_router()
        with self.assertRaises(LLMConfigError):
            await router.complete("summarize", ())

    async def test_provider_pin_wins_over_route_order(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1")], p2=[ok("p2")]))
        response = await router.complete("summarize", MESSAGES, provider="p2")
        self.assertEqual(response.provider, "p2")
        self.assertEqual(providers["p1"].calls, [])

    async def test_model_pin_resolves_by_concrete_id(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1")], p2=[ok("p2")]))
        response = await router.complete("summarize", MESSAGES, model="p2-model")
        self.assertEqual(response.provider, "p2")
        self.assertEqual(providers["p1"].calls, [])

    async def test_unknown_provider_pin_is_rejected(self) -> None:
        router, _, _, _ = make_router()
        with self.assertRaises(LLMConfigError):
            await router.complete("summarize", MESSAGES, provider="missing")

    async def test_unknown_model_pin_is_rejected(self) -> None:
        router, _, _, _ = make_router()
        with self.assertRaises(LLMConfigError):
            await router.complete("summarize", MESSAGES, model="missing")

    async def test_environment_provider_override(self) -> None:
        router, _, _, _ = make_router(
            providers=scripted(p1=[ok("p1")], p2=[ok("p2")]),
            env={**KEYS, "AI_NEWS_LLM_PROVIDER": "p2"},
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p2")

    async def test_environment_model_override(self) -> None:
        router, _, _, _ = make_router(
            providers=scripted(p1=[ok("p1")], p2=[ok("p2")]),
            env={**KEYS, "AI_NEWS_LLM_MODEL": "p1-model"},
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p1")


class TestPinnedFailure(unittest.IsolatedAsyncioTestCase):
    """A pin fails loudly unless fallback is explicitly enabled."""

    async def test_pinned_failure_does_not_fall_back_by_default(self) -> None:
        router, providers, _, _ = make_router(
            providers=scripted(p1=[LLMTransportError("down")], p2=[ok("p2")])
        )
        with self.assertRaises(LLMUnavailableError):
            await router.complete("summarize", MESSAGES, provider="p1")
        self.assertEqual(len(providers["p1"].calls), 1)
        self.assertEqual(providers["p2"].calls, [])

    async def test_pinned_failure_can_opt_into_fallback(self) -> None:
        router, providers, _, _ = make_router(
            providers=scripted(p1=[LLMTransportError("down")], p2=[ok("p2")])
        )
        response = await router.complete("summarize", MESSAGES, provider="p1", allow_fallback=True)
        self.assertEqual(response.provider, "p2")
        self.assertEqual(len(providers["p1"].calls), 1)
        self.assertEqual(len(providers["p2"].calls), 1)

    async def test_environment_enables_pinned_fallback(self) -> None:
        router, _, _, _ = make_router(
            providers=scripted(p1=[LLMTransportError("down")], p2=[ok("p2")]),
            env={**KEYS, "AI_NEWS_LLM_ALLOW_FALLBACK": "yes"},
        )
        response = await router.complete("summarize", MESSAGES, provider="p1")
        self.assertEqual(response.provider, "p2")


class TestRetriesAndFallback(unittest.IsolatedAsyncioTestCase):
    """Retryable failures retry, non-retryable failures move on."""

    async def test_retryable_error_falls_back_to_next_candidate(self) -> None:
        router, providers, _, _ = make_router(
            providers=scripted(p1=[LLMTransportError("down")], p2=[ok("p2")])
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p2")
        self.assertEqual(len(providers["p1"].calls), 1)
        self.assertEqual(len(providers["p2"].calls), 1)

    async def test_retries_same_candidate_before_fallback(self) -> None:
        router, providers, sleeps, recorded = make_router(
            config=make_config(retries=2),
            providers=scripted(
                p1=[LLMTransportError("one"), LLMTransportError("two"), ok("p1")],
                p2=[ok("p2")],
            ),
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p1")
        self.assertEqual(len(providers["p1"].calls), 3)
        self.assertEqual(providers["p2"].calls, [])
        self.assertEqual(len(recorded), 2)

    async def test_exhausted_retries_fall_back(self) -> None:
        router, providers, _, recorded = make_router(
            config=make_config(retries=2),
            providers=scripted(
                p1=[LLMTransportError("one"), LLMTransportError("two"), LLMTransportError("three")],
                p2=[ok("p2")],
            ),
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p2")
        self.assertEqual(len(providers["p1"].calls), 3)
        self.assertEqual(len(recorded), 2)

    async def test_rate_limit_retry_after_is_honored(self) -> None:
        router, providers, _, recorded = make_router(
            config=make_config(retries=1),
            providers=scripted(
                p1=[LLMRateLimitError("slow down", retry_after=3.5), ok("p1")],
                p2=[ok("p2")],
            ),
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p1")
        self.assertEqual(recorded, [3.5])

    async def test_non_retryable_error_skips_retries(self) -> None:
        router, providers, _, recorded = make_router(
            config=make_config(retries=2),
            providers=scripted(
                p1=[LLMHTTPError("bad request", status_code=400), ok("p1")],
                p2=[ok("p2")],
            ),
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p2")
        self.assertEqual(len(providers["p1"].calls), 1)
        self.assertEqual(recorded, [])

    async def test_every_candidate_failing_raises_last_error(self) -> None:
        router, _, _, _ = make_router(
            providers=scripted(
                p1=[LLMTransportError("down")],
                p2=[LLMHTTPError("server down", status_code=503)],
            )
        )
        with self.assertRaises(LLMHTTPError):
            await router.complete("summarize", MESSAGES)


class TestAvailability(unittest.IsolatedAsyncioTestCase):
    """Credentials and circuit breakers decide candidate availability."""

    async def test_missing_credentials_skip_provider(self) -> None:
        router, providers, _, _ = make_router(
            providers=scripted(p1=[ok("p1")], p2=[ok("p2")]),
            env={"P2_KEY": "k2"},
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p2")
        self.assertEqual(providers["p1"].calls, [])

    async def test_no_credentials_raises_unavailable(self) -> None:
        router, _, _, _ = make_router(env={})
        with self.assertRaises(LLMUnavailableError):
            await router.complete("summarize", MESSAGES)

    async def test_breaker_opens_after_threshold(self) -> None:
        router, providers, _, _ = make_router(
            providers=scripted(
                p1=[LLMTransportError("one"), LLMTransportError("two")],
                p2=[ok("p2"), ok("p2"), ok("p2")],
            ),
            failure_threshold=2,
        )
        await router.complete("summarize", MESSAGES, use_cache=False)
        await router.complete("summarize", MESSAGES, use_cache=False)
        response = await router.complete("summarize", MESSAGES, use_cache=False)
        self.assertEqual(response.provider, "p2")
        self.assertEqual(len(providers["p1"].calls), 2)
        self.assertEqual(len(providers["p2"].calls), 3)

    async def test_breaker_half_opens_after_cooldown(self) -> None:
        router, providers, clock, _ = make_router(
            providers=scripted(
                p1=[LLMTransportError("one"), LLMTransportError("two"), ok("p1")],
                p2=[ok("p2"), ok("p2")],
            ),
            failure_threshold=2,
            cooldown_s=10.0,
        )
        await router.complete("summarize", MESSAGES, use_cache=False)
        await router.complete("summarize", MESSAGES, use_cache=False)
        clock.advance(10.0)
        response = await router.complete("summarize", MESSAGES, use_cache=False)
        self.assertEqual(response.provider, "p1")
        self.assertEqual(len(providers["p1"].calls), 3)

    async def test_auth_failure_counts_toward_breaker(self) -> None:
        router, providers, _, _ = make_router(
            providers=scripted(
                p1=[LLMAuthError("denied", status_code=401)],
                p2=[ok("p2")],
            ),
            failure_threshold=1,
        )
        await router.complete("summarize", MESSAGES)
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "p2")
        self.assertEqual(len(providers["p1"].calls), 1)


class TestCacheAndAccounting(unittest.IsolatedAsyncioTestCase):
    """Caching, token budgets, and usage accounting."""

    async def test_cache_avoids_a_second_provider_call(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1")]))
        first = await router.complete("summarize", MESSAGES)
        second = await router.complete("summarize", MESSAGES)
        self.assertEqual(first, second)
        self.assertEqual(len(providers["p1"].calls), 1)

    async def test_cache_can_be_bypassed(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1"), ok("p1")]))
        await router.complete("summarize", MESSAGES)
        await router.complete("summarize", MESSAGES, use_cache=False)
        self.assertEqual(len(providers["p1"].calls), 2)

    async def test_clear_cache_forces_a_new_call(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1"), ok("p1")]))
        await router.complete("summarize", MESSAGES)
        router.clear_cache()
        await router.complete("summarize", MESSAGES)
        self.assertEqual(len(providers["p1"].calls), 2)

    async def test_daily_budget_stops_further_calls(self) -> None:
        router, providers, _, _ = make_router(
            config=make_config(daily_token_budget=3),
            providers=scripted(p1=[ok("p1")]),
        )
        await router.complete("summarize", MESSAGES)
        with self.assertRaises(LLMConfigError):
            await router.complete("summarize", MESSAGES)
        self.assertEqual(len(providers["p1"].calls), 1)

    async def test_usage_is_accumulated_by_route(self) -> None:
        router, _, _, _ = make_router(providers=scripted(p1=[ok("p1")]))
        await router.complete("summarize", MESSAGES)
        self.assertEqual(router.usage_by_route["summarize"].total_tokens, 3)
        self.assertEqual(router.total_usage.total_tokens, 3)

    async def test_cached_usage_is_counted_again(self) -> None:
        router, _, _, _ = make_router(providers=scripted(p1=[ok("p1")]))
        await router.complete("summarize", MESSAGES)
        await router.complete("summarize", MESSAGES)
        self.assertEqual(router.total_usage.total_tokens, 6)


class TestRequestShaping(unittest.IsolatedAsyncioTestCase):
    """Overrides, schema handling, and input fitting."""

    async def test_overrides_are_forwarded(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1")]))
        await router.complete(
            "summarize",
            MESSAGES,
            temperature=0.9,
            max_tokens=42,
            metadata={"trace": "abc"},
        )
        request = providers["p1"].calls[0]
        self.assertEqual(request.temperature, 0.9)
        self.assertEqual(request.max_tokens, 42)
        self.assertEqual(request.metadata, {"trace": "abc"})

    async def test_unknown_override_is_rejected(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1")]))
        with self.assertRaises(LLMConfigError):
            await router.complete("summarize", MESSAGES, top_p=0.5)
        self.assertEqual(providers["p1"].calls, [])

    async def test_invalid_override_values_are_rejected(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1")]))
        invalid = (
            {"temperature": -1},
            {"max_tokens": 0},
            {"response_format": "xml"},
            {"json_schema": "not-a-mapping"},
            {"metadata": "not-a-mapping"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(LLMConfigError):
                await router.complete("summarize", MESSAGES, **overrides)
        self.assertEqual(providers["p1"].calls, [])

    async def test_json_schema_route_requires_an_explicit_schema(self) -> None:
        router, _, _, _ = make_router(
            config=make_config(response_format="json_schema", schema_ref="graph.v1"),
        )
        with self.assertRaises(LLMConfigError):
            await router.complete("summarize", MESSAGES)

    async def test_json_schema_override_is_forwarded(self) -> None:
        router, providers, _, _ = make_router(
            config=make_config(response_format="json_schema", schema_ref="graph.v1"),
            providers=scripted(p1=[ok("p1")]),
        )
        schema = {"type": "object"}
        await router.complete("summarize", MESSAGES, json_schema=schema)
        request = providers["p1"].calls[0]
        self.assertEqual(request.response_format, "json_schema")
        self.assertEqual(request.json_schema, schema)

    async def test_large_input_is_clipped(self) -> None:
        router, providers, _, _ = make_router(
            config=make_config(max_input_tokens=20),
            providers=scripted(p1=[ok("p1")]),
        )
        messages = (
            Message("system", "be brief"),
            Message("user", "x" * 500),
        )
        await router.complete("summarize", messages)
        fitted = providers["p1"].calls[0].messages
        self.assertLess(len(fitted[-1].content), 500)
        self.assertIn("[...truncated...]", fitted[-1].content)


class TestDryRunAndLifecycle(unittest.IsolatedAsyncioTestCase):
    """Dry run never touches a provider; close releases every adapter."""

    async def test_dry_run_returns_fixture_without_calls(self) -> None:
        router, providers, _, _ = make_router(
            providers=scripted(p1=[ok("p1")]),
            dry_run=True,
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "dry-run")
        self.assertEqual(response.text, "dry-run:summarize")
        self.assertEqual(providers["p1"].calls, [])

    async def test_dry_run_json_route_returns_empty_object(self) -> None:
        router, _, _, _ = make_router(
            config=make_config(response_format="json_object"),
            dry_run=True,
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.text, "{}")

    async def test_dry_run_environment_flag(self) -> None:
        router, providers, _, _ = make_router(
            providers=scripted(p1=[ok("p1")]),
            env={**KEYS, "LLM_DRY_RUN": "1"},
        )
        response = await router.complete("summarize", MESSAGES)
        self.assertEqual(response.provider, "dry-run")
        self.assertEqual(providers["p1"].calls, [])

    async def test_close_closes_created_providers(self) -> None:
        router, providers, _, _ = make_router(providers=scripted(p1=[ok("p1")]))
        await router.complete("summarize", MESSAGES)
        await router.aclose()
        self.assertTrue(providers["p1"].closed)
        await router.aclose()

    def test_estimate_tokens_is_conservative(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcde"), 2)


class TestConstructorValidation(unittest.TestCase):
    """Invalid router bounds fail at construction."""

    def test_config_type_is_validated(self) -> None:
        with self.assertRaises(LLMConfigError):
            LLMRouter("not-a-config")  # type: ignore[arg-type]

    def test_failure_threshold_is_validated(self) -> None:
        with self.assertRaises(LLMConfigError):
            LLMRouter(make_config(), failure_threshold=0)

    def test_cooldown_is_validated(self) -> None:
        with self.assertRaises(LLMConfigError):
            LLMRouter(make_config(), cooldown_s=-1.0)

    def test_extra_route_can_be_selected(self) -> None:
        config = make_config()
        routes = dict(config.routes)
        routes["classify"] = RouteConfig(
            name="classify",
            candidates=("p2/default",),
        )
        extended = type(config)(
            providers=config.providers,
            routes=routes,
        )
        self.assertIn("classify", extended.routes)


if __name__ == "__main__":
    unittest.main()
