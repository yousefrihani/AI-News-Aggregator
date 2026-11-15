"""Shared fakes for LLM-layer tests.

Nothing here opens a socket. Fake providers return scripted responses or raise
scripted errors, and the fake clock removes all timing from breaker tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from Articles.llm import (
    CompletionRequest,
    CompletionResponse,
    LLMConfig,
    LLMRouter,
    ModelConfig,
    ProviderConfig,
    ProviderRegistry,
    RouteConfig,
    Usage,
)


class FakeClock:
    """Monotonic clock controlled by the test."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the clock.

        Args:
            seconds: Seconds to add.
        """
        self.now += seconds


class FakeProvider:
    """Scripted provider used by router tests."""

    def __init__(self, name: str = "fake", responses: Sequence[Any] = ()) -> None:
        self.name = name
        self.responses = list(responses)
        self.calls: list[CompletionRequest] = []
        self.closed = False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        if not self.responses:
            raise AssertionError(f"no scripted response left for {self.name}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            result: CompletionResponse = item(request)
            return result
        return item

    async def aclose(self) -> None:
        self.closed = True


def make_response(
    provider: str = "p1",
    model: str = "p1-model",
    text: str = "ok",
    *,
    prompt_tokens: int = 1,
    completion_tokens: int = 2,
    raw: dict[str, Any] | None = None,
) -> CompletionResponse:
    """Build a representative completion response.

    Args:
        provider: Provider name recorded on the response.
        model: Model id recorded on the response.
        text: Assistant text.
        prompt_tokens: Reported prompt tokens.
        completion_tokens: Reported completion tokens.
        raw: Optional raw provider payload.

    Returns:
        A normalized completion response.
    """
    return CompletionResponse(
        text=text,
        provider=provider,
        model=model,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        finish_reason="stop",
        raw=raw or {},
    )


def make_model(
    model_id: str = "p-model",
    *,
    context_window: int = 128_000,
    supports: frozenset[str] | None = None,
) -> ModelConfig:
    """Build a model configuration for registry tests.

    Args:
        model_id: Concrete provider model id.
        context_window: Maximum context size.
        supports: Optional protocol features.

    Returns:
        A validated model configuration.
    """
    return ModelConfig(
        id=model_id,
        context_window=context_window,
        supports=supports if supports is not None else frozenset(),
    )


def make_provider(
    name: str = "p",
    *,
    kind: str = "fake",
    base_url: str | None = None,
    api_key_env: str | None = None,
    models: dict[str, ModelConfig] | None = None,
    headers: dict[str, str] | None = None,
) -> ProviderConfig:
    """Build a provider configuration for registry tests.

    Args:
        name: Provider name.
        kind: Adapter kind.
        base_url: Explicit base URL. ``None`` selects a test default.
        api_key_env: Optional API key environment variable.
        models: Optional model map. Defaults to one ``default`` model.
        headers: Optional extra headers.

    Returns:
        A validated provider configuration.
    """
    return ProviderConfig(
        name=name,
        kind=kind,
        base_url=f"https://{name}.test" if base_url is None else base_url,
        api_key_env=api_key_env,
        headers=headers or {},
        models=models or {"default": make_model(f"{name}-model")},
    )


def make_config(
    *,
    candidates: tuple[str, ...] = ("p1/default", "p2/default"),
    retries: int = 0,
    timeout_s: float = 1.0,
    max_input_tokens: int | None = None,
    daily_token_budget: int | None = None,
    response_format: str = "text",
    schema_ref: str | None = None,
) -> LLMConfig:
    """Build a two-provider configuration for router tests."""
    supports = frozenset({"json_object", "json_schema"})
    providers = {
        name: ProviderConfig(
            name=name,
            kind="fake",
            base_url=f"https://{name}.test",
            api_key_env=f"{name.upper()}_KEY",
            models={"default": ModelConfig(id=f"{name}-model", supports=supports)},
        )
        for name in ("p1", "p2")
    }
    route = RouteConfig(
        name="summarize",
        candidates=candidates,
        retries=retries,
        timeout_s=timeout_s,
        max_input_tokens=max_input_tokens,
        daily_token_budget=daily_token_budget,
        response_format=response_format,  # type: ignore[arg-type]
        schema_ref=schema_ref,
    )
    return LLMConfig(providers=providers, routes={"summarize": route})


def make_router(
    config: LLMConfig | None = None,
    providers: dict[str, FakeProvider] | None = None,
    *,
    env: dict[str, str] | None = None,
    sleeps: list[float] | None = None,
    **kwargs: Any,
) -> tuple[LLMRouter, dict[str, FakeProvider], FakeClock, list[float]]:
    """Build a router backed by fake providers.

    Args:
        config: Optional configuration.
        providers: Optional scripted providers keyed by name.
        env: Optional environment mapping.
        sleeps: Optional list that records requested delays.
        **kwargs: Extra router constructor arguments.

    Returns:
        The router, providers, fake clock, and recorded sleeps.
    """
    active_config = config or make_config()
    active_providers = providers or {
        "p1": FakeProvider("p1", [make_response("p1", "p1-model")]),
        "p2": FakeProvider("p2", [make_response("p2", "p2-model")]),
    }
    registry = ProviderRegistry()
    for name, provider in active_providers.items():
        registry.register(name, lambda _config, _key, provider=provider: provider)
    clock = FakeClock()
    recorded_sleeps: list[float] = sleeps if sleeps is not None else []

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    router = LLMRouter(
        active_config,
        registry,
        env=env if env is not None else {"P1_KEY": "k1", "P2_KEY": "k2"},
        cache={},
        clock=clock,
        sleep=fake_sleep,
        **kwargs,
    )
    return router, active_providers, clock, recorded_sleeps
