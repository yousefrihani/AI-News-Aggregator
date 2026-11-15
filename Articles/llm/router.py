"""Health-aware LLM routing with retries, fallback, and user overrides.

Patterns and agents name a **route**, never a vendor. The router resolves the
route to an ordered candidate chain, skips providers that are unconfigured or
circuit-broken, retries retryable failures, and falls back to the next
candidate. An explicit provider/model pin wins over route order and fails
loudly unless fallback is allowed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from Articles.llm.base import (
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    Message,
    Usage,
)
from Articles.llm.config import LLMConfig, ProviderConfig, RouteConfig
from Articles.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMTransportError,
    LLMUnavailableError,
)
from Articles.llm.registry import ProviderRegistry

__all__ = ["LLMRouter", "estimate_tokens"]

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_TRUNCATION_MARKER = "\n[...truncated...]\n"
_DEFAULT_ROUTE = "summarize"
_VALID_FORMATS: frozenset[str] = frozenset({"text", "json_object", "json_schema"})
_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"temperature", "max_tokens", "response_format", "json_schema", "metadata"}
)


def estimate_tokens(text: str) -> int:
    """Estimate tokens with a conservative character heuristic.

    Args:
        text: Input text.

    Returns:
        Estimated token count using four characters per token.
    """
    return max(0, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


@dataclass(slots=True)
class _Breaker:
    """Per-provider circuit-breaker state."""

    failures: int = 0
    opened_at: float | None = None


@dataclass(slots=True)
class _Candidate:
    """A resolved route candidate."""

    raw: str
    provider: str
    alias: str
    model_id: str


class LLMRouter:
    """Route completions across configured providers.

    Args:
        config: Validated provider and route configuration.
        registry: Provider factory registry. Defaults to the built-ins plus
            installed plugins.
        env: Optional environment mapping for tests. Defaults to ``os.environ``.
        dry_run: When true, return fixture responses and never call a provider.
            Defaults to the ``LLM_DRY_RUN`` environment flag.
        cache: Optional completion cache. A new in-memory dict is used by
            default.
        failure_threshold: Consecutive failures before a provider breaker opens.
        cooldown_s: Seconds an open breaker stays open before a half-open probe.
        clock: Monotonic clock, injectable for tests.
        sleep: Async sleep function, injectable for tests.

    Raises:
        LLMConfigError: If a constructor bound is invalid.
    """

    def __init__(
        self,
        config: LLMConfig,
        registry: ProviderRegistry | None = None,
        *,
        env: Mapping[str, str] | None = None,
        dry_run: bool | None = None,
        cache: MutableMapping[str, CompletionResponse] | None = None,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if not isinstance(config, LLMConfig):
            raise LLMConfigError("config must be an LLMConfig")
        if failure_threshold < 1:
            raise LLMConfigError("failure_threshold must be at least 1")
        if cooldown_s < 0:
            raise LLMConfigError("cooldown_s must be non-negative")
        self._config = config
        self._registry = registry or ProviderRegistry.with_defaults()
        self._env = env
        self._dry_run = (
            dry_run if dry_run is not None else self._env_flag("LLM_DRY_RUN", default=False)
        )
        self._cache: MutableMapping[str, CompletionResponse] = cache if cache is not None else {}
        self._failure_threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._sleep = sleep or self._default_sleep
        self._breakers: dict[str, _Breaker] = {}
        self._providers: dict[str, LLMProvider] = {}
        self._usage: dict[str, Usage] = {}

    # ------------------------------------------------------------------
    # Environment and provider lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    async def _default_sleep(delay: float) -> None:
        """Sleep for ``delay`` seconds.

        Args:
            delay: Seconds to sleep.
        """
        await asyncio.sleep(delay)

    def _env_value(self, name: str) -> str:
        """Read a stripped environment value.

        Args:
            name: Environment variable name.

        Returns:
            The stripped value, or an empty string.
        """
        if self._env is not None:
            return str(self._env.get(name, "")).strip()
        return os.getenv(name, "").strip()

    def _env_flag(self, name: str, *, default: bool) -> bool:
        """Read a boolean environment flag.

        Args:
            name: Environment variable name.
            default: Value when the variable is unset or blank.

        Returns:
            ``True`` for ``1``, ``true``, ``yes``, or ``on``.
        """
        raw = self._env_value(name)
        if not raw:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    def _api_key(self, provider: ProviderConfig) -> str | None:
        """Resolve a provider API key from its configured environment name.

        Args:
            provider: Provider configuration.

        Returns:
            The key, or ``None`` when no key is configured or required.
        """
        if provider.api_key_env is None:
            return None
        return self._env_value(provider.api_key_env) or None

    def _has_credentials(self, provider: ProviderConfig) -> bool:
        """Return whether a provider has the credentials it requires.

        Args:
            provider: Provider configuration.

        Returns:
            ``True`` for keyless endpoints or when the named env var is set.
        """
        return provider.api_key_env is None or bool(self._api_key(provider))

    def _get_provider(self, name: str) -> LLMProvider:
        """Return or lazily create a provider adapter.

        Args:
            name: Provider name.

        Returns:
            A provider instance.

        Raises:
            LLMConfigError: If the provider is unknown or cannot be created.
        """
        existing = self._providers.get(name)
        if existing is not None:
            return existing
        config = self._config.providers[name]
        provider = self._registry.create(config, self._api_key(config))
        self._providers[name] = provider
        return provider

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _provider_available(self, name: str, now: float) -> bool:
        """Return whether a provider may be attempted.

        Args:
            name: Provider name.
            now: Current monotonic time.

        Returns:
            ``False`` while the breaker is open, ``True`` otherwise or after the
            cooldown has elapsed (half-open).
        """
        breaker = self._breakers.get(name)
        if breaker is None or breaker.opened_at is None:
            return True
        return now - breaker.opened_at >= self._cooldown_s

    def _record_success(self, name: str) -> None:
        """Close a provider breaker after a success.

        Args:
            name: Provider name.
        """
        self._breakers.pop(name, None)

    def _record_failure(self, name: str, now: float) -> None:
        """Count a failure and open the breaker at the threshold.

        Args:
            name: Provider name.
            now: Current monotonic time.
        """
        breaker = self._breakers.setdefault(name, _Breaker())
        if breaker.opened_at is not None:
            breaker.opened_at = now
            return
        breaker.failures += 1
        if breaker.failures >= self._failure_threshold:
            breaker.opened_at = now

    @staticmethod
    def _counts_toward_breaker(error: LLMError) -> bool:
        """Return whether an error indicates provider health problems.

        Args:
            error: Failure raised by a candidate.

        Returns:
            ``True`` for retryable errors and authentication failures.
        """
        return error.retryable or isinstance(error, LLMAuthError)

    # ------------------------------------------------------------------
    # Route resolution
    # ------------------------------------------------------------------

    def _default_route(self) -> str:
        """Return the default route name.

        Returns:
            ``AI_NEWS_LLM_ROUTE`` when set, otherwise ``summarize``.
        """
        return self._env_value("AI_NEWS_LLM_ROUTE") or _DEFAULT_ROUTE

    def _allow_fallback(self, override: bool | None, *, pinned: bool) -> bool:
        """Resolve the fallback policy.

        Args:
            override: Explicit caller value, when provided.
            pinned: Whether a provider or model was pinned.

        Returns:
            ``True`` when another candidate may be tried after a pinned failure.
        """
        if override is not None:
            return override
        raw = self._env_value("AI_NEWS_LLM_ALLOW_FALLBACK")
        if raw:
            return raw.lower() in {"1", "true", "yes", "on"}
        return not pinned

    def _candidate_parts(self, candidate: str) -> _Candidate:
        """Parse and validate a ``provider/model-alias`` candidate.

        Args:
            candidate: Raw candidate string.

        Returns:
            A resolved candidate.

        Raises:
            LLMConfigError: If the candidate is malformed or unknown.
        """
        provider_name, separator, alias = candidate.partition("/")
        if not separator or not provider_name or not alias:
            raise LLMConfigError(f"invalid route candidate: {candidate!r}")
        provider = self._config.providers.get(provider_name)
        if provider is None:
            raise LLMConfigError(f"unknown provider: {provider_name!r}")
        model = provider.models.get(alias)
        if model is None:
            raise LLMConfigError(f"unknown model alias {alias!r} on provider {provider_name!r}")
        return _Candidate(
            raw=candidate,
            provider=provider_name,
            alias=alias,
            model_id=model.id,
        )

    @staticmethod
    def _resolve_model_alias(provider: ProviderConfig, model: str) -> str:
        """Resolve an alias or concrete model id to an alias.

        Args:
            provider: Provider configuration.
            model: Alias or concrete model id.

        Returns:
            The model alias.

        Raises:
            LLMConfigError: If the model is not exposed by the provider.
        """
        if model in provider.models:
            return model
        for alias, config in provider.models.items():
            if config.id == model:
                return alias
        raise LLMConfigError(f"provider {provider.name!r} does not expose model {model!r}")

    def _pin_candidate(
        self, route: RouteConfig, provider_name: str, model: str | None
    ) -> _Candidate:
        """Build the candidate requested by an explicit provider/model pin.

        Args:
            route: Route being executed.
            provider_name: Pinned provider.
            model: Optional pinned alias or model id.

        Returns:
            A resolved candidate.

        Raises:
            LLMConfigError: If the provider is unknown or does not expose the
                requested model.
        """
        provider = self._config.providers.get(provider_name)
        if provider is None:
            raise LLMConfigError(f"unknown provider: {provider_name!r}")
        if model is not None:
            alias = self._resolve_model_alias(provider, model)
        else:
            alias = next(
                (
                    candidate.partition("/")[2]
                    for candidate in route.candidates
                    if candidate.partition("/")[0] == provider_name
                ),
                next(iter(provider.models)),
            )
        return _Candidate(
            raw=f"{provider_name}/{alias}",
            provider=provider_name,
            alias=alias,
            model_id=provider.models[alias].id,
        )

    def _candidate_for_model(self, route: RouteConfig, model: str) -> _Candidate:
        """Find a candidate exposing a model alias or id.

        Args:
            route: Route being executed.
            model: Alias or concrete model id.

        Returns:
            The first matching candidate.

        Raises:
            LLMConfigError: If no configured provider exposes the model.
        """
        for candidate in route.candidates:
            parts = self._candidate_parts(candidate)
            provider = self._config.providers[parts.provider]
            if parts.alias == model or provider.models[parts.alias].id == model:
                return parts
        for provider_name, provider in self._config.providers.items():
            for alias, model_config in provider.models.items():
                if alias == model or model_config.id == model:
                    return self._candidate_parts(f"{provider_name}/{alias}")
        raise LLMConfigError(f"no configured provider exposes model {model!r}")

    def _resolve_candidates(
        self,
        route: RouteConfig,
        provider_override: str | None,
        model_override: str | None,
        allow_fallback: bool | None,
    ) -> tuple[tuple[_Candidate, ...], bool]:
        """Resolve the ordered candidate chain for a request.

        Args:
            route: Route being executed.
            provider_override: Explicit provider from the caller.
            model_override: Explicit model from the caller.
            allow_fallback: Explicit fallback policy.

        Returns:
            The candidates and whether the first one is pinned.

        Raises:
            LLMConfigError: If an override is unknown.
        """
        env_provider = self._env_value("AI_NEWS_LLM_PROVIDER")
        env_model = self._env_value("AI_NEWS_LLM_MODEL")
        explicit_provider = provider_override.strip() if provider_override else None
        explicit_model = model_override.strip() if model_override else None
        pinned_provider = explicit_provider or env_provider or None
        pinned_model = explicit_model or (env_model if explicit_provider is None else None) or None

        if pinned_provider is not None:
            pinned_candidate = self._pin_candidate(route, pinned_provider, pinned_model)
        elif pinned_model is not None:
            pinned_candidate = self._candidate_for_model(route, pinned_model)
        else:
            return (
                tuple(self._candidate_parts(candidate) for candidate in route.candidates),
                False,
            )

        fallback = self._allow_fallback(allow_fallback, pinned=True)
        if not fallback:
            return (pinned_candidate,), True
        rest = tuple(
            self._candidate_parts(candidate)
            for candidate in route.candidates
            if candidate != pinned_candidate.raw
        )
        return (pinned_candidate, *rest), True

    # ------------------------------------------------------------------
    # Request shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_overrides(overrides: Mapping[str, Any]) -> None:
        """Reject unknown or malformed per-call overrides.

        Args:
            overrides: Caller overrides.

        Raises:
            LLMConfigError: If a key or value is invalid.
        """
        unknown = set(overrides) - _OVERRIDE_KEYS
        if unknown:
            raise LLMConfigError(f"unknown completion overrides: {sorted(unknown)}")
        if "temperature" in overrides:
            value = overrides["temperature"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise LLMConfigError("temperature override must be a non-negative number")
        if "max_tokens" in overrides and overrides["max_tokens"] is not None:
            value = overrides["max_tokens"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LLMConfigError("max_tokens override must be an integer >= 1")
        if "response_format" in overrides and overrides["response_format"] not in _VALID_FORMATS:
            raise LLMConfigError("response_format override is invalid")
        if (
            "json_schema" in overrides
            and overrides["json_schema"] is not None
            and not isinstance(overrides["json_schema"], Mapping)
        ):
            raise LLMConfigError("json_schema override must be a mapping")
        if "metadata" in overrides and not isinstance(overrides["metadata"], Mapping):
            raise LLMConfigError("metadata override must be a mapping")

    @staticmethod
    def _message_tokens(message: Message) -> int:
        """Estimate tokens for one message including role overhead.

        Args:
            message: Message to estimate.

        Returns:
            Estimated token count.
        """
        return 4 + estimate_tokens(message.content)

    def _messages_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate total tokens for a message list.

        Args:
            messages: Messages to estimate.

        Returns:
            Estimated token count.
        """
        return sum(self._message_tokens(message) for message in messages)

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        """Clip text to a character budget, preserving both ends.

        Args:
            text: Text to clip.
            max_chars: Maximum output length.

        Returns:
            The original text when it fits, otherwise a head/tail excerpt with a
            truncation marker.
        """
        if max_chars >= len(text):
            return text
        if max_chars <= len(_TRUNCATION_MARKER) + 2:
            return text[:max_chars]
        budget = max_chars - len(_TRUNCATION_MARKER)
        head = budget * 2 // 3
        tail = budget - head
        return f"{text[:head]}{_TRUNCATION_MARKER}{text[-tail:]}"

    def _fit_messages(
        self, messages: Sequence[Message], max_input_tokens: int | None
    ) -> tuple[Message, ...]:
        """Fit messages into the route's input-token budget.

        Oldest non-system messages are dropped first. If the remaining messages
        still exceed the budget, the largest message is clipped from the middle.

        Args:
            messages: Original messages.
            max_input_tokens: Route budget, or ``None`` for no limit.

        Returns:
            A non-empty fitted message tuple.
        """
        fitted = list(messages)
        if max_input_tokens is None:
            return tuple(fitted)
        while len(fitted) > 1 and self._messages_tokens(fitted) > max_input_tokens:
            index = next(
                (offset for offset, message in enumerate(fitted[:-1]) if message.role != "system"),
                None,
            )
            if index is None:
                break
            fitted.pop(index)
        for _ in range(8):
            total = self._messages_tokens(fitted)
            if total <= max_input_tokens:
                break
            largest = max(
                range(len(fitted)),
                key=lambda offset: (fitted[offset].role != "system", len(fitted[offset].content)),
            )
            message = fitted[largest]
            overflow_tokens = total - max_input_tokens
            allowed = max(1, len(message.content) - overflow_tokens * _CHARS_PER_TOKEN)
            clipped = self._clip_text(message.content, allowed)
            if clipped == message.content:
                break
            fitted[largest] = Message(role=message.role, content=clipped)
        return tuple(fitted)

    @staticmethod
    def _build_request(
        route: RouteConfig,
        messages: tuple[Message, ...],
        model_id: str,
        overrides: Mapping[str, Any],
    ) -> CompletionRequest:
        """Build a concrete request for one candidate.

        Args:
            route: Route policy.
            messages: Fitted messages.
            model_id: Concrete provider model id.
            overrides: Validated caller overrides.

        Returns:
            A completion request.

        Raises:
            LLMConfigError: If the request is internally invalid.
        """
        try:
            return CompletionRequest(
                messages=messages,
                model=model_id,
                temperature=float(overrides.get("temperature", route.temperature)),
                max_tokens=overrides.get("max_tokens", route.max_tokens),
                response_format=overrides.get("response_format", route.response_format),
                json_schema=overrides.get("json_schema"),
                metadata=overrides.get("metadata", {}),
            )
        except ValueError as err:
            raise LLMConfigError(str(err)) from err

    # ------------------------------------------------------------------
    # Cache and accounting
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(route: str, request: CompletionRequest) -> str:
        """Build a deterministic cache key for a request.

        Args:
            route: Route name.
            request: Concrete request.

        Returns:
            A SHA-256 hex digest.
        """
        payload = {
            "route": route,
            "model": request.model,
            "messages": [(message.role, message.content) for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "response_format": request.response_format,
            "json_schema": request.json_schema,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def clear_cache(self) -> None:
        """Remove every cached completion."""
        self._cache.clear()

    def _record_usage(self, route: str, usage: Usage) -> None:
        """Accumulate usage for a route.

        Args:
            route: Route name.
            usage: Reported usage.
        """
        current = self._usage.get(route, Usage())
        self._usage[route] = Usage(
            prompt_tokens=current.prompt_tokens + usage.prompt_tokens,
            completion_tokens=current.completion_tokens + usage.completion_tokens,
        )

    @property
    def usage_by_route(self) -> Mapping[str, Usage]:
        """Return accumulated usage per route as an immutable view."""
        return MappingProxyType(dict(self._usage))

    @property
    def total_usage(self) -> Usage:
        """Return accumulated usage across every route."""
        prompt = sum(usage.prompt_tokens for usage in self._usage.values())
        completion = sum(usage.completion_tokens for usage in self._usage.values())
        return Usage(prompt_tokens=prompt, completion_tokens=completion)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _retry_delay(self, error: LLMError, attempt: int) -> float:
        """Choose the delay before retrying a candidate.

        Args:
            error: The failed attempt.
            attempt: One-based attempt number.

        Returns:
            Delay in seconds, honoring ``Retry-After`` when supplied.
        """
        if isinstance(error, LLMRateLimitError) and error.retry_after is not None:
            return max(0.0, error.retry_after)
        return 0.25 * 2 ** (attempt - 1) + random.uniform(0.0, 0.1)

    async def _call_provider(
        self,
        candidate: _Candidate,
        request: CompletionRequest,
        timeout_s: float,
    ) -> CompletionResponse:
        """Call one provider with a timeout.

        Args:
            candidate: Resolved candidate.
            request: Concrete request.
            timeout_s: Per-attempt timeout.

        Returns:
            A normalized completion.

        Raises:
            LLMTransportError: If the call times out.
        """
        provider = self._get_provider(candidate.provider)
        try:
            result: CompletionResponse = await asyncio.wait_for(
                provider.complete(request), timeout=timeout_s
            )
        except TimeoutError as err:
            raise LLMTransportError(
                "Provider request timed out",
                provider=candidate.provider,
                model=candidate.model_id,
            ) from err
        return result

    def _dry_run_response(self, route: str, config: RouteConfig) -> CompletionResponse:
        """Build a deterministic dry-run response.

        Args:
            route: Route name.
            config: Route configuration.

        Returns:
            A response that never touched a provider.
        """
        text = f"dry-run:{route}" if config.response_format == "text" else "{}"
        return CompletionResponse(text=text, provider="dry-run", model="dry-run")

    @staticmethod
    def _log_success(
        route: str,
        response: CompletionResponse,
        elapsed_ms: float,
    ) -> None:
        """Log one successful completion without prompt content.

        Args:
            route: Route name.
            response: Provider response.
            elapsed_ms: Wall-clock latency in milliseconds.
        """
        logger.info(
            "llm route=%s provider=%s model=%s latency_ms=%.1f "
            "prompt_tokens=%d completion_tokens=%d",
            route,
            response.provider,
            response.model,
            elapsed_ms,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    async def complete(
        self,
        route: str,
        messages: Sequence[Message],
        *,
        provider: str | None = None,
        model: str | None = None,
        allow_fallback: bool | None = None,
        use_cache: bool = True,
        **overrides: Any,
    ) -> CompletionResponse:
        """Execute a route with retries, fallback, and optional overrides.

        Args:
            route: Route name. An empty value selects the default route.
            messages: Conversation messages.
            provider: Optional provider pin.
            model: Optional model alias or id pin.
            allow_fallback: Override the fallback policy for a pinned request.
            use_cache: Whether successful completions may be served from cache.
            **overrides: Optional ``temperature``, ``max_tokens``,
                ``response_format``, ``json_schema``, or ``metadata``.

        Returns:
            A normalized completion.

        Raises:
            LLMConfigError: If the route, provider, model, schema, or override is
                invalid.
            LLMUnavailableError: If no provider with credentials is available,
                or a pinned provider failed and fallback was disabled.
            LLMError: The last provider error when every candidate failed.
        """
        route_name = (route or "").strip() or self._default_route()
        route_config = self._config.routes.get(route_name)
        if route_config is None:
            raise LLMConfigError(f"unknown LLM route: {route_name!r}")
        message_tuple = tuple(messages)
        if not message_tuple:
            raise LLMConfigError("messages must contain at least one message")
        self._validate_overrides(overrides)
        if route_config.daily_token_budget is not None:
            used = self._usage.get(route_name, Usage()).total_tokens
            if used >= route_config.daily_token_budget:
                raise LLMConfigError(
                    f"route {route_name!r} exhausted its token budget "
                    f"({used}/{route_config.daily_token_budget})"
                )
        if self._dry_run:
            return self._dry_run_response(route_name, route_config)

        json_schema = overrides.get("json_schema")
        response_format = overrides.get("response_format", route_config.response_format)
        if response_format == "json_schema" and json_schema is None:
            raise LLMConfigError(
                f"route {route_name!r} requires an explicit json_schema override "
                f"for schema_ref {route_config.schema_ref!r}"
            )

        candidates, pinned = self._resolve_candidates(route_config, provider, model, allow_fallback)
        allow_fallback = self._allow_fallback(allow_fallback, pinned=pinned)
        fitted = self._fit_messages(message_tuple, route_config.max_input_tokens)
        last_error: LLMError | None = None
        usable = False
        pinned_candidate = candidates[0] if pinned else None

        for candidate in candidates:
            provider_config = self._config.providers[candidate.provider]
            if not self._provider_available(candidate.provider, self._clock()):
                logger.info(
                    "Skipping LLM provider %s: circuit breaker open",
                    candidate.provider,
                )
                continue
            if not self._has_credentials(provider_config):
                logger.info(
                    "Skipping LLM provider %s: %s is not set",
                    candidate.provider,
                    provider_config.api_key_env,
                )
                continue
            usable = True
            request = self._build_request(route_config, fitted, candidate.model_id, overrides)
            cache_key = self._cache_key(route_name, request)
            if use_cache:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._record_usage(route_name, cached.usage)
                    return cached

            candidate_error: LLMError | None = None
            for attempt in range(1, route_config.retries + 2):
                started = time.monotonic()
                try:
                    response = await self._call_provider(candidate, request, route_config.timeout_s)
                except LLMConfigError:
                    raise
                except LLMError as error:
                    candidate_error = error
                    last_error = error
                    if error.retryable and attempt <= route_config.retries:
                        delay = self._retry_delay(error, attempt)
                        logger.info(
                            "Retrying LLM candidate %s after attempt %d/%d in %.2fs",
                            candidate.raw,
                            attempt,
                            route_config.retries + 1,
                            delay,
                        )
                        await self._sleep(delay)
                        continue
                    break
                else:
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    self._record_success(candidate.provider)
                    self._record_usage(route_name, response.usage)
                    if use_cache:
                        self._cache[cache_key] = response
                    self._log_success(route_name, response, elapsed_ms)
                    return response

            if candidate_error is not None and self._counts_toward_breaker(candidate_error):
                self._record_failure(candidate.provider, self._clock())
            if candidate_error is not None:
                logger.warning("LLM candidate %s failed: %s", candidate.raw, candidate_error)

        if pinned and not allow_fallback and pinned_candidate is not None:
            detail = f": {last_error}" if last_error is not None else ""
            raise LLMUnavailableError(
                f"pinned provider {pinned_candidate.provider!r} is unavailable{detail}",
                provider=pinned_candidate.provider,
                model=pinned_candidate.model_id,
            ) from last_error
        if not usable:
            raise LLMUnavailableError(
                f"no provider with credentials is available for route {route_name!r}"
            )
        if last_error is not None:
            raise last_error
        raise LLMUnavailableError(f"no available provider could serve route {route_name!r}")

    async def aclose(self) -> None:
        """Close every provider created by this router."""
        providers = list(self._providers.values())
        self._providers.clear()
        for provider in providers:
            try:
                await provider.aclose()
            except Exception as err:  # pragma: no cover - defensive cleanup
                logger.warning("Error closing LLM provider: %r", err)
