"""Shared pattern protocol, context, registry, and single-shot implementation.

A **pattern** is the unit that turns a typed task into messages, asks a route
for a completion, and validates the result. Patterns know route names, never
vendors, so adding a provider or changing fallback order never touches them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from Articles.llm.base import CompletionResponse, Message
from Articles.llm.errors import LLMConfigError, LLMResponseError

__all__ = [
    "CompletionRouter",
    "Pattern",
    "PatternContext",
    "PatternRegistry",
    "SingleShotPattern",
    "parse_json_object",
]


@runtime_checkable
class CompletionRouter(Protocol):
    """The narrow router surface patterns depend on."""

    async def complete(
        self,
        route: str,
        messages: Sequence[Message],
        *,
        provider: str | None = None,
        model: str | None = None,
        allow_fallback: bool | None = None,
        **overrides: Any,
    ) -> CompletionResponse:
        """Execute one route and return a normalized completion."""


@runtime_checkable
class Pattern(Protocol):
    """Structural contract shared by every pattern."""

    name: str

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the pattern."""


@dataclass(frozen=True, slots=True)
class PatternContext:
    """Routing hints applied to every call made by a pattern.

    Args:
        provider: Optional provider pin.
        model: Optional model alias or id pin.
        allow_fallback: Optional fallback policy for a pinned provider.
        overrides: Extra completion overrides such as ``temperature``.
    """

    provider: str | None = None
    model: str | None = None
    allow_fallback: bool | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate hints and freeze the override mapping.

        Raises:
            LLMConfigError: If a pin is blank or overrides are not a mapping.
        """
        for name, value in (("provider", self.provider), ("model", self.model)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise LLMConfigError(f"pattern context {name} must be a non-empty string")
        if not isinstance(self.overrides, Mapping):
            raise LLMConfigError("pattern context overrides must be a mapping")
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))


class SingleShotPattern:
    """One request with an optional system prompt.

    Subclasses shape the prompt and parse the response; retries, fallback, and
    provider selection stay in the router.

    Args:
        router: Router facade used for every completion.
        route: Logical route name.
        system_prompt: Optional system message prepended to each request.
        context: Optional routing hints.
    """

    name = "single_shot"

    def __init__(
        self,
        router: CompletionRouter,
        route: str,
        *,
        system_prompt: str | None = None,
        context: PatternContext | None = None,
    ) -> None:
        if not isinstance(route, str) or not route.strip():
            raise LLMConfigError("pattern route must be a non-empty string")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise LLMConfigError("system_prompt must be a string")
        self._router = router
        self.route = route.strip()
        self.system_prompt = system_prompt
        self.context = context or PatternContext()

    def build_messages(self, prompt: str) -> tuple[Message, ...]:
        """Build the message sequence for one prompt.

        Args:
            prompt: User prompt.

        Returns:
            Optional system message followed by the user message.
        """
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message("system", self.system_prompt))
        messages.append(Message("user", prompt))
        return tuple(messages)

    async def complete(
        self,
        messages: Sequence[Message],
        **overrides: Any,
    ) -> CompletionResponse:
        """Call the configured route.

        Args:
            messages: Conversation messages.
            **overrides: Completion overrides merged over the context.

        Returns:
            A normalized completion.
        """
        merged = {**dict(self.context.overrides), **overrides}
        for reserved in ("provider", "model", "allow_fallback"):
            if reserved in merged:
                raise LLMConfigError(
                    f"use PatternContext to set {reserved!r}, not a completion override"
                )
        return await self._router.complete(
            self.route,
            messages,
            provider=self.context.provider,
            model=self.context.model,
            allow_fallback=self.context.allow_fallback,
            **merged,
        )

    async def complete_text(self, prompt: str, **overrides: Any) -> CompletionResponse:
        """Send a single prompt and return the raw completion.

        Subclasses expose their own typed ``run`` method; this helper keeps the
        one-shot prompt path in one place.

        Args:
            prompt: User prompt.
            **overrides: Completion overrides.

        Returns:
            A normalized completion.
        """
        return await self.complete(self.build_messages(prompt), **overrides)


class PatternRegistry:
    """Name-to-factory registry for pattern lookup."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._factories: dict[str, Callable[..., Pattern]] = {}

    def register(self, name: str, factory: Callable[..., Pattern]) -> None:
        """Register or replace a pattern factory.

        Args:
            name: Pattern name.
            factory: Callable returning a pattern.

        Raises:
            LLMConfigError: If the name is blank or the factory is not callable.
        """
        if not isinstance(name, str) or not name.strip():
            raise LLMConfigError("pattern name must be a non-empty string")
        if not callable(factory):
            raise LLMConfigError(f"pattern factory {name!r} must be callable")
        self._factories[name.strip()] = factory

    def create(self, name: str, *args: Any, **kwargs: Any) -> Pattern:
        """Create a registered pattern.

        Args:
            name: Registered pattern name.
            *args: Positional factory arguments.
            **kwargs: Keyword factory arguments.

        Returns:
            The constructed pattern.

        Raises:
            LLMConfigError: If the pattern is not registered.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise LLMConfigError(f"unknown pattern: {name!r}")
        return factory(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered names in sorted order."""
        return tuple(sorted(self._factories))

    @classmethod
    def with_defaults(cls) -> PatternRegistry:
        """Return a registry containing the shipped patterns.

        Imports are local so pattern modules can import this module without a
        circular import at load time.
        """
        from Articles.llm.patterns.classify import ClassifyPattern
        from Articles.llm.patterns.extract_graph import ExtractGraphPattern
        from Articles.llm.patterns.summarize import SummarizePattern

        registry = cls()
        registry.register("summarize", SummarizePattern)
        registry.register("classify", ClassifyPattern)
        registry.register("extract_graph", ExtractGraphPattern)
        return registry


def parse_json_object(text: str) -> Mapping[str, Any]:
    """Parse a JSON object, tolerating a single fenced code block.

    Args:
        text: Raw assistant text.

    Returns:
        The parsed object.

    Raises:
        LLMResponseError: If the text is not a JSON object.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        last_fence = candidate.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            candidate = candidate[first_newline + 1 : last_fence].strip()
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError) as err:
        raise LLMResponseError("pattern response is not valid JSON") from err
    if not isinstance(parsed, Mapping):
        raise LLMResponseError("pattern response must be a JSON object")
    return parsed
