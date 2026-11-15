"""Provider registry and plugin discovery.

The registry is the only place that maps a provider name or adapter ``kind`` to
a concrete class. Adding a vendor through an entry point requires no change to
the router, patterns, or agents.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable

from Articles.llm.base import LLMProvider
from Articles.llm.config import ProviderConfig
from Articles.llm.errors import LLMConfigError
from Articles.llm.groq import GroqProvider
from Articles.llm.openai_compat import OpenAICompatibleProvider
from Articles.llm.openrouter import OpenRouterProvider

__all__ = ["ProviderFactory", "ProviderRegistry"]

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[ProviderConfig, str | None], LLMProvider]

_ENTRY_POINT_GROUP = "ai_news_aggregator.llm_providers"


class ProviderRegistry:
    """Map provider names and adapter kinds to factories.

    Args:
        factories: Optional initial factories keyed by name.
    """

    def __init__(self, factories: dict[str, ProviderFactory] | None = None) -> None:
        self._factories: dict[str, ProviderFactory] = dict(factories or {})

    def register(self, name: str, factory: ProviderFactory) -> ProviderRegistry:
        """Register or replace a provider factory.

        Args:
            name: Provider name or adapter kind.
            factory: Callable building a provider from config and API key.

        Returns:
            ``self`` to allow chaining.

        Raises:
            LLMConfigError: If the name is blank or the factory is not callable.
        """
        if not isinstance(name, str) or not name.strip():
            raise LLMConfigError("provider factory name must be a non-empty string")
        if not callable(factory):
            raise LLMConfigError(f"provider factory {name!r} must be callable")
        self._factories[name] = factory
        return self

    def names(self) -> tuple[str, ...]:
        """Return registered factory names in sorted order."""
        return tuple(sorted(self._factories))

    def create(self, config: ProviderConfig, api_key: str | None) -> LLMProvider:
        """Instantiate a provider using its name, then its adapter kind.

        Args:
            config: Provider configuration.
            api_key: Resolved API key, or ``None`` for keyless endpoints.

        Returns:
            A provider instance.

        Raises:
            LLMConfigError: If no factory matches the provider name or kind, or
                if the factory returns an invalid object.
        """
        factory = self._factories.get(config.name) or self._factories.get(config.kind)
        if factory is None:
            raise LLMConfigError(
                f"no LLM provider adapter registered for name {config.name!r} "
                f"or kind {config.kind!r}"
            )
        provider = factory(config, api_key)
        if not isinstance(provider, LLMProvider):
            raise LLMConfigError(f"provider factory {config.name!r} returned an invalid provider")
        return provider

    def _discover_entry_points(self) -> None:
        """Register factories exposed by installed plugin packages.

        Discovery is best-effort: broken metadata or a broken plugin must never
        prevent the built-in adapters from being usable.
        """
        try:
            entry_points = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except Exception as err:  # pragma: no cover - depends on installed metadata
            logger.warning("Skipping LLM provider discovery: %r", err)
            return
        for entry_point in entry_points:
            try:
                factory = entry_point.load()
            except Exception as err:  # pragma: no cover - broken third-party plugin
                logger.warning("Skipping LLM provider plugin %r: %r", entry_point.name, err)
                continue
            if not callable(factory):
                logger.warning("Skipping LLM provider plugin %r: not callable", entry_point.name)
                continue
            self.register(entry_point.name, factory)

    @classmethod
    def with_defaults(cls) -> ProviderRegistry:
        """Build a registry containing the built-in adapters and plugins.

        Returns:
            A registry with Groq, OpenRouter, and the generic OpenAI-compatible
            adapter registered, plus any discovered entry-point plugins.
        """

        def _generic(config: ProviderConfig, api_key: str | None) -> LLMProvider:
            return OpenAICompatibleProvider(
                config.name,
                config.base_url_or_default(),
                api_key,
                headers=config.headers,
            )

        registry = cls()
        registry.register("groq", lambda config, key: GroqProvider(config, key))
        registry.register("openrouter", lambda config, key: OpenRouterProvider(config, key))
        registry.register("openai_compatible", _generic)
        registry._discover_entry_points()
        return registry
