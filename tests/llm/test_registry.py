"""Tests for provider resolution and entry-point discovery."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from Articles.llm.errors import LLMConfigError
from Articles.llm.groq import GroqProvider
from Articles.llm.openai_compat import OpenAICompatibleProvider
from Articles.llm.openrouter import OpenRouterProvider
from Articles.llm.registry import ProviderRegistry

from tests.llm import FakeProvider, make_model, make_provider


def plugin(name: str, factory: Any) -> SimpleNamespace:
    """Build a fake ``importlib.metadata.EntryPoint``-like object."""
    return SimpleNamespace(name=name, load=lambda: factory)


class TestDefaults(unittest.TestCase):
    """Built-in adapters resolve by provider name first, then kind."""

    def test_with_defaults_registers_builtins(self) -> None:
        registry = ProviderRegistry.with_defaults()
        names = registry.names()
        self.assertIn("groq", names)
        self.assertIn("openrouter", names)
        self.assertIn("openai_compatible", names)

    def test_provider_name_wins_over_kind(self) -> None:
        registry = ProviderRegistry.with_defaults()
        config = make_provider("groq", kind="openrouter")
        provider = registry.create(config, "key")
        self.assertIsInstance(provider, GroqProvider)

    def test_kind_resolution(self) -> None:
        registry = ProviderRegistry.with_defaults()
        openrouter = registry.create(make_provider("custom", kind="openrouter"), "key")
        compatible = registry.create(make_provider("local", kind="openai_compatible"), "key")
        self.assertIsInstance(openrouter, OpenRouterProvider)
        self.assertIsInstance(compatible, OpenAICompatibleProvider)

    def test_unknown_kind_raises(self) -> None:
        registry = ProviderRegistry.with_defaults()
        with self.assertRaises(LLMConfigError) as raised:
            registry.create(make_provider("custom", kind="mystery"), "key")
        self.assertIn("mystery", str(raised.exception))

    def test_generic_provider_requires_base_url(self) -> None:
        registry = ProviderRegistry.with_defaults()
        with self.assertRaises(LLMConfigError) as raised:
            registry.create(make_provider("local", kind="openai_compatible", base_url=""), "key")
        self.assertIn("base_url", str(raised.exception))


class TestRegistration(unittest.TestCase):
    """Custom factories receive configuration and the resolved key."""

    def test_register_and_create(self) -> None:
        seen: dict[str, Any] = {}
        expected = FakeProvider("custom", [])

        def factory(config: Any, api_key: str | None) -> Any:
            seen["config"] = config
            seen["api_key"] = api_key
            return expected

        registry = ProviderRegistry()
        registry.register("custom", factory)
        config = make_provider("custom", models={"fast": make_model("m-1")})
        result = registry.create(config, "secret")
        self.assertIs(result, expected)
        self.assertIs(seen["config"], config)
        self.assertEqual(seen["api_key"], "secret")
        self.assertEqual(registry.names(), ("custom",))

    def test_register_replaces_existing(self) -> None:
        first = FakeProvider("first", [])
        second = FakeProvider("second", [])
        registry = ProviderRegistry()
        registry.register("x", lambda config, key: first)
        registry.register("x", lambda config, key: second)
        self.assertIs(registry.create(make_provider("x"), None), second)


class TestEntryPointDiscovery(unittest.TestCase):
    """Plugins are optional and must never break registry construction."""

    def test_loads_entry_point_factory(self) -> None:
        expected = FakeProvider("acme", [])

        def factory(config: Any, api_key: str | None) -> Any:
            return expected

        with patch(
            "importlib.metadata.entry_points",
            return_value=[plugin("acme", factory)],
        ):
            registry = ProviderRegistry.with_defaults()
        self.assertIn("acme", registry.names())
        self.assertIs(registry.create(make_provider("acme"), None), expected)

    def test_broken_plugin_is_skipped(self) -> None:
        def load() -> Any:
            raise RuntimeError("boom")

        broken = SimpleNamespace(name="broken", load=load)
        with patch("importlib.metadata.entry_points", return_value=[broken]):
            registry = ProviderRegistry.with_defaults()
        self.assertNotIn("broken", registry.names())
        self.assertIn("groq", registry.names())

    def test_non_callable_plugin_is_skipped(self) -> None:
        with patch(
            "importlib.metadata.entry_points",
            return_value=[plugin("weird", "not-callable")],
        ):
            registry = ProviderRegistry.with_defaults()
        self.assertNotIn("weird", registry.names())

    def test_discovery_failure_is_survivable(self) -> None:
        with patch("importlib.metadata.entry_points", side_effect=RuntimeError("metadata broken")):
            registry = ProviderRegistry.with_defaults()
        self.assertIn("groq", registry.names())


if __name__ == "__main__":
    unittest.main()
