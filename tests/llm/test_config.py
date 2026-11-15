"""Unit tests for LLM configuration loading and validation."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Articles.llm.config import (
    LLMConfig,
    ModelConfig,
    ProviderConfig,
    RouteConfig,
    load_llm_config,
)
from Articles.llm.errors import LLMConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "config" / "llm_routes.example.yaml"

VALID = """
version: 1
providers:
  p:
    kind: openai_compatible
    base_url: https://p.test/v1
    api_key_env: P_KEY
    models:
      default:
        id: p-model
        context_window: 1000
        supports: [json_object]
routes:
  summarize:
    candidates: [p/default]
    response_format: json_object
    max_tokens: 10
    retries: 1
"""


class TestLoadConfig(unittest.TestCase):
    """YAML loading, examples, and environment selection."""

    def test_loads_example_config(self) -> None:
        config = load_llm_config(EXAMPLE)
        self.assertEqual(config.version, 1)
        self.assertIn("groq", config.providers)
        self.assertIn("summarize", config.routes)
        self.assertEqual(config.source_path, EXAMPLE)

    def test_from_env_uses_configured_path(self) -> None:
        with patch.dict(os.environ, {"AI_NEWS_LLM_ROUTES_FILE": str(EXAMPLE)}):
            config = LLMConfig.from_env()
        self.assertIn("extract_graph", config.routes)

    def test_from_env_accepts_explicit_path(self) -> None:
        self.assertIn("agent.answer", LLMConfig.from_env(EXAMPLE).routes)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(LLMConfigError):
            load_llm_config("/nonexistent/llm_routes.yaml")

    def test_invalid_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("providers: [", encoding="utf-8")
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_unsupported_version_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.yaml"
            path.write_text(VALID.replace("version: 1", "version: 2"), encoding="utf-8")
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_unknown_candidate_provider_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(VALID.replace("p/default", "missing/default"), encoding="utf-8")
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_unknown_candidate_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(VALID.replace("p/default", "p/missing"), encoding="utf-8")
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_unsupported_response_format_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                VALID.replace("response_format: json_object", "response_format: json_schema"),
                encoding="utf-8",
            )
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_candidates_must_be_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                VALID.replace("candidates: [p/default]", "candidates: p/default"),
                encoding="utf-8",
            )
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_provider_requires_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                VALID.replace("models:\n      default:", "models: {}"), encoding="utf-8"
            )
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_provider_key_must_match_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(VALID.replace("providers:\n  p:", "providers:\n  q:"), encoding="utf-8")
            with self.assertRaises(LLMConfigError):
                load_llm_config(path)

    def test_json_schema_requires_schema_ref(self) -> None:
        with self.assertRaises(LLMConfigError):
            RouteConfig(
                name="r",
                candidates=("p/default",),
                response_format="json_schema",
            )


class TestConfigModels(unittest.TestCase):
    """Direct construction of configuration dataclasses."""

    def test_base_url_defaults_for_builtin_provider(self) -> None:
        provider = ProviderConfig(
            name="groq",
            models={"fast": ModelConfig(id="m")},
        )
        self.assertEqual(provider.base_url_or_default(), "https://api.groq.com/openai/v1")

    def test_base_url_required_for_generic_provider(self) -> None:
        provider = ProviderConfig(
            name="custom",
            kind="openai_compatible",
            models={"fast": ModelConfig(id="m")},
        )
        with self.assertRaises(LLMConfigError):
            provider.base_url_or_default()

    def test_provider_requires_a_model(self) -> None:
        with self.assertRaises(LLMConfigError):
            ProviderConfig(name="p")

    def test_model_supports_is_normalized(self) -> None:
        model = ModelConfig(id="m", supports=["json_object"])  # type: ignore[arg-type]
        self.assertEqual(model.supports, frozenset({"json_object"}))

    def test_route_candidates_are_normalized(self) -> None:
        route = RouteConfig(name="r", candidates=["p/default"])  # type: ignore[arg-type]
        self.assertEqual(route.candidates, ("p/default",))

    def test_route_rejects_invalid_candidate(self) -> None:
        with self.assertRaises(LLMConfigError):
            RouteConfig(name="r", candidates=("no-slash",))

    def test_route_rejects_negative_retries(self) -> None:
        with self.assertRaises(LLMConfigError):
            RouteConfig(name="r", candidates=("p/default",), retries=-1)

    def test_config_rejects_duplicate_route_name(self) -> None:
        provider = ProviderConfig(
            name="p",
            base_url="https://p.test",
            models={"default": ModelConfig(id="m")},
        )
        route = RouteConfig(name="other", candidates=("p/default",))
        with self.assertRaises(LLMConfigError):
            LLMConfig(providers={"p": provider}, routes={"r": route})

    def test_provider_for_candidate(self) -> None:
        provider = ProviderConfig(
            name="p",
            base_url="https://p.test",
            models={"default": ModelConfig(id="m")},
        )
        config = LLMConfig(
            providers={"p": provider},
            routes={"r": RouteConfig(name="r", candidates=("p/default",))},
        )
        resolved_provider, resolved_model = config.provider_for_candidate("p/default")
        self.assertEqual(resolved_provider.name, "p")
        self.assertEqual(resolved_model.id, "m")

    def test_provider_for_candidate_rejects_malformed(self) -> None:
        provider = ProviderConfig(
            name="p",
            base_url="https://p.test",
            models={"default": ModelConfig(id="m")},
        )
        config = LLMConfig(
            providers={"p": provider},
            routes={"r": RouteConfig(name="r", candidates=("p/default",))},
        )
        with self.assertRaises(LLMConfigError):
            config.provider_for_candidate("p")


if __name__ == "__main__":
    unittest.main()
