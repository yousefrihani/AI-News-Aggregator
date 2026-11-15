"""Offline tests for the LLM pattern layer.

Every pattern runs against a scripted router, so no test here needs a network
connection or a provider SDK.
"""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import Any

from Articles.llm import CompletionResponse, Message
from Articles.llm.errors import LLMConfigError, LLMResponseError
from Articles.llm.patterns import (
    Classification,
    ClassifyPattern,
    ExtractGraphPattern,
    PatternContext,
    PatternRegistry,
    SingleShotPattern,
    SummarizePattern,
    parse_classification,
    parse_graph_extraction,
    parse_json_object,
)

from tests.llm.support import make_response


class ScriptedRouter:
    """Router double returning scripted completions and recording calls."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)
        self.calls: list[dict[str, Any]] = []

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
        """Record one call and return the next scripted text."""
        if not self.texts:
            raise AssertionError("no scripted response left")
        self.calls.append(
            {
                "route": route,
                "messages": tuple(messages),
                "provider": provider,
                "model": model,
                "allow_fallback": allow_fallback,
                "overrides": overrides,
            }
        )
        return make_response(text=self.texts.pop(0))


class TestSingleShotPattern(unittest.TestCase):
    """Base pattern validation and prompt shaping."""

    def test_rejects_blank_route(self) -> None:
        with self.assertRaises(LLMConfigError):
            SingleShotPattern(ScriptedRouter(["x"]), "  ")

    def test_rejects_non_string_system_prompt(self) -> None:
        with self.assertRaises(LLMConfigError):
            SingleShotPattern(
                ScriptedRouter(["x"]),
                "summarize",
                system_prompt=3,  # type: ignore[arg-type]
            )

    def test_build_messages_prepends_system_prompt(self) -> None:
        pattern = SingleShotPattern(ScriptedRouter(["x"]), "summarize", system_prompt="sys")
        messages = pattern.build_messages("hello")
        self.assertEqual([message.role for message in messages], ["system", "user"])
        self.assertEqual(messages[1].content, "hello")

    def test_build_messages_without_system_prompt(self) -> None:
        pattern = SingleShotPattern(ScriptedRouter(["x"]), "summarize")
        self.assertEqual(len(pattern.build_messages("hello")), 1)

    def test_complete_merges_context_and_overrides(self) -> None:
        router = ScriptedRouter(["x"])
        context = PatternContext(provider="groq", model="fast", overrides={"temperature": 0.1})
        pattern = SingleShotPattern(router, "summarize", context=context)

        async def call() -> CompletionResponse:
            return await pattern.complete(
                pattern.build_messages("hi"),
                temperature=0.9,
                max_tokens=10,
            )

        response = run(call())
        self.assertEqual(response.text, "x")
        call_record = router.calls[0]
        self.assertEqual(call_record["route"], "summarize")
        self.assertEqual(call_record["provider"], "groq")
        self.assertEqual(call_record["model"], "fast")
        self.assertEqual(call_record["overrides"], {"temperature": 0.9, "max_tokens": 10})

    def test_reserved_override_rejected(self) -> None:
        pattern = SingleShotPattern(ScriptedRouter(["x"]), "summarize")

        async def call() -> CompletionResponse:
            return await pattern.complete(pattern.build_messages("hi"), provider="groq")

        with self.assertRaises(LLMConfigError):
            run(call())

    def test_complete_text_returns_raw_completion(self) -> None:
        pattern = SingleShotPattern(ScriptedRouter(["plain"]), "summarize")
        self.assertEqual(run(pattern.complete_text("hi")).text, "plain")


class TestPatternContext(unittest.TestCase):
    """Context validation and immutability."""

    def test_blank_pin_rejected(self) -> None:
        with self.assertRaises(LLMConfigError):
            PatternContext(provider=" ")

    def test_non_mapping_overrides_rejected(self) -> None:
        with self.assertRaises(LLMConfigError):
            PatternContext(overrides=[("temperature", 1)])  # type: ignore[arg-type]

    def test_overrides_are_frozen(self) -> None:
        context = PatternContext(overrides={"temperature": 0.0})
        with self.assertRaises(TypeError):
            context.overrides["temperature"] = 1.0  # type: ignore[index]


class TestPatternRegistry(unittest.TestCase):
    """Default registry contents and lookup failures."""

    def test_default_names(self) -> None:
        registry = PatternRegistry.with_defaults()
        self.assertEqual(
            registry.names(),
            ("classify", "extract_graph", "graph_rag", "summarize"),
        )

    def test_create_unknown_pattern(self) -> None:
        with self.assertRaises(LLMConfigError):
            PatternRegistry.with_defaults().create("missing", ScriptedRouter(["x"]))

    def test_register_rejects_blank_name(self) -> None:
        with self.assertRaises(LLMConfigError):
            PatternRegistry().register(" ", SummarizePattern)

    def test_register_rejects_non_callable(self) -> None:
        with self.assertRaises(LLMConfigError):
            PatternRegistry().register("x", None)  # type: ignore[arg-type]

    def test_create_returns_configured_pattern(self) -> None:
        registry = PatternRegistry.with_defaults()
        pattern = registry.create("summarize", ScriptedRouter(["x"]))
        self.assertIsInstance(pattern, SummarizePattern)


class TestSummarizePattern(unittest.IsolatedAsyncioTestCase):
    """Prompt shape and empty-response handling."""

    async def test_run_returns_trimmed_summary(self) -> None:
        router = ScriptedRouter(["  a summary  "])
        pattern = SummarizePattern(router)
        self.assertEqual(await pattern.run("Title", "Body"), "a summary")
        call = router.calls[0]
        self.assertEqual(call["route"], "summarize")
        self.assertIn("<article>\nBody\n</article>", call["messages"][-1].content)

    async def test_prompt_includes_word_limit_and_title(self) -> None:
        pattern = SummarizePattern(ScriptedRouter(["x"]), max_words=42)
        prompt = pattern.build_prompt("Title", "Body")
        self.assertIn("Article title: Title", prompt)
        self.assertIn("at most 42 words", prompt)

    async def test_per_call_word_limit_overrides_default(self) -> None:
        pattern = SummarizePattern(ScriptedRouter(["x"]), max_words=42)
        self.assertIn("at most 7 words", pattern.build_prompt("T", "B", max_words=7))

    async def test_invalid_max_words_rejected(self) -> None:
        with self.assertRaises(LLMConfigError):
            SummarizePattern(ScriptedRouter(["x"]), max_words=0)
        pattern = SummarizePattern(ScriptedRouter(["x"]))
        with self.assertRaises(LLMConfigError):
            pattern.build_prompt("T", "B", max_words=-1)

    async def test_empty_summary_raises(self) -> None:
        pattern = SummarizePattern(ScriptedRouter(["   "]))
        with self.assertRaises(LLMResponseError):
            await pattern.run("Title", "Body")


class TestClassifyPattern(unittest.IsolatedAsyncioTestCase):
    """Classification prompt, parsing, and label validation."""

    async def test_run_parses_valid_json(self) -> None:
        router = ScriptedRouter(
            [
                '{"labels": ["technology", "business"], "primary": "technology", "rationale": "chips"}'
            ]
        )
        pattern = ClassifyPattern(router, labels=("technology", "business"))
        result = await pattern.run("Title", "Body")
        self.assertEqual(result, Classification(("technology", "business"), "technology", "chips"))
        self.assertEqual(router.calls[0]["overrides"]["response_format"], "json_object")

    async def test_prompt_lists_candidates(self) -> None:
        pattern = ClassifyPattern(ScriptedRouter(["{}"]), labels=("policy", "safety"))
        prompt = pattern.build_prompt("T", "B")
        self.assertIn("Candidate labels: policy, safety", prompt)

    async def test_unknown_labels_are_filtered(self) -> None:
        result = parse_classification(
            '{"labels": ["technology", "sports"], "primary": "sports"}',
            ("technology", "business"),
        )
        self.assertEqual(result.labels, ("technology",))
        self.assertEqual(result.primary, "technology")

    async def test_primary_defaults_to_first_label(self) -> None:
        result = parse_classification('{"labels": ["research", "policy"]}')
        self.assertEqual(result.primary, "research")

    async def test_empty_labels_allowed(self) -> None:
        result = parse_classification('{"labels": []}')
        self.assertEqual(result.labels, ())
        self.assertIsNone(result.primary)

    async def test_invalid_json_raises(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_classification("not json")

    async def test_non_array_labels_raise(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_classification('{"labels": "technology"}')

    async def test_non_string_label_raises(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_classification('{"labels": [3]}')

    async def test_invalid_label_configuration_raises(self) -> None:
        with self.assertRaises(LLMConfigError):
            ClassifyPattern(ScriptedRouter(["{}"]), labels=("ok", " "))
        with self.assertRaises(LLMConfigError):
            ClassifyPattern(ScriptedRouter(["{}"]), response_format="xml")


class TestExtractionParsing(unittest.TestCase):
    """Graph extraction normalization, filtering, and validation."""

    def test_valid_extraction_is_normalized(self) -> None:
        payload = """
        {
          "entities": [
            {"name": "OpenAI", "type": "Organization"},
            {"name": "GPT-5", "type": "Product", "canonical": "GPT-5"},
            {"name": "OpenAI", "type": "Organization"}
          ],
          "relations": [
            {"source": "OpenAI", "target": "gpt-5", "type": "Released", "confidence": 1.4}
          ],
          "mentions": [
            {"chunk_id": "c1", "entity": "OpenAI"},
            {"chunk_id": "c1", "entity": "OpenAI"},
            {"chunk_id": "c9", "entity": "OpenAI"}
          ]
        }
        """
        extraction = parse_graph_extraction(payload, allowed_chunk_ids=frozenset({"c1"}))
        self.assertEqual([entity.name for entity in extraction.entities], ["OpenAI", "GPT-5"])
        self.assertEqual(extraction.entities[0].type, "organization")
        self.assertEqual(len(extraction.relations), 1)
        self.assertEqual(extraction.relations[0].confidence, 1.0)
        self.assertEqual(extraction.relations[0].type, "released")
        self.assertEqual([(m.chunk_id, m.entity) for m in extraction.mentions], [("c1", "OpenAI")])

    def test_unknown_entities_are_dropped(self) -> None:
        payload = """
        {
          "entities": [{"name": "OpenAI", "type": "organization"}],
          "relations": [{"source": "OpenAI", "target": "Unknown", "type": "knows"}],
          "mentions": [{"chunk_id": "c1", "entity": "Unknown"}]
        }
        """
        extraction = parse_graph_extraction(payload)
        self.assertEqual(extraction.relations, ())
        self.assertEqual(extraction.mentions, ())

    def test_negative_confidence_clamped(self) -> None:
        payload = """
        {
          "entities": [
            {"name": "A", "type": "x"},
            {"name": "B", "type": "x"}
          ],
          "relations": [{"source": "A", "target": "B", "type": "r", "confidence": -3}]
        }
        """
        self.assertEqual(parse_graph_extraction(payload).relations[0].confidence, 0.0)

    def test_disallowed_evidence_is_dropped(self) -> None:
        payload = """
        {
          "entities": [
            {"name": "A", "type": "x"},
            {"name": "B", "type": "x"}
          ],
          "relations": [
            {"source": "A", "target": "B", "type": "r", "evidence_chunk_id": "c9"}
          ]
        }
        """
        extraction = parse_graph_extraction(payload, allowed_chunk_ids=frozenset({"c1"}))
        self.assertIsNone(extraction.relations[0].evidence_chunk_id)

    def test_missing_entities_array_raises(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_graph_extraction('{"relations": []}')

    def test_relations_must_be_array(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_graph_extraction('{"entities": [], "relations": {}}')

    def test_entity_requires_name_and_type(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_graph_extraction('{"entities": [{"name": "A"}], "relations": []}')

    def test_relation_requires_reference_fields(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_graph_extraction('{"entities": [], "relations": [{"source": "A", "type": "r"}]}')

    def test_confidence_must_be_numeric(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_graph_extraction(
                '{"entities": [{"name": "A", "type": "x"}, {"name": "B", "type": "x"}], '
                '"relations": [{"source": "A", "target": "B", "type": "r", '
                '"confidence": "high"}]}'
            )


class TestExtractGraphPattern(unittest.IsolatedAsyncioTestCase):
    """Schema request, repair, and failure behaviour."""

    async def test_run_returns_extraction_and_requests_schema(self) -> None:
        router = ScriptedRouter(['{"entities": [], "relations": []}'])
        pattern = ExtractGraphPattern(router)
        result = await pattern.run("Title", "Body", chunk_ids=["c1"])
        self.assertEqual(result.entities, ())
        self.assertEqual(router.calls[0]["overrides"]["response_format"], "json_schema")
        self.assertIn("Extract named entities", pattern.build_prompt("T", "B"))

    async def test_json_object_mode_omits_schema(self) -> None:
        router = ScriptedRouter(['{"entities": [], "relations": []}'])
        pattern = ExtractGraphPattern(router, response_format="json_object")
        await pattern.run("Title", "Body")
        self.assertNotIn("json_schema", router.calls[0]["overrides"])

    async def test_invalid_json_triggers_one_repair(self) -> None:
        router = ScriptedRouter(["not json", '{"entities": [], "relations": []}'])
        pattern = ExtractGraphPattern(router)
        result = await pattern.run("Title", "Body")
        self.assertEqual(result.entities, ())
        self.assertEqual(len(router.calls), 2)
        repair_messages = router.calls[1]["messages"]
        self.assertEqual(repair_messages[-2].role, "assistant")
        self.assertIn("failed validation", repair_messages[-1].content)

    async def test_repair_exhaustion_raises(self) -> None:
        router = ScriptedRouter(["not json", "still not json"])
        pattern = ExtractGraphPattern(router)
        with self.assertRaises(LLMResponseError):
            await pattern.run("Title", "Body")
        self.assertEqual(len(router.calls), 2)

    async def test_repairs_can_be_disabled(self) -> None:
        router = ScriptedRouter(["not json"])
        pattern = ExtractGraphPattern(router, max_repairs=0)
        with self.assertRaises(LLMResponseError):
            await pattern.run("Title", "Body")
        self.assertEqual(len(router.calls), 1)

    async def test_invalid_repair_configuration(self) -> None:
        with self.assertRaises(LLMConfigError):
            ExtractGraphPattern(ScriptedRouter(["x"]), max_repairs=-1)
        with self.assertRaises(LLMConfigError):
            ExtractGraphPattern(ScriptedRouter(["x"]), response_format="text")

    async def test_prompt_lists_valid_chunk_ids(self) -> None:
        pattern = ExtractGraphPattern(ScriptedRouter(["x"]))
        prompt = pattern.build_prompt("Title", "Body", chunk_ids=["b", "a"])
        self.assertIn("chunk_id=a", prompt)
        self.assertIn("chunk_id=b", prompt)


class TestParseJsonObject(unittest.TestCase):
    """JSON parsing tolerates one fenced block and rejects non-objects."""

    def test_plain_object(self) -> None:
        self.assertEqual(dict(parse_json_object('{"a": 1}')), {"a": 1})

    def test_fenced_object(self) -> None:
        self.assertEqual(dict(parse_json_object('```json\n{"a": 1}\n```')), {"a": 1})

    def test_array_rejected(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_json_object("[1, 2]")

    def test_malformed_rejected(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_json_object("{oops}")


def run(awaitable: Any) -> Any:
    """Run a coroutine to completion.

    Args:
        awaitable: Coroutine to run.

    Returns:
        The coroutine result.
    """
    import asyncio

    return asyncio.run(awaitable)


if __name__ == "__main__":
    unittest.main()
