"""Offline tests for the Graph RAG answer pattern."""

import unittest
from collections.abc import Sequence
from typing import Any

from Articles.graph.base import Chunk
from Articles.graph.retriever import Citation, RetrievedContext
from Articles.llm import CompletionResponse, Message, Usage
from Articles.llm.errors import LLMConfigError, LLMResponseError
from Articles.llm.patterns import (
    GraphRAGAnswer,
    GraphRAGAnswerPattern,
    PatternRegistry,
    parse_citations,
)

from tests.llm.support import make_response


class ScriptedRouter:
    """Router double returning scripted text and recording calls."""

    def __init__(self, text: str = "Answer [a1].") -> None:
        self.text = text
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
        """Record one call and return the scripted text."""
        self.calls.append({"route": route, "messages": tuple(messages), "overrides": overrides})
        return make_response(text=self.text, prompt_tokens=10, completion_tokens=5)


def make_context() -> RetrievedContext:
    """Build a retrieved context citing two articles."""
    chunks = (
        Chunk("c1", "a1", 0, "OpenAI released GPT-5."),
        Chunk("c2", "a2", 0, "Analysts reacted."),
    )
    citations = (
        Citation("a1", ("c1",)),
        Citation("a2", ("c2",)),
    )
    return RetrievedContext(
        query="What happened?",
        chunks=chunks,
        citations=citations,
        context_text="[a1]\n  c1: OpenAI released GPT-5.\n\n[a2]\n  c2: Analysts reacted.",
    )


class TestGraphRAGAnswerPattern(unittest.IsolatedAsyncioTestCase):
    """Prompt constraints, citation parsing, and failure modes."""

    async def test_run_returns_answer_and_resolved_citations(self) -> None:
        router = ScriptedRouter("OpenAI released GPT-5 [a1]. Analysts reacted [a2] [a1].")
        pattern = GraphRAGAnswerPattern(router)

        answer = await pattern.run("What happened?", make_context())

        self.assertEqual(answer.article_ids, ("a1", "a2"))
        self.assertEqual(answer.unknown_citations, ())
        self.assertEqual(answer.usage, Usage(prompt_tokens=10, completion_tokens=5))
        self.assertTrue(answer.is_grounded)
        self.assertEqual(router.calls[0]["route"], "graph_rag")
        self.assertEqual(router.calls[0]["overrides"]["response_format"], "text")

    async def test_prompt_contains_question_context_and_rule(self) -> None:
        router = ScriptedRouter()
        pattern = GraphRAGAnswerPattern(router)
        prompt = pattern.build_prompt("What happened?", make_context())
        self.assertIn("Question: What happened?", prompt)
        self.assertIn("<context>", prompt)
        self.assertIn("[a1]", prompt)
        self.assertIn("square brackets", prompt)

    async def test_unknown_citations_are_reported_not_resolved(self) -> None:
        router = ScriptedRouter("Claim [a1] and [invented-9].")
        pattern = GraphRAGAnswerPattern(router)
        answer = await pattern.run("What happened?", make_context())
        self.assertEqual(answer.article_ids, ("a1",))
        self.assertEqual(answer.unknown_citations, ("invented-9",))

    async def test_answer_without_citations_is_allowed(self) -> None:
        router = ScriptedRouter("The context is insufficient.")
        pattern = GraphRAGAnswerPattern(router)
        answer = await pattern.run("What happened?", make_context())
        self.assertFalse(answer.is_grounded)
        self.assertEqual(answer.citations, ())

    async def test_empty_answer_raises(self) -> None:
        pattern = GraphRAGAnswerPattern(ScriptedRouter("   "))
        with self.assertRaises(LLMResponseError):
            await pattern.run("What happened?", make_context())

    async def test_blank_question_rejected(self) -> None:
        pattern = GraphRAGAnswerPattern(ScriptedRouter())
        with self.assertRaises(LLMConfigError):
            pattern.build_prompt("  ", make_context())

    async def test_empty_context_rejected(self) -> None:
        pattern = GraphRAGAnswerPattern(ScriptedRouter())
        with self.assertRaises(LLMConfigError):
            pattern.build_prompt("Question", RetrievedContext(query="Question"))

    async def test_non_context_value_rejected(self) -> None:
        pattern = GraphRAGAnswerPattern(ScriptedRouter())
        with self.assertRaises(LLMConfigError):
            pattern.build_prompt("Question", "not context")  # type: ignore[arg-type]

    def test_registry_includes_graph_rag(self) -> None:
        registry = PatternRegistry.with_defaults()
        self.assertIn("graph_rag", registry.names())
        self.assertIsInstance(registry.create("graph_rag", ScriptedRouter()), GraphRAGAnswerPattern)


class TestParseCitations(unittest.TestCase):
    """Bracket parsing against sequences, mappings, and unknown ids."""

    def test_sequence_input(self) -> None:
        resolved, unknown = parse_citations("A [a1] B [a2]", (Citation("a1"), Citation("a2")))
        self.assertEqual([item.article_id for item in resolved], ["a1", "a2"])
        self.assertEqual(unknown, ())

    def test_mapping_input(self) -> None:
        resolved, unknown = parse_citations(
            "A [a1]", {"a1": Citation("a1", ("c1",)), "a2": Citation("a2")}
        )
        self.assertEqual(resolved, (Citation("a1", ("c1",)),))
        self.assertEqual(unknown, ())

    def test_duplicates_and_unknowns(self) -> None:
        resolved, unknown = parse_citations("[a1] [a1] [x] [x]", (Citation("a1"),))
        self.assertEqual(resolved, (Citation("a1"),))
        self.assertEqual(unknown, ("x",))

    def test_lowercase_is_case_sensitive(self) -> None:
        resolved, unknown = parse_citations("[A1]", (Citation("a1"),))
        self.assertEqual(resolved, ())
        self.assertEqual(unknown, ("A1",))

    def test_empty_markers_ignored(self) -> None:
        resolved, unknown = parse_citations("[] [ ]", (Citation("a1"),))
        self.assertEqual(resolved, ())
        self.assertEqual(unknown, ())

    def test_graph_rag_answer_validates(self) -> None:
        with self.assertRaises(ValueError):
            GraphRAGAnswer("   ")
