"""Offline tests for the agent runtime seam.

The seam is exercised with fake steps, tools, and checkpoint stores. No
provider, graph adapter, or agent framework is imported.
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from Articles.agents import (
    AgentBudgetError,
    AgentCheckpointError,
    AgentConfigError,
    AgentError,
    AgentResult,
    AgentRuntime,
    AgentStep,
    AgentToolError,
    Budget,
    BudgetSnapshot,
    Checkpoint,
    CheckpointStore,
    SingleStepRuntime,
    StepContext,
    StepOutcome,
    Tool,
    ToolResult,
)
from Articles.llm import CompletionResponse, Message, Usage


class EchoTool:
    """Tool that echoes its ``text`` argument and records calls."""

    name = "echo"
    description = "Echo text back"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Record the arguments and echo ``text``."""
        self.calls.append(dict(arguments))
        return ToolResult(f"echo:{arguments.get('text', '')}")


class FakeRouter:
    """Router double that satisfies the completion protocol."""

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
        """Return a fixed completion."""
        return CompletionResponse(text=route, provider=provider or "fake", model=model or "fake")


class RecordingStep:
    """Step that increments a counter and captures its context."""

    name = "increment"

    def __init__(self, *, done: bool = True, usage: Usage | None = None) -> None:
        self.done = done
        self.usage = usage or Usage()
        self.contexts: list[StepContext] = []
        self.states: list[dict[str, int]] = []

    async def run(
        self,
        state: dict[str, int],
        context: StepContext,
    ) -> StepOutcome[dict[str, int]]:
        """Increment ``value`` and record the call."""
        self.contexts.append(context)
        self.states.append(dict(state))
        return StepOutcome(
            state={"value": state["value"] + 1},
            done=self.done,
            usage=self.usage,
        )


class MemoryCheckpoint:
    """In-memory checkpoint store used by runtime tests."""

    def __init__(self) -> None:
        self.saved: dict[str, Checkpoint[dict[str, int]]] = {}

    async def save(self, checkpoint: Checkpoint[dict[str, int]]) -> None:
        """Store the checkpoint."""
        self.saved[checkpoint.run_id] = checkpoint

    async def load(self, run_id: str) -> Checkpoint[dict[str, int]] | None:
        """Return a stored checkpoint, if any."""
        return self.saved.get(run_id)


class TestErrorHierarchy(unittest.TestCase):
    """Typed failures derive from the seam base class."""

    def test_errors_share_a_base(self) -> None:
        for error_type in (
            AgentBudgetError,
            AgentCheckpointError,
            AgentConfigError,
            AgentToolError,
        ):
            self.assertTrue(issubclass(error_type, AgentError))

    def test_error_carries_context(self) -> None:
        error = AgentError("failed", run_id="run-1", step="plan", retryable=True)
        self.assertEqual(str(error), "failed")
        self.assertEqual(error.run_id, "run-1")
        self.assertEqual(error.step, "plan")
        self.assertTrue(error.retryable)


class TestToolResult(unittest.TestCase):
    """Observation validation and metadata freezing."""

    def test_defaults(self) -> None:
        result = ToolResult("done")
        self.assertEqual(result.content, "done")
        self.assertFalse(result.is_error)
        self.assertEqual(dict(result.metadata), {})

    def test_metadata_is_frozen(self) -> None:
        result = ToolResult("done", metadata={"retries": 1})
        with self.assertRaises(TypeError):
            result.metadata["retries"] = 2  # type: ignore[index]

    def test_non_string_content_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            ToolResult(3)  # type: ignore[arg-type]

    def test_non_mapping_metadata_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            ToolResult("done", metadata=[("retries", 1)])  # type: ignore[arg-type]


class TestToolProtocol(unittest.IsolatedAsyncioTestCase):
    """Tools are callable through the structural protocol."""

    async def test_tool_protocol_is_usable(self) -> None:
        tool: Tool = EchoTool()
        result = await tool.run({"text": "hello"})
        self.assertEqual(result.content, "echo:hello")


class TestBudget(unittest.TestCase):
    """Step, token, and tool-call accounting."""

    def test_defaults(self) -> None:
        budget = Budget()
        self.assertEqual(budget.remaining_tokens, 20_000)
        self.assertEqual(budget.snapshot().steps_used, 0)

    def test_start_step_increments_and_exhausts(self) -> None:
        budget = Budget(max_steps=2)
        budget.start_step()
        budget.start_step()
        self.assertEqual(budget.steps_used, 2)
        with self.assertRaises(AgentBudgetError):
            budget.start_step()
        self.assertEqual(budget.steps_used, 2)

    def test_record_usage_adds_tokens(self) -> None:
        budget = Budget(max_tokens=10)
        budget.record_usage(Usage(prompt_tokens=2, completion_tokens=3))
        budget.record_usage(Usage(prompt_tokens=1, completion_tokens=1))
        self.assertEqual(budget.tokens_used, 7)
        self.assertEqual(budget.remaining_tokens, 3)

    def test_record_usage_overflow_does_not_mutate(self) -> None:
        budget = Budget(max_tokens=5)
        with self.assertRaises(AgentBudgetError):
            budget.record_usage(Usage(prompt_tokens=3, completion_tokens=3))
        self.assertEqual(budget.tokens_used, 0)

    def test_record_tool_call(self) -> None:
        budget = Budget(max_tool_calls=1)
        budget.record_tool_call()
        with self.assertRaises(AgentBudgetError):
            budget.record_tool_call()
        self.assertEqual(budget.tool_calls_used, 1)

    def test_invalid_limits_rejected(self) -> None:
        for kwargs in (
            {"max_steps": 0},
            {"max_tokens": 0},
            {"max_tool_calls": 0},
            {"max_steps": True},
            {"max_tokens": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(AgentConfigError):
                Budget(**kwargs)  # type: ignore[arg-type]

    def test_invalid_counters_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            Budget(max_steps=1, steps_used=2)
        with self.assertRaises(AgentConfigError):
            Budget(max_tokens=1, tokens_used=2)
        with self.assertRaises(AgentConfigError):
            Budget(max_tool_calls=1, tool_calls_used=2)
        with self.assertRaises(AgentConfigError):
            Budget(steps_used=-1)

    def test_snapshot_is_detached(self) -> None:
        budget = Budget(max_steps=3, max_tokens=5, max_tool_calls=2)
        budget.start_step()
        budget.record_usage(Usage(prompt_tokens=1, completion_tokens=1))
        budget.record_tool_call()
        snapshot = budget.snapshot()
        self.assertEqual(
            snapshot,
            BudgetSnapshot(
                max_steps=3,
                max_tokens=5,
                max_tool_calls=2,
                steps_used=1,
                tokens_used=2,
                tool_calls_used=1,
            ),
        )
        budget.start_step()
        self.assertEqual(snapshot.steps_used, 1)
        self.assertEqual(snapshot.remaining_tokens, 3)


class TestStepContext(unittest.TestCase):
    """Context validation and collaborator freezing."""

    def test_freezes_tools(self) -> None:
        tool = EchoTool()
        context = StepContext("run-1", 1, Budget(), tools={"echo": tool})
        self.assertIs(context.tools["echo"], tool)
        with self.assertRaises(TypeError):
            context.tools["other"] = tool  # type: ignore[index]

    def test_blank_run_id_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            StepContext(" ", 1, Budget())

    def test_invalid_step_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            StepContext("run-1", 0, Budget())

    def test_non_budget_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            StepContext("run-1", 1, 3)  # type: ignore[arg-type]

    def test_non_mapping_tools_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            StepContext("run-1", 1, Budget(), tools=[])  # type: ignore[arg-type]


class TestCheckpoint(unittest.TestCase):
    """Checkpoint validation."""

    def test_valid_checkpoint(self) -> None:
        checkpoint = Checkpoint("run-1", 1, {"value": 1})
        self.assertEqual(checkpoint.state, {"value": 1})

    def test_blank_run_id_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            Checkpoint(" ", 1, {})

    def test_invalid_step_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            Checkpoint("run-1", 0, {})


class TestAgentResult(unittest.TestCase):
    """Result validation."""

    def test_valid_result(self) -> None:
        result = AgentResult(
            run_id="run-1",
            state={"value": 1},
            steps=("increment",),
            budget=Budget().snapshot(),
        )
        self.assertEqual(result.steps, ("increment",))
        self.assertFalse(result.completed)

    def test_blank_fields_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            AgentResult(run_id=" ", state={}, steps=(), budget=Budget().snapshot())
        with self.assertRaises(AgentConfigError):
            AgentResult(run_id="run-1", state={}, steps=(" ",), budget=Budget().snapshot())


class TestSingleStepRuntime(unittest.IsolatedAsyncioTestCase):
    """Reference runtime behaviour and checkpoint replay."""

    async def test_runs_one_step_and_reports_budget(self) -> None:
        step = RecordingStep(usage=Usage(prompt_tokens=4, completion_tokens=6))
        runtime: AgentRuntime[dict[str, int]] = SingleStepRuntime(step)
        budget = Budget(max_steps=3, max_tokens=50, max_tool_calls=2)

        result = await runtime.run({"value": 1}, budget=budget, run_id="run-1")

        self.assertEqual(runtime.name, "single-step")
        self.assertEqual(result.run_id, "run-1")
        self.assertEqual(result.state, {"value": 2})
        self.assertEqual(result.steps, ("increment",))
        self.assertEqual(result.usage, Usage(prompt_tokens=4, completion_tokens=6))
        self.assertTrue(result.completed)
        self.assertEqual(result.budget.steps_used, 1)
        self.assertEqual(result.budget.tokens_used, 10)
        self.assertEqual(step.states, [{"value": 1}])
        context = step.contexts[0]
        self.assertEqual(context.run_id, "run-1")
        self.assertEqual(context.step, 1)
        self.assertIs(context.budget, budget)

    async def test_passes_tools_and_router(self) -> None:
        step = RecordingStep()
        tool = EchoTool()
        router = FakeRouter()
        runtime = SingleStepRuntime(step, tools=[tool], router=router)

        await runtime.run({"value": 0}, run_id="run-2")

        context = step.contexts[0]
        self.assertIs(context.tools["echo"], tool)
        self.assertIs(context.router, router)
        self.assertEqual(
            (await context.tools["echo"].run({"text": "hi"})).content,
            "echo:hi",
        )

    async def test_done_flag_is_propagated(self) -> None:
        runtime = SingleStepRuntime(RecordingStep(done=False))
        result = await runtime.run({"value": 0}, run_id="run-3")
        self.assertFalse(result.completed)

    async def test_default_budget_and_run_id(self) -> None:
        runtime = SingleStepRuntime(RecordingStep())
        first = await runtime.run({"value": 0})
        second = await runtime.run({"value": 0})
        self.assertEqual(first.budget.max_steps, 1)
        self.assertTrue(first.run_id.startswith("agent-"))
        self.assertNotEqual(first.run_id, second.run_id)

    async def test_exhausted_budget_propagates(self) -> None:
        runtime = SingleStepRuntime(RecordingStep(usage=Usage(prompt_tokens=5)))
        with self.assertRaises(AgentBudgetError):
            await runtime.run({"value": 0}, budget=Budget(max_tokens=3), run_id="run-4")

    async def test_invalid_budget_rejected(self) -> None:
        runtime = SingleStepRuntime(RecordingStep())
        with self.assertRaises(AgentConfigError):
            await runtime.run({"value": 0}, budget=3, run_id="run-5")  # type: ignore[arg-type]

    async def test_duplicate_tool_names_rejected(self) -> None:
        with self.assertRaises(AgentConfigError):
            SingleStepRuntime(RecordingStep(), tools=[EchoTool(), EchoTool()])

    async def test_checkpoint_is_saved(self) -> None:
        store = MemoryCheckpoint()
        runtime = SingleStepRuntime(RecordingStep())
        await runtime.run({"value": 1}, checkpoint=store, run_id="run-6")
        self.assertEqual(
            store.saved["run-6"],
            Checkpoint("run-6", 1, {"value": 2}),
        )

    async def test_existing_checkpoint_skips_execution(self) -> None:
        store = MemoryCheckpoint()
        step = RecordingStep()
        runtime = SingleStepRuntime(step)
        await runtime.run({"value": 1}, checkpoint=store, run_id="run-7")
        budget = Budget(max_steps=2)

        result = await runtime.run({"value": 99}, budget=budget, checkpoint=store, run_id="run-7")

        self.assertEqual(result.state, {"value": 2})
        self.assertEqual(result.steps, ())
        self.assertTrue(result.completed)
        self.assertEqual(budget.steps_used, 0)
        self.assertEqual(len(step.contexts), 1)

    async def test_structural_protocol_conformance(self) -> None:
        step: AgentStep[dict[str, int]] = RecordingStep()
        store: CheckpointStore[dict[str, int]] = MemoryCheckpoint()
        runtime: AgentRuntime[dict[str, int]] = SingleStepRuntime(step)
        result = await runtime.run({"value": 0}, checkpoint=store, run_id="run-8")
        self.assertEqual(result.state, {"value": 1})


if __name__ == "__main__":
    unittest.main()
