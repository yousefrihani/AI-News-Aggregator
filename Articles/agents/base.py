"""Agent runtime seam: protocols and a reference single-step runtime.

This module freezes the contract a custom agent workflow will implement. It
contains **no production workflow**: state, steps, tools, budgets, checkpoints,
and the runtime entry point are defined here, while the workflow itself is
supplied separately. Nothing imports a provider SDK or a graph adapter.

Design rules:

* State is opaque to the seam. A workflow chooses its own state type and only
  the concrete runtime needs to understand it.
* Steps are pure async callables returning a new state, so a workflow can be
  composed, replayed, and checkpointed without shared mutable scratch space.
* Budgets are enforced by the runtime and are also available to steps, which
  keeps tool loops bounded by construction.
* Routes, never vendor names, are passed to steps through the narrow
  :class:`~Articles.llm.patterns.base.CompletionRouter` protocol.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar

from Articles.llm.base import Usage
from Articles.llm.patterns.base import CompletionRouter

__all__ = [
    "AgentBudgetError",
    "AgentCheckpointError",
    "AgentConfigError",
    "AgentError",
    "AgentResult",
    "AgentRuntime",
    "AgentStep",
    "AgentToolError",
    "Budget",
    "BudgetSnapshot",
    "Checkpoint",
    "CheckpointStore",
    "SingleStepRuntime",
    "StepContext",
    "StepOutcome",
    "Tool",
    "ToolResult",
]

StateT = TypeVar("StateT")


class AgentError(Exception):
    """Base class for every agent-runtime failure.

    Args:
        message: Human-readable description.
        run_id: Run identifier, when known.
        step: Step name, when known.
        retryable: Whether the caller may retry the same operation.
    """

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        step: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.run_id = run_id
        self.step = step
        self.retryable = retryable


class AgentConfigError(AgentError):
    """Raised when the seam receives an invalid configuration."""


class AgentBudgetError(AgentError):
    """Raised when a step, token, or tool-call budget is exhausted."""


class AgentToolError(AgentError):
    """Raised when a tool call cannot be completed."""


class AgentCheckpointError(AgentError):
    """Raised when a checkpoint cannot be saved or restored."""


def _require_non_empty(name: str, value: str) -> None:
    """Validate a required string.

    Args:
        name: Field name used in the error message.
        value: Value to validate.

    Raises:
        AgentConfigError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise AgentConfigError(f"{name} must be a non-empty string")


def _require_positive_int(name: str, value: int) -> None:
    """Validate a positive integer.

    Args:
        name: Field name used in the error message.
        value: Value to validate.

    Raises:
        AgentConfigError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AgentConfigError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: int) -> None:
    """Validate a non-negative integer.

    Args:
        name: Field name used in the error message.
        value: Value to validate.

    Raises:
        AgentConfigError: If the value is not a non-negative integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AgentConfigError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Observation returned by a tool.

    Args:
        content: Text the workflow can place in the next prompt.
        is_error: Whether the tool failed in a way the model may recover from.
        metadata: Structured, JSON-friendly details for logs and tracing.
    """

    content: str
    is_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze the observation.

        Raises:
            AgentConfigError: If ``content`` is not a string or ``metadata`` is
                not a mapping.
        """
        if not isinstance(self.content, str):
            raise AgentConfigError("tool result content must be a string")
        if not isinstance(self.metadata, Mapping):
            raise AgentConfigError("tool result metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class Tool(Protocol):
    """Structural contract for a callable tool."""

    name: str
    description: str

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Execute the tool.

        Args:
            arguments: Tool arguments supplied by a step or a model.

        Returns:
            A validated observation.
        """


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Immutable view of budget consumption at one point in a run.

    Args:
        max_steps: Maximum steps allowed.
        max_tokens: Maximum total tokens allowed.
        max_tool_calls: Maximum tool calls allowed.
        steps_used: Steps consumed so far.
        tokens_used: Tokens consumed so far.
        tool_calls_used: Tool calls consumed so far.
    """

    max_steps: int
    max_tokens: int
    max_tool_calls: int
    steps_used: int
    tokens_used: int
    tool_calls_used: int

    @property
    def remaining_tokens(self) -> int:
        """Return the token allowance left in the run."""
        return self.max_tokens - self.tokens_used


@dataclass(slots=True)
class Budget:
    """Mutable budget ledger shared by a runtime and its steps.

    Args:
        max_steps: Maximum steps allowed.
        max_tokens: Maximum total prompt and completion tokens allowed.
        max_tool_calls: Maximum tool calls allowed.

    Raises:
        AgentConfigError: If a limit is not a positive integer.
    """

    max_steps: int = 8
    max_tokens: int = 20_000
    max_tool_calls: int = 16
    steps_used: int = 0
    tokens_used: int = 0
    tool_calls_used: int = 0

    def __post_init__(self) -> None:
        """Validate limits and consumed counters.

        Raises:
            AgentConfigError: If a value is out of range.
        """
        _require_positive_int("max_steps", self.max_steps)
        _require_positive_int("max_tokens", self.max_tokens)
        _require_positive_int("max_tool_calls", self.max_tool_calls)
        _require_non_negative_int("steps_used", self.steps_used)
        _require_non_negative_int("tokens_used", self.tokens_used)
        _require_non_negative_int("tool_calls_used", self.tool_calls_used)
        if self.steps_used > self.max_steps:
            raise AgentConfigError("steps_used cannot exceed max_steps")
        if self.tokens_used > self.max_tokens:
            raise AgentConfigError("tokens_used cannot exceed max_tokens")
        if self.tool_calls_used > self.max_tool_calls:
            raise AgentConfigError("tool_calls_used cannot exceed max_tool_calls")

    @property
    def remaining_tokens(self) -> int:
        """Return the token allowance left in the run."""
        return self.max_tokens - self.tokens_used

    def start_step(self) -> None:
        """Consume one step.

        Raises:
            AgentBudgetError: If the step budget is exhausted.
        """
        if self.steps_used >= self.max_steps:
            raise AgentBudgetError("step budget exhausted")
        self.steps_used += 1

    def record_usage(self, usage: Usage) -> None:
        """Consume token usage reported by a provider.

        Args:
            usage: Token counters to add.

        Raises:
            AgentBudgetError: If the token budget would be exceeded. The
                ledger is left unchanged in that case.
        """
        total = usage.total_tokens
        if self.tokens_used + total > self.max_tokens:
            raise AgentBudgetError(
                f"token budget exhausted ({self.tokens_used}+{total}>{self.max_tokens})"
            )
        self.tokens_used += total

    def record_tool_call(self) -> None:
        """Consume one tool call.

        Raises:
            AgentBudgetError: If the tool-call budget is exhausted.
        """
        if self.tool_calls_used >= self.max_tool_calls:
            raise AgentBudgetError("tool-call budget exhausted")
        self.tool_calls_used += 1

    def snapshot(self) -> BudgetSnapshot:
        """Return an immutable copy of the ledger."""
        return BudgetSnapshot(
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            max_tool_calls=self.max_tool_calls,
            steps_used=self.steps_used,
            tokens_used=self.tokens_used,
            tool_calls_used=self.tool_calls_used,
        )


@dataclass(frozen=True, slots=True)
class StepContext:
    """Collaborators handed to a step for one execution.

    Args:
        run_id: Stable identifier for the run.
        step: One-based step number.
        budget: Shared budget ledger.
        tools: Available tools keyed by name.
        router: Route-based completion facade, when the step needs an LLM.
    """

    run_id: str
    step: int
    budget: Budget
    tools: Mapping[str, Tool] = field(default_factory=dict)
    router: CompletionRouter | None = None

    def __post_init__(self) -> None:
        """Validate the context and freeze the tool mapping.

        Raises:
            AgentConfigError: If an identifier, step number, or collaborator is
                invalid.
        """
        _require_non_empty("run_id", self.run_id)
        _require_positive_int("step", self.step)
        if not isinstance(self.budget, Budget):
            raise AgentConfigError("step context requires a Budget")
        if not isinstance(self.tools, Mapping):
            raise AgentConfigError("step context tools must be a mapping")
        object.__setattr__(self, "tools", MappingProxyType(dict(self.tools)))


@dataclass(frozen=True, slots=True)
class StepOutcome(Generic[StateT]):
    """Result of one step.

    Args:
        state: State produced by the step.
        done: Whether the workflow has finished.
        usage: Token usage incurred by the step.
    """

    state: StateT
    done: bool = False
    usage: Usage = field(default_factory=Usage)


class AgentStep(Protocol[StateT]):
    """Structural contract for one workflow step."""

    name: str

    async def run(self, state: StateT, context: StepContext) -> StepOutcome[StateT]:
        """Run the step.

        Args:
            state: Current workflow state.
            context: Collaborators and budget for this execution.

        Returns:
            The next state and a completion flag.
        """


@dataclass(frozen=True, slots=True)
class Checkpoint(Generic[StateT]):
    """A restorable workflow snapshot.

    Args:
        run_id: Stable identifier for the run.
        step: One-based step number that produced the state.
        state: Opaque workflow state.
    """

    run_id: str
    step: int
    state: StateT

    def __post_init__(self) -> None:
        """Validate identifiers.

        Raises:
            AgentConfigError: If ``run_id`` is blank or ``step`` is not
                positive.
        """
        _require_non_empty("run_id", self.run_id)
        _require_positive_int("step", self.step)


class CheckpointStore(Protocol[StateT]):
    """Persistence port for workflow checkpoints."""

    async def save(self, checkpoint: Checkpoint[StateT]) -> None:
        """Persist a checkpoint.

        Args:
            checkpoint: Snapshot to store.
        """

    async def load(self, run_id: str) -> Checkpoint[StateT] | None:
        """Load the latest checkpoint for a run.

        Args:
            run_id: Run identifier.

        Returns:
            The stored checkpoint, or ``None`` when the run is new.
        """


@dataclass(frozen=True, slots=True)
class AgentResult(Generic[StateT]):
    """Final output of a runtime.

    Args:
        run_id: Stable identifier for the run.
        state: Final workflow state.
        steps: Names of the steps that executed, in order.
        usage: Total token usage for the run.
        budget: Budget consumption at completion.
        completed: Whether the workflow signalled completion.
    """

    run_id: str
    state: StateT
    steps: tuple[str, ...]
    budget: BudgetSnapshot
    usage: Usage = field(default_factory=Usage)
    completed: bool = False

    def __post_init__(self) -> None:
        """Validate identifiers and step names.

        Raises:
            AgentConfigError: If the run identifier or a step name is blank.
        """
        _require_non_empty("run_id", self.run_id)
        for step in self.steps:
            _require_non_empty("step", step)


class AgentRuntime(Protocol[StateT]):
    """Structural contract for an agent runtime."""

    name: str

    async def run(
        self,
        state: StateT,
        *,
        budget: Budget | None = None,
        checkpoint: CheckpointStore[StateT] | None = None,
        run_id: str | None = None,
    ) -> AgentResult[StateT]:
        """Execute a workflow.

        Args:
            state: Initial workflow state.
            budget: Optional budget ledger. Runtimes create one when omitted.
            checkpoint: Optional checkpoint store for replay and recovery.
            run_id: Optional stable run identifier.

        Returns:
            The final result.
        """


class SingleStepRuntime(Generic[StateT]):
    """Reference runtime that executes exactly one step.

    This exists to prove the seam is usable and testable. It is not a
    production workflow; custom runtimes implement :class:`AgentRuntime`
    directly.

    Args:
        step: Step to execute.
        tools: Tools exposed to the step. Names must be unique.
        router: Optional route-based completion facade.

    Raises:
        AgentConfigError: If two tools share a name.
    """

    name = "single-step"

    def __init__(
        self,
        step: AgentStep[StateT],
        *,
        tools: Sequence[Tool] = (),
        router: CompletionRouter | None = None,
    ) -> None:
        self._step = step
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise AgentConfigError(f"duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool
        self._router = router

    async def run(
        self,
        state: StateT,
        *,
        budget: Budget | None = None,
        checkpoint: CheckpointStore[StateT] | None = None,
        run_id: str | None = None,
    ) -> AgentResult[StateT]:
        """Run the single configured step.

        When a checkpoint already exists for ``run_id``, the stored state is
        returned as a completed result without executing the step again.

        Args:
            state: Initial workflow state.
            budget: Optional budget ledger. Defaults to a one-step budget.
            checkpoint: Optional checkpoint store.
            run_id: Optional stable run identifier.

        Returns:
            The step result and budget snapshot.

        Raises:
            AgentConfigError: If ``budget`` is not a :class:`Budget`.
            AgentBudgetError: If the step or token budget is exhausted.
        """
        active_budget = budget if budget is not None else Budget(max_steps=1)
        if not isinstance(active_budget, Budget):
            raise AgentConfigError("budget must be a Budget")
        active_run_id = run_id or f"agent-{uuid.uuid4().hex}"
        _require_non_empty("run_id", active_run_id)

        if checkpoint is not None:
            existing = await checkpoint.load(active_run_id)
            if existing is not None:
                return AgentResult(
                    run_id=active_run_id,
                    state=existing.state,
                    steps=(),
                    budget=active_budget.snapshot(),
                    completed=True,
                )

        active_budget.start_step()
        context = StepContext(
            run_id=active_run_id,
            step=active_budget.steps_used,
            budget=active_budget,
            tools=self._tools,
            router=self._router,
        )
        outcome = await self._step.run(state, context)
        active_budget.record_usage(outcome.usage)
        if checkpoint is not None:
            await checkpoint.save(
                Checkpoint(
                    run_id=active_run_id,
                    step=active_budget.steps_used,
                    state=outcome.state,
                )
            )
        return AgentResult(
            run_id=active_run_id,
            state=outcome.state,
            steps=(self._step.name,),
            budget=active_budget.snapshot(),
            usage=outcome.usage,
            completed=outcome.done,
        )
