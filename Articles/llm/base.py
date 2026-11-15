"""Immutable request and response models shared by every LLM adapter.

The models mirror :mod:`Articles.Models`: standard library only, frozen
dataclasses, and validation that fails at construction rather than halfway
through a provider call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "CompletionRequest",
    "CompletionResponse",
    "LLMProvider",
    "Message",
    "MessageRole",
    "ResponseFormat",
    "Usage",
]

MessageRole = Literal["system", "user", "assistant", "tool"]
ResponseFormat = Literal["text", "json_object", "json_schema"]

_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})
_FORMATS: frozenset[str] = frozenset({"text", "json_object", "json_schema"})


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message.

    Args:
        role: Conversation role.
        content: Message text.

    Raises:
        ValueError: If ``role`` is unknown or ``content`` is not a string.
    """

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        """Validate the role and content type.

        Raises:
            ValueError: If the role is unknown or content is not a string.
        """
        if self.role not in _ROLES:
            raise ValueError(f"unknown message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError("message content must be a string")


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """A provider-independent completion request.

    Args:
        messages: Ordered conversation messages. Must not be empty.
        model: Concrete provider model id, never a route alias.
        temperature: Sampling temperature, at least zero.
        max_tokens: Optional completion token limit.
        response_format: Expected response shape.
        json_schema: Schema used when ``response_format`` is ``json_schema``.
        metadata: Optional caller metadata. Adapters may ignore it.

    Raises:
        ValueError: If a field violates its documented constraint.
    """

    messages: tuple[Message, ...]
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None
    response_format: ResponseFormat = "text"
    json_schema: Mapping[str, Any] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize sequences and validate request invariants.

        Raises:
            ValueError: If messages are missing, the model is blank, the
                temperature is negative, the token limit is not positive, the
                response format is unknown, or a JSON schema is missing.
        """
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("messages must contain at least one message")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1 when provided")
        if self.response_format not in _FORMATS:
            raise ValueError(f"unknown response_format: {self.response_format!r}")
        if self.response_format == "json_schema" and self.json_schema is None:
            raise ValueError("json_schema is required when response_format is json_schema")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage reported by a provider.

    Args:
        prompt_tokens: Input tokens consumed.
        completion_tokens: Output tokens generated.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        """Reject negative counters.

        Raises:
            ValueError: If either counter is negative.
        """
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counters must be non-negative")

    @property
    def total_tokens(self) -> int:
        """Return the sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """A normalized provider completion.

    Args:
        text: Assistant response text.
        provider: Provider name that served the request.
        model: Concrete model id that served the request.
        usage: Reported token usage.
        finish_reason: Provider finish reason, when supplied.
        raw: Original provider payload, excluded from equality and repr.
    """

    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate required response fields.

        Raises:
            ValueError: If text, provider, or model is not a string.
        """
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")


@runtime_checkable
class LLMProvider(Protocol):
    """Structural contract for a chat-completion provider."""

    name: str

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute one completion request.

        Args:
            request: Provider-independent request.

        Returns:
            A normalized completion response.
        """
        ...

    async def aclose(self) -> None:
        """Release provider-owned resources."""
        ...
