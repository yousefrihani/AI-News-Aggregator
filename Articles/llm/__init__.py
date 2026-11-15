"""Provider-agnostic LLM layer.

The public surface is intentionally small: request/response models, typed
errors, route configuration, a provider registry, and the routing facade. No
business logic in this package knows a concrete vendor by name.
"""

from Articles.llm.base import (
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    Message,
    Usage,
)
from Articles.llm.config import (
    LLMConfig,
    ModelConfig,
    ProviderConfig,
    RouteConfig,
    load_llm_config,
)
from Articles.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTransportError,
    LLMUnavailableError,
)
from Articles.llm.registry import ProviderRegistry
from Articles.llm.router import LLMRouter, estimate_tokens

__all__ = [
    "CompletionRequest",
    "CompletionResponse",
    "LLMAuthError",
    "LLMConfig",
    "LLMConfigError",
    "LLMError",
    "LLMHTTPError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMRouter",
    "LLMTransportError",
    "LLMUnavailableError",
    "Message",
    "ModelConfig",
    "ProviderConfig",
    "ProviderRegistry",
    "RouteConfig",
    "Usage",
    "estimate_tokens",
    "load_llm_config",
]
