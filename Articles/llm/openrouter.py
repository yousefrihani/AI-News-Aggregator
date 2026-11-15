"""OpenRouter adapter.

OpenRouter exposes an OpenAI-compatible chat-completions API and asks clients to
identify themselves with two optional attribution headers.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx

from Articles.llm.config import ProviderConfig
from Articles.llm.openai_compat import OpenAICompatibleProvider

__all__ = ["OpenRouterProvider"]


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter chat-completion provider.

    Args:
        config: Provider configuration containing models and optional headers.
            When omitted, the built-in OpenRouter endpoint is used.
        api_key: OpenRouter API key.
        headers: Extra headers merged over the attribution defaults and the
            configuration headers.
        client: Optional pre-configured async client, mainly for tests.
        timeout: Default request timeout in seconds.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_HEADERS: Mapping[str, str] = {
        "HTTP-Referer": "https://github.com/yousefrihani/AI-News-Aggregator",
        "X-Title": "AI-News-Aggregator",
    }

    def __init__(
        self,
        config: ProviderConfig | None = None,
        api_key: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        base_url = self.DEFAULT_BASE_URL
        merged = dict(self.DEFAULT_HEADERS)
        if config is not None:
            base_url = config.base_url.strip() or base_url
            merged.update(config.headers)
        merged.update(headers or {})
        super().__init__(
            "openrouter",
            base_url,
            api_key,
            headers=merged,
            client=client,
            timeout=timeout,
        )
