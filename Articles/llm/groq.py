"""Groq adapter.

Groq exposes an OpenAI-compatible chat-completions API, so the adapter only
supplies the default endpoint and provider name.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx

from Articles.llm.config import ProviderConfig
from Articles.llm.openai_compat import OpenAICompatibleProvider

__all__ = ["GroqProvider"]


class GroqProvider(OpenAICompatibleProvider):
    """Groq chat-completion provider.

    Args:
        config: Provider configuration containing models and optional headers.
            When omitted, the built-in Groq endpoint is used.
        api_key: Groq API key.
        headers: Extra headers merged over the configuration headers.
        client: Optional pre-configured async client, mainly for tests.
        timeout: Default request timeout in seconds.
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

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
        merged: dict[str, str] = {}
        if config is not None:
            base_url = config.base_url.strip() or base_url
            merged.update(config.headers)
        merged.update(headers or {})
        super().__init__(
            "groq",
            base_url,
            api_key,
            headers=merged,
            client=client,
            timeout=timeout,
        )
