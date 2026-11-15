"""Configuration models and YAML loading for the LLM layer.

Routes are the stable contract between patterns/agents and providers. A route
lists ordered ``provider/model-alias`` candidates plus policy such as retries,
token limits, and response format. Configuration is validated once at load
time so malformed routes fail at startup, never halfway through a batch.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

from Articles.llm.base import ResponseFormat
from Articles.llm.errors import LLMConfigError

__all__ = [
    "LLMConfig",
    "ModelConfig",
    "ProviderConfig",
    "RouteConfig",
    "load_llm_config",
]

_VALID_FORMATS: frozenset[str] = frozenset({"text", "json_object", "json_schema"})
_BUILTIN_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return ``value`` as a string-keyed mapping.

    Args:
        value: Raw configuration value.
        name: Field name used in error messages.

    Returns:
        The validated mapping.

    Raises:
        LLMConfigError: If the value is not a mapping with string keys.
    """
    if not isinstance(value, Mapping):
        raise LLMConfigError(f"{name} must be a mapping")
    for key in value:
        if not isinstance(key, str) or not key.strip():
            raise LLMConfigError(f"{name} keys must be non-empty strings")
    return value


def _require_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    """Return ``value`` as a string with optional emptiness.

    Args:
        value: Raw configuration value.
        name: Field name used in error messages.
        allow_empty: Whether an empty or blank string is accepted.

    Returns:
        The validated string.

    Raises:
        LLMConfigError: If the value is not a string or is blank.
    """
    if not isinstance(value, str):
        raise LLMConfigError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise LLMConfigError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 1) -> int:
    """Return ``value`` as an integer at or above ``minimum``.

    Args:
        value: Raw configuration value.
        name: Field name used in error messages.
        minimum: Smallest accepted value.

    Returns:
        The validated integer.

    Raises:
        LLMConfigError: If the value is not an integer or is too small.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMConfigError(f"{name} must be an integer")
    if value < minimum:
        raise LLMConfigError(f"{name} must be at least {minimum}")
    return value


def _require_float(value: Any, name: str, *, minimum: float = 0.0) -> float:
    """Return ``value`` as a float at or above ``minimum``.

    Args:
        value: Raw configuration value.
        name: Field name used in error messages.
        minimum: Smallest accepted value.

    Returns:
        The validated float.

    Raises:
        LLMConfigError: If the value is not numeric or is too small.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMConfigError(f"{name} must be a number")
    number = float(value)
    if number < minimum:
        raise LLMConfigError(f"{name} must be at least {minimum}")
    return number


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A concrete model exposed by a provider.

    Args:
        id: Provider model identifier sent over the wire.
        context_window: Maximum context size in tokens.
        supports: Optional protocol features such as ``json_object``.
    """

    id: str
    context_window: int = 128_000
    supports: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate the model definition.

        Raises:
            LLMConfigError: If the id is blank or the context window is invalid.
        """
        if not isinstance(self.id, str) or not self.id.strip():
            raise LLMConfigError("model id must be a non-empty string")
        if self.context_window < 1:
            raise LLMConfigError(f"model {self.id!r} context_window must be at least 1")
        if not isinstance(self.supports, frozenset):
            object.__setattr__(self, "supports", frozenset(self.supports))


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Connection settings for one LLM provider.

    Args:
        name: Provider name used in route candidates.
        kind: Adapter kind, for example ``openai_compatible``.
        base_url: API root. Vendor adapters may supply a default.
        api_key_env: Environment variable holding the API key.
        headers: Extra request headers. Values are not secrets.
        models: Model aliases exposed by this provider.
    """

    name: str
    kind: str = "openai_compatible"
    base_url: str = ""
    api_key_env: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    models: Mapping[str, ModelConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize mappings and validate provider invariants.

        Raises:
            LLMConfigError: If a required field is malformed.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise LLMConfigError("provider name must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise LLMConfigError(f"provider {self.name!r} kind must be non-empty")
        if not isinstance(self.base_url, str):
            raise LLMConfigError(f"provider {self.name!r} base_url must be a string")
        if self.api_key_env is not None and not isinstance(self.api_key_env, str):
            raise LLMConfigError(f"provider {self.name!r} api_key_env must be a string")
        clean_headers = {
            _require_str(key, "header name"): _require_str(
                value, f"header {key!r}", allow_empty=True
            )
            for key, value in _require_mapping(self.headers, "headers").items()
        }
        object.__setattr__(self, "headers", MappingProxyType(clean_headers))
        models = {
            _require_str(alias, "model alias"): model
            for alias, model in _require_mapping(
                self.models, f"provider {self.name!r} models"
            ).items()
        }
        for alias, model in models.items():
            if not isinstance(model, ModelConfig):
                raise LLMConfigError(f"model {alias!r} must be a ModelConfig")
        if not models:
            raise LLMConfigError(f"provider {self.name!r} must define at least one model")
        object.__setattr__(self, "models", MappingProxyType(models))

    def base_url_or_default(self) -> str:
        """Return the configured base URL, falling back to a vendor default.

        Returns:
            A non-empty base URL.

        Raises:
            LLMConfigError: If no base URL is configured and the provider has
                no built-in default.
        """
        if self.base_url.strip():
            return self.base_url.rstrip("/")
        default = _BUILTIN_BASE_URLS.get(self.name)
        if default is None:
            raise LLMConfigError(f"provider {self.name!r} requires base_url for kind {self.kind!r}")
        return default


@dataclass(frozen=True, slots=True)
class RouteConfig:
    """Ordered provider candidates and policy for one logical task.

    Args:
        name: Route name used by patterns and agents.
        candidates: Ordered ``provider/model-alias`` candidates.
        temperature: Sampling temperature for this route.
        max_tokens: Optional completion token limit.
        max_input_tokens: Optional estimated input-token limit.
        response_format: Expected response shape.
        schema_ref: Schema name required for ``json_schema`` routes.
        retries: Extra attempts per candidate after the first failure.
        timeout_s: Per-attempt timeout in seconds.
        daily_token_budget: Optional token budget across the router lifetime.
    """

    name: str
    candidates: tuple[str, ...]
    temperature: float = 0.2
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    response_format: ResponseFormat = "text"
    schema_ref: str | None = None
    retries: int = 0
    timeout_s: float = 30.0
    daily_token_budget: int | None = None

    def __post_init__(self) -> None:
        """Normalize candidates and validate route invariants.

        Raises:
            LLMConfigError: If a field violates its documented constraint.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise LLMConfigError("route name must be a non-empty string")
        if not isinstance(self.candidates, tuple):
            object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates:
            raise LLMConfigError(f"route {self.name!r} must define at least one candidate")
        for candidate in self.candidates:
            if not isinstance(candidate, str) or "/" not in candidate:
                raise LLMConfigError(
                    f"route {self.name!r} candidate {candidate!r} must be provider/model"
                )
        if self.temperature < 0:
            raise LLMConfigError(f"route {self.name!r} temperature must be non-negative")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise LLMConfigError(f"route {self.name!r} max_tokens must be at least 1")
        if self.max_input_tokens is not None and self.max_input_tokens < 1:
            raise LLMConfigError(f"route {self.name!r} max_input_tokens must be at least 1")
        if self.response_format not in _VALID_FORMATS:
            raise LLMConfigError(
                f"route {self.name!r} has unknown response_format {self.response_format!r}"
            )
        if self.response_format == "json_schema" and not self.schema_ref:
            raise LLMConfigError(
                f"route {self.name!r} requires schema_ref for json_schema responses"
            )
        if self.retries < 0:
            raise LLMConfigError(f"route {self.name!r} retries must be non-negative")
        if self.timeout_s <= 0:
            raise LLMConfigError(f"route {self.name!r} timeout_s must be greater than 0")
        if self.daily_token_budget is not None and self.daily_token_budget < 1:
            raise LLMConfigError(f"route {self.name!r} daily_token_budget must be at least 1")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Validated provider and route configuration.

    Args:
        version: Schema version. Only version ``1`` is supported.
        providers: Provider configurations keyed by name.
        routes: Route configurations keyed by name.
        defaults: Reserved scalar defaults for future releases.
        source_path: Path the configuration was loaded from, when known.
    """

    version: int = 1
    providers: Mapping[str, ProviderConfig] = field(default_factory=dict)
    routes: Mapping[str, RouteConfig] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        """Normalize mappings and validate the whole configuration.

        Raises:
            LLMConfigError: If the configuration is internally inconsistent.
        """
        object.__setattr__(
            self,
            "providers",
            MappingProxyType(
                {
                    _require_str(name, "provider name"): provider
                    for name, provider in _require_mapping(self.providers, "providers").items()
                }
            ),
        )
        object.__setattr__(
            self,
            "routes",
            MappingProxyType(
                {
                    _require_str(name, "route name"): route
                    for name, route in _require_mapping(self.routes, "routes").items()
                }
            ),
        )
        object.__setattr__(
            self,
            "defaults",
            MappingProxyType(dict(_require_mapping(self.defaults, "defaults"))),
        )
        self._validate()

    def _validate(self) -> None:
        """Validate cross-references between providers, models, and routes.

        Raises:
            LLMConfigError: If a route points at an unknown provider/model, or a
                model cannot produce the route's response format.
        """
        if self.version != 1:
            raise LLMConfigError(f"unsupported LLM config version: {self.version}")
        if not self.providers:
            raise LLMConfigError("at least one provider must be configured")
        if not self.routes:
            raise LLMConfigError("at least one route must be configured")
        for name, provider in self.providers.items():
            if not isinstance(provider, ProviderConfig):
                raise LLMConfigError(f"provider {name!r} must be a ProviderConfig")
            if provider.name != name:
                raise LLMConfigError(
                    f"provider key {name!r} does not match provider name {provider.name!r}"
                )
        for name, route in self.routes.items():
            if not isinstance(route, RouteConfig):
                raise LLMConfigError(f"route {name!r} must be a RouteConfig")
            if route.name != name:
                raise LLMConfigError(f"route key {name!r} does not match route name {route.name!r}")
            for candidate in route.candidates:
                provider_name, _, alias = candidate.partition("/")
                provider_config = self.providers.get(provider_name)
                if provider_config is None:
                    raise LLMConfigError(
                        f"route {name!r} references unknown provider {provider_name!r}"
                    )
                model = provider_config.models.get(alias)
                if model is None:
                    raise LLMConfigError(
                        f"route {name!r} references unknown model alias "
                        f"{alias!r} on provider {provider_name!r}"
                    )
                if route.response_format != "text" and route.response_format not in model.supports:
                    raise LLMConfigError(
                        f"route {name!r} requires {route.response_format!r}, but "
                        f"{provider_name}/{alias} does not declare support"
                    )

    @classmethod
    def from_env(cls, path: str | Path | None = None) -> LLMConfig:
        """Load configuration from the environment or an explicit path.

        Args:
            path: Optional explicit path that overrides
                ``AI_NEWS_LLM_ROUTES_FILE``.

        Returns:
            A validated configuration.
        """
        return load_llm_config(path if path is not None else _default_config_path())

    def provider_for_candidate(self, candidate: str) -> tuple[ProviderConfig, ModelConfig]:
        """Resolve one ``provider/model-alias`` candidate.

        Args:
            candidate: Candidate string from a route.

        Returns:
            The provider and model configuration.

        Raises:
            LLMConfigError: If the candidate is malformed or unknown.
        """
        provider_name, separator, alias = candidate.partition("/")
        if not separator or not provider_name or not alias:
            raise LLMConfigError(f"invalid route candidate: {candidate!r}")
        provider = self.providers.get(provider_name)
        if provider is None:
            raise LLMConfigError(f"unknown provider: {provider_name!r}")
        model = provider.models.get(alias)
        if model is None:
            raise LLMConfigError(f"unknown model alias {alias!r} on provider {provider_name!r}")
        return provider, model


def _load_model(alias: str, raw: Any, provider: str) -> ModelConfig:
    """Build one :class:`ModelConfig` from a YAML mapping.

    Args:
        alias: Model alias under the provider.
        raw: Raw YAML value.
        provider: Provider name, used in error messages.

    Returns:
        A validated model configuration.

    Raises:
        LLMConfigError: If the model definition is malformed.
    """
    body = _require_mapping(raw, f"provider {provider!r} model {alias!r}")
    model_id = _require_str(body.get("id"), f"model {alias!r} id")
    context_window = _require_int(
        body.get("context_window", 128_000), f"model {alias!r} context_window"
    )
    supports_raw = body.get("supports", [])
    if isinstance(supports_raw, str):
        supports = frozenset({supports_raw})
    elif isinstance(supports_raw, (list, tuple, set, frozenset)):
        supports = frozenset(
            _require_str(item, f"model {alias!r} support") for item in supports_raw
        )
    else:
        raise LLMConfigError(f"model {alias!r} supports must be a list")
    return ModelConfig(id=model_id, context_window=context_window, supports=supports)


def _load_provider(name: str, raw: Any) -> ProviderConfig:
    """Build one :class:`ProviderConfig` from a YAML mapping.

    Args:
        name: Provider name.
        raw: Raw YAML value.

    Returns:
        A validated provider configuration.

    Raises:
        LLMConfigError: If the provider definition is malformed.
    """
    body = _require_mapping(raw, f"provider {name!r}")
    headers_raw = body.get("headers", {})
    headers = {
        _require_str(key, "header name"): _require_str(value, f"header {key!r}", allow_empty=True)
        for key, value in _require_mapping(headers_raw, f"provider {name!r} headers").items()
    }
    models_raw = _require_mapping(body.get("models"), f"provider {name!r} models")
    models = {alias: _load_model(alias, value, name) for alias, value in models_raw.items()}
    api_key_env = body.get("api_key_env")
    if api_key_env is not None:
        api_key_env = _require_str(api_key_env, f"provider {name!r} api_key_env")
    return ProviderConfig(
        name=name,
        kind=_require_str(body.get("kind", "openai_compatible"), f"provider {name!r} kind"),
        base_url=_require_str(
            body.get("base_url", ""), f"provider {name!r} base_url", allow_empty=True
        ),
        api_key_env=api_key_env,
        headers=headers,
        models=models,
    )


def _load_route(name: str, raw: Any) -> RouteConfig:
    """Build one :class:`RouteConfig` from a YAML mapping.

    Args:
        name: Route name.
        raw: Raw YAML value.

    Returns:
        A validated route configuration.

    Raises:
        LLMConfigError: If the route definition is malformed.
    """
    body = _require_mapping(raw, f"route {name!r}")
    candidates_raw = body.get("candidates")
    if not isinstance(candidates_raw, (list, tuple)):
        raise LLMConfigError(f"route {name!r} candidates must be a list")
    candidates = tuple(
        _require_str(candidate, f"route {name!r} candidate") for candidate in candidates_raw
    )
    response_format = body.get("response_format", "text")
    if not isinstance(response_format, str) or response_format not in _VALID_FORMATS:
        raise LLMConfigError(f"route {name!r} has invalid response_format")
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        max_tokens = _require_int(max_tokens, f"route {name!r} max_tokens")
    max_input_tokens = body.get("max_input_tokens")
    if max_input_tokens is not None:
        max_input_tokens = _require_int(max_input_tokens, f"route {name!r} max_input_tokens")
    daily_token_budget = body.get("daily_token_budget")
    if daily_token_budget is not None:
        daily_token_budget = _require_int(daily_token_budget, f"route {name!r} daily_token_budget")
    schema_ref = body.get("schema_ref")
    if schema_ref is not None:
        schema_ref = _require_str(schema_ref, f"route {name!r} schema_ref")
    return RouteConfig(
        name=name,
        candidates=candidates,
        temperature=_require_float(body.get("temperature", 0.2), f"route {name!r} temperature"),
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        response_format=cast(ResponseFormat, response_format),
        schema_ref=schema_ref,
        retries=_require_int(body.get("retries", 0), f"route {name!r} retries", minimum=0),
        timeout_s=_require_float(
            body.get("timeout_s", 30.0), f"route {name!r} timeout_s", minimum=0.001
        ),
        daily_token_budget=daily_token_budget,
    )


def load_llm_config(path: str | Path) -> LLMConfig:
    """Load and validate a YAML route configuration.

    Args:
        path: Path to the YAML file.

    Returns:
        A validated :class:`LLMConfig`.

    Raises:
        LLMConfigError: If the file is missing, unreadable, not valid YAML, or
            fails validation.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise LLMConfigError(f"LLM route config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as err:
        raise LLMConfigError(f"could not read LLM route config {config_path}: {err}") from err
    except yaml.YAMLError as err:
        raise LLMConfigError(f"invalid YAML in {config_path}: {err}") from err
    if raw is None:
        raw = {}
    body = _require_mapping(raw, "LLM config")
    providers_raw = _require_mapping(body.get("providers", {}), "providers")
    routes_raw = _require_mapping(body.get("routes", {}), "routes")
    defaults_raw = body.get("defaults", {})
    return LLMConfig(
        version=_require_int(body.get("version", 1), "version"),
        providers={
            _require_str(name, "provider name"): _load_provider(name, value)
            for name, value in providers_raw.items()
        },
        routes={
            _require_str(name, "route name"): _load_route(name, value)
            for name, value in routes_raw.items()
        },
        defaults=dict(_require_mapping(defaults_raw, "defaults")),
        source_path=config_path,
    )


def _default_config_path() -> Path:
    """Return the configured route file path.

    Returns:
        ``AI_NEWS_LLM_ROUTES_FILE`` when set, otherwise ``config/llm_routes.yaml``.
    """
    return Path(os.getenv("AI_NEWS_LLM_ROUTES_FILE", "config/llm_routes.yaml"))
