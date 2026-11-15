"""Environment-driven configuration for the Graph RAG layer.

Settings are immutable, validated at construction time, and read from the
``GRAPH_*``/``NEO4J_*``/``EMBEDDING_*`` environment variables documented in the
project plan. Invalid values raise :class:`GraphConfigError` so deployments fail
at startup rather than at the first query.
"""

import os
from dataclasses import dataclass
from typing import Literal, cast

from Articles.graph.errors import GraphConfigError

__all__ = ["EmbeddingSettings", "GraphSettings"]

Backend = Literal["memory", "neo4j"]
EmbeddingProviderName = Literal["fastembed", "deterministic"]


def _env(name: str, default: str) -> str:
    """Return a stripped environment value or ``default`` when unset/blank.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or blank.

    Returns:
        The configured value.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _optional_env(name: str) -> str | None:
    """Return a stripped environment value or ``None`` when unset/blank.

    Args:
        name: Environment variable name.

    Returns:
        The configured value or ``None``.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


@dataclass(frozen=True, slots=True)
class GraphSettings:
    """Graph backend configuration.

    Args:
        backend: ``"memory"`` or ``"neo4j"``.
        neo4j_uri: Bolt/Neo4j connection URI.
        neo4j_user: Database user.
        neo4j_password: Database password. Required for the Neo4j backend.
        neo4j_database: Target database name.

    Raises:
        GraphConfigError: If the backend is unknown or a Neo4j password is
            missing.
    """

    backend: Backend = "memory"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"

    def __post_init__(self) -> None:
        """Validate the backend selection and required credentials.

        Raises:
            GraphConfigError: If the backend is unsupported or the Neo4j
                backend is selected without a password.
        """
        if self.backend not in ("memory", "neo4j"):
            raise GraphConfigError(
                f"GRAPH_BACKEND must be 'memory' or 'neo4j', got {self.backend!r}"
            )
        if not self.neo4j_uri.strip():
            raise GraphConfigError("NEO4J_URI must not be empty")
        if not self.neo4j_user.strip():
            raise GraphConfigError("NEO4J_USER must not be empty")
        if not self.neo4j_database.strip():
            raise GraphConfigError("NEO4J_DATABASE must not be empty")
        if self.backend == "neo4j" and not (self.neo4j_password or "").strip():
            raise GraphConfigError("NEO4J_PASSWORD is required when GRAPH_BACKEND=neo4j")

    @classmethod
    def from_env(cls) -> "GraphSettings":
        """Build settings from ``GRAPH_*``/``NEO4J_*`` environment variables.

        Returns:
            A validated :class:`GraphSettings` instance.

        Raises:
            GraphConfigError: If a configured value is invalid.
        """
        backend = cast(Backend, _env("GRAPH_BACKEND", "memory").lower())
        return cls(
            backend=backend,
            neo4j_uri=_env("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=_env("NEO4J_USER", "neo4j"),
            neo4j_password=_optional_env("NEO4J_PASSWORD"),
            neo4j_database=_env("NEO4J_DATABASE", "neo4j"),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """Embedding provider configuration.

    Args:
        provider: ``"fastembed"`` or ``"deterministic"``.
        model: Model identifier for the local embedding provider.
        dim: Vector dimension, which must match the graph vector index.

    Raises:
        GraphConfigError: If the provider is unsupported or the dimension is
            not a positive integer.
    """

    provider: EmbeddingProviderName = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384

    def __post_init__(self) -> None:
        """Validate provider, model, and dimension.

        Raises:
            GraphConfigError: If any value violates its documented contract.
        """
        if self.provider not in ("fastembed", "deterministic"):
            raise GraphConfigError(
                f"EMBEDDING_PROVIDER must be 'fastembed' or 'deterministic', got {self.provider!r}"
            )
        if not self.model.strip():
            raise GraphConfigError("EMBEDDING_MODEL must not be empty")
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise GraphConfigError("EMBEDDING_DIM must be an integer")
        if self.dim <= 0:
            raise GraphConfigError("EMBEDDING_DIM must be greater than 0")

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        """Build settings from ``EMBEDDING_*`` environment variables.

        Returns:
            A validated :class:`EmbeddingSettings` instance.

        Raises:
            GraphConfigError: If the dimension is not a positive integer or the
                provider is unsupported.
        """
        raw_dim = os.getenv("EMBEDDING_DIM")
        dim = 384
        if raw_dim is not None and raw_dim.strip():
            try:
                dim = int(raw_dim)
            except ValueError as err:
                raise GraphConfigError(
                    f"EMBEDDING_DIM must be an integer, got {raw_dim!r}"
                ) from err
        provider = cast(
            EmbeddingProviderName,
            _env("EMBEDDING_PROVIDER", "fastembed").lower(),
        )
        return cls(
            provider=provider,
            model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            dim=dim,
        )
