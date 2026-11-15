"""Embedding provider port and reference implementations.

``DeterministicEmbedding`` is a dependency-free hashed embedder used by tests
and offline development. ``FastEmbedEmbedding`` wraps the optional ``fastembed``
package and never downloads a model at import time or in the constructor.
"""

import asyncio
import hashlib
import importlib
import math
from collections.abc import Sequence
from typing import Any, Protocol

from Articles.graph.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    GraphConfigError,
)

__all__ = [
    "DeterministicEmbedding",
    "EmbeddingProvider",
    "FastEmbedEmbedding",
]

_INSTALL_HINT = (
    "fastembed is not installed; install local embeddings with "
    "`pip install ai-news-aggregator[embeddings-local]`"
)


class EmbeddingProvider(Protocol):
    """Asynchronous embedding provider port."""

    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed; an empty sequence returns an empty list.

        Returns:
            One vector per input text, each of length :attr:`dim`.
        """

    async def aclose(self) -> None:
        """Release any resources owned by the provider."""


def _normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalize a vector, leaving an all-zero vector unchanged.

    Args:
        vector: Input values.

    Returns:
        A new normalized list of floats.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _import_text_embedding() -> Any:
    """Import ``fastembed.TextEmbedding`` lazily.

    Returns:
        The ``TextEmbedding`` class.

    Raises:
        GraphConfigError: If the optional package is missing.
    """
    try:
        module = importlib.import_module("fastembed")
    except ImportError as err:
        raise GraphConfigError(_INSTALL_HINT) from err
    try:
        return module.TextEmbedding
    except AttributeError as err:
        raise GraphConfigError(_INSTALL_HINT) from err


class DeterministicEmbedding:
    """Stable SHA-256-based embeddings for tests and offline development.

    Vectors depend only on the input text and dimension, so they are identical
    across calls, processes, and machines. No model is loaded and no network
    access is performed.

    Args:
        dim: Vector dimension. Must be greater than zero.

    Raises:
        GraphConfigError: If ``dim`` is not a positive integer.
    """

    def __init__(self, dim: int = 384) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise GraphConfigError("embedding dim must be an integer")
        if dim <= 0:
            raise GraphConfigError("embedding dim must be greater than 0")
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        """Build one deterministic normalized vector.

        Args:
            text: Input text.

        Returns:
            A normalized vector of length :attr:`dim`.
        """
        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{text}\x1f{counter}".encode()).digest()
            for offset in range(0, len(digest), 4):
                if len(values) >= self.dim:
                    break
                word = int.from_bytes(digest[offset : offset + 4], "big")
                values.append((word / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        return _normalize(values)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts deterministically.

        Args:
            texts: Texts to embed.

        Returns:
            One vector per input text.
        """
        return [self._vector(text) for text in texts]

    async def aclose(self) -> None:
        """Release resources; this provider owns none."""


class FastEmbedEmbedding:
    """Local embeddings backed by the optional ``fastembed`` package.

    CPU-bound work runs in a worker thread. The model itself is instantiated on
    first use, so constructing the provider never downloads weights.

    Args:
        model_name: fastembed model identifier.
        dim: Expected vector dimension. Must match the graph vector index.
        cache_dir: Optional model cache directory.

    Raises:
        GraphConfigError: If the dimension is invalid or ``fastembed`` is not
            installed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dim: int = 384,
        cache_dir: str | None = None,
    ) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise GraphConfigError("embedding dim must be an integer")
        if dim <= 0:
            raise GraphConfigError("embedding dim must be greater than 0")
        if not model_name.strip():
            raise GraphConfigError("embedding model name must not be empty")
        self.model_name = model_name
        self.dim = dim
        self.cache_dir = cache_dir
        self._text_embedding_cls = _import_text_embedding()
        self._embedder: Any = None

    def _ensure_embedder(self) -> Any:
        """Create the fastembed model on first use.

        Returns:
            A ``TextEmbedding`` instance.
        """
        if self._embedder is None:
            kwargs: dict[str, Any] = {}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            self._embedder = self._text_embedding_cls(model_name=self.model_name, **kwargs)
        return self._embedder

    def _embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        """Run fastembed synchronously and validate/normalize the output.

        Args:
            texts: Texts to embed.

        Returns:
            One normalized vector per input text.

        Raises:
            EmbeddingDimensionError: If a returned vector has the wrong length.
        """
        embedder = self._ensure_embedder()
        vectors: list[list[float]] = []
        for index, raw in enumerate(embedder.embed(list(texts))):
            vector = [float(value) for value in raw]
            if len(vector) != self.dim:
                raise EmbeddingDimensionError(
                    f"embedding {index} has dimension {len(vector)}, expected {self.dim}"
                )
            if any(not math.isfinite(value) for value in vector):
                raise EmbeddingError(f"embedding {index} contains a non-finite value")
            vectors.append(_normalize(vector))
        return vectors

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in a worker thread.

        Args:
            texts: Texts to embed; an empty sequence returns an empty list.

        Returns:
            One normalized vector per input text.

        Raises:
            EmbeddingDimensionError: If the model returns an unexpected size.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    async def aclose(self) -> None:
        """Release resources; fastembed owns no closeable handle."""
