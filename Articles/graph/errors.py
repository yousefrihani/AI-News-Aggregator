"""Typed exceptions raised by the Graph RAG layer.

Every graph failure derives from :class:`GraphError`, allowing callers to
isolate storage problems from unrelated application errors. Configuration
failures raised here never include credentials.
"""

__all__ = [
    "EmbeddingDimensionError",
    "EmbeddingError",
    "GraphConfigError",
    "GraphConnectionError",
    "GraphError",
    "GraphQueryError",
]


class GraphError(Exception):
    """Base class for every Graph RAG failure.

    Args:
        message: Human-readable description of the failure. Must not contain
            credentials or other secrets.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """Return the failure message."""
        return self.message


class GraphConfigError(GraphError):
    """Raised when graph or embedding configuration is invalid."""


class GraphConnectionError(GraphError):
    """Raised when a graph database connection cannot be established."""


class GraphQueryError(GraphError):
    """Raised when a graph query fails or returns an unusable payload."""


class EmbeddingError(GraphError):
    """Base class for embedding provider failures."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when an embedding provider returns an unexpected vector size."""
