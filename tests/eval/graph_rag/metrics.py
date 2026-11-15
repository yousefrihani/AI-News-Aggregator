"""Retrieval and citation metrics for the Graph RAG evaluation harness.

The metrics are pure functions so they can be unit-tested in the offline suite
without loading a model, opening a socket, or calling a provider.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["citation_precision", "recall_at_k"]


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Compute retrieval recall over the first ``k`` retrieved identifiers.

    Args:
        retrieved_ids: Retrieved chunk identifiers in ranked order.
        relevant_ids: Chunk identifiers that contain the answer.
        k: Cut-off rank. The first ``k`` entries of ``retrieved_ids`` are
            considered.

    Returns:
        The fraction of relevant chunks found in the top ``k``. Returns
        ``1.0`` when there are no relevant chunks.

    Raises:
        ValueError: If ``k`` is less than 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant) / len(relevant)


def citation_precision(
    cited_ids: Sequence[str],
    supporting_ids: Sequence[str],
) -> float:
    """Compute the fraction of cited documents that support the answer.

    Args:
        cited_ids: Documents cited by the answer, including hallucinated
            identifiers that do not appear in the retrieved context.
        supporting_ids: Documents that actually support the answer.

    Returns:
        ``len(cited & supporting) / len(cited)``. An answer with no citations
        scores ``0.0`` because it makes unsupported claims.
    """
    cited = set(cited_ids)
    if not cited:
        return 0.0
    return len(cited & set(supporting_ids)) / len(cited)
