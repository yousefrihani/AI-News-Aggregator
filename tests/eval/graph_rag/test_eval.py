"""Opt-in Graph RAG evaluation over the fixed golden corpus.

Run with ``RUN_EVAL=1 python -m unittest tests.eval.graph_rag.test_eval``.
The harness uses deterministic bag-of-words embeddings and recorded answers, so
it needs no API key and no model download. Metrics are printed for the CI log
and the quality gates from the roadmap are enforced as assertions.
"""

from __future__ import annotations

import os
import unittest

from Articles.graph.retriever import GraphRAGRetriever
from Articles.llm.patterns.graph_rag import parse_citations

from tests.eval.graph_rag.corpus import GOLDEN, BagOfWordsEmbedding, build_store
from tests.eval.graph_rag.metrics import citation_precision, recall_at_k

RUN_EVAL = os.getenv("RUN_EVAL", "0").strip() == "1"
RECALL_K = 8
MIN_RECALL = 0.8
MIN_CITATION_PRECISION = 0.9


@unittest.skipUnless(RUN_EVAL, "set RUN_EVAL=1 to run the Graph RAG evaluation")
class TestGraphRAGEval(unittest.IsolatedAsyncioTestCase):
    """Golden-question retrieval and citation quality gates."""

    async def asyncSetUp(self) -> None:
        self.embedder = BagOfWordsEmbedding()
        self.store = await build_store(self.embedder)
        self.retriever = GraphRAGRetriever(self.store, self.embedder, k=RECALL_K)

    async def asyncTearDown(self) -> None:
        await self.store.aclose()

    async def test_golden_set_meets_quality_gate(self) -> None:
        recalls: list[float] = []
        precisions: list[float] = []
        rows: list[str] = []
        for golden in GOLDEN:
            context = await self.retriever.retrieve(golden.question)
            retrieved = [chunk.id for chunk in context.chunks]
            recall = recall_at_k(retrieved, golden.relevant_chunk_ids, RECALL_K)
            resolved, unknown = parse_citations(golden.answer, context.citations)
            cited = [citation.article_id for citation in resolved] + list(unknown)
            precision = citation_precision(cited, golden.supporting_article_ids)
            recalls.append(recall)
            precisions.append(precision)
            rows.append(
                f"recall@k={recall:.2f} precision={precision:.2f} "
                f"cited={','.join(cited) or '-'} | {golden.question}"
            )

        mean_recall = sum(recalls) / len(recalls)
        mean_precision = sum(precisions) / len(precisions)
        print(
            "\nGraph RAG evaluation\n"
            + "\n".join(rows)
            + f"\nrecall@{RECALL_K}={mean_recall:.3f} "
            + f"citation_precision={mean_precision:.3f} "
            + f"questions={len(GOLDEN)}"
        )
        self.assertGreaterEqual(mean_recall, MIN_RECALL)
        self.assertGreaterEqual(mean_precision, MIN_CITATION_PRECISION)


if __name__ == "__main__":
    unittest.main()
