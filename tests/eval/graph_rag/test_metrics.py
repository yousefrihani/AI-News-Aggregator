"""Offline tests for the Graph RAG evaluation metrics and fixtures."""

from __future__ import annotations

import unittest

from Articles.llm.patterns.graph_rag import parse_citations

from tests.eval.graph_rag.corpus import (
    GOLDEN,
    chunk_ids,
    supporting_article_ids,
)
from tests.eval.graph_rag.metrics import citation_precision, recall_at_k


class TestRecallAtK(unittest.TestCase):
    """Ranked retrieval recall."""

    def test_all_relevant_found(self) -> None:
        self.assertEqual(recall_at_k(["c1", "c2", "c3"], ["c1", "c3"], 3), 1.0)

    def test_partial_recall(self) -> None:
        self.assertAlmostEqual(recall_at_k(["c1", "c2", "c3"], ["c1", "c4"], 3), 0.5)

    def test_cutoff_is_respected(self) -> None:
        self.assertEqual(recall_at_k(["c9", "c8", "c1"], ["c1"], 2), 0.0)
        self.assertEqual(recall_at_k(["c9", "c8", "c1"], ["c1"], 3), 1.0)

    def test_duplicates_do_not_inflate_recall(self) -> None:
        self.assertAlmostEqual(recall_at_k(["c1", "c1", "c2"], ["c1", "c3"], 3), 0.5)

    def test_no_relevant_chunks_is_vacuously_perfect(self) -> None:
        self.assertEqual(recall_at_k(["c1"], [], 1), 1.0)

    def test_invalid_k_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k(["c1"], ["c1"], 0)


class TestCitationPrecision(unittest.TestCase):
    """Cited-versus-supporting document precision."""

    def test_all_citations_supported(self) -> None:
        self.assertEqual(citation_precision(["a1", "a2"], ["a1", "a2"]), 1.0)

    def test_partial_precision(self) -> None:
        self.assertAlmostEqual(citation_precision(["a1", "a9"], ["a1"]), 0.5)

    def test_duplicate_citations_do_not_change_precision(self) -> None:
        self.assertEqual(citation_precision(["a1", "a1"], ["a1"]), 1.0)

    def test_uncited_answer_scores_zero(self) -> None:
        self.assertEqual(citation_precision([], ["a1"]), 0.0)

    def test_metric_detects_an_imprecise_answer(self) -> None:
        golden = GOLDEN[0]
        resolved, unknown = parse_citations(f"{golden.answer} [invented]", {})
        cited = [citation.article_id for citation in resolved] + list(unknown)
        precision = citation_precision(cited, golden.supporting_article_ids)
        self.assertLess(precision, 0.9)
        self.assertAlmostEqual(precision, 0.5)


class TestCorpusIntegrity(unittest.TestCase):
    """Golden questions must reference real chunks, articles, and citations."""

    def test_has_enough_golden_questions(self) -> None:
        self.assertGreaterEqual(len(GOLDEN), 20)

    def test_questions_are_unique_and_non_blank(self) -> None:
        questions = [golden.question for golden in GOLDEN]
        self.assertEqual(len(questions), len(set(questions)))
        self.assertTrue(all(question.strip() for question in questions))

    def test_every_relevant_chunk_exists(self) -> None:
        known = chunk_ids()
        for golden in GOLDEN:
            with self.subTest(question=golden.question):
                self.assertTrue(golden.relevant_chunk_ids)
                self.assertTrue(set(golden.relevant_chunk_ids) <= known)

    def test_every_supporting_article_exists(self) -> None:
        known = supporting_article_ids()
        for golden in GOLDEN:
            with self.subTest(question=golden.question):
                self.assertTrue(golden.supporting_article_ids)
                self.assertTrue(set(golden.supporting_article_ids) <= known)

    def test_every_answer_cites_a_supporting_article(self) -> None:
        for golden in GOLDEN:
            with self.subTest(question=golden.question):
                _, cited = parse_citations(golden.answer, {})
                self.assertTrue(cited)
                self.assertTrue(set(cited) <= set(golden.supporting_article_ids))


if __name__ == "__main__":
    unittest.main()
