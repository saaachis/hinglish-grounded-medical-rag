import numpy as np
import pytest

from src.evaluation.retrieval_metrics import (
    ndcg_binary, ndcg_graded, rank_metrics, reciprocal_rank, success_at_k,
)


def test_success_and_mrr():
    hits = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 1]], dtype=bool)
    assert success_at_k(hits, 1).tolist() == [0, 0, 1]
    assert success_at_k(hits, 2).tolist() == [1, 0, 1]
    assert reciprocal_rank(hits).tolist() == [0.5, 0.0, 1.0]


def test_ndcg_single_relevant_matches_old_formula():
    # With exactly one relevant document the ideal DCG is 1, as the old code assumed.
    hits = np.array([[0, 1, 0]], dtype=bool)
    assert ndcg_binary(hits, np.array([1]), 3)[0] == pytest.approx(1 / np.log2(3))


def test_ndcg_many_relevant_is_not_saturated():
    # The defect this module fixes: one hit at rank 1 among many relevant documents
    # is NOT a perfect ranking. The old formula returned 1.0 here.
    hits = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
    ideal = (1 / np.log2(np.arange(2, 12))).sum()
    assert ndcg_binary(hits, np.array([555]), 10)[0] == pytest.approx(1 / ideal)
    assert ndcg_binary(hits, np.array([555]), 10)[0] < 0.25


def test_ndcg_perfect_ranking_is_one():
    hits = np.ones((1, 10), dtype=bool)
    assert ndcg_binary(hits, np.array([555]), 10)[0] == pytest.approx(1.0)
    # fewer relevant documents than k: all of them at the top is still perfect
    hits = np.array([[1, 1, 0, 0, 0]], dtype=bool)
    assert ndcg_binary(hits, np.array([2]), 5)[0] == pytest.approx(1.0)


def test_ndcg_no_relevant_documents_scores_zero():
    assert ndcg_binary(np.zeros((1, 5), dtype=bool), np.array([0]), 5)[0] == 0.0


def test_ndcg_graded_prefers_higher_grade_first():
    counts = np.array([[0, 5, 2]])  # 5 docs at grade 1, 2 at grade 2 -> ideal is [2, 2, 1]
    best = ndcg_graded(np.array([[2, 2, 1]]), counts, 3)[0]
    worse = ndcg_graded(np.array([[1, 2, 2]]), counts, 3)[0]
    assert best == pytest.approx(1.0)
    assert worse < best


def test_ndcg_graded_ideal_uses_lower_grades_when_high_run_out():
    counts = np.array([[0, 5, 1]])  # only one grade-2 document exists
    assert ndcg_graded(np.array([[2, 1, 1]]), counts, 3)[0] == pytest.approx(1.0)


def test_rank_metrics_keys():
    hits = np.zeros((4, 10), dtype=bool)
    hits[0, 0] = True
    m = rank_metrics(hits, np.full(4, 555))
    assert set(m) == {"success@1", "success@3", "success@5", "success@10", "MRR@10", "nDCG@10"}
    assert m["success@1"] == 0.25
