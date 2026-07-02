"""Tests for evaluation metrics."""

from evaluation.evaluate import _recall_at_k, _mrr, _ndcg


def test_recall_at_k_perfect():
    ranks = [1, 1, 1]
    assert _recall_at_k(ranks, 1) == 1.0


def test_recall_at_k_partial():
    ranks = [1, 5, 10]
    assert _recall_at_k(ranks, 1) == 1 / 3
    assert _recall_at_k(ranks, 5) == 2 / 3
    assert _recall_at_k(ranks, 10) == 1.0


def test_recall_at_k_empty():
    assert _recall_at_k([], 5) == 0.0


def test_mrr():
    ranks = [1, 2, 3]
    expected = (1 / 1 + 1 / 2 + 1 / 3) / 3
    assert abs(_mrr(ranks) - expected) < 1e-6


def test_mrr_empty():
    assert _mrr([]) == 0.0


def test_ndcg():
    import math
    ranks = [1, 2]
    # DCG@k uses 1/log2(r+1), iDCG=1.0 (binary relevance, ideal rank=1)
    per_query_ndcg = [1.0 / math.log2(r + 1) for r in ranks]
    expected = sum(per_query_ndcg) / len(per_query_ndcg)
    assert abs(_ndcg(ranks, 10) - expected) < 1e-6


def test_ndcg_empty():
    assert _ndcg([], 10) == 0.0


def test_ndcg_outside_k():
    ranks = [11, 12]
    assert _ndcg(ranks, 10) == 0.0
