"""Reciprocal Rank Fusion (RRF).

Combines several ranked id-lists into one ranking using only positions, not
raw scores — which makes it robust to the incomparable scales of dense cosine
similarity and BM25.  The same id appearing in multiple lists has its
contributions summed, so fusion also deduplicates.

Formula (Cormack et al. 2009): ``score(d) = Σ_i w_i / (k + rank_i(d))`` where
``rank`` is 0-based and ``k`` (default 60) dampens the influence of top ranks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, List, Optional, Sequence, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[Hashable, float]]:
    """Fuse ranked id-lists into one ranking.

    Parameters
    ----------
    ranked_lists:
        Each inner sequence is a list of item ids ordered best-first.
    k:
        RRF damping constant (default 60).
    weights:
        Optional per-list weights (default 1.0 each).  A weight of 0 makes a
        list contribute nothing (useful to disable one retriever).

    Returns
    -------
    List[Tuple[id, score]]
        Fused ``(id, score)`` pairs sorted best-first.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    scores: dict = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        if weight == 0:
            continue
        for rank, item_id in enumerate(ranked):
            scores[item_id] += weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
