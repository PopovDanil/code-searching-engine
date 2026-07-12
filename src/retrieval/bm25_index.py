"""BM25 lexical index over code entities.

Dense embeddings miss exact-term matches (identifier names, rare API tokens);
BM25 catches them.  The key trick for code is **identifier splitting**:
``read_json_file`` and ``parseHTTPResponse`` are broken into their component
words so a natural-language query ("read json file") lexically overlaps the
function name.  Used as the sparse arm of hybrid (BM25 + dense) retrieval.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from parser.extract import CodeEntity

# Split camelCase / PascalCase, including acronym boundaries:
#   fooBar   -> foo Bar
#   HTTPServer -> HTTP Server
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")


def tokenize_code(text: str) -> List[str]:
    """Tokenize *text* into lowercased sub-word units for BM25.

    Alphanumeric runs are extracted (so ``_``, punctuation and whitespace are
    separators), then each run is split on camelCase boundaries.  This turns
    identifiers into their component words.
    """
    tokens: List[str] = []
    for run in _ALNUM_RUN.findall(text):
        for part in _CAMEL_BOUNDARY.sub(" ", run).split():
            tokens.append(part.lower())
    return tokens


class BM25Index:
    """In-memory BM25 index paired with the same ``CodeEntity`` objects used
    by the FAISS index, so hybrid fusion can match by object identity."""

    def __init__(self) -> None:
        self._entities: List[CodeEntity] = []
        self._bm25 = None  # rank_bm25.BM25Okapi

    def build(self, texts: Sequence[str], entities: Sequence[CodeEntity]) -> None:
        """Build the index from document *texts* aligned to *entities*."""
        from rank_bm25 import BM25Okapi

        if len(texts) != len(entities):
            raise ValueError("texts and entities must be the same length")
        self._entities = list(entities)
        corpus = [tokenize_code(t) for t in texts]
        # Guard against fully-empty tokenizations (BM25Okapi needs non-empty docs).
        corpus = [toks if toks else ["∅"] for toks in corpus]
        self._bm25 = BM25Okapi(corpus)

    def search(
        self, query: str, top_k: int = 100
    ) -> List[Tuple[CodeEntity, float]]:
        """Return the *top_k* entities most relevant to *query* by BM25 score."""
        if self._bm25 is None:
            raise RuntimeError("BM25Index.search called before build()")
        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        # argsort descending without importing numpy: enumerate + sort.
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [(self._entities[i], float(scores[i])) for i in ranked]

    @property
    def size(self) -> int:
        return len(self._entities)
