"""Search engine: embed query → FAISS retrieval → reranking → scoring."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from config import CodeSearchConfig
from embedding.embedder import BaseEmbedder, create_embedder
from indexing.faiss_index import FaissCodeIndex
from parser.extract import CodeEntity
from retrieval.reranker import BaseReranker, create_reranker

logger = logging.getLogger(__name__)


# ── Result dataclass ────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single ranked search result."""

    entity: CodeEntity
    final_score: float
    embedding_similarity: float
    reranker_score: Optional[float]
    metadata_bonus: float

    def __str__(self) -> str:
        lines = [
            f"  Score:     {self.final_score:.4f}",
            f"  File:      {self.entity.file_path}",
            f"  Function:  {self.entity.identifier}",
            f"  Language:  {self.entity.language.capitalize()}",
            f"  Lines:     {self.entity.start_line}–{self.entity.end_line}",
            "  Code:",
        ]
        for line in self.entity.source_code.splitlines():
            lines.append(f"    {line}")
        return "\n".join(lines)


# ── Metadata bonus ──────────────────────────────────────────────────────

def _compute_metadata_bonus(entity: CodeEntity, query: str) -> float:
    """Compute a bonus score (0–1) based on simple metadata heuristics.

    * function/class name contains query words          → up to 0.4
    * exact identifier match                             → +0.3
    * docstring contains query words                     → up to 0.3
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    bonus = 0.0

    name = (entity.function_name or entity.class_name or "").lower()
    if name:
        if query_lower in name or any(w in name for w in query_words):
            bonus += 0.4
        # Exact identifier match (query "read json" → "read_json")
        if name == query_lower.replace(" ", "_"):
            bonus += 0.3

    if entity.docstring:
        doc_lower = entity.docstring.lower()
        matching = sum(1 for w in query_words if w in doc_lower)
        if query_words and matching:
            bonus += 0.3 * (matching / len(query_words))

    return min(bonus, 1.0)


# ── Search engine ───────────────────────────────────────────────────────

class SearchEngine:
    """End-to-end semantic code search.

    Parameters
    ----------
    config:
        System configuration.
    embedder:
        Optional pre-initialised embedder.
    reranker:
        Optional pre-initialised reranker.
    index:
        Optional pre-loaded FAISS index.
    """

    def __init__(
        self,
        config: CodeSearchConfig,
        embedder: Optional[BaseEmbedder] = None,
        reranker: Optional[BaseReranker] = None,
        index: Optional[FaissCodeIndex] = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._reranker = reranker
        self._index = index

    # ── Lazy initialisation ─────────────────────────────────────────────

    def _ensure_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = create_embedder(
                model_name=self._config.embedding_model,
                device=self._config.device,
                max_seq_length=self._config.max_seq_length,
                batch_size=self._config.batch_size,
                query_instruction=self._config.query_instruction,
                torch_dtype=self._config.get_torch_dtype(),
            )
        return self._embedder

    def _ensure_reranker(self) -> BaseReranker:
        if self._reranker is None:
            self._reranker = create_reranker(
                model_name=self._config.reranker_model,
                device=self._config.device,
                max_seq_length=self._config.max_seq_length,
                batch_size=self._config.batch_size,
                enabled=self._config.enable_reranking,
                torch_dtype=self._config.get_torch_dtype(),
                include_docstring=self._config.include_docstring,
            )
        return self._reranker

    def _ensure_index(self) -> FaissCodeIndex:
        if self._index is None:
            self._index = FaissCodeIndex.load(self._config.index_dir)
        return self._index

    # ── Public API ──────────────────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Execute a semantic search for *query*.

        Steps:
        1. Embed the query.
        2. Retrieve ``retrieval_top_k`` candidates from FAISS.
        3. (Optionally) rerank with the cross-encoder.
        4. Compute final weighted score.
        5. Return top-K results.
        """
        k = top_k or self._config.top_k
        w = self._config.weights
        embedder = self._ensure_embedder()
        faiss_idx = self._ensure_index()

        # 1. Embed query
        t0 = time.perf_counter()
        query_vec = embedder.embed_query(query)
        logger.info("Query embedding latency: %.2f ms", (time.perf_counter() - t0) * 1000)

        # 2. FAISS retrieval
        t0 = time.perf_counter()
        candidates = faiss_idx.search(query_vec, top_k=self._config.retrieval_top_k)
        logger.info(
            "FAISS search latency: %.2f ms (candidates=%d)",
            (time.perf_counter() - t0) * 1000,
            len(candidates),
        )

        # 3. Reranking
        reranker = self._ensure_reranker()
        t0 = time.perf_counter()
        reranked = reranker.rerank(query, candidates, batch_size=self._config.batch_size)
        logger.info("Reranking latency: %.2f ms", (time.perf_counter() - t0) * 1000)

        # 4. Scoring
        results: List[SearchResult] = []
        for entity, reranker_score in reranked:
            emb_sim = 0.0
            # Recover embedding similarity from candidates list
            for ent, sim in candidates:
                if ent is entity:
                    emb_sim = sim
                    break

            meta_bonus = _compute_metadata_bonus(entity, query)

            # If reranking is disabled, reranker_score is the embedding similarity
            # and we adjust the weights accordingly
            if not self._config.enable_reranking:
                final = emb_sim
            else:
                final = (
                    w.get("reranker", 0.75) * reranker_score
                    + w.get("embedding", 0.20) * emb_sim
                    + w.get("metadata", 0.05) * meta_bonus
                )

            results.append(
                SearchResult(
                    entity=entity,
                    final_score=final,
                    embedding_similarity=emb_sim,
                    reranker_score=reranker_score if self._config.enable_reranking else None,
                    metadata_bonus=meta_bonus,
                )
            )

        # Sort by final score descending
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:k]

    def set_index(self, index: FaissCodeIndex) -> None:
        """Replace the current FAISS index."""
        self._index = index
