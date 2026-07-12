"""Search engine: embed query → FAISS retrieval → reranking → scoring."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import CodeSearchConfig
from embedding.embedder import BaseEmbedder, create_embedder
from indexing.faiss_index import FaissCodeIndex
from parser.extract import CodeEntity
from retrieval.reranker import BaseReranker, create_reranker
from retrieval.query_rewriter import BaseQueryRewriter, create_query_rewriter

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
        Optional pre-loaded FAISS index (single-index mode).
    """

    def __init__(
        self,
        config: CodeSearchConfig,
        embedder: Optional[BaseEmbedder] = None,
        reranker: Optional[BaseReranker] = None,
        index: Optional[FaissCodeIndex] = None,
        query_rewriter: Optional[BaseQueryRewriter] = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._reranker = reranker
        self._index = index
        self._query_rewriter = query_rewriter
        self._language_indexes: Dict[str, FaissCodeIndex] = {}
        self._manifest_loaded = False

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
                language_hint=self._config.reranker_language_hint,
            )
        return self._reranker

    def _ensure_query_rewriter(self) -> BaseQueryRewriter:
        if self._query_rewriter is None:
            self._query_rewriter = create_query_rewriter(
                enabled=self._config.enable_query_rewriting,
                strategy=self._config.query_rewrite_strategy,
                model_name=self._config.query_rewriter_model,
                device=self._config.device,
                max_new_tokens=self._config.query_rewriter_max_new_tokens,
                torch_dtype=self._config.get_torch_dtype(),
            )
        return self._query_rewriter

    def _ensure_index(self) -> FaissCodeIndex:
        if self._index is None:
            self._index = FaissCodeIndex.load(self._config.index_dir)
        return self._index

    def _load_manifest(self) -> List[str]:
        """Load the language manifest for separate-index mode."""
        if self._manifest_loaded:
            return list(self._language_indexes.keys())

        manifest_path = os.path.join(self._config.index_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            self._manifest_loaded = True
            return []

        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        languages = manifest.get("languages", [])

        for lang in languages:
            lang_dir = os.path.join(self._config.index_dir, lang)
            if os.path.isdir(lang_dir):
                self._language_indexes[lang] = FaissCodeIndex.load(lang_dir)

        self._manifest_loaded = True
        return list(self._language_indexes.keys())

    def _ensure_language_index(self, language: str) -> FaissCodeIndex:
        """Load a specific language index, raising if not found."""
        self._load_manifest()
        if language not in self._language_indexes:
            lang_dir = os.path.join(self._config.index_dir, language)
            if os.path.isdir(lang_dir):
                self._language_indexes[language] = FaissCodeIndex.load(lang_dir)
            else:
                raise FileNotFoundError(
                    f"No index found for language '{language}'. "
                    f"Available: {list(self._language_indexes.keys())}"
                )
        return self._language_indexes[language]

    # ── Public API ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        language: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute a semantic search for *query*.

        Parameters
        ----------
        query:
            The search query string.
        top_k:
            Number of final results to return.
        language:
            If provided, restrict search to this language index
            (only effective in separate-index mode).

        Steps:
        1. Embed the query.
        2. Retrieve ``retrieval_top_k`` candidates from FAISS.
        3. (Optionally) rerank with the cross-encoder.
        4. Compute final weighted score.
        5. Return top-K results.
        """
        k = top_k or self._config.top_k
        w = self._config.weights
        original_query = query
        query = self._ensure_query_rewriter().rewrite(original_query)
        if query != original_query:
            logger.info("Rewritten query: %r -> %r", original_query, query)
        embedder = self._ensure_embedder()

        # 1. Embed query
        t0 = time.perf_counter()
        query_vec = embedder.embed_query(query)
        logger.info("Query embedding latency: %.2f ms", (time.perf_counter() - t0) * 1000)

        # 2. FAISS retrieval
        t0 = time.perf_counter()
        candidates = self._retrieve_candidates(query_vec, language)
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

        # 4. Scoring — build a lookup dict for O(1) embedding similarity retrieval
        emb_sim_map = {id(ent): sim for ent, sim in candidates}
        results: List[SearchResult] = []
        for entity, reranker_score in reranked:
            emb_sim = emb_sim_map.get(id(entity), 0.0)
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

    def _retrieve_candidates(
        self,
        query_vec,
        language: Optional[str] = None,
    ):
        """Retrieve candidates from FAISS, supporting both index modes."""
        if self._config.separate_indexes:
            return self._retrieve_separate(query_vec, language)
        faiss_idx = self._ensure_index()
        return faiss_idx.search(query_vec, top_k=self._config.retrieval_top_k)

    def _retrieve_separate(
        self,
        query_vec,
        language: Optional[str] = None,
    ):
        """Retrieve candidates from per-language indexes."""
        if language:
            faiss_idx = self._ensure_language_index(language)
            return faiss_idx.search(query_vec, top_k=self._config.retrieval_top_k)

        # Search all language indexes and merge results
        self._load_manifest()
        all_candidates = []
        per_lang_k = max(1, self._config.retrieval_top_k // max(1, len(self._language_indexes)))
        for lang, faiss_idx in self._language_indexes.items():
            candidates = faiss_idx.search(query_vec, top_k=per_lang_k)
            all_candidates.extend(candidates)

        # Sort by similarity and take top retrieval_top_k
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        return all_candidates[: self._config.retrieval_top_k]

    def set_index(self, index: FaissCodeIndex) -> None:
        """Replace the current FAISS index."""
        self._index = index
