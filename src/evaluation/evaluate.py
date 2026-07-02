"""Evaluation utilities for the CodeSearchNet benchmark.

Computes Recall@K, MRR, and NDCG by treating each query in the test
split as a search and checking whether the correct code snippet appears
in the top-K results.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from tqdm import tqdm

from config import CodeSearchConfig
from parser.extract import CodeEntity
from retrieval.search import SearchEngine

logger = logging.getLogger(__name__)


# ── Metric helpers ──────────────────────────────────────────────────────

def _recall_at_k(ranks: List[int], k: int) -> float:
    """Fraction of queries where the correct doc appears in top-K."""
    return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0.0


def _mrr(ranks: List[int]) -> float:
    """Mean Reciprocal Rank."""
    return sum(1.0 / r for r in ranks) / len(ranks) if ranks else 0.0


def _ndcg(ranks: List[int], k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at *k*.

    With binary relevance (the single correct doc is relevant), the
    ideal rank is always 1, so iDCG = 1.
    """
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k)
    return dcg / len(ranks) if ranks else 0.0


# ── Public evaluation API ───────────────────────────────────────────────

def evaluate_on_codesearchnet(
    config: CodeSearchConfig,
    languages: Optional[List[str]] = None,
    max_queries: Optional[int] = None,
    split: str = "test",
) -> Dict[str, Dict[str, float]]:
    """Evaluate the search system on CodeSearchNet.

    Parameters
    ----------
    config:
        System configuration.
    languages:
        Subset of languages to evaluate.  ``None`` means all six.
    max_queries:
        Cap on the number of queries per language (useful for fast
        iteration).  ``None`` means no cap.
    split:
        Dataset split to use (``"test"`` or ``"validation"``).

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping ``language → {metric_name: value}``.
    """
    from datasets import load_dataset

    target_langs = languages or ["python", "java", "javascript", "go", "ruby", "php"]
    all_results: Dict[str, Dict[str, float]] = {}

    for lang in target_langs:
        logger.info("Evaluating on CodeSearchNet / %s", lang)
        try:
            ds = load_dataset("code-search-net/code_search_net", lang, split=split, trust_remote_code=True)
        except Exception:
            logger.exception("Failed to load CodeSearchNet for %s — skipping", lang)
            continue

        # Build a corpus of code entities from the dataset
        corpus_entities: List[CodeEntity] = []
        query_to_relevant: Dict[int, int] = {}  # query_idx → corpus_idx
        queries: List[str] = []

        limit = max_queries or len(ds)
        for i, example in enumerate(tqdm(ds, desc=f"Loading {lang}", total=min(limit, len(ds)))):
            if i >= limit:
                break

            code = example.get("func_code_string", "")
            doc = example.get("func_documentation_string", "")
            func_name = example.get("func_name", "")
            repo = example.get("repository", "")
            path = example.get("path", "")

            if not code.strip() or not doc.strip():
                continue

            entity = CodeEntity(
                repository=repo,
                file_path=path,
                language=lang,
                entity_type="function",
                function_name=func_name,
                class_name=None,
                signature=code.split("\n")[0].strip(),
                docstring=doc if doc else None,
                source_code=code,
                start_line=1,
                end_line=code.count("\n") + 1,
            )
            corpus_idx = len(corpus_entities)
            corpus_entities.append(entity)

            query_idx = len(queries)
            queries.append(doc)
            query_to_relevant[query_idx] = corpus_idx

        if not queries:
            logger.warning("No valid examples for %s — skipping", lang)
            continue

        # Index the corpus
        from embedding.embedder import create_embedder

        embedder = create_embedder(
            model_name=config.embedding_model,
            device=config.device,
            max_seq_length=config.max_seq_length,
            batch_size=config.batch_size,
            query_instruction=config.query_instruction,
            torch_dtype=config.get_torch_dtype(),
        )

        # Embed corpus entities
        texts = [e.to_structured_text(include_docstring=config.include_docstring) for e in corpus_entities]
        embeddings = embedder.embed_documents(texts)

        from indexing.faiss_index import FaissCodeIndex
        faiss_index = FaissCodeIndex(
            dimension=embeddings.shape[1],
            index_type="flat",
        )
        faiss_index.build(embeddings, corpus_entities)

        # Create search engine with the pre-built index
        engine = SearchEngine(config=config, embedder=embedder, index=faiss_index)

        # Evaluate
        ranks: List[int] = []
        for qi, query in enumerate(tqdm(queries, desc=f"Searching {lang}")):
            results = engine.search(query, top_k=10)
            relevant_idx = query_to_relevant[qi]
            # Find rank of the correct entity
            rank = 9999
            for ri, res in enumerate(results, start=1):
                if res.entity is corpus_entities[relevant_idx]:
                    rank = ri
                    break
            ranks.append(rank)

        metrics = {
            "Recall@1": _recall_at_k(ranks, 1),
            "Recall@5": _recall_at_k(ranks, 5),
            "Recall@10": _recall_at_k(ranks, 10),
            "MRR": _mrr(ranks),
            "NDCG@10": _ndcg(ranks, 10),
        }
        all_results[lang] = metrics

        for metric, val in metrics.items():
            logger.info("  %s / %s: %.4f", lang, metric, val)

    return all_results
