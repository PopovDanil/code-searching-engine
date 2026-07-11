"""Evaluation utilities for the CodeSearchNet benchmark.

Computes Recall@K, MRR, and NDCG by treating each query in the test
split as a search and checking whether the correct logical function
appears in the top-K results.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from config import CodeSearchConfig
from parser.extract import CodeEntity, extract_entities
from parser.parser import parse_file

logger = logging.getLogger(__name__)


# ── Metric helpers ──────────────────────────────────────────────────────

def _recall_at_k(ranks: Sequence[float], k: int) -> float:
    """Fraction of queries where the correct parent appears in top-K."""
    return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0.0


def _mrr(ranks: Sequence[float]) -> float:
    """Mean Reciprocal Rank."""
    return sum(1.0 / r for r in ranks) / len(ranks) if ranks else 0.0


def _ndcg(ranks: Sequence[float], k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at *k*.

    With binary relevance (the single correct doc is relevant), the
    ideal rank is always 1, so iDCG = 1.
    """
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k)
    return dcg / len(ranks) if ranks else 0.0


# ── Chunk-aware corpus preparation ─────────────────────────────────────

def _blank_entity_docstring(source_code: str, docstring: Optional[str]) -> str:
    """Blank an extracted Python docstring without changing its line count."""
    if not docstring:
        return source_code
    position = source_code.find(docstring)
    if position < 0:
        return source_code

    blank = "''" + "".join(char for char in docstring if char in "\r\n")
    return source_code[:position] + blank + source_code[position + len(docstring):]


def _contains_documentation(code: str, documentation: str) -> bool:
    """Return whether the query's word sequence remains in *code*."""
    normalised_doc = " ".join(re.findall(r"\w+", documentation.casefold()))
    normalised_code = " ".join(re.findall(r"\w+", code.casefold()))
    return bool(
        normalised_doc
        and f" {normalised_doc} " in f" {normalised_code} "
    )


def _parent_key(entity: CodeEntity) -> Tuple[object, ...]:
    """Identify all chunks emitted from the same extracted entity."""
    parent_start = (
        entity.parent_start_line
        if entity.parent_start_line is not None
        else entity.start_line
    )
    parent_end = (
        entity.parent_end_line
        if entity.parent_end_line is not None
        else entity.end_line
    )
    return (
        entity.entity_type,
        entity.class_name,
        entity.function_name,
        parent_start,
        parent_end,
    )


def _select_paired_chunks(
    entities: List[CodeEntity], dataset_function_name: str
) -> List[CodeEntity]:
    """Select only chunks belonging to the dataset row's function."""
    if not entities:
        return []

    simple_name = dataset_function_name.rsplit(".", 1)[-1]
    expected_names = {dataset_function_name, simple_name}
    matching = [
        entity
        for entity in entities
        if entity.entity_type != "class"
        and entity.function_name in expected_names
    ]
    non_classes = [entity for entity in entities if entity.entity_type != "class"]
    candidates = matching or non_classes
    if not candidates:
        return []
    anchor = candidates[0]
    parent_key = _parent_key(anchor)
    return [entity for entity in entities if _parent_key(entity) == parent_key]


def _prepare_evaluation_example(
    example: dict,
    language: str,
    config: CodeSearchConfig,
) -> Tuple[str, List[CodeEntity]]:
    """Turn one CodeSearchNet row into a query and its code chunks."""
    code = str(example.get("func_code_string", "") or "")
    documentation = str(example.get("func_documentation_string", "") or "")
    if not code.strip() or not documentation.strip():
        return documentation, []

    repository = str(
        example.get("repository_name", example.get("repository", "")) or ""
    )
    file_path = str(
        example.get("func_path_in_repository", example.get("path", "")) or ""
    )
    function_name = str(example.get("func_name", "") or "")
    tree = parse_file(code, language)
    if tree is None:
        return documentation, []

    parent_entities = extract_entities(
        source_code=code,
        tree=tree,
        language=language,
        repository=repository,
        file_path=file_path,
    )
    selected_parents = _select_paired_chunks(parent_entities, function_name)
    if not selected_parents:
        return documentation, []

    parent_source = selected_parents[0].source_code
    if language == "python":
        code = _blank_entity_docstring(parent_source, selected_parents[0].docstring)
        tree = parse_file(code, language)
        if tree is None:
            return documentation, []
        parent_source = code

    if _contains_documentation(parent_source, documentation):
        logger.debug("Skipping evaluation row with documentation leakage")
        return documentation, []

    chunks = extract_entities(
        source_code=code,
        tree=tree,
        language=language,
        repository=repository,
        file_path=file_path,
        max_chunk_chars=config.max_chunk_chars,
        chunk_overlap_chars=config.chunk_overlap_chars,
    )
    paired_chunks = _select_paired_chunks(chunks, function_name)
    if not paired_chunks:
        return documentation, []

    return documentation, [
        replace(entity, docstring=None) for entity in paired_chunks
    ]


def _find_parent_rank(
    results: Sequence[object],
    entity_to_parent: Dict[int, int],
    relevant_parent: int,
    top_k: int = 10,
) -> float:
    """Return the rank among unique parents, or infinity for a miss."""
    seen_parents = set()
    parent_rank = 0

    for result in results:
        parent = entity_to_parent.get(id(result.entity))
        if parent is None or parent in seen_parents:
            continue
        seen_parents.add(parent)
        parent_rank += 1
        if parent == relevant_parent:
            return float(parent_rank)
        if parent_rank >= top_k:
            break
    return math.inf


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
    from retrieval.search import SearchEngine

    target_langs = languages or [
        "python",
        "java",
        "javascript",
        "go",
        "ruby",
        "php",
    ]
    all_results: Dict[str, Dict[str, float]] = {}
    evaluation_config = replace(config, include_docstring=False)

    for lang in target_langs:
        logger.info("Evaluating on CodeSearchNet / %s", lang)
        try:
            ds = load_dataset(
                "code-search-net/code_search_net",
                lang,
                split=split,
                trust_remote_code=True,
            )
        except Exception:
            logger.exception("Failed to load CodeSearchNet for %s — skipping", lang)
            continue

        # Build a chunked corpus while retaining parent-level relevance.
        corpus_entities: List[CodeEntity] = []
        entity_to_parent: Dict[int, int] = {}
        query_to_relevant: Dict[int, int] = {}
        queries: List[str] = []

        limit = max_queries if max_queries is not None else len(ds)
        progress = tqdm(ds, desc=f"Loading {lang}", total=min(limit, len(ds)))
        for i, example in enumerate(progress):
            if i >= limit:
                break

            query, chunks = _prepare_evaluation_example(example, lang, config)
            if not chunks:
                continue

            parent_id = len(queries)
            corpus_entities.extend(chunks)
            for chunk in chunks:
                entity_to_parent[id(chunk)] = parent_id
            query_idx = len(queries)
            queries.append(query)
            query_to_relevant[query_idx] = parent_id

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
        texts = [
            entity.to_structured_text(include_docstring=False)
            for entity in corpus_entities
        ]
        embeddings = embedder.embed_documents(texts)

        from indexing.faiss_index import FaissCodeIndex
        faiss_index = FaissCodeIndex(
            dimension=embeddings.shape[1],
            index_type="flat",
        )
        faiss_index.build(embeddings, corpus_entities)

        # Create search engine with the pre-built index
        engine = SearchEngine(
            config=evaluation_config,
            embedder=embedder,
            index=faiss_index,
        )

        # Evaluate
        ranks: List[float] = []
        for qi, query in enumerate(tqdm(queries, desc=f"Searching {lang}")):
            results = engine.search(query, top_k=config.retrieval_top_k)
            rank = _find_parent_rank(
                results,
                entity_to_parent,
                query_to_relevant[qi],
                top_k=10,
            )
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
