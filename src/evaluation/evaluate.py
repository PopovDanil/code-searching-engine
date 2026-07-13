"""Evaluation utilities for the CodeSearchNet benchmark.

Computes Recall@K, MRR, and NDCG by treating each query in the test
split as a search and checking whether the correct logical function
appears in the top-K results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
from dataclasses import replace
from pathlib import Path
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


# Recall is reported at several K: low K reflects final-ranking quality,
# high K (@50/@100) reflects the retriever's real job — getting the answer
# into the candidate pool that feeds the reranker.
_RECALL_KS = (1, 5, 10, 20, 50, 100)


def _precision_at_k(ranks: Sequence[float], k: int) -> float:
    """Precision@K. Each query has exactly one relevant doc, so this equals
    ``Recall@K / K`` — kept for completeness; NDCG@100 is the more useful
    pool-quality metric here."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r <= k)
    return hits / (k * len(ranks))


def _build_metrics(ranks: Sequence[float]) -> Dict[str, float]:
    """Build the full metric dict for a list of ranks."""
    metrics = {f"Recall@{k}": _recall_at_k(ranks, k) for k in _RECALL_KS}
    metrics["Precision@100"] = _precision_at_k(ranks, 100)
    metrics["MRR"] = _mrr(ranks)
    metrics["NDCG@10"] = _ndcg(ranks, 10)
    metrics["NDCG@100"] = _ndcg(ranks, 100)
    return metrics


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

    # CodeSearchNet stores PHP snippets without the opening "<?php" tag, so
    # tree-sitter-php parses them as inline HTML and finds no functions.
    if language == "php" and not code.lstrip().startswith("<?"):
        code = "<?php\n" + code

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
    max_rank: int = 100,
) -> float:
    """Return the rank of the correct function among unique retrieved parents.

    *max_rank* caps how deep we look (the size of the candidate pool that
    matters for the retriever, e.g. what would be handed to the reranker).
    A miss returns infinity.
    """
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
        if parent_rank >= max_rank:
            break
    return math.inf


# ── Evaluation cache ───────────────────────────────────────────────────

def _cache_key(
    max_dataset_records: Optional[int],
    embedding_model: str,
    split: str,
) -> str:
    """Return a deterministic cache key for the given parameters."""
    raw = f"{max_dataset_records}|{embedding_model}|{split}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _eval_cache_dir(config: CodeSearchConfig, lang: str) -> Path:
    """Return the cache directory for a language's evaluation index."""
    return Path(config.index_dir) / "eval_cache" / lang


def _save_eval_cache(
    cache_dir: Path,
    key: str,
    faiss_index,  # FaissCodeIndex
    entity_parent_ids: List[int],
    all_queries: List[Tuple[str, int]],
) -> None:
    """Persist evaluation state to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    faiss_index.save(str(cache_dir))
    with open(cache_dir / "eval_state.pkl", "wb") as fh:
        pickle.dump(
            {"entity_parent_ids": entity_parent_ids, "all_queries": all_queries},
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with open(cache_dir / "cache_key.json", "w") as fh:
        json.dump({"key": key}, fh)
    logger.info("Saved evaluation cache to %s", cache_dir)


def _load_eval_cache(
    cache_dir: Path,
    key: str,
):
    """Load evaluation state from disk if the cache key matches.

    Returns ``(faiss_index, entity_parent_ids, all_queries)`` or
    ``None`` if the cache is missing or stale.
    """
    key_path = cache_dir / "cache_key.json"
    index_path = cache_dir / "index.faiss"
    state_path = cache_dir / "eval_state.pkl"

    if not all([key_path.exists(), index_path.exists(), state_path.exists()]):
        return None

    with open(key_path) as fh:
        stored = json.load(fh)
    if stored.get("key") != key:
        logger.info("Cache key mismatch in %s - rebuilding", cache_dir)
        return None

    from indexing.faiss_index import FaissCodeIndex
    faiss_index = FaissCodeIndex.load(str(cache_dir))
    with open(state_path, "rb") as fh:
        state = pickle.load(fh)

    logger.info("Loaded evaluation cache from %s (%d vectors)", cache_dir, faiss_index.ntotal)
    return faiss_index, state["entity_parent_ids"], state["all_queries"]


def _build_entity_to_parent(
    metadata: List[CodeEntity],
    entity_parent_ids: List[int],
) -> Dict[int, int]:
    """Rebuild ``entity_to_parent`` mapping from loaded metadata."""
    return {id(entity): entity_parent_ids[i] for i, entity in enumerate(metadata)}


# ── Public evaluation API ───────────────────────────────────────────────

def evaluate_on_codesearchnet(
    config: CodeSearchConfig,
    languages: Optional[List[str]] = None,
    max_queries: Optional[int] = None,
    max_dataset_records: Optional[int] = None,
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
        Cap on the number of evaluation queries per language (useful
        for fast iteration).  ``None`` means use all loaded examples.
    max_dataset_records:
        Total number of records to load across all languages.
        Divided evenly among target languages.  ``None`` means load
        all records.
    split:
        Dataset split to use (``"test"`` or ``"validation"``).

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping ``language -> {metric_name: value}`` plus an ``"overall"``
        key with aggregate metrics across all languages.
    """
    from datasets import load_dataset

    from embedding.embedder import create_embedder
    from indexing.faiss_index import FaissCodeIndex
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
    separate = config.separate_indexes

    # Divide total records evenly across languages
    num_langs = len(target_langs)
    if max_dataset_records is not None:
        per_lang_base = max_dataset_records // num_langs
        remainder = max_dataset_records % num_langs
        per_lang_limits = {
            lang: per_lang_base + (1 if i < remainder else 0)
            for i, lang in enumerate(target_langs)
        }
    else:
        per_lang_limits = {lang: None for lang in target_langs}

    cache_key = _cache_key(max_dataset_records, config.embedding_model, split)

    # ── Create embedder once ───────────────────────────────────────
    embedder = create_embedder(
        model_name=config.embedding_model,
        device=config.device,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        query_instruction=config.query_instruction,
        torch_dtype=config.get_torch_dtype(),
        query_prefix=config.st_query_prefix,
        trust_remote_code=config.embedder_trust_remote_code,
        config_kwargs=config.st_config_kwargs,
    )

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Load corpora for all target languages
    # ═══════════════════════════════════════════════════════════════
    # per_lang_data[lang] = (corpus_entities, entity_to_parent, all_queries)
    per_lang_data: Dict[str, Tuple[List[CodeEntity], Dict[int, int], List[Tuple[str, int]]]] = {}

    for lang in target_langs:
        logger.info("Loading CodeSearchNet / %s", lang)
        per_lang_limit = per_lang_limits[lang]

        # ── Try loading from cache (separate mode only) ────────────
        if separate:
            cache_dir = _eval_cache_dir(config, lang)
            cached = _load_eval_cache(cache_dir, cache_key)
            if cached is not None:
                faiss_index, entity_parent_ids, all_queries = cached
                entity_to_parent = _build_entity_to_parent(
                    faiss_index.metadata, entity_parent_ids,
                )
                per_lang_data[lang] = ([], entity_to_parent, all_queries)
                continue

        # ── Load dataset ──────────────────────────────────────────
        try:
            ds = load_dataset(
                "code-search-net/code_search_net",
                lang,
                split=split,
                trust_remote_code=True,
            )
        except Exception:
            logger.exception("Failed to load CodeSearchNet for %s - skipping", lang)
            continue

        corpus_entities: List[CodeEntity] = []
        entity_to_parent: Dict[int, int] = {}
        all_queries: List[Tuple[str, int]] = []

        corpus_limit = per_lang_limit if per_lang_limit is not None else len(ds)
        load_progress = tqdm(ds, desc=f"Loading {lang}", total=min(corpus_limit, len(ds)))
        for i, example in enumerate(load_progress):
            if i >= corpus_limit:
                break

            query, chunks = _prepare_evaluation_example(example, lang, config)
            if not chunks:
                continue

            parent_id = len(all_queries)
            corpus_entities.extend(chunks)
            for chunk in chunks:
                entity_to_parent[id(chunk)] = parent_id
            all_queries.append((query, parent_id))

        if not all_queries:
            logger.warning("No valid examples for %s - skipping", lang)
            continue

        logger.info(
            "Loaded %d records (%d chunks) for %s",
            len(all_queries), len(corpus_entities), lang,
        )
        per_lang_data[lang] = (corpus_entities, entity_to_parent, all_queries)

    if not per_lang_data:
        logger.error("No languages loaded successfully - nothing to evaluate")
        return all_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Build index(es) based on separate_indexes config
    # ═══════════════════════════════════════════════════════════════
    if separate:
        # ── Build separate per-language indexes ────────────────────
        lang_indexes: Dict[str, FaissCodeIndex] = {}
        for lang, (corpus_entities, entity_to_parent, all_queries) in per_lang_data.items():
            if not corpus_entities:
                # Loaded from cache — skip rebuild
                continue

            texts = [
                entity.to_structured_text(include_docstring=False)
                for entity in corpus_entities
            ]
            embeddings = embedder.embed_documents(texts)

            faiss_index = FaissCodeIndex(
                dimension=embeddings.shape[1],
                index_type=config.index_type,
            )
            faiss_index.build(embeddings, corpus_entities)
            lang_indexes[lang] = faiss_index

            # Save cache
            cache_dir = _eval_cache_dir(config, lang)
            entity_parent_ids = [entity_to_parent[id(e)] for e in corpus_entities]
            _save_eval_cache(cache_dir, cache_key, faiss_index, entity_parent_ids, all_queries)
    else:
        # ── Build a single combined index for all languages ────────
        combined_entities: List[CodeEntity] = []
        combined_entity_to_parent: Dict[int, int] = {}
        combined_queries: List[Tuple[str, int]] = []
        # Per-language parent-id offsets: entity parent ids are globalised
        # below, so query-side ids must be shifted by the same amount when
        # ranks are computed in Phase 3.
        lang_parent_offsets: Dict[str, int] = {}

        for lang, (corpus_entities, entity_to_parent, all_queries) in per_lang_data.items():
            parent_offset = len(combined_queries)
            lang_parent_offsets[lang] = parent_offset
            combined_queries.extend(all_queries)
            combined_entities.extend(corpus_entities)
            for entity in corpus_entities:
                parent_id = entity_to_parent[id(entity)]
                combined_entity_to_parent[id(entity)] = parent_id + parent_offset

        texts = [
            entity.to_structured_text(include_docstring=False)
            for entity in combined_entities
        ]
        embeddings = embedder.embed_documents(texts)

        combined_index = FaissCodeIndex(
            dimension=embeddings.shape[1],
            index_type=config.index_type,
        )
        combined_index.build(embeddings, combined_entities)

        # Sparse arm for hybrid retrieval (BM25 over the same corpus/entities).
        combined_bm25 = None
        if config.enable_hybrid:
            from retrieval.bm25_index import BM25Index

            combined_bm25 = BM25Index()
            combined_bm25.build(texts, combined_entities)
            logger.info("Built BM25 index for hybrid retrieval (%d docs)", combined_bm25.size)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Run queries and compute metrics
    # ═══════════════════════════════════════════════════════════════
    all_ranks: Dict[str, List[float]] = {lang: [] for lang in per_lang_data}

    if separate:
        for lang, (_, entity_to_parent, all_queries) in per_lang_data.items():
            faiss_idx = lang_indexes.get(lang)
            if faiss_idx is None:
                # Must be from cache — load it
                cache_dir = _eval_cache_dir(config, lang)
                cached = _load_eval_cache(cache_dir, cache_key)
                if cached is None:
                    continue
                faiss_idx, _, _ = cached

            query_limit = max_queries if max_queries is not None else len(all_queries)
            query_limit = min(query_limit, len(all_queries))
            selected_queries = all_queries[:query_limit]

            logger.info(
                "Evaluating %d queries out of %d loaded records for %s",
                query_limit, len(all_queries), lang,
            )

            engine = SearchEngine(config=evaluation_config, index=faiss_idx)

            for qi, (query, relevant_parent) in enumerate(
                tqdm(selected_queries, desc=f"Searching {lang}")
            ):
                results = engine.search(query, top_k=config.retrieval_top_k)
                rank = _find_parent_rank(
                    results, entity_to_parent, relevant_parent,
                    max_rank=config.retrieval_top_k,
                )
                all_ranks[lang].append(rank)
    else:
        # Combined index — run all queries from all languages
        for lang, (_, entity_to_parent, all_queries) in per_lang_data.items():
            query_limit = max_queries if max_queries is not None else len(all_queries)
            query_limit = min(query_limit, len(all_queries))
            selected_queries = all_queries[:query_limit]

            logger.info(
                "Evaluating %d queries out of %d loaded records for %s",
                query_limit, len(all_queries), lang,
            )

            engine = SearchEngine(
                config=evaluation_config,
                index=combined_index,
                bm25_index=combined_bm25,
            )

            for qi, (query, relevant_parent) in enumerate(
                tqdm(selected_queries, desc=f"Searching {lang}")
            ):
                results = engine.search(query, top_k=config.retrieval_top_k)
                rank = _find_parent_rank(
                    results,
                    combined_entity_to_parent,
                    relevant_parent + lang_parent_offsets[lang],
                    max_rank=config.retrieval_top_k,
                )
                all_ranks[lang].append(rank)

    # ── Compute per-language metrics ───────────────────────────────
    for lang, ranks in all_ranks.items():
        if not ranks:
            continue
        metrics = _build_metrics(ranks)
        all_results[lang] = metrics
        for metric, val in metrics.items():
            logger.info("  %s / %s: %.4f", lang, metric, val)

    # ── Compute overall aggregate metrics ──────────────────────────
    combined = [r for ranks in all_ranks.values() for r in ranks]
    if combined:
        overall = _build_metrics(combined)
        all_results["overall"] = overall
        logger.info("  overall / total queries: %d", len(combined))
        for metric, val in overall.items():
            logger.info("  overall / %s: %.4f", metric, val)

    return all_results
