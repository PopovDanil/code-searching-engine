"""High-level pipeline: walk repository → parse → embed → build index."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from config import CodeSearchConfig
from embedding.embedder import BaseEmbedder, create_embedder
from indexing.faiss_index import FaissCodeIndex
from parser.extract import CodeEntity, extract_entities
from parser.parser import detect_language, parse_file

logger = logging.getLogger(__name__)


# ── Single-file processing (top-level for pickling) ─────────────────────

def _process_file(args: tuple) -> List[CodeEntity]:
    """Parse a single source file and return extracted entities.

    Accepts a tuple so this function is picklable for ProcessPoolExecutor.
    """
    file_path, repository, language, *chunk_options = args
    max_chunk_chars = chunk_options[0] if chunk_options else None
    chunk_overlap_chars = chunk_options[1] if len(chunk_options) > 1 else 0
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            source_code = fh.read()
    except (OSError, UnicodeDecodeError):
        logger.warning("Cannot read file: %s", file_path)
        return []

    if not source_code.strip():
        return []

    if language is None:
        language = detect_language(file_path)
    if language is None:
        return []

    tree = parse_file(source_code, language)
    if tree is None:
        return []

    rel_path = os.path.relpath(file_path, repository) if repository else file_path
    return extract_entities(
        source_code=source_code,
        tree=tree,
        language=language,
        repository=os.path.basename(repository),
        file_path=rel_path,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )


# ── Public API ──────────────────────────────────────────────────────────

def build_index(
    repository_path: str,
    config: CodeSearchConfig,
    embedder: Optional[BaseEmbedder] = None,
) -> FaissCodeIndex:
    """Walk *repository_path*, parse, embed, and return a ``FaissCodeIndex``.

    Parameters
    ----------
    repository_path:
        Root directory of the source repository.
    config:
        System configuration.
    embedder:
        Optional pre-initialised embedder.  If ``None`` one is created
        from *config*.

    Returns
    -------
    FaissCodeIndex
        Fully-built and persisted index.
    """
    repository_path = os.path.abspath(repository_path)
    if not os.path.isdir(repository_path):
        raise FileNotFoundError(f"Repository path does not exist: {repository_path}")
    config.validate_chunking()

    # 1. Collect source files ------------------------------------------------
    supported_exts = set()
    from parser.parser import SUPPORTED_LANGUAGES
    for info in SUPPORTED_LANGUAGES.values():
        supported_exts.update(info.extensions)

    file_args: List[tuple] = []
    for root, _dirs, files in os.walk(repository_path):
        # Skip hidden / vendor directories
        dirs_to_skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "vendor"}
        root_obj = Path(root)
        if any(part in dirs_to_skip or part.startswith(".") for part in root_obj.parts):
            continue
        for fname in files:
            full = os.path.join(root, fname)
            if Path(fname).suffix.lower() in supported_exts:
                lang = detect_language(full)
                file_args.append(
                    (
                        full,
                        repository_path,
                        lang,
                        config.max_chunk_chars,
                        config.chunk_overlap_chars,
                    )
                )

    logger.info("Found %d source files in %s", len(file_args), repository_path)
    if not file_args:
        raise ValueError("No supported source files found in the repository.")

    # 2. Parse files in parallel ---------------------------------------------
    all_entities: List[CodeEntity] = []
    workers = min(config.num_parser_workers, len(file_args))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_file, a): a for a in file_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing"):
            try:
                entities = future.result()
                all_entities.extend(entities)
            except Exception:
                logger.exception("Error processing file %s", futures[future])

    logger.info("Extracted %d code entities", len(all_entities))
    if not all_entities:
        raise ValueError("No code entities were extracted from the repository.")

    # 3. Build structured texts ----------------------------------------------
    texts = [entity.to_structured_text(include_docstring=config.include_docstring) for entity in all_entities]

    # 4. Initialise embedder if needed ---------------------------------------
    if embedder is None:
        embedder = create_embedder(
            model_name=config.embedding_model,
            device=config.device,
            max_seq_length=config.max_seq_length,
            batch_size=config.batch_size,
            query_instruction=config.query_instruction,
            torch_dtype=config.get_torch_dtype(),
        )

    # 5. Embed all documents -------------------------------------------------
    logger.info("Embedding %d documents (batch_size=%d)…", len(texts), config.batch_size)
    embeddings: np.ndarray = embedder.embed_documents(texts)
    logger.info("Embedding shape: %s", embeddings.shape)

    # 6. Build FAISS index ---------------------------------------------------
    faiss_index = FaissCodeIndex(
        dimension=embeddings.shape[1],
        index_type=config.index_type,
        hnsw_m=config.hnsw_m,
        hnsw_ef_construction=config.hnsw_ef_construction,
        hnsw_ef_search=config.hnsw_ef_search,
    )
    faiss_index.build(embeddings, all_entities)

    # 7. Persist -------------------------------------------------------------
    faiss_index.save(config.index_dir)
    return faiss_index
