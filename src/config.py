"""Global configuration for the codesearch system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from parser.chunker import SUPPORTED_CHUNKER_TYPES


def _auto_device() -> str:
    """Select CUDA if available, otherwise CPU."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class CodeSearchConfig:
    """Configuration for the semantic code search pipeline.

    All weights and model names are configurable so the system can be
    adapted without modifying source code.
    """

    # ── Models ──────────────────────────────────────────────────────────
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"

    # ── Processing ──────────────────────────────────────────────────────
    batch_size: int = 16
    max_seq_length: int = 512
    num_parser_workers: int = 4
    max_chunk_chars: Optional[int] = 1500
    chunk_overlap_chars: int = 150
    chunker_type: str = "recursive"

    # ── Retrieval ───────────────────────────────────────────────────────
    top_k: int = 10
    retrieval_top_k: int = 50  # candidates fetched before reranking

    # ── Dataset / Evaluation ────────────────────────────────────────────
    max_dataset_records: Optional[int] = None  # total records across all languages to load into the database (None = all)

    # ── Index ───────────────────────────────────────────────────────────
    index_type: str = "flat"  # "flat" | "hnsw"
    index_dir: str = "index"
    separate_indexes: bool = False  # build a separate index per language
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    # ── Device ──────────────────────────────────────────────────────────
    device: str = field(default_factory=_auto_device)

    # ── Scoring weights ─────────────────────────────────────────────────
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "reranker": 0.75,
            "embedding": 0.20,
            "metadata": 0.05,
        }
    )

    # ── Reranker toggle ─────────────────────────────────────────────────
    enable_reranking: bool = True

    # Query rewriting is opt-in to preserve the original retrieval behaviour.
    enable_query_rewriting: bool = False
    query_rewrite_strategy: str = "none"  # "none" | "rewrite" | "hyde"
    query_rewriter_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    query_rewriter_max_new_tokens: int = 128

    # Add the candidate's programming language to the reranker prompt.
    reranker_language_hint: bool = False

    # ── Reranker prompt settings ────────────────────────────────────────
    # Token budget for the full reranker prompt (prefix + pair + suffix).
    # Kept separate from max_seq_length: Qwen3-Reranker handles long
    # contexts, while 512 would truncate most (query, code) pairs.
    reranker_max_length: int = 512
    reranker_instruction: str = (
        "Given a natural-language search query, judge whether the code "
        "snippet implements the functionality described in the query."
    )

    # ── Docstring inclusion in structured text ──────────────────────────
    include_docstring: bool = True

    # ── Embedding instruction (for Qwen3-Embedding) ────────────────────
    query_instruction: str = "Retrieve relevant source code based on the user query"

    # ── Persistence ─────────────────────────────────────────────────────
    embedding_dtype: str = "float16"  # "float16" | "float32" | "bfloat16"

    # -------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate recursive chunking limits before parser workers start."""
        self.validate_chunking()
        if self.query_rewrite_strategy not in {"none", "rewrite", "hyde"}:
            raise ValueError(
                "query_rewrite_strategy must be one of: none, rewrite, hyde"
            )
        if self.query_rewriter_max_new_tokens <= 0:
            raise ValueError("query_rewriter_max_new_tokens must be greater than zero")

    def validate_chunking(self) -> None:
        """Validate the current recursive chunking settings."""
        if self.max_chunk_chars is None:
            return
        if not isinstance(self.chunker_type, str) or not self.chunker_type:
            raise ValueError("chunker_type must be a non-empty string")
        if self.chunker_type not in SUPPORTED_CHUNKER_TYPES:
            raise ValueError(
                "chunker_type must be one of: " + ", ".join(SUPPORTED_CHUNKER_TYPES)
            )
        if not isinstance(self.max_chunk_chars, int) or isinstance(
            self.max_chunk_chars, bool
        ):
            raise ValueError("max_chunk_chars must be an integer or null")
        if not isinstance(self.chunk_overlap_chars, int) or isinstance(
            self.chunk_overlap_chars, bool
        ):
            raise ValueError("chunk_overlap_chars must be an integer")
        if self.max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be greater than zero")
        if self.chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars cannot be negative")
        if self.chunk_overlap_chars >= self.max_chunk_chars:
            raise ValueError(
                "chunk_overlap_chars must be smaller than max_chunk_chars"
            )

    @classmethod
    def from_yaml(cls, path: str) -> "CodeSearchConfig":
        """Load configuration from a YAML file.

        Missing keys fall back to class defaults.
        """
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        # Only pass keys that the dataclass actually accepts
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def get_torch_dtype(self):
        """Return the torch dtype matching ``embedding_dtype``."""
        import torch

        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(self.embedding_dtype, torch.float16)
