"""Embedding model abstraction and implementations.

Provides a ``BaseEmbedder`` interface with two concrete implementations:

* ``Qwen3Embedder`` — wraps Qwen3-Embedding-8B (or any Qwen3-Embedding
  variant) via the ``transformers`` library.
* ``SentenceTransformerEmbedder`` — lightweight fallback using the
  ``sentence-transformers`` ecosystem.

Use ``create_embedder()`` to instantiate the correct class based on the
model name.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Abstract interface ──────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Interface that every embedder must implement."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings for a batch of document texts.

        Parameters
        ----------
        texts:
            List of structured text representations.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), dim)`` with float32 values,
            L2-normalised along the last axis.
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Return a single L2-normalised embedding for *query*.

        Returns
        -------
        np.ndarray
            Shape ``(dim,)`` with float32 values.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality."""


# ── Utility: last-token pooling ─────────────────────────────────────────

def _last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the last hidden state by selecting the last *non-padding* token.

    This is the recommended pooling strategy for Qwen3-Embedding models.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


# ── Qwen3-Embedding implementation ─────────────────────────────────────

class Qwen3Embedder(BaseEmbedder):
    """Embedder backed by Qwen3-Embedding (via ``transformers``).

    For *queries*, an instruction prefix is prepended so the model can
    differentiate between query-side and document-side encoding.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "auto",
        max_seq_length: int = 512,
        batch_size: int = 16,
        query_instruction: str = "Retrieve relevant source code based on the user query",
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu" and torch.cuda.is_available():
            logger.warning(
                "CUDA is available but embedder is on CPU — embedding will be slow. "
                "Set device='cuda' or device='auto' in your config."
            )

        self._device = torch.device(device)
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._query_instruction = query_instruction

        dtype = torch_dtype or (torch.float16 if self._device.type == "cuda" else torch.float32)

        logger.info("Loading embedding model %s on %s (dtype=%s)", model_name, device, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

        # Determine dimension from a dummy forward pass
        self._dim: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = self._model.config.hidden_size
        return self._dim

    # ── internal helpers ────────────────────────────────────────────────

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts and return normalised embeddings."""
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoded)

        embeddings = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _batched_encode(self, texts: List[str]) -> np.ndarray:
        """Encode *texts* in batches and return a numpy array."""
        all_embs: List[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            embs = self._encode_batch(batch)
            all_embs.append(embs.cpu().float().numpy())
            if start % (self._batch_size * 10) == 0 and start > 0:
                logger.info("Embedded %d / %d texts", start + len(batch), len(texts))
        return np.vstack(all_embs) if all_embs else np.empty((0, self.dimension), dtype=np.float32)

    # ── public API ──────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed document (code) texts — no instruction prefix."""
        return self._batched_encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with the instruction prefix."""
        prefixed = f"Instruct: {self._query_instruction}\nQuery: {query}"
        return self._batched_encode([prefixed])[0]


# ── Sentence-Transformers fallback ──────────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """Lightweight embedder using the ``sentence-transformers`` library.

    Use this for quick testing or when GPU resources are limited.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 64,
        query_prefix: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device
        self._batch_size = batch_size
        # Some code embedders (e.g. CodeRankEmbed) were trained with a fixed
        # query-side prefix; without it their retrieval quality degrades.
        self._query_prefix = query_prefix
        logger.info("Loading sentence-transformers model %s on %s", model_name, device)
        self._model = SentenceTransformer(
            model_name, device=device, trust_remote_code=trust_remote_code
        )
        self._dim: int = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        text = f"{self._query_prefix}{query}" if self._query_prefix else query
        emb = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb[0], dtype=np.float32)


# ── Factory ─────────────────────────────────────────────────────────────

def create_embedder(
    model_name: str,
    device: str = "auto",
    max_seq_length: int = 512,
    batch_size: int = 16,
    query_instruction: str = "Retrieve relevant source code based on the user query",
    torch_dtype: Optional[torch.dtype] = None,
    query_prefix: str = "",
    trust_remote_code: bool = False,
) -> BaseEmbedder:
    """Instantiate the correct embedder based on *model_name*.

    Models whose path contains ``"Qwen3-Embedding"`` use the Qwen3 path;
    everything else falls back to ``SentenceTransformerEmbedder``.
    """
    if "Qwen3-Embedding" in model_name or "qwen3-embedding" in model_name.lower():
        return Qwen3Embedder(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            query_instruction=query_instruction,
            torch_dtype=torch_dtype,
        )

    try:
        return SentenceTransformerEmbedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            query_prefix=query_prefix,
            trust_remote_code=trust_remote_code,
        )
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for non-Qwen3 embedding models. "
            "Install with: pip install sentence-transformers"
        )
