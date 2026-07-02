"""Cross-encoder reranker abstraction and implementations.

The default reranker is Qwen3-Reranker-8B, which formulates reranking
as a binary yes/no relevance classification task.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from parser.extract import CodeEntity

logger = logging.getLogger(__name__)


# ── Abstract interface ──────────────────────────────────────────────────

class BaseReranker(ABC):
    """Interface for reranker models."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[CodeEntity, float]],
        batch_size: int = 16,
    ) -> List[Tuple[CodeEntity, float]]:
        """Rerank *candidates* and return them sorted by relevance.

        Parameters
        ----------
        query:
            Natural-language search query.
        candidates:
            List of ``(entity, embedding_similarity)`` pairs.
        batch_size:
            Batch size for reranking inference.

        Returns
        -------
        List[Tuple[CodeEntity, float]]
            Entities paired with reranker relevance scores, sorted
            descending.
        """


# ── Qwen3-Reranker implementation ───────────────────────────────────────

class Qwen3Reranker(BaseReranker):
    """Reranker using Qwen3-Reranker-8B (or any Qwen3-Reranker variant).

    The model receives (query, document) pairs formatted as a chat prompt
    and outputs the probability of ``"yes"`` as the relevance score.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-8B",
        device: str = "auto",
        max_seq_length: int = 2048,
        batch_size: int = 16,
        torch_dtype: Optional[torch.dtype] = None,
        include_docstring: bool = True,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._include_docstring = include_docstring

        dtype = torch_dtype or (torch.float16 if self._device.type == "cuda" else torch.float32)

        logger.info("Loading reranker model %s on %s (dtype=%s)", model_name, device, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

        # Resolve yes/no token ids
        self._yes_token_id = self._tokenizer("yes", add_special_tokens=False)["input_ids"][0]
        self._no_token_id = self._tokenizer("no", add_special_tokens=False)["input_ids"][0]

    # ── internal ─────────────────────────────────────────────────────────

    def _format_prompt(self, query: str, document: str) -> str:
        """Build the chat prompt for the reranker."""
        return (
            "<|im_start|>system\n"
            'Judge whether the Document is relevant to the Query. Answer only "yes" or "no".'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Query>{query}</Query>\n"
            f"<Document>{document}</Document>"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _score_batch(self, prompts: List[str]) -> List[float]:
        """Score a batch of prompts, returning the P("yes") for each."""
        encoded = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoded)

        logits = outputs.logits[:, -1, :]  # last token logits
        yes_no_logits = logits[:, [self._yes_token_id, self._no_token_id]]
        probs = torch.softmax(yes_no_logits, dim=-1)
        scores = probs[:, 0].cpu().tolist()  # probability of "yes"
        return scores

    # ── public API ───────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[CodeEntity, float]],
        batch_size: int = 16,
    ) -> List[Tuple[CodeEntity, float]]:
        """Rerank candidates by relevance to *query*."""
        if not candidates:
            return []

        scored: List[Tuple[CodeEntity, float]] = []
        bs = batch_size or self._batch_size

        for start in range(0, len(candidates), bs):
            batch = candidates[start : start + bs]
            prompts = [
                self._format_prompt(query, ent.to_structured_text(include_docstring=self._include_docstring))
                for ent, _ in batch
            ]
            scores = self._score_batch(prompts)
            for (ent, _), score in zip(batch, scores):
                scored.append((ent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ── No-op reranker (when reranking is disabled) ─────────────────────────

class NoOpReranker(BaseReranker):
    """Pass-through reranker that preserves embedding similarity scores."""

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[CodeEntity, float]],
        batch_size: int = 16,
    ) -> List[Tuple[CodeEntity, float]]:
        return sorted(candidates, key=lambda x: x[1], reverse=True)


# ── Factory ─────────────────────────────────────────────────────────────

def create_reranker(
    model_name: str,
    device: str = "auto",
    max_seq_length: int = 2048,
    batch_size: int = 16,
    enabled: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
    include_docstring: bool = True,
) -> BaseReranker:
    """Instantiate the correct reranker based on *model_name*."""
    if not enabled:
        logger.info("Reranking disabled — using no-op reranker")
        return NoOpReranker()

    if "Qwen3-Reranker" in model_name or "qwen3-reranker" in model_name.lower():
        return Qwen3Reranker(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            include_docstring=include_docstring,
        )

    # Fallback: use Qwen3-Reranker for any unrecognised name
    logger.warning("Unrecognised reranker model %s; defaulting to Qwen3-Reranker", model_name)
    return Qwen3Reranker(
        model_name=model_name,
        device=device,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        include_docstring=include_docstring,
    )
