"""Cross-encoder reranker abstraction and implementations.

The default reranker is Qwen3-Reranker-8B, which formulates reranking
as a binary yes/no relevance classification task.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from console import console, log_model_loaded, log_model_loading
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

# Official Qwen3-Reranker chat template. The prefix and suffix are
# tokenized separately so that truncation can never eat the assistant
# tag or the <think> block the model was trained with.
_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_DEFAULT_INSTRUCTION = (
    "Given a natural-language search query, judge whether the code "
    "snippet implements the functionality described in the query."
)


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
        language_hint: bool = False,
        instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu" and torch.cuda.is_available():
            logger.warning(
                "CUDA is available but reranker is on CPU — reranking will be slow. "
                "Set device='cuda' or device='auto' in your config."
            )

        self._device = torch.device(device)
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._include_docstring = include_docstring
        self._language_hint = language_hint
        self._instruction = instruction

        if torch_dtype is not None:
            dtype = torch_dtype
        elif self._device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32

        log_model_loading(console, model_name, device, str(dtype))
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
        log_model_loaded(console, model_name)

        # Resolve yes/no token ids
        self._yes_token_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_token_id = self._tokenizer.convert_tokens_to_ids("no")

        # Pre-tokenize the fixed prompt parts; only the pair text in the
        # middle is ever truncated.
        self._prefix_ids = self._tokenizer.encode(_PROMPT_PREFIX, add_special_tokens=False)
        self._suffix_ids = self._tokenizer.encode(_PROMPT_SUFFIX, add_special_tokens=False)
        self._pair_budget = max_seq_length - len(self._prefix_ids) - len(self._suffix_ids)
        if self._pair_budget <= 0:
            raise ValueError(
                f"max_seq_length={max_seq_length} is too small for the reranker "
                f"prompt template ({len(self._prefix_ids) + len(self._suffix_ids)} "
                "tokens of fixed prefix/suffix)"
            )

    # ── internal ─────────────────────────────────────────────────────────

    def _format_pair(
        self,
        query: str,
        document: str,
        language: Optional[str] = None,
    ) -> str:
        """Build the instruction/query/document block (without chat tags)."""
        language_line = ""
        if self._language_hint and language:
            language_line = (
                f"The document is source code written in {language.capitalize()}.\n"
            )
        return (
            f"<Instruct>: {self._instruction}\n"
            f"<Query>: {query}\n"
            f"{language_line}"
            f"<Document>: {document}"
        )

    def _score_batch(self, pairs: List[str]) -> List[float]:
        """Score a batch of pair texts, returning the P("yes") for each."""
        encoded = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            max_length=self._pair_budget,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        input_ids = [
            self._prefix_ids + ids + self._suffix_ids
            for ids in encoded["input_ids"]
        ]
        batch = self._tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**batch)

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
            pairs = [
                self._format_pair(
                    query,
                    ent.to_structured_text(include_docstring=self._include_docstring),
                    ent.language,
                )
                for ent, _ in batch
            ]
            scores = self._score_batch(pairs)
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
    language_hint: bool = False,
    instruction: str = _DEFAULT_INSTRUCTION,
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
            language_hint=language_hint,
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
        language_hint=language_hint,
        instruction=instruction,
    )
