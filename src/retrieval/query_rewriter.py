"""Optional LLM-based query rewriting for code retrieval."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class BaseQueryRewriter(ABC):
    """Interface for retrieval-query transformations."""

    @abstractmethod
    def rewrite(self, query: str) -> str:
        """Return the text that should be used for retrieval."""


class NoOpQueryRewriter(BaseQueryRewriter):
    """Return the original query unchanged."""

    def rewrite(self, query: str) -> str:
        return query


class LLMQueryRewriter(BaseQueryRewriter):
    """Rewrite a query with a small instruction-tuned causal language model."""

    def __init__(
        self,
        model_name: str,
        strategy: str = "rewrite",
        device: str = "auto",
        max_new_tokens: int = 128,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if strategy not in {"rewrite", "hyde"}:
            raise ValueError("LLM query rewriting strategy must be rewrite or hyde")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._strategy = strategy
        self._device = torch.device(device)
        self._max_new_tokens = max_new_tokens
        dtype = torch_dtype or (
            torch.float16 if self._device.type == "cuda" else torch.float32
        )

        logger.info(
            "Loading query rewriter %s on %s (strategy=%s, dtype=%s)",
            model_name, device, strategy, dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype,
        ).to(self._device)
        self._model.eval()

    def _instruction(self, query: str) -> str:
        if self._strategy == "hyde":
            return (
                "Generate a short hypothetical description of source code that would "
                "answer the user's request. Mention likely APIs, identifiers, and behavior. "
                "Return only the description, not a full implementation.\n\n"
                f"User request: {query}"
            )
        return (
            "Rewrite the user's natural-language request as one concise, precise semantic "
            "code-search query. Preserve the intent and add useful programming terms. "
            "Return only the rewritten query.\n\n"
            f"User request: {query}"
        )

    def rewrite(self, query: str) -> str:
        messages = [{"role": "user", "content": self._instruction(query)}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        input_length = encoded["input_ids"].shape[1]
        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        rewritten = self._tokenizer.decode(
            output[0, input_length:], skip_special_tokens=True,
        ).strip()
        if not rewritten:
            logger.warning("Query rewriter returned empty text; using original query")
            return query
        return rewritten


def create_query_rewriter(
    enabled: bool,
    strategy: str,
    model_name: str,
    device: str = "auto",
    max_new_tokens: int = 128,
    torch_dtype: Optional[torch.dtype] = None,
) -> BaseQueryRewriter:
    """Create a no-op or LLM query rewriter from configuration."""
    if not enabled or strategy == "none":
        return NoOpQueryRewriter()
    return LLMQueryRewriter(
        model_name=model_name,
        strategy=strategy,
        device=device,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
    )
