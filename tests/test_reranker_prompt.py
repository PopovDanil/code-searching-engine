"""Prompt-format tests that do not load the reranker model."""

from retrieval.reranker import Qwen3Reranker


def _uninitialised_reranker(language_hint):
    reranker = object.__new__(Qwen3Reranker)
    reranker._language_hint = language_hint
    return reranker


def test_language_hint_disabled_preserves_prompt():
    prompt = _uninitialised_reranker(False)._format_prompt(
        "read json", "def read_json(): pass", "python",
    )
    assert "written in Python" not in prompt


def test_language_hint_enabled_adds_language():
    prompt = _uninitialised_reranker(True)._format_prompt(
        "read json", "def read_json(): pass", "python",
    )
    assert "The document is source code written in Python." in prompt
