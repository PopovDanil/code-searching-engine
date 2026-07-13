"""Prompt-format tests that do not load the reranker model."""

from retrieval.reranker import Qwen3Reranker

_INSTRUCTION = "judge relevance"


def _uninitialised_reranker(language_hint):
    reranker = object.__new__(Qwen3Reranker)
    reranker._language_hint = language_hint
    reranker._instruction = _INSTRUCTION
    return reranker


def test_language_hint_disabled_preserves_prompt():
    prompt = _uninitialised_reranker(False)._format_pair(
        "read json", "def read_json(): pass", "python",
    )
    assert "written in Python" not in prompt


def test_language_hint_enabled_adds_language():
    prompt = _uninitialised_reranker(True)._format_pair(
        "read json", "def read_json(): pass", "python",
    )
    assert "The document is source code written in Python." in prompt


def test_language_hint_enabled_without_language_is_noop():
    prompt = _uninitialised_reranker(True)._format_pair(
        "read json", "def read_json(): pass",
    )
    assert "written in" not in prompt


def test_instruction_included_in_pair():
    prompt = _uninitialised_reranker(False)._format_pair(
        "read json", "def read_json(): pass", "python",
    )
    assert _INSTRUCTION in prompt
