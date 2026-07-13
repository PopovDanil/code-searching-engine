"""Prompt-format tests that do not load the reranker model."""

from retrieval.reranker import Qwen3Reranker
from parser.extract import CodeEntity


def _uninitialised_reranker(language_hint):
    reranker = object.__new__(Qwen3Reranker)
    reranker._language_hint = language_hint
    reranker._instruction = "Judge code relevance."
    return reranker


def test_language_hint_disabled_preserves_prompt():
    pair = _uninitialised_reranker(False)._format_pair(
        "read json", "def read_json(): pass", "python",
    )
    assert pair == (
        "<Instruct>: Judge code relevance.\n"
        "<Query>: read json\n"
        "<Document>: def read_json(): pass"
    )


def test_language_hint_enabled_adds_language():
    pair = _uninitialised_reranker(True)._format_pair(
        "read json", "def read_json(): pass", "python",
    )
    assert "The document is source code written in Python.\n" in pair
    assert pair.index("written in Python") < pair.index("<Document>")


def test_rerank_passes_entity_language_to_formatted_pair():
    reranker = _uninitialised_reranker(True)
    reranker._include_docstring = True
    reranker._batch_size = 1
    captured = []
    reranker._score_batch = lambda pairs: captured.extend(pairs) or [0.9]
    entity = CodeEntity(
        repository="repo",
        file_path="main.py",
        language="python",
        entity_type="function",
        function_name="read_json",
        class_name=None,
        signature="def read_json():",
        docstring=None,
        source_code="def read_json(): pass",
        start_line=1,
        end_line=1,
    )

    reranker.rerank("read json", [(entity, 0.5)], batch_size=1)

    assert len(captured) == 1
    assert "The document is source code written in Python." in captured[0]
