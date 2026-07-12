"""Tests for BM25 tokenization, BM25 index, and RRF fusion."""

from retrieval.bm25_index import BM25Index, tokenize_code
from retrieval.fusion import reciprocal_rank_fusion
from parser.extract import CodeEntity


def _ent(name):
    return CodeEntity(
        repository="r", file_path="f.py", language="python",
        entity_type="function", function_name=name, class_name=None,
        signature="", docstring=None, source_code=f"def {name}(): pass",
        start_line=1, end_line=1,
    )


# ── tokenizer ────────────────────────────────────────────────────────────

def test_tokenize_snake_case():
    assert tokenize_code("read_json_file") == ["read", "json", "file"]


def test_tokenize_camel_case():
    assert tokenize_code("parseHttpResponse") == ["parse", "http", "response"]


def test_tokenize_acronym_boundary():
    assert tokenize_code("HTTPServer") == ["http", "server"]


def test_tokenize_mixed_and_punctuation():
    assert tokenize_code("def read_json(path):") == ["def", "read", "json", "path"]


def test_query_matches_identifier_after_split():
    # A natural-language query overlaps a snake_case identifier only after split
    assert set(tokenize_code("read json")) <= set(tokenize_code("read_json_file"))


# ── RRF fusion ───────────────────────────────────────────────────────────

def test_rrf_agreement_ranks_first():
    # id "a" is top of both lists -> must win
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]], k=60)
    assert fused[0][0] == "a"


def test_rrf_dedupes_across_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    ids = [item for item, _ in fused]
    assert sorted(ids) == ["a", "b"]


def test_rrf_zero_weight_disables_list():
    # With the second list weighted 0, ranking follows the first list only
    fused = reciprocal_rank_fusion(
        [["a", "b"], ["b", "a"]], k=60, weights=[1.0, 0.0]
    )
    assert [item for item, _ in fused] == ["a", "b"]


def test_rrf_weight_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


# ── BM25 index ───────────────────────────────────────────────────────────

def test_bm25_finds_relevant_entity():
    ents = [_ent("read_json_file"), _ent("bubble_sort"), _ent("connect_database")]
    texts = [
        "Function: read_json_file\ndef read_json_file(path): ...",
        "Function: bubble_sort\ndef bubble_sort(arr): ...",
        "Function: connect_database\ndef connect_database(): ...",
    ]
    idx = BM25Index()
    idx.build(texts, ents)
    results = idx.search("read json file", top_k=3)
    assert results[0][0].function_name == "read_json_file"


def test_bm25_empty_query_returns_empty():
    idx = BM25Index()
    idx.build(["Function: foo"], [_ent("foo")])
    assert idx.search("!!!", top_k=5) == []
