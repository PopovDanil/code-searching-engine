"""Tests for retrieval/search module."""

import json
import os
import tempfile

import numpy as np
from retrieval.search import _compute_metadata_bonus, SearchResult
from parser.extract import CodeEntity
from indexing.faiss_index import FaissCodeIndex
from retrieval.reranker import NoOpReranker
from retrieval.query_rewriter import NoOpQueryRewriter


def _make_entity(function_name="test_func", class_name=None, docstring=None, language="python"):
    return CodeEntity(
        repository="repo",
        file_path="f.py",
        language=language,
        entity_type="function",
        function_name=function_name,
        class_name=class_name,
        signature="def test_func():",
        docstring=docstring,
        source_code="def test_func(): pass",
        start_line=1,
        end_line=1,
    )


def test_metadata_bonus_exact_name():
    entity = _make_entity(function_name="read_json")
    bonus = _compute_metadata_bonus(entity, "read json")
    assert bonus >= 0.4  # name match + exact match


def test_metadata_bonus_name_contains_word():
    entity = _make_entity(function_name="read_json_file")
    bonus = _compute_metadata_bonus(entity, "read json")
    assert bonus >= 0.4


def test_metadata_bonus_no_match():
    entity = _make_entity(function_name="write_csv")
    bonus = _compute_metadata_bonus(entity, "read json")
    assert bonus == 0.0


def test_metadata_bonus_docstring():
    entity = _make_entity(
        function_name="func",
        docstring="Reads a JSON file from disk",
    )
    bonus = _compute_metadata_bonus(entity, "read json file")
    assert bonus > 0.0


def test_metadata_bonus_max_1():
    entity = _make_entity(
        function_name="read_json",
        docstring="Read JSON",
    )
    bonus = _compute_metadata_bonus(entity, "read json")
    assert bonus <= 1.0


def test_noop_query_rewriter_preserves_query():
    assert NoOpQueryRewriter().rewrite("read json") == "read json"


def test_search_uses_rewritten_query_for_embedding_and_metadata():
    from config import CodeSearchConfig
    from retrieval.search import SearchEngine

    class FakeRewriter:
        def rewrite(self, query):
            assert query == "original"
            return "read json"

    class FakeEmbedder:
        def __init__(self):
            self.query = None

        def embed_query(self, query):
            self.query = query
            return np.array([1.0, 0.0], dtype=np.float32)

    entity = _make_entity(function_name="read_json")
    index = FaissCodeIndex(dimension=2, index_type="flat")
    index.build(np.array([[1.0, 0.0]], dtype=np.float32), [entity])
    embedder = FakeEmbedder()
    engine = SearchEngine(
        config=CodeSearchConfig(enable_reranking=False),
        embedder=embedder,
        reranker=NoOpReranker(),
        query_rewriter=FakeRewriter(),
        index=index,
    )

    results = engine.search("original")
    assert embedder.query == "read json"
    assert results[0].metadata_bonus >= 0.4


def test_search_engine_loads_manifest():
    """SearchEngine loads per-language indexes from a manifest."""
    from config import CodeSearchConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        dim = 8
        # Create two per-language indexes
        for lang in ("python", "go"):
            idx = FaissCodeIndex(dimension=dim, index_type="flat")
            embeddings = np.random.rand(3, dim).astype(np.float32)
            entities = [_make_entity(f"{lang}_f{i}", language=lang) for i in range(3)]
            idx.build(embeddings, entities)
            idx.save(os.path.join(tmpdir, lang))

        with open(os.path.join(tmpdir, "manifest.json"), "w") as fh:
            json.dump({"languages": ["python", "go"]}, fh)

        config = CodeSearchConfig(index_dir=tmpdir, separate_indexes=True)
        from retrieval.search import SearchEngine
        engine = SearchEngine(config=config)

        langs = engine._load_manifest()
        assert sorted(langs) == ["go", "python"]
        assert "python" in engine._language_indexes
        assert "go" in engine._language_indexes
        assert engine._language_indexes["python"].ntotal == 3


def test_search_engine_language_filter():
    """When language filter is set, only that index is queried."""
    from config import CodeSearchConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        dim = 8
        for lang in ("python", "go"):
            idx = FaissCodeIndex(dimension=dim, index_type="flat")
            embeddings = np.random.rand(5, dim).astype(np.float32)
            entities = [_make_entity(f"{lang}_f{i}", language=lang) for i in range(5)]
            idx.build(embeddings, entities)
            idx.save(os.path.join(tmpdir, lang))

        with open(os.path.join(tmpdir, "manifest.json"), "w") as fh:
            json.dump({"languages": ["python", "go"]}, fh)

        config = CodeSearchConfig(
            index_dir=tmpdir,
            separate_indexes=True,
            retrieval_top_k=100,
            enable_reranking=False,
        )
        from retrieval.search import SearchEngine
        engine = SearchEngine(config=config)

        # Search only in python
        query_vec = np.random.rand(dim).astype(np.float32)
        candidates = engine._retrieve_separate(query_vec, language="python")
        assert len(candidates) > 0
        for entity, _score in candidates:
            assert entity.language == "python"


def test_search_engine_merges_all_languages():
    """Without language filter, results from all languages are merged."""
    from config import CodeSearchConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        dim = 8
        for lang in ("python", "java"):
            idx = FaissCodeIndex(dimension=dim, index_type="flat")
            embeddings = np.random.rand(5, dim).astype(np.float32)
            entities = [_make_entity(f"{lang}_f{i}", language=lang) for i in range(5)]
            idx.build(embeddings, entities)
            idx.save(os.path.join(tmpdir, lang))

        with open(os.path.join(tmpdir, "manifest.json"), "w") as fh:
            json.dump({"languages": ["python", "java"]}, fh)

        config = CodeSearchConfig(
            index_dir=tmpdir,
            separate_indexes=True,
            retrieval_top_k=100,
        )
        from retrieval.search import SearchEngine
        engine = SearchEngine(config=config)

        query_vec = np.random.rand(dim).astype(np.float32)
        candidates = engine._retrieve_separate(query_vec, language=None)
        assert len(candidates) > 0
        # Should have results from both languages
        languages_found = {e.language for e, _ in candidates}
        assert "python" in languages_found
        assert "java" in languages_found
