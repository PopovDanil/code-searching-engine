"""Tests for indexing module (FAISS index)."""

import numpy as np
from parser.extract import CodeEntity
from indexing.faiss_index import FaissCodeIndex


def _make_entity(name="func"):
    return CodeEntity(
        repository="repo",
        file_path="f.py",
        language="python",
        entity_type="function",
        function_name=name,
        class_name=None,
        signature=f"def {name}():",
        docstring=None,
        source_code=f"def {name}(): pass",
        start_line=1,
        end_line=1,
    )


def test_flat_index_build_and_search():
    dim = 8
    index = FaissCodeIndex(dimension=dim, index_type="flat")
    embeddings = np.random.rand(10, dim).astype(np.float32)
    entities = [_make_entity(f"func_{i}") for i in range(10)]
    index.build(embeddings, entities)
    assert index.ntotal == 10

    query = np.random.rand(dim).astype(np.float32)
    results = index.search(query, top_k=3)
    assert len(results) == 3
    for entity, score in results:
        assert isinstance(entity, CodeEntity)
        assert isinstance(score, float)


def test_index_search_empty():
    index = FaissCodeIndex(dimension=8, index_type="flat")
    query = np.random.rand(8).astype(np.float32)
    results = index.search(query, top_k=5)
    assert results == []


def test_index_dimension():
    index = FaissCodeIndex(dimension=16, index_type="flat")
    assert index.dimension == 16
