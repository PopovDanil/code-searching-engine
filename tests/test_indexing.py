"""Tests for indexing module (FAISS index)."""

import json
import os
import tempfile

import numpy as np
from parser.extract import CodeEntity
from indexing.faiss_index import FaissCodeIndex


def _make_entity(name="func", language="python"):
    return CodeEntity(
        repository="repo",
        file_path="f.py",
        language=language,
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


def test_separate_indexes_save_and_load():
    """Per-language indexes can be saved to subdirectories and loaded back."""
    dim = 8
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build and save two separate language indexes
        for lang in ("python", "go"):
            index = FaissCodeIndex(dimension=dim, index_type="flat")
            embeddings = np.random.rand(5, dim).astype(np.float32)
            entities = [_make_entity(f"{lang}_func_{i}", language=lang) for i in range(5)]
            index.build(embeddings, entities)
            index.save(os.path.join(tmpdir, lang))

        # Write manifest
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump({"languages": ["python", "go"]}, fh)

        # Verify files exist
        assert os.path.isfile(os.path.join(tmpdir, "python", "index.faiss"))
        assert os.path.isfile(os.path.join(tmpdir, "python", "metadata.pkl"))
        assert os.path.isfile(os.path.join(tmpdir, "go", "index.faiss"))
        assert os.path.isfile(os.path.join(tmpdir, "go", "metadata.pkl"))
        assert os.path.isfile(manifest_path)

        # Load and verify
        py_index = FaissCodeIndex.load(os.path.join(tmpdir, "python"))
        assert py_index.ntotal == 5
        assert py_index.metadata[0].language == "python"

        go_index = FaissCodeIndex.load(os.path.join(tmpdir, "go"))
        assert go_index.ntotal == 5
        assert go_index.metadata[0].language == "go"

        # Search within a language index
        query = np.random.rand(dim).astype(np.float32)
        results = py_index.search(query, top_k=3)
        assert len(results) == 3
        for entity, _score in results:
            assert entity.language == "python"


def test_separate_indexes_manifest_content():
    """Manifest JSON contains the correct language list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump({"languages": ["java", "javascript", "python"]}, fh)

        with open(manifest_path, "r") as fh:
            manifest = json.load(fh)

        assert sorted(manifest["languages"]) == ["java", "javascript", "python"]
