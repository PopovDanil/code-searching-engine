"""Tests for retrieval/search module."""

from retrieval.search import _compute_metadata_bonus, SearchResult
from parser.extract import CodeEntity


def _make_entity(function_name="test_func", class_name=None, docstring=None):
    return CodeEntity(
        repository="repo",
        file_path="f.py",
        language="python",
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
