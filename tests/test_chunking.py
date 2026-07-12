"""Focused tests for AST-first recursive code chunking."""

from __future__ import annotations

import pytest

from config import CodeSearchConfig
from parser.chunker import language_aware_recursive_chunk_node, recursive_chunk_node
from parser.extract import extract_entities
from parser.parser import get_parser


def _python_node(source: str, node_type: str = "function_definition"):
    """Return the first node of *node_type* from a parsed Python source."""
    tree = get_parser("python").parse(source.encode("utf-8"))
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == node_type:
            return tree, node
        stack.extend(reversed(node.children))
    raise AssertionError(f"No {node_type!r} node found in test source")


def _assert_span_matches_source(span, source_bytes: bytes) -> None:
    assert span.start_byte < span.end_byte
    assert span.text == source_bytes[span.start_byte : span.end_byte].decode("utf-8")
    assert span.start_line == source_bytes[: span.start_byte].count(b"\n") + 1

    # ``end_line`` is inclusive. Looking at the byte immediately before the
    # exclusive end offset avoids assigning a trailing newline to the next line.
    assert span.end_line == source_bytes[: span.end_byte - 1].count(b"\n") + 1


def test_recursive_chunking_config_defaults():
    config = CodeSearchConfig()

    assert config.max_chunk_chars == 1500
    assert config.chunk_overlap_chars == 150


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chunk_chars": 0, "chunk_overlap_chars": 0},
        {"max_chunk_chars": 10, "chunk_overlap_chars": -1},
        {"max_chunk_chars": 10, "chunk_overlap_chars": 10},
        {"max_chunk_chars": 10.5, "chunk_overlap_chars": 1},
    ],
)
def test_recursive_chunking_config_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        CodeSearchConfig(**kwargs)


def test_recursive_chunking_can_be_disabled():
    config = CodeSearchConfig(max_chunk_chars=None)

    assert config.max_chunk_chars is None


@pytest.mark.parametrize("chunker_type", ["unsupported", None, 1])
def test_recursive_chunking_config_rejects_invalid_chunker_type(chunker_type):
    with pytest.raises(ValueError):
        CodeSearchConfig(chunker_type=chunker_type)


def test_recursive_chunking_revalidates_mutated_config():
    config = CodeSearchConfig()
    config.max_chunk_chars = 10
    config.chunk_overlap_chars = 10

    with pytest.raises(ValueError):
        config.validate_chunking()


def test_recursive_chunking_config_loads_from_yaml(tmp_path):
    config_path = tmp_path / "chunking.yaml"
    config_path.write_text(
        "max_chunk_chars: 80\nchunk_overlap_chars: 8\n",
        encoding="utf-8",
    )

    config = CodeSearchConfig.from_yaml(str(config_path))

    assert config.max_chunk_chars == 80
    assert config.chunk_overlap_chars == 8


def test_under_limit_node_is_returned_as_one_identity_span():
    source = "# module comment\n\ndef greet(name):\n    return f'Hello, {name}!'\n"
    source_bytes = source.encode("utf-8")
    _tree, node = _python_node(source)
    node_text = source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    spans = recursive_chunk_node(
        node,
        source_bytes,
        max_chars=len(node_text),
    )

    assert len(spans) == 1
    span = spans[0]
    assert span.start_byte == node.start_byte
    assert span.end_byte == node.end_byte
    assert span.start_line == 3
    assert span.end_line == 4
    assert span.text == node_text
    _assert_span_matches_source(span, source_bytes)


def test_under_limit_node_stays_whole_when_overlap_is_configured():
    source = "def greet():\n    return 'hello'"
    source_bytes = source.encode("utf-8")
    _tree, node = _python_node(source)

    spans = recursive_chunk_node(
        node,
        source_bytes,
        max_chars=len(source) + 1,
        overlap_chars=5,
    )

    assert len(spans) == 1
    assert spans[0].text == source


def test_oversized_node_is_split_recursively_in_source_order():
    source = (
        "# preface\n"
        "\n"
        "def transform(value):\n"
        "    first = value + 1\n"
        "    second = first * 2\n"
        "    third = second - 3\n"
        "    fourth = third / 4\n"
        "    return fourth\n"
    )
    source_bytes = source.encode("utf-8")
    _tree, node = _python_node(source)
    max_chars = 38

    spans = recursive_chunk_node(node, source_bytes, max_chars=max_chars)

    assert len(spans) > 1
    assert spans[0].start_byte == node.start_byte
    assert spans[-1].end_byte == node.end_byte
    assert all(len(span.text) <= max_chars for span in spans)
    assert all(left.end_byte == right.start_byte for left, right in zip(spans, spans[1:]))
    assert "".join(span.text for span in spans) == source_bytes[
        node.start_byte : node.end_byte
    ].decode("utf-8")
    assert [(span.start_byte, span.end_byte) for span in spans] == sorted(
        (span.start_byte, span.end_byte) for span in spans
    )
    for span in spans:
        _assert_span_matches_source(span, source_bytes)

    # Chunking is deterministic, including the chosen AST boundaries.
    repeated = recursive_chunk_node(node, source_bytes, max_chars=max_chars)
    assert [
        (span.start_byte, span.end_byte, span.start_line, span.end_line, span.text)
        for span in repeated
    ] == [
        (span.start_byte, span.end_byte, span.start_line, span.end_line, span.text)
        for span in spans
    ]


def test_oversized_leaf_uses_exact_character_fallback_and_terminates():
    identifier = "abcdefghijklmnopqrstuvwxyz0123456789X"
    source_bytes = identifier.encode("utf-8")
    _tree, leaf = _python_node(identifier, "identifier")

    spans = recursive_chunk_node(leaf, source_bytes, max_chars=10)

    assert [span.text for span in spans] == [
        "abcdefghij",
        "klmnopqrst",
        "uvwxyz0123",
        "456789X",
    ]
    assert [(span.start_byte, span.end_byte) for span in spans] == [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 37),
    ]
    assert all((span.start_line, span.end_line) == (1, 1) for span in spans)


def test_overlap_is_exact_for_oversized_leaf():
    source = "abcdefghijklmnopqrstuvwxyz"
    source_bytes = source.encode("utf-8")
    _tree, leaf = _python_node(source, "identifier")

    spans = recursive_chunk_node(
        leaf,
        source_bytes,
        max_chars=10,
        overlap_chars=3,
    )

    assert len(spans) > 1
    assert all(len(span.text) <= 10 for span in spans)
    assert all(
        left.end_byte - right.start_byte == 3
        for left, right in zip(spans, spans[1:])
    )
    assert spans[0].start_byte == leaf.start_byte
    assert spans[-1].end_byte == leaf.end_byte
    assert all(
        left.end_byte < right.end_byte for left, right in zip(spans, spans[1:])
    )
    for span in spans:
        _assert_span_matches_source(span, source_bytes)


def test_high_overlap_uses_full_first_window_without_nested_prefix_chunks():
    source = "abcdefghijklmnopqrst"
    source_bytes = source.encode("utf-8")
    _tree, leaf = _python_node(source, "identifier")

    spans = recursive_chunk_node(
        leaf,
        source_bytes,
        max_chars=10,
        overlap_chars=9,
    )

    assert spans[0].text == "abcdefghij"
    assert [span.start_byte for span in spans] == list(range(11))
    assert all(len(span.text) == 10 for span in spans)


def test_language_aware_chunker_respects_python_structure():
    source = (
        "def greet(name):\n"
        "    if name:\n"
        "        return name\n"
        "    return 'hello'\n"
    )
    source_bytes = source.encode("utf-8")
    _tree, node = _python_node(source)

    spans = language_aware_recursive_chunk_node(
        node,
        source_bytes,
        max_chars=24,
        language="python",
    )

    assert len(spans) > 1
    assert all(len(span.text) <= 24 for span in spans)
    assert any("if name:" in span.text for span in spans)


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_limits_are_rejected(max_chars: int, overlap_chars: int):
    source_bytes = b"identifier"
    _tree, leaf = _python_node(source_bytes.decode("ascii"), "identifier")

    with pytest.raises(ValueError):
        recursive_chunk_node(
            leaf,
            source_bytes,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )


def test_unicode_uses_character_limit_but_preserves_utf8_byte_spans():
    identifier = "переменнаяпеременная"
    source = f"# π prefix\n{identifier}"
    source_bytes = source.encode("utf-8")
    _tree, leaf = _python_node(source, "identifier")

    spans = recursive_chunk_node(leaf, source_bytes, max_chars=6)

    assert len(spans) > 1
    assert "".join(span.text for span in spans) == identifier
    assert all(len(span.text) <= 6 for span in spans)
    assert spans[0].start_byte == len("# π prefix\n".encode("utf-8"))
    assert spans[-1].end_byte == len(source_bytes)
    assert all((span.start_line, span.end_line) == (2, 2) for span in spans)
    for span in spans:
        _assert_span_matches_source(span, source_bytes)


def test_crlf_source_keeps_exact_coverage_and_line_numbers():
    source = (
        "def calculate(value):\r\n"
        "    first = value + 1\r\n"
        "    second = first * 2\r\n"
        "    return second"
    )
    source_bytes = source.encode("utf-8")
    _tree, node = _python_node(source)

    spans = recursive_chunk_node(node, source_bytes, max_chars=35)

    assert len(spans) > 1
    assert "".join(span.text for span in spans) == source
    assert spans[0].start_line == 1
    assert spans[-1].end_line == 4
    assert all(len(span.text) <= 35 for span in spans)
    for span in spans:
        _assert_span_matches_source(span, source_bytes)


def test_extract_entities_wraps_chunks_and_preserves_entity_metadata():
    source = (
        "# preface\n"
        "\n"
        "def calculate(value):\n"
        "    \"\"\"Keep the calculation documented.\"\"\"\n"
        "    first = value + 100\n"
        "    second = first * 200\n"
        "    third = second - 300\n"
        "    return third\n"
    )
    tree = get_parser("python").parse(source.encode("utf-8"))
    max_chars = 42

    chunks = extract_entities(
        source_code=source,
        tree=tree,
        language="python",
        repository="example-repository",
        file_path="pkg/calculation.py",
        max_chunk_chars=max_chars,
        chunk_overlap_chars=0,
    )

    assert len(chunks) > 1
    assert all(len(chunk.source_code) <= max_chars for chunk in chunks)
    assert [chunk.start_line for chunk in chunks] == sorted(
        chunk.start_line for chunk in chunks
    )
    assert chunks[0].start_line == 3
    assert chunks[-1].end_line == 8
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    for chunk in chunks:
        assert chunk.repository == "example-repository"
        assert chunk.file_path == "pkg/calculation.py"
        assert chunk.language == "python"
        assert chunk.entity_type == "function"
        assert chunk.function_name == "calculate"
        assert chunk.class_name is None
        assert chunk.signature == "def calculate(value):"
        assert "Keep the calculation documented." in (chunk.docstring or "")
        assert 3 <= chunk.start_line <= chunk.end_line <= 8
        assert chunk.chunk_count == len(chunks)
        assert chunk.parent_start_line == 3
        assert chunk.parent_end_line == 8
