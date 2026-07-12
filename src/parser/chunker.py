"""AST-aware recursive chunking for parsed source code."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from tree_sitter import Node


_TEXT_SEPARATORS: Tuple[str, ...] = (
    "\r\n\r\n",
    "\n\n",
    "\r\n",
    "\n",
    "\r",
    " ",
    "\t",
    "",
)

SUPPORTED_CHUNKER_TYPES: Tuple[str, ...] = (
    "recursive",
    "language_aware_recursive",
)

_LANGUAGE_AWARE_BOUNDARY_TYPES: Dict[str, Tuple[str, ...]] = {
    "python": (
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "with_statement",
        "match_statement",
        "return_statement",
        "expression_statement",
        "function_definition",
        "class_definition",
        "block",
        "suite",
    ),
    "javascript": (
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "switch_statement",
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "statement_block",
    ),
    "java": (
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "switch_statement",
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "block",
    ),
    "go": (
        "if_statement",
        "for_statement",
        "switch_statement",
        "function_declaration",
        "block",
    ),
    "ruby": (
        "if",
        "case",
        "while",
        "until",
        "method",
        "singleton_method",
        "class",
        "module",
        "begin",
        "do_block",
    ),
    "php": (
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "switch_statement",
        "function_definition",
        "class_declaration",
        "method_declaration",
        "compound_statement",
    ),
}


@dataclass(frozen=True)
class ChunkSpan:
    """A source-backed code chunk with absolute byte and line positions.

    Byte offsets use Tree-sitter's half-open ``[start_byte, end_byte)``
    convention.  Line numbers are one-based and inclusive.
    """

    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    text: str


ByteSpan = Tuple[int, int]


def _span_text(source_bytes: bytes, span: ByteSpan) -> str:
    return source_bytes[span[0] : span[1]].decode("utf-8")


def _span_length(source_bytes: bytes, span: ByteSpan) -> int:
    return len(_span_text(source_bytes, span))


def _merge_adjacent_spans(
    spans: Sequence[ByteSpan], source_bytes: bytes, limit: int
) -> List[ByteSpan]:
    """Greedily merge contiguous spans while respecting *limit*."""
    if not spans:
        return []

    merged: List[ByteSpan] = []
    current_start, current_end = spans[0]

    for start, end in spans[1:]:
        candidate = (current_start, end)
        if start == current_end and _span_length(source_bytes, candidate) <= limit:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def _merge_text_ranges(
    ranges: Sequence[Tuple[int, int]], limit: int
) -> List[Tuple[int, int]]:
    """Greedily merge contiguous character ranges up to *limit*."""
    if not ranges:
        return []

    merged: List[Tuple[int, int]] = []
    current_start, current_end = ranges[0]

    for start, end in ranges[1:]:
        if start == current_end and end - current_start <= limit:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def _whitespace_or_hard_ranges(text: str, limit: int) -> List[Tuple[int, int]]:
    """Split at any whitespace within the limit, then hard-cut if needed."""
    ranges: List[Tuple[int, int]] = []
    start = 0

    while len(text) - start > limit:
        window_end = start + limit
        boundary = window_end
        for index in range(window_end, start, -1):
            if text[index - 1].isspace():
                boundary = index
                break
        ranges.append((start, boundary))
        start = boundary

    if start < len(text):
        ranges.append((start, len(text)))
    return ranges


def _recursive_text_ranges(
    text: str, limit: int, separators: Sequence[str]
) -> List[Tuple[int, int]]:
    """Recursively split *text*, preserving every character exactly once."""
    if len(text) <= limit:
        return [(0, len(text))]

    if not separators or separators[0] == "":
        return _whitespace_or_hard_ranges(text, limit)

    separator = separators[0]
    parts: List[Tuple[int, int]] = []
    cursor = 0

    while cursor < len(text):
        position = text.find(separator, cursor)
        if position < 0:
            parts.append((cursor, len(text)))
            break

        boundary = position + len(separator)
        parts.append((cursor, boundary))
        cursor = boundary

    # This separator did not create a useful boundary.  Try a finer one.
    if len(parts) <= 1:
        return _recursive_text_ranges(text, limit, separators[1:])

    split_ranges: List[Tuple[int, int]] = []
    for start, end in parts:
        if end - start <= limit:
            split_ranges.append((start, end))
            continue

        nested = _recursive_text_ranges(text[start:end], limit, separators[1:])
        split_ranges.extend(
            (start + nested_start, start + nested_end)
            for nested_start, nested_end in nested
        )

    return _merge_text_ranges(split_ranges, limit)


def _split_text_span(
    start_byte: int, end_byte: int, source_bytes: bytes, limit: int
) -> List[ByteSpan]:
    """Split a byte span recursively at increasingly fine text boundaries."""
    text = source_bytes[start_byte:end_byte].decode("utf-8")
    ranges = _recursive_text_ranges(text, limit, _TEXT_SEPARATORS)

    char_to_byte = [0]
    byte_offset = 0
    for char in text:
        byte_offset += len(char.encode("utf-8"))
        char_to_byte.append(byte_offset)

    return [
        (start_byte + char_to_byte[start], start_byte + char_to_byte[end])
        for start, end in ranges
        if start < end
    ]


def _strict_children(node: Node) -> List[Node]:
    """Return non-empty, source-ordered children smaller than *node*."""
    children = [
        child
        for child in node.children
        if child.end_byte > child.start_byte
        and child.start_byte >= node.start_byte
        and child.end_byte <= node.end_byte
        and (child.start_byte > node.start_byte or child.end_byte < node.end_byte)
    ]
    children.sort(key=lambda child: (child.start_byte, child.end_byte))
    return children


def _select_language_aware_children(
    node: Node, children: Sequence[Node], language: Optional[str]
) -> List[Node]:
    """Prefer children that form natural boundaries for *language*."""
    if not children:
        return []
    if language is None:
        return list(children)

    normalized = language.lower()
    boundary_types = _LANGUAGE_AWARE_BOUNDARY_TYPES.get(normalized, ())
    if not boundary_types:
        return list(children)

    preferred = [child for child in children if child.type in boundary_types]
    if preferred:
        return preferred
    return list(children)


def _recursive_node_spans(
    node: Node,
    source_bytes: bytes,
    limit: int,
    *,
    language: Optional[str] = None,
) -> List[ByteSpan]:
    """Recursively split *node* into non-overlapping spans up to *limit*."""
    node_span = (node.start_byte, node.end_byte)
    if node.end_byte <= node.start_byte:
        return []
    if _span_length(source_bytes, node_span) <= limit:
        return [node_span]

    children = _strict_children(node)
    children = _select_language_aware_children(node, children, language)
    if not children:
        return _split_text_span(node.start_byte, node.end_byte, source_bytes, limit)

    pieces: List[ByteSpan] = []
    cursor = node.start_byte

    for child in children:
        # Defensive handling for malformed trees with overlapping children.
        if child.start_byte < cursor:
            continue

        if cursor < child.start_byte:
            pieces.extend(_split_text_span(cursor, child.start_byte, source_bytes, limit))

        pieces.extend(_recursive_node_spans(child, source_bytes, limit, language=language))
        cursor = child.end_byte

    if cursor < node.end_byte:
        pieces.extend(_split_text_span(cursor, node.end_byte, source_bytes, limit))

    if not pieces:
        return _split_text_span(node.start_byte, node.end_byte, source_bytes, limit)

    return _merge_adjacent_spans(pieces, source_bytes, limit)


def _pack_for_overlap(
    spans: Sequence[ByteSpan],
    source_bytes: bytes,
    first_limit: int,
    later_limit: int,
) -> List[ByteSpan]:
    """Let the first unique span use the full budget before overlap starts."""
    if not spans:
        return []

    packed: List[ByteSpan] = []
    current_start, current_end = spans[0]
    current_limit = first_limit

    for start, end in spans[1:]:
        candidate = (current_start, end)
        if start == current_end and _span_length(source_bytes, candidate) <= current_limit:
            current_end = end
            continue

        packed.append((current_start, current_end))
        current_start, current_end = start, end
        current_limit = later_limit

    packed.append((current_start, current_end))
    return packed


def _with_overlap(
    spans: Sequence[ByteSpan], source_bytes: bytes, node_start: int, overlap_chars: int
) -> List[ByteSpan]:
    if overlap_chars == 0 or len(spans) < 2:
        return list(spans)

    node_end = spans[-1][1]
    node_text = source_bytes[node_start:node_end].decode("utf-8")
    char_to_byte = [node_start]
    byte_offset = node_start
    for char in node_text:
        byte_offset += len(char.encode("utf-8"))
        char_to_byte.append(byte_offset)
    byte_to_char = {offset: index for index, offset in enumerate(char_to_byte)}

    overlapped: List[ByteSpan] = [spans[0]]
    for start, end in spans[1:]:
        start_char = byte_to_char[start]
        overlap_start = max(0, start_char - overlap_chars)
        overlapped.append((char_to_byte[overlap_start], end))
    return overlapped


def _validate_chunk_limits(max_chars: int, overlap_chars: int) -> None:
    """Validate chunking limits used by both chunker implementations."""
    if not isinstance(max_chars, int) or isinstance(max_chars, bool):
        raise ValueError("max_chars must be an integer")
    if not isinstance(overlap_chars, int) or isinstance(overlap_chars, bool):
        raise ValueError("overlap_chars must be an integer")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")


def recursive_chunk_node(
    node: Node,
    source_bytes: bytes,
    *,
    max_chars: int,
    overlap_chars: int = 0,
) -> List[ChunkSpan]:
    """Recursively split a Tree-sitter node into bounded source chunks.

    The algorithm first follows AST children.  When an oversized node cannot
    be divided structurally, it falls back through blank-line, line,
    whitespace, and finally character boundaries.  ``max_chars`` measures
    Unicode characters and includes any requested overlap.
    """
    _validate_chunk_limits(max_chars, overlap_chars)
    if node.start_byte < 0 or node.end_byte > len(source_bytes):
        raise ValueError("node byte range is outside source_bytes")

    node_span = (node.start_byte, node.end_byte)
    if _span_length(source_bytes, node_span) <= max_chars:
        base_spans = [node_span]
    else:
        content_limit = max_chars - overlap_chars if overlap_chars else max_chars
        base_spans = _recursive_node_spans(node, source_bytes, content_limit)
        if overlap_chars:
            base_spans = _pack_for_overlap(
                base_spans,
                source_bytes,
                first_limit=max_chars,
                later_limit=content_limit,
            )
    spans = _with_overlap(base_spans, source_bytes, node.start_byte, overlap_chars)

    newline_offsets: List[int] = []
    newline = source_bytes.find(b"\n", node.start_byte, node.end_byte)
    while newline >= 0:
        newline_offsets.append(newline)
        newline = source_bytes.find(b"\n", newline + 1, node.end_byte)
    first_line = node.start_point[0] + 1

    return [
        ChunkSpan(
            start_byte=start,
            end_byte=end,
            start_line=first_line + bisect_left(newline_offsets, start),
            # A trailing newline belongs to the preceding inclusive line.
            end_line=first_line + bisect_left(newline_offsets, end - 1),
            text=source_bytes[start:end].decode("utf-8"),
        )
        for start, end in spans
        if start < end
    ]


def language_aware_recursive_chunk_node(
    node: Node,
    source_bytes: bytes,
    *,
    max_chars: int,
    overlap_chars: int = 0,
    language: Optional[str] = None,
) -> List[ChunkSpan]:
    """Chunk based on AST boundaries that are meaningful for the target language."""
    _validate_chunk_limits(max_chars, overlap_chars)
    if node.start_byte < 0 or node.end_byte > len(source_bytes):
        raise ValueError("node byte range is outside source_bytes")

    node_span = (node.start_byte, node.end_byte)
    if _span_length(source_bytes, node_span) <= max_chars:
        base_spans = [node_span]
    else:
        content_limit = max_chars - overlap_chars if overlap_chars else max_chars
        base_spans = _recursive_node_spans(
            node,
            source_bytes,
            content_limit,
            language=language,
        )
        if overlap_chars:
            base_spans = _pack_for_overlap(
                base_spans,
                source_bytes,
                first_limit=max_chars,
                later_limit=content_limit,
            )
    spans = _with_overlap(base_spans, source_bytes, node.start_byte, overlap_chars)

    newline_offsets: List[int] = []
    newline = source_bytes.find(b"\n", node.start_byte, node.end_byte)
    while newline >= 0:
        newline_offsets.append(newline)
        newline = source_bytes.find(b"\n", newline + 1, node.end_byte)
    first_line = node.start_point[0] + 1

    return [
        ChunkSpan(
            start_byte=start,
            end_byte=end,
            start_line=first_line + bisect_left(newline_offsets, start),
            end_line=first_line + bisect_left(newline_offsets, end - 1),
            text=source_bytes[start:end].decode("utf-8"),
        )
        for start, end in spans
        if start < end
    ]
