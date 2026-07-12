"""Extract functions, methods, and classes from a tree-sitter AST."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import List, Optional

from tree_sitter import Node, Tree

from parser.chunker import (
    SUPPORTED_CHUNKER_TYPES,
    language_aware_recursive_chunk_node,
    recursive_chunk_node,
)
from parser.parser import SUPPORTED_LANGUAGES, LangInfo

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────────────────

@dataclass
class CodeEntity:
    """A single extracted code entity (function, method, or class)."""

    repository: str
    file_path: str
    language: str
    entity_type: str  # "function" | "method" | "class"
    function_name: Optional[str]
    class_name: Optional[str]
    signature: str
    docstring: Optional[str]
    source_code: str
    start_line: int
    end_line: int
    chunk_index: int = 0
    chunk_count: int = 1
    parent_start_line: Optional[int] = None
    parent_end_line: Optional[int] = None

    # ── Derived helpers ─────────────────────────────────────────────────

    @property
    def identifier(self) -> str:
        """Human-readable identifier like ``ClassName.method_name``."""
        parts = []
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts) if parts else "<anonymous>"

    def to_structured_text(self, include_docstring: bool = True) -> str:
        """Build the structured text representation used for embedding.

        Format::

            Language: Python
            Function: read_json_file
            Signature: def read_json(path):
            Documentation: Reads JSON from disk.
            Code:
            def read_json(path):
                ...

        The Documentation section is omitted when no docstring is available
        or when *include_docstring* is ``False``.
        """
        lines: List[str] = [f"Language: {self.language.capitalize()}"]

        if self.entity_type == "class":
            lines.append(f"Class: {self.function_name or self.class_name or '<unknown>'}")
        else:
            if self.class_name:
                lines.append(f"Class: {self.class_name}")
                lines.append(f"Method: {self.function_name or '<unknown>'}")
            else:
                lines.append(f"Function: {self.function_name or '<unknown>'}")

        lines.append(f"Signature: {self.signature}")

        if include_docstring and self.docstring:
            # Collapse to single line for the header, preserve full text below
            lines.append(f"Documentation: {self.docstring.strip()}")

        lines.append("Code:")
        lines.append(self.source_code)
        return "\n".join(lines)


# ── Internal extraction helpers ─────────────────────────────────────────

def _node_text(node: Node, source_bytes: bytes) -> str:
    """Extract UTF-8 text covered by *node*."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _line_text(source_code: str, line_number: int) -> str:
    """Return the content of 1-indexed *line_number*."""
    lines = source_code.splitlines()
    if 0 < line_number <= len(lines):
        return lines[line_number - 1]
    return ""


def _extract_name(node: Node, source_bytes: bytes) -> Optional[str]:
    """Best-effort extraction of the identifier name from a definition node."""
    # Walk immediate children for a node whose field name or type contains "name"
    for i, child in enumerate(node.children):
        if child.type == "identifier" or child.type == "property_identifier":
            return _node_text(child, source_bytes)
        if child.is_named and child.type.endswith("identifier"):
            return _node_text(child, source_bytes)
        field_name = node.field_name_for_child(i)
        if field_name and "name" in field_name:
            return _node_text(child, source_bytes)
    # Try child_by_field_name
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node, source_bytes)
    return None


def _extract_signature(node: Node, source_bytes: bytes, language: str) -> str:
    """Extract the function/method/class signature (header without body).

    Heuristic: take text from the start of the node up to the opening brace
    or colon that starts the body.
    """
    full_text = _node_text(node, source_bytes)

    if language == "python":
        # Python: everything up to and including the first ':'
        colon_pos = full_text.find(":")
        if colon_pos != -1:
            return full_text[: colon_pos + 1].strip()
        return full_text.split("\n")[0].strip()

    # Brace-based languages: everything before the first '{'
    brace_pos = full_text.find("{")
    if brace_pos != -1:
        return full_text[:brace_pos].strip()

    # Ruby: everything before 'do' or first newline after parameters
    if language == "ruby":
        do_pos = full_text.find("\n")
        if do_pos != -1:
            return full_text[:do_pos].strip()
        return full_text.split("\n")[0].strip()

    return full_text.split("\n")[0].strip()


def _extract_docstring(node: Node, source_bytes: bytes, language: str) -> Optional[str]:
    """Extract documentation string / comment attached to *node*."""
    if language == "python":
        # Python docstring: first statement in the body is a string expression
        body_node = node.child_by_field_name("body")
        if body_node is not None:
            for child in body_node.children:
                if child.type == "expression_statement":
                    for sub in child.children:
                        if sub.type in ("string", "concatenated_string"):
                            return _node_text(sub, source_bytes).strip()
                break  # only check first child
        return None

    # For other languages, look for comment nodes immediately before this node
    comments: List[str] = []
    sibling = node.prev_named_sibling
    # Walk backwards through consecutive comment siblings
    while sibling is not None and sibling.type in ("comment", "block_comment", "line_comment"):
        comments.insert(0, _node_text(sibling, source_bytes).strip())
        sibling = sibling.prev_named_sibling

    if comments:
        combined = "\n".join(comments)
        # Strip common comment prefixes
        combined = re.sub(r"^\s*//+\s?", "", combined, flags=re.MULTILINE)
        combined = re.sub(r"^\s*\*+\s?", "", combined, flags=re.MULTILINE)
        combined = re.sub(r"/\*+=?", "", combined)
        combined = re.sub(r"\*+/", "", combined)
        combined = re.sub(r"^\s*#+\s?", "", combined, flags=re.MULTILINE)
        return combined.strip() or None

    return None


def _extract_class_name(node: Node, source_bytes: bytes) -> Optional[str]:
    """Walk up the tree to find an enclosing class name."""
    parent = node.parent
    while parent is not None:
        if parent.type in ("class_definition", "class_declaration",
                           "class", "module"):
            name_node = parent.child_by_field_name("name")
            if name_node is not None:
                return _node_text(name_node, source_bytes)
            # Try children
            for child in parent.children:
                if child.type in ("identifier", "type_identifier"):
                    return _node_text(child, source_bytes)
        parent = parent.parent
    return None


def _is_inside_class(node: Node) -> bool:
    """Return True if *node* is nested inside a class body."""
    parent = node.parent
    while parent is not None:
        if parent.type in ("class_definition", "class_declaration",
                           "class", "class_body"):
            return True
        parent = parent.parent
    return False


def _walk_tree(node: Node) -> List[Node]:
    """Depth-first traversal yielding all named and unnamed descendants."""
    result: List[Node] = []
    stack = list(node.children)
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(reversed(current.children))
    return result


# ── Public extraction API ───────────────────────────────────────────────

def extract_entities(
    source_code: str,
    tree: Tree,
    language: str,
    repository: str,
    file_path: str,
    *,
    max_chunk_chars: Optional[int] = None,
    chunk_overlap_chars: int = 0,
    chunker_type: str = "recursive",
) -> List[CodeEntity]:
    """Walk *tree* and return every function, method, and class found.

    Parameters
    ----------
    source_code:
        The raw source text.
    tree:
        The parsed tree-sitter tree.
    language:
        Canonical language name (``"python"``, ``"java"``, …).
    repository:
        Repository identifier (directory name or dataset label).
    file_path:
        Relative file path inside the repository.
    max_chunk_chars:
        Maximum number of Unicode characters per recursive chunk, including
        overlap.  ``None`` keeps each extracted entity whole.
    chunk_overlap_chars:
        Number of characters repeated from the preceding chunk.
    chunker_type:
        Chunking strategy selector. Supported values are
        ``"recursive"`` and ``"language_aware_recursive"``.

    Returns
    -------
    List[CodeEntity]
        All extracted entities, each with metadata.
    """
    if chunker_type not in SUPPORTED_CHUNKER_TYPES:
        raise ValueError(
            "chunker_type must be one of: " + ", ".join(SUPPORTED_CHUNKER_TYPES)
        )

    lang_info: LangInfo = SUPPORTED_LANGUAGES[language]
    source_bytes = source_code.encode("utf-8")
    entities: List[CodeEntity] = []

    # All node types we care about
    target_types = set(
        lang_info.function_nodes
        + lang_info.method_nodes
        + lang_info.class_nodes
    )

    all_nodes = _walk_tree(tree.root_node)

    for node in all_nodes:
        if node.type not in target_types:
            continue

        is_class = node.type in lang_info.class_nodes
        is_method = (not is_class) and _is_inside_class(node)

        name = _extract_name(node, source_bytes)
        signature = _extract_signature(node, source_bytes, language)
        docstring = _extract_docstring(node, source_bytes, language)
        code = _node_text(node, source_bytes)

        # Normalise: collapse excessive blank lines
        code = re.sub(r"\n{3,}", "\n\n", code)

        class_name: Optional[str] = None
        entity_type: str

        if is_class:
            entity_type = "class"
            class_name = name
        elif is_method:
            entity_type = "method"
            class_name = _extract_class_name(node, source_bytes)
        else:
            entity_type = "function"

        entity = CodeEntity(
            repository=repository,
            file_path=file_path,
            language=language,
            entity_type=entity_type,
            function_name=name,
            class_name=class_name,
            signature=signature,
            docstring=docstring,
            source_code=code,
            start_line=node.start_point[0] + 1,  # 1-indexed
            end_line=node.end_point[0] + 1,
        )

        if max_chunk_chars is None:
            entities.append(entity)
            continue

        if chunker_type == "language_aware_recursive":
            spans = language_aware_recursive_chunk_node(
                node,
                source_bytes,
                max_chars=max_chunk_chars,
                overlap_chars=chunk_overlap_chars,
                language=language,
            )
        else:
            spans = recursive_chunk_node(
                node,
                source_bytes,
                max_chars=max_chunk_chars,
                overlap_chars=chunk_overlap_chars,
            )
        chunk_count = len(spans)
        for chunk_index, span in enumerate(spans):
            chunk_code = re.sub(r"\n{3,}", "\n\n", span.text)
            entities.append(
                replace(
                    entity,
                    source_code=chunk_code,
                    start_line=span.start_line,
                    end_line=span.end_line,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    parent_start_line=entity.start_line,
                    parent_end_line=entity.end_line,
                )
            )

    return entities
