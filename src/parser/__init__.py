"""Tree-sitter-based source code parsing and entity extraction."""

from parser.chunker import ChunkSpan, recursive_chunk_node
from parser.extract import CodeEntity, extract_entities
from parser.parser import (
    SUPPORTED_LANGUAGES,
    LangInfo,
    detect_language,
    get_parser,
    parse_file,
)

__all__ = [
    "CodeEntity",
    "ChunkSpan",
    "extract_entities",
    "recursive_chunk_node",
    "LangInfo",
    "detect_language",
    "get_parser",
    "parse_file",
    "SUPPORTED_LANGUAGES",
]
