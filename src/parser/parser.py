"""Tree-sitter language definitions, parser creation, and file parsing."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from tree_sitter import Language, Parser, Tree

logger = logging.getLogger(__name__)

# ── Language registry ───────────────────────────────────────────────────

@dataclass(frozen=True)
class LangInfo:
    """Metadata needed to create a parser for one language."""

    name: str
    extensions: Tuple[str, ...]
    module_name: str  # Python module that ships the grammar
    function_nodes: Tuple[str, ...]
    method_nodes: Tuple[str, ...]
    class_nodes: Tuple[str, ...]
    language_fn: str = "language"  # function name inside the module


SUPPORTED_LANGUAGES: Dict[str, LangInfo] = {
    "python": LangInfo(
        name="python",
        extensions=(".py",),
        module_name="tree_sitter_python",
        function_nodes=("function_definition",),
        method_nodes=("function_definition",),  # methods are also function_definition
        class_nodes=("class_definition",),
    ),
    "java": LangInfo(
        name="java",
        extensions=(".java",),
        module_name="tree_sitter_java",
        function_nodes=("method_declaration",),
        method_nodes=("method_declaration",),
        class_nodes=("class_declaration", "interface_declaration", "enum_declaration"),
    ),
    "javascript": LangInfo(
        name="javascript",
        extensions=(".js", ".jsx", ".mjs", ".cjs"),
        module_name="tree_sitter_javascript",
        function_nodes=("function_declaration", "arrow_function", "function_expression"),
        method_nodes=("method_definition",),
        class_nodes=("class_declaration",),
    ),
    "go": LangInfo(
        name="go",
        extensions=(".go",),
        module_name="tree_sitter_go",
        function_nodes=("function_declaration",),
        method_nodes=("method_declaration",),
        class_nodes=(),  # Go has no classes; methods belong to receiver types
    ),
    "ruby": LangInfo(
        name="ruby",
        extensions=(".rb",),
        module_name="tree_sitter_ruby",
        function_nodes=("method", "singleton_method"),
        method_nodes=("method", "singleton_method"),
        class_nodes=("class", "module"),
    ),
    "php": LangInfo(
        name="php",
        extensions=(".php",),
        module_name="tree_sitter_php",
        function_nodes=("function_definition",),
        method_nodes=("method_declaration",),
        class_nodes=("class_declaration",),
        language_fn="language_php_only",  # standalone snippets (no <?php tag)
    ),
}

# Reverse map: extension → language name
_EXTENSION_MAP: Dict[str, str] = {
    ext: lang
    for lang, info in SUPPORTED_LANGUAGES.items()
    for ext in info.extensions
}


# ── Public helpers ──────────────────────────────────────────────────────

def detect_language(file_path: str) -> Optional[str]:
    """Return the canonical language name for *file_path*, or ``None``."""
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_MAP.get(ext)


def _load_language(lang_info: LangInfo) -> Language:
    """Import the grammar module and build a ``Language`` object."""
    mod = importlib.import_module(lang_info.module_name)
    fn = getattr(mod, lang_info.language_fn, None)
    if fn is None:
        # Fall back to default ``language`` function
        fn = getattr(mod, "language")
    return Language(fn())


def get_parser(language: str) -> Parser:
    """Create a tree-sitter ``Parser`` for the given language.

    Parameters
    ----------
    language:
        Canonical language name (e.g. ``"python"``).

    Raises
    ------
    ValueError
        If the language is not supported.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language!r}. "
            f"Supported: {list(SUPPORTED_LANGUAGES)}"
        )
    lang_info = SUPPORTED_LANGUAGES[language]
    ts_language = _load_language(lang_info)
    parser = Parser()
    parser.language = ts_language
    return parser


def parse_file(source_code: str, language: str) -> Optional[Tree]:
    """Parse *source_code* and return the tree, or ``None`` on failure."""
    try:
        parser = get_parser(language)
        tree = parser.parse(source_code.encode("utf-8"))
        return tree
    except Exception:
        logger.exception("Failed to parse as %s", language)
        return None
