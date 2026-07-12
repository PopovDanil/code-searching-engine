# AGENTS.md

## Project Description

**codesearch** is a semantic code search engine that enables natural-language searching of code repositories. It parses source code using Tree-sitter AST parsing, extracts functions/methods/classes, embeds them into vector representations, indexes them with FAISS, and provides a search interface with cross-encoder reranking. The project includes an evaluation pipeline against the CodeSearchNet benchmark.

**Supported languages:** Python, Java, JavaScript, Go, Ruby, PHP

**Tech stack:** Python 3.10, PyTorch, HuggingFace Transformers, FAISS, Tree-sitter, Typer CLI, Rich console output.

## Build & Test Commands

```bash
# Run all tests (must run from src/ directory)
cd src && python -m pytest ../tests -v

# Quick tests (no model downloads required)
cd src && python -m pytest ../tests/test_chunking.py ../tests/test_parser.py ../tests/test_evaluation.py -q

# CLI usage
python src/cli.py index <repo_path> --config example_config.yaml
python src/cli.py search "<query>" --config example_config.yaml
python src/cli.py evaluate --config example_config.yaml
```

No linting or formatting tools are configured.

## Code Conventions

- **Architecture:** Pipeline pattern: Parse -> Extract -> Chunk -> Embed -> Index -> Retrieve -> Rerank -> Score
- **Design patterns:** Strategy/Factory for embedders, rerankers, query rewriters. ABCs define interfaces (`BaseEmbedder`, `BaseReranker`, `BaseQueryRewriter`).
- **Configuration:** Single `@dataclass` (`CodeSearchConfig`) with YAML loading via `from_yaml()` classmethod.
- **Type hints:** Use `Optional`, `List`, `Dict`, `Tuple` from `typing`. All modules use `from __future__ import annotations`.
- **Docstrings:** Google/numpy-style with parameter descriptions and return types.
- **Logging:** Standard `logging` module, `logger = logging.getLogger(__name__)` at module level.
- **Dataclasses:** Used extensively (`CodeEntity`, `SearchResult`, `CodeSearchConfig`, `ChunkSpan`, `LangInfo`). Some frozen (`LangInfo`, `ChunkSpan`).
- **Private helpers:** Prefixed with underscore (e.g., `_process_file()`, `_compute_metadata_bonus()`).
- **Section markers:** Module sections marked with `# -- Section Name --` banner comments.
- **CLI output:** All user-facing output uses `rich` for styled panels, tables, progress bars, and syntax highlighting.
- **Lazy initialization:** `SearchEngine` loads models/indexes/rerankers on first use via `_ensure_*()` methods.

## Agent Rules

1. **Tests must run from `src/` directory** due to import structure. Always use `workdir` parameter when running pytest.
2. **No model-dependent tests for CI.** The quick test subset (`test_chunking.py`, `test_parser.py`, `test_evaluation.py`) runs without model downloads. Prefer this for validation.
3. **Follow the factory pattern.** When adding new embedders, rerankers, or rewriters, create an ABC, implement the concrete class, and register it in the corresponding `create_*()` factory function.
4. **Use `from __future__ import annotations`** in every new module.
5. **Use `@dataclass`** for data containers. Prefer `frozen=True` for immutable value objects.
6. **Use `typing` imports** (`Optional`, `List`, `Dict`, `Tuple`) rather than built-in generic syntax.
7. **Include type hints** on all function signatures and return types.
8. **Write Google/numpy-style docstrings** for public functions and classes.
9. **Use `logging`** instead of `print()` for debug/info messages. Use `rich` console for user-facing output.
10. **Prefix private helpers** with underscore.
11. **Mark module sections** with `# -- Section Name --` banner comments when adding significant new functionality.
12. **Tests should be self-contained.** Prefer inline test data over fixtures. Use `tempfile.TemporaryDirectory()` or `tmp_path` for filesystem tests.
13. **Do not commit secrets or keys.** ML models are downloaded at runtime from HuggingFace, never stored in the repo.
14. **Keep the `index/` directory gitignored.** Never commit built indexes or evaluation caches.
