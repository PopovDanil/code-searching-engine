"""CLI entry point for the codesearch system."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List, Optional

import typer

from console import (
    console,
    render_error,
    render_evaluation_table,
    render_index_summary,
    render_search_result,
    setup_logging,
)

app = typer.Typer(help="Semantic code search CLI")

if TYPE_CHECKING:
    from config import CodeSearchConfig


# ── CLI override helpers ───────────────────────────────────────────────

def _apply_chunker_type(
    config: CodeSearchConfig,
    chunker_type: Optional[str],
) -> None:
    """Apply and validate an optional CLI chunker override."""
    if chunker_type is None:
        return

    from parser.chunker import SUPPORTED_CHUNKER_TYPES

    if chunker_type not in SUPPORTED_CHUNKER_TYPES:
        choices = ", ".join(SUPPORTED_CHUNKER_TYPES)
        raise typer.BadParameter(
            f"must be one of: {choices}",
            param_hint="--chunker-type",
        )
    config.chunker_type = chunker_type


@app.command()
def index(
    repository_path: str = typer.Argument(..., help="Path to the repository to index"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Embedding model name"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device (cpu/cuda/auto)"),
    index_type: Optional[str] = typer.Option(None, "--index-type", help="Index type (flat/hnsw)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    separate_indexes: bool = typer.Option(False, "--separate-indexes", "-s", help="Build a separate index per language"),
    chunker_type: Optional[str] = typer.Option(None, "--chunker-type", help="Chunker: recursive or language_aware_recursive"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Index a repository for semantic search."""
    setup_logging(verbose)

    from config import CodeSearchConfig

    if config_path:
        config = CodeSearchConfig.from_yaml(config_path)
    else:
        config = CodeSearchConfig()

    if model:
        config.embedding_model = model
    if device:
        config.device = device
    if index_type:
        config.index_type = index_type
    if batch_size:
        config.batch_size = batch_size
    if separate_indexes:
        config.separate_indexes = True
    _apply_chunker_type(config, chunker_type)

    from indexing.build_index import build_index

    console.print(f"\n[bold]\\U0001f4c2[/] Indexing repository: [cyan]{repository_path}[/]\n")
    try:
        faiss_index = build_index(repository_path, config)
        if faiss_index is not None:
            panel = render_index_summary(
                path=config.index_dir,
                vectors=faiss_index.ntotal,
                dimension=faiss_index.dimension,
            )
            console.print(panel)
        else:
            console.print(
                render_index_summary(
                    path=config.index_dir,
                    vectors=0,
                    dimension=0,
                )
            )
    except Exception as e:
        console.print(render_error(str(e)))
        sys.exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Number of results"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Restrict search to a language (requires --separate-indexes)"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Search indexed code semantically."""
    setup_logging(verbose)

    from config import CodeSearchConfig

    if config_path:
        config = CodeSearchConfig.from_yaml(config_path)
    else:
        config = CodeSearchConfig()

    if top_k:
        config.top_k = top_k
    if no_rerank:
        config.enable_reranking = False

    from retrieval.search import SearchEngine

    engine = SearchEngine(config=config)

    try:
        results = engine.search(query, top_k=config.top_k, language=language)
    except FileNotFoundError:
        console.print(render_error("No index found. Run [bold]'index'[/] command first."))
        sys.exit(1)

    if not results:
        console.print("[yellow]No results found.[/]")
        return

    console.print()
    console.rule(f'[bold cyan]Results for: [white]"{query}"[/][/]')
    console.print()
    for i, result in enumerate(results, 1):
        panel = render_search_result(i, result)
        console.print(panel)
        console.print()


@app.command()
def evaluate(
    languages: Optional[str] = typer.Option(None, "--languages", "-l", help="Comma-separated languages"),
    max_queries: Optional[int] = typer.Option(None, "--max-queries", help="Max evaluation queries per language"),
    max_dataset_records: Optional[int] = typer.Option(None, "--max-dataset-records", help="Total records across all languages to load into the database"),
    separate_indexes: Optional[bool] = typer.Option(None, "--separate-indexes", "-s", help="Build separate per-language indexes (default: combined)"),
    rewrite: Optional[bool] = typer.Option(None, "--rewrite", "-r", help="Enable query rewriting"),
    rewrite_strategy: Optional[str] = typer.Option(None, "--rewrite-strategy", help="Rewrite strategy: rewrite or hyde"),
    rewrite_model: Optional[str] = typer.Option(None, "--rewrite-model", help="Query rewriter model name"),
    reranker_hint: Optional[bool] = typer.Option(None, "--reranker-hint", help="Add language hint to reranker prompt"),
    primary_split: Optional[str] = typer.Option(None, "--primary-split", help="Primary CodeSearchNet split used for corpus and queries"),
    fill_split: Optional[str] = typer.Option(None, "--fill-split", help="Secondary split used to fill the evaluation corpus"),
    fill_from_split: Optional[bool] = typer.Option(None, "--fill-from-split", help="Fill the corpus from a secondary split"),
    no_deduplicate: bool = typer.Option(False, "--no-deduplicate", help="Allow duplicate functions across evaluation splits"),
    train_as_queries: Optional[bool] = typer.Option(None, "--train-as-queries", help="Use fill-split records as evaluation queries"),
    chunker_type: Optional[str] = typer.Option(None, "--chunker-type", help="Chunker: recursive or language_aware_recursive"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate on CodeSearchNet benchmark."""
    setup_logging(verbose)

    from config import CodeSearchConfig

    if config_path:
        config = CodeSearchConfig.from_yaml(config_path)
    else:
        config = CodeSearchConfig()

    if separate_indexes is not None:
        config.separate_indexes = separate_indexes
    if rewrite is not None:
        config.enable_query_rewriting = rewrite
    if rewrite_strategy is not None:
        config.query_rewrite_strategy = rewrite_strategy
    if rewrite_model is not None:
        config.query_rewriter_model = rewrite_model
    if reranker_hint is not None:
        config.reranker_language_hint = reranker_hint
    if primary_split is not None:
        config.evaluation_primary_split = primary_split
    if fill_split is not None:
        config.evaluation_fill_split = fill_split
    if fill_from_split is not None:
        config.evaluation_fill_from_split = fill_from_split
    if no_deduplicate:
        config.evaluation_deduplicate = False
    if train_as_queries is not None:
        config.evaluation_train_as_queries = train_as_queries
    _apply_chunker_type(config, chunker_type)

    effective_max_queries = max_queries
    effective_max_dataset_records = max_dataset_records or config.max_dataset_records

    lang_list = languages.split(",") if languages else None

    from evaluation.evaluate import evaluate_on_codesearchnet

    results = evaluate_on_codesearchnet(
        config=config,
        languages=lang_list,
        max_queries=effective_max_queries,
        max_dataset_records=effective_max_dataset_records,
    )

    console.print()
    table = render_evaluation_table(results)
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
