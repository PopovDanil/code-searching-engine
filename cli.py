"""CLI entry point for the codesearch system."""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

import typer

app = typer.Typer(help="Semantic code search CLI")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def index(
    repository_path: str = typer.Argument(..., help="Path to the repository to index"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Embedding model name"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device (cpu/cuda/auto)"),
    index_type: Optional[str] = typer.Option(None, "--index-type", help="Index type (flat/hnsw)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Index a repository for semantic search."""
    _setup_logging(verbose)
    logger = logging.getLogger("codesearch.cli")

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

    from indexing.build_index import build_index

    logger.info("Indexing repository: %s", repository_path)
    try:
        faiss_index = build_index(repository_path, config)
        logger.info("Index built successfully (%d vectors)", faiss_index.ntotal)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Number of results"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Search indexed code semantically."""
    _setup_logging(verbose)
    logger = logging.getLogger("codesearch.cli")

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
        results = engine.search(query, top_k=config.top_k)
    except FileNotFoundError:
        typer.echo("Error: No index found. Run 'index' command first.", err=True)
        sys.exit(1)

    if not results:
        typer.echo("No results found.")
        return

    typer.echo(f"\nResults for: \"{query}\"\n")
    for i, result in enumerate(results, 1):
        typer.echo(f"--- Result {i} ---")
        typer.echo(str(result))
        typer.echo()


@app.command()
def evaluate(
    languages: Optional[str] = typer.Option(None, "--languages", "-l", help="Comma-separated languages"),
    max_queries: Optional[int] = typer.Option(None, "--max-queries", help="Max queries per language"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate on CodeSearchNet benchmark."""
    _setup_logging(verbose)

    from config import CodeSearchConfig

    if config_path:
        config = CodeSearchConfig.from_yaml(config_path)
    else:
        config = CodeSearchConfig()

    lang_list = languages.split(",") if languages else None

    from evaluation.evaluate import evaluate_on_codesearchnet

    results = evaluate_on_codesearchnet(
        config=config,
        languages=lang_list,
        max_queries=max_queries,
    )

    typer.echo("\nEvaluation Results:")
    for lang, metrics in results.items():
        typer.echo(f"\n  {lang}:")
        for metric, val in metrics.items():
            typer.echo(f"    {metric}: {val:.4f}")


if __name__ == "__main__":
    app()
