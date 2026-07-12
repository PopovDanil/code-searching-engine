"""Rich console configuration and shared helpers for codesearch CLI."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# ── Shared console ──────────────────────────────────────────────────────

console = Console(highlight=True)


# ── Logging setup ───────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with RichHandler for styled log output."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Progress bar factory ───────────────────────────────────────────────

def create_progress(**kwargs) -> Progress:
    """Create a Rich progress bar with standard columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        **kwargs,
    )


# ── Search result rendering ────────────────────────────────────────────

def _score_style(score: float) -> str:
    """Return a Rich style string based on score magnitude."""
    if score >= 0.8:
        return "bold green"
    if score >= 0.5:
        return "yellow"
    return "red"


def render_search_result(index: int, result) -> Panel:
    """Render a single SearchResult as a Rich Panel."""
    lines = []

    # Score line
    score_color = _score_style(result.final_score)
    meta_parts = [
        f"[{score_color}]Score:     {result.final_score:.4f}[/]",
        f"[bold]File:      [/]{result.entity.file_path}",
        f"[bold]Function:  [/]{result.entity.identifier}",
        f"[bold]Language:  [/]{result.entity.language.capitalize()}",
        f"[bold]Lines:     [/]\\u2013".join(
            [f"{result.entity.start_line}", f"{result.entity.end_line}"]
        ),
    ]
    if result.reranker_score is not None:
        meta_parts.append(
            f"[dim]Reranker:  {result.reranker_score:.4f}  |  "
            f"Embed: {result.embedding_similarity:.4f}  |  "
            f"Meta: {result.metadata_bonus:.4f}[/]"
        )
    lines.append("  ".join(meta_parts))

    # Syntax-highlighted code
    lang = result.entity.language or "python"
    code = result.entity.source_code.rstrip()
    syntax = Syntax(code, lang, theme="monokai", line_numbers=False, word_wrap=True)
    lines.append(syntax)

    title = Text(f"Result {index}", style="bold cyan")
    return Panel(
        "\n".join(str(l) for l in lines),
        title=title,
        border_style="bright_blue",
        padding=(0, 1),
    )


# ── Evaluation table ───────────────────────────────────────────────────

def render_evaluation_table(results: Dict[str, Dict[str, float]]) -> Table:
    """Render evaluation results as a Rich Table."""
    table = Table(
        title="Evaluation Results",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_blue",
    )
    table.add_column("Language", style="cyan", no_wrap=True)
    table.add_column("Recall@1", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("NDCG@10", justify="right")

    metric_keys = ["Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@10"]

    for lang, metrics in results.items():
        row = [lang]
        for key in metric_keys:
            val = metrics.get(key, 0.0)
            row.append(f"{val:.4f}")
        style = "bold white" if lang == "overall" else None
        table.add_row(*row, style=style)

    return table


# ── Index summary panel ────────────────────────────────────────────────

def render_index_summary(
    path: str,
    vectors: int,
    dimension: int,
    languages: Optional[List[str]] = None,
) -> Panel:
    """Render index build summary as a Rich Panel."""
    parts = [
        f"[bold]Path:[/]       {path}",
        f"[bold]Vectors:[/]    {vectors:,}",
        f"[bold]Dimension:[/]  {dimension}",
    ]
    if languages:
        parts.append(f"[bold]Languages:[/]  {', '.join(languages)}")
    content = "\n".join(parts)
    return Panel(content, title="[bold green]Index Built Successfully[/]", border_style="green")


# ── Error panel ────────────────────────────────────────────────────────

def render_error(message: str) -> Panel:
    """Render an error message as a Rich Panel."""
    return Panel(
        f"[bold red]{message}[/]",
        title="[bold red]Error[/]",
        border_style="red",
    )


def render_warning(message: str) -> Panel:
    """Render a warning message as a Rich Panel."""
    return Panel(
        f"[yellow]{message}[/]",
        title="[bold yellow]Warning[/]",
        border_style="yellow",
    )


# ── Model loading spinner ─────────────────────────────────────────────

def log_model_loading(console_instance: Console, model_name: str, device: str, dtype: str) -> None:
    """Print a styled model loading message."""
    console_instance.print(
        f"  [dim]\\u25cf[/] Loading [bold]{model_name}[/] on [cyan]{device}[/] [dim](dtype={dtype})[/]"
    )


def log_model_loaded(console_instance: Console, model_name: str) -> None:
    """Print a styled model loaded confirmation."""
    console_instance.print(
        f"  [green]\\u2714[/] [bold]{model_name}[/] loaded"
    )


def log_latency(console_instance: Console, label: str, ms: float) -> None:
    """Print a styled latency measurement."""
    console_instance.print(f"  [dim]\\u25cf[/] {label}: [cyan]{ms:.1f}ms[/]")
