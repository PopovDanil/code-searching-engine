"""Tests for query rewriting without loading transformer models."""

import pytest

from retrieval.query_rewriter import NoOpQueryRewriter, create_query_rewriter


def test_disabled_factory_is_noop():
    rewriter = create_query_rewriter(
        enabled=False,
        strategy="rewrite",
        model_name="unused",
    )
    assert isinstance(rewriter, NoOpQueryRewriter)
    assert rewriter.rewrite("query") == "query"


def test_none_strategy_is_noop_even_when_enabled():
    rewriter = create_query_rewriter(
        enabled=True,
        strategy="none",
        model_name="unused",
    )
    assert isinstance(rewriter, NoOpQueryRewriter)
