"""Tests for chunk-aware evaluation and its metrics."""

import math
from types import SimpleNamespace

from config import CodeSearchConfig
from evaluation.evaluate import (
    _contains_documentation,
    _find_parent_rank,
    _mrr,
    _ndcg,
    _prepare_evaluation_example,
    _recall_at_k,
    _cache_key,
)


def test_recall_at_k_perfect():
    ranks = [1, 1, 1]
    assert _recall_at_k(ranks, 1) == 1.0


def test_recall_at_k_partial():
    ranks = [1, 5, 10]
    assert _recall_at_k(ranks, 1) == 1 / 3
    assert _recall_at_k(ranks, 5) == 2 / 3
    assert _recall_at_k(ranks, 10) == 1.0


def test_recall_at_k_empty():
    assert _recall_at_k([], 5) == 0.0


def test_mrr():
    ranks = [1, 2, 3]
    expected = (1 / 1 + 1 / 2 + 1 / 3) / 3
    assert abs(_mrr(ranks) - expected) < 1e-6


def test_mrr_empty():
    assert _mrr([]) == 0.0


def test_ndcg():
    ranks = [1, 2]
    # DCG@k uses 1/log2(r+1), iDCG=1.0 (binary relevance, ideal rank=1)
    per_query_ndcg = [1.0 / math.log2(r + 1) for r in ranks]
    expected = sum(per_query_ndcg) / len(per_query_ndcg)
    assert abs(_ndcg(ranks, 10) - expected) < 1e-6


def test_ndcg_empty():
    assert _ndcg([], 10) == 0.0


def test_ndcg_outside_k():
    ranks = [11, 12]
    assert _ndcg(ranks, 10) == 0.0


def test_evaluation_example_uses_recursive_chunks_without_query_leakage():
    documentation = "Compute several intermediate values."
    example = {
        "repository_name": "example/project",
        "func_path_in_repository": "src/calculation.py",
        "func_name": "Calculator.calculate",
        "func_documentation_string": documentation,
        "func_code_string": (
            "def calculate(value):\n"
            f'    """{documentation}"""\n'
            "    first = value + 100\n"
            "    second = first * 200\n"
            "    third = second - 300\n"
            "    fourth = third / 400\n"
            "    return fourth\n"
        ),
    }
    config = CodeSearchConfig(max_chunk_chars=55, chunk_overlap_chars=5)

    query, chunks = _prepare_evaluation_example(example, "python", config)

    assert query == documentation
    assert len(chunks) > 1
    assert all(len(chunk.source_code) <= config.max_chunk_chars for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.chunk_count == len(chunks) for chunk in chunks)
    assert all(chunk.function_name == "calculate" for chunk in chunks)
    assert all(chunk.repository == "example/project" for chunk in chunks)
    assert all(chunk.file_path == "src/calculation.py" for chunk in chunks)
    assert all(chunk.docstring is None for chunk in chunks)
    assert all(
        not _contains_documentation(chunk.source_code, documentation)
        for chunk in chunks
    )
    assert all(
        "Documentation:" not in chunk.to_structured_text(include_docstring=False)
        for chunk in chunks
    )


def test_removed_multiline_docstring_does_not_create_empty_chunks():
    documentation = (
        "Calculate a value using a deliberately long explanation.\n"
        "Keep the explanation out of every indexed code chunk."
    )
    example = {
        "func_name": "calculate",
        "func_documentation_string": documentation,
        "func_code_string": (
            "def calculate(value):\n"
            f'    """{documentation}"""\n'
            "    adjusted = value + 1\n"
            "    return adjusted * 2\n"
        ),
    }
    config = CodeSearchConfig(max_chunk_chars=35, chunk_overlap_chars=5)

    _query, chunks = _prepare_evaluation_example(example, "python", config)

    assert chunks
    assert all(chunk.source_code.strip() for chunk in chunks)
    assert all(len(chunk.source_code) <= config.max_chunk_chars for chunk in chunks)


def test_evaluation_selects_only_the_dataset_function_parent():
    example = {
        "func_name": "module.outer",
        "func_documentation_string": "Transform a value.",
        "func_code_string": (
            "def outer(value):\n"
            "    def inner(item):\n"
            "        return item + 1\n"
            "    adjusted = inner(value)\n"
            "    return adjusted * 2\n"
        ),
    }
    config = CodeSearchConfig(max_chunk_chars=45, chunk_overlap_chars=5)

    _query, chunks = _prepare_evaluation_example(example, "python", config)

    assert chunks
    assert all(chunk.function_name == "outer" for chunk in chunks)
    assert len(
        {
            (chunk.parent_start_line, chunk.parent_end_line)
            for chunk in chunks
        }
    ) == 1


def test_parent_rank_collapses_duplicate_chunks():
    wrong_chunk_1 = object()
    wrong_chunk_2 = object()
    relevant_chunk = object()
    later_chunk = object()
    results = [
        SimpleNamespace(entity=wrong_chunk_1),
        SimpleNamespace(entity=wrong_chunk_2),
        SimpleNamespace(entity=relevant_chunk),
        SimpleNamespace(entity=later_chunk),
    ]
    entity_to_parent = {
        id(wrong_chunk_1): 10,
        id(wrong_chunk_2): 10,
        id(relevant_chunk): 20,
        id(later_chunk): 30,
    }

    rank = _find_parent_rank(results, entity_to_parent, relevant_parent=20)

    assert rank == 2


def test_parent_rank_miss_has_zero_metric_credit():
    unrelated_chunk = object()
    results = [SimpleNamespace(entity=unrelated_chunk)]

    rank = _find_parent_rank(
        results,
        {id(unrelated_chunk): 10},
        relevant_parent=20,
    )

    assert math.isinf(rank)
    assert _recall_at_k([rank], 10) == 0.0
    assert _mrr([rank]) == 0.0
    assert _ndcg([rank], 10) == 0.0


def test_documentation_detection_normalises_case_and_whitespace():
    assert _contains_documentation(
        "COMPUTE   several\nintermediate values.",
        "Compute several intermediate values.",
    )


def test_documentation_detection_ignores_comment_decoration():
    documentation = "Compute several intermediate values."

    assert _contains_documentation(
        "/**\n * Compute several\n * intermediate values.\n */",
        documentation,
    )
    assert _contains_documentation(
        "// Compute several\n// intermediate values.",
        documentation,
    )


def test_non_python_attached_documentation_is_not_indexed():
    documentation = "Return the incremented value."
    example = {
        "func_name": "Calculator.increment",
        "func_documentation_string": documentation,
        "func_code_string": (
            "/**\n"
            " * Return the incremented value.\n"
            " */\n"
            "public int increment(int value) {\n"
            "    return value + 1;\n"
            "}\n"
        ),
    }
    config = CodeSearchConfig(max_chunk_chars=80, chunk_overlap_chars=5)

    _query, chunks = _prepare_evaluation_example(example, "java", config)

    assert chunks
    assert all(chunk.function_name == "increment" for chunk in chunks)
    assert all(chunk.docstring is None for chunk in chunks)
    assert all(
        not _contains_documentation(chunk.source_code, documentation)
        for chunk in chunks
    )


def test_cache_key_deterministic():
    key1 = _cache_key(500000, "Qwen/Qwen3-Embedding-0.6B", "test")
    key2 = _cache_key(500000, "Qwen/Qwen3-Embedding-0.6B", "test")
    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) == 16


def test_cache_key_varies_with_params():
    base = _cache_key(500000, "Qwen/Qwen3-Embedding-0.6B", "test")
    assert _cache_key(100000, "Qwen/Qwen3-Embedding-0.6B", "test") != base
    assert _cache_key(500000, "other-model", "test") != base
    assert _cache_key(500000, "Qwen/Qwen3-Embedding-0.6B", "validation") != base
    assert _cache_key(
        500000,
        "Qwen/Qwen3-Embedding-0.6B",
        "test",
        "language_aware_recursive",
    ) != base


def test_max_dataset_records_division_even():
    total = 600000
    langs = ["python", "java", "javascript", "go", "ruby", "php"]
    num_langs = len(langs)
    per_lang_base = total // num_langs
    remainder = total % num_langs
    limits = {
        lang: per_lang_base + (1 if i < remainder else 0)
        for i, lang in enumerate(langs)
    }
    assert all(v == 100000 for v in limits.values())
    assert sum(limits.values()) == total


def test_max_dataset_records_division_uneven():
    total = 500000
    langs = ["python", "java", "javascript", "go", "ruby", "php"]
    num_langs = len(langs)
    per_lang_base = total // num_langs
    remainder = total % num_langs
    limits = {
        lang: per_lang_base + (1 if i < remainder else 0)
        for i, lang in enumerate(langs)
    }
    assert sum(limits.values()) == total
    assert limits["python"] == 83334  # first 2 languages get +1
    assert limits["java"] == 83334
    assert limits["javascript"] == 83333
