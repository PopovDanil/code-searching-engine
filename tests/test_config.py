"""Tests for config module."""

import tempfile

from config import CodeSearchConfig


def test_default_config():
    config = CodeSearchConfig()
    assert config.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert config.batch_size == 16
    assert config.top_k == 10
    assert config.index_type == "flat"
    assert config.enable_reranking is True
    assert config.enable_query_rewriting is False
    assert config.query_rewrite_strategy == "none"
    assert config.reranker_language_hint is False


def test_query_rewriting_from_yaml():
    import yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({
            "enable_query_rewriting": True,
            "query_rewrite_strategy": "hyde",
            "query_rewriter_max_new_tokens": 64,
            "reranker_language_hint": True,
        }, f)
        f.flush()
        config = CodeSearchConfig.from_yaml(f.name)
    assert config.enable_query_rewriting is True
    assert config.query_rewrite_strategy == "hyde"
    assert config.query_rewriter_max_new_tokens == 64
    assert config.reranker_language_hint is True


def test_weights_default():
    config = CodeSearchConfig()
    assert config.weights["reranker"] == 0.75
    assert config.weights["embedding"] == 0.20
    assert config.weights["metadata"] == 0.05


def test_custom_config():
    config = CodeSearchConfig(
        embedding_model="custom-model",
        batch_size=32,
        top_k=5,
    )
    assert config.embedding_model == "custom-model"
    assert config.batch_size == 32
    assert config.top_k == 5


def test_separate_indexes_default():
    config = CodeSearchConfig()
    assert config.separate_indexes is False


def test_separate_indexes_from_yaml():
    import yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"separate_indexes": True, "index_dir": "my_index"}, f)
        f.flush()
        config = CodeSearchConfig.from_yaml(f.name)
    assert config.separate_indexes is True
    assert config.index_dir == "my_index"


def test_max_dataset_records_default():
    config = CodeSearchConfig()
    assert config.max_dataset_records is None


def test_max_dataset_records_from_yaml():
    import yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"max_dataset_records": 500000}, f)
        f.flush()
        config = CodeSearchConfig.from_yaml(f.name)
    assert config.max_dataset_records == 500000
