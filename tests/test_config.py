"""Tests for config module."""

from config import CodeSearchConfig


def test_default_config():
    config = CodeSearchConfig()
    assert config.embedding_model == "Qwen/Qwen3-Embedding-8B"
    assert config.batch_size == 16
    assert config.top_k == 10
    assert config.index_type == "flat"
    assert config.enable_reranking is True


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
