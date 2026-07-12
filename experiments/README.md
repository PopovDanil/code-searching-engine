# Experiment Reproduction Guide

This directory contains experiment notes and results for query rewriting and language-aware reranking.

## Main evaluation setup

All main experiments use the Python subset of CodeSearchNet:

```bash
--languages python
--max-dataset-records 500
--max-queries 50
```

For each run, 500 dataset records are requested. In the recorded Colab experiments, preprocessing produced 498 valid records and 579 indexed code chunks.

## Run baseline

```bash
python src/cli.py evaluate \
  --config configs/colab_baseline.yaml \
  --languages python \
  --max-dataset-records 500 \
  --max-queries 50
```

## Run direct query rewriting

```bash
python src/cli.py evaluate \
  --config configs/colab_rewrite.yaml \
  --languages python \
  --max-dataset-records 500 \
  --max-queries 50
```

## Run HyDE

```bash
python src/cli.py evaluate \
  --config configs/colab_hyde.yaml \
  --languages python \
  --max-dataset-records 500 \
  --max-queries 50
```

## Run language-aware reranker

```bash
python src/cli.py evaluate \
  --config configs/colab_language_hint.yaml \
  --languages python \
  --max-dataset-records 500 \
  --max-queries 50
```

## Run HyDE + language-aware reranker

```bash
python src/cli.py evaluate \
  --config configs/colab_hyde_language_hint.yaml \
  --languages python \
  --max-dataset-records 500 \
  --max-queries 50
```

## Run embedding-only control

```bash
python src/cli.py evaluate \
  --config configs/colab_embedding_only_100.yaml \
  --languages python \
  --max-dataset-records 500 \
  --max-queries 50
```

## Important notes

Do not use `--separate-indexes` for these experiments.

Use:

```yaml
separate_indexes: false
```

A valid run should show a positive number of retrieval candidates, for example:

```text
candidates=100
```

If the logs show:

```text
candidates=0
```

the run is invalid and should not be used for metrics.

## Large CodeSearchNet evaluation index

The default evaluation uses a single split, usually the test split.

For larger-scale evaluation, the index can be built from the full primary
split plus additional examples from another split:

```bash
python src/cli.py evaluate \
  --config configs/colab_language_hint.yaml \
  --languages python \
  --max-dataset-records 500000 \
  --max-queries 50 \
  --primary-split test \
  --fill-split train \
  --fill-from-split
```

In this mode, test examples are used both as indexed corpus documents and as
evaluation queries. Train examples are added only as additional indexed corpus
documents by default. They are not used as evaluation queries unless
`--train-as-queries` is explicitly passed.

Deduplication is enabled by default to avoid adding the same function twice if
CodeSearchNet splits overlap. Pass `--no-deduplicate` only when an experiment
explicitly requires duplicate corpus records.

`500000` means target records before chunking. The number of indexed FAISS
vectors may be larger because one function can produce multiple chunks.
