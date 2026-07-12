# Evaluation CLI

Evaluate the search system on the [CodeSearchNet](https://github.com/github/CodeSearchNet) benchmark.

```bash
python src/cli.py evaluate [OPTIONS]
```

## CLI Options

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--languages` | `-l` | string | all six | Comma-separated list of languages to evaluate (e.g. `"python,java"`). |
| `--max-queries` | | int | all loaded | Maximum number of evaluation queries **per language**. Useful for quick iteration. |
| `--max-dataset-records` | | int | all | Total records to load across all languages. Divided evenly among target languages. |
| `--separate-indexes` | `-s` | bool | `false` | Build a separate FAISS index per language instead of one combined index. |
| `--rewrite` | `-r` | bool | `false` | Enable query rewriting before search. |
| `--rewrite-strategy` | | string | `none` | Rewrite strategy: `rewrite` (rewrite query) or `hyde` (generate hypothetical code description). Requires `--rewrite`. |
| `--rewrite-model` | | string | `SmolLM2-135M-Instruct` | Model name for the query rewriter. |
| `--reranker-hint` | | bool | `false` | Add a language hint (`"The document is source code written in {Language}."`) to the reranker prompt. |
| `--config` | `-c` | path | none | Path to a YAML configuration file. |
| `--verbose` | `-v` | flag | `false` | Enable debug logging. |

## Examples

Run a full evaluation on all languages with defaults:

```bash
python src/cli.py evaluate
```

Quick smoke test with 1000 total records and 50 queries per language:

```bash
python src/cli.py evaluate --max-dataset-records 1000 --max-queries 50
```

Evaluate only Python and Java:

```bash
python src/cli.py evaluate --languages python,java
```

Use separate per-language indexes:

```bash
python src/cli.py evaluate --separate-indexes
```

Enable query rewriting:

```bash
python src/cli.py evaluate --rewrite --rewrite-strategy rewrite
```

Enable HyDE (hypothetical document embedding):

```bash
python src/cli.py evaluate --rewrite --rewrite-strategy hyde
```

Enable reranker language hint:

```bash
python src/cli.py evaluate --reranker-hint
```

Override model and device via a config file:

```bash
python src/cli.py evaluate --config my_config.yaml -v
```

## Output

Per-language metrics are printed to stdout:

```
Evaluation Results:

  python:
    Recall@1: 0.4520
    Recall@5: 0.6810
    Recall@10: 0.7430
    MRR: 0.5312
    NDCG@10: 0.5890

  java:
    Recall@1: 0.3910
    Recall@5: 0.6120
    Recall@10: 0.6900
    MRR: 0.4780
    NDCG@10: 0.5430

  overall:
    Recall@1: 0.4215
    Recall@5: 0.6465
    Recall@10: 0.7165
    MRR: 0.5046
    NDCG@10: 0.5660
```

| Metric | Meaning |
|---|---|
| **Recall\@K** | Fraction of queries where the correct function appears in the top-K results. |
| **MRR** | Mean Reciprocal Rank -- average of `1/rank` for each query. |
| **NDCG\@10** | Normalised Discounted Cumulative Gain at 10. Measures ranking quality. |

## Caching

Evaluation indexes are cached to `{index_dir}/eval_cache/` keyed by
`max_dataset_records`, `embedding_model`, and `split`. On a subsequent
run with the same parameters the cached index is loaded directly,
skipping dataset download, parsing, and embedding.

Cache directories:

- `eval_cache/combined/` -- single combined index (default mode)
- `eval_cache/{language}/` -- per-language indexes (with `--separate-indexes`)

To force a rebuild, delete the cache directory or change the `embedding_model` / `max_dataset_records` / `split`.

## Config file

All `CodeSearchConfig` fields can be set via YAML. The evaluate command
respects the following fields that are not exposed as CLI flags:

```yaml
# Models
embedding_model: "Qwen/Qwen3-Embedding-0.6B"
reranker_model: "Qwen/Qwen3-Reranker-0.6B"

# Retrieval
top_k: 10
retrieval_top_k: 100

# Reranker
enable_reranking: true

# Scoring weights
weights:
  reranker: 0.75
  embedding: 0.20
  metadata: 0.05

# Index
index_type: "flat"
index_dir: "index"

# Device
device: "auto"

# Processing
batch_size: 16
max_seq_length: 512
```

### Query rewriting (opt-in)

Query rewriting is disabled by default. Enable via config:

```yaml
enable_query_rewriting: true
query_rewrite_strategy: "rewrite"   # "rewrite" or "hyde"
query_rewriter_model: "HuggingFaceTB/SmolLM2-135M-Instruct"
query_rewriter_max_new_tokens: 128
```

### Reranker language hint

Adds `"The document is source code written in {Language}."` to the reranker prompt:

```yaml
reranker_language_hint: true
```
