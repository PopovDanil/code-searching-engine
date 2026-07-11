# 🔍 codesearch — Semantic Code Search

A production-quality semantic code search system built **entirely on
pretrained open-source models**. No training, no fine-tuning, no model
optimization required.

## Supported Languages

| Language     | Extensions       |
|--------------|------------------|
| Python       | `.py`            |
| Java         | `.java`          |
| JavaScript   | `.js .jsx .mjs`  |
| Go           | `.go`            |
| Ruby         | `.rb`            |
| PHP          | `.php`           |

## Architecture

```
Repository
    ↓
Tree-sitter Parser ─── Parallel file parsing
    ↓
Extract function / class / method AST nodes
    ↓
Recursive AST-aware chunking
    ↓
Normalize each chunk
    ↓
Create structured text representation
    ↓
Embedding Model (Qwen3-Embedding-8B)
    ↓
FAISS Index (IndexFlatIP / IndexHNSWFlat)
    ↓
Top-100 Retrieval
    ↓
Cross-Encoder Reranker (Qwen3-Reranker-8B)
    ↓
Weighted Scoring → Final Results
```

### Recursive Chunking

Tree-sitter does not create the final chunks itself. It parses a source file
into an abstract syntax tree (AST), and the extractor selects the AST nodes
that represent functions, classes, and methods. The recursive chunker then
processes each selected node as follows:

1. If the complete node fits within `max_chunk_chars`, it becomes one chunk.
2. If it is too large, the chunker visits its direct Tree-sitter children in
   source order and recursively splits children that are still too large.
3. Text between child nodes is retained, so comments, punctuation, and
   whitespace are not lost.
4. If an oversized node has no smaller useful AST children, the chunker falls
   back to text boundaries in increasingly fine order: blank lines, line
   breaks, spaces or tabs, any other whitespace, and finally a hard
   Unicode-character boundary.
5. Adjacent spans are packed up to the configured limit. Later chunks repeat
   up to `chunk_overlap_chars` characters from the preceding source region.

In short, the hierarchy is:

```text
extracted AST node
├── fits the limit → emit one chunk
└── too large → recursively process AST children
    └── no useful smaller child → recursively use text separators
        └── no separator available → hard character split
```

Every emitted chunk keeps the original entity metadata, including repository,
file, language, function/class name, signature, and docstring. It also records
its zero-based chunk index, total chunk count, exact one-based source line
range, and the original parent entity's line range. Each chunk is embedded and
stored as a separate FAISS vector.

`max_chunk_chars` bounds the source-code slice, including overlap, and both
chunk settings count Unicode characters rather than model tokens. The
structured metadata added before embedding is not part of this character
budget. Overlap must be non-negative and smaller than the maximum chunk size.
Set `max_chunk_chars: null` to disable recursive chunking.

### Scoring Formula

```
final = 0.75 × reranker_score
      + 0.20 × embedding_similarity
      + 0.05 × metadata_bonus
```

Metadata bonus rewards:
- Function name containing query words
- Exact identifier match (e.g., `read_json` for query "read json")
- Docstring keyword overlap

All weights are configurable.

## Installation

```bash
# Clone / copy the project
cd codesearch

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration, install the appropriate PyTorch build:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Model Downloads

The models are downloaded automatically from HuggingFace on first use.
You need approximately:

| Model                   | Disk  | GPU RAM (fp16) | CPU RAM   |
|-------------------------|-------|-----------------|-----------|
| Qwen3-Embedding-8B      | ~16 GB| ~16 GB          | ~32 GB    |
| Qwen3-Reranker-8B       | ~16 GB| ~16 GB          | ~32 GB    |

> **Tip:** For machines without a powerful GPU, swap the embedding model
> to a smaller one in the config:
> ```yaml
> embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
> ```
> This requires `pip install sentence-transformers` and uses ~0.5 GB.

## Quick Start

### 1. Index a repository

```bash
python -m codesearch.cli index path/to/my-repo
```

Options:
```bash
python -m codesearch.cli index path/to/repo \
    --config example_config.yaml \
    --model Qwen/Qwen3-Embedding-8B \
    --device cuda \
    --index-type flat \
    --batch-size 16
```

### 2. Search

```bash
python -m codesearch.cli search "read json file"
python -m codesearch.cli search "database connection"
python -m codesearch.cli search "calculate cosine similarity"
```

Options:
```bash
python -m codesearch.cli search "parse command line arguments" \
    --top-k 5 \
    --no-rerank
```

### 3. Evaluate on CodeSearchNet

```bash
python -m codesearch.cli evaluate --languages python,go --max-queries 500
```

## Configuration

Copy `example_config.yaml` and edit to taste:

```yaml
embedding_model: "Qwen/Qwen3-Embedding-8B"
reranker_model: "Qwen/Qwen3-Reranker-8B"
batch_size: 16
max_chunk_chars: 1500
chunk_overlap_chars: 150
top_k: 10
index_type: "flat"
device: "auto"
weights:
  reranker: 0.75
  embedding: 0.20
  metadata: 0.05
```

Pass the config with `--config path/to/config.yaml`.

The default chunk size is 1500 characters with 150 characters of overlap.
Smaller chunks improve retrieval precision but create more vectors; larger
chunks preserve more surrounding context but approach the embedding model's
sequence limit.

## Project Structure

```
codesearch/
├── __init__.py
├── cli.py                   # Typer CLI entry point
├── config.py                # Configuration dataclass
├── parser/
│   ├── __init__.py
│   ├── parser.py            # Tree-sitter language setup & parsing
│   ├── extract.py           # Entity extraction & structured text
│   └── chunker.py           # Recursive AST-aware code chunking
├── embedding/
│   ├── __init__.py
│   └── embedder.py          # BaseEmbedder, Qwen3Embedder, factory
├── indexing/
│   ├── __init__.py
│   ├── faiss_index.py       # FAISS wrapper (build / save / load / search)
│   └── build_index.py       # Repository → index pipeline
├── retrieval/
│   ├── __init__.py
│   ├── search.py            # SearchEngine + scoring
│   └── reranker.py          # BaseReranker, Qwen3Reranker, factory
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py          # CodeSearchNet evaluation & metrics
├── models/                  # Placeholder for model artifacts
└── utils/
    └── __init__.py           # Logging setup & shared helpers
```

## Performance Notes

| Technique                       | Detail                                     |
|---------------------------------|--------------------------------------------|
| Batch embedding                 | Configurable batch size for GPU throughput |
| GPU inference                   | Automatic CUDA detection, fp16 by default  |
| Parallel parsing                | `ProcessPoolExecutor` with configurable workers |
| Streaming indexing              | Embeddings built in batches, FAISS add is incremental |
| Memory-efficient loading        | fp16 / bfloat16 dtypes on GPU              |
| HNSW approximate index          | Sub-linear retrieval for large corpora     |

**Typical latency (8B models, A100 GPU):**

| Step              | Time (approx.) |
|-------------------|-----------------|
| Query embedding   | 30–50 ms        |
| FAISS search      | < 1 ms          |
| Reranking (100)   | 2–5 s           |

On CPU, expect 10–50× slower inference.

## Limitations

- **Model size:** 8B-parameter models require significant GPU memory.
  Use `sentence-transformers/all-MiniLM-L6-v2` as a lightweight
  alternative.
- **Reranking cost:** Cross-encoder reranking is expensive; disable with
  `--no-rerank` or `enable_reranking: false`.
- **Anonymous functions:** Arrow functions and lambdas may lack
  identifiers and are harder to retrieve by name.
- **Go classes:** Go has no classes; method receivers are handled as
  methods with a class name derived from the receiver type.
- **PHP parsing:** Uses the PHP-only grammar; HTML-embedded PHP is not
  supported.
- **CodeSearchNet evaluation:** The `datasets` download can be large
  (~10 GB total for all languages).
- **Character-based chunks:** Chunk limits are measured in Unicode characters,
  not tokenizer tokens. Structured metadata also consumes part of the model's
  sequence length.

## License

This project uses only open-source models and libraries. Check individual
model licenses (Apache 2.0 for Qwen3 models) before deploying in
production.
