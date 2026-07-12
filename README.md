# 🔍 codesearch — Semantic Code Search

A local semantic code-search pipeline and CLI built around pretrained
embedding and reranking models. The repository contains indexing, search,
recursive chunking, and evaluation code; it does not contain a model-training
or fine-tuning pipeline.

## Supported Languages

| Language     | Extensions       |
|--------------|------------------|
| Python       | `.py`            |
| Java         | `.java`          |
| JavaScript   | `.js .jsx .mjs .cjs` |
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
Normalize each chunk (collapse runs of 3+ newlines)
    ↓
Create structured text representation
    ↓
Embedding Model (default: Qwen3-Embedding-0.6B)
    ↓
FAISS Index (IndexFlatIP / IndexHNSWFlat)
    ↓
Candidate Retrieval (`retrieval_top_k`, default: 100)
    ↓
Qwen3 Reranker (default: Qwen3-Reranker-0.6B)
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
stored as a separate FAISS vector. Normal search is chunk-level, so multiple
chunks from the same function can appear as separate results. CodeSearchNet
evaluation groups chunk hits back to their parent function before calculating
rank metrics.

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

All weights are configurable. When reranking is disabled, the final score is
the embedding similarity alone; the weighted formula and metadata bonus are
not applied.

## Installation

```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run all remaining commands from the repository root.

## Model Downloads

Models download from Hugging Face on first use. The defaults are:

- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-Reranker-0.6B`

For a lighter setup, install `sentence-transformers`, select a smaller
embedding model in YAML, and set `enable_reranking: false`.

## CLI Commands

There is no installable `codesearch` package. Run `src/cli.py` directly and
use the same configuration for indexing and searching.

### `index` — Build a search index

```bash
python src/cli.py index <repository_path> [OPTIONS]
```

Walks the repository, parses source files with Tree-sitter, embeds code
entities, and saves a FAISS index.

| Option | Description |
|---|---|
| `--config`, `-c` | YAML config file |
| `--model`, `-m` | Embedding model name override |
| `--device`, `-d` | Device: `cpu`, `cuda`, or `auto` |
| `--index-type` | Index type: `flat` (exact) or `hnsw` (approximate) |
| `--batch-size`, `-b` | Inference batch size |
| `--separate-indexes`, `-s` | Build a separate index per language |
| `--verbose`, `-v` | Enable debug logging |

```bash
# Single index (default)
python src/cli.py index path/to/my-repo --config example_config.yaml

# Per-language indexes
python src/cli.py index path/to/my-repo --config example_config.yaml --separate-indexes
```

With `--separate-indexes`, the output directory is structured as:

```
index/
├── manifest.json
├── python/
│   ├── index.faiss
│   └── metadata.pkl
├── go/
│   ├── index.faiss
│   └── metadata.pkl
└── ...
```

### `search` — Search indexed code

```bash
python src/cli.py search "<query>" [OPTIONS]
```

Embeds the query, retrieves candidates from FAISS, optionally reranks with
the cross-encoder, and returns ranked results.

| Option | Description |
|---|---|
| `--config`, `-c` | YAML config file |
| `--top-k`, `-k` | Number of results to return |
| `--language`, `-l` | Restrict search to a language (requires `--separate-indexes`) |
| `--no-rerank` | Disable cross-encoder reranking |
| `--verbose`, `-v` | Enable debug logging |

```bash
# Search all indexed code
python src/cli.py search "read json file" --config example_config.yaml

# Return only top 5 results, skip reranking
python src/cli.py search "parse args" --config example_config.yaml -k 5 --no-rerank

# Search only Python code (requires separate indexes)
python src/cli.py search "parse args" --config example_config.yaml --language python
```

### `evaluate` — CodeSearchNet benchmark

```bash
python src/cli.py evaluate [OPTIONS]
```

Runs a paired-row proxy evaluation against the CodeSearchNet dataset.
Reports Recall@k, MRR, and NDCG at parent-function level. The first run
downloads the dataset and models.

Query rewriting is opt-in. Set `enable_query_rewriting: true` and choose
`query_rewrite_strategy: "rewrite"` or `"hyde"`; the small rewriter model is
loaded lazily on the first search. `reranker_language_hint: true` adds the
candidate's programming language to the reranker prompt and has no effect when
reranking is disabled. Both features are disabled by default, so existing
search and evaluation commands preserve their previous behavior.

| Option | Description |
|---|---|
| `--config`, `-c` | YAML config file |
| `--languages`, `-l` | Comma-separated languages (e.g. `python,go`) |
| `--max-queries` | Max queries per language |
| `--verbose`, `-v` | Enable debug logging |

```bash
python src/cli.py evaluate --config example_config.yaml --languages python --max-queries 10
```

## Testing

Tests are in `tests/` and use pytest. Because imports are rooted in `src`,
tests must be run from that directory:

```bash
cd src
python -m pytest ../tests -v
```

### What each test file covers

| Test file | What it tests |
|---|---|
| `test_parser.py` | Language detection by file extension, Tree-sitter parser creation for each supported language, entity extraction (functions, classes, methods), `CodeEntity` identifier and structured text output |
| `test_chunking.py` | Recursive AST-aware chunking: config validation, node splitting, overlap handling, Unicode/CRLF edge cases, chunk-entity metadata preservation |
| `test_config.py` | `CodeSearchConfig` defaults, custom values, YAML loading, `separate_indexes` option |
| `test_indexing.py` | FAISS index build and search (flat), empty index behavior, dimension property, per-language index save/load/manifest |
| `test_search.py` | Metadata bonus scoring (name match, docstring match, exact match), `SearchEngine` manifest loading, language-filtered search, multi-language index merging |
| `test_evaluation.py` | Recall@k, MRR, NDCG metrics, parent-rank collapsing, documentation detection, chunk-aware evaluation logic |

### Running specific test files

```bash
cd src

# Parser and chunking only
python -m pytest ../tests/test_parser.py ../tests/test_chunking.py -v

# Indexing and search
python -m pytest ../tests/test_indexing.py ../tests/test_search.py -v

# Config and evaluation
python -m pytest ../tests/test_config.py ../tests/test_evaluation.py -v

# Single test by name
python -m pytest ../tests/test_search.py::test_search_engine_language_filter -v
```

### Quick (no model downloads)

```bash
cd src
python -m pytest ../tests/test_chunking.py ../tests/test_parser.py ../tests/test_evaluation.py -q
```

These tests do not require downloading embedding or reranking models.

## Configuration

Copy `example_config.yaml` and edit to taste:

```yaml
embedding_model: "Qwen/Qwen3-Embedding-0.6B"
reranker_model: "Qwen/Qwen3-Reranker-0.6B"
batch_size: 16
max_seq_length: 512
num_parser_workers: 4
max_chunk_chars: 1500
chunk_overlap_chars: 150
top_k: 10
retrieval_top_k: 100
index_type: "flat"
index_dir: "index"
separate_indexes: false
device: "auto"
enable_reranking: true
include_docstring: true
embedding_dtype: "float16"
weights:
  reranker: 0.75
  embedding: 0.20
  metadata: 0.05
```

Pass the config with `--config path/to/config.yaml`.

The default chunk size is 1500 characters with 150 characters of overlap.
Smaller chunks may improve locality but create more vectors and lose context;
larger chunks preserve context but approach the embedding model's sequence
limit. The source-code character limit does not include structured metadata,
and the complete structured text is still subject to tokenizer truncation at
`max_seq_length` tokens. CPU users may need `embedding_dtype: "float32"` for
model compatibility.

## Project Structure

```
.
├── src/
│   ├── cli.py                   # Typer CLI entry point
│   ├── config.py                # Configuration dataclass
│   ├── parser/
│   │   ├── parser.py            # Tree-sitter language setup and parsing
│   │   ├── extract.py           # Entity extraction and structured text
│   │   └── chunker.py           # Recursive AST-aware code chunking
│   ├── embedding/
│   │   └── embedder.py          # Embedding interfaces and implementations
│   ├── indexing/
│   │   ├── faiss_index.py       # FAISS build, persistence, and search
│   │   └── build_index.py       # Repository-to-index pipeline
│   ├── retrieval/
│   │   ├── search.py            # Retrieval, reranking, and final scoring
│   │   └── reranker.py          # Reranker implementations
│   ├── evaluation/
│   │   └── evaluate.py          # Chunk-aware CodeSearchNet proxy evaluation
│   └── models/
│       └── __init__.py          # Package marker; no downloaded weights
├── tests/                       # Unit and integration-style tests
├── example_config.yaml          # Example configuration
├── requirements.txt             # Python dependencies
├── README.md
└── LICENCE                      # MIT license
```

## Performance Notes

- Parsing uses `ProcessPoolExecutor`; model inference is batched.
- `flat` provides exact FAISS search and `hnsw` provides approximate search.
- Index construction keeps entities and embeddings in memory; it is not
  streaming.
- Runtime and memory usage depend on the selected models and corpus.

## Limitations

- Model loading and reranking can be expensive on CPU.
- Anonymous functions may not have recoverable names; Python lambdas are not
  extracted.
- Go receiver types are not stored as `class_name` metadata.
- PHP uses the PHP-only grammar; HTML-embedded PHP is unsupported.
- Chunk limits use Unicode characters, not tokenizer tokens.

## License

The repository code is licensed under the MIT License; see [LICENCE](LICENCE).
Models, datasets, and third-party dependencies have their own licenses and
terms, which should be reviewed separately.
