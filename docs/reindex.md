# Reindexing

Rebuild a cached FAISS index into a different index type without re-embedding.

The evaluation pipeline caches its index to `eval_cache/` after the first run. This script
reads the vectors from that cache, builds a new index type, and saves it alongside the
original — the original is never modified.

```bash
python src/reindex.py --source SOURCE --type TYPE [OPTIONS]
```

## Supported Index Types

| Type | FAISS Class | Training | Description |
|---|---|---|---|
| `flat` | `IndexFlatIP` | No | Exact inner product (same as original). |
| `hnsw` | `IndexHNSWFlat` | No | Hierarchical Navigable Small World — fast approximate nearest-neighbour. Best recall/latency trade-off. |
| `ivf_flat` | `IndexIVFFlat` | Yes | Inverted file index with exact cells. Faster search than flat, slight recall loss. |
| `ivf_pq` | `IndexIVFPQ` | Yes | IVF + Product Quantization — compressed vectors, much smaller on disk. |
| `ivf_sq` | `IndexIVFScalarQuantizer` | Yes | IVF + scalar quantization (fp16). |
| `pq` | `IndexPQ` | Yes | Product quantization only (no IVF). |
| `lsh` | `IndexLSH` | No | Locality-Sensitive Hashing. |

All indexes use **inner product** metric on L2-normalized vectors (equivalent to cosine similarity).

## CLI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--source` | path | (required) | Directory containing `index.faiss` + `metadata.pkl`. Typically `index/eval_cache/combined`. |
| `--output` | path | auto | Output directory. Auto-generated as `{source}_{type}` if omitted. |
| `--type` | string | (required) | Target index type (see table above). |
| `--dry-run` | flag | `false` | Show what would happen without writing any files. |
| `-f`, `--force` | flag | `false` | Overwrite output directory if it already exists. |

### HNSW options

| Option | Default | Description |
|---|---|---|
| `--hnsw-m` | 32 | Number of connections per node. Higher = better recall, more memory. |
| `--hnsw-ef-construction` | 200 | Build-time search depth. Higher = better quality index, slower build. |
| `--hnsw-ef-search` | 64 | Search-time search depth. Higher = better recall, slower search. |

### IVF options

| Option | Default | Description |
|---|---|---|
| `--nlist` | 256 | Number of Voronoi cells (clusters). Higher = finer partitioning, needs more data per cell. |
| `--nprobe` | 32 | Cells to visit at search time. Higher = better recall, slower search. |

### PQ / IVF-PQ options

| Option | Default | Description |
|---|---|---|
| `--pq-m` | 32 | Number of sub-quantizers. Lower = more compression, more distortion. |

### LSH options

| Option | Default | Description |
|---|---|---|
| `--lsh-nbits` | 1024 | Number of hash bits. More bits = more precise but slower. |

## Examples

Rebuild as HNSW (fast ANN, ~99%+ recall):

```bash
python src/reindex.py --source index/eval_cache/combined --type hnsw
```

Rebuild as IVF-Flat with 512 clusters:

```bash
python src/reindex.py --source index/eval_cache/combined --type ivf_flat --nlist 512
```

Rebuild as IVF-PQ (compressed):

```bash
python src/reindex.py --source index/eval_cache/combined --type ivf_pq --nlist 256 --pq-m 32
```

Save to a custom location (original stays untouched):

```bash
python src/reindex.py --source index/eval_cache/combined --type hnsw --output index_hnsw/eval_cache/combined
```

Preview without writing:

```bash
python src/reindex.py --source index/eval_cache/combined --type ivf_flat --dry-run
```

## Output Structure

The output directory mirrors the source layout:

```
{output}/
  index.faiss          # New FAISS index
  metadata.pkl         # CodeEntity list (same as source)
  eval_state.pkl       # Parent IDs + queries (copied from source)
  cache_key.json       # Cache key (copied from source)
```

Default output paths:

| Source | Output |
|---|---|
| `index/eval_cache/combined` | `index/eval_cache_{type}/combined` |
| `index_minilm/eval_cache/combined` | `index_minilm/eval_cache_{type}/combined` |
| `index/` (standalone) | `index_{type}/` |

## How It Works

1. **Load** — reads `index.faiss` and `metadata.pkl` from the source directory.
2. **Extract** — pulls all vectors into a numpy array (~1s for 559K vectors via fast-path storage access).
3. **Build** — creates the new FAISS index type and inserts all vectors. For IVF/PQ types, a training step runs first on the vector set.
4. **Validate** — runs 100 random queries against both old and new indexes, reports **Recall@10** so you can verify quality before using the new index.
5. **Save** — writes `index.faiss`, `metadata.pkl`, `eval_state.pkl`, and `cache_key.json` to the output directory.

## Choosing an Index Type

| Scenario | Recommended | Why |
|---|---|---|
| Small dataset (<100K vectors) | `flat` | Exact results, fast enough. |
| Large dataset, low latency | `hnsw` | Sub-millisecond search, ~99%+ recall. |
| Large dataset, memory-constrained | `ivf_pq` | Compressed vectors (4-16x smaller on disk). |
| Balanced speed/recall | `ivf_flat` | Good middle ground, faster than flat. |
| Disk-based search | `ivf_pq` | Smallest footprint. |

### Parameter Tuning Tips

- **HNSW**: Start with `--hnsw-m 32 --hnsw-ef-search 64`. Increase `efSearch` if recall is low.
- **IVF**: Rule of thumb: `nlist = sqrt(n)` where n is the number of vectors. For 559K vectors, `nlist=512-768` works well.
- **IVF nprobe**: Start with `nlist/8`, increase for better recall.
- **PQ m**: Use `m = dim/8` or `m = dim/4` for good compression vs quality trade-off. For 1024-dim, `m=32-128`.
