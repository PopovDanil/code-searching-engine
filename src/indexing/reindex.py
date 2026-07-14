#!/usr/bin/env python3
"""Quickly reindex a cached FAISS index into a different index type.

Reads embeddings from an existing eval_cache (or any FAISS index directory),
builds a new index with the requested type, and saves it alongside (or
separately from) the original — without re-embedding anything.

Usage examples
--------------
    # Rebuild as HNSW (fast ANN, no training)
    python src/reindex.py --source index/eval_cache/combined --type hnsw --hnsw-m 32

    # Rebuild as IVF-Flat (needs training, faster search than flat)
    python src/reindex.py --source index/eval_cache/combined --type ivf_flat --nlist 1024

    # Rebuild as IVF-PQ (compressed, much smaller on disk)
    python src/reindex.py --source index/eval_cache/combined --type ivf_pq --nlist 256 --pq-m 32

    # Save to a separate directory (preserves original untouched)
    python src/reindex.py --source index/eval_cache/combined --type hnsw --output index_hnsw/eval_cache/combined

    # Reindex from a standalone FaissCodeIndex dir (not eval_cache)
    python src/reindex.py --source index --type hnsw --output index_hnsw

    # Dry run: just show what would happen
    python src/reindex.py --source index/eval_cache/combined --type ivf_flat --dry-run
"""

from __future__ import annotations

import argparse
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reindex")


# ── Helpers ────────────────────────────────────────────────────────────


def _load_faiss_index(index_path: Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    logger.info("Loading FAISS index from %s", index_path)
    idx = faiss.read_index(str(index_path))
    logger.info("  ntotal=%d  dim=%d  type=%s", idx.ntotal, idx.d, type(idx).__name__)
    return idx


def _load_metadata(meta_path: Path) -> list:
    """Load CodeEntity metadata from pickle."""
    logger.info("Loading metadata from %s", meta_path)
    with open(meta_path, "rb") as fh:
        metadata = pickle.load(fh)
    logger.info("  %d entities loaded", len(metadata))
    return metadata


def _load_eval_state(state_path: Path) -> dict:
    """Load eval_state.pkl (parent IDs, queries, etc.) if present."""
    if not state_path.exists():
        return {}
    logger.info("Loading eval state from %s", state_path)
    with open(state_path, "rb") as fh:
        state = pickle.load(fh)
    return state


def _extract_vectors(index: faiss.Index) -> np.ndarray:
    """Extract all vectors from a FAISS index as a numpy array.

    Works for IndexFlatIP (direct storage access) and reconstruct-capable
    indexes (HNSW, IVF, etc.).
    """
    ntotal = index.ntotal
    dim = index.d

    # Fast path: IndexFlatCodes stores vectors contiguously
    if isinstance(index, faiss.IndexFlatCodes):
        logger.info("  Fast-path extraction (IndexFlatCodes storage)")
        # faiss.swig_ptr gives us a view into the internal storage
        try:
            arr = faiss.rev_swig_ptr(index.get_xb(), ntotal * dim)
            return arr.reshape(ntotal, dim).copy()
        except Exception:
            pass

    # Generic path: reconstruct vectors one by one (works for any index)
    logger.info("  Reconstructing vectors from index (%d vectors)...", ntotal)
    t0 = time.time()
    vectors = np.empty((ntotal, dim), dtype=np.float32)
    batch = 10000
    for start in range(0, ntotal, batch):
        end = min(start + batch, ntotal)
        for i in range(start, end):
            vectors[i] = index.reconstruct(i)
        elapsed = time.time() - t0
        logger.info("    %d/%d reconstructed (%.1fs)", end, ntotal, elapsed)
    return vectors


# ── Index builders ─────────────────────────────────────────────────────


def _build_flat(d: int, vectors: np.ndarray) -> faiss.Index:
    """Exact inner-product index (same as original)."""
    idx = faiss.IndexFlatIP(d)
    idx.add(vectors)
    return idx


def _build_hnsw(d: int, vectors: np.ndarray, m: int, ef_construction: int, ef_search: int) -> faiss.Index:
    """HNSW approximate nearest-neighbour index."""
    idx = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = ef_construction
    idx.hnsw.efSearch = ef_search
    idx.add(vectors)
    return idx


def _build_ivf_flat(d: int, vectors: np.ndarray, nlist: int, nprobe: int) -> faiss.Index:
    """IVF-Flat: partition space into Voronoi cells, exact within each cell."""
    quantizer = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    logger.info("  Training IVF-Flat with nlist=%d ...", nlist)
    idx.train(vectors)
    idx.nprobe = nprobe
    idx.add(vectors)
    return idx


def _build_ivf_pq(
    d: int, vectors: np.ndarray, nlist: int, m: int, nprobe: int
) -> faiss.Index:
    """IVF-PQ: IVF with product quantization (compressed vectors)."""
    quantizer = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    logger.info("  Training IVF-PQ with nlist=%d, m=%d ...", nlist, m)
    idx.train(vectors)
    idx.nprobe = nprobe
    idx.add(vectors)
    return idx


def _build_ivf_sq(
    d: int, vectors: np.ndarray, nlist: int, nprobe: int
) -> faiss.Index:
    """IVF-ScalarQuantizer: IVF with scalar quantization."""
    quantizer = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
    logger.info("  Training IVF-SQ with nlist=%d ...", nlist)
    idx.train(vectors)
    idx.nprobe = nprobe
    idx.add(vectors)
    return idx


def _build_pq(d: int, vectors: np.ndarray, m: int) -> faiss.Index:
    """Product Quantization only (no IVF)."""
    idx = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
    logger.info("  Training PQ with m=%d ...", m)
    idx.train(vectors)
    idx.add(vectors)
    return idx


def _build_lsh(d: int, vectors: np.ndarray, nbits: int) -> faiss.Index:
    """Locality-Sensitive Hashing index."""
    idx = faiss.IndexLSH(d, nbits)
    idx.add(vectors)
    return idx


BUILDERS = {
    "flat": lambda d, v, **kw: _build_flat(d, v),
    "hnsw": lambda d, v, **kw: _build_hnsw(d, v, kw.get("hnsw_m", 32), kw.get("hnsw_ef_construction", 200), kw.get("hnsw_ef_search", 64)),
    "ivf_flat": lambda d, v, **kw: _build_ivf_flat(d, v, kw.get("nlist", 256), kw.get("nprobe", 32)),
    "ivf_pq": lambda d, v, **kw: _build_ivf_pq(d, v, kw.get("nlist", 256), kw.get("pq_m", 32), kw.get("nprobe", 32)),
    "ivf_sq": lambda d, v, **kw: _build_ivf_sq(d, v, kw.get("nlist", 256), kw.get("nprobe", 32)),
    "pq": lambda d, v, **kw: _build_pq(d, v, kw.get("pq_m", 32)),
    "lsh": lambda d, v, **kw: _build_lsh(d, v, kw.get("lsh_nbits", 1024)),
}


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reindex a cached FAISS index into a different index type.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported index types:
  flat      IndexFlatIP           — exact, identical to original
  hnsw      IndexHNSWFlat         — fast ANN, good recall/latency trade-off
  ivf_flat  IndexIVFFlat          — IVF + exact cells, needs training
  ivf_pq    IndexIVFPQ            — IVF + compressed vectors, needs training
  ivf_sq    IndexIVFScalarQuantizer — IVF + scalar quantization, needs training
  pq        IndexPQ               — product quantization only, needs training
  lsh       IndexLSH              — locality-sensitive hashing

Examples:
  %(prog)s --source index/eval_cache/combined --type hnsw
  %(prog)s --source index/eval_cache/combined --type ivf_flat --nlist 512
  %(prog)s --source index/eval_cache/combined --type hnsw --output index_hnsw/eval_cache/combined
        """,
    )

    parser.add_argument(
        "--source", required=True,
        help="Path to the source index directory (must contain index.faiss + metadata.pkl)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory. Defaults to {source}_{type}/ (or {source}_{type}/eval_cache/combined/ "
             "if source looks like an eval_cache path).",
    )
    parser.add_argument(
        "--type", required=True, choices=list(BUILDERS.keys()),
        dest="index_type",
        help="Target index type.",
    )

    # HNSW params
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW connectivity (default: 32)")
    parser.add_argument("--hnsw-ef-construction", type=int, default=200, help="HNSW efConstruction (default: 200)")
    parser.add_argument("--hnsw-ef-search", type=int, default=64, help="HNSW efSearch (default: 64)")

    # IVF params
    parser.add_argument("--nlist", type=int, default=256, help="Number of Voronoi cells for IVF (default: 256)")
    parser.add_argument("--nprobe", type=int, default=32, help="Cells to probe at search time (default: 32)")

    # PQ params
    parser.add_argument("--pq-m", type=int, default=32, help="Number of sub-quantizers for PQ (default: 32)")

    # LSH params
    parser.add_argument("--lsh-nbits", type=int, default=1024, help="LSH hash bits (default: 1024)")

    # Safety
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing.")
    parser.add_argument("--keep-vectors", action="store_true",
                        help="When source is an eval_cache, copy eval_state.pkl + cache_key.json to output.")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output directory if it exists.")

    args = parser.parse_args()

    # ── Locate source files ────────────────────────────────────────────
    src = Path(args.source)
    if not src.exists():
        logger.error("Source directory does not exist: %s", src)
        sys.exit(1)

    faiss_path = src / "index.faiss"
    meta_path = src / "metadata.pkl"
    state_path = src / "eval_state.pkl"
    key_path = src / "cache_key.json"

    if not faiss_path.exists():
        logger.error("No index.faiss found in %s", src)
        sys.exit(1)
    if not meta_path.exists():
        logger.error("No metadata.pkl found in %s", src)
        sys.exit(1)

    # ── Determine output path ──────────────────────────────────────────
    if args.output:
        out = Path(args.output)
    else:
        # Auto-name: append _{type} to the parent of the source dir
        # e.g. index/eval_cache/combined -> index_hnsw/eval_cache/combined
        # or  index/ -> index_hnsw/
        parent = src.parent
        suffix = src.name
        out = parent / f"{suffix}_{args.index_type}"
        if suffix in ("combined",) or (src / "eval_state.pkl").exists():
            # eval_cache path: preserve the eval_cache/combined structure
            out = parent.parent / f"{parent.name}_{args.index_type}" / suffix

    logger.info("Output directory: %s", out)

    if out.exists() and not args.force:
        logger.error("Output directory already exists. Use --force to overwrite or pick a different --output.")
        sys.exit(1)

    # ── Load ───────────────────────────────────────────────────────────
    faiss_index = _load_faiss_index(faiss_path)
    metadata = _load_metadata(meta_path)
    eval_state = _load_eval_state(state_path)

    assert faiss_index.ntotal == len(metadata), (
        f"Mismatch: FAISS has {faiss_index.ntotal} vectors but metadata has {len(metadata)} entries"
    )

    ntotal = faiss_index.ntotal
    dim = faiss_index.d

    # ── Extract vectors ────────────────────────────────────────────────
    logger.info("Extracting %d vectors (dim=%d) ...", ntotal, dim)
    t0 = time.time()
    vectors = _extract_vectors(faiss_index)
    elapsed = time.time() - t0
    logger.info("Extraction done in %.1fs (shape=%s)", elapsed, vectors.shape)

    # ── Sanity check ───────────────────────────────────────────────────
    norms = np.linalg.norm(vectors, axis=1)
    logger.info(
        "Vector stats: min_norm=%.4f  max_norm=%.4f  mean_norm=%.4f",
        norms.min(), norms.max(), norms.mean(),
    )

    if args.dry_run:
        logger.info("=== DRY RUN — would build %s index ===", args.index_type)
        logger.info("  Source: %s", src)
        logger.info("  Output: %s", out)
        logger.info("  Vectors: %d x %d", ntotal, dim)
        logger.info("  Index type: %s", args.index_type)
        if args.index_type == "hnsw":
            logger.info("  HNSW: m=%d  efConstruction=%d  efSearch=%d",
                        args.hnsw_m, args.hnsw_ef_construction, args.hnsw_ef_search)
        elif args.index_type.startswith("ivf"):
            logger.info("  IVF: nlist=%d  nprobe=%d", args.nlist, args.nprobe)
            if args.index_type == "ivf_pq":
                logger.info("  PQ: m=%d", args.pq_m)
        elif args.index_type == "pq":
            logger.info("  PQ: m=%d", args.pq_m)
        elif args.index_type == "lsh":
            logger.info("  LSH: nbits=%d", args.lsh_nbits)
        return

    # ── Build new index ────────────────────────────────────────────────
    logger.info("Building %s index ...", args.index_type)
    t0 = time.time()
    builder = BUILDERS[args.index_type]
    new_index = builder(dim, vectors,
                        hnsw_m=args.hnsw_m,
                        hnsw_ef_construction=args.hnsw_ef_construction,
                        hnsw_ef_search=args.hnsw_ef_search,
                        nlist=args.nlist,
                        nprobe=args.nprobe,
                        pq_m=args.pq_m,
                        lsh_nbits=args.lsh_nbits)
    elapsed = time.time() - t0
    logger.info("Index built in %.1fs  ntotal=%d  type=%s", elapsed, new_index.ntotal, type(new_index).__name__)

    # ── Quick recall sanity check ──────────────────────────────────────
    logger.info("Running quick recall check (100 random queries) ...")
    n_queries = min(100, ntotal)
    rng = np.random.default_rng(42)
    query_ids = rng.choice(ntotal, size=n_queries, replace=False)
    queries = vectors[query_ids]  # already L2-normalized

    # Ground truth from flat index
    _, gt_ids = faiss_index.search(queries, 10)
    # New index search
    _, new_ids = new_index.search(queries, 10)

    hits = 0
    for gt_row, new_row in zip(gt_ids, new_ids):
        gt_set = set(gt_row[gt_row >= 0])
        new_set = set(new_row[new_row >= 0])
        hits += len(gt_set & new_set)
    recall_at_10 = hits / (n_queries * 10)
    logger.info("  Recall@10 vs original: %.1f%%", recall_at_10 * 100)

    if recall_at_10 < 0.5 and args.index_type != "flat":
        logger.warning("  Low recall! Consider increasing nlist/nprobe or using a different index type.")

    # ── Save ───────────────────────────────────────────────────────────
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Saving new index to %s ...", out)
    faiss.write_index(new_index, str(out / "index.faiss"))

    # Metadata and eval_state are shared across all index types
    with open(out / "metadata.pkl", "wb") as fh:
        pickle.dump(metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)

    if eval_state:
        with open(out / "eval_state.pkl", "wb") as fh:
            pickle.dump(eval_state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    if key_path.exists():
        shutil.copy2(key_path, out / "cache_key.json")

    logger.info("Done. Saved %d vectors to %s", new_index.ntotal, out)

    # ── Summary ────────────────────────────────────────────────────────
    orig_size = faiss_path.stat().st_size
    new_size = (out / "index.faiss").stat().st_size
    logger.info("Index file size: original=%s  new=%s  ratio=%.2fx",
                _fmt_size(orig_size), _fmt_size(new_size), new_size / orig_size if orig_size else 0)


def _fmt_size(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


if __name__ == "__main__":
    main()
