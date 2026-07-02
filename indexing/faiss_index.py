"""FAISS vector-index wrapper with persistence.

Supports ``IndexFlatIP`` (exact inner-product) and ``IndexHNSWFlat``
(approximate nearest-neighbour) index types.  All vectors are
L2-normalised before insertion so that inner product equals cosine
similarity.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from parser.extract import CodeEntity

logger = logging.getLogger(__name__)


class FaissCodeIndex:
    """Persistent FAISS index with associated metadata.

    Parameters
    ----------
    dimension:
        Embedding dimensionality.
    index_type:
        ``"flat"`` for ``IndexFlatIP`` or ``"hnsw"`` for ``IndexHNSWFlat``.
    hnsw_m:
        HNSW connectivity parameter (ignored for flat index).
    hnsw_ef_construction:
        HNSW build-time search depth.
    hnsw_ef_search:
        HNSW search-time search depth.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ) -> None:
        self._dimension = dimension
        self._index_type = index_type
        self._metadata: List[CodeEntity] = []

        if index_type == "flat":
            self._index = faiss.IndexFlatIP(dimension)
        elif index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = hnsw_ef_construction
            self._index.hnsw.efSearch = hnsw_ef_search
        else:
            raise ValueError(f"Unknown index_type: {index_type!r}")

    # ── Build ───────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, metadata: List[CodeEntity]) -> None:
        """Normalise *embeddings*, add to the index, and store *metadata*.

        Parameters
        ----------
        embeddings:
            Shape ``(n, dim)``, float32.
        metadata:
            One ``CodeEntity`` per row in *embeddings*.
        """
        assert embeddings.shape[0] == len(metadata), (
            f"Embeddings count ({embeddings.shape[0]}) != metadata count ({len(metadata)})"
        )
        assert embeddings.shape[1] == self._dimension

        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (embeddings / norms).astype(np.float32)

        self._index.add(normed)
        self._metadata.extend(metadata)
        logger.info(
            "Built index: %d vectors (type=%s)", self._index.ntotal, self._index_type
        )

    # ── Search ──────────────────────────────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[CodeEntity, float]]:
        """Return the *top_k* nearest neighbours of *query_vector*.

        Parameters
        ----------
        query_vector:
            Shape ``(dim,)`` — will be L2-normalised internally.
        top_k:
            Number of results to return.

        Returns
        -------
        List[Tuple[CodeEntity, float]]
            Sorted by descending similarity score.
        """
        q = query_vector.reshape(1, -1).astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        distances, indices = self._index.search(q, k)
        results: List[Tuple[CodeEntity, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._metadata[idx], float(dist)))
        return results

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Write the index and metadata to *directory*."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(dir_path / "index.faiss"))
        with open(dir_path / "metadata.pkl", "wb") as fh:
            pickle.dump(self._metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Saved index (%d vectors) to %s", self._index.ntotal, directory)

    @classmethod
    def load(cls, directory: str) -> "FaissCodeIndex":
        """Load a previously saved index and metadata from *directory*."""
        dir_path = Path(directory)
        index = faiss.read_index(str(dir_path / "index.faiss"))
        with open(dir_path / "metadata.pkl", "rb") as fh:
            metadata = pickle.load(fh)

        obj = cls.__new__(cls)
        obj._index = index
        obj._metadata = metadata
        obj._dimension = index.d
        obj._index_type = "flat" if isinstance(index, faiss.IndexFlatIP) else "hnsw"
        logger.info("Loaded index (%d vectors) from %s", index.ntotal, directory)
        return obj

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def ntotal(self) -> int:
        """Total number of indexed vectors."""
        return self._index.ntotal

    @property
    def metadata(self) -> List[CodeEntity]:
        """Shallow copy of the stored metadata."""
        return list(self._metadata)

    @property
    def dimension(self) -> int:
        return self._dimension
