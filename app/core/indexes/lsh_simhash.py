import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from app.core.models import Chunk, SearchResult
from app.core.indexing import VectorIndex
from app.core.similarity_metrics import CosineSimilarity


@dataclass
class _VecItem:
    vec: np.ndarray  # unit-norm embedding vector float32
    keys: np.ndarray # simhash keys for vector. shape: (n_tables,), dtype=uint64


class SimHashLSHIndex(VectorIndex):
    """
    Multi-table SimHash LSH (cosine).

    Parameters:
      - dimension: dimension of the vectors
      - n_bits: bits per table (band width)
      - n_tables: number of independent tables (bands)
      - rng_seed: random seed for reproducible hyperplane generation
    """

    def __init__(self, dimension: int, n_bits: int = 16, n_tables: int = 8, rng_seed: int = 42):
        super().__init__(dimension, similarity_metric=CosineSimilarity())
        
        if n_bits <= 0 or n_tables <= 0:
            raise ValueError("n_bits and n_tables must be >= 1")

        if n_bits > 64:
            raise ValueError("n_bits must be <= 64")

        self.n_bits   = int(n_bits)
        self.n_tables = int(n_tables)
        self.rng      = np.random.default_rng(rng_seed)

        self.vec_items: Dict[str, _VecItem] = {} # in-memory chunks

        # A list of dictionaries (one dictionary per hash table)
        # Each dictionary maps hash keys (integers) to sets/buckets of chunk IDs
        self.tables: List[Dict[int, Set[str]]] = [dict() for _ in range(self.n_tables)]

        # Random hyperplanes
        H = self.rng.standard_normal(size=(self.n_tables, self.n_bits, self.dimension)).astype(np.float32)
        self.hyperplanes = H / (np.linalg.norm(H, axis=2, keepdims=True) + 1e-12)

        # Precomputed weights (onehot encoding)
        self._bit_weights = (1 << np.arange(self.n_bits, dtype=np.uint64)).astype(np.uint64)

    def _simhash_keys(self, v: np.ndarray) -> np.ndarray:
        # (n_tables, n_bits, d) @ (d,) -> (n_tables, n_bits)
        dots = np.tensordot(self.hyperplanes, v, axes=([2], [0]))
        bits = (dots >= 0)

        # int keys for each table (n_tables,)
        return (bits * self._bit_weights).sum(axis=1, dtype=np.uint64)

    def _add_to_buckets(self, chunk_id: str, keys: np.ndarray) -> None:
        """Add chunk_id to buckets based on hash keys."""
        for t, key in enumerate(keys):
            bucket = self.tables[t].setdefault(int(key), set())
            bucket.add(chunk_id)

    def _remove_from_buckets(self, chunk_id: str, keys: np.ndarray) -> None:
        """Remove chunk_id from buckets based on hash keys."""
        for t, key in enumerate(keys):
            bucket = self.tables[t].get(int(key))
            if bucket is not None:
                bucket.discard(chunk_id)

    def add_chunk(self, chunk: Chunk) -> None:
        vec = self._normalize_if_needed(chunk.embedding)
        keys = self._simhash_keys(vec)
        self.vec_items[chunk.id] = _VecItem(vec=vec, keys=keys)

        self._add_to_buckets(chunk.id, keys)

    def update_chunk(self, chunk_id: str, new_chunk: Chunk) -> bool:
        existed = chunk_id in self.vec_items
        old_vec_item = self.vec_items.get(chunk_id)

        # remove old
        if old_vec_item is not None:
            self._remove_from_buckets(chunk_id, old_vec_item.keys)

        # compute and add new
        vec  = self._normalize_if_needed(new_chunk.embedding)
        keys = self._simhash_keys(vec)
        self.vec_items[chunk_id] = _VecItem(vec=vec, keys=keys)
        self._add_to_buckets(chunk_id, keys)

        return existed

    def remove_chunk(self, chunk_id: str) -> bool:
        item = self.vec_items.pop(chunk_id, None)
        if item is None:
            return False

        self._remove_from_buckets(chunk_id, item.keys)
        return True

    def search(self, query_embedding: List[float], k: int,
               metadata_filters=None) -> List[SearchResult]:
        if k <= 0:
            return []

        q = self._normalize_if_needed(query_embedding)

        qkeys = self._simhash_keys(q)
        cand_chunk_ids: Set[str] = set() # candidates = union of buckets across tables for query's keys
        for t, key in enumerate(qkeys):
            bucket = self.tables[t].get(int(key))
            if bucket:
                cand_chunk_ids.update(bucket)

        if not cand_chunk_ids:
            return []

        # exact rerank by cosine (vectorized)
        ids = list(cand_chunk_ids)
        X = np.stack([self.vec_items[i].vec for i in ids], dtype=np.float32)
        scores = X @ q
        k_eff = min(k, len(scores))
        top_idx = np.argpartition(-scores, kth=k_eff-1)[:k_eff]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]

        return [SearchResult(chunk_id=ids[i], similarity_score=float(scores[i]))
                for i in top_sorted]

    def get_complexity(self) -> tuple[str, str]:
        return ("Space: O(n_tables*n) + O(n_tables*n_bits*d)",
                "Build: O(n_tables*n_bits*n*d) | Query: O(n_tables*n_bits*d) + O(C*d) + top-k")

