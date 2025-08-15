from __future__ import annotations
import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.core.models import Chunk, SearchResult
from app.core.indexing import VectorIndex
from app.core.similarity_metrics import SimilarityMetric, CosineSimilarity


class FlatIndex(VectorIndex):
    """
    Flat/linearindex
    Stores: chunk_id -> embedding vector
    """

    def __init__(self, dimension: int, similarity_metric: Optional[SimilarityMetric] = None):
        super().__init__(dimension)
        self.similarity_metric: SimilarityMetric = similarity_metric or CosineSimilarity()
        self._vecs: Dict[str, np.ndarray] = {}  # index: chunk_id -> vector

    def add_chunk(self, chunk: Chunk) -> None:
        emb = chunk.embedding
        if len(emb) != self.dimension:
            raise ValueError(f"dim mismatch for chunk {chunk.id}: {len(emb)} != {self.dimension}")
        self._vecs[chunk.id] = self._normalize_if_needed(emb)

    def update_chunk(self, chunk_id: str, new_chunk: Chunk) -> bool:
        existed = chunk_id in self._vecs
        if existed:
            if len(new_chunk.embedding) != self.dimension:
                raise ValueError(f"dim mismatch for chunk {chunk_id}: {len(new_chunk.embedding)} != {self.dimension}")
            self._vecs[chunk_id] = self._normalize_if_needed(new_chunk.embedding)
        else:
            # if new id -> treat as add
            self.add_chunk(new_chunk)
        return existed

    def remove_chunk(self, chunk_id: str) -> bool:
        return self._vecs.pop(chunk_id, None) is not None

    def search(
        self,
        query_embedding: List[float],
        k: int,
        metadata_filters: Optional[Dict[str, str]] = None,  # TODO: not used for now
    ) -> List[SearchResult]:
        if k <= 0:
            return []
        if len(query_embedding) != self.dimension:
            raise ValueError(f"dim mismatch: query dim {len(query_embedding)} != {self.dimension}")
        if not self._vecs:
            return []

        q = self._normalize_if_needed(query_embedding)
        higher_is_better = self.similarity_metric.higher_is_better
        k_eff = min(k, len(self._vecs)) # effective k

        # size-k min-heap over score (higher is better)
        heap: List[Tuple[float, str, float]] = [] # format: (score, chunk id, raw similarity metric value)
        for cid, v in self._vecs.items():
            raw = self.similarity_metric.compute(q, v)
            score = raw if higher_is_better else -raw
            if len(heap) < k_eff:
                heapq.heappush(heap, (score, cid, raw))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, cid, raw))

        heap.sort(key=lambda x: x[0], reverse=True)
        return [SearchResult(chunk_id=cid, similarity_score=raw, chunk=None) for _, cid, raw in heap]

    def get_complexity(self) -> tuple[str, str]:
        return ("O(N*d)", "Build: O(1) | Query: O(N*d) + O(N log k)")

    # TODO: optional snapshot hooks for fast index loading from disk
    # def export_snapshot(self) -> Tuple[List[str], np.ndarray]:
    #     ids = list(self._vecs.keys())
    #     mat = np.vstack([self._vecs[i] for i in ids]) if ids else np.empty((0, self.dimension), dtype=np.float32)
    #     return ids, mat

    # def import_snapshot(self, ids: List[str], mat: np.ndarray) -> None:
    #     self._vecs.clear()
    #     for i, cid in enumerate(ids):
    #         self._vecs[cid] = np.asarray(mat[i], dtype=np.float32)
