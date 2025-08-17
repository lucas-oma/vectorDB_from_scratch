import numpy as np
from typing import Dict, List, Optional, Set
from app.core.models import Chunk, SearchResult
from app.core.indexing import VectorIndex
from app.core.similarity_metrics import CosineSimilarity


class IVFIndex(VectorIndex):
    """
    IVF (Inverted File) index (FAISS-style)

    Parameters:
      - dimension: dimension of the vectors
      - n_clusters: number of clusters
      - n_probes: number of clusters to probe per query
      - train_iters: number of k-means iterations
      - rng_seed: random seed for reproducible initialization
      
    FAISS-style: Must be trained before any add/update/search operations.
    """

    def __init__(self, dimension: int, n_clusters: int = 64, n_probes: int = 1,
                 train_iters: int = 20, rng_seed: int = 42):
        super().__init__(dimension, similarity_metric=CosineSimilarity())

        self.n_clusters = max(1, int(n_clusters))                      # number of clusters/lists (>=1)
        self.n_probes   = max(1, int(n_probes))                        # clusters to probe per query (>=1)
        self.train_iters = int(train_iters)                             # k-means iterations
        self.rng = np.random.default_rng(rng_seed)                      # RNG for reproducible init/reseeds

        self.centroids: Optional[np.ndarray] = None                     # (n_clusters, dim) unit-norm; None before train
        self.inverted_lists: Dict[int, Set[str]] = {i: set() for i in range(self.n_clusters)}  # cluster_id -> ids
        self.chunk_vectors: Dict[str, np.ndarray] = {}                  # chunk_id -> unit-norm vector cached in RAM
        self.chunk_to_cluster: Dict[str, int] = {}                      # chunk_id -> cluster_id (for O(1) reassign/removal)
        self.is_initializing: bool = True                               # Flag to allow adding chunks before training. This prevents error raising when index initializes for the first time.


    def _assign_cluster(self, chunk_id: str, vec: np.ndarray) -> None:
        """Assign unit-norm vec to its nearest centroid (by cosine) and record membership."""
        # centroids shape may be < n_clusters if k > n during training; use what we have
        sims = self.centroids @ vec  # (n_actual_clusters,)
        cid = int(np.argmax(sims))
        self.inverted_lists[cid].add(chunk_id)
        self.chunk_to_cluster[chunk_id] = cid

    def _ensure_trained(self) -> None:
        if self.centroids is None:
            raise RuntimeError("IVFIndex is not trained. Call train(...) before add/update/search")

    def add_chunk(self, chunk: Chunk) -> None:
        v = self._normalize_if_needed(chunk.embedding)
        self.chunk_vectors[chunk.id] = v
        
        # If not initializing, ensure trained and assign to cluster
        if not self.is_initializing:
            self._ensure_trained()
            self._assign_cluster(chunk.id, v)
        # If initializing, just store the chunk (will be assigned during training)

    def update_chunk(self, chunk_id: str, new_chunk: Chunk) -> bool:
        if not self.is_initializing:
            self._ensure_trained()
        
        existed = chunk_id in self.chunk_vectors
        v = self._normalize_if_needed(new_chunk.embedding)
        self.chunk_vectors[chunk_id] = v

        # Only update cluster assignment if not initializing
        if not self.is_initializing:
            old = self.chunk_to_cluster.pop(chunk_id, None)
            if old is not None:
                self.inverted_lists[old].discard(chunk_id)
            self._assign_cluster(chunk_id, v)
        
        return existed

    def remove_chunk(self, chunk_id: str) -> bool:
        removed = False

        if chunk_id in self.chunk_vectors:
            self.chunk_vectors.pop(chunk_id, None)
            removed = True

        cid = self.chunk_to_cluster.pop(chunk_id, None)
        if cid is not None:
            self.inverted_lists[cid].discard(chunk_id)
            removed = True

        return removed

    def search(self, query_embedding: List[float], k: int, metadata_filters=None) -> List[SearchResult]:
        self._ensure_trained()

        if k <= 0:
            return []

        q = self._normalize_if_needed(query_embedding)

        n_probe  = min(self.n_probes, self.centroids.shape[0])

        centroid_sims = self.centroids @ q
        probe_ids = np.argpartition(-centroid_sims, kth=n_probe-1)[:n_probe]
        
        # rank the probed clusters by similarity
        probe_ids = probe_ids[np.argsort(-centroid_sims[probe_ids])]

        # Gather candidates
        cand_ids: List[str] = []
        for cid in probe_ids:
            cand_ids.extend(self.inverted_lists[int(cid)])

        if not cand_ids:
            return []

        # Vectorized re-rank:
        Xc = np.stack([self.chunk_vectors[i] for i in cand_ids])  # (m, d)
        scores = Xc @ q # (m,)
        k_eff = min(k, len(scores))
        top_idx = np.argpartition(-scores, kth=k_eff-1)[:k_eff]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]

        return [SearchResult(chunk_id=cand_ids[i], similarity_score=float(scores[i]))
                for i in top_sorted]

    # TODO: double check complexities
    def get_complexity(self) -> tuple[str, str]:
        return ("Space: O(n) + O(k*d)",
                "Build: O(n*k*iters) k-means + O(n) assign | Query: O(n_probes*avg_list*d) + O(m + k log k)")


    def train(self, sample_vectors: Optional[np.ndarray] = None) -> None:
        """
        Compute centroids. If sample_vectors is None, use current vectors (incl. pending).
        After training, assign all pending ids to clusters.
        """
        # get training data
        if sample_vectors is None:
            if not self.chunk_vectors:
                return
            X = np.stack(list(self.chunk_vectors.values())).astype(np.float32, copy=False)
        else:
            X = np.asarray(sample_vectors, dtype=np.float32)

        # k-means (returns unit-norm centers)
        centers = self._kmeans(X, self.n_clusters, iters=self.train_iters)
        self.centroids = centers  # (k_actual, d)

        # Rebuild inverted lists from scratch under new centroids
        self.inverted_lists = {i: set() for i in range(self.centroids.shape[0])}
        self.chunk_to_cluster.clear()
        
        # assign all chunks to new clusters
        for chunk_id, vec in self.chunk_vectors.items():
            self._assign_cluster(chunk_id, vec)
        
        # Set initializing flag to False - now require training check for future operations
        self.is_initializing = False        

    def _kmeans(self, X: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
        """
        Cosine k-means (vectorized): returns unit-norm centers of shape (k_actual, dim)
        If k > n, uses k_actual = n (i.e. cap k)
        """
        n, _ = X.shape
        k = min(k, n)

        # unit-normalize once for cosine
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        # init centers from data
        idx = self.rng.choice(n, size=k, replace=False)
        C = Xn[idx].copy()  # (k, d), unit

        for _ in range(iters):
            # assign all points at once
            sim = Xn @ C.T                      # (n, k)
            labels = np.argmax(sim, axis=1)     # (n,)

            # update: sum per cluster (vectorized) then renormalize
            new_C = np.zeros_like(C)
            np.add.at(new_C, labels, Xn)        # accumulate rows of Xn into cluster rows of new_C
            counts = np.bincount(labels, minlength=k).astype(np.float32).reshape(-1, 1)
            # avoid div by zero; reseed empty clusters
            empty = (counts[:, 0] == 0)
            new_C[~empty] /= counts[~empty]
            # renormalize non-empty
            norms = np.linalg.norm(new_C, axis=1, keepdims=True) + 1e-12
            new_C[~empty] /= norms[~empty]
            # reseed empties randomly
            if np.any(empty):
                reseed_idx = self.rng.choice(n, size=int(empty.sum()), replace=False)
                new_C[empty] = Xn[reseed_idx]

            # convergence
            if np.allclose(new_C, C, rtol=1e-5, atol=1e-7):
                C = new_C
                break
            C = new_C

        return C
