from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from app.core.models import Chunk, SearchResult
from app.core.similarity_metrics import SimilarityMetric, CosineSimilarity


class VectorIndex(ABC):
    """Abstract base class for vector indexing algorithms."""
    
    def __init__(self, dimension: int, similarity_metric: SimilarityMetric = None):
        self.dimension = dimension
        self.similarity_metric = similarity_metric or CosineSimilarity()


    def _normalize_if_needed(self, emb: List[float]) -> np.ndarray:
        arr = np.asarray(emb, dtype=np.float32)
        if getattr(self.similarity_metric, "requires_unit_norm", False):
            n = float(np.linalg.norm(arr))
            if n > 0.0:
                arr = arr / n
        return arr

    @abstractmethod
    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the index."""
        pass

    @abstractmethod
    def update_chunk(self, chunk_id: str, new_chunk: Chunk) -> bool:
        """Update a chunk in the index."""
        pass

    @abstractmethod
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from the index."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], k: int, 
               metadata_filters: Optional[Dict[str, str]] = None) -> List[SearchResult]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    def get_complexity(self) -> tuple[str, str]:
        """Return (space_complexity, time_complexity) as strings."""
        pass
