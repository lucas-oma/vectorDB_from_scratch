from abc import ABC, abstractmethod
from typing import List
import numpy as np


class SimilarityMetric(ABC):
    higher_is_better: bool = False  # meaning lower = more similar (e.g. L2 distance)
    requires_unit_norm: bool = False  # meaning vectors do not need to be unit normalized
    
    @abstractmethod
    def compute(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute similarity between two vectors."""
        pass


class CosineSimilarity(SimilarityMetric):
    higher_is_better: bool = True  # meaning higher = more similar
    requires_unit_norm: bool = True  # meaning vectors must be unit normalized

    def compute(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity: (A.B) / (||A|| * ||B||)"""
        a = np.array(vec1)
        b = np.array(vec2)
        
        dot_product = np.dot(a, b)
        norm1 = np.linalg.norm(a)
        norm2 = np.linalg.norm(b)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class L2Similarity(SimilarityMetric):
    higher_is_better: bool = False
    requires_unit_norm: bool = False
    
    def compute(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute L2 distance"""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.linalg.norm(a - b)


class ManhattanSimilarity(SimilarityMetric):
    higher_is_better: bool = False
    requires_unit_norm: bool = False
    
    def compute(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute Manhattan distance"""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.sum(np.abs(a - b))
