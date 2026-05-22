"""Ensemble aggregation strategies for GeoPrompt."""
import numpy as np
from typing import List, Optional


class EnsembleAggregator:
    METHODS = ("confidence_weighted", "majority_voting", "max_pooling")

    def __init__(self, method: str = "confidence_weighted"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method

    def aggregate(self, sim_maps: List[np.ndarray],
                  confidence_scores: Optional[List[float]] = None) -> np.ndarray:
        if not sim_maps:
            raise ValueError("sim_maps must be non-empty")
        stack = np.stack(sim_maps, axis=0)
        if self.method == "confidence_weighted":
            return self._confidence_weighted(stack, confidence_scores)
        elif self.method == "majority_voting":
            return self._majority_voting(stack)
        return stack.max(axis=0)

    @staticmethod
    def _confidence_weighted(stack, scores):
        T = stack.shape[0]
        if scores is None:
            w = np.ones(T, dtype=np.float32) / T
        else:
            s = np.array(scores, dtype=np.float32)
            s = np.clip(s, 0, None)
            total = s.sum()
            w = s / total if total > 0 else np.ones(T) / T
        return (stack * w[:, None, None]).sum(axis=0)

    @staticmethod
    def _majority_voting(stack):
        T = stack.shape[0]
        thresholds = np.median(stack.reshape(T, -1), axis=1)
        binary = (stack > thresholds[:, None, None]).astype(np.float32)
        return binary.mean(axis=0)
