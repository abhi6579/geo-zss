"""Hierarchical prompt aggregation."""
import numpy as np
from typing import Dict, List, Optional

LEVEL_NAMES = ("scene", "object", "subclass", "pixel")
DEFAULT_LEVEL_WEIGHTS = {"scene": 0.15, "object": 0.40, "subclass": 0.30, "pixel": 0.15}


class HierarchicalPromptAggregator:
    def __init__(self, level_weights: Optional[Dict[str, float]] = None):
        self.level_weights = level_weights or DEFAULT_LEVEL_WEIGHTS.copy()
        total = sum(self.level_weights.values())
        if abs(total - 1.0) > 1e-3:
            for k in self.level_weights:
                self.level_weights[k] /= total

    def aggregate(self, level_maps: Dict[str, np.ndarray]) -> np.ndarray:
        out = None
        for level, w in self.level_weights.items():
            if level not in level_maps or w == 0:
                continue
            out = level_maps[level] * w if out is None else out + level_maps[level] * w
        return out if out is not None else next(iter(level_maps.values()))
