"""Fixed-spread market maker: always quotes the same spread."""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List


class FixedSpreadMM:
    """Always quotes a single fixed spread. Serves as a passive baseline."""

    def __init__(self, spread: float):
        self.spread          = float(spread)
        self.reward_history: List[float] = []
        self.t:              int = 0

    def get_distribution(self) -> np.ndarray:
        raise NotImplementedError

    def choose_spread(self, context: Optional[dict] = None) -> float:
        return self.spread

    def update(self, reward: float, counterfactual_rewards=None):
        self.reward_history.append(reward)
        self.t += 1

    def reset(self):
        self.reward_history.clear()
        self.t = 0

    def __repr__(self) -> str:
        return f"FixedSpreadMM(spread={self.spread:.4f})"
