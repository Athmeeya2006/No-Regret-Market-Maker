"""
RegimeGenerator: generates market regimes for testing adaptation speed.

Each regime is a dict with keys:
  name   : human-readable label
  sigma  : mid-price volatility
  mu     : fraction of arrivals that are informed (0-1)
  lambda : total arrival rate

Transitions:
  abrupt   : parameters jump at regime boundaries
  gradual  : linear interpolation between regimes over `blend_frac` fraction
             of regime_duration (e.g. blend_frac=0.1 => 10% blend period)
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional


_DEFAULT_REGIMES = [
    {"name": "calm",           "sigma": 0.005, "mu": 0.10, "lambda": 5.0},
    {"name": "volatile",       "sigma": 0.020, "mu": 0.30, "lambda": 8.0},
    {"name": "illiquid",       "sigma": 0.010, "mu": 0.05, "lambda": 2.0},
    {"name": "informed_heavy", "sigma": 0.015, "mu": 0.50, "lambda": 6.0},
]


class RegimeGenerator:
    def __init__(
        self,
        regimes:          List[Dict] = None,
        regime_duration:  int        = 2000,
        transition:       str        = "abrupt",
        blend_frac:       float      = 0.1,
    ):
        self.regimes    = regimes if regimes is not None else _DEFAULT_REGIMES
        self.duration   = int(regime_duration)
        self.transition = transition
        self.blend_frac = float(blend_frac)
        self._n         = len(self.regimes)

    def get_params(self, t: int) -> Dict:
        idx  = (t // self.duration) % self._n
        frac = (t % self.duration) / self.duration

        if self.transition == "abrupt":
            return dict(self.regimes[idx])

        # Gradual: blend last `blend_frac` of current regime into next
        blend_start = 1.0 - self.blend_frac
        if frac < blend_start:
            return dict(self.regimes[idx])

        alpha = (frac - blend_start) / self.blend_frac   # 0 -> 1
        next_idx = (idx + 1) % self._n
        return self._blend(self.regimes[idx], self.regimes[next_idx], alpha)

    @staticmethod
    def _blend(a: Dict, b: Dict, alpha: float) -> Dict:
        result = {}
        for key in a:
            if key == "name":
                result["name"] = a["name"] + "->" + b["name"]
            else:
                result[key] = (1.0 - alpha) * a[key] + alpha * b[key]
        return result

    def get_change_times(self, total_rounds: int) -> List[int]:
        return list(range(self.duration, total_rounds, self.duration))

    def regime_at(self, t: int) -> str:
        idx = (t // self.duration) % self._n
        return self.regimes[idx]["name"]

    def __repr__(self) -> str:
        names = [r["name"] for r in self.regimes]
        return f"RegimeGenerator(regimes={names}, duration={self.duration}, transition={self.transition})"
