"""External regret tracking against the best fixed spread in hindsight."""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple

# Re-export RegretDecomposer and TradeRecord for backward compatibility
from python.evaluation.regret_decomposer import RegretDecomposer, TradeRecord


class RegretTracker:
    """
    Tracks regret for a single market maker over a simulation run.

    For full accuracy, the simulator provides counterfactual_rewards:
    a dict {spread: reward_if_chosen} at every round. These are used
    to compute exact per-round counterfactual regret.
    """

    def __init__(self, spread_choices: List[float]):
        self.spreads = spread_choices
        self.K       = len(spread_choices)

        self.actual_rewards:            List[float] = []
        self.chosen_spread_indices:     List[int]   = []
        self.counterfactual_rewards:    Dict[float, List[float]] = {
            s: [] for s in spread_choices
        }

    def record_round(
        self,
        chosen_spread:          float,
        actual_reward:          float,
        counterfactual_rewards: Dict[float, float],
    ):
        self.actual_rewards.append(actual_reward)
        # Find closest spread in choices
        idx = int(np.argmin(np.abs(np.array(self.spreads) - chosen_spread)))
        self.chosen_spread_indices.append(idx)

        for s in self.spreads:
            cf = counterfactual_rewards.get(s, actual_reward)
            self.counterfactual_rewards[s].append(cf)

    # ---- Core regret metrics ------------------------------------

    def total_regret(self) -> float:
        if not self.actual_rewards:
            return 0.0
        actual_total = sum(self.actual_rewards)
        best_fixed   = max(
            sum(v) for v in self.counterfactual_rewards.values()
        )
        return max(0.0, best_fixed - actual_total)

    def regret_curve(self) -> np.ndarray:
        """
        Cumulative regret at each timestep.
        Shape: (T,)
        """
        T = len(self.actual_rewards)
        if T == 0:
            return np.array([])

        actual_cum = np.cumsum(self.actual_rewards)

        cf_mat     = np.array([
            self.counterfactual_rewards[s] for s in self.spreads
        ])                                     # (K, T)
        best_fixed_cum = np.cumsum(cf_mat, axis=1).max(axis=0)   # (T,)

        return np.maximum(0.0, best_fixed_cum - actual_cum)

    def per_spread_regret(self) -> Dict[float, float]:
        """Regret vs each individual fixed arm (not just the best)."""
        actual_total = sum(self.actual_rewards)
        return {
            s: max(0.0, sum(v) - actual_total)
            for s, v in self.counterfactual_rewards.items()
        }

    def best_fixed_arm(self) -> float:
        """Which spread would have been best in hindsight?"""
        totals = {s: sum(v) for s, v in self.counterfactual_rewards.items()}
        return max(totals, key=totals.get)

    def rolling_regret(self, window: int = 500) -> np.ndarray:
        """Rolling regret over a sliding window."""
        T    = len(self.actual_rewards)
        out  = np.zeros(T)
        for t in range(T):
            start     = max(0, t - window + 1)
            actual_w  = sum(self.actual_rewards[start:t+1])
            best_fixed_w = max(
                sum(v[start:t+1])
                for v in self.counterfactual_rewards.values()
            )
            out[t] = max(0.0, best_fixed_w - actual_w)
        return out

    def theoretical_bound(self, T: Optional[int] = None) -> np.ndarray:
        """
        Theoretical Exp3 regret bound at each timestep:
          B(t) = 2 * sqrt(t * K * ln(K))
        """
        T  = T or len(self.actual_rewards)
        ts = np.arange(1, T + 1)
        return 2.0 * np.sqrt(ts * self.K * np.log(self.K))

    def ci_bootstrap(
        self,
        n_bootstrap: int = 200,
        alpha:       float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap confidence interval for the regret curve.
        Returns (lower, upper) each of shape (T,).
        """
        T = len(self.actual_rewards)
        if T < 2:
            curve = self.regret_curve()
            return curve, curve

        boot_curves = np.zeros((n_bootstrap, T))
        actual_arr  = np.array(self.actual_rewards)
        cf_arr      = np.array([
            self.counterfactual_rewards[s] for s in self.spreads
        ])   # (K, T)

        rng = np.random.default_rng(42)
        for b in range(n_bootstrap):
            idx        = rng.integers(0, T, size=T)
            actual_b   = actual_arr[idx]
            cf_b       = cf_arr[:, idx]
            actual_cum = np.cumsum(actual_b)
            best_cum   = np.cumsum(cf_b, axis=1).max(axis=0)
            boot_curves[b] = np.maximum(0.0, best_cum - actual_cum)

        lower = np.quantile(boot_curves, alpha / 2,     axis=0)
        upper = np.quantile(boot_curves, 1 - alpha / 2, axis=0)
        return lower, upper

    def summary(self) -> Dict:
        curve = self.regret_curve()
        T     = len(self.actual_rewards)
        return {
            "T":                 T,
            "total_regret":      float(self.total_regret()),
            "regret_per_round":  float(self.total_regret() / max(T, 1)),
            "theoretical_bound": float(self.theoretical_bound()[-1]) if T > 0 else 0.0,
            "bound_slack":       float(
                self.theoretical_bound()[-1] - self.total_regret()
            ) if T > 0 else 0.0,
            "best_fixed_arm":    self.best_fixed_arm(),
            "actual_total":      float(sum(self.actual_rewards)),
        }
