"""Adaptation-speed metrics for regime-switching simulations."""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AdaptationEvent:
    change_time:       int
    old_regime:        str
    new_regime:        str
    delta:             float        # reward gap in new regime
    baseline_reward:   float        # pre-change rolling average
    adaptation_time:   Optional[int]  # None if never recovered
    recovery_reward:   Optional[float]
    theoretical_pred:  float        # O(log K / delta^2)


class AdaptationSpeedAnalyzer:
    """
    Measures how fast an algorithm recovers after regime changes.

    Parameters
    ----------
    K       : number of arms (spread choices)
    window  : rolling window for computing baseline and recovery averages
    epsilon : recovery threshold - how close to baseline counts as "recovered"
              (fraction of baseline, e.g. 0.1 = within 10%)
    """

    def __init__(
        self,
        K:       int   = 5,
        window:  int   = 50,
        epsilon: float = 0.1,
    ):
        self.K       = K
        self.window  = window
        self.epsilon = epsilon

    def measure_adaptation(
        self,
        reward_series: np.ndarray,
        change_times:  List[int],
        regime_names:  Optional[List[str]] = None,
        spread_choices: Optional[List[float]] = None,
        counterfactual_matrix: Optional[np.ndarray] = None,
    ) -> List[AdaptationEvent]:
        """
        For each regime change time, measure how long until the algorithm
        recovers to within epsilon of its pre-change performance.

        Parameters
        ----------
        reward_series   : (T,) array of per-round rewards
        change_times    : list of round indices where regime changed
        regime_names    : optional list of (old, new) pairs per change
        counterfactual_matrix : (T, K) - used to compute reward gap delta
        """
        T       = len(reward_series)
        events  = []

        for i, t_change in enumerate(change_times):
            if t_change >= T:
                break

            # Pre-change baseline
            pre_start = max(0, t_change - self.window)
            baseline  = float(np.mean(reward_series[pre_start:t_change]))

            # Reward gap delta in new regime (from counterfactuals)
            delta = self._estimate_reward_gap(
                reward_series, counterfactual_matrix, t_change, T
            )

            # Theoretical prediction
            theoretical = np.log(self.K) / (delta ** 2 + 1e-8)

            # Find recovery time
            adaptation_time  = None
            recovery_reward  = None
            max_search       = min(t_change + 1000, T)

            for t in range(t_change, max_search):
                win_start = max(t_change, t - self.window + 1)
                post_avg  = float(np.mean(reward_series[win_start:t + 1]))
                if abs(baseline) < 1e-8:
                    recovered = abs(post_avg - baseline) < self.epsilon
                else:
                    recovered = abs(post_avg - baseline) / abs(baseline) < self.epsilon
                if recovered and t > t_change:
                    adaptation_time = t - t_change
                    recovery_reward = post_avg
                    break

            old_name = regime_names[i][0] if regime_names else f"regime_{i}"
            new_name = regime_names[i][1] if regime_names else f"regime_{i+1}"

            events.append(AdaptationEvent(
                change_time      = t_change,
                old_regime       = old_name,
                new_regime       = new_name,
                delta            = delta,
                baseline_reward  = baseline,
                adaptation_time  = adaptation_time,
                recovery_reward  = recovery_reward,
                theoretical_pred = theoretical,
            ))

        return events

    def _estimate_reward_gap(
        self,
        reward_series:         np.ndarray,
        counterfactual_matrix: Optional[np.ndarray],
        t_change:              int,
        T:                     int,
    ) -> float:
        """
        Estimate delta = gap between best and second-best arm in the new regime.
        Uses counterfactual rewards in the window after regime change.
        """
        if counterfactual_matrix is None:
            return 0.1  # default

        window_end = min(t_change + self.window, T)
        if window_end <= t_change:
            return 0.1

        cf_window = counterfactual_matrix[t_change:window_end, :]  # (W, K)
        arm_means = cf_window.mean(axis=0)                          # (K,)
        sorted_means = np.sort(arm_means)[::-1]
        if len(sorted_means) < 2:
            return 0.1
        delta = float(sorted_means[0] - sorted_means[1])
        return max(delta, 0.005)

    def compare_algorithms(
        self,
        reward_dict:      Dict[str, np.ndarray],   # {name: reward_series}
        change_times:     List[int],
        cf_matrix:        Optional[np.ndarray] = None,
        regime_names:     Optional[List[str]]  = None,
    ) -> Dict[str, List[AdaptationEvent]]:
        """
        Run adaptation analysis for multiple algorithms.
        Returns {algo_name: [AdaptationEvent, ...]}
        """
        results = {}
        for name, rewards in reward_dict.items():
            results[name] = self.measure_adaptation(
                rewards, change_times,
                counterfactual_matrix=cf_matrix,
            )
        return results

    def summary_table(
        self,
        algo_results: Dict[str, List[AdaptationEvent]],
    ) -> Dict:
        """
        Produce a summary comparing adaptation speeds across algorithms.
        """
        summary = {}
        for name, events in algo_results.items():
            valid = [e for e in events if e.adaptation_time is not None]
            summary[name] = {
                "mean_adaptation_time": float(np.mean(
                    [e.adaptation_time for e in valid]
                )) if valid else float("inf"),
                "median_adaptation_time": float(np.median(
                    [e.adaptation_time for e in valid]
                )) if valid else float("inf"),
                "recovery_rate": len(valid) / max(len(events), 1),
                "n_changes": len(events),
                "n_recovered": len(valid),
            }
        return summary

    def theoretical_curve(
        self,
        K_values:    np.ndarray,
        delta_values: np.ndarray,
    ) -> np.ndarray:
        """
        Theoretical adaptation time as a function of K and delta:
          t_adapt ~ C * log(K) / delta^2
        Returns array of predicted adaptation times.
        """
        return np.log(K_values) / (delta_values ** 2 + 1e-10)

    def sw_exp3_theoretical(
        self,
        window: int,
        delta:  float,
    ) -> float:
        """
        SW-Exp3 adaptation time: O(W) regardless of delta.
        Exact expression: W * (1 + 1/delta) approximately.
        """
        return float(window)

    def correlation_test(
        self,
        events: List[AdaptationEvent],
    ) -> Dict:
        """
        Test whether measured adaptation times correlate with
        theoretical predictions (O(log K / delta^2)).
        Returns Pearson correlation and p-value.
        """
        from scipy import stats

        valid = [e for e in events if e.adaptation_time is not None]
        if len(valid) < 3:
            return {"correlation": float("nan"), "p_value": float("nan"),
                    "n": len(valid)}

        measured    = np.array([e.adaptation_time   for e in valid])
        theoretical = np.array([e.theoretical_pred  for e in valid])

        r, p = stats.pearsonr(theoretical, measured)
        return {
            "correlation": float(r),
            "p_value":     float(p),
            "n":           len(valid),
            "significant": p < 0.05,
        }
