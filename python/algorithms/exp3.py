"""Exp3 variants for adversarial spread selection.

Includes known-horizon Exp3, a doubling-trick wrapper, and sliding-window
Exp3 for non-stationary rewards. The update follows Auer et al. (2002).
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# ================================================================
# Base: Exp3
# ================================================================

class Exp3MarketMaker:
    """Adversarial bandit market maker with importance-weighted updates (Auer et al., 2002)."""

    def __init__(
        self,
        spread_choices: List[float],
        gamma: Optional[float] = None,
        T: Optional[int] = None,
    ):
        self.K      = len(spread_choices)
        self.spreads = np.array(spread_choices, dtype=float)

        # Auer et al. (2002) learning rate for a known horizon.
        if gamma is not None:
            self.gamma = float(gamma)
        elif T is not None:
            self.gamma = float(np.sqrt(np.log(self.K) / (self.K * T)))
        else:
            self.gamma = 0.1

        # Log-space weights: log_w[k] = 0 initially (=> w[k] = 1)
        self.log_w = np.zeros(self.K)

        # Tracking
        self.last_idx: Optional[int]   = None
        self.last_p:   Optional[float] = None
        self.t: int = 0

        self.reward_history:  List[float] = []
        self.spread_history:  List[int]   = []   # arm indices
        self.dist_history:    List[np.ndarray] = []

    # ---- Core interface ----------------------------------------

    def get_distribution(self) -> np.ndarray:
        """Mixed strategy: (1-gamma) * softmax(log_w) + gamma/K."""
        log_w_shifted = self.log_w - self.log_w.max()
        w = np.exp(log_w_shifted)
        p = (1.0 - self.gamma) * w / w.sum() + self.gamma / self.K
        # Clip tiny probabilities before renormalizing.
        p = np.clip(p, 1e-15, 1.0)
        p /= p.sum()
        return p

    def choose_spread(self, context: Optional[dict] = None) -> float:
        p = self.get_distribution()
        idx = int(np.random.choice(self.K, p=p))
        self.last_idx = idx
        self.last_p   = float(p[idx])
        self.dist_history.append(p.copy())
        return float(self.spreads[idx])

    def record_choice(self, idx: int, p: np.ndarray) -> None:
        """Record a manually-chosen arm and its probability.
        
        Use this when sampling from the distribution externally
        (e.g., in analysis code). Ensures dist_history is updated.
        """
        self.last_idx = int(idx)
        self.last_p   = float(p[idx])
        self.dist_history.append(p.copy())

    def update(self, reward: float,
               counterfactual_rewards: Optional[Dict[float, float]] = None):
        """
        Parameters
        ----------
        reward : actual reward observed this round
        counterfactual_rewards : dict {spread_value: reward_if_chosen},
            used only for regret tracking (not for weight update)
        """
        if self.last_idx is None:
            return

        p = self.get_distribution()
        # Importance-weighted unbiased estimator for chosen arm only
        x_hat_chosen = reward / p[self.last_idx]
        self.log_w[self.last_idx] += self.gamma * x_hat_chosen / self.K

        self.reward_history.append(reward)
        self.spread_history.append(self.last_idx)
        self.t += 1

    # ---- Regret computation ------------------------------------

    def theoretical_regret_bound(self, T: Optional[int] = None) -> float:
        T = T or self.t
        return 2.0 * np.sqrt(T * self.K * np.log(self.K))

    def empirical_regret(
        self,
        counterfactual_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute regret curve from counterfactual matrix.

        Parameters
        ----------
        counterfactual_matrix : shape (T, K)
            counterfactual_matrix[t, k] = reward if arm k had been chosen at round t

        Returns
        -------
        regret_curve : shape (T,)
            cumulative regret at each timestep
        """
        T = counterfactual_matrix.shape[0]
        actual = np.array(self.reward_history[:T])
        best_fixed = counterfactual_matrix.cumsum(axis=0).max(axis=1)
        actual_cum = np.cumsum(actual)
        return best_fixed - actual_cum

    def reset(self):
        self.log_w       = np.zeros(self.K)
        self.last_idx    = None
        self.last_p      = None
        self.t           = 0
        self.reward_history.clear()
        self.spread_history.clear()
        self.dist_history.clear()

    def __repr__(self) -> str:
        return (f"Exp3MarketMaker(K={self.K}, gamma={self.gamma:.4f}, "
                f"t={self.t})")


# ================================================================
# Exp3 with Doubling Trick (unknown horizon)
# ================================================================

class Exp3DoublingTrick:
    """Exp3 for unknown horizons using geometric restarts."""

    def __init__(self, spread_choices: List[float]):
        self.spread_choices = spread_choices
        self.K = len(spread_choices)

        self._epoch_len  = 1
        self._epoch_t    = 0
        self._global_t   = 0
        self._current    = Exp3MarketMaker(spread_choices, T=1)

        # Aggregate history across epochs
        self.reward_history:  List[float] = []
        self.spread_history:  List[int]   = []
        self.dist_history:    List[np.ndarray] = []

    def _maybe_restart(self):
        if self._epoch_t >= self._epoch_len:
            self._epoch_len *= 2
            self._epoch_t = 0
            self._current = Exp3MarketMaker(
                list(self.spread_choices), T=self._epoch_len
            )
            logger.debug(f"Exp3DoublingTrick: new epoch len={self._epoch_len}, "
                         f"gamma={self._current.gamma:.5f}")
            return True
        return False

    def get_distribution(self) -> np.ndarray:
        return self._current.get_distribution()

    def choose_spread(self, context=None) -> float:
        self._epoch_t += 1
        self._maybe_restart()
        s = self._current.choose_spread(context)
        self.dist_history.append(self._current.dist_history[-1].copy())
        return s

    def record_choice(self, idx: int, p: np.ndarray) -> None:
        self._epoch_t += 1
        self._current.record_choice(idx, p)
        self.dist_history.append(self._current.dist_history[-1].copy())
        if self._epoch_t >= self._epoch_len:
            self._maybe_restart()

    def update(self, reward: float, counterfactual_rewards=None):
        self._current.update(reward, counterfactual_rewards)
        self._global_t += 1
        self.reward_history.append(reward)
        if self._current.spread_history:
            self.spread_history.append(self._current.spread_history[-1])

    def theoretical_regret_bound(self, T=None) -> float:
        T = T or self._global_t
        return 4.0 * np.sqrt(T * self.K * np.log(self.K))

    def reset(self):
        self._epoch_len = 1
        self._epoch_t   = 0
        self._global_t  = 0
        self._current   = Exp3MarketMaker(list(self.spread_choices), T=1)
        self.reward_history.clear()
        self.spread_history.clear()
        self.dist_history.clear()

    @property
    def spreads(self): return self._current.spreads
    @property
    def K(self): return len(self.spread_choices)
    @K.setter
    def K(self, v): pass   # kept for symmetry


# ================================================================
# SW-Exp3: Sliding Window Exp3 (non-stationary extension)
# ================================================================

class SWExp3MarketMaker:
    """Sliding-window variant of Exp3 for non-stationary reward sequences."""

    def __init__(
        self,
        spread_choices: List[float],
        window: int = 200,
        gamma: Optional[float] = None,
        T: Optional[int] = None,
        upsilon: Optional[int] = None,
    ):
        self.K       = len(spread_choices)
        self.spreads = np.array(spread_choices, dtype=float)
        self.W       = int(window)

        if gamma is not None:
            self.gamma = float(gamma)
        elif T is not None and upsilon is not None:
            # Optimal gamma for known T and Upsilon
            self.gamma = np.sqrt(
                np.log(self.K) / (self.K * T / max(upsilon, 1))
            )
        else:
            self.gamma = 0.15

        # Circular buffer stores (arm_idx, reward_estimate) for last W rounds
        # We recompute log_w from scratch each round using the window
        self._buf_arms: List[int]   = []
        self._buf_xhat: List[float] = []  # importance-weighted reward

        self.last_idx: Optional[int]   = None
        self.last_p:   Optional[float] = None
        self.t: int = 0

        self.reward_history: List[float] = []
        self.spread_history: List[int]   = []
        self.dist_history:   List[np.ndarray] = []

    def _compute_log_w(self) -> np.ndarray:
        """Recompute weights from the sliding window buffer."""
        log_w = np.zeros(self.K)
        window_start = max(0, len(self._buf_arms) - self.W)
        for i in range(window_start, len(self._buf_arms)):
            k    = self._buf_arms[i]
            xhat = self._buf_xhat[i]
            log_w[k] += self.gamma * xhat / self.K
        return log_w

    def get_distribution(self) -> np.ndarray:
        log_w = self._compute_log_w()
        log_w -= log_w.max()
        w = np.exp(log_w)
        p = (1.0 - self.gamma) * w / w.sum() + self.gamma / self.K
        p = np.clip(p, 1e-15, 1.0)
        p /= p.sum()
        return p

    def choose_spread(self, context=None) -> float:
        p = self.get_distribution()
        idx = int(np.random.choice(self.K, p=p))
        self.last_idx = idx
        self.last_p   = float(p[idx])
        self.dist_history.append(p.copy())
        return float(self.spreads[idx])

    def record_choice(self, idx: int, p: np.ndarray) -> None:
        self.last_idx = int(idx)
        self.last_p   = float(p[idx])
        self.dist_history.append(p.copy())

    def update(self, reward: float, counterfactual_rewards=None):
        if self.last_idx is None:
            return
        p    = self.get_distribution()
        xhat = reward / p[self.last_idx]

        self._buf_arms.append(self.last_idx)
        self._buf_xhat.append(xhat)

        # Trim buffer beyond window
        if len(self._buf_arms) > self.W * 2:
            self._buf_arms = self._buf_arms[-self.W:]
            self._buf_xhat = self._buf_xhat[-self.W:]

        self.reward_history.append(reward)
        self.spread_history.append(self.last_idx)
        self.t += 1

    def theoretical_regret_bound(self, T=None, upsilon=1) -> float:
        T = T or self.t
        return 2.0 * np.sqrt(upsilon * T * self.K * np.log(self.K))

    def reset(self):
        self._buf_arms.clear()
        self._buf_xhat.clear()
        self.last_idx = None
        self.last_p   = None
        self.t        = 0
        self.reward_history.clear()
        self.spread_history.clear()
        self.dist_history.clear()

    def __repr__(self) -> str:
        return (f"SWExp3MarketMaker(K={self.K}, W={self.W}, "
                f"gamma={self.gamma:.4f}, t={self.t})")
