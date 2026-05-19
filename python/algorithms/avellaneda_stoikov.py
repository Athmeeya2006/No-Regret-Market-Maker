"""Avellaneda-Stoikov market maker.

Implements the closed-form quotes from Avellaneda and Stoikov (2008) and
basic helpers for volatility and arrival-rate calibration.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


class AvellanedaStoikovMM:
    """
    Model-based optimal market maker using the A-S closed-form solution.

    This is the benchmark against which Exp3 / EXP4 are compared.
    It performs well when the underlying model is correctly specified,
    and degrades gracefully (but measurably) when parameters shift.
    """

    def __init__(
        self,
        sigma:  float = 0.01,
        kappa:  float = 1.5,
        gamma:  float = 0.1,
        T:      float = 1.0,
        dt:     float = 0.001,
        lambda_: float = 1.0,   # Reference arrival rate at zero spread
    ):
        """
        Parameters
        ----------
        sigma   : Volatility of mid-price (per unit time)
        kappa   : Order-arrival decay (how fast rate drops with spread)
        gamma   : Coefficient of risk aversion
        T       : Total trading horizon (normalised)
        dt      : Timestep size
        lambda_ : Poisson rate at zero spread (not used in deterministic quotes
                  but needed for arrival simulation)
        """
        self.sigma   = float(sigma)
        self.kappa   = float(kappa)
        self.gamma   = float(gamma)
        self.T       = float(T)
        self.dt      = float(dt)
        self.lambda_ = float(lambda_)

        self.t:         float = 0.0   # Current time (updated each round)
        self.inventory: int   = 0
        self.cash:      float = 0.0

        # History for diagnostics
        self.quote_history:  list = []
        self.reward_history: list = []

    # ---- Core quoting ------------------------------------------

    def reservation_price(self, mid: float, inventory: int,
                           time_remaining: float) -> float:
        """
        r(t, q) = S - q * gamma * sigma^2 * (T - t)

        Intuition: if the MM is long (q > 0), the reservation price is
        below mid because the MM wants to sell to reduce inventory.
        """
        return mid - inventory * self.gamma * self.sigma**2 * time_remaining

    def optimal_spread(self, time_remaining: float) -> float:
        """
        s* = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)
        """
        inventory_risk   = self.gamma * self.sigma**2 * time_remaining
        adverse_selection = (2.0 / self.gamma) * np.log(
            1.0 + self.gamma / self.kappa
        )
        return max(1e-4, inventory_risk + adverse_selection)

    def quote(
        self,
        mid: float,
        inventory: int,
        time_remaining: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Returns (bid, ask) optimal quotes.

        Parameters
        ----------
        mid            : Current mid price
        inventory      : Current inventory (positive = long)
        time_remaining : Time left to horizon; if None uses self.T - self.t
        """
        if time_remaining is None:
            time_remaining = max(self.T - self.t, 1e-4)

        r  = self.reservation_price(mid, inventory, time_remaining)
        s  = self.optimal_spread(time_remaining)
        bid = r - s / 2.0
        ask = r + s / 2.0
        return bid, ask

    def choose_spread(self, context: dict) -> float:
        """
        Derive the spread from context dict.
        Returns the full bid-ask spread s*.

        Also stores quote for later introspection.
        """
        mid           = context.get("mid_price", 100.0)
        inventory     = context.get("inventory", 0)
        time_remaining = context.get("time_remaining", 0.5)

        bid, ask = self.quote(mid, inventory, time_remaining)
        spread = ask - bid

        self.quote_history.append({
            "mid": mid, "bid": bid, "ask": ask,
            "spread": spread, "inventory": inventory,
        })
        return spread

    def update(self, reward: float, counterfactual_rewards=None):
        """A-S is model-based; no learning update. Just record reward."""
        self.reward_history.append(reward)
        self.t += self.dt

    def reset(self):
        self.t         = 0.0
        self.inventory = 0
        self.cash      = 0.0
        self.quote_history.clear()
        self.reward_history.clear()

    def get_distribution(self) -> np.ndarray:
        """
        A-S is deterministic: return a point mass on the computed spread.
        Spreads here must be discretised to match Exp3's arm set.
        """
        raise NotImplementedError(
            "A-S is deterministic; it does not maintain a distribution "
            "over spread arms. Use choose_spread() directly."
        )

    @staticmethod
    def estimate_sigma(mid_prices: np.ndarray, dt: float = 1.0) -> float:
        """
        MLE estimate of volatility from mid-price series.
        sigma^2 = Var[log(S_{t+1}/S_t)] / dt
        """
        log_ret = np.diff(np.log(mid_prices))
        return float(np.std(log_ret) / np.sqrt(dt))

    @staticmethod
    def estimate_kappa(
        spreads: np.ndarray,
        arrival_counts: np.ndarray,
    ) -> float:
        """
        MLE estimate of kappa from empirical arrival data.
        Fits: log(lambda(delta)) = log(lambda_0) - kappa * delta
        via OLS on log-counts vs. spreads.
        """
        log_counts = np.log(np.maximum(arrival_counts, 1e-8))
        # OLS: slope = -kappa
        A = np.vstack([spreads, np.ones_like(spreads)]).T
        slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
        return max(0.1, -slope)

    def arrival_rate(self, half_spread: float) -> float:
        """lambda(delta) = lambda_0 * exp(-kappa * delta)"""
        return self.lambda_ * np.exp(-self.kappa * half_spread)

    def expected_trades_per_dt(self, half_spread: float,
                                dt: float = 1.0) -> float:
        """Expected number of trades against one side per unit time."""
        return self.arrival_rate(half_spread) * dt

    def optimal_inventory_distribution(
        self,
        n_rounds: int,
        spread: Optional[float] = None,
    ) -> dict:
        """
        Under A-S, approximate the stationary inventory distribution.
        Buy fills and sell fills balance at the optimal spread.
        Returns mean and std of inventory under the model.
        """
        if spread is None:
            spread = self.optimal_spread(self.T / 2)

        rate = self.arrival_rate(spread / 2)
        # Both sides arrive at same rate => inventory is a symmetric random walk
        # Std of RW after T steps: sqrt(2 * rate * T)
        std_inv = np.sqrt(2.0 * rate * n_rounds)
        return {"mean": 0.0, "std": std_inv, "arrival_rate": rate}

    def __repr__(self) -> str:
        s = self.optimal_spread(max(self.T - self.t, 1e-4))
        return (f"AvellanedaStoikovMM(sigma={self.sigma:.4f}, "
                f"kappa={self.kappa:.2f}, gamma={self.gamma:.2f}, "
                f"optimal_spread={s:.5f})")
