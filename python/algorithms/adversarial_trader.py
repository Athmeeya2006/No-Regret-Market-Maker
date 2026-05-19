"""Order-flow models, including a best-response informed trader for Exp3.

The adversarial trader observes the market maker's spread distribution and
switches between informed flow and small bait orders. See Auer et al. (2002)
for the adversarial bandit setting used by Exp3.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# ================================================================
# True Value Process
# ================================================================

class TrueValueProcess:
    """Fundamental value process observed by informed traders."""

    def __init__(
        self,
        initial_value: float = 100.0,
        drift:         float = 0.0,
        vol:           float = 0.02,
        jump_prob:     float = 0.01,
        jump_size:     float = 0.5,
        seed:          Optional[int] = None,
    ):
        self.V0         = initial_value
        self.V          = initial_value
        self.drift      = drift
        self.vol        = vol
        self.jump_prob  = jump_prob
        self.jump_size  = jump_size
        self.rng        = np.random.default_rng(seed)
        self.history: List[float] = [initial_value]

    def step(self, dt: float = 1.0) -> float:
        """Advance true value by one timestep."""
        dW   = self.rng.standard_normal() * np.sqrt(dt)
        jump = 0.0
        if self.rng.random() < self.jump_prob * dt:
            jump = self.rng.choice([-1, 1]) * self.jump_size
        self.V += self.drift * dt + self.vol * dW + jump
        self.V  = max(self.V, 0.01)   # stay positive
        self.history.append(self.V)
        return self.V

    def value(self) -> float:
        return self.V

    def reset(self):
        self.V = self.V0
        self.history = [self.V0]


# ================================================================
# Noise Trader
# ================================================================

class NoiseTrader:
    """
    Uninformed trader whose orders arrive as a Poisson process.
    Order sizes follow a power-law distribution.
    These traders do not respond to prices.
    """

    TRADER_ID = 1

    def __init__(
        self,
        arrival_rate:  float = 5.0,
        alpha:         float = 1.5,    # power-law exponent
        size_min:      int   = 1,
        size_max:      int   = 50,
        seed:          Optional[int] = None,
    ):
        self.rate     = float(arrival_rate)
        self.alpha    = float(alpha)
        self.size_min = int(size_min)
        self.size_max = int(size_max)
        self.rng      = np.random.default_rng(seed)

    def _sample_size(self) -> int:
        """
        Power-law sample via inverse-CDF method.
        P(X >= x) ~ x^{-alpha}, x >= x_min
        CDF: F(x) = 1 - (x_min/x)^alpha
        Inverse: x = x_min * (1-u)^{-1/alpha}
        """
        u = self.rng.random()
        x = self.size_min * (1.0 - u) ** (-1.0 / self.alpha)
        return int(np.clip(round(x), self.size_min, self.size_max))

    def maybe_arrive(self, dt: float = 1.0) -> List[Dict]:
        """
        Returns a list of market orders for this timestep.
        n ~ Poisson(rate * dt)
        """
        n = self.rng.poisson(self.rate * dt)
        orders = []
        for _ in range(n):
            side = "BUY" if self.rng.random() < 0.5 else "SELL"
            size = self._sample_size()
            orders.append({
                "side": side, "qty": size, "type": "MARKET",
                "trader_id": self.TRADER_ID,
            })
        return orders


# ================================================================
# Stochastic Informed Trader
# ================================================================

class StochasticInformedTrader:
    """
    Informed trader with private value signal.
    Trades when the price offers an edge relative to true value.
    """

    TRADER_ID = 2

    def __init__(
        self,
        true_value_process: TrueValueProcess,
        arrival_rate_frac:  float = 0.2,   # fraction of noise rate
        base_arrival_rate:  float = 5.0,
        size_scale:         int   = 10,
        seed:               Optional[int] = None,
    ):
        self.V_proc   = true_value_process
        self.rate     = arrival_rate_frac * base_arrival_rate
        self.size_scale = int(size_scale)
        self.rng      = np.random.default_rng(seed)

    def maybe_arrive(
        self,
        dt:      float,
        mid:     float,
        spread:  float,
    ) -> List[Dict]:
        n = self.rng.poisson(self.rate * dt)
        orders = []
        V = self.V_proc.value()
        ask = mid + spread / 2
        bid = mid - spread / 2

        for _ in range(n):
            if V > ask:
                edge = V - ask
                size = max(1, int(edge / mid * self.size_scale * 100))
                size = min(size, self.size_scale * 5)
                orders.append({
                    "side": "BUY", "qty": size, "type": "MARKET",
                    "trader_id": self.TRADER_ID,
                })
            elif V < bid:
                edge = bid - V
                size = max(1, int(edge / mid * self.size_scale * 100))
                size = min(size, self.size_scale * 5)
                orders.append({
                    "side": "SELL", "qty": size, "type": "MARKET",
                    "trader_id": self.TRADER_ID,
                })
        return orders


# ================================================================
# Adversarial Informed Trader
# ================================================================

class AdversarialInformedTrader:
    """Best-response order-flow adversary for spread-selection experiments."""

    TRADER_ID = 3

    def __init__(
        self,
        true_value_process:  TrueValueProcess,
        mm_spread_choices:   List[float],
        aggression:          float = 1.0,
        tight_threshold:     float = 0.5,   # mass below median spread => "tight"
        seed:                Optional[int] = None,
    ):
        self.V_proc           = true_value_process
        self.spreads          = np.array(mm_spread_choices)
        self.K                = len(mm_spread_choices)
        self.median_spread_idx = self.K // 2
        self.aggression       = float(aggression)
        self.tight_threshold  = tight_threshold
        self.rng              = np.random.default_rng(seed)

        self.mm_strategy_history: List[np.ndarray] = []
        self._phase: str = "observe"   # "observe" | "exploit" | "bait"

    def observe_mm_strategy(self, spread_weights: np.ndarray):
        """Record MM's current weight distribution (called before trading)."""
        self.mm_strategy_history.append(spread_weights.copy())

    def _is_tight(self, weights: np.ndarray) -> bool:
        """Returns True if MM currently favours tight spreads."""
        tight_mass = weights[:self.median_spread_idx].sum()
        return bool(tight_mass > self.tight_threshold)

    def compute_best_response(
        self,
        current_mm_weights: np.ndarray,
        mid:                float,
        spread:             float,
    ) -> str:
        """
        Returns "exploit" (informed trade) or "bait" (noise trade).

        Exploit: MM quotes tight => send large informed order
        Bait:    MM quotes wide  => send small noise to encourage tightening
        """
        if self._is_tight(current_mm_weights):
            return "exploit"
        else:
            return "bait"

    def maybe_arrive(
        self,
        dt:                 float,
        mid:                float,
        spread:             float,
        current_mm_weights: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Generate adversarial orders for this timestep."""
        if current_mm_weights is None:
            current_mm_weights = np.ones(self.K) / self.K

        phase  = self.compute_best_response(current_mm_weights, mid, spread)
        V      = self.V_proc.value()
        orders = []

        if phase == "exploit":
            n = self.rng.poisson(max(1, 3.0 * dt))  # higher arrival rate
            for _ in range(n):
                ask = mid + spread / 2
                bid = mid - spread / 2
                if V > ask:
                    edge = max(0.0, V - ask)
                    base_size = max(1, int(edge / mid * 500))
                    size = int(base_size * self.aggression)
                    orders.append({
                        "side": "BUY", "qty": size, "type": "MARKET",
                        "trader_id": self.TRADER_ID, "informed": True,
                    })
                elif V < bid:
                    edge = max(0.0, bid - V)
                    base_size = max(1, int(edge / mid * 500))
                    size = int(base_size * self.aggression)
                    orders.append({
                        "side": "SELL", "qty": size, "type": "MARKET",
                        "trader_id": self.TRADER_ID, "informed": True,
                    })
        else:
            # Bait: small balanced noise to encourage MM to tighten
            n = self.rng.poisson(0.5 * dt)
            for _ in range(n):
                side = "BUY" if self.rng.random() < 0.5 else "SELL"
                orders.append({
                    "side": side, "qty": 1, "type": "MARKET",
                    "trader_id": self.TRADER_ID, "informed": False,
                })

        return orders

    # ---- Regret analysis helpers --------------------------------

    def compute_mm_regret_impact(
        self,
        chosen_spread:   float,
        fills_this_round: List[Dict],
        mid_move_after:  float,
    ) -> Dict[str, float]:
        """
        Compute how much this adversary contributed to the MM's regret.

        adverse_selection_cost: fill_qty * |mid_move_after| (adversary trades
            in the direction of the subsequent mid-price move)
        spread_revenue: fill_qty * chosen_spread / 2
        net_impact: adverse_selection_cost - spread_revenue
        """
        my_fills = [f for f in fills_this_round
                    if f.get("trader_id") == self.TRADER_ID]
        total_qty = sum(f.get("qty", 0) for f in my_fills)

        adverse_sel   = total_qty * abs(mid_move_after)
        spread_rev    = total_qty * chosen_spread / 2.0
        return {
            "adverse_selection_cost": adverse_sel,
            "spread_revenue":         spread_rev,
            "net_impact":             adverse_sel - spread_rev,
            "total_qty":              total_qty,
        }
