"""Simulation loop for market makers, order flow, regimes, and rewards."""

from __future__ import annotations

import numpy as np
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import C++ engine; fall back to pure-Python stub for unit tests
try:
    import lob_engine as _lob
    _HAS_CPP = True
except ImportError:
    logger.warning("lob_engine not found - using Python stub (slow).")
    _HAS_CPP = False
    from .python_lob_stub import PythonLOBStub as _lob_stub


@dataclass
class RoundResult:
    """Everything recorded in a single simulation round."""
    t:               int
    mid_before:      float
    mid_after:       float
    spread_quoted:   float
    reward:          float
    spread_revenue:  float
    inventory_pnl:   float
    inventory_penalty: float
    inventory:       int
    n_trades:        int
    n_informed:      int
    was_adversarial: bool
    regime_name:     str
    counterfactual_rewards: Dict[float, float] = field(default_factory=dict)
    mm_distribution: Optional[np.ndarray] = None


class MarketSimulator:
    """
    Full market simulation.

    Parameters
    ----------
    engine          : MatchingEngine (C++ or stub)
    noise_trader    : NoiseTrader
    informed_trader : StochasticInformedTrader or AdversarialInformedTrader
    market_maker    : Any MM with choose_spread(context) / update(reward) API
    spread_choices  : List of possible spreads (for counterfactual eval)
    regime_gen      : Optional RegimeGenerator
    inventory_penalty : Quadratic inventory penalty coefficient
    seed            : RNG seed
    """

    def __init__(
        self,
        engine,
        noise_trader,
        informed_trader,
        market_maker,
        spread_choices:       List[float],
        regime_gen=None,
        inventory_penalty:    float = 0.001,
        mid_price_vol:        float = 0.001,
        tick_size:            float = 0.01,
        seed:                 Optional[int] = 42,
    ):
        self.engine            = engine
        self.noise_trader      = noise_trader
        self.informed_trader   = informed_trader
        self.mm                = market_maker
        self.spread_choices    = spread_choices
        self.regime_gen        = regime_gen
        self.inventory_penalty = inventory_penalty
        self.mid_price_vol     = mid_price_vol
        self.tick_size         = tick_size
        self.rng               = np.random.default_rng(seed)

        # State
        self.inventory:   int   = 0
        self.cash:        float = 0.0
        self.t:           int   = 0
        engine_mid = engine.mid_price() if _HAS_CPP else 100.0
        self._mid:        float = engine_mid if np.isfinite(engine_mid) and engine_mid > 0 else 100.0

        # Records
        self.results:     List[RoundResult] = []
        self._bid_id:     Optional[int]     = None
        self._ask_id:     Optional[int]     = None

    # ================================================================
    # Main simulation loop
    # ================================================================

    def run(self, n_rounds: int, dt: float = 1.0) -> List[RoundResult]:
        """Run n_rounds of the market simulation."""
        self.results = []

        for t in range(n_rounds):
            self.t = t
            result = self._run_one_round(dt)
            self.results.append(result)

            if t % 1000 == 0:
                logger.info(
                    f"t={t:5d} | mid={self._mid:.3f} | "
                    f"inv={self.inventory:+4d} | "
                    f"cum_reward={sum(r.reward for r in self.results):.2f} | "
                    f"regime={result.regime_name}"
                )

        return self.results

    def _run_one_round(self, dt: float) -> RoundResult:
        # 1. Update regime parameters if applicable
        regime_name = "default"
        if self.regime_gen is not None:
            params = self.regime_gen.get_params(self.t)
            regime_name = params.get("name", "unknown")
            self._apply_regime(params)

        mid_before = self._mid

        # 2. Build context for the MM
        context = self._build_context(dt)

        # 3. MM chooses spread; record its distribution if available
        mm_distribution = None
        if hasattr(self.mm, "get_distribution"):
            try:
                mm_distribution = self.mm.get_distribution().copy()
            except NotImplementedError:
                pass

        spread = self.mm.choose_spread(context)
        spread = max(self.tick_size, round(spread / self.tick_size) * self.tick_size)

        bid_price = self._mid - spread / 2
        ask_price = self._mid + spread / 2

        # 4. Submit MM quotes to LOB
        self._cancel_stale_mm_quotes()
        bid_id = self._submit_mm_quote("BID", bid_price)
        ask_id = self._submit_mm_quote("ASK", ask_price)
        self._bid_id = bid_id
        self._ask_id = ask_id

        # 5. Noise traders arrive
        n_noise_trades = 0
        noise_orders = self.noise_trader.maybe_arrive(dt)
        for order_dict in noise_orders:
            fills = self._submit_order(order_dict)
            n_noise_trades += len(fills)
            self._process_fills(fills, order_dict)

        # 6. Informed / adversarial traders arrive
        n_informed_trades = 0
        was_adversarial   = False
        mm_weights = mm_distribution if mm_distribution is not None \
                     else np.ones(len(self.spread_choices)) / len(self.spread_choices)

        if hasattr(self.informed_trader, "observe_mm_strategy"):
            self.informed_trader.observe_mm_strategy(mm_weights)
            informed_orders = self.informed_trader.maybe_arrive(
                dt, self._mid, spread, current_mm_weights=mm_weights
            )
            was_adversarial = True
        else:
            informed_orders = self.informed_trader.maybe_arrive(
                dt, self._mid, spread
            )

        for order_dict in informed_orders:
            fills = self._submit_order(order_dict)
            n_informed_trades += len(fills)
            self._process_fills(fills, order_dict)

        # 7. Advance true value and mid price (GBM + mean reversion to V)
        self._advance_mid(dt)

        mid_after = self._mid

        # 8. Cancel remaining MM quotes (fresh quotes each round)
        self._cancel_stale_mm_quotes()

        # 9. Compute reward components
        spread_revenue    = self._spread_revenue_this_round
        inventory_pnl     = self.inventory * (mid_after - mid_before)
        inventory_penalty = self.inventory_penalty * self.inventory ** 2
        reward            = spread_revenue + inventory_pnl - inventory_penalty

        # 10. Compute counterfactual rewards (replay each spread)
        counterfactual_rewards = self._compute_counterfactuals(
            mid_before, mid_after, noise_orders, informed_orders, dt
        )

        # 11. Update MM
        self.mm.update(reward, counterfactual_rewards)

        # 12. Reset per-round accumulators
        self._spread_revenue_this_round = 0.0

        return RoundResult(
            t=self.t,
            mid_before=mid_before,
            mid_after=mid_after,
            spread_quoted=spread,
            reward=reward,
            spread_revenue=spread_revenue,
            inventory_pnl=inventory_pnl,
            inventory_penalty=inventory_penalty,
            inventory=self.inventory,
            n_trades=n_noise_trades + n_informed_trades,
            n_informed=n_informed_trades,
            was_adversarial=was_adversarial,
            regime_name=regime_name,
            counterfactual_rewards=counterfactual_rewards,
            mm_distribution=mm_distribution,
        )

    # ================================================================
    # Context builder
    # ================================================================

    def _build_context(self, dt: float) -> dict:
        T = getattr(self.mm, "T", 1.0) if hasattr(self.mm, "T") else 1.0
        time_remaining = max(T - self.t * dt / 10000, 0.01)

        mid_hist = []
        if _HAS_CPP and hasattr(self.engine, "mid_price_history"):
            mid_hist = self.engine.mid_price_history()[-20:]

        momentum = 0.0
        if len(mid_hist) >= 2:
            momentum = (mid_hist[-1] - mid_hist[0]) / (mid_hist[0] + 1e-8)

        return {
            "mid_price":      self._mid,
            "spread":         self.engine.spread() if _HAS_CPP else 0.02,
            "ofi":            self.engine.ofi() if _HAS_CPP else 0.0,
            "inventory":      self.inventory,
            "realized_vol":   self.engine.realized_volatility(20) if _HAS_CPP else self.mid_price_vol,
            "time_remaining": time_remaining,
            "bid_volume":     self.engine.total_bid_volume(5) if _HAS_CPP else 0,
            "ask_volume":     self.engine.total_ask_volume(5) if _HAS_CPP else 0,
            "momentum":       momentum,
            "t":              self.t,
        }

    # ================================================================
    # LOB interaction helpers
    # ================================================================

    _spread_revenue_this_round: float = 0.0

    def _submit_mm_quote(self, side: str, price: float) -> Optional[int]:
        price = max(0.01, price)
        if not _HAS_CPP:
            return None
        s = _lob.Side.BID if side == "BID" else _lob.Side.ASK
        fills = self.engine.submit_limit(s, price, qty=100, trader_id=0)
        # MM quotes that immediately fill => this shouldn't happen
        # (mid +/- spread/2 should not cross existing resting orders
        #  from the same MM unless spread is very tight)
        if fills:
            for f in fills:
                qty = f.quantity
                if side == "BID":
                    self.inventory       += qty
                    self.cash            -= f.price * qty
                    self._spread_revenue_this_round += spread_earned_from_fill(f, "BID")
                else:
                    self.inventory       -= qty
                    self.cash            += f.price * qty
                    self._spread_revenue_this_round += spread_earned_from_fill(f, "ASK")
        return int(self.engine.last_order_id())

    def _cancel_stale_mm_quotes(self):
        """Cancel the current market-maker quotes if they are still resting."""
        if not _HAS_CPP:
            return
        for oid in (self._bid_id, self._ask_id):
            if oid is not None:
                self.engine.cancel(int(oid))
        self._bid_id = None
        self._ask_id = None

    def _submit_order(self, order_dict: dict) -> list:
        if not _HAS_CPP:
            return []
        side = _lob.Side.BID if order_dict["side"] == "BUY" else _lob.Side.ASK
        tid  = order_dict.get("trader_id", -1)
        qty  = order_dict.get("qty", 1)
        return self.engine.submit_market(side, qty, tid)

    def _process_fills(self, fills: list, order_dict: dict):
        """Update inventory and cash from fills against MM quotes."""
        for f in fills:
            if not _HAS_CPP:
                continue
            qty = f.quantity
            # MM is trader_id=0 on one side of every fill
            if f.buy_trader_id == 0:
                # MM bought: inventory goes up, cash goes down
                self.inventory       += qty
                self.cash            -= f.price * qty
                self._spread_revenue_this_round += (self._mid - f.price) * qty
            elif f.sell_trader_id == 0:
                # MM sold: inventory goes down, cash goes up
                self.inventory       -= qty
                self.cash            += f.price * qty
                self._spread_revenue_this_round += (f.price - self._mid) * qty

    # ================================================================
    # Counterfactual reward computation
    # ================================================================

    def _compute_counterfactuals(
        self,
        mid_before: float,
        mid_after:  float,
        noise_orders:    List[dict],
        informed_orders: List[dict],
        dt: float,
    ) -> Dict[float, float]:
        """
        For each hypothetical spread s in spread_choices, compute what
        reward the MM would have earned if it had quoted that spread.

        Simplified model (avoids re-running LOB for each arm):
          spread_revenue(s) = n_fills(s) * s / 2
          n_fills(s): Poisson with rate proportional to exp(-kappa*s/2)
            normalised to expected actual fills at chosen spread.

        inventory_pnl and inventory_penalty are computed identically
        because those depend on inventory accumulated, which depends
        on fill counts.
        """
        all_orders = noise_orders + informed_orders
        total_qty  = sum(o.get("qty", 1) for o in all_orders)
        mid_move   = mid_after - mid_before

        counterfactuals: Dict[float, float] = {}
        for s in self.spread_choices:
            # Expected fill rate: higher for tighter spreads
            # Use exponential model: fill_rate(s) ~ exp(-kappa * s)
            # kappa = 1.5 (default); normalise to total_qty at s=0
            kappa = 1.5
            fill_frac     = np.exp(-kappa * s)
            fill_frac_ref = np.exp(-kappa * min(self.spread_choices))
            expected_fills = max(0, total_qty * fill_frac / (fill_frac_ref + 1e-8))
            expected_fills = min(expected_fills, total_qty)

            # Spread revenue
            spread_rev = expected_fills * s / 2.0

            # Inventory pnl: assume same direction of trades
            # If all noise (random dir) => inventory ~ 0 net
            # Expected inventory pnl is near zero for noise traders
            inv_pnl = self.inventory * mid_move * (expected_fills / (total_qty + 1e-8))

            # Inventory penalty: same penalty (conservative)
            inv_pen = self.inventory_penalty * self.inventory ** 2

            counterfactuals[s] = spread_rev + inv_pnl - inv_pen

        return counterfactuals

    # ================================================================
    # Mid-price dynamics
    # ================================================================

    def _advance_mid(self, dt: float):
        """Advance the mid price by one GBM step."""
        dW = self.rng.standard_normal() * np.sqrt(dt)
        self._mid += self.mid_price_vol * self._mid * dW
        self._mid  = max(0.01, self._mid)

    def _apply_regime(self, params: dict):
        """Apply regime parameters to traders."""
        if hasattr(self.noise_trader, "rate"):
            self.noise_trader.rate = params.get("lambda", self.noise_trader.rate)
        self.mid_price_vol = params.get("sigma", self.mid_price_vol)
        if hasattr(self.informed_trader, "rate"):
            base  = params.get("lambda", 5.0)
            mu    = params.get("mu", 0.2)
            self.informed_trader.rate = mu * base

    # ================================================================
    # Result accessors
    # ================================================================

    def reward_series(self) -> np.ndarray:
        return np.array([r.reward for r in self.results])

    def spread_series(self) -> np.ndarray:
        return np.array([r.spread_quoted for r in self.results])

    def inventory_series(self) -> np.ndarray:
        return np.array([r.inventory for r in self.results])

    def counterfactual_matrix(self) -> np.ndarray:
        """Returns (T, K) array of counterfactual rewards."""
        T = len(self.results)
        K = len(self.spread_choices)
        mat = np.zeros((T, K))
        for t, r in enumerate(self.results):
            for k, s in enumerate(self.spread_choices):
                mat[t, k] = r.counterfactual_rewards.get(s, 0.0)
        return mat


def spread_earned_from_fill(fill, mm_side: str) -> float:
    """Compute spread revenue for the MM from a single fill."""
    return 0.0
