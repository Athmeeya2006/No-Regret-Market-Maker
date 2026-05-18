"""EXP4 market maker with contextual expert advice.

Each expert maps a market context to a spread; EXP4 samples experts and
updates their weights with an importance-weighted reward estimate.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# ================================================================
# Expert base class and concrete implementations
# ================================================================

class Expert:
    """Abstract expert: maps context dict -> spread (float)."""
    def quote(self, context: dict) -> float:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class FixedSpreadExpert(Expert):
    """Always quotes a fixed spread. Serves as anchoring baselines."""
    def __init__(self, spread: float):
        self.spread = spread

    def quote(self, context: dict) -> float:
        return self.spread

    @property
    def name(self) -> str:
        return f"Fixed({self.spread:.3f})"


class VolatilityExpert(Expert):
    """
    Quote proportional to current realised volatility.
    s = scale * sigma * mid,  where scale ~ 2 (two sigma move cover).
    """
    def __init__(self, scale: float = 2.0, min_spread: float = 0.005):
        self.scale      = scale
        self.min_spread = min_spread

    def quote(self, context: dict) -> float:
        vol = context.get("realized_vol", 0.01)
        mid = context.get("mid_price", 100.0)
        return max(self.min_spread, self.scale * vol * mid)


class InventoryExpert(Expert):
    """
    Widen spread when inventory is large to discourage further accumulation.
    s = base + k * |inventory|
    """
    def __init__(self, base: float = 0.02, k: float = 0.002):
        self.base = base
        self.k    = k

    def quote(self, context: dict) -> float:
        inv = abs(context.get("inventory", 0))
        return self.base + self.k * inv


class OFIExpert(Expert):
    """
    Widen spread when order-flow imbalance is high (informed order pressure).
    s = base + k * |OFI|
    """
    def __init__(self, base: float = 0.02, k: float = 0.05):
        self.base = base
        self.k    = k

    def quote(self, context: dict) -> float:
        ofi = abs(context.get("ofi", 0.0))
        return self.base + self.k * ofi


class DepthImbalanceExpert(Expert):
    """
    Use depth imbalance (bid_vol / ask_vol ratio) to adjust quote.
    High bid volume relative to ask => likely upward pressure => widen ask.
    """
    def __init__(self, base: float = 0.02, sensitivity: float = 0.03):
        self.base        = base
        self.sensitivity = sensitivity

    def quote(self, context: dict) -> float:
        bid_vol = max(context.get("bid_volume", 1), 1)
        ask_vol = max(context.get("ask_volume", 1), 1)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)  # in [-1, 1]
        return self.base + self.sensitivity * abs(imbalance)


class ASHeuristicExpert(Expert):
    """
    Avellaneda-Stoikov closed-form spread used as an expert:
      s* = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)

    Parameters from A-S paper:
      sigma : mid-price volatility
      kappa : order-arrival decay
      gamma : risk aversion
    """
    def __init__(self, sigma: float = 0.01, kappa: float = 1.5,
                 gamma: float = 0.1):
        self.sigma = sigma
        self.kappa = kappa
        self.gamma = gamma

    def quote(self, context: dict) -> float:
        T_rem = context.get("time_remaining", 0.5)
        T_rem = max(T_rem, 1e-4)
        inventory_risk  = self.gamma * self.sigma**2 * T_rem
        adverse_sel     = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        return max(0.005, inventory_risk + adverse_sel)

    @property
    def name(self) -> str:
        return f"AS(sigma={self.sigma:.3f})"


class MomentumExpert(Expert):
    """
    Quote wide when recent mid-price changes are large (trending market).
    Uses the magnitude of the last N mid-price returns as a signal.
    """
    def __init__(self, base: float = 0.02, scale: float = 5.0):
        self.base  = base
        self.scale = scale

    def quote(self, context: dict) -> float:
        momentum = abs(context.get("momentum", 0.0))
        return self.base + self.scale * momentum


# ================================================================
# EXP4 Market Maker
# ================================================================

class EXP4MarketMaker:
    """EXP4 contextual market maker."""

    def __init__(
        self,
        experts: List[Expert],
        gamma: float = 0.1,
        T: Optional[int] = None,
    ):
        self.experts = experts
        self.N       = len(experts)

        if T is not None:
            # EXP4 learning rate for a known horizon.
            self.gamma = float(np.sqrt(np.log(self.N) / (self.N * T)))
        else:
            self.gamma = float(gamma)

        self.log_w = np.zeros(self.N)   # log-space weights

        self.last_idx:    Optional[int]       = None
        self.last_p:      Optional[float]     = None
        self.last_context: Optional[dict]     = None
        self.t: int = 0

        self.reward_history: List[float]      = []
        self.expert_history: List[int]        = []
        self.dist_history:   List[np.ndarray] = []

    # ---- Context builder (static method for reuse) --------------

    @staticmethod
    def build_context(
        mid:            float,
        spread:         float,
        ofi:            float,
        inventory:      int,
        realized_vol:   float,
        time_remaining: float,
        bid_volume:     int = 0,
        ask_volume:     int = 0,
        momentum:       float = 0.0,
    ) -> dict:
        return {
            "mid_price":      mid,
            "spread":         spread,
            "ofi":            ofi,
            "inventory":      inventory,
            "realized_vol":   realized_vol,
            "time_remaining": time_remaining,
            "bid_volume":     bid_volume,
            "ask_volume":     ask_volume,
            "momentum":       momentum,
        }

    # ---- Core interface ----------------------------------------

    def get_distribution(self) -> np.ndarray:
        log_w_shifted = self.log_w - self.log_w.max()
        w = np.exp(log_w_shifted)
        q = (1.0 - self.gamma) * w / w.sum() + self.gamma / self.N
        q = np.clip(q, 1e-15, 1.0)
        q /= q.sum()
        return q

    def choose_spread(self, context: dict) -> float:
        """
        Sample an expert, return that expert's recommended spread.
        The expert may use context; the MM is agnostic to which expert was chosen.
        """
        q   = self.get_distribution()
        idx = int(np.random.choice(self.N, p=q))

        self.last_idx     = idx
        self.last_p       = float(q[idx])
        self.last_context = context
        self.dist_history.append(q.copy())

        return float(self.experts[idx].quote(context))

    def update(self, reward: float, counterfactual_rewards=None):
        if self.last_idx is None:
            return

        q = self.get_distribution()
        # Importance-weighted update for chosen expert only
        x_hat_chosen = reward / q[self.last_idx]
        self.log_w[self.last_idx] += self.gamma * x_hat_chosen / self.N

        self.reward_history.append(reward)
        self.expert_history.append(self.last_idx)
        self.t += 1

    # ---- Introspection -----------------------------------------

    def expert_weights(self) -> Dict[str, float]:
        """Return current unnormalised weights per expert (for logging)."""
        q = self.get_distribution()
        return {self.experts[i].name: float(q[i]) for i in range(self.N)}

    def dominant_expert(self) -> str:
        q = self.get_distribution()
        return self.experts[int(np.argmax(q))].name

    def theoretical_regret_bound(self, T: Optional[int] = None) -> float:
        T = T or self.t
        return 2.0 * np.sqrt(T * self.N * np.log(self.N))

    def reset(self):
        self.log_w       = np.zeros(self.N)
        self.last_idx    = None
        self.last_p      = None
        self.last_context = None
        self.t           = 0
        self.reward_history.clear()
        self.expert_history.clear()
        self.dist_history.clear()

    def __repr__(self) -> str:
        return (f"EXP4MarketMaker(N={self.N}, gamma={self.gamma:.4f}, "
                f"dominant={self.dominant_expert()}, t={self.t})")


# ================================================================
# Factory for building the default expert pool from config
# ================================================================

def build_expert_pool(cfg: dict) -> List[Expert]:
    """Build the expert pool described in config.yaml."""
    experts: List[Expert] = []
    for spec in cfg.get("experts", []):
        etype = spec["type"]
        if etype == "fixed_spread":
            experts.append(FixedSpreadExpert(spec["spread"]))
        elif etype == "volatility":
            experts.append(VolatilityExpert())
        elif etype == "inventory":
            experts.append(InventoryExpert())
        elif etype == "ofi":
            experts.append(OFIExpert())
        elif etype == "depth_imbalance":
            experts.append(DepthImbalanceExpert())
        elif etype == "avellaneda_stoikov":
            experts.append(ASHeuristicExpert(
                sigma=spec.get("sigma", 0.01),
                kappa=spec.get("kappa", 1.5),
                gamma=spec.get("gamma", 0.1),
            ))
        elif etype == "momentum":
            experts.append(MomentumExpert())
        else:
            logger.warning(f"Unknown expert type: {etype}")
    if not experts:
        # Sensible defaults if nothing specified
        experts = [
            FixedSpreadExpert(0.01),
            FixedSpreadExpert(0.05),
            VolatilityExpert(),
            InventoryExpert(),
            OFIExpert(),
            ASHeuristicExpert(),
        ]
    return experts
