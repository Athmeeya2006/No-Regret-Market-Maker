"""Economic PnL decomposition for market-making simulations."""

from __future__ import annotations

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TradeRecord:
    spread_earned:  float
    was_informed:   bool
    inventory_change: int
    mid_move_after: float


class RegretDecomposer:
    """
    Decomposes total MM regret into three economic components:

    1. Spread regret      : revenue from bid-ask spread
    2. Adverse selection  : loss from trading with informed traders
    3. Inventory regret   : mark-to-market loss from accumulated position

    The components are tracked separately so plots can attribute PnL to
    spread capture, informed flow, and inventory exposure.
    """

    def __init__(self):
        self.spread_gains:           List[float] = []
        self.adverse_selection_losses: List[float] = []
        self.inventory_losses:       List[float] = []
        self.trades:                 List[TradeRecord] = []

    def record_trade(
        self,
        spread_earned:    float,
        was_informed:     bool,
        inventory_change: int,
        mid_move_after:   float,
    ):
        # Spread revenue
        self.spread_gains.append(spread_earned)

        # Adverse selection: if the counterparty was informed and the mid
        # moved against us after the trade, we lose that amount
        if was_informed:
            # Informed buyer => mid goes up => MM (who sold) loses
            # Informed seller => mid goes down => MM (who bought) loses
            as_loss = abs(inventory_change) * abs(mid_move_after)
        else:
            as_loss = 0.0
        self.adverse_selection_losses.append(as_loss)

        # Inventory loss: mark-to-market movement of current inventory
        # Positive inventory (long) loses when mid falls
        inv_loss = max(0.0, -(inventory_change * mid_move_after))
        self.inventory_losses.append(inv_loss)

        self.trades.append(TradeRecord(
            spread_earned    = spread_earned,
            was_informed     = was_informed,
            inventory_change = inventory_change,
            mid_move_after   = mid_move_after,
        ))

    def summary(self) -> Dict:
        s  = sum(self.spread_gains)
        as_ = sum(self.adverse_selection_losses)
        inv = sum(self.inventory_losses)
        return {
            "spread_revenue":       s,
            "adverse_selection":    as_,
            "inventory_loss":       inv,
            "net_pnl":              s - as_ - inv,
            "informed_fraction":    sum(t.was_informed for t in self.trades)
                                    / max(len(self.trades), 1),
            "n_trades":             len(self.trades),
        }

    def component_curves(self) -> Dict[str, np.ndarray]:
        """Cumulative component PnL curves."""
        return {
            "spread":           np.cumsum(self.spread_gains),
            "adverse_selection": -np.cumsum(self.adverse_selection_losses),
            "inventory":        -np.cumsum(self.inventory_losses),
            "net":              np.cumsum([
                s - a - i for s, a, i in zip(
                    self.spread_gains,
                    self.adverse_selection_losses,
                    self.inventory_losses,
                )
            ]),
        }
