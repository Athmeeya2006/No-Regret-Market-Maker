"""Small pure-Python fallback used when the C++ extension is unavailable."""

from __future__ import annotations

import math
from collections import deque


class PythonLOBStub:
    """Minimal MatchingEngine-compatible object for light simulations."""

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self._mid = 100.0
        self._mid_history = deque([self._mid], maxlen=2000)

    def mid_price(self) -> float:
        return self._mid

    def spread(self) -> float:
        return 0.02

    def ofi(self) -> float:
        return 0.0

    def total_bid_volume(self, levels: int = 10) -> int:
        return 0

    def total_ask_volume(self, levels: int = 10) -> int:
        return 0

    def realized_volatility(self, window: int = 20) -> float:
        values = list(self._mid_history)[-max(window + 1, 2):]
        if len(values) < 2:
            return 0.0
        returns = [
            math.log(values[i + 1] / values[i])
            for i in range(len(values) - 1)
            if values[i] > 0 and values[i + 1] > 0
        ]
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(var)

    def mid_price_history(self):
        return list(self._mid_history)

    def submit_limit(self, side, price: float, qty: int, trader_id: int = -1):
        self._mid = max(0.01, float(price))
        self._mid_history.append(self._mid)
        return []

    def submit_market(self, side, qty: int, trader_id: int = -1):
        return []

    def cancel(self, order_id: int) -> bool:
        return False

    def last_order_id(self) -> int:
        return 0
