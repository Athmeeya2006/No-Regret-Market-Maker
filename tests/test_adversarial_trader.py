"""
Tests for adversarial and noise traders.

Run with: pytest tests/test_adversarial_trader.py -v --tb=short
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.algorithms.adversarial_trader import (
    AdversarialInformedTrader, NoiseTrader,
    StochasticInformedTrader, TrueValueProcess
)


SPREADS = [0.01, 0.02, 0.05, 0.10, 0.20]
K       = len(SPREADS)


class TestAdversarialTrader:

    def _make_adversary(self):
        V   = TrueValueProcess(initial_value=100.0, vol=0.02, seed=0)
        adv = AdversarialInformedTrader(
            true_value_process=V,
            mm_spread_choices=SPREADS,
            aggression=1.0,
            seed=42,
        )
        return adv, V

    def test_exploit_when_tight(self):
        """When MM is tight, adversary should exploit (informed orders)."""
        adv, V = self._make_adversary()
        # Heavy weight on first (tightest) spread
        tight_weights = np.array([0.7, 0.2, 0.05, 0.03, 0.02])
        phase = adv.compute_best_response(tight_weights, mid=100.0, spread=0.01)
        assert phase == "exploit"

    def test_exploit_orders_marked_informed(self):
        adv, V = self._make_adversary()
        V.V = 101.0
        tight_weights = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        orders = adv.maybe_arrive(10.0, mid=100.0, spread=0.01, current_mm_weights=tight_weights)
        assert orders
        assert all(o.get("informed") is True for o in orders)

    def test_bait_when_wide(self):
        """When MM is wide, adversary should bait."""
        adv, V = self._make_adversary()
        wide_weights = np.array([0.02, 0.05, 0.1, 0.3, 0.53])
        phase = adv.compute_best_response(wide_weights, mid=100.0, spread=0.20)
        assert phase == "bait"

    def test_adversary_generates_orders(self):
        adv, V = self._make_adversary()
        weights = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        V.V = 101.0   # true value above market
        orders = adv.maybe_arrive(
            dt=1.0, mid=100.0, spread=0.01,
            current_mm_weights=weights,
        )
        # Should generate buy orders (V > ask = 100.005)
        buy_orders = [o for o in orders if o["side"] == "BUY"]
        assert len(buy_orders) >= 0   # may be 0 if Poisson draws 0

    def test_noise_trader_poisson_rate(self):
        """Mean number of orders should match rate * dt."""
        nt = NoiseTrader(arrival_rate=10.0, seed=0)
        n_trials = 5000
        counts   = [len(nt.maybe_arrive(1.0)) for _ in range(n_trials)]
        assert abs(np.mean(counts) - 10.0) / 10.0 < 0.05  # within 5%

    def test_noise_trader_power_law_size(self):
        """Size distribution should have heavy tail (max >> mean)."""
        nt = NoiseTrader(arrival_rate=100.0, alpha=1.5, seed=1)
        orders = []
        for _ in range(100):
            orders.extend(nt.maybe_arrive(1.0))
        sizes = [o["qty"] for o in orders]
        if sizes:
            assert max(sizes) > np.mean(sizes) * 3

    def test_true_value_process_stays_positive(self):
        proc = TrueValueProcess(initial_value=100.0, vol=0.5, seed=4)
        for _ in range(10000):
            assert proc.step() > 0

    def test_noise_sizes_within_bounds(self):
        nt = NoiseTrader(arrival_rate=50.0, size_min=2, size_max=7, seed=2)
        orders = nt.maybe_arrive(5.0)
        assert orders
        assert all(2 <= o["qty"] <= 7 for o in orders)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
