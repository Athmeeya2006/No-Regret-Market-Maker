"""
Tests for Avellaneda-Stoikov model-based market maker.

Run with: pytest tests/test_as_model.py -v --tb=short
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.algorithms.avellaneda_stoikov import AvellanedaStoikovMM


class TestAvellanedaStoikov:

    def test_optimal_spread_formula(self):
        """Verify the closed-form spread matches hand computation."""
        sigma, kappa, gamma = 0.01, 1.5, 0.1
        T_rem = 0.5
        mm    = AvellanedaStoikovMM(sigma, kappa, gamma)
        s     = mm.optimal_spread(T_rem)
        expected = gamma * sigma**2 * T_rem + (2/gamma) * np.log(1 + gamma/kappa)
        assert abs(s - expected) < 1e-12

    def test_reservation_price_direction(self):
        """Long inventory => reservation price < mid."""
        mm  = AvellanedaStoikovMM(sigma=0.01, kappa=1.5, gamma=0.1)
        mid = 100.0
        r   = mm.reservation_price(mid, inventory=5, time_remaining=0.5)
        assert r < mid

        r2 = mm.reservation_price(mid, inventory=-5, time_remaining=0.5)
        assert r2 > mid

    def test_spread_narrows_as_horizon_approaches(self):
        """Inventory risk term -> 0 as T - t -> 0, so spread narrows."""
        mm = AvellanedaStoikovMM(sigma=0.01, kappa=1.5, gamma=0.1)
        s1 = mm.optimal_spread(1.0)
        s2 = mm.optimal_spread(0.01)
        assert s2 < s1

    def test_spread_increases_with_gamma(self):
        low = AvellanedaStoikovMM(sigma=5.0, kappa=1.5, gamma=0.05)
        high = AvellanedaStoikovMM(sigma=5.0, kappa=1.5, gamma=0.2)
        assert high.optimal_spread(1.0) > low.optimal_spread(1.0)

    def test_spread_increases_with_sigma(self):
        low = AvellanedaStoikovMM(sigma=0.01, kappa=1.5, gamma=0.1)
        high = AvellanedaStoikovMM(sigma=0.20, kappa=1.5, gamma=0.1)
        assert high.optimal_spread(1.0) > low.optimal_spread(1.0)

    def test_quotes_straddle_reservation_price(self):
        mm = AvellanedaStoikovMM(sigma=0.01, kappa=1.5, gamma=0.1)
        mid = 100.0
        bid, ask = mm.quote(mid, inventory=0, time_remaining=0.5)
        r = mm.reservation_price(mid, 0, 0.5)
        s = mm.optimal_spread(0.5)
        assert abs(ask - bid - s) < 1e-10
        assert abs((bid + ask) / 2 - r) < 1e-10

    def test_sigma_estimation(self):
        """estimated sigma should be close to true sigma."""
        true_sigma = 0.01
        n          = 5000
        mid_prices = 100 * np.exp(
            np.cumsum(np.random.default_rng(5).normal(0, true_sigma, n))
        )
        est = AvellanedaStoikovMM.estimate_sigma(mid_prices, dt=1.0)
        assert abs(est - true_sigma) / true_sigma < 0.15   # within 15%

    def test_choose_spread_returns_positive_float(self):
        mm = AvellanedaStoikovMM(sigma=0.01, kappa=1.5, gamma=0.1)
        spread = mm.choose_spread({"mid_price": 100.0, "inventory": 0, "time_remaining": 0.5})
        assert isinstance(spread, float)
        assert spread > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
