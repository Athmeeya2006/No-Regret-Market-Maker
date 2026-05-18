"""
Tests for EXP4 (expert advice) algorithm.

Run with: pytest tests/test_exp4_experts.py -v --tb=short
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.algorithms.exp4 import (
    EXP4MarketMaker, FixedSpreadExpert, VolatilityExpert,
    InventoryExpert, OFIExpert, ASHeuristicExpert
)
from python.algorithms.avellaneda_stoikov import AvellanedaStoikovMM


SPREADS = [0.01, 0.02, 0.05, 0.10, 0.20]
K       = len(SPREADS)


class TestEXP4:

    def _make_mm(self):
        experts = [
            FixedSpreadExpert(0.01),
            FixedSpreadExpert(0.05),
            VolatilityExpert(),
            InventoryExpert(),
            OFIExpert(),
        ]
        return EXP4MarketMaker(experts, gamma=0.1)

    def test_spread_is_positive(self):
        mm = self._make_mm()
        ctx = EXP4MarketMaker.build_context(
            mid=100.0, spread=0.02, ofi=0.1,
            inventory=0, realized_vol=0.01,
            time_remaining=0.5,
        )
        s = mm.choose_spread(ctx)
        assert s > 0

    def test_all_experts_return_positive_spreads(self):
        mm = self._make_mm()
        ctx = EXP4MarketMaker.build_context(100.0, 0.02, 0.2, 3, 0.01, 0.5)
        for expert in mm.experts:
            assert expert.quote(ctx) > 0

    def test_distribution_valid(self):
        mm = self._make_mm()
        q  = mm.get_distribution()
        assert abs(q.sum() - 1.0) < 1e-9
        assert (q >= 0).all()
        assert len(q) == 5

    def test_weight_update_chosen_expert_increases(self):
        mm = self._make_mm()
        q_before = mm.get_distribution().copy()
        ctx = EXP4MarketMaker.build_context(
            mid=100.0, spread=0.02, ofi=0.0,
            inventory=0, realized_vol=0.01,
            time_remaining=0.5,
        )
        mm.choose_spread(ctx)
        idx = mm.last_idx
        mm.update(1.0)
        q_after = mm.get_distribution()
        assert q_after[idx] >= q_before[idx] - 1e-9

    def test_as_expert_closed_form(self):
        """ASHeuristicExpert should match A-S formula."""
        expert = ASHeuristicExpert(sigma=0.01, kappa=1.5, gamma=0.1)
        as_mm  = AvellanedaStoikovMM(sigma=0.01, kappa=1.5, gamma=0.1)
        ctx    = {"time_remaining": 0.5}
        s_expert = expert.quote(ctx)
        s_as     = as_mm.optimal_spread(0.5)
        assert abs(s_expert - s_as) < 1e-6

    def test_regret_bound_holds(self):
        np.random.seed(99)
        T  = 1000
        mm = self._make_mm()
        mm.gamma = np.sqrt(np.log(mm.N) / (mm.N * T))

        total_reward  = 0.0
        best_fixed    = 0.0
        expert_totals = np.zeros(mm.N)
        ctx = EXP4MarketMaker.build_context(100, 0.02, 0, 0, 0.01, 0.5)

        for _ in range(T):
            mm.choose_spread(ctx)
            reward = np.random.random()
            for i, e in enumerate(mm.experts):
                expert_totals[i] += reward  # same reward for all (simplified)
            mm.update(reward)
            total_reward += reward

        best_expert_total = expert_totals.max()
        regret            = best_expert_total - total_reward
        bound             = mm.theoretical_regret_bound(T)
        assert regret < bound * 1.2

    def test_learns_volatility_expert(self):
        np.random.seed(8)
        mm = self._make_mm()
        ctx = EXP4MarketMaker.build_context(100.0, 0.02, 0.0, 0, 0.01, 0.5)
        vol_idx = 2
        for _ in range(500):
            mm.choose_spread(ctx)
            reward = 1.0 if mm.last_idx == vol_idx else 0.0
            mm.update(reward)
        assert mm.get_distribution()[vol_idx] > 1.0 / mm.N

    def test_dominant_expert_matches_max_weight(self):
        mm = self._make_mm()
        mm.log_w[1] = 3.0
        assert mm.dominant_expert() == mm.experts[1].name


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
