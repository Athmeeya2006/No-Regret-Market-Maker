"""
Tests for RegretTracker and RegretDecomposer.

Run with: pytest tests/test_regret_decomposition.py -v --tb=short
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.evaluation.regret_tracker import RegretTracker
from python.evaluation.regret_decomposer import RegretDecomposer

SPREADS = [0.01, 0.02, 0.05, 0.10, 0.20]
K       = len(SPREADS)


class TestRegretTracker:

    def test_regret_nonnegative(self):
        tracker = RegretTracker(SPREADS)
        for t in range(100):
            cf = {s: np.random.random() for s in SPREADS}
            tracker.record_round(SPREADS[0], np.random.random(), cf)
        assert tracker.total_regret() >= -1e-9

    def test_regret_zero_when_always_optimal(self):
        tracker = RegretTracker(SPREADS)
        for t in range(100):
            cf     = {s: 1.0 if s == SPREADS[2] else 0.5 for s in SPREADS}
            actual = cf[SPREADS[2]]
            tracker.record_round(SPREADS[2], actual, cf)
        assert tracker.total_regret() < 1e-6

    def test_regret_curve_monotone(self):
        """Cumulative regret should be non-decreasing when best arm is fixed."""
        tracker = RegretTracker(SPREADS)
        for t in range(200):
            # Arm index 3 always best (1.0), others get 0.3
            cf = {s: 1.0 if s == SPREADS[3] else 0.3 for s in SPREADS}
            tracker.record_round(SPREADS[1], 0.3, cf)
        curve = tracker.regret_curve()
        diffs = np.diff(curve)
        assert (diffs >= -1e-6).all()

    def test_best_fixed_arm_found(self):
        tracker = RegretTracker(SPREADS)
        for t in range(50):
            cf = {s: 1.0 if s == SPREADS[3] else 0.3 for s in SPREADS}
            tracker.record_round(SPREADS[0], 0.3, cf)
        assert tracker.best_fixed_arm() == SPREADS[3]

    def test_bootstrap_ci_coverage(self):
        tracker = RegretTracker(SPREADS)
        for t in range(200):
            cf = {s: np.random.random() for s in SPREADS}
            tracker.record_round(np.random.choice(SPREADS), np.random.random(), cf)
        lower, upper = tracker.ci_bootstrap(n_bootstrap=50)
        curve = tracker.regret_curve()
        within = np.mean((curve >= lower - 1e-3) & (curve <= upper + 1e-3))
        assert within > 0.7


class TestRegretDecomposer:

    def test_components_nonnegative(self):
        decomp = RegretDecomposer()
        for _ in range(100):
            decomp.record_trade(
                spread_earned=np.random.uniform(0, 0.1),
                was_informed=np.random.random() < 0.3,
                inventory_change=np.random.randint(-5, 6),
                mid_move_after=np.random.normal(0, 0.01),
            )
        s = decomp.summary()
        assert s["adverse_selection"] >= 0
        assert s["inventory_loss"] >= 0
        assert s["spread_revenue"] >= 0

    def test_components_sum_to_net(self):
        decomp = RegretDecomposer()
        decomp.record_trade(0.10, True, 2, -0.03)
        decomp.record_trade(0.05, False, -1, 0.02)
        s = decomp.summary()
        expected = s["spread_revenue"] - s["adverse_selection"] - s["inventory_loss"]
        assert s["net_pnl"] == pytest.approx(expected)

    def test_informed_only_adds_adverse_selection_loss(self):
        decomp = RegretDecomposer()
        decomp.record_trade(0.02, False, 1, 0.10)
        decomp.record_trade(0.02, True, 1, 0.10)
        assert decomp.adverse_selection_losses[0] == 0.0
        assert decomp.adverse_selection_losses[1] > 0.0

    def test_curves_correct_length(self):
        decomp = RegretDecomposer()
        N = 50
        for _ in range(N):
            decomp.record_trade(0.02, False, 1, 0.001)
        curves = decomp.component_curves()
        for k, v in curves.items():
            assert len(v) == N, f"Curve {k} has length {len(v)}, expected {N}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
