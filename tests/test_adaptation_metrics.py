"""
Tests for AdaptationSpeedAnalyzer.

Run with: pytest tests/test_adaptation_metrics.py -v --tb=short
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.evaluation.adaptation_metrics import AdaptationSpeedAnalyzer

SPREADS = [0.01, 0.02, 0.05, 0.10, 0.20]
K       = len(SPREADS)


class TestAdaptationMetrics:

    def test_adaptation_time_measured(self):
        T1, T2 = 300, 300
        T      = T1 + T2
        rewards = np.concatenate([np.ones(T1), 0.5 * np.ones(T2)])
        rng = np.random.default_rng(1)
        rewards += rng.normal(0, 0.05, T)
        analyzer = AdaptationSpeedAnalyzer(K=K, window=30, epsilon=0.3)
        events   = analyzer.measure_adaptation(rewards, change_times=[T1])
        assert len(events) == 1
        assert events[0].change_time == T1

    def test_immediate_recovery_has_small_adaptation_time(self):
        rewards = np.ones(80)
        analyzer = AdaptationSpeedAnalyzer(K=K, window=10, epsilon=0.05)
        event = analyzer.measure_adaptation(rewards, change_times=[40])[0]
        assert event.adaptation_time == 1

    def test_never_recovers_has_none(self):
        rewards = np.r_[np.ones(40), np.zeros(80)]
        analyzer = AdaptationSpeedAnalyzer(K=K, window=10, epsilon=0.05)
        event = analyzer.measure_adaptation(rewards, change_times=[40])[0]
        assert event.adaptation_time is None

    def test_adaptation_times_nonnegative(self):
        rewards = np.r_[np.ones(40), np.ones(80)]
        analyzer = AdaptationSpeedAnalyzer(K=K, window=10, epsilon=0.05)
        events = analyzer.measure_adaptation(rewards, change_times=[40])
        for event in events:
            if event.adaptation_time is not None:
                assert event.adaptation_time >= 0

    def test_theoretical_curve_shape(self):
        analyzer = AdaptationSpeedAnalyzer(K=5)
        K_vals   = np.array([2, 5, 10, 20])
        delta    = np.ones(4) * 0.1
        curve    = analyzer.theoretical_curve(K_vals, delta)
        assert (np.diff(curve) > 0).all()

    def test_sw_prediction_equals_window(self):
        analyzer = AdaptationSpeedAnalyzer(K=5)
        t = analyzer.sw_exp3_theoretical(window=200, delta=0.1)
        assert t == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
