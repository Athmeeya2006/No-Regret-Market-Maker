"""
Tests for RegimeGenerator and FixedSpreadMM.

Run with: pytest tests/test_regime_generator.py -v --tb=short
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.simulation.regime_generator import RegimeGenerator
from python.algorithms.fixed_spread_mm import FixedSpreadMM


class TestRegimeGenerator:

    def test_correct_regime_at_boundary(self):
        gen = RegimeGenerator(regime_duration=100, transition="abrupt")
        assert gen.get_params(0)["name"]   == "calm"
        assert gen.get_params(99)["name"]  == "calm"
        assert gen.get_params(100)["name"] == "volatile"
        assert gen.get_params(200)["name"] == "illiquid"

    def test_cycles_through_all_regimes(self):
        gen    = RegimeGenerator(regime_duration=50, transition="abrupt")
        names  = {gen.get_params(t)["name"] for t in range(200)}
        expected = {r["name"] for r in gen.regimes}
        assert names == expected

    def test_gradual_interpolation(self):
        gen = RegimeGenerator(regime_duration=100, transition="gradual",
                               blend_frac=0.2)
        p90 = gen.get_params(90)
        p0  = gen.get_params(0)
        p100 = gen.get_params(100)
        assert p0["sigma"] <= p90["sigma"] <= p100["sigma"] or \
               p0["sigma"] >= p90["sigma"] >= p100["sigma"]

    def test_gradual_mid_blend_strictly_between_regimes(self):
        regimes = [
            {"name": "a", "sigma": 1.0, "mu": 1.0, "lambda": 1.0},
            {"name": "b", "sigma": 3.0, "mu": 3.0, "lambda": 3.0},
        ]
        gen = RegimeGenerator(regimes=regimes, regime_duration=100, transition="gradual", blend_frac=0.2)
        params = gen.get_params(90)
        assert 1.0 < params["sigma"] < 3.0
        assert 1.0 < params["mu"] < 3.0
        assert 1.0 < params["lambda"] < 3.0

    def test_change_times(self):
        gen   = RegimeGenerator(regime_duration=100, transition="abrupt")
        times = gen.get_change_times(400)
        assert times == [100, 200, 300]


class TestFixedSpreadMM:

    def test_always_returns_same_spread(self):
        mm = FixedSpreadMM(0.05)
        for _ in range(100):
            assert mm.choose_spread() == 0.05

    def test_update_records_reward(self):
        mm = FixedSpreadMM(0.05)
        mm.update(1.0)
        mm.update(-0.5)
        assert len(mm.reward_history) == 2
        assert mm.reward_history[0] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
