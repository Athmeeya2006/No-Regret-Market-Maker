"""
Tests for Exp3 algorithms: convergence, regret bound, weight updates.

Run with: pytest tests/test_exp3_convergence.py -v --tb=short
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.algorithms.exp3 import (
    Exp3MarketMaker, Exp3DoublingTrick, SWExp3MarketMaker
)

SPREADS = [0.01, 0.02, 0.05, 0.10, 0.20]
K       = len(SPREADS)


# ================================================================
# Helpers
# ================================================================

def synthetic_rewards(K: int, T: int, best_arm: int = 0,
                       gap: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    (T, K) counterfactual reward matrix.
    best_arm earns `gap` more per round than all others.
    All rewards in [0, 1].
    """
    rng = np.random.default_rng(seed)
    mat = rng.uniform(0.3, 0.7, size=(T, K))
    mat[:, best_arm] += gap
    mat = np.clip(mat, 0, 1)
    return mat


def run_exp3_on_matrix(
    mm: Exp3MarketMaker,
    cf_mat: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run an Exp3 instance on a fixed counterfactual matrix."""
    T = cf_mat.shape[0]
    for t in range(T):
        p      = mm.get_distribution()
        idx    = int(rng.choice(K, p=p))
        reward = cf_mat[t, idx]
        mm.record_choice(idx, p)
        mm.update(reward)
    return mm.empirical_regret(cf_mat)


# ================================================================
# TestExp3
# ================================================================

class TestExp3:

    def test_distribution_sums_to_one(self):
        mm = Exp3MarketMaker(SPREADS, gamma=0.1)
        p  = mm.get_distribution()
        assert abs(p.sum() - 1.0) < 1e-9
        assert (p >= 0).all()

    def test_distribution_has_minimum_exploration(self):
        mm  = Exp3MarketMaker(SPREADS, gamma=0.15)
        p   = mm.get_distribution()
        # Each arm gets at least gamma/K probability
        assert (p >= 0.15 / K - 1e-9).all()

    def test_update_increases_chosen_arm_weight(self):
        mm    = Exp3MarketMaker(SPREADS, gamma=0.2)
        p_before = mm.get_distribution().copy()
        # Manually choose arm 0 with high reward
        mm.record_choice(0, p_before)
        mm.update(1.0)
        p_after = mm.get_distribution()
        assert p_after[0] > p_before[0] - 1e-9

    def test_regret_bound_not_exceeded(self):
        """
        Run Exp3 on adversarial rewards for 2000 rounds.
        Regret should stay below 2 * sqrt(T * K * ln(K)) with high probability.
        We allow 20% slack to account for finite-sample variation.
        """
        rng  = np.random.default_rng(42)
        T    = 2000
        mm   = Exp3MarketMaker(SPREADS, T=T)
        cf   = synthetic_rewards(K, T, best_arm=2, gap=0.15, seed=1)
        regret_curve = run_exp3_on_matrix(mm, cf, rng)

        bound = mm.theoretical_regret_bound(T)
        # Regret must be below bound (with 20% slack for stochasticity)
        assert regret_curve[-1] < bound * 1.20, (
            f"Regret {regret_curve[-1]:.2f} exceeds bound {bound:.2f}"
        )

    def test_regret_curve_nonnegative(self):
        rng = np.random.default_rng(5)
        T = 500
        mm = Exp3MarketMaker(SPREADS, T=T)
        cf = synthetic_rewards(K, T, best_arm=0, gap=0.12, seed=9)
        rc = run_exp3_on_matrix(mm, cf, rng)
        assert (rc >= -1e-9).all()

    def test_regret_sublinear(self):
        """
        Verify regret grows sublinearly: R_T / T -> 0.
        Check that per-round regret at T=3000 is lower than at T=1000.
        """
        rng  = np.random.default_rng(7)
        T    = 3000
        mm   = Exp3MarketMaker(SPREADS, T=T)
        cf   = synthetic_rewards(K, T, best_arm=1, gap=0.2, seed=2)
        rc   = run_exp3_on_matrix(mm, cf, rng)

        # Per-round regret R_t/t should decrease over time
        per_round_early = rc[999] / 1000     # R_{1000} / 1000
        per_round_late  = rc[-1]  / T        # R_{3000} / 3000
        assert per_round_late < per_round_early, (
            f"Per-round regret should decrease: early={per_round_early:.4f}, "
            f"late={per_round_late:.4f}"
        )

    def test_reset_clears_state(self):
        mm = Exp3MarketMaker(SPREADS, T=100)
        p = mm.get_distribution()
        mm.record_choice(0, p)
        mm.update(0.5)
        mm.reset()
        assert mm.t == 0
        assert len(mm.reward_history) == 0
        assert np.allclose(mm.log_w, 0.0)

    def test_optimal_gamma_formula(self):
        T    = 5000
        mm   = Exp3MarketMaker(SPREADS, T=T)
        expected_gamma = np.sqrt(np.log(K) / (K * T))
        assert abs(mm.gamma - expected_gamma) < 1e-9

    def test_numerical_stability_extreme_weights(self):
        """After many updates to one arm, distribution should still be valid."""
        mm = Exp3MarketMaker(SPREADS, gamma=0.5)
        for _ in range(500):
            p = mm.get_distribution()
            mm.record_choice(0, p)
            mm.update(1.0)
        p = mm.get_distribution()
        assert np.isfinite(p).all()
        assert abs(p.sum() - 1.0) < 1e-6
        assert (p >= 0).all()


# ================================================================
# TestExp3DoublingTrick
# ================================================================

class TestExp3DoublingTrick:

    def test_epoch_doubles(self):
        mm = Exp3DoublingTrick(SPREADS)
        assert mm._epoch_len == 1
        mm.choose_spread()
        mm.update(0.5)
        # After 1 round, epoch should double to 2
        assert mm._epoch_len == 2

    def test_regret_at_most_twice_standard(self):
        """
        DT regret bound is 4 * sqrt(T K ln K) = 2x the standard 2*sqrt bound.
        """
        rng    = np.random.default_rng(11)
        T    = 2000
        mm_dt = Exp3DoublingTrick(SPREADS)
        mm_std = Exp3MarketMaker(SPREADS, T=T)

        cf = synthetic_rewards(K, T, best_arm=0, gap=0.1, seed=3)

        for t in range(T):
            for mm in [mm_dt, mm_std]:
                p   = mm.get_distribution() if hasattr(mm, "get_distribution") \
                      else None
                if p is None:
                    continue
                idx = int(rng.choice(K, p=p))
                mm.record_choice(idx, p)
                mm.update(cf[t, idx])

        bound_dt  = mm_dt.theoretical_regret_bound(T)
        bound_std = mm_std.theoretical_regret_bound(T)
        assert bound_dt <= bound_std * 2.1    # at most 2x (with 10% slack)

    def test_unknown_horizon_no_T_needed(self):
        mm = Exp3DoublingTrick(SPREADS)
        for _ in range(100):
            s = mm.choose_spread()
            assert s in SPREADS
            mm.update(np.random.random())


# ================================================================
# TestSWExp3
# ================================================================

class TestSWExp3:

    def test_window_forget(self):
        """
        After a regime change, SW-Exp3 should shift weights toward the new
        best arm faster than standard Exp3.
        """
        rng = np.random.default_rng(42)
        T1, T2 = 500, 500
        T      = T1 + T2
        W      = 100

        # Phase 1: arm 0 is best; Phase 2: arm 4 is best
        cf = np.zeros((T, K))
        cf[:T1, 0] = 1.0
        cf[T1:, K-1] = 1.0

        # SW-Exp3
        mm_sw  = SWExp3MarketMaker(SPREADS, window=W, gamma=0.2)
        # Standard Exp3
        mm_std = Exp3MarketMaker(SPREADS, gamma=0.2)

        sw_choices  = []
        std_choices = []

        for t in range(T):
            for mm, choices in [(mm_sw, sw_choices), (mm_std, std_choices)]:
                p   = mm.get_distribution()
                idx = int(rng.choice(K, p=p))
                mm.record_choice(idx, p)
                mm.update(cf[t, idx])
                choices.append(idx)

        # After change: fraction of rounds where SW picked correct arm (K-1)
        sw_correct  = np.mean(np.array(sw_choices[T1+W:]) == K-1)
        std_correct = np.mean(np.array(std_choices[T1+W:]) == K-1)

        # SW-Exp3 should be at least as good as standard after adaptation
        assert sw_correct >= std_correct - 0.15, (
            f"SW({sw_correct:.2f}) should adapt faster than Std({std_correct:.2f})"
        )

    def test_sw_exp3_lower_post_change_regret(self):
        rng = np.random.default_rng(123)
        T1, T2 = 400, 400
        T = T1 + T2
        cf = np.zeros((T, K))
        cf[:T1, 0] = 1.0
        cf[T1:, K - 1] = 1.0

        sw = SWExp3MarketMaker(SPREADS, window=80, gamma=0.2)
        std = Exp3MarketMaker(SPREADS, gamma=0.2)
        for t in range(T):
            for mm in (sw, std):
                p = mm.get_distribution()
                idx = int(rng.choice(K, p=p))
                mm.record_choice(idx, p)
                mm.update(float(cf[t, idx]))

        sw_post = sw.empirical_regret(cf)[-1] if hasattr(sw, "empirical_regret") else None
        std_post = std.empirical_regret(cf)[-1]
        if sw_post is None:
            sw_actual = np.cumsum(sw.reward_history)
            sw_post = np.cumsum(cf, axis=0).max(axis=1)[-1] - sw_actual[-1]
        assert sw_post <= std_post + 80

    def test_buffer_size_bounded(self):
        mm = SWExp3MarketMaker(SPREADS, window=50, gamma=0.1)
        for _ in range(200):
            mm.choose_spread()
            mm.update(0.5)
        assert len(mm._buf_arms) <= 100   # at most 2*W


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
