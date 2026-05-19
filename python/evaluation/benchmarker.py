"""Benchmark market-making algorithms on a shared simulation factory."""

from __future__ import annotations

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    algo_name:          str
    reward_series:      np.ndarray
    spread_series:      np.ndarray
    inventory_series:   np.ndarray
    total_reward:       float
    total_regret:       float
    theoretical_bound:  float
    bound_ratio:        float      # total_regret / theoretical_bound
    sharpe:             float
    max_drawdown:       float
    mean_inventory:     float
    runtime_sec:        float
    regret_curve:       np.ndarray
    counterfactual_matrix: Optional[np.ndarray] = None
    extra:              Dict[str, Any] = field(default_factory=dict)


class Benchmarker:
    """
    Runs all algorithms on the same simulation and records results.

    Usage
    -----
    bench = Benchmarker(sim_factory, spread_choices, n_rounds)
    results = bench.run_all()
    bench.print_summary(results)
    """

    def __init__(
        self,
        sim_factory,           # callable(mm) -> MarketSimulator
        spread_choices: List[float],
        n_rounds:       int   = 10000,
        dt:             float = 1.0,
        seed:           int   = 42,
        config:         Optional[dict] = None,
        config_path:    Optional[str] = None,
    ):
        self.sim_factory   = sim_factory
        self.spread_choices = spread_choices
        self.n_rounds      = n_rounds
        self.dt            = dt
        self.seed          = seed
        self.config        = config
        if config_path is None:
            root = Path(__file__).resolve().parents[2]
            config_path = str(root / "config.yaml")
        self.config_path   = config_path

    def run_all(self) -> Dict[str, BenchmarkResult]:
        """Build and run all algorithms. Returns results dict."""
        algos = self._build_all_algos()
        results = {}
        for name, mm in algos.items():
            logger.info(f"Running {name} for {self.n_rounds} rounds ...")
            result = self._run_single(name, mm)
            results[name] = result
            logger.info(
                f"  {name}: total_reward={result.total_reward:.2f}, "
                f"regret={result.total_regret:.2f}, "
                f"bound={result.theoretical_bound:.2f}, "
                f"ratio={result.bound_ratio:.3f}, "
                f"rt={result.runtime_sec:.1f}s"
            )
        return results

    def _run_single(self, name: str, mm) -> BenchmarkResult:
        from python.evaluation.regret_tracker import RegretTracker

        sim     = self.sim_factory(mm)
        tracker = RegretTracker(self.spread_choices)

        t0 = time.perf_counter()
        sim_results = sim.run(self.n_rounds, self.dt)
        rt = time.perf_counter() - t0

        for r in sim_results:
            tracker.record_round(
                chosen_spread          = r.spread_quoted,
                actual_reward          = r.reward,
                counterfactual_rewards = r.counterfactual_rewards,
            )

        rewards   = sim.reward_series()
        spreads   = sim.spread_series()
        inventory = sim.inventory_series()
        cf_mat    = sim.counterfactual_matrix()

        regret_curve      = tracker.regret_curve()
        total_regret      = tracker.total_regret()
        theoretical_bound = tracker.theoretical_bound()[-1] if len(rewards) > 0 else 0.0

        return BenchmarkResult(
            algo_name          = name,
            reward_series      = rewards,
            spread_series      = spreads,
            inventory_series   = inventory,
            total_reward       = float(rewards.sum()),
            total_regret       = float(total_regret),
            theoretical_bound  = float(theoretical_bound),
            bound_ratio        = float(total_regret / (theoretical_bound + 1e-8)),
            sharpe             = self._sharpe(rewards),
            max_drawdown       = self._max_drawdown(rewards),
            mean_inventory     = float(np.mean(np.abs(inventory))),
            runtime_sec        = rt,
            regret_curve       = regret_curve,
            counterfactual_matrix = cf_mat,
        )

    # ================================================================
    # Algorithm factory
    # ================================================================

    def _build_all_algos(self) -> Dict:
        from python.algorithms.exp3 import (
            Exp3MarketMaker, Exp3DoublingTrick, SWExp3MarketMaker
        )
        from python.algorithms.exp4 import EXP4MarketMaker, build_expert_pool
        from python.algorithms.avellaneda_stoikov import AvellanedaStoikovMM
        from python.algorithms.fixed_spread_mm import FixedSpreadMM

        K  = len(self.spread_choices)
        T  = self.n_rounds

        cfg = self._load_config()

        algos = {
            "Exp3":              Exp3MarketMaker(self.spread_choices, T=T),
            "Exp3-DoublingTrick": Exp3DoublingTrick(self.spread_choices),
            "SW-Exp3(W=200)":    SWExp3MarketMaker(self.spread_choices, window=200),
            "SW-Exp3(W=500)":    SWExp3MarketMaker(self.spread_choices, window=500),
            "EXP4":              EXP4MarketMaker(
                                     build_expert_pool(cfg), gamma=0.1, T=T
                                 ),
            "AvellanedaStoikov":  AvellanedaStoikovMM(
                                     sigma=0.01, kappa=1.5, gamma=0.1
                                 ),
        }

        # Fixed-spread baselines
        for s in self.spread_choices:
            algos[f"Fixed({s:.3f})"] = FixedSpreadMM(s)

        return algos

    def _load_config(self) -> dict:
        if self.config is not None:
            return self.config
        path = Path(self.config_path) if self.config_path else None
        if path is None or not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s", path, exc)
            return {}
        return cfg if isinstance(cfg, dict) else {}

    # ================================================================
    # Metrics helpers
    # ================================================================

    @staticmethod
    def _sharpe(rewards: np.ndarray, rf: float = 0.0) -> float:
        if len(rewards) < 2:
            return 0.0
        excess = rewards - rf
        std    = np.std(excess)
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(len(rewards)))

    @staticmethod
    def _max_drawdown(rewards: np.ndarray) -> float:
        cum   = np.cumsum(rewards)
        peak  = np.maximum.accumulate(cum)
        dd    = peak - cum
        return float(dd.max()) if len(dd) > 0 else 0.0

    # ================================================================
    # Reporting
    # ================================================================

    def print_summary(self, results: Dict[str, BenchmarkResult]):
        print("\n" + "=" * 80)
        print(f"{'Algorithm':<25} {'Reward':>10} {'Regret':>10} "
              f"{'Bound':>10} {'Ratio':>7} {'Sharpe':>7} {'MaxDD':>8}")
        print("-" * 80)

        for name, r in sorted(results.items(),
                               key=lambda x: -x[1].total_reward):
            print(
                f"{name:<25} {r.total_reward:>10.2f} {r.total_regret:>10.2f} "
                f"{r.theoretical_bound:>10.2f} {r.bound_ratio:>7.3f} "
                f"{r.sharpe:>7.2f} {r.max_drawdown:>8.2f}"
            )
        print("=" * 80)

    def to_dataframe(self, results: Dict[str, BenchmarkResult]):
        import pandas as pd
        rows = []
        for name, r in results.items():
            rows.append({
                "algo":             name,
                "total_reward":     r.total_reward,
                "total_regret":     r.total_regret,
                "theoretical_bound": r.theoretical_bound,
                "bound_ratio":      r.bound_ratio,
                "sharpe":           r.sharpe,
                "max_drawdown":     r.max_drawdown,
                "mean_inventory":   r.mean_inventory,
                "runtime_sec":      r.runtime_sec,
            })
        return pd.DataFrame(rows).sort_values("total_reward", ascending=False)
