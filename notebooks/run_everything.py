"""Run build, benchmark, plot, and invariant checks for the project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).resolve().parents[1]


def report(label: str, fn) -> None:
    print(f"{label}...", end=" ", flush=True)
    try:
        fn()
    except Exception as exc:
        print(f"failed: {exc}")
        raise
    print("ok")


def main():
    parser = argparse.ArgumentParser(description="Run project validation")
    parser.add_argument("--n-rounds", type=int, default=400, help="Simulation rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-build", action="store_true", help="Skip C++ build")
    args = parser.parse_args()

    if not args.skip_build:
        report(
            "building C++ extension",
            lambda: subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ROOT),
        )

    from python.algorithms.adversarial_trader import (
        AdversarialInformedTrader,
        NoiseTrader,
        TrueValueProcess,
    )
    from python.algorithms.avellaneda_stoikov import AvellanedaStoikovMM
    from python.algorithms.exp3 import Exp3MarketMaker
    from python.evaluation.adaptation_metrics import AdaptationSpeedAnalyzer
    from python.evaluation.benchmarker import Benchmarker
    from python.simulation.market_simulator import MarketSimulator
    from python.simulation.python_lob_stub import PythonLOBStub
    from python.simulation.regime_generator import RegimeGenerator

    try:
        import lob_engine

        def make_engine():
            return lob_engine.MatchingEngine(tick_size=0.01)
    except ImportError:
        def make_engine():
            return PythonLOBStub(tick_size=0.01)

    spread_choices = [0.01, 0.02, 0.05, 0.10, 0.20]
    regime_gen = RegimeGenerator(regime_duration=max(args.n_rounds // 4, 1))

    def check_exp3_bound() -> None:
        mm = Exp3MarketMaker(spread_choices, T=100)
        assert mm.theoretical_regret_bound(100) > 0

    def sim_factory(mm):
        value_process = TrueValueProcess(initial_value=100.0, vol=0.02, seed=args.seed)
        return MarketSimulator(
            engine=make_engine(),
            noise_trader=NoiseTrader(arrival_rate=5.0, seed=args.seed),
            informed_trader=AdversarialInformedTrader(
                true_value_process=value_process,
                mm_spread_choices=spread_choices,
                seed=args.seed,
            ),
            market_maker=mm,
            spread_choices=spread_choices,
            regime_gen=regime_gen,
            seed=args.seed,
        )

    def check_as_closed_form() -> None:
        mm = AvellanedaStoikovMM(sigma=0.01, kappa=1.5, gamma=0.1)
        assert mm.optimal_spread(0.5) > 0
        assert mm.reservation_price(100.0, 5, 0.5) < 100.0

    def run_benchmark() -> None:
        bench = Benchmarker(sim_factory, spread_choices, args.n_rounds, seed=args.seed)
        results = bench.run_all()
        assert results
        change_times = regime_gen.get_change_times(args.n_rounds)
        analyzer = AdaptationSpeedAnalyzer(K=len(spread_choices))
        analyzer.compare_algorithms(
            {name: result.reward_series for name, result in results.items()},
            change_times,
        )

    def generate_analysis_outputs() -> None:
        scripts = [
            ROOT / "analysis" / "convergence_plots.py",
            ROOT / "analysis" / "pnl_comparison.py",
            ROOT / "analysis" / "adaptation_plots.py",
            ROOT / "analysis" / "regret_decomposition_plots.py",
        ]
        for script in scripts:
            subprocess.check_call([sys.executable, str(script)], cwd=ROOT)

    def check_outputs() -> None:
        expected = [
            ROOT / "analysis" / "convergence_kuhn.png",
            ROOT / "analysis" / "pnl_comparison.png",
            ROOT / "analysis" / "adaptation_speed.png",
            ROOT / "analysis" / "regret_decomposition.png",
        ]
        missing = [path for path in expected if not path.exists()]
        assert not missing, f"missing outputs: {missing}"

    report("checking Exp3 regret bound", check_exp3_bound)
    report("verifying A-S closed form", check_as_closed_form)
    report("running benchmark smoke test", run_benchmark)
    report("generating analysis outputs", generate_analysis_outputs)
    report("checking output files", check_outputs)


if __name__ == "__main__":
    main()
