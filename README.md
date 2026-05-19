# No-Regret Market Making Engine

This repository is a full-stack sandbox for studying no-regret learning in market making. It combines a C++17 limit order book, Python bindings, adversarial and stochastic order flow, a simulation loop with regime shifts, and evaluation tooling for regret, adaptation speed, and PnL decomposition. The goal is to compare model-free bandit market makers (Exp3, SW-Exp3, EXP4) against model-based baselines (Avellaneda-Stoikov) under both stationary and non-stationary market conditions.

## Why no-regret market making

Classical market making assumes calibrated order arrival models and stationary volatility. In practice, flow changes and parameters drift. Exp3 and its variants guarantee sublinear regret even when rewards are adversarial, which makes them a natural fit for spread selection when the environment is not trusted. This project uses a controlled simulator to stress-test that idea.

## What you get

- A C++17 matching engine with price-time priority, cancellations, depth, OFI, and volatility statistics.
- A Python-facing `lob_engine` module via pybind11.
- Exp3, Exp3 with doubling trick, SW-Exp3, and EXP4 market makers.
- Avellaneda-Stoikov and fixed-spread baselines.
- Noise, stochastic informed, and adversarial informed traders.
- Regime generator for abrupt or gradual market shifts.
- Regret tracking, regret decomposition, and adaptation-speed metrics.
- Synthetic microstructure processes with volatility clustering and mean-reverting noise.
- Analysis scripts that generate plots and CSVs.
- LaTeX writeups with mathematical background and lower bounds.

## Repository layout

```text
cpp/
  lob/                 C++ order book and matching engine
  bindings/            pybind11 module definition
python/
  algorithms/          Exp3, EXP4, A-S, fixed spread, trader logic
  simulation/          simulator, regimes, synthetic microstructure
  evaluation/          regret, decomposition, adaptation, benchmark tools
  analysis/            plotting helpers for scripts
tests/                 unit tests for engine, algorithms, and metrics
writeup/               LaTeX math notes
notebooks/             runnable experiment entrypoint
analysis/              plotting scripts and generated CSV/PNG outputs
config.yaml            default experiment configuration
```

## Quick start

Requirements: Python 3.12+, a C++17 compiler, and CMake.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify the C++ extension:

```bash
python -c "import lob_engine; print(lob_engine.MatchingEngine().mid_price())"
```

Run the full smoke test + plots (builds the extension, runs a short benchmark, and generates figures):

```bash
python notebooks/run_everything.py
```

You can skip the build step if you already compiled:

```bash
python notebooks/run_everything.py --skip-build
```

## Tests

```bash
python -m pytest tests -q
```


## The simulation loop

`MarketSimulator` orchestrates each round as follows:

1. Apply the current regime (volatility, informed fraction, arrival rate).
2. Build a context vector for the market maker from book state and inventory.
3. Ask the market maker for a spread and submit quotes to the LOB.
4. Sample noise orders and informed/adversarial orders, submit as market orders.
5. Advance the mid-price by a GBM step.
6. Compute reward: spread revenue + inventory PnL - inventory penalty.
7. Compute counterfactual rewards for every spread arm.
8. Update the market maker with the realized reward.

The context includes mid-price, spread, OFI, inventory, realized volatility, depth imbalance, momentum, and time remaining. It is designed to be compatible with EXP4 experts and model-based baselines.

## Synthetic data for controlled experiments

Real-world data adds engineering overhead and noise without improving the core guarantees this project tests. The experiments target regret bounds, adaptation speed, and failure modes under controlled regime shifts, so the primary data source is a synthetic microstructure process with realistic statistical properties:

- Volatility clustering from a GARCH(1,1) true value process.
- Microstructure noise from a mean-reverting observed-price perturbation.
- Fat-tailed order sizes already captured by the power-law size distribution in `NoiseTrader`.

See [python/simulation/synthetic_lob.py](python/simulation/synthetic_lob.py) for the synthetic processes and price-path helper.

## Algorithms

- **Exp3MarketMaker**: Standard adversarial bandit update with importance-weighted rewards.
- **Exp3DoublingTrick**: Restarts Exp3 on doubling epochs to handle unknown horizons.
- **SWExp3MarketMaker**: Sliding-window Exp3 that forgets old rewards to track regime shifts.
- **EXP4MarketMaker**: Contextual bandit that mixes a pool of experts.
- **AvellanedaStoikovMM**: Closed-form model-based spread quote; serves as a benchmark.
- **FixedSpreadMM**: Deterministic baseline used to anchor comparisons.

## Order flow models

- **NoiseTrader**: Poisson arrivals with power-law order sizes.
- **StochasticInformedTrader**: Trades when the true value exceeds the quoted prices.
- **AdversarialInformedTrader**: Observes the market maker's spread distribution and alternates between exploit and bait phases depending on whether the maker is tight or wide.
- **TrueValueProcess**: GBM with optional jump component; drives the informed traders.

## Evaluation toolkit

- **RegretTracker**: Exact external regret vs the best fixed spread in hindsight, using per-round counterfactuals.
- **RegretDecomposer**: Economic decomposition into spread revenue, adverse selection loss, and inventory loss.
- **AdaptationSpeedAnalyzer**: Measures recovery time after regime changes and compares to $O(\log K / \Delta^2)$ predictions.
- **Benchmarker**: Runs all algorithms on a shared simulation factory and reports reward, regret, bound ratio, Sharpe, max drawdown, and mean inventory.

## C++ engine notes

The matching engine is price-time priority with FIFO queues at each price level. It exposes:

- `submit_limit`, `submit_market`, and `cancel` for order handling.
- `best_bid`, `best_ask`, `mid_price`, `spread`, and `ofi` for state.
- `bid_levels`, `ask_levels`, `total_bid_volume`, `total_ask_volume` for depth.
- `vwap` and `realized_volatility` derived statistics.
- `fill_history` and `mid_price_history` for diagnostics.

The Python build exposes this as the `lob_engine` module and is used by the simulator. If the extension is unavailable, the simulator falls back to a minimal Python stub for light testing (slower and less detailed).

## Configuration

`config.yaml` defines simulation and algorithm parameters:

- Simulation horizon, tick size, inventory penalty, and mid-price volatility.
- Spread choices used by Exp3 and regret tracking.
- Noise and informed trader arrival rates and size scales.
- Adversarial trader aggression and tightness threshold.
- Avellaneda-Stoikov parameters.
- Exp3/EXP4 learning rates and SW-Exp3 window size.
- Regime duration, transition type, and regime list.
- EXP4 expert pool configuration.

## Analysis scripts and outputs

The top-level scripts in `analysis/` generate plots and CSVs:

- `convergence_plots.py` -> convergence plots and `convergence_regret.csv`.
- `pnl_comparison.py` -> cumulative PnL comparison and `pnl_comparison.csv`.
- `adaptation_plots.py` -> adaptation-speed plot and `adaptation_speed.csv`.
- `regret_decomposition_plots.py` -> PnL component plot and `regret_decomposition.csv`.

The PNGs are saved alongside the CSVs: `convergence_kuhn.png`, `pnl_comparison.png`, `adaptation_speed.png`, and `regret_decomposition.png`.

## Reproducibility notes

- Most components accept RNG seeds for deterministic runs.
- The simulator uses counterfactual rewards for exact regret curves.
- Use the C++ engine for realistic LOB mechanics; the Python stub is only for lightweight testing.

## CI

GitHub Actions builds the C++ extension, validates import of `lob_engine`, and runs the test suite on every push and pull request.
