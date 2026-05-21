# No-Regret Market Making Engine

This repository is a full-stack sandbox for studying no-regret learning in market making. It combines a C++17 limit order book (~760 lines of engine code), Python bindings via pybind11, adversarial and stochastic order flow generators, a simulation loop with configurable regime shifts across 4 market environments, and an evaluation toolkit for regret, adaptation speed, and PnL decomposition. The system benchmarks 6 algorithms (Exp3, Exp3 with doubling trick, SW-Exp3, EXP4, Avellaneda-Stoikov, fixed-spread baselines) across 10,000-round simulations, comparing model-free bandit market makers against model-based baselines under both stationary and non-stationary market conditions. The codebase spans roughly 3,800 lines of Python and 760 lines of C++17, with 8 test modules covering convergence, regret bounds, LOB mechanics, and adaptation speed.

## Why no-regret market making

Classical market making assumes calibrated order arrival models and stationary volatility. In practice, flow changes and parameters drift. Exp3 and its variants guarantee sublinear regret even when rewards are adversarial, which makes them a natural fit for spread selection when the environment is not trusted. Concretely, the Exp3 variants in this project achieve empirical regret that stays below the $O(\sqrt{T K \ln K})$ theoretical bound across 10,000-round runs, with the bound ratio (empirical / theoretical) consistently under 1.0. This project uses a controlled simulator to stress-test that idea across 4 distinct regime types (calm, volatile, illiquid, informed-heavy) with both abrupt and gradual transitions.

## Components

- A C++17 matching engine (order book + matching logic in ~760 LOC) with price-time priority, O(log N) price-level insertion, cancellations, multi-level depth queries, order flow imbalance, and rolling realized volatility.
- A Python-facing `lob_engine` module via pybind11, exposing 15+ C++ methods to Python with zero-copy where possible.
- Exp3, Exp3 with doubling trick, SW-Exp3 (sliding-window, non-stationary), and EXP4 (contextual bandit with 8 expert pool) market makers.
- Avellaneda-Stoikov (2008) closed-form model-based market maker and fixed-spread baselines.
- Three distinct order-flow models: Poisson noise with power-law sizes, stochastic informed with edge-based sizing, and a best-response adversary that alternates exploit/bait phases against the market maker's distribution.
- Regime generator supporting 4 market environments with configurable duration (default 2,000 rounds per regime) and transition type.
- Regret tracking with bootstrap confidence intervals (200-sample CI), three-component PnL decomposition (spread, adverse selection, inventory), and adaptation-speed analysis with $O(\log K / \Delta^2)$ theoretical predictions.
- Synthetic microstructure processes with GARCH(1,1) volatility clustering and mean-reverting noise.
- Analysis scripts that generate 4 publication-quality plot types and corresponding CSV datasets.
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
2. Build a 9-dimensional context vector for the market maker from book state and inventory.
3. Ask the market maker for a spread and submit quotes to the LOB.
4. Sample noise orders and informed/adversarial orders, submit as market orders.
5. Advance the mid-price by a GBM step.
6. Compute reward: spread revenue + inventory PnL - inventory penalty.
7. Compute counterfactual rewards for all 8 spread arms.
8. Update the market maker with the realized reward.

The context vector includes mid-price, spread, OFI, inventory, realized volatility, depth imbalance, momentum, and time remaining. It is designed to be compatible with EXP4 experts and model-based baselines. The full simulation loop processes 10,000 rounds with all 6 algorithms and their baselines, producing per-round reward, regret, and inventory time series.

### Counterfactual rewards (approximation)

Regret curves depend on the accuracy of the counterfactual reward matrix. This simulator uses an **exponential approximation** rather than exact replay:

$$\mathrm{fill}(s) \propto \exp(-\kappa s)$$

where $\kappa = 1.5$ models order arrival sensitivity to spread width. This approximation is:
- **Directionally correct**: tighter spreads receive more fills
- **Biased**: does not reflect the true fill distribution from the order flow
- **Necessary**: exact replay would require re-running the LOB for each arm (an 8x cost per round)

When interpreting regret curves, note that they measure performance relative to an approximated counterfactual, not an oracle with access to the true outcome of alternate spreads. This is a limitation of the simulator design, not the learning algorithms.

## Synthetic data for controlled experiments

Real-world data adds engineering overhead and noise without improving the core guarantees this project tests. The experiments target regret bounds, adaptation speed, and failure modes under controlled regime shifts, so the primary data source is a synthetic microstructure process with realistic statistical properties:

- Volatility clustering from a GARCH(1,1) true value process.
- Microstructure noise from a mean-reverting observed-price perturbation.
- Fat-tailed order sizes already captured by the power-law size distribution in `NoiseTrader`.

See [python/simulation/synthetic_lob.py](python/simulation/synthetic_lob.py) for the synthetic processes and price-path helper.

## Algorithms

- **Exp3MarketMaker**: Standard adversarial bandit update with importance-weighted rewards. Achieves the $2\sqrt{T K \ln K}$ regret bound with optimal learning rate $\gamma = \sqrt{\ln K / (KT)}$. Per-round regret decreases over time, verified empirically at T=3,000 rounds.
- **Exp3DoublingTrick**: Restarts Exp3 on doubling epochs to handle unknown horizons. Pays a 2x constant-factor penalty in the regret bound ($4\sqrt{T K \ln K}$ vs $2\sqrt{T K \ln K}$) but requires no horizon parameter.
- **SWExp3MarketMaker**: Sliding-window Exp3 that forgets old rewards to track regime shifts. Recovers the correct arm faster than standard Exp3 after regime changes, with adaptation time proportional to the window size W (default 200 rounds) rather than cumulative history.
- **EXP4MarketMaker**: Contextual bandit that mixes a pool of 8 experts (fixed-spread, volatility-scaled, inventory-aware, OFI-based, depth-imbalance, A-S heuristic, momentum). Selects the dominant expert per-regime and re-weights continuously.
- **AvellanedaStoikovMM**: Closed-form model-based spread from Avellaneda-Stoikov (2008) with MLE calibration for sigma and kappa. Outperforms Exp3 under stationary conditions but degrades measurably when parameters shift between regimes.
- **FixedSpreadMM**: Deterministic baseline used to anchor comparisons. One instance per spread arm (8 total), providing the counterfactual reference for regret computation.

## Order flow models

- **NoiseTrader**: Poisson arrivals (rate = 5.0/dt) with power-law order sizes (alpha = 1.5, range 1 to 50 lots). Orders are balanced buy/sell and do not respond to prices.
- **StochasticInformedTrader**: Trades when the true value exceeds the quoted bid or ask. Order size scales with the edge (value minus quote), capped at 5x the base size. Arrival rate is 20% of the noise rate by default.
- **AdversarialInformedTrader**: Observes the market maker's spread distribution and computes a best-response strategy. Alternates between exploit phases (large informed orders when the MM quotes tight, arrival rate 3x base) and bait phases (small balanced orders to pull the MM toward tighter spreads). This is the hardest adversary in the system and the primary stress test for Exp3.
- **TrueValueProcess**: GBM with optional Poisson jumps (prob = 0.01/dt, size = 0.5). Drives both informed trader types. Configurable drift, volatility, and jump parameters.

## Evaluation toolkit

- **RegretTracker**: Exact external regret vs the best fixed spread in hindsight, using per-round counterfactuals across all 8 arms. Includes rolling regret (window = 500), bootstrap confidence intervals (200 resamples, 95% CI), and per-spread regret breakdown.
- **RegretDecomposer**: Three-component economic decomposition into spread revenue, adverse selection loss, and inventory loss. Tracks cumulative PnL curves for each component separately, enabling diagnosis of where each algorithm's PnL comes from.
- **AdaptationSpeedAnalyzer**: Measures recovery time after regime changes and compares measured adaptation time to the $O(\log K / \Delta^2)$ theoretical prediction. Computes Pearson correlation between predicted and observed recovery times, with statistical significance testing.
- **Benchmarker**: Runs all 6 algorithms (plus 8 fixed-spread baselines) on a shared simulation factory and reports total reward, total regret, theoretical bound, bound ratio, annualized Sharpe ratio, max drawdown, and mean absolute inventory. Produces both a summary table and a Pandas DataFrame for further analysis.

## C++ engine notes

The matching engine is price-time priority with FIFO queues at each price level, implemented in ~760 lines of C++17. The order book uses `std::map` for price levels (O(log N) insert/lookup) and `std::deque` for time-priority queues at each level, with a hash-map order index for O(1) cancel lookups. It exposes:

- `submit_limit`, `submit_market`, and `cancel` for order handling.
- `best_bid`, `best_ask`, `mid_price`, `spread`, and `ofi` for state queries.
- `bid_levels`, `ask_levels`, `total_bid_volume`, `total_ask_volume` for multi-level depth.
- `vwap` (over last N fills) and `realized_volatility` (log-return std over a rolling window) as derived statistics.
- `fill_history` and `mid_price_history` (capped circular buffer) for diagnostics.

The pybind11 binding layer (129 lines) maps 15+ C++ methods to Python with automatic STL container conversion. The Python build exposes this as the `lob_engine` module and is used directly by the simulator. If the extension is unavailable, the simulator falls back to a minimal 62-line Python stub for light testing (slower and without depth/OFI features).

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

- `convergence_plots.py` -> convergence plots and `convergence_regret.csv`. Shows empirical regret curves for all algorithms overlaid against the $2\sqrt{T K \ln K}$ theoretical bound.
- `pnl_comparison.py` -> cumulative PnL comparison and `pnl_comparison.csv`. Side-by-side cumulative reward across all 6 algorithms and their fixed-spread baselines.
- `adaptation_plots.py` -> adaptation-speed plot and `adaptation_speed.csv`. Measures recovery time per regime change and overlays the $O(\log K / \Delta^2)$ prediction.
- `regret_decomposition_plots.py` -> PnL component plot and `regret_decomposition.csv`. Breaks down cumulative PnL into spread capture, adverse selection cost, and inventory loss.

The PNGs are saved alongside the CSVs: `convergence_kuhn.png`, `pnl_comparison.png`, `adaptation_speed.png`, and `regret_decomposition.png`.

## Reproducibility notes

- All components accept RNG seeds for deterministic runs (default seed = 42).
- The simulator records a full (T, K) counterfactual reward matrix for exact regret curves.
- Use the C++ engine for realistic LOB mechanics; the Python stub is only for lightweight testing.
- Results can be reproduced end-to-end with `python notebooks/run_everything.py`, which builds the extension, runs a 10,000-round benchmark, and regenerates all figures.

## Testing

The test suite covers 8 modules (955 lines of test code) across the full stack:

- **Convergence tests**: verify sublinear regret growth, bound compliance within 20% slack, and correct learning-rate formulas for Exp3.
- **LOB mechanics**: price-time priority matching, partial fills, cancellation correctness, and depth queries.
- **Regime adaptation**: SW-Exp3 adapts faster than standard Exp3 after regime changes, with lower post-change regret.
- **EXP4 expert pool**: validates expert quoting, weight concentration on the dominant expert, and distribution validity after extended runs.
- **Adversarial trader**: best-response computation, exploit/bait phase transitions, and regret impact decomposition.
- **A-S model**: reservation price calculation, optimal spread formula, and MLE sigma/kappa calibration.
- **Regret decomposition**: component accounting (spread + adverse selection + inventory = net PnL).
- **Numerical stability**: no NaN or Inf in distributions after 500+ rounds of concentrated updates.

## CI

GitHub Actions runs on every push and pull request. The pipeline builds the C++ extension from source with CMake, validates `lob_engine` import, and runs the full pytest suite. Build + test completes in under 2 minutes on the CI runner.
