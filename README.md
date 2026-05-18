# No-Regret Market Making Engine

This project studies no-regret learning for market making. It implements a C++ limit order book with Python bindings, Exp3/EXP4 market makers, an Avellaneda-Stoikov benchmark, simulated order flow, regime shifts, regret tracking, and CI.

## What It Builds

- C++17 limit order book and matching engine with price-time priority.
- Python bindings through pybind11 as `lob_engine`.
- Exp3 market maker for adversarial spread selection.
- Exp3 with the doubling trick for unknown horizons.
- Sliding-window Exp3 for regime-switching markets.
- EXP4 contextual market maker with expert advice.
- Fixed-spread and Avellaneda-Stoikov baselines.
- Noise, stochastic informed, and adversarial informed traders.
- Regime generator for abrupt or gradual market shifts.
- Regret tracker, regret decomposition, and adaptation-speed metrics.
- Binance data pipeline for optional real market data experiments.
- Mathematical writeups connecting Exp3, lower bounds, and CFR.

## Project Layout

```text
cpp/
  lob/                 C++ order book and matching engine
  bindings/            pybind11 module definition
python/
  algorithms/          Exp3, EXP4, A-S, fixed spread, trader logic
  simulation/          simulator, regimes, data pipeline
  evaluation/          regret, decomposition, adaptation, benchmark tools
  analysis/            plotting helpers
tests/                 unit tests for engine, algorithms, and metrics
writeup/               LaTeX math notes
notebooks/             runnable experiment entrypoint
config.yaml            default experiment configuration
```

## Mathematical Idea

At each round `t`, the market maker chooses one spread from `K` discrete spread choices. The environment then produces order flow, fills, inventory changes, and a reward:

```text
reward = PnL from spread capture + inventory PnL - inventory risk penalty
```

Exp3 maintains a probability distribution over spreads. It samples a spread, observes only the reward for that spread, and performs an importance-weighted exponential update. Its external regret is:

```text
R_T = max_k sum_t r_{t,k} - sum_t r_{t,I_t}
```

With the usual Exp3 learning rate, the expected regret satisfies:

```text
E[R_T] <= 2 sqrt(T K log K)
```

That guarantee is model-free: it does not assume GBM prices, Poisson arrivals, calibrated intensities, or stationary regimes. This is the same philosophical guarantee as CFR in poker: time-average regret goes to zero, so the strategy becomes difficult to exploit in the relevant game model.

## Benchmarks

The intended comparison is:

1. Fixed-spread market makers as simple baselines.
2. Avellaneda-Stoikov as a model-based stochastic-control benchmark.
3. Exp3/SW-Exp3 as model-free no-regret market makers.
4. EXP4 as a contextual extension using volatility, inventory, OFI, depth, momentum, and A-S experts.

The main comparison is performance under stationary flow versus regimes where A-S assumptions are misspecified: adversarial informed flow, changing volatility, shifting informed-trader fractions, and regime switches.

## Setup

Use Python 3.12 or newer with a C++17 compiler.

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install -e .
```

Verify the C++ extension:

```bash
venv/bin/python -c "import lob_engine; print(lob_engine.MatchingEngine().mid_price())"
```

## Tests

```bash
venv/bin/python -m pytest tests -q
```

The network download test for Binance data is skipped by default. Everything else should run locally, including the C++ extension tests.

## Analysis Outputs

The top-level `analysis/` scripts generate:

- `analysis/convergence_kuhn.png`
- `analysis/pnl_comparison.png`
- `analysis/adaptation_speed.png`
- `analysis/regret_decomposition.png`

## CI

GitHub Actions is configured in `.github/workflows/ci.yml`. On every push and pull request it:

- installs Python dependencies,
- builds the C++/pybind11 extension with `pip install -e .`,
- verifies that `lob_engine` imports,
- runs the full pytest suite.

## Current Status

The local suite passes with the compiled engine:

```text
53 passed, 1 skipped
```

The skipped test is the Binance download test, which intentionally requires network access.
