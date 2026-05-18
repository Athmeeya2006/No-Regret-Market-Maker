"""Generate regret-convergence plots for Exp3 variants."""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python.algorithms.exp3 import Exp3DoublingTrick, Exp3MarketMaker, SWExp3MarketMaker


OUT_DIR = ROOT / "analysis"
SPREADS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]


def reward_matrix(T: int, K: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rewards = rng.uniform(0.25, 0.75, size=(T, K))
    rewards[:, 2] += 0.16
    return np.clip(rewards, 0.0, 1.0)


def run_algorithm(mm, rewards: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for t in range(rewards.shape[0]):
        p = mm.get_distribution()
        idx = int(rng.choice(rewards.shape[1], p=p))
        mm.last_idx = idx
        mm.last_p = float(p[idx])
        mm.update(float(rewards[t, idx]))
    actual = np.cumsum(mm.reward_history)
    best = np.cumsum(rewards, axis=0).max(axis=1)
    return np.maximum(0.0, best[: len(actual)] - actual)


def main() -> None:
    T = 1500
    K = len(SPREADS)
    rewards = reward_matrix(T, K)
    curves = {
        "Exp3": run_algorithm(Exp3MarketMaker(SPREADS, T=T), rewards, 10),
        "Exp3DoublingTrick": run_algorithm(Exp3DoublingTrick(SPREADS), rewards, 11),
        "SWExp3": run_algorithm(SWExp3MarketMaker(SPREADS, window=150, gamma=0.18), rewards, 12),
    }
    ts = np.arange(1, T + 1)
    bound = 2.0 * np.sqrt(ts * K * np.log(K))

    frame = pd.DataFrame({"t": ts, "bound": bound, **curves})
    frame.to_csv(OUT_DIR / "convergence_regret.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, curve in curves.items():
        axes[0].plot(ts, curve, label=name)
        axes[1].plot(ts, curve, label=name)
    for ax in axes:
        ax.plot(ts, bound, color="black", linestyle="--", label="2 sqrt(T K log K)")
        ax.set_xlabel("round")
        ax.set_ylabel("cumulative regret")
        ax.grid(alpha=0.25)
    axes[0].set_title("linear scale")
    axes[1].set_title("log scale")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "convergence_kuhn.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
