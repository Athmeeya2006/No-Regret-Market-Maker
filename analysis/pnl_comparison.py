"""Generate cumulative PnL comparison plots."""

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

from python.algorithms.avellaneda_stoikov import AvellanedaStoikovMM
from python.algorithms.exp3 import Exp3MarketMaker
from python.algorithms.exp4 import EXP4MarketMaker, FixedSpreadExpert, OFIExpert, VolatilityExpert


OUT_DIR = ROOT / "analysis"
SPREADS = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20])


def rewards_for_regime(T: int, regime_switch: bool) -> np.ndarray:
    rng = np.random.default_rng(21 if regime_switch else 20)
    base = rng.normal(0.0, 0.03, size=(T, len(SPREADS)))
    centers = np.full(T, 3)
    if regime_switch:
        centers[T // 2 :] = 6
    for t, center in enumerate(centers):
        distance = np.abs(np.arange(len(SPREADS)) - center)
        base[t] += 0.12 - 0.035 * distance
    return base


def nearest_arm(spread: float) -> int:
    return int(np.argmin(np.abs(SPREADS - spread)))


def run_exp3(rewards: np.ndarray) -> np.ndarray:
    mm = Exp3MarketMaker(SPREADS.tolist(), gamma=0.12)
    rng = np.random.default_rng(22)
    pnl = []
    for t in range(rewards.shape[0]):
        p = mm.get_distribution()
        idx = int(rng.choice(len(SPREADS), p=p))
        reward = float(rewards[t, idx])
        mm.record_choice(idx, p)
        mm.update(reward)
        pnl.append(reward)
    return np.cumsum(pnl)


def run_exp4(rewards: np.ndarray) -> np.ndarray:
    experts = [FixedSpreadExpert(0.02), FixedSpreadExpert(0.08), VolatilityExpert(), OFIExpert()]
    mm = EXP4MarketMaker(experts, gamma=0.1)
    pnl = []
    for t in range(rewards.shape[0]):
        ctx = EXP4MarketMaker.build_context(100, 0.02, 0.1, 0, 0.0002 + 0.00001 * t, 1.0)
        spread = mm.choose_spread(ctx)
        reward = float(rewards[t, nearest_arm(spread)])
        mm.update(reward)
        pnl.append(reward)
    return np.cumsum(pnl)


def run_as(rewards: np.ndarray) -> np.ndarray:
    mm = AvellanedaStoikovMM(sigma=0.01, kappa=1.5, gamma=0.1)
    pnl = []
    for t in range(rewards.shape[0]):
        spread = mm.choose_spread({"mid_price": 100.0, "inventory": 0, "time_remaining": 1.0})
        pnl.append(float(rewards[t, nearest_arm(spread)]))
    return np.cumsum(pnl)


def run_fixed(rewards: np.ndarray, spread: float = 0.05) -> np.ndarray:
    return np.cumsum(rewards[:, nearest_arm(spread)])


def main() -> None:
    T = 900
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, title, switching in [
        (axes[0], "stationary", False),
        (axes[1], "regime switching", True),
    ]:
        rewards = rewards_for_regime(T, switching)
        curves = {
            "Exp3": run_exp3(rewards),
            "EXP4": run_exp4(rewards),
            "A-S": run_as(rewards),
            "Fixed": run_fixed(rewards),
        }
        for name, curve in curves.items():
            ax.plot(curve, label=name)
            rows.extend(
                {"condition": title, "algorithm": name, "t": i + 1, "cum_pnl": value}
                for i, value in enumerate(curve)
            )
        if switching:
            ax.axvline(T // 2, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cumulative PnL")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pnl_comparison.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_DIR / "pnl_comparison.csv", index=False)


if __name__ == "__main__":
    main()
