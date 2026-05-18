"""Generate PnL component decomposition plots."""

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

from python.evaluation.regret_decomposer import RegretDecomposer


OUT_DIR = ROOT / "analysis"


def main() -> None:
    rng = np.random.default_rng(41)
    decomp = RegretDecomposer()
    n = 600
    for _ in range(n):
        informed = bool(rng.random() < 0.28)
        inventory_change = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        mid_move = float(rng.normal(0.0, 0.015 if informed else 0.006))
        spread_earned = float(rng.uniform(0.01, 0.05) * abs(inventory_change))
        decomp.record_trade(spread_earned, informed, inventory_change, mid_move)

    curves = decomp.component_curves()
    summary = decomp.summary()
    frame = pd.DataFrame({"t": np.arange(n), **curves})
    frame.to_csv(OUT_DIR / "regret_decomposition.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    t = frame["t"].to_numpy()
    spread = frame["spread"].to_numpy()
    adverse = -frame["adverse_selection"].to_numpy()
    inventory = -frame["inventory"].to_numpy()
    ax.stackplot(
        t,
        spread,
        -adverse,
        -inventory,
        labels=["spread revenue", "adverse selection loss", "inventory loss"],
        colors=["#2ca02c", "#d62728", "#ff7f0e"],
        alpha=0.72,
    )
    ax.plot(t, frame["net"], color="black", linewidth=1.8, label="net PnL")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("trade")
    ax.set_ylabel("cumulative component value")
    ax.set_title(
        "PnL decomposition "
        f"(net={summary['net_pnl']:.2f}, informed={summary['informed_fraction']:.2f})"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "regret_decomposition.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
