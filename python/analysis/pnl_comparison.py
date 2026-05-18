"""
PnL comparison plots: cumulative PnL and spread distribution.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/CI
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---- consistent colour palette across all plots ----------------
PALETTE = {
    "Exp3":               "#1f77b4",
    "Exp3-DoublingTrick": "#ff7f0e",
    "SW-Exp3(W=200)":     "#2ca02c",
    "SW-Exp3(W=500)":     "#d62728",
    "EXP4":               "#9467bd",
    "AvellanedaStoikov":  "#8c564b",
    "theoretical":        "#7f7f7f",
    "Fixed":              "#bcbd22",
}

def _color(name: str) -> str:
    for key, col in PALETTE.items():
        if key in name:
            return col
    return "#17becf"

def _save(fig, fname: str, dpi: int = 150):
    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_pnl_comparison(
    results:     Dict,
    save:        bool = True,
) -> plt.Figure:
    """Cumulative PnL for all algorithms on a single axis."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, r in sorted(results.items(),
                           key=lambda x: -x[1].total_reward):
        cum = np.cumsum(r.reward_series)
        lw  = 2.5 if "Fixed" not in name else 1.0
        ls  = "-"  if "Fixed" not in name else "--"
        ax.plot(cum, label=f"{name} ({r.total_reward:+.0f})",
                color=_color(name), lw=lw, ls=ls)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Cumulative PnL: All Algorithms")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        _save(fig, "pnl_comparison.pdf")
    return fig


def plot_spread_distribution(
    results: Dict,
    spread_choices: List[float],
    save: bool = True,
) -> plt.Figure:
    """Bar chart of how often each spread was chosen per algorithm."""
    K    = len(spread_choices)
    algos = [n for n in results if "Fixed" not in n and
             hasattr(results[n], "spread_series")]

    fig, axes = plt.subplots(1, len(algos), figsize=(4 * len(algos), 4),
                              sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, name in zip(axes, algos):
        r    = results[name]
        s    = r.spread_series
        bins = np.array(spread_choices)
        counts = np.array([np.sum(np.isclose(s, b, atol=1e-3)) for b in bins])
        ax.bar(range(K), counts / len(s), color=_color(name))
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"{b:.3f}" for b in bins], rotation=45, fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Spread")
        if ax is axes[0]:
            ax.set_ylabel("Fraction of rounds")

    fig.suptitle("Spread Choice Distribution", fontsize=12)
    fig.tight_layout()
    if save:
        _save(fig, "spread_distribution.pdf")
    return fig
