"""
Convergence plots: regret curves vs theoretical bound.
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


def plot_regret_curves(
    results: Dict,
    spread_choices: List[float],
    T: int,
    include_bound: bool = True,
    ci: bool = True,
    save: bool = True,
) -> plt.Figure:
    """
    Plot cumulative regret curves for all algorithms.
    Overlays the theoretical 2*sqrt(T*K*ln(K)) bound.
    """
    K   = len(spread_choices)
    ts  = np.arange(1, T + 1)
    bound = 2.0 * np.sqrt(ts * K * np.log(K))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw cumulative regret.
    ax = axes[0]
    for name, r in results.items():
        if "Fixed" in name:
            continue
        curve = r.regret_curve
        if len(curve) == 0:
            continue
        ax.plot(curve, label=name, color=_color(name), lw=1.8)

    if include_bound:
        ax.plot(bound, "--", label="Theoretical bound", color=PALETTE["theoretical"],
                lw=2, zorder=10)

    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative Regret $R_t$")
    ax.set_title("Regret Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: regret / sqrt(T) (should flatten if bound is tight)
    ax2 = axes[1]
    sqrt_ts = np.sqrt(ts)
    for name, r in results.items():
        if "Fixed" in name:
            continue
        curve = r.regret_curve
        if len(curve) == 0:
            continue
        ax2.plot(curve / (sqrt_ts[:len(curve)] + 1e-8),
                 label=name, color=_color(name), lw=1.8)

    theo_flat = bound / sqrt_ts
    ax2.plot(theo_flat, "--", label="Bound / $\\sqrt{t}$",
             color=PALETTE["theoretical"], lw=2, zorder=10)
    ax2.set_xlabel("Round t")
    ax2.set_ylabel("$R_t / \\sqrt{t}$")
    ax2.set_title("Normalised Regret (should flatten)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Exp3 Regret Convergence  (K={K}, T={T})", fontsize=13)
    fig.tight_layout()
    if save:
        _save(fig, "regret_curves.pdf")
    return fig
