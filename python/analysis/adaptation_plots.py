"""
Adaptation plots: adaptation speed, regime overlay, regret decomposition,
weight evolution (Exp3 and EXP4).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/CI
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

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


def plot_adaptation_speed(
    algo_events:    Dict[str, list],   # {name: [AdaptationEvent]}
    save:           bool = True,
) -> plt.Figure:
    """
    Two panels:
    Left:  measured adaptation time per regime change (time-series)
    Right: measured vs theoretical O(log K / delta^2) scatter
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: adaptation times over regime changes
    ax = axes[0]
    for name, events in algo_events.items():
        times = [e.adaptation_time if e.adaptation_time else 1000
                 for e in events]
        ax.plot(times, "o-", label=name, color=_color(name), lw=1.5, ms=5)
    ax.set_xlabel("Regime change #")
    ax.set_ylabel("Adaptation time (rounds)")
    ax.set_title("Adaptation Time per Regime Change")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: measured vs theoretical scatter
    ax2 = axes[1]
    for name, events in algo_events.items():
        valid = [e for e in events if e.adaptation_time is not None]
        if not valid:
            continue
        measured    = [e.adaptation_time   for e in valid]
        theoretical = [e.theoretical_pred  for e in valid]
        ax2.scatter(theoretical, measured, label=name, color=_color(name),
                    alpha=0.7, s=40)

    # Reference line y = x
    all_vals = []
    for events in algo_events.values():
        all_vals.extend([e.theoretical_pred for e in events if e.adaptation_time])
    if all_vals:
        lim = (0, max(all_vals) * 1.1)
        ax2.plot(lim, lim, "k--", lw=1, label="y = x (perfect prediction)")
        ax2.set_xlim(lim)
        ax2.set_ylim(0, max(
            e.adaptation_time for ev in algo_events.values()
            for e in ev if e.adaptation_time
        ) * 1.1)

    ax2.set_xlabel("Theoretical prediction  $O(\\log K / \\delta^2)$")
    ax2.set_ylabel("Measured adaptation time")
    ax2.set_title("Measured vs Theoretical Adaptation")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Adaptation Speed Analysis", fontsize=13)
    fig.tight_layout()
    if save:
        _save(fig, "adaptation_plots.pdf")
    return fig


def plot_regime_overlay(
    reward_dict:   Dict[str, np.ndarray],
    regime_gen,
    n_rounds:      int,
    save:          bool = True,
) -> plt.Figure:
    """
    Plot per-round rewards with regime boundaries highlighted.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ts = np.arange(n_rounds)

    # Rolling mean for readability
    W = 100
    for name, rewards in reward_dict.items():
        if "Fixed" in name:
            continue
        roll = np.convolve(rewards, np.ones(W) / W, mode="same")
        ax.plot(roll, label=name, color=_color(name), lw=1.5)

    # Shade regime regions
    change_times = regime_gen.get_change_times(n_rounds)
    boundaries   = [0] + change_times + [n_rounds]
    colors_bg    = ["#e8f4f8", "#fff3e0", "#e8f5e9", "#fce4ec"]
    n_regimes    = len(regime_gen.regimes)

    for i in range(len(boundaries) - 1):
        regime_idx = i % n_regimes
        ax.axvspan(boundaries[i], boundaries[i + 1],
                   alpha=0.15, color=colors_bg[regime_idx % len(colors_bg)],
                   label=f"Regime: {regime_gen.regimes[regime_idx]['name']}"
                         if i < n_regimes else "")

    for ct in change_times:
        ax.axvline(ct, color="black", lw=1, ls="--", alpha=0.5)

    ax.set_xlabel("Round t")
    ax.set_ylabel(f"Rolling reward (window={W})")
    ax.set_title("Per-Round Rewards with Regime Changes")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if save:
        _save(fig, "regime_overlay.pdf")
    return fig


def plot_regret_decomposition(
    decomposer,    # RegretDecomposer instance
    algo_name: str = "Exp3",
    save:      bool = True,
) -> plt.Figure:
    """
    Stacked area chart of cumulative PnL components:
      spread revenue, adverse selection loss, inventory loss.
    """
    curves = decomposer.component_curves()
    T      = len(curves["net"])
    ts     = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: stacked area
    ax = axes[0]
    ax.fill_between(ts, 0, curves["spread"],
                    alpha=0.4, color="green",  label="Spread revenue")
    ax.fill_between(ts, 0, curves["adverse_selection"],
                    alpha=0.4, color="red",    label="Adverse selection")
    ax.fill_between(ts, 0, curves["inventory"],
                    alpha=0.4, color="orange", label="Inventory loss")
    ax.plot(ts, curves["net"], color="black", lw=2, label="Net PnL")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative PnL component")
    ax.set_title(f"Regret Decomposition: {algo_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: pie chart of total magnitudes
    ax2 = axes[1]
    s    = decomposer.summary()
    vals = [
        max(s["spread_revenue"],    0),
        max(s["adverse_selection"], 0),
        max(s["inventory_loss"],    0),
    ]
    lbls   = ["Spread\nRevenue", "Adverse\nSelection", "Inventory\nLoss"]
    colors = ["#2ca02c", "#d62728", "#ff7f0e"]
    if sum(vals) > 0:
        ax2.pie(vals, labels=lbls, colors=colors, autopct="%1.1f%%",
                startangle=90)
    ax2.set_title("Component Magnitudes")

    fig.suptitle("Regret Decomposition (CFR Analogy)", fontsize=13)
    fig.tight_layout()
    if save:
        _save(fig, "regret_decomposition.pdf")
    return fig


def plot_weight_evolution(
    dist_history:  List[np.ndarray],
    spread_choices: List[float],
    regime_gen=None,
    n_rounds: int = 0,
    save:     bool = True,
) -> plt.Figure:
    """
    Heatmap of Exp3 arm probabilities over time.
    Rows = arms (spreads), columns = time.
    """
    if not dist_history:
        return None

    mat = np.array(dist_history).T    # (K, T)
    K, T = mat.shape

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"{s:.3f}" for s in spread_choices])
    ax.set_xlabel("Round t")
    ax.set_ylabel("Spread arm")
    ax.set_title("Exp3 Arm Probability Heatmap")
    plt.colorbar(im, ax=ax, label="Probability")

    if regime_gen is not None and n_rounds > 0:
        for ct in regime_gen.get_change_times(n_rounds):
            if ct < T:
                ax.axvline(ct, color="white", lw=1.5, ls="--", alpha=0.8)

    fig.tight_layout()
    if save:
        _save(fig, "weight_evolution.pdf")
    return fig


def plot_exp4_weights(
    dist_history:  List[np.ndarray],
    expert_names:  List[str],
    regime_gen=None,
    n_rounds:      int = 0,
    save:          bool = True,
) -> plt.Figure:
    """Line plot of EXP4 expert weights over time."""
    if not dist_history:
        return None

    mat = np.array(dist_history)   # (T, N)
    T, N = mat.shape

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, name in enumerate(expert_names):
        ax.plot(mat[:, i], label=name, lw=1.5)

    if regime_gen is not None and n_rounds > 0:
        for ct in regime_gen.get_change_times(n_rounds):
            if ct < T:
                ax.axvline(ct, color="black", lw=1, ls="--", alpha=0.5)

    ax.set_xlabel("Round t")
    ax.set_ylabel("Expert weight")
    ax.set_title("EXP4 Expert Weight Evolution")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        _save(fig, "exp4_weights.pdf")
    return fig
