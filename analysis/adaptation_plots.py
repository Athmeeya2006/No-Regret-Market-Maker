"""Generate adaptation-speed plots around a regime change."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def adaptation_time(rewards: np.ndarray, change_time: int, window: int, epsilon: float) -> int | None:
    baseline = float(np.mean(rewards[change_time - window : change_time]))
    for t in range(change_time + 1, len(rewards)):
        post = float(np.mean(rewards[max(change_time, t - window + 1) : t + 1]))
        if abs(post - baseline) <= epsilon * max(abs(baseline), 1e-8):
            return t - change_time
    return None


def main() -> None:
    rng = np.random.default_rng(31)
    T = 800
    change = 400
    window = 35
    algorithms = {
        "Exp3": np.r_[rng.normal(0.10, 0.03, change), rng.normal(0.03, 0.04, 130), rng.normal(0.09, 0.03, T - change - 130)],
        "SWExp3": np.r_[rng.normal(0.10, 0.03, change), rng.normal(0.04, 0.04, 45), rng.normal(0.10, 0.03, T - change - 45)],
        "A-S": np.r_[rng.normal(0.11, 0.03, change), rng.normal(0.02, 0.05, T - change)],
        "Fixed": np.r_[rng.normal(0.08, 0.03, change), rng.normal(0.07, 0.03, T - change)],
    }

    rows = []
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for name, rewards in algorithms.items():
        roll = rolling_mean(rewards, window)
        at = adaptation_time(rewards, change, window, epsilon=0.18)
        ax.plot(roll, label=f"{name} (adapt={at})")
        if at is not None:
            ax.annotate(str(at), xy=(change + at, roll[change + at]), fontsize=8)
        rows.append({"algorithm": name, "change_time": change, "adaptation_time": at})
    ax.axvline(change, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(change - 180, change + 260)
    ax.set_xlabel("round")
    ax.set_ylabel(f"rolling reward, window={window}")
    ax.set_title("adaptation around regime change")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "adaptation_speed.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_DIR / "adaptation_speed.csv", index=False)


if __name__ == "__main__":
    main()
