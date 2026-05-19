"""Synthetic microstructure processes for controlled experiments.

Provides:
- GARCHTrueValueProcess for volatility clustering.
- MicrostructureNoise for mean-reverting observed-price noise.
- generate_observed_prices helper to build mid-price series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class GARCHState:
    variance: float
    last_return: float


class GARCHTrueValueProcess:
    """True value process with GARCH(1,1) volatility clustering."""

    def __init__(
        self,
        initial_value: float = 100.0,
        drift: float = 0.0,
        omega: float = 1e-6,
        alpha: float = 0.08,
        beta: float = 0.9,
        seed: Optional[int] = None,
    ):
        self.V0 = float(initial_value)
        self.V = float(initial_value)
        self.drift = float(drift)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)

        if self.alpha + self.beta < 1.0:
            long_run = self.omega / max(1.0 - self.alpha - self.beta, 1e-8)
        else:
            long_run = max(self.omega, 1e-8)
        self.state = GARCHState(variance=float(long_run), last_return=0.0)
        self.history: List[float] = [self.V]

    def step(self, dt: float = 1.0) -> float:
        z = self.rng.standard_normal()
        variance = (
            self.omega
            + self.alpha * (self.state.last_return ** 2)
            + self.beta * self.state.variance
        )
        variance = max(variance, 1e-12)
        ret = self.drift * dt + np.sqrt(variance * dt) * z

        self.state = GARCHState(variance=variance, last_return=ret)
        self.V = max(self.V * np.exp(ret), 0.01)
        self.history.append(self.V)
        return self.V

    def value(self) -> float:
        return self.V

    def reset(self):
        self.V = self.V0
        if self.alpha + self.beta < 1.0:
            long_run = self.omega / max(1.0 - self.alpha - self.beta, 1e-8)
        else:
            long_run = max(self.omega, 1e-8)
        self.state = GARCHState(variance=float(long_run), last_return=0.0)
        self.history = [self.V]


class MicrostructureNoise:
    """Mean-reverting noise applied to observed prices."""

    def __init__(
        self,
        reversion: float = 0.15,
        sigma: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.reversion = float(reversion)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.state = 0.0

    def step(self, dt: float = 1.0) -> float:
        eps = self.rng.standard_normal()
        self.state = (1.0 - self.reversion * dt) * self.state + self.sigma * np.sqrt(dt) * eps
        return self.state

    def apply(self, true_price: float, dt: float = 1.0) -> float:
        noise = self.step(dt)
        return max(true_price + noise, 0.01)


def generate_observed_prices(
    n_steps: int,
    dt: float = 1.0,
    value_process: Optional[GARCHTrueValueProcess] = None,
    noise_process: Optional[MicrostructureNoise] = None,
) -> np.ndarray:
    """Generate a synthetic observed mid-price series."""
    if value_process is None:
        value_process = GARCHTrueValueProcess()
    if noise_process is None:
        noise_process = MicrostructureNoise()

    out = np.zeros(n_steps, dtype=float)
    for i in range(n_steps):
        true_value = value_process.step(dt)
        out[i] = noise_process.apply(true_value, dt)
    return out
