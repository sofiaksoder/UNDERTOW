"""Stochastic volatility (SV) model and simulation.

    h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,   eta_t ~ N(0, 1)
    r_t = exp(h_t / 2) * eps_t,                            eps_t ~ N(0, 1)

h_t is the log-variance. Volatility is exp(h_t / 2), per day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


def annualize(daily_vol: np.ndarray | float) -> np.ndarray | float:
    """Daily volatility -> annualized, in percent."""
    return daily_vol * np.sqrt(TRADING_DAYS) * 100.0


@dataclass(frozen=True)
class SVParams:
    mu: float = -9.2  # long-run mean of h; exp(mu / 2) is about 1% daily vol
    phi: float = 0.97  # persistence
    sigma_eta: float = 0.15  # volatility of volatility

    @property
    def stationary_std(self) -> float:
        return self.sigma_eta / np.sqrt(1.0 - self.phi**2)


@dataclass(frozen=True)
class JumpParams:
    """Optional jump in the state equation: each day, with probability `prob`, h gets an extra
    N(mean, std^2) added. Makes a sudden large rise in volatility plausible under the prior."""
    prob: float = 0.02
    mean: float = 1.0
    std: float = 0.5

    @property
    def expected_exp(self) -> float:
        """E[exp(J)] for the jump J (1 when there is no jump)."""
        return (1.0 - self.prob) + self.prob * float(np.exp(self.mean + 0.5 * self.std**2))


@dataclass
class SVPath:
    h: np.ndarray  # log-variance
    returns: np.ndarray
    vol: np.ndarray  # true daily volatility, exp(h / 2)
    regime: np.ndarray | None = None  # 0 = calm, 1 = stressed (regime model only)


def simulate(
    n: int,
    params: SVParams = SVParams(),
    seed: int = 0,
    shocks: dict[int, float] | None = None,
) -> SVPath:
    """Simulate n days. `shocks` maps day -> amount added to h on that day."""
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    h = np.empty(n)
    h[0] = params.mu + params.stationary_std * eta[0]
    for t in range(1, n):
        h[t] = params.mu + params.phi * (h[t - 1] - params.mu) + params.sigma_eta * eta[t]
    h = _apply_shocks(h, params.phi, shocks)
    vol = np.exp(h / 2.0)
    return SVPath(h=h, returns=vol * eps, vol=vol)


def simulate_regime(
    n: int,
    calm: SVParams = SVParams(),
    stressed: SVParams = SVParams(mu=-8.0, phi=0.97, sigma_eta=0.25),
    p_calm_to_stressed: float = 0.01,
    p_stressed_to_calm: float = 0.04,
    seed: int = 0,
    shocks: dict[int, float] | None = None,
) -> SVPath:
    """SV model whose parameters switch between a calm and a stressed regime (Markov chain)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    eta = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    regime = np.zeros(n, dtype=int)
    h = np.empty(n)
    h[0] = calm.mu + calm.stationary_std * eta[0]
    for t in range(1, n):
        prev = regime[t - 1]
        flip = u[t] < (p_stressed_to_calm if prev == 1 else p_calm_to_stressed)
        regime[t] = 1 - prev if flip else prev
        p = stressed if regime[t] == 1 else calm
        h[t] = p.mu + p.phi * (h[t - 1] - p.mu) + p.sigma_eta * eta[t]
    h = _apply_shocks(h, calm.phi, shocks)
    vol = np.exp(h / 2.0)
    return SVPath(h=h, returns=vol * eps, vol=vol, regime=regime)


def _apply_shocks(h: np.ndarray, phi: float, shocks: dict[int, float] | None) -> np.ndarray:
    """Add each shock to h on its day and let it decay through the AR(1) persistence."""
    if not shocks:
        return h
    h = h.copy()
    extra = np.zeros(len(h))
    for day, size in shocks.items():
        extra[day] += size
    carry = 0.0
    for t in range(len(h)):
        carry = phi * carry + extra[t]
        h[t] += carry
    return h
