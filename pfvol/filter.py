"""Bootstrap particle filter for the SV model (NumPy only).

Each day: propagate particles through the state equation, weight them by the
likelihood of the return, then resample if the effective sample size (ESS)
falls below `ess_threshold * N`. Weights are kept in log space.

Two optional robustness variants (off by default, so the plain model is unchanged):
- `jump`: a jump component in the state equation (see JumpParams).
- `obs_dof`: Student-t observation noise (unit variance) instead of Gaussian.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pfvol.model import JumpParams, SVParams

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class FilterResult:
    mean_vol: np.ndarray  # filtered posterior mean of daily vol, E[exp(h_t/2) | r_1:t]
    quantiles: np.ndarray  # (T, 5) daily vol quantiles at 5/25/50/75/95%
    ess: np.ndarray  # effective sample size after weighting, before resampling
    resampled: np.ndarray  # True on days where resampling happened
    pred_var: np.ndarray  # forecast of next-day variance given r_1:t
    prior_mean: np.ndarray  # mean daily vol after predict, before weighting
    loglik_t: np.ndarray  # log p(r_t | r_1:t-1), one value per day
    loglik: float  # log-likelihood estimate of the whole series


def normalize_log_weights(logw: np.ndarray) -> tuple[np.ndarray, float]:
    """Return normalized weights and log(sum(exp(logw)))."""
    m = np.max(logw)
    if not np.isfinite(m):
        raise FloatingPointError("all particle weights are zero")
    s = np.exp(logw - m)
    total = s.sum()
    return s / total, float(m + np.log(total))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return N parent indices drawn with systematic resampling."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # guard against rounding
    return np.searchsorted(cdf, positions)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) - 0.5 * weights[order]
    return np.interp(qs, cdf, values[order])


def gaussian_logpdf(x: float, h: np.ndarray) -> np.ndarray:
    """log N(x; 0, exp(h)) for each particle."""
    return -0.5 * (h + x * x * np.exp(-h) + _LOG_2PI)


def student_t_logpdf(x: float, h: np.ndarray, dof: float) -> np.ndarray:
    """log density of x = exp(h/2) * e, where e is Student-t with `dof` degrees of freedom,
    scaled to unit variance (needs dof > 2)."""
    c = math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2) - 0.5 * math.log(math.pi * (dof - 2))
    return c - 0.5 * h - 0.5 * (dof + 1) * np.log1p(x * x * np.exp(-h) / (dof - 2))


def particle_filter(
    returns: np.ndarray,
    params: SVParams = SVParams(),
    n_particles: int = 1000,
    seed: int = 0,
    ess_threshold: float = 0.5,
    compute_quantiles: bool = True,
    obs_dof: float | None = None,
    jump: JumpParams | None = None,
) -> FilterResult:
    if obs_dof is not None and obs_dof <= 2:
        raise ValueError("obs_dof must be above 2 (unit-variance Student-t)")
    rng = np.random.default_rng(seed)
    n_days = len(returns)
    n = n_particles
    mu, phi, sig = params.mu, params.phi, params.sigma_eta
    jump_factor = 1.0 if jump is None else jump.expected_exp

    mean_vol = np.empty(n_days)
    quantiles = np.full((n_days, len(QUANTILES)), np.nan)
    ess = np.empty(n_days)
    resampled = np.zeros(n_days, dtype=bool)
    pred_var = np.empty(n_days)
    prior_mean = np.empty(n_days)
    loglik_t = np.empty(n_days)

    h = mu + params.stationary_std * rng.standard_normal(n)  # draw from the stationary law
    logw = np.full(n, -np.log(n))
    for t in range(n_days):
        if t > 0:
            h = mu + phi * (h - mu) + sig * rng.standard_normal(n)
            if jump is not None:
                hit = rng.random(n) < jump.prob
                h = h + hit * (jump.mean + jump.std * rng.standard_normal(n))
        prior_mean[t] = np.exp(logw) @ np.exp(h / 2.0)
        r = returns[t]
        loglik = gaussian_logpdf(r, h) if obs_dof is None else student_t_logpdf(r, h, obs_dof)
        logw = logw + loglik
        w, log_incr = normalize_log_weights(logw)
        loglik_t[t] = log_incr  # previous weights sum to 1, so this is log p(r_t | r_1:t-1)
        logw = np.log(w)

        vol = np.exp(h / 2.0)
        mean_vol[t] = w @ vol
        if compute_quantiles:
            quantiles[t] = weighted_quantiles(vol, w, QUANTILES)
        ess[t] = 1.0 / np.sum(w * w)
        # E[exp(h_{t+1})] for h_{t+1} ~ N(mu + phi (h_t - mu), sig^2), plus the jump term
        pred_var[t] = jump_factor * (w @ np.exp(mu + phi * (h - mu) + 0.5 * sig * sig))

        if ess[t] < ess_threshold * n:
            h = h[systematic_resample(w, rng)]
            logw = np.full(n, -np.log(n))
            resampled[t] = True

    return FilterResult(mean_vol, quantiles, ess, resampled, pred_var, prior_mean,
                        loglik_t, float(loglik_t.sum()))
