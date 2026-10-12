"""Error metrics.

On simulated data the true volatility is known, so RMSE / MAE against it are exact.
On real data true volatility is unobservable. There we score next-day volatility
forecasts against noisy proxies of what happened next:
  - QLIKE against the next day's squared return (robust to the proxy's noise, Patton 2011),
  - MAE against the realized vol of the next `horizon` days.
"""
from __future__ import annotations

import math

import numpy as np


def rmse(est: np.ndarray, truth: np.ndarray) -> float:
    m = np.isfinite(est) & np.isfinite(truth)
    return float(np.sqrt(np.mean((est[m] - truth[m]) ** 2)))


def mae(est: np.ndarray, truth: np.ndarray) -> float:
    m = np.isfinite(est) & np.isfinite(truth)
    return float(np.mean(np.abs(est[m] - truth[m])))


def forward_realized_vol(returns: np.ndarray, horizon: int = 5) -> np.ndarray:
    """Element t = sqrt(mean(r_{t+1..t+horizon}^2)). NaN where the window runs past the end."""
    r2 = np.asarray(returns) ** 2
    out = np.full(len(r2), np.nan)
    csum = np.cumsum(np.insert(r2, 0, 0.0))
    for t in range(len(r2) - horizon):
        out[t] = np.sqrt((csum[t + 1 + horizon] - csum[t + 1]) / horizon)
    return out


def qlike(vol_forecast: np.ndarray, next_returns: np.ndarray) -> float:
    """Mean of log(var) + r^2 / var. Lower is better."""
    var = np.asarray(vol_forecast) ** 2
    return float(np.mean(np.log(var) + np.asarray(next_returns) ** 2 / var))


def score_forecasts(returns: np.ndarray, forecasts: dict[str, np.ndarray],
                    mask: np.ndarray | None = None, horizon: int = 5) -> dict[str, dict[str, float]]:
    """Score next-day vol forecasts. forecasts[name][t] is made after seeing r_t, for day t+1.

    All methods are scored on the same days (where every method has a forecast).
    """
    r = np.asarray(returns)
    n = len(r)
    ok = np.ones(n, dtype=bool) if mask is None else mask.copy()
    for f in forecasts.values():
        ok &= np.isfinite(f)
    ok[-horizon:] = False  # need `horizon` future days for the proxy
    proxy = forward_realized_vol(r, horizon)
    idx = np.where(ok)[0]
    out = {}
    for name, f in forecasts.items():
        out[name] = {
            "qlike": qlike(f[idx], r[idx + 1]),
            f"mae_rv{horizon}": mae(f[idx], proxy[idx]),
            "n_days": int(len(idx)),
        }
    return out


def qlike_losses(vol_forecast: np.ndarray, next_returns: np.ndarray) -> np.ndarray:
    """Per-day QLIKE loss: log(var) + r^2 / var. Lower is better."""
    var = np.asarray(vol_forecast) ** 2
    return np.log(var) + np.asarray(next_returns) ** 2 / var


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int | None = None
                    ) -> tuple[float, float, float]:
    """Diebold-Mariano test of equal predictive accuracy for one-step forecasts.

    d_t = loss_a - loss_b. Returns (mean_d, z, two_sided_p). A negative mean_d means
    forecast A has the lower loss. The variance of mean(d) is a Newey-West (Bartlett)
    long-run variance with `lags` lags (default floor(T^(1/3))), because daily loss
    differences are autocorrelated.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if lags is None:
        lags = int(n ** (1 / 3))
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(dc[k:] @ dc[:-k]) / n
    se = np.sqrt(max(lrv, 1e-300) / n)
    z = float(d.mean() / se)
    p = float(math.erfc(abs(z) / math.sqrt(2.0)))
    return float(d.mean()), z, p
