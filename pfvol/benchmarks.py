"""Baseline volatility estimators to compare the filter against. All return daily vol."""
from __future__ import annotations

import numpy as np


def rolling_std(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling sample std of returns (zero-mean). NaN until the window is full."""
    out = np.full(len(returns), np.nan)
    sq = np.asarray(returns) ** 2
    csum = np.cumsum(np.insert(sq, 0, 0.0))
    out[window - 1:] = np.sqrt((csum[window:] - csum[:-window]) / window)
    return out


def ewma_vol(returns: np.ndarray, lam: float = 0.94, warmup: int = 20) -> np.ndarray:
    """RiskMetrics EWMA: var_t = lam * var_{t-1} + (1 - lam) * r_t^2."""
    r = np.asarray(returns)
    var = np.empty(len(r))
    var[0] = np.mean(r[:warmup] ** 2)
    for t in range(len(r)):
        prev = var[t - 1] if t > 0 else var[0]
        var[t] = lam * prev + (1 - lam) * r[t] ** 2
    return np.sqrt(var)


def garch_vol(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a zero-mean GARCH(1,1) with `arch` on the full sample.

    Returns (vol, next_vol): conditional vol for day t (uses r up to t-1) and the
    one-step-ahead forecast for day t+1 made after seeing r_t.
    """
    from arch import arch_model

    scale = 100.0  # arch is better conditioned with returns in percent
    res = arch_model(np.asarray(returns) * scale, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
    cond = np.asarray(res.conditional_volatility)
    omega, alpha, beta = (res.params[k] for k in ("omega", "alpha[1]", "beta[1]"))
    next_var = omega + alpha * (np.asarray(returns) * scale) ** 2 + beta * cond**2
    return cond / scale, np.sqrt(next_var) / scale


def garch_params(train_returns: np.ndarray) -> dict[str, float]:
    """Fit a zero-mean GARCH(1,1) on `train_returns` with `arch`; return omega, alpha, beta in
    raw (not percent) return units, so the recursion below is plain arithmetic."""
    from arch import arch_model

    scale = 100.0
    res = arch_model(np.asarray(train_returns) * scale, mean="Zero", vol="GARCH", p=1, q=1).fit(
        disp="off")
    return {"omega": float(res.params["omega"]) / scale**2, "alpha": float(res.params["alpha[1]"]),
            "beta": float(res.params["beta[1]"])}


def garch_next_vol(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """One-step-ahead GARCH vol: element t is sigma_{t+1} given returns up to t.

    sigma^2_{t+1} = omega + alpha r_t^2 + beta sigma^2_t, started from the unconditional variance.
    """
    r = np.asarray(returns)
    persistence = alpha + beta
    var_t = omega / (1.0 - persistence) if persistence < 1.0 else float(np.mean(r[:20] ** 2))
    out = np.empty(len(r))
    for t in range(len(r)):
        var_t = omega + alpha * r[t] ** 2 + beta * var_t
        out[t] = np.sqrt(var_t)
    return out
