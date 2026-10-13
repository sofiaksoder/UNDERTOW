"""Volatility-targeting backtest.

A long-only position in one asset, scaled each day so its expected volatility is a fixed target:

    w = clip(target_daily_vol / vol_forecast, 0, cap)

The only thing that differs between strategies is the volatility forecast. Timing is strict:
`vol_forecast[t]` is made after seeing r_t (every forecaster in this repo is built that way), the weight
it sets is held over day t+1, and the cost of moving to it is charged on day t+1.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def target_weights(vol_forecast: np.ndarray, target_ann: float = 0.10, cap: float = 1.5) -> np.ndarray:
    """Position sizes from daily-vol forecasts. NaN forecasts stay NaN (no position decided)."""
    target_daily = target_ann / np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(target_daily / np.asarray(vol_forecast, dtype=float), 0.0, cap)


def positions_from_forecasts(vol_forecast: np.ndarray, target_ann: float = 0.10, cap: float = 1.5
                             ) -> np.ndarray:
    """Weight held over each day: element i uses the forecast made after day i-1. Element 0 is NaN."""
    w = target_weights(vol_forecast, target_ann, cap)
    pos = np.full(len(w), np.nan)
    pos[1:] = w[:-1]
    return pos


def run_strategy(simple_returns: np.ndarray, positions: np.ndarray, cost_bps: float = 5.0,
                 initial_position: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Daily net returns and daily turnover for the positions held over each day.

    Cost = cost_bps / 1e4 per unit of turnover |w_t - w_{t-1}|; the first day trades from
    `initial_position`. The uninvested part of the capital earns zero.
    """
    r = np.asarray(simple_returns, dtype=float)
    w = np.asarray(positions, dtype=float)
    prev = np.concatenate([[initial_position], w[:-1]])
    turnover = np.abs(w - prev)
    return w * r - cost_bps / 1e4 * turnover, turnover


def summarize(net_returns: np.ndarray, positions: np.ndarray, turnover: np.ndarray) -> dict[str, float]:
    """Annualised return / vol, Sharpe (zero risk-free rate), max drawdown, turnover, average leverage."""
    r = np.asarray(net_returns, dtype=float)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.insert(equity, 0, 1.0))[1:]
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann_ret = float(r.mean() * TRADING_DAYS)
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else float("nan"),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "turnover_per_year": float(np.sum(turnover) / len(r) * TRADING_DAYS),
        "avg_leverage": float(np.mean(positions)),
        "total_return": float(equity[-1] - 1.0),
    }


def sharpe(r: np.ndarray) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, mean_block: int = 20, reps: int = 4000,
                          seed: int = 0) -> tuple[float, float, float, float]:
    """Stationary block bootstrap (Politis-Romano) of Sharpe(a) - Sharpe(b) on paired days.

    Returns (observed difference, 2.5%, 97.5% quantiles of the bootstrap distribution, two-sided
    p-value for a zero difference from the bootstrap sign proportion).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    diffs = np.empty(reps)
    for k in range(reps):
        starts = rng.integers(0, n, size=n)
        new_block = rng.random(n) < p
        new_block[0] = True
        idx = np.empty(n, dtype=int)
        cur = 0
        for i in range(n):
            cur = starts[i] if new_block[i] else (cur + 1) % n
            idx[i] = cur
        diffs[k] = sharpe(a[idx]) - sharpe(b[idx])
    obs = sharpe(a) - sharpe(b)
    centred = diffs - diffs.mean()
    pval = float(np.mean(np.abs(centred) >= abs(obs)))
    return obs, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), pval
