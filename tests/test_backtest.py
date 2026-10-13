from __future__ import annotations

import numpy as np

from pfvol.backtest import (TRADING_DAYS, bootstrap_sharpe_diff, positions_from_forecasts, run_strategy,
                            summarize, target_weights)


def test_weight_hits_target_and_respects_cap() -> None:
    daily = 0.10 / np.sqrt(TRADING_DAYS)
    w = target_weights(np.array([daily, 2 * daily, daily / 10, np.nan]), target_ann=0.10, cap=1.5)
    assert np.allclose(w[:3], [1.0, 0.5, 1.5])
    assert np.isnan(w[3])


def test_position_uses_only_the_previous_days_forecast() -> None:
    f = np.array([0.02, 0.01, 0.04, 0.005])
    pos = positions_from_forecasts(f, target_ann=0.10, cap=10.0)
    w = target_weights(f, target_ann=0.10, cap=10.0)
    assert np.isnan(pos[0])
    assert np.allclose(pos[1:], w[:-1])  # a forecast made after day t sets the weight held on t+1


def test_constant_position_has_no_turnover_after_entry_and_costs_only_entry() -> None:
    r = np.full(10, 0.001)
    net, turn = run_strategy(r, np.full(10, 0.8), cost_bps=10.0, initial_position=0.0)
    assert np.isclose(turn[0], 0.8) and np.allclose(turn[1:], 0.0)
    assert np.isclose(net[0], 0.8 * 0.001 - 10e-4 * 0.8)
    assert np.allclose(net[1:], 0.8 * 0.001)


def test_costs_only_ever_lower_returns() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 500)
    pos = np.clip(rng.normal(1, 0.3, 500), 0, 1.5)
    free, _ = run_strategy(r, pos, cost_bps=0.0)
    paid, _ = run_strategy(r, pos, cost_bps=20.0)
    assert (paid <= free + 1e-15).all() and paid.sum() < free.sum()


def test_summary_of_a_known_series() -> None:
    r = np.array([0.10, -0.20, 0.10, 0.10])
    s = summarize(r, np.ones(4), np.zeros(4))
    assert np.isclose(s["max_drawdown"], -0.20)  # peak 1.10 -> trough 0.88
    assert np.isclose(s["total_return"], 1.1 * 0.8 * 1.1 * 1.1 - 1)
    assert s["turnover_per_year"] == 0.0


def test_bootstrap_detects_a_clear_sharpe_gap_and_not_a_tie() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 1500)
    good, bad = noise + 0.0015, noise - 0.0015
    obs, lo, hi, p = bootstrap_sharpe_diff(good, bad, reps=400)
    assert obs > 0 and lo > 0 and p < 0.05
    _, lo2, hi2, p2 = bootstrap_sharpe_diff(noise, noise.copy(), reps=200)
    assert lo2 <= 0 <= hi2 and p2 > 0.5
