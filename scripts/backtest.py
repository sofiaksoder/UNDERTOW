"""Volatility-targeting backtest on SPY, out of sample (2019-2024), driven by each volatility forecast.

Every strategy holds w = clip(10% / forecast, 0, 1.5) of SPY and the rest in cash (0% return); only the
forecast differs. All parameters (particle filter, GARCH) were fitted on 2010-2018, so nothing here is
tuned on the test years. Costs are charged per unit of turnover. The filter results are the mean over
5 seeds (the forecast is stochastic); the figure and the bootstrap use seed 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style

from pfvol import JumpParams, SVParams, particle_filter
from pfvol.backtest import bootstrap_sharpe_diff, positions_from_forecasts, run_strategy, summarize
from pfvol.benchmarks import ewma_vol, garch_next_vol, rolling_std
from pfvol.data import load_prices, log_returns

OUT = Path(__file__).resolve().parent.parent / "docs"
N, N_SEEDS = 2000, 5
TEST_START, TEST_END = "2019-01-01", "2024-12-31"
TARGET, CAP = 0.10, 1.5
COST_BPS, COST_GRID = 5.0, [0.0, 5.0, 10.0, 25.0]
PF, PFJ, GARCH, ROLL, EWMA, BH = ("Undertow (particle filter)", "Undertow + jump", "GARCH(1,1)",
                                  "Rolling std (20d)", "EWMA (0.94)", "Buy and hold")
COLORS = {PF: style.ACCENT, PFJ: style.CORAL, GARCH: style.GARCH, ROLL: style.ROLL, EWMA: style.EWMA,
          BH: style.INK}


def evaluate(simple: np.ndarray, pos: np.ndarray, mask: np.ndarray, cost: float) -> dict:
    net, turn = run_strategy(simple[mask], pos[mask], cost, initial_position=0.0)
    return summarize(net, pos[mask], turn), net, turn


def main() -> None:
    fit = json.loads((OUT / "results" / "fit.json").read_text())
    spy = json.loads((OUT / "results" / "spy.json").read_text())
    ret = log_returns(load_prices("SPY", "2010-01-01", "2024-12-31"))
    r, dates = ret.to_numpy(), ret.index
    simple = np.expm1(r)
    mask = (dates >= TEST_START) & (dates <= TEST_END)
    base, jump = SVParams(**fit["baseline"]), JumpParams(**fit["jump"])
    gp = spy["garch_params_fit_2010_2018"]

    # forecast[t] is made after seeing r_t, for day t+1
    runs = {name: [np.sqrt(particle_filter(r, base, N, seed=s, compute_quantiles=False,
                                           jump=j).pred_var) for s in range(N_SEEDS)]
            for name, j in ((PF, None), (PFJ, jump))}
    runs[GARCH] = [garch_next_vol(r, gp["omega"], gp["alpha"], gp["beta"])]
    runs[ROLL] = [rolling_std(r, 20)]
    runs[EWMA] = [ewma_vol(r)]
    positions = {k: [positions_from_forecasts(f, TARGET, CAP) for f in fs] for k, fs in runs.items()}
    positions[BH] = [np.ones(len(r))]
    assert all(np.isfinite(p[mask]).all() for ps in positions.values() for p in ps), "missing forecast"

    results: dict[str, dict] = {}
    for name, ps in positions.items():
        per_cost = {}
        for c in COST_GRID:
            ms = [evaluate(simple, p, mask, 0.0 if name == BH else c)[0] for p in ps]
            per_cost[str(c)] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            if len(ps) > 1:
                per_cost[str(c)]["sharpe_range_over_seeds"] = [float(min(m["sharpe"] for m in ms)),
                                                               float(max(m["sharpe"] for m in ms))]
        results[name] = per_cost

    # seed-0 net daily returns at the headline cost, for the figure and the bootstrap
    net = {k: evaluate(simple, ps[0], mask, 0.0 if k == BH else COST_BPS)[1] for k, ps in positions.items()}
    boot = {}
    for a, b in ((PF, ROLL), (PF, GARCH), (PF, EWMA), (PFJ, GARCH), (PF, BH)):
        obs, lo, hi, p = bootstrap_sharpe_diff(net[a], net[b])
        boot[f"{a} vs {b}"] = {"sharpe_diff": obs, "ci95": [lo, hi], "p_value": p}

    # worst drawdown inside the 2020 crash window, for context
    crash = (dates[mask] >= "2020-02-01") & (dates[mask] <= "2020-06-30")
    crash_dd = {}
    for k, nr in net.items():
        eq = np.cumprod(1.0 + nr[crash])
        crash_dd[k] = float((eq / np.maximum.accumulate(np.insert(eq, 0, 1.0))[1:] - 1.0).min())

    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "backtest.json").write_text(json.dumps({
        "asset": "SPY", "test": [TEST_START, TEST_END], "n_days": int(mask.sum()),
        "target_ann_vol": TARGET, "leverage_cap": CAP, "cost_bps_headline": COST_BPS,
        "cost_grid_bps": COST_GRID, "particles": N, "seeds_averaged": N_SEEDS,
        "garch_params_fit_2010_2018": gp, "results": results, "sharpe_diff_bootstrap_seed0": boot,
        "max_drawdown_2020_feb_jun_seed0": crash_dd}, indent=2))

    print(f"\n{COST_BPS:g} bps per unit of turnover, {int(mask.sum())} days")
    print(f"{'strategy':28s} {'ret':>6s} {'vol':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'turn/yr':>8s} {'lev':>5s}")
    for k, pc in results.items():
        m = pc[str(0.0 if k == BH else COST_BPS)]
        print(f"{k:28s} {m['ann_return']:6.1%} {m['ann_vol']:6.1%} {m['sharpe']:7.2f} "
              f"{m['max_drawdown']:7.1%} {m['turnover_per_year']:8.1f} {m['avg_leverage']:5.2f}")

    # figure: net growth of $1, drawdown, position size
    style.apply()
    x = dates[mask]
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    for k, nr in net.items():
        eq = np.cumprod(1.0 + nr)
        lw, ls = (1.8, "-") if k in (PF, PFJ) else (1.1, "--" if k == BH else "-")
        ax[0].plot(x, eq, color=COLORS[k], lw=lw, ls=ls, label=k)
        ax[1].plot(x, eq / np.maximum.accumulate(eq) - 1.0, color=COLORS[k], lw=lw * 0.8, ls=ls)
    for k in (PF, GARCH, ROLL):
        ax[2].plot(x, positions[k][0][mask], color=COLORS[k], lw=1.0 if k != PF else 1.4, label=k)
    ax[0].set_ylabel("growth of $1, net of costs"); ax[0].legend(ncol=3, fontsize=8)
    ax[1].set_ylabel("drawdown"); ax[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax[2].set_ylabel("position (x capital)")
    fig.suptitle(f"SPY 2019-2024: {TARGET:.0%} volatility target, {COST_BPS:g} bps per unit of turnover",
                 x=0.01, ha="left", color=style.INK)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "backtest.png", dpi=160)


if __name__ == "__main__":
    main()
