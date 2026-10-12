"""Out-of-sample evaluation on SPY.

Everything with parameters is fitted on 2010-2018 only (the particle filter via scripts/fit_params.py,
GARCH here). The methods are then scored on 2019-2024 on next-day volatility forecasts, against
two noisy proxies of what happened next (true volatility is unobservable on real data):
QLIKE against the next day's squared return, and MAE against the next 5 days' realized vol.
Diebold-Mariano tests compare the QLIKE losses.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style

from pfvol import JumpParams, SVParams, particle_filter
from pfvol.benchmarks import ewma_vol, garch_next_vol, garch_params, rolling_std
from pfvol.data import load_prices, log_returns
from pfvol.metrics import diebold_mariano, forward_realized_vol, qlike_losses
from pfvol.model import annualize

OUT = Path(__file__).resolve().parent.parent / "docs"
N, N_SEEDS = 2000, 5
TRAIN_END, TEST_START, TEST_END = "2018-12-31", "2019-01-01", "2024-12-31"
HORIZON = 5


def moment_mu(r: np.ndarray, phi: float, sig: float) -> float:
    return float(np.log(np.mean(r**2)) - 0.5 * sig**2 / (1 - phi**2))


def main() -> None:
    fit = json.loads((OUT / "results" / "fit.json").read_text())
    prices = load_prices("SPY", "2010-01-01", "2024-12-31")
    ret = log_returns(prices)
    r, dates = ret.to_numpy(), ret.index
    n_train = int((dates <= TRAIN_END).sum())
    train = r[:n_train]

    base = SVParams(**fit["baseline"])
    jump = JumpParams(**fit["jump"])
    dof = float(fit["dof"])
    default = SVParams(mu=moment_mu(train, 0.97, 0.15), phi=0.97, sigma_eta=0.15)
    pf_models = {
        "Particle filter (default params)": (default, None, None),
        "Particle filter (fitted)": (base, None, None),
        "Particle filter (fitted + jump)": (base, jump, None),
        "Particle filter (fitted + Student-t)": (base, None, dof),
        "Particle filter (fitted + jump + t)": (base, jump, dof),
    }
    runs = {name: [particle_filter(r, p, N, seed=s, compute_quantiles=(name == "Particle filter (fitted)"
                                                                     and s == 0), obs_dof=d, jump=j)
                   for s in range(N_SEEDS)] for name, (p, j, d) in pf_models.items()}

    gp = garch_params(train)  # GARCH fitted on the training years only
    forecasts: dict[str, list[np.ndarray]] = {
        name: [np.sqrt(f.pred_var) for f in fs] for name, fs in runs.items()}
    forecasts["Rolling std (20d)"] = [rolling_std(r, 20)]
    forecasts["EWMA (0.94)"] = [ewma_vol(r)]
    forecasts["GARCH(1,1), fit on 2010-2018"] = [garch_next_vol(r, gp["omega"], gp["alpha"], gp["beta"])]

    n = len(r)
    nxt = np.append(r[1:], np.nan)  # r_{t+1}, aligned with the forecast made at t
    rv5 = forward_realized_vol(r, HORIZON)
    in_test = (dates >= TEST_START) & (dates <= TEST_END)
    valid = np.arange(n) < n - HORIZON
    periods = {
        "2019-2024 (out of sample)": in_test,
        "2020 Feb-Jun": (dates >= "2020-02-01") & (dates <= "2020-06-30"),
        "2022": (dates >= "2022-01-01") & (dates <= "2022-12-31"),
        "2019 + 2021 (calmer)": ((dates >= "2019-01-01") & (dates <= "2019-12-31"))
        | ((dates >= "2021-01-01") & (dates <= "2021-12-31")),
    }
    # per-day loss series; for the particle filters, averaged over seeds
    losses = {k: np.mean([qlike_losses(f, nxt) for f in fs], axis=0) for k, fs in forecasts.items()}
    maes = {k: np.mean([np.abs(f - rv5) for f in fs], axis=0) for k, fs in forecasts.items()}

    scores = {}
    for pname, mask in periods.items():
        m = mask & valid & np.all([np.isfinite(losses[k]) for k in losses], axis=0)
        scores[pname] = {k: {"qlike": float(np.mean(losses[k][m])), "mae_rv5": float(np.mean(maes[k][m])),
                             "n_days": int(m.sum())} for k in forecasts}

    # Diebold-Mariano on the whole out-of-sample window (negative mean_diff: first method has lower loss)
    m = in_test & valid & np.all([np.isfinite(losses[k]) for k in losses], axis=0)
    dm = {}
    for a in ("Particle filter (fitted)", "Particle filter (fitted + jump + t)"):
        dm[a] = {}
        for b in forecasts:
            if b == a:
                continue
            mean_d, z, p = diebold_mariano(losses[a][m], losses[b][m])
            dm[a][b] = {"mean_qlike_diff": mean_d, "z": z, "p_value": p}

    # The same tests on the crash window only. This window was singled out in advance because the shock
    # experiments predicted the plain filter's weakness there; with ~100 days it is a small sample.
    crash = periods["2020 Feb-Jun"] & valid & np.all([np.isfinite(losses[k]) for k in losses], axis=0)
    gname = "GARCH(1,1), fit on 2010-2018"
    dm_crash = {}
    for a, b in (("Particle filter (fitted)", gname), ("Particle filter (fitted + jump)", gname),
                 ("Particle filter (fitted + jump)", "Particle filter (fitted)"),
                 ("Particle filter (fitted + jump + t)", "Particle filter (fitted)")):
        mean_d, z, p = diebold_mariano(losses[a][crash], losses[b][crash])
        dm_crash[f"{a} vs {b}"] = {"mean_qlike_diff": mean_d, "z": z, "p_value": p, "n_days": int(crash.sum())}

    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "spy.json").write_text(json.dumps({
        "ticker": "SPY", "train": ["2010-01-04", TRAIN_END], "test": [TEST_START, TEST_END],
        "particles": N, "seeds_averaged": N_SEEDS, "filter_params": fit["baseline"],
        "jump": fit["jump"], "dof": dof, "garch_params_fit_2010_2018": gp, "scores": scores,
        "diebold_mariano_qlike_2019_2024": dm, "diebold_mariano_qlike_2020_feb_jun": dm_crash}, indent=2))

    # figure: fitted filter vs baselines through the 2020 crash
    f0 = runs["Particle filter (fitted)"][0]
    sel = (dates >= "2019-10-01") & (dates <= "2020-12-31")
    x = dates[sel]
    fig, ax = plt.subplots(3, 1, figsize=(8, 7.4), sharex=True, gridspec_kw={"height_ratios": [1, 2, 0.8]})
    ax[0].plot(x, prices.reindex(dates)[sel], color=style.INK, lw=1); ax[0].set_ylabel("SPY adj. close")
    ax[1].fill_between(x, annualize(f0.quantiles[sel, 0]), annualize(f0.quantiles[sel, 4]),
                       color=style.DEEP, alpha=0.3, lw=0, label="PF 90% band")
    ax[1].fill_between(x, annualize(f0.quantiles[sel, 1]), annualize(f0.quantiles[sel, 3]),
                       color=style.DEEP, alpha=0.5, lw=0, label="PF 50% band")
    ax[1].plot(x, annualize(f0.quantiles[sel, 2]), color=style.ACCENT, lw=1.6, label="PF median (fitted)")
    ax[1].plot(x, annualize(forecasts["EWMA (0.94)"][0][sel]), color=style.EWMA, lw=1, label="EWMA 0.94")
    ax[1].plot(x, annualize(forecasts["GARCH(1,1), fit on 2010-2018"][0][sel]), color=style.GARCH, lw=1,
               ls="-.", label="GARCH (fit 2010-18)")
    ax[1].set_ylabel("vol, % annualized"); ax[1].legend(ncol=3, fontsize=8)
    ax[2].bar(x, f0.ess[sel] / N, color=style.DEEP, width=1.0)
    ax[2].bar(x[f0.resampled[sel]], f0.ess[sel][f0.resampled[sel]] / N, color=style.ACCENT, width=1.0)
    ax[2].axhline(0.5, color=style.ACCENT, ls=":", lw=0.8); ax[2].set_ylabel("ESS / N")
    fig.suptitle("SPY, late 2019 to 2020: out-of-sample filter vs baselines", x=0.01, ha="left", color=style.INK)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "spy_2020.png", dpi=160)

    for pname, rows in scores.items():
        print(f"\n{pname} ({rows['EWMA (0.94)']['n_days']} days)")
        for k, v in rows.items():
            print(f"  {k:40s} QLIKE {v['qlike']:8.4f}   MAE vs 5d RV {v['mae_rv5']*100:.4f}%")
    print("\nDiebold-Mariano on the 2020 Feb-Jun window (small sample)")
    for k, v in dm_crash.items():
        print(f"  {k:92s} diff {v['mean_qlike_diff']:+.4f}  z {v['z']:+.2f}  p {v['p_value']:.3f}")
    print("\nDiebold-Mariano, QLIKE, 2019-2024 (mean diff < 0: first method better)")
    for a, row in dm.items():
        for b, v in row.items():
            print(f"  {a[:38]:38s} vs {b[:38]:38s} diff {v['mean_qlike_diff']:+.4f}  z {v['z']:+.2f}  p {v['p_value']:.3f}")


if __name__ == "__main__":
    main()
