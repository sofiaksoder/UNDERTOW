"""How fast do the estimators react to a volatility shock? Simulated data, true vol known.

Fair setup: GARCH is fitted only on the days before the shock; the jump / Student-t variants use
the parameters fitted on SPY 2010-2018 (docs/results/fit.json), so nothing is tuned to these shocks.
The plain filter and its variants all use the true model parameters (a best case).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style

from pfvol import JumpParams, particle_filter, simulate
from pfvol.benchmarks import ewma_vol, garch_next_vol, garch_params, rolling_std
from pfvol.metrics import mae, rmse
from pfvol.model import annualize

OUT = Path(__file__).resolve().parent.parent / "docs"
DAYS, SHOCK_DAY, GARCH_FIT_DAYS, N, N_PATHS, WINDOW = 1050, 1000, 950, 2000, 100, 15
SHOCKS = [0.0, 0.5, 1.0, 1.5]  # added to log-variance; 0.15 is the model's per-day noise
PLOT_SHOCK = 1.5
METHODS = ["Particle filter", "PF + jump", "PF + Student-t", "PF + jump + t",
           "Rolling std (20d)", "EWMA (0.94)", "GARCH(1,1)"]
fit = json.loads((OUT / "results" / "fit.json").read_text()) if (OUT / "results" / "fit.json").exists() else {}
JUMP = JumpParams(**fit["jump"]) if "jump" in fit else JumpParams()
DOF = float(fit.get("dof", 6.0))


def estimates(returns: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Every estimate at day t only uses returns up to and including day t."""
    kw = {"n_particles": N, "seed": seed, "compute_quantiles": False}
    gp = garch_params(returns[:GARCH_FIT_DAYS])
    return {
        "Particle filter": particle_filter(returns, **kw).mean_vol,
        "PF + jump": particle_filter(returns, jump=JUMP, **kw).mean_vol,
        "PF + Student-t": particle_filter(returns, obs_dof=DOF, **kw).mean_vol,
        "PF + jump + t": particle_filter(returns, jump=JUMP, obs_dof=DOF, **kw).mean_vol,
        "Rolling std (20d)": rolling_std(returns, 20), "EWMA (0.94)": ewma_vol(returns),
        "GARCH(1,1)": garch_next_vol(returns, gp["omega"], gp["alpha"], gp["beta"]),
    }


def one_path(job: tuple[float, int]) -> dict:
    size, s = job
    p = simulate(DAYS, seed=1000 + s, shocks={SHOCK_DAY: size} if size else None)
    est = estimates(p.returns, s)
    win = slice(SHOCK_DAY, SHOCK_DAY + WINDOW)
    sl = slice(SHOCK_DAY - 10, SHOCK_DAY + 30)
    return {"size": size, "mae": {k: mae(v[win], p.vol[win]) for k, v in est.items()},
            "rmse": {k: rmse(v[win], p.vol[win]) for k, v in est.items()},
            "curve": {k: np.abs(v[sl] - p.vol[sl]) for k, v in est.items()}}


def main() -> None:
    style.apply()
    jobs = [(size, s) for size in SHOCKS for s in range(N_PATHS)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one_path, jobs, chunksize=2))

    table: dict[str, dict[str, dict[str, float]]] = {}
    curves = {k: np.zeros(40) for k in METHODS}
    for size in SHOCKS:
        rs = [r for r in res if r["size"] == size]
        table[str(size)] = {}
        gar = np.array([r["mae"]["GARCH(1,1)"] for r in rs])
        pf0 = np.array([r["mae"]["Particle filter"] for r in rs])
        for k in METHODS:
            m = np.array([r["mae"][k] for r in rs])
            row = {"mae": float(m.mean()), "mae_se": float(m.std(ddof=1) / np.sqrt(len(m))),
                   "rmse": float(np.mean([r["rmse"][k] for r in rs]))}
            if k not in ("Particle filter", "GARCH(1,1)"):  # paired difference to the plain filter
                dp = m - pf0
                row["mae_minus_plain_pf"] = float(dp.mean())
                row["mae_minus_plain_pf_se"] = float(dp.std(ddof=1) / np.sqrt(len(dp)))
                row["better_than_plain_pf_on_paths"] = int(np.sum(m < pf0))
            if k != "GARCH(1,1)":  # paired difference to GARCH across paths
                d = m - gar
                row["mae_minus_garch"] = float(d.mean())
                row["mae_minus_garch_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
                row["better_than_garch_on_paths"] = int(np.sum(m < gar))
            table[str(size)][k] = row
        if size == PLOT_SHOCK:
            for r in rs:
                for k in METHODS:
                    curves[k] += r["curve"][k] / len(rs)

    path = simulate(DAYS, seed=3, shocks={SHOCK_DAY: PLOT_SHOCK})
    est = estimates(path.returns, 3)
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 6.4))
    x = np.arange(DAYS); s = slice(SHOCK_DAY - 40, SHOCK_DAY + 60)
    a.plot(x[s], annualize(path.vol[s]), color=style.INK, lw=1.2, ls="--", label="true vol")
    colors = {"Particle filter": style.ACCENT, "PF + jump": style.CORAL, "GARCH(1,1)": style.GARCH,
              "EWMA (0.94)": style.EWMA}
    for k, c in colors.items():
        a.plot(x[s], annualize(est[k][s]), color=c, lw=1.8 if k == "PF + jump" else 1.2, label=k)
    a.axvline(SHOCK_DAY, color=style.CORAL, ls=":", lw=1); a.set_ylabel("vol, % annualized")
    a.set_title(f"One path, +{PLOT_SHOCK} shock in log-variance on day {SHOCK_DAY}", loc="left", color=style.INK)
    a.legend(ncol=3, fontsize=8)
    d = np.arange(-10, 30)
    for k, c in {**colors, "Rolling std (20d)": style.ROLL}.items():
        b.plot(d, annualize(curves[k]), color=c, lw=1.8 if k == "PF + jump" else 1.2, label=k)
    b.axvline(0, color=style.CORAL, ls=":", lw=1)
    b.set_xlabel("days since shock"); b.set_ylabel("mean abs error, % ann.")
    b.set_title(f"Mean over {N_PATHS} paths", loc="left", color=style.INK)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "shock_response.png", dpi=160)

    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "shock_response.json").write_text(json.dumps({
        "days": DAYS, "shock_day": SHOCK_DAY, "garch_fit_days": GARCH_FIT_DAYS, "particles": N,
        "paths": N_PATHS, "window_days_after_shock": WINDOW,
        "jump": {"prob": JUMP.prob, "mean": JUMP.mean, "std": JUMP.std}, "dof": DOF,
        "errors_in_daily_vol_by_shock_size": table}, indent=2))
    for size, rows in table.items():
        print(f"shock +{size}")
        for k, v in rows.items():
            extra = "" if "better_than_garch_on_paths" not in v else (
                f"  vs GARCH {v['mae_minus_garch']*1e4:+.2f}e-4 (se {v['mae_minus_garch_se']*1e4:.2f}), "
                f"better on {v['better_than_garch_on_paths']}/{N_PATHS}")
            if "mae_minus_plain_pf" in v:
                extra += (f"\n{'':20s} vs plain PF {v['mae_minus_plain_pf']*1e4:+.2f}e-4 "
                          f"(se {v['mae_minus_plain_pf_se']*1e4:.2f}), better on "
                          f"{v['better_than_plain_pf_on_paths']}/{N_PATHS}")
            print(f"  {k:18s} MAE {v['mae']*1e4:6.2f}e-4 (se {v['mae_se']*1e4:.2f}){extra}")


if __name__ == "__main__":
    main()
