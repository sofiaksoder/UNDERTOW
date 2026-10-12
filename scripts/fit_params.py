"""Estimate phi, sigma_eta (and mu) on SPY by maximizing the particle-filter log-likelihood.

Fit on 2010-2018 only; the 2019-2024 log-likelihood is then a true out-of-sample score. A coarse
grid search comes first, then the best cells are re-scored with more seeds because the
log-likelihood estimate is itself noisy (Monte Carlo). The jump and Student-t variants are
fitted the same way, starting from the baseline fit.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style

from pfvol import JumpParams, SVParams, particle_filter
from pfvol.data import load_prices, log_returns

OUT = Path(__file__).resolve().parent.parent / "docs"
TRAIN_END, TEST_START, TEST_END = "2018-12-31", "2019-01-01", "2024-12-31"
N = 2000
PHIS = [0.90, 0.94, 0.96, 0.97, 0.98, 0.99, 0.995]
SIGS = [0.05, 0.08, 0.11, 0.15, 0.20, 0.27, 0.35, 0.45]
MU_OFFSETS = [-0.4, 0.0, 0.4]
JUMP_P, JUMP_M, JUMP_S = [0.003, 0.01, 0.03], [0.5, 1.0, 1.5], 0.5
DOFS = [4.0, 6.0, 10.0, 20.0, 50.0]

_R: np.ndarray | None = None


def _init(r: np.ndarray) -> None:
    global _R
    _R = r


def _loglik(job: dict) -> float:
    """Total log-likelihood of the training returns for one parameter set and seed."""
    assert _R is not None
    p = SVParams(mu=job["mu"], phi=job["phi"], sigma_eta=job["sig"])
    jump = JumpParams(*job["jump"]) if job.get("jump") else None
    try:
        res = particle_filter(_R[: job["n"]], p, N, seed=job["seed"], compute_quantiles=False,
                              obs_dof=job.get("dof"), jump=jump)
    except FloatingPointError:
        return float("-inf")
    return res.loglik


def moment_mu(r: np.ndarray, phi: float, sig: float) -> float:
    return float(np.log(np.mean(r**2)) - 0.5 * sig**2 / (1 - phi**2))


def main() -> None:
    style.apply()
    ret = log_returns(load_prices("SPY", "2010-01-01", "2024-12-31"))
    r = ret.to_numpy()
    n_train = int((ret.index <= TRAIN_END).sum())
    train = r[:n_train]
    n_test = int(((ret.index >= TEST_START) & (ret.index <= TEST_END)).sum())
    print(f"train {n_train} days, test {n_test} days")

    with ProcessPoolExecutor(8, initializer=_init, initargs=(r,)) as ex:
        # 1) coarse grid, one seed
        jobs = [{"phi": ph, "sig": sg, "mu": moment_mu(train, ph, sg) + off, "off": off, "seed": 0,
                 "n": n_train} for ph in PHIS for sg in SIGS for off in MU_OFFSETS]
        ll = list(ex.map(_loglik, jobs, chunksize=4))
        order = np.argsort(ll)[::-1]
        # 2) re-score the 8 best cells with four more seeds
        top = [jobs[i] for i in order[:8]]
        rescored = []
        for j in top:
            seeds = [dict(j, seed=s) for s in range(1, 5)]
            rescored.append(float(np.mean(list(ex.map(_loglik, seeds)))))
        best = top[int(np.argmax(rescored))]
        base_params = {"mu": best["mu"], "phi": best["phi"], "sigma_eta": best["sig"]}
        print("baseline fit:", base_params, "train loglik", round(max(rescored), 1))

        # 3) variants, starting from the baseline fit
        def run(extra: dict, seeds: int = 4) -> float:
            js = [dict(best, seed=s, **extra) for s in range(seeds)]
            return float(np.mean(list(ex.map(_loglik, js))))

        jump_grid = {(p, m): run({"jump": (p, m, JUMP_S)}) for p in JUMP_P for m in JUMP_M}
        (jp, jm), jump_ll = max(jump_grid.items(), key=lambda kv: kv[1])
        dof_grid = {d: run({"dof": d}) for d in DOFS}
        dof, dof_ll = max(dof_grid.items(), key=lambda kv: kv[1])
        both_ll = run({"jump": (jp, jm, JUMP_S), "dof": dof})
        print(f"jump: p={jp}, mean={jm} (train loglik {jump_ll:.1f}); t: dof={dof} ({dof_ll:.1f}); "
              f"both {both_ll:.1f}")

        # 4) score every model on train and test (the filter runs on the full series)
        base = SVParams(**base_params)
        default = SVParams(mu=moment_mu(train, 0.97, 0.15), phi=0.97, sigma_eta=0.15)
        jump = JumpParams(jp, jm, JUMP_S)
        models = {
            "default (phi 0.97, sigma_eta 0.15)": (default, None, None),
            "fitted": (base, None, None),
            "fitted + jump": (base, jump, None),
            "fitted + student-t": (base, None, dof),
            "fitted + jump + student-t": (base, jump, dof),
        }
        scores = {}
        for name, (p, jmp, dd) in models.items():
            tr, te = [], []
            for s in range(6):
                res = particle_filter(r, p, N, seed=s, compute_quantiles=False, obs_dof=dd, jump=jmp)
                tr.append(res.loglik_t[:n_train].sum())
                te.append(res.loglik_t[n_train:n_train + n_test].sum())
            scores[name] = {"train_loglik": float(np.mean(tr)), "test_loglik": float(np.mean(te)),
                            "test_loglik_per_day": float(np.mean(te) / n_test)}
            print(f"{name:38s} train {np.mean(tr):9.1f}   test {np.mean(te):9.1f}")

    # heatmap at the best mu offset
    grid = np.full((len(PHIS), len(SIGS)), np.nan)
    for j, v in zip(jobs, ll):
        if abs(j["off"] - best["off"]) < 1e-9:
            grid[PHIS.index(j["phi"]), SIGS.index(j["sig"])] = v
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(grid - np.nanmax(grid), origin="lower", aspect="auto", cmap=style.AQUA,
                   vmin=-200, vmax=0)
    ax.set_xticks(range(len(SIGS)), [f"{x:.2f}" for x in SIGS]); ax.set_yticks(range(len(PHIS)),
                                                                            [f"{x:.3f}" for x in PHIS])
    ax.set_xlabel(r"$\sigma_\eta$"); ax.set_ylabel(r"$\phi$"); ax.grid(False)
    ax.plot(SIGS.index(best["sig"]), PHIS.index(best["phi"]), "o", color=style.CORAL, ms=8,
            mfc="none", mew=1.5)
    cb = fig.colorbar(im, ax=ax); cb.set_label("log-likelihood minus its maximum (clipped at -200)")
    ax.set_title("SPY 2010-2018: filter log-likelihood", loc="left", color=style.INK)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "loglik_surface.png", dpi=160)

    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "fit.json").write_text(json.dumps({
        "train": ["2010-01-04", TRAIN_END], "test": [TEST_START, TEST_END], "n_train": n_train,
        "n_test": n_test, "particles": N, "baseline": base_params,
        "jump": {"prob": jp, "mean": jm, "std": JUMP_S}, "dof": dof,
        "grid": {"phi": PHIS, "sigma_eta": SIGS, "mu_offsets": MU_OFFSETS},
        "train_loglik_best_cells": [float(x) for x in sorted(rescored, reverse=True)],
        "jump_grid_train_loglik": {f"{p}|{m}": v for (p, m), v in jump_grid.items()},
        "dof_grid_train_loglik": {str(d): v for d, v in dof_grid.items()},
        "scores": scores}, indent=2))
    print("wrote docs/results/fit.json")


if __name__ == "__main__":
    main()
