"""Monte Carlo error of the filter vs number of particles (log-log), against 1/sqrt(N)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style

from pfvol import particle_filter, simulate

OUT = Path(__file__).resolve().parent.parent / "docs"
NS = [100, 200, 500, 1000, 2000, 5000, 10000]
N_SEEDS, REF_N, DAYS = 30, 200_000, 500


def main() -> None:
    style.apply()
    r = simulate(DAYS, seed=0).returns
    ref = particle_filter(r, n_particles=REF_N, seed=12345, compute_quantiles=False).mean_vol
    errs = np.array([
        [np.sqrt(np.mean((particle_filter(r, n_particles=n, seed=s, compute_quantiles=False)
                          .mean_vol - ref) ** 2)) for s in range(N_SEEDS)]
        for n in NS
    ])
    med = np.median(errs, axis=1)
    lo, hi = np.percentile(errs, [5, 95], axis=1)
    slope = np.polyfit(np.log(NS), np.log(med), 1)[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(NS, lo, hi, color=style.DEEP, alpha=0.3, lw=0, label="5-95% over seeds")
    ax.plot(NS, med, "o-", color=style.ACCENT, ms=4, lw=1.5, label="median RMSE")
    ref_line = med[0] * np.sqrt(NS[0] / np.array(NS))
    ax.plot(NS, ref_line, "--", color=style.CORAL, lw=1, label=r"$\propto 1/\sqrt{N}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("particles N"); ax.set_ylabel("RMSE vs reference run (daily vol)")
    ax.set_title(f"Filter error vs N  (fitted slope {slope:.2f})", color=style.INK, loc="left")
    ax.legend()
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "convergence.png", dpi=160)

    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "convergence.json").write_text(json.dumps({
        "days": DAYS, "reference_particles": REF_N, "seeds": N_SEEDS, "n": NS,
        "median_rmse": med.tolist(), "p05": lo.tolist(), "p95": hi.tolist(),
        "loglog_slope": slope,
    }, indent=2))
    print("slope", round(slope, 3))
    for n, m in zip(NS, med):
        print(n, f"{m:.3e}")


if __name__ == "__main__":
    main()
