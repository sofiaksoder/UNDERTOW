"""Animated GIF of the shock response (docs/figures/shock_response.gif).

The average over many simulated paths, each with a +1.5 jump in log-variance on the same day;
the lines reveal day by day. Same setup as
scripts/shock_response.py: GARCH is fitted only on the days before the shock, the filter variants
use the true model parameters and the jump settings fitted on SPY.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style
from matplotlib.animation import FuncAnimation, PillowWriter

from pfvol import JumpParams, particle_filter, simulate
from pfvol.benchmarks import ewma_vol, garch_next_vol, garch_params
from pfvol.model import annualize

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
DAYS, SHOCK_DAY, GARCH_FIT_DAYS, SIZE, N_PATHS = 1050, 1000, 950, 1.5, 80
FIT = json.loads((OUT.parent / "results" / "fit.json").read_text())
LO, HI = SHOCK_DAY - 30, SHOCK_DAY + 45


def one_path(seed: int) -> dict[str, np.ndarray]:
    path = simulate(DAYS, seed=2000 + seed, shocks={SHOCK_DAY: SIZE})
    r = path.returns
    kw = {"n_particles": 2000, "seed": seed, "compute_quantiles": False}
    gp = garch_params(r[:GARCH_FIT_DAYS])
    est = {
        "true volatility": path.vol,
        "particle filter": particle_filter(r, **kw).mean_vol,
        "filter + jump": particle_filter(r, jump=JumpParams(**FIT["jump"]), **kw).mean_vol,
        "EWMA (0.94)": ewma_vol(r),
        "GARCH(1,1)": garch_next_vol(r, gp["omega"], gp["alpha"], gp["beta"]),
    }
    return {k: annualize(v[LO:HI + 1]) for k, v in est.items()}


def main() -> None:
    style.apply()
    with ProcessPoolExecutor(8) as ex:
        paths = list(ex.map(one_path, range(N_PATHS)))
    style_of = {"true volatility": (style.INK, "--", 1.3), "particle filter": (style.ACCENT, "-", 2.0),
                "filter + jump": (style.CORAL, "-", 2.0), "EWMA (0.94)": (style.EWMA, "-", 1.2),
                "GARCH(1,1)": (style.GARCH, "-.", 1.2)}
    lines = {k: np.mean([p[k] for p in paths], axis=0) for k in style_of}
    x = np.arange(LO, HI + 1)
    ymax = max(v.max() for v in lines.values()) * 1.1
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(LO, HI); ax.set_ylim(0, ymax)
    ax.set_xlabel("day"); ax.set_ylabel("volatility, % annualized")
    ax.axvline(SHOCK_DAY, color=style.CORAL, ls=":", lw=1)
    ax.text(SHOCK_DAY + 0.8, ymax * 0.93, "shock", color=style.CORAL, fontsize=9)
    artists = {k: ax.plot([], [], color=c, ls=ls, lw=lw, label=k)[0] for k, (c, ls, lw) in style_of.items()}
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"A sudden volatility shock: average of {N_PATHS} simulated paths", loc="left",
                 color=style.INK, fontsize=10)
    fig.tight_layout()
    n_frames = len(x) + 12  # hold the last frame for a moment

    def draw(i: int):
        k = min(i, len(x) - 1)
        for name, v in lines.items():
            artists[name].set_data(x[:k + 1], v[:k + 1])
        return list(artists.values())

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=100, blit=True)
    anim.save(OUT / "shock_response.gif", writer=PillowWriter(fps=10), dpi=90)
    print("wrote", OUT / "shock_response.gif", f"{(OUT / 'shock_response.gif').stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
