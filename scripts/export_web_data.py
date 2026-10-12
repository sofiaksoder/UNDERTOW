"""Write web/data/spy.json for the demo's "Real: SPY 2020" preset.

Window: 2018-10-01 to 2020-12-31. The first months are warm-up for the filter (the demo's day-70
start is just after them). Parameters come from the 2010-2018 fits only, so the demo's real-data
run is out of sample. Volatility fields are daily volatility fractions except VIX, which is the
index level (annualized %).
"""
from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np

from pfvol.benchmarks import garch_params
from pfvol.data import load_prices, log_returns
from pfvol.metrics import forward_realized_vol

ROOT = Path(__file__).resolve().parent.parent
FROM, TO, TRAIN_END = "2018-10-01", "2020-12-31", "2018-12-31"


def main() -> None:
    fit = json.loads((ROOT / "docs" / "results" / "fit.json").read_text())
    prices = load_prices("SPY", "2010-01-01", "2024-12-31")
    ret = log_returns(prices)
    r = ret.to_numpy()
    rv5 = forward_realized_vol(r, 5)  # realized vol of the next 5 days, computed on the full series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch = garch_params(r[(ret.index <= TRAIN_END)])
    vix = load_prices("^VIX", "2018-01-01", "2021-03-31")

    sel = np.where((ret.index >= FROM) & (ret.index <= TO))[0]
    dates = [ret.index[i].strftime("%Y-%m-%d") for i in sel]
    vix_aligned = [None if d not in vix.index.strftime("%Y-%m-%d") else round(float(vix.loc[d]), 2)
                   for d in dates]
    out = {
        "meta": {
            "ticker": "SPY", "source": "Yahoo Finance adjusted close via yfinance (VIX: ^VIX close)",
            "exported": date.today().isoformat(), "window": [FROM, TO],
            "notes": "filter and GARCH parameters fitted on 2010-2018 only, so 2019-2020 is out of sample; "
                     "realized vol is the realized volatility of the NEXT 5 days (hindsight)",
            "units": "ret: daily log return; rv5: daily volatility; vix: annualized %",
        },
        "params": {"mu": fit["baseline"]["mu"], "phi": fit["baseline"]["phi"],
                   "sigma_eta": fit["baseline"]["sigma_eta"]},
        "jump": fit["jump"], "dof": fit["dof"], "garch": garch,
        "dates": dates, "close": [round(float(prices.loc[d]), 2) for d in dates],
        "ret": [round(float(r[i]), 6) for i in sel],
        "rv5": [None if not np.isfinite(rv5[i]) else round(float(rv5[i]), 6) for i in sel],
        "vix": vix_aligned,
    }
    path = ROOT / "web" / "data" / "spy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    missing = sum(v is None for v in vix_aligned)
    print(path, f"{path.stat().st_size/1024:.0f} KB", len(sel), "days", dates[0], dates[-1],
          "| VIX missing on", missing, "days | garch", {k: round(v, 5) for k, v in garch.items()},
          "| start index of 2020-02-14:", dates.index("2020-02-14"))


if __name__ == "__main__":
    main()
