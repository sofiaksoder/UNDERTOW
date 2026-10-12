"""Write tests/data/parity.json: one return series plus the Python filter's output on it.

The browser filter (web/index.html) is checked against this: same input, agree within Monte Carlo
error. The plain filter and both robust variants (jump, Student-t) are covered.
"""
import json
from pathlib import Path

from pfvol import JumpParams, SVParams, particle_filter, simulate

ROOT = Path(__file__).resolve().parent.parent
fit = json.loads((ROOT / "docs" / "results" / "fit.json").read_text())
JUMP = JumpParams(**fit["jump"])
DOF = float(fit["dof"])
REF_N = 100_000

p = SVParams()
path = simulate(250, p, seed=42, shocks={150: 1.0})
runs = {
    "plain": particle_filter(path.returns, p, REF_N, seed=1),
    "jump": particle_filter(path.returns, p, REF_N, seed=1, jump=JUMP),
    "t": particle_filter(path.returns, p, REF_N, seed=1, obs_dof=DOF),
}
out = {
    "params": {"mu": p.mu, "phi": p.phi, "sigma_eta": p.sigma_eta},
    "jump": fit["jump"], "dof": DOF,
    "returns": path.returns.tolist(), "reference_particles": REF_N,
    "variants": {k: {"mean_vol": r.mean_vol.tolist(), "q50": r.quantiles[:, 2].tolist(),
                     "ess_ratio": (r.ess / REF_N).tolist(), "loglik": r.loglik} for k, r in runs.items()},
}
# keep the original top-level keys for the plain variant (older tooling reads them)
out["mean_vol"] = out["variants"]["plain"]["mean_vol"]
out["loglik"] = out["variants"]["plain"]["loglik"]
dest = ROOT / "tests" / "data" / "parity.json"
dest.write_text(json.dumps(out))
print(dest, f"{dest.stat().st_size/1024:.0f} KB", {k: round(v["loglik"], 2) for k, v in out["variants"].items()})
