"""Design check before pre-registration: anchoring strength of the R2 refit.

Uses dry-run sessions only (fake backends with injected drift), never hardware data.
Run after `uv run qdrift dryrun-campaign`: uv run python scripts/select_refit_lambda.py
"""

import numpy as np

from qdrift.analysis import features, load_session
from qdrift.config import load_config
from qdrift.repairs import anchored_refit
from qdrift.session import session_dirs
from qdrift.train import head, load_processed

cfg = load_config(dry_run=True)
proc = load_processed(cfg)
grid = {"fixed 1": lambda k: 1.0, "fixed 3": lambda k: 3.0, "fixed 10": lambda k: 10.0,
        "0.25 k": lambda k: 0.25 * k, "0.5 k": lambda k: 0.5 * k, "1 k": lambda k: 1.0 * k}
gain = {g: {k: [] for k in (8, 16, 32)} for g in grid}
for device in cfg["dryrun"]["devices"]:
    for path in session_dirs(cfg, device):
        S = load_session(path)
        P = len(S["meta"]["patches"])
        for dn in cfg["data"]["datasets"]:
            ds = proc["datasets"][dn]
            sp, y = ds["splits"], ds["y"]
            ncal, ycal, ytest = len(sp["cal"]), y[sp["cal"]], y[sp["test"]]
            for spec in proc["specs"]:
                R0 = head(proc, dn, spec.name, "R0")
                F = features(spec, S["counts"], dn, P)
                base = np.mean((R0.proba(F[ncal:]) >= 0.5) == ytest)
                for g, lam in grid.items():
                    for k in (8, 16, 32):
                        h = anchored_refit(R0, F[:k], ycal[:k], lam(k))
                        gain[g][k].append(np.mean((h.proba(F[ncal:]) >= 0.5) == ytest) - base)
for g in grid:
    print(f"lambda {g:9s} " + "  ".join(f"k={k}: {np.mean(v):+.4f}" for k, v in gain[g].items()))
