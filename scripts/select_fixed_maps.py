"""Design check before pre-registration: depth of the fixed feature maps (M2, M3).

Selection uses validation accuracy only; the test split is never looked at here.
Run: uv run python scripts/select_fixed_maps.py
"""

import numpy as np

from qdrift.circuits import exact_features, m2_projected, m3_qelm
from qdrift.config import load_config
from qdrift.data import build_dataset
from qdrift.repairs import fit_head

cfg = load_config()
n = cfg["models"]["n_qubits"]
for name in cfg["data"]["datasets"]:
    ds = build_dataset(name, cfg)
    X, y, sp = ds["X"], ds["y"], ds["splits"]
    for spec in [m2_projected(n, L) for L in (1, 2, 3)] + [m3_qelm(n, L, s) for L in (2, 3) for s in (7, 8, 9)]:
        F = exact_features(spec, X)
        h = fit_head(F[sp["train"]], y[sp["train"]])
        val = np.mean((h.proba(F[sp["val"]]) >= 0.5) == y[sp["val"]])
        cz = sum(op.gate == "cz" for op in spec.ops)
        print(f"{name:14s} {spec.name} ops={len(spec.ops):3d} cz={cz:2d}  val acc {val:.3f}")
