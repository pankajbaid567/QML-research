"""Simulator training (Tasks 3-4): datasets, classical baselines, quantum models and heads."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .circuits import ModelSpec, exact_features, feature_grad, model_specs
from .data import build_dataset, load_datasets, save_datasets
from .metrics import accuracy_ci, classification_metrics
from .repairs import Head, fit_head


def _log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_m1(spec: ModelSpec, Xtr, ytr, Xval, yval, steps: int, lr: float, restarts: int, seed: int):
    """Train circuit parameters jointly with a linear head (Adam, exact parameter shift)."""
    P, F, B = spec.n_params, spec.n_features, len(Xtr)
    best, log = None, []
    for r in range(restarts):
        rng = np.random.default_rng([seed, r])
        v = np.r_[rng.uniform(-np.pi, np.pi, P), np.zeros(F + 1)]
        m1, m2 = np.zeros_like(v), np.zeros_like(v)
        for t in range(1, steps + 1):
            theta, w, b = v[:P], v[P:P + F], v[-1]
            z, dz = feature_grad(spec, Xtr, theta)
            g = (expit(z @ w + b) - ytr) / B
            grad = np.r_[np.einsum("pbf,f,b->p", dz, w, g), z.T @ g + 2e-3 * w, g.sum()]
            m1 = 0.9 * m1 + 0.1 * grad
            m2 = 0.999 * m2 + 0.001 * grad**2
            v = v - lr * (m1 / (1 - 0.9**t)) / (np.sqrt(m2 / (1 - 0.999**t)) + 1e-8)
        theta, w, b = v[:P], v[P:P + F], v[-1]
        val = _log_loss(yval, expit(exact_features(spec, Xval, theta) @ w + b))
        tr = _log_loss(ytr, expit(exact_features(spec, Xtr, theta) @ w + b))
        log.append({"restart": r, "train_loss": tr, "val_loss": val})
        if best is None or val < best[0]:
            best = (val, theta.copy())
    return best[1], log


def _baselines(seed: int):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return {
        "logreg": GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                               {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}, cv=cv),
        "rbf_svm": GridSearchCV(make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed)),
                                {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.1, 1.0]}, cv=cv),
        "mlp": GridSearchCV(make_pipeline(StandardScaler(), MLPClassifier(max_iter=3000, random_state=seed)),
                            {"mlpclassifier__hidden_layer_sizes": [(8,), (16,), (16, 16)],
                             "mlpclassifier__alpha": [1e-4, 1e-2]}, cv=cv),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prepare(cfg, log=print) -> dict:
    """Build datasets, train all models in simulation and write data/processed + results."""
    seed, names = cfg["experiment"]["seed"], cfg["data"]["datasets"]
    specs = model_specs(cfg)
    datasets = {n: build_dataset(n, cfg) for n in names}
    cfg.processed.mkdir(parents=True, exist_ok=True)
    save_datasets(datasets, cfg.processed / "datasets.npz")
    m, nb = cfg["models"], cfg["experiment"]["bootstrap"]
    models, feats, base_rows, sim_rows = {"datasets": {}}, {}, [], []
    for name, ds in datasets.items():
        X, y, sp = ds["X"], ds["y"], ds["splits"]
        tr, va, te = sp["train"], sp["val"], sp["test"]
        entry = {"hash": ds["hash"], "clip_fraction": ds["clip_fraction"], "heads": {}}
        for bname, est in _baselines(seed).items():
            est.fit(X[tr], y[tr])
            p = est.predict_proba(X[te])[:, 1]
            lo, hi = accuracy_ci(y[te], p, nb, seed)
            base_rows.append({"dataset": name, "model": bname, **classification_metrics(y[te], p),
                              "acc_lo": lo, "acc_hi": hi, "params": json.dumps(est.best_params_)})
        for spec in specs:
            theta = None
            if spec.n_params:
                theta, tlog = train_m1(spec, X[tr], y[tr], X[va], y[va], m["m1_steps"], m["m1_lr"],
                                       m["m1_restarts"], seed)
                entry["theta_M1"], entry["m1_log"] = theta.tolist(), tlog
            Fall = exact_features(spec, X, theta)
            feats[f"{name}__{spec.name}"] = Fall
            heads = {"R0": fit_head(Fall[tr], y[tr], "fixed", m["head_Cs"], seed),
                     "R1": fit_head(Fall[tr], y[tr], "batch", m["head_Cs"], seed)}
            entry["heads"][spec.name] = {k: h.to_dict() for k, h in heads.items()}
            p = heads["R0"].proba(Fall[te])
            lo, hi = accuracy_ci(y[te], p, nb, seed)
            sim_rows.append({"dataset": name, "model": spec.name, **classification_metrics(y[te], p),
                             "acc_lo": lo, "acc_hi": hi, "val_acc": float(np.mean((heads["R0"].proba(Fall[va]) >= .5) == y[va]))})
            log(f"{name:14s} {spec.name}  sim test acc {sim_rows[-1]['acc']:.3f}")
        models["datasets"][name] = entry
    np.savez_compressed(cfg.processed / "sim_features.npz", **feats)
    (cfg.processed / "models.json").write_text(json.dumps(models, indent=1))
    _write_csv(cfg.results / "baselines.csv", base_rows)
    _write_csv(cfg.results / "sim_metrics.csv", sim_rows)
    return {"baselines": base_rows, "sim": sim_rows}


def load_processed(cfg) -> dict:
    """Datasets, M1 parameters, heads and exact simulator features written by `prepare`."""
    names = cfg["data"]["datasets"]
    models = json.loads((cfg.processed / "models.json").read_text())
    datasets = load_datasets(cfg.processed / "datasets.npz", names)
    z = np.load(cfg.processed / "sim_features.npz")
    for name in names:
        if datasets[name]["hash"] != models["datasets"][name]["hash"]:
            raise RuntimeError(f"{name}: processed data hash mismatch; rerun `qdrift prepare`")
    return {"datasets": datasets, "models": models["datasets"],
            "sim_features": {k: z[k] for k in z.files}, "specs": model_specs(cfg)}


def theta_for(proc: dict, dataset: str, spec: ModelSpec):
    return np.asarray(proc["models"][dataset]["theta_M1"]) if spec.n_params else None


def head(proc: dict, dataset: str, model: str, kind: str) -> Head:
    return Head.from_dict(proc["models"][dataset]["heads"][model][kind])
