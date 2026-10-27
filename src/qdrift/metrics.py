"""Classification, calibration and bootstrap metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def ece(y, p1, bins: int = 10) -> float:
    """Expected calibration error of p(y=1) with equal-width bins on [0, 1]."""
    y, p1 = np.asarray(y, float), np.asarray(p1, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p1, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p1[m].mean())
    return float(total)


def classification_metrics(y, p1) -> dict[str, float]:
    y, p1 = np.asarray(y, int), np.asarray(p1, float)
    pred = (p1 >= 0.5).astype(int)
    return {
        "acc": float(np.mean(pred == y)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p1)) if len(np.unique(y)) == 2 else float("nan"),
        "ece": ece(y, p1),
        "brier": float(np.mean((p1 - y) ** 2)),
    }


def bootstrap_ci(stat, n: int, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of stat(indices) over resampled index sets (pass a closure)."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))


def accuracy_ci(y, p1, n_boot: int, seed: int):
    correct = ((np.asarray(p1) >= 0.5).astype(int) == np.asarray(y)).astype(float)
    return bootstrap_ci(lambda i: correct[i].mean(), len(correct), n_boot, seed)
