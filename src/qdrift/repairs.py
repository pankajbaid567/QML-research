"""Classical heads and the low-cost repairs R1-R5.

R0 simulator head as trained       R3 tensored readout mitigation (measured or reported)
R1 QuantumNAT-style normalisation  R4 head trained on emulated noisy features
R2 anchored refit on k labels      R5 split and adaptive conformal prediction
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class Head:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    mode: str = "fixed"   # "fixed": stored statistics; "batch": statistics of the batch scored (R1)
    C: float = 1.0

    def standardize(self, F):
        F = np.atleast_2d(F)
        if self.mode == "batch":
            return (F - F.mean(0)) / (F.std(0) + 1e-9)
        return (F - self.mean) / self.scale

    def proba(self, F) -> np.ndarray:
        return expit(self.standardize(F) @ self.coef + self.intercept)

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Head":
        return cls(**{k: (np.asarray(v) if k in ("mean", "scale", "coef") else v) for k, v in d.items()})


def fit_head(F, y, mode: str = "fixed", Cs=(0.01, 0.1, 1.0, 10.0, 100.0), seed: int = 0) -> Head:
    F = np.asarray(F, float)
    mean, scale = F.mean(0), F.std(0) + 1e-9
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    lr = LogisticRegressionCV(Cs=list(Cs), cv=cv, scoring="neg_log_loss", max_iter=5000, l1_ratios=(0.0,),
                              use_legacy_attributes=False)
    lr.fit((F - mean) / scale, y)
    return Head(mean, scale, np.ravel(lr.coef_).copy(), float(np.ravel(lr.intercept_)[0]), mode,
                float(np.ravel(lr.C_)[0]))


def anchored_refit(head: Head, F, y, lam: float) -> Head:
    """R2: refit on k labelled hardware samples, penalising distance to the simulator head."""
    A = np.c_[head.standardize(F), np.ones(len(F))]
    y = np.asarray(y, float)
    w0 = np.r_[head.coef, head.intercept]

    def loss(v):
        z = A @ v
        val = np.sum(np.logaddexp(0, z) - y * z) + lam * np.sum((v - w0) ** 2)
        return val, A.T @ (expit(z) - y) + 2 * lam * (v - w0)

    v = minimize(loss, w0, jac=True, method="L-BFGS-B").x
    return replace(head, coef=v[:-1], intercept=float(v[-1]))


# ---------------------------------------------------------------- readout mitigation (R3)

def readout_from_cal(zeros, ones, n: int):
    """Per-qubit e0 = P(read 1 | prep 0) and e1 = P(read 0 | prep 1) from two cal circuits."""
    bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    p0, p1 = np.asarray(zeros, float), np.asarray(ones, float)
    p0, p1 = p0 / p0.sum(), p1 / p1.sum()
    return p0 @ bits, p1 @ (1 - bits)


def mitigate(probs, e0, e1) -> np.ndarray:
    """Apply the inverse of independent per-qubit confusion matrices (quasi-probabilities)."""
    probs = np.atleast_2d(np.asarray(probs, float))
    B, n = len(probs), len(e0)
    t = probs.reshape((B,) + (2,) * n)
    for q in range(n):
        A = np.array([[1 - e0[q], e1[q]], [e0[q], 1 - e1[q]]])
        a = n - q
        t = np.moveaxis(np.moveaxis(t, a, -1) @ np.linalg.inv(A).T, -1, a)
    return t.reshape(B, -1)


# ---------------------------------------------------------------- conformal prediction (R5)

def nonconformity(p1, y) -> np.ndarray:
    p1 = np.asarray(p1, float)
    return np.where(np.asarray(y) == 1, 1 - p1, p1)


def conformal_quantile(scores, alpha: float) -> float:
    s = np.sort(np.asarray(scores, float))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    if alpha <= 0 or k > len(s):
        return np.inf
    if k < 1:
        return -np.inf
    return float(s[k - 1])


def prediction_sets(p1, qhat: float) -> np.ndarray:
    """Boolean (B, 2): label 0 is in the set if p1 <= qhat, label 1 if 1 - p1 <= qhat."""
    p1 = np.asarray(p1, float)
    return np.stack([p1 <= qhat, 1 - p1 <= qhat], axis=1)


def set_stats(sets, y) -> tuple[float, float]:
    covered = sets[np.arange(len(y)), np.asarray(y, int)]
    return float(covered.mean()), float(sets.sum(1).mean())


def aci_stream(p1, y, cal_scores, alpha: float, gamma: float, update: bool = False):
    """Adaptive conformal inference (Gibbs & Candes 2021) over an ordered test stream.

    With update=True each revealed test score joins the calibration set (as in AQCP).
    """
    a_t, covered, sizes = alpha, [], []
    scores = list(np.asarray(cal_scores, float))
    for p, label in zip(np.asarray(p1, float), np.asarray(y, int)):
        q = conformal_quantile(scores, a_t) if a_t < 1 else -np.inf
        s = prediction_sets([p], q)[0]
        err = 0.0 if s[label] else 1.0
        covered.append(1 - err)
        sizes.append(int(s.sum()))
        a_t += gamma * (alpha - err)
        if update:
            scores.append(float(nonconformity([p], [label])[0]))
    return np.array(covered), np.array(sizes)
