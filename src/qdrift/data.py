"""Datasets, fixed stratified splits and angle scaling."""

from __future__ import annotations

import hashlib
import zlib

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SPLITS = ("train", "val", "cal", "test")


def two_curves(n_samples: int, n_features: int, degree: int, offset: float, noise: float, rng):
    """'Two curves' from XanaduAI/qml-benchmarks (Bowles, Ahmed & Schuld 2024, Apache-2.0).

    Re-implemented with a seeded generator instead of NumPy's global state. Each feature is
    a normalised Fourier series of a point t on [0, 1]; class 1 is class 0 shifted by
    `offset`. Labels are mapped from {-1, 1} to {0, 1}.
    """
    coeffs = rng.uniform(size=(n_features, degree + 1, 2))
    coeffs /= np.linalg.norm(coeffs)
    freqs = np.arange(1, degree + 1)

    def curve(t):
        ang = np.pi * np.outer(t, freqs)
        out = coeffs[:, 0, 0] + coeffs[:, 0, 1] + np.cos(ang) @ coeffs[:, 1:, 0].T + np.sin(ang) @ coeffs[:, 1:, 1].T
        return out + rng.normal(scale=noise, size=out.shape)

    half = n_samples // 2
    X = np.vstack([curve(rng.random(half)), curve(rng.random(half)) + offset])
    y = np.r_[np.zeros(half, int), np.ones(half, int)]
    return StandardScaler().fit_transform(X), y


def stratified_splits(y: np.ndarray, sizes: dict[str, int], rng) -> dict[str, np.ndarray]:
    """Fixed splits. Test and cal are drawn first so changing n_train never moves them.

    The cal pool alternates classes, so its first k samples are balanced for every k.
    """
    pools = [list(rng.permutation(np.flatnonzero(y == c))) for c in (0, 1)]
    frac1 = np.mean(y == 1)
    out = {}
    for name in ("test", "cal", "val", "train"):
        m = sizes[name]
        n1 = int(round(m * frac1)) if name != "cal" else m // 2
        take = [pools[0][: m - n1], pools[1][:n1]]
        if len(take[0]) < m - n1 or len(take[1]) < n1:
            raise ValueError(f"not enough samples for split {name}")
        pools = [pools[0][m - n1:], pools[1][n1:]]
        if name == "cal":
            a, b = take
            idx = [v for pair in zip(a, b) for v in pair] + a[len(b):] + b[len(a):]
        else:
            idx = list(rng.permutation(take[0] + take[1]))
        out[name] = np.asarray(idx, int)
    return out


def _angles(X: np.ndarray, train_idx: np.ndarray):
    lo, hi = X[train_idx].min(0), X[train_idx].max(0)
    A = (X - lo) / np.where(hi > lo, hi - lo, 1.0) * np.pi
    clipped = float(np.mean((A < 0) | (A > np.pi)))
    return np.clip(A, 0.0, np.pi), clipped


def dataset_hash(X, y, splits) -> str:
    h = hashlib.sha256(np.round(X, 10).tobytes() + np.asarray(y).tobytes())
    for name in SPLITS:
        h.update(name.encode() + np.asarray(splits[name]).tobytes())
    return h.hexdigest()


def build_dataset(name: str, cfg) -> dict:
    d, n = cfg["data"], cfg["models"]["n_qubits"]
    sizes = {s: d[f"n_{s}" if s != "cal" else "n_cal_pool"] for s in SPLITS}
    total = sum(sizes.values())
    rng = np.random.default_rng([cfg["experiment"]["seed"], zlib.crc32(name.encode())])
    if name == "two_curves":
        p = d["two_curves"]
        X, y = two_curves(total + total % 2, n, p["degree"], p["offset"], p["noise"], rng)
        splits = stratified_splits(y, sizes, rng)
    elif name == "breast_cancer":
        raw = load_breast_cancer()
        X0, y = raw.data, raw.target
        splits = stratified_splits(y, sizes, rng)
        tr = splits["train"]
        sc = StandardScaler().fit(X0[tr])
        pca = PCA(n_components=n, random_state=0).fit(sc.transform(X0[tr]))
        X = pca.transform(sc.transform(X0))
    else:
        raise ValueError(f"unknown dataset {name}")
    A, clipped = _angles(X, splits["train"])
    return {"name": name, "X": A, "y": np.asarray(y, int), "splits": splits,
            "clip_fraction": clipped, "hash": dataset_hash(A, y, splits)}


def save_datasets(datasets: dict, path) -> None:
    arrays = {}
    for name, ds in datasets.items():
        arrays[f"{name}__X"], arrays[f"{name}__y"] = ds["X"], ds["y"]
        for s in SPLITS:
            arrays[f"{name}__{s}"] = ds["splits"][s]
    np.savez_compressed(path, **arrays)


def load_datasets(path, names) -> dict:
    z = np.load(path)
    out = {}
    for name in names:
        X, y = z[f"{name}__X"], z[f"{name}__y"]
        splits = {s: z[f"{name}__{s}"] for s in SPLITS}
        out[name] = {"name": name, "X": X, "y": y, "splits": splits, "hash": dataset_hash(X, y, splits)}
    return out
