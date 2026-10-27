"""Model circuits.

One gate-list spec drives the NumPy simulator, the Qiskit builder and the feature
definitions, so training in simulation and execution on hardware use the same circuit.
Probability vectors follow Qiskit's little-endian order: bit q of the index is qubit q.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

# An angle is ("x", feature index) | ("p", trainable parameter index) | ("c", constant).
Angle = tuple


@dataclass(frozen=True)
class Op:
    gate: str
    qubits: tuple[int, ...]
    angle: Angle | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_qubits: int
    ops: tuple[Op, ...]
    n_params: int
    bases: tuple[str, ...]
    terms: tuple[tuple[int, ...], ...]

    @property
    def n_features(self) -> int:
        return len(self.bases) * len(self.terms)

    def feature_names(self) -> list[str]:
        return ["".join(f"{b.upper()}{q}" for q in t) for b in self.bases for t in self.terms]


BASIS_OPS = {"z": (), "x": ("h",), "y": ("sdg", "h")}


def basis_ops(n: int, basis: str) -> tuple[Op, ...]:
    return tuple(Op(g, (q,)) for q in range(n) for g in BASIS_OPS[basis])


def _chain(n: int) -> list[Op]:
    return [Op("cz", (q, q + 1)) for q in range(n - 1)]


def singles(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple((q,) for q in range(n))


def m1_reupload(n: int, layers: int) -> ModelSpec:
    """Data re-uploading classifier: RY(x) then trainable RZ, RY per qubit, CZ chain."""
    ops, k = [], 0
    for _ in range(layers):
        for q in range(n):
            ops += [Op("ry", (q,), ("x", q)), Op("rz", (q,), ("p", k)), Op("ry", (q,), ("p", k + 1))]
            k += 2
        ops += _chain(n)
    for q in range(n):
        ops.append(Op("ry", (q,), ("p", k)))
        k += 1
    return ModelSpec("M1", n, tuple(ops), k, ("z",), singles(n))


def m2_projected(n: int, layers: int) -> ModelSpec:
    """Fixed feature map; features are single-qubit X, Y and Z expectations (1-RDMs)."""
    ops = []
    for _ in range(layers):
        for q in range(n):
            ops += [Op("ry", (q,), ("x", q)), Op("rz", (q,), ("x", q))]
        ops += _chain(n)
    return ModelSpec("M2", n, tuple(ops), 0, ("x", "y", "z"), singles(n))


def m3_qelm(n: int, layers: int, seed: int) -> ModelSpec:
    """Quantum extreme learning machine: data, then fixed random layers; Z and ZZ features."""
    rng = np.random.default_rng(seed)
    ops = [Op("ry", (q,), ("x", q)) for q in range(n)]
    for layer in range(layers):
        for q in range(n):
            ops += [Op("ry", (q,), ("c", float(rng.uniform(0, 2 * np.pi)))),
                    Op("rz", (q,), ("c", float(rng.uniform(0, 2 * np.pi))))]
        ops += _chain(n)
        if layer < layers - 1:
            ops += [Op("rz", (q,), ("x", q)) for q in range(n)]
    return ModelSpec("M3", n, tuple(ops), 0, ("z",), singles(n) + tuple(combinations(range(n), 2)))


def model_specs(cfg) -> list[ModelSpec]:
    m = cfg["models"]
    n = m["n_qubits"]
    return [m1_reupload(n, m["m1_layers"]), m2_projected(n, m["m2_layers"]), m3_qelm(n, m["m3_layers"], m["m3_seed"])]


# ---------------------------------------------------------------- NumPy simulator

_FIXED = {
    "h": np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
    "sdg": np.diag([1, -1j]).astype(complex),
    "x": np.array([[0, 1], [1, 0]], complex),
}


def _rotation(gate: str, t: np.ndarray) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    m = np.zeros(t.shape + (2, 2), complex)
    if gate == "ry":
        m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1] = c, -s, s, c
    elif gate == "rz":
        m[..., 0, 0], m[..., 1, 1] = np.exp(-0.5j * t), np.exp(0.5j * t)
    elif gate == "rx":
        m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1] = c, -1j * s, -1j * s, c
    else:
        raise ValueError(f"unknown rotation {gate}")
    return m


def _angle_values(angle: Angle, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    kind, v = angle
    if kind == "x":
        return X[:, v]
    if kind == "p":
        return theta[:, v]
    return np.full(len(X), float(v))


def probabilities(spec: ModelSpec, X, theta=None, basis: str = "z") -> np.ndarray:
    """Exact outcome probabilities for a batch of inputs, shape (B, 2**n)."""
    X = np.atleast_2d(np.asarray(X, float))
    B, n = len(X), spec.n_qubits
    th = np.zeros((B, 0)) if spec.n_params == 0 else np.broadcast_to(np.atleast_2d(theta), (B, spec.n_params))
    psi = np.zeros((B,) + (2,) * n, complex)
    psi[(slice(None),) + (0,) * n] = 1.0
    # Axis n - q holds qubit q, so a C-order reshape gives the little-endian index.
    for op in spec.ops + basis_ops(n, basis):
        if op.gate == "cz":
            idx = [slice(None)] * (n + 1)
            idx[n - op.qubits[0]] = 1
            idx[n - op.qubits[1]] = 1
            psi[tuple(idx)] *= -1
            continue
        a = n - op.qubits[0]
        st = np.moveaxis(psi, a, -1)
        shape = st.shape
        if op.angle is None:
            st = st @ _FIXED[op.gate].T
        else:
            U = _rotation(op.gate, _angle_values(op.angle, X, th))
            st = (st.reshape(B, -1, 2) @ np.swapaxes(U, 1, 2)).reshape(shape)
        psi = np.moveaxis(st, -1, a)
    return np.abs(psi.reshape(B, -1)) ** 2


def sign_matrix(n: int, terms) -> np.ndarray:
    bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    return np.stack([np.prod(1 - 2 * bits[:, list(t)], axis=1) for t in terms]).astype(float)


def features_from_probs(spec: ModelSpec, probs_by_basis: dict) -> np.ndarray:
    """Pauli expectation features from (quasi-)probabilities, one block per basis."""
    S = sign_matrix(spec.n_qubits, spec.terms)
    return np.concatenate([np.atleast_2d(probs_by_basis[b]) @ S.T for b in spec.bases], axis=1)


def exact_features(spec: ModelSpec, X, theta=None) -> np.ndarray:
    return features_from_probs(spec, {b: probabilities(spec, X, theta, b) for b in spec.bases})


def feature_grad(spec: ModelSpec, X, theta):
    """Features and their exact parameter-shift gradients, shapes (B, F) and (P, B, F)."""
    X = np.atleast_2d(X)
    P, B = spec.n_params, len(X)
    E = np.eye(P) * (np.pi / 2)
    stack = np.vstack([theta[None], theta + E, theta - E])
    F = exact_features(spec, np.tile(X, (len(stack), 1)), np.repeat(stack, B, axis=0))
    F = F.reshape(len(stack), B, -1)
    return F[0], (F[1:P + 1] - F[P + 1:]) / 2


# ---------------------------------------------------------------- Qiskit builder

def append_model(qc, spec: ModelSpec, qubits, x, theta, basis: str, creg=None):
    """Append the spec on physical `qubits` (logical q -> qubits[q]); measure into `creg`."""
    for op in spec.ops + basis_ops(spec.n_qubits, basis):
        qs = [qubits[q] for q in op.qubits]
        if op.gate == "cz":
            qc.cz(*qs)
        elif op.angle is None:
            getattr(qc, op.gate)(qs[0])
        else:
            kind, v = op.angle
            value = x[v] if kind == "x" else theta[v] if kind == "p" else float(v)
            getattr(qc, op.gate)(value, qs[0])
    if creg is not None:
        qc.measure(list(qubits), list(creg))
    return qc
