"""Device side: calibration snapshots, patch packing, circuit templates and emulation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .circuits import ModelSpec, append_model

FAKES = {"fake_fez": "FakeFez", "fake_marrakesh": "FakeMarrakesh", "fake_kingston": "FakeKingston",
         "fake_aachen": "FakeAachen", "fake_torino": "FakeTorino"}


def fake_backend(name: str):
    import qiskit_ibm_runtime.fake_provider as fp
    return getattr(fp, FAKES[name])()


def two_qubit_gate(target) -> str:
    for g in ("cz", "ecr", "cx"):
        if g in target.operation_names:
            return g
    raise ValueError("no supported two-qubit gate")


def snapshot(backend) -> dict:
    """Calibration data available before submission (target plus reported properties)."""
    t, n = backend.target, backend.num_qubits
    g2 = two_qubit_gate(t)
    try:
        props = backend.properties()
    except Exception:
        props = None

    def prop(q, name):
        try:
            return float(props.qubit_property(q, name)[0])
        except Exception:
            return None

    qubits = []
    for q in range(n):
        qp = t.qubit_properties[q] if t.qubit_properties else None
        meas = t["measure"].get((q,)) if "measure" in t else None
        sx = t["sx"].get((q,)) if "sx" in t else None
        qubits.append({"q": q, "t1": getattr(qp, "t1", None), "t2": getattr(qp, "t2", None),
                       "readout_error": getattr(meas, "error", None), "sx_error": getattr(sx, "error", None),
                       "prob_meas1_prep0": prop(q, "prob_meas1_prep0"),
                       "prob_meas0_prep1": prop(q, "prob_meas0_prep1")})
    edges = [{"a": a, "b": b, "error": getattr(p, "error", None)} for (a, b), p in t[g2].items()]
    return {"backend": backend.name, "num_qubits": n, "two_qubit_gate": g2,
            "last_update": str(getattr(props, "last_update_date", None)) if props else None,
            "captured_utc": datetime.now(timezone.utc).isoformat(), "qubits": qubits, "edges": edges}


def snapshot_from_properties(backend, props, when: str) -> dict:
    """Same schema as `snapshot`, built from historical BackendProperties (e.g. at run time)."""
    g2 = two_qubit_gate(backend.target)

    def qp(q, name):
        try:
            return float(props.qubit_property(q, name)[0])
        except Exception:
            return None

    def ge(gate, qubits):
        try:
            return float(props.gate_error(gate, qubits))
        except Exception:
            return None

    qubits = [{"q": q, "t1": qp(q, "T1"), "t2": qp(q, "T2"), "readout_error": qp(q, "readout_error"),
               "sx_error": ge("sx", [q]), "prob_meas1_prep0": qp(q, "prob_meas1_prep0"),
               "prob_meas0_prep1": qp(q, "prob_meas0_prep1")} for q in range(backend.num_qubits)]
    edges = [{"a": a, "b": b, "error": ge(g2, [a, b])} for (a, b) in backend.target[g2]]
    return {"backend": backend.name, "num_qubits": backend.num_qubits, "two_qubit_gate": g2,
            "last_update": str(props.last_update_date), "properties_at": when,
            "captured_utc": datetime.now(timezone.utc).isoformat(), "qubits": qubits, "edges": edges}


def _graph(snap: dict):
    adj = {q["q"]: set() for q in snap["qubits"]}
    err = {}
    for e in snap["edges"]:
        a, b = e["a"], e["b"]
        adj[a].add(b)
        adj[b].add(a)
        key = frozenset((a, b))
        if e["error"] is not None:
            err[key] = min(err.get(key, 1.0), e["error"])
    return adj, err


def plan_patches(snap: dict, n: int, max_patches: int, buffer: int = 1,
                 max_readout_error: float = 0.1, max_twoq_error: float = 0.05) -> list[list[int]]:
    """Greedy choice of the best disjoint n-qubit paths, kept `buffer` idle qubits apart."""
    adj, err = _graph(snap)
    ro = {q["q"]: q["readout_error"] for q in snap["qubits"]}
    good = {q for q, e in ro.items() if e is not None and e <= max_readout_error}
    paths = []

    def extend(path):
        if len(path) == n:
            if path[0] < path[-1]:
                paths.append(tuple(path))
            return
        for nb in adj[path[-1]]:
            if nb in good and nb not in path and err.get(frozenset((path[-1], nb)), 1.0) <= max_twoq_error:
                extend(path + [nb])

    for q in sorted(good):
        extend([q])
    scored = sorted(paths, key=lambda p: (sum(ro[q] for q in p) + sum(err[frozenset(e)] for e in zip(p, p[1:])), p))
    blocked, chosen = set(), []
    for p in scored:
        if len(chosen) == max_patches:
            break
        if blocked.isdisjoint(p):
            chosen.append(list(p))
            ring = set(p)
            for _ in range(buffer):
                ring |= {nb for q in ring for nb in adj[q]}
            blocked |= ring
    return chosen


def patch_edges(patches) -> set[frozenset]:
    return {frozenset(e) for p in patches for e in zip(p, p[1:])}


# ---------------------------------------------------------------- templates

def model_template(spec: ModelSpec, basis: str, patches, slots, num_qubits: int) -> QuantumCircuit:
    """One circuit running the model on several patches; slot k reads into register s{k}."""
    qc = QuantumCircuit(QuantumRegister(num_qubits, "q"),
                        *[ClassicalRegister(spec.n_qubits, f"s{k}") for k in slots])
    for k, patch in zip(slots, patches):
        x = ParameterVector(f"x{k}", spec.n_qubits)
        th = ParameterVector(f"t{k}", spec.n_params) if spec.n_params else []
        append_model(qc, spec, patch, x, th, basis, qc.cregs[slots.index(k)])
    return qc


def readout_template(patches, slots, n: int, num_qubits: int, ones: bool) -> QuantumCircuit:
    qc = QuantumCircuit(QuantumRegister(num_qubits, "q"), *[ClassicalRegister(n, f"s{k}") for k in slots])
    for i, patch in enumerate(patches):
        if ones:
            qc.x(patch)
        qc.measure(patch, qc.cregs[i])
    return qc


def slot_values(spec: ModelSpec, slot: int, x, theta) -> dict[str, float]:
    vals = {f"x{slot}[{j}]": float(v) for j, v in enumerate(x)}
    if spec.n_params:
        vals.update({f"t{slot}[{j}]": float(v) for j, v in enumerate(theta)})
    return vals


def bind_array(isa: QuantumCircuit, rows: list[dict[str, float]]) -> np.ndarray:
    names = [p.name for p in isa.parameters]
    return np.array([[r[name] for name in names] for r in rows], float)


def transpile_fixed(circuits, backend, opt_level: int, seed: int):
    """Translate to native gates with the identity layout, so no qubit is moved."""
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend,
                                      initial_layout=list(range(backend.num_qubits)), seed_transpiler=seed)
    return pm.run(circuits)


def verify_isa(isa: QuantumCircuit, patches) -> None:
    allowed = {q for p in patches for q in p}
    edges = patch_edges(patches)
    for inst in isa.data:
        qs = [isa.find_bit(q).index for q in inst.qubits]
        if not set(qs) <= allowed:
            raise AssertionError(f"{inst.operation.name} touches qubits {qs} outside the patches")
        if len(qs) == 2 and frozenset(qs) not in edges:
            raise AssertionError(f"two-qubit gate on {qs} is not a patch edge")
        if len(qs) > 2:
            raise AssertionError(f"unexpected {len(qs)}-qubit operation {inst.operation.name}")


# ---------------------------------------------------------------- emulation

def calibration_emulator(backend) -> AerSimulator:
    """Emulator built only from the backend's reported calibration (RQ1 prediction)."""
    return AerSimulator.from_backend(backend, method="density_matrix")


def drifted_emulator(backend, patches, ro_extra: float, twoq_extra: float) -> AerSimulator:
    """Dry-run stand-in for hardware: the calibration noise plus unreported extra noise."""
    nm = NoiseModel.from_backend(backend)
    g2 = two_qubit_gate(backend.target)
    for q in {q for p in patches for q in p}:
        nm.add_readout_error(ReadoutError([[1 - ro_extra, ro_extra], [ro_extra, 1 - ro_extra]]), [q], warnings=False)
    for a, b in (tuple(e) for e in patch_edges(patches)):
        for pair in ((a, b), (b, a)):
            if pair in backend.target[g2]:
                nm.add_quantum_error(depolarizing_error(twoq_extra, 2), [g2], list(pair), warnings=False)
    return AerSimulator(method="density_matrix", noise_model=nm)


def counts_vector(counts: dict[str, int], n: int) -> list[int]:
    """Counts as a list indexed by the little-endian outcome integer (bit q = qubit q)."""
    out = [0] * (2**n)
    for key, c in counts.items():
        out[int(key.replace(" ", ""), 2)] += int(c)
    return out


def run_bound(sim: AerSimulator, circuits, shots: int, seed: int) -> list[list[int]]:
    if not circuits:
        return []
    res = sim.run(circuits, shots=shots, seed_simulator=seed).result()
    n = circuits[0].cregs[0].size
    return [counts_vector(res.get_counts(i), n) for i in range(len(circuits))]
