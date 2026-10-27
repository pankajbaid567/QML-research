"""Unit tests: circuits, data splits, repairs, metrics and the budget meter."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qdrift.budget import BudgetExceeded, BudgetMeter, estimate_seconds
from qdrift.circuits import append_model, exact_features, feature_grad, model_specs, probabilities
from qdrift.config import ROOT, load_config
from qdrift.data import build_dataset, stratified_splits
from qdrift.metrics import ece
from qdrift.repairs import (aci_stream, conformal_quantile, fit_head, mitigate, prediction_sets,
                            readout_from_cal, set_stats)

CFG = load_config()


@pytest.mark.parametrize("spec", model_specs(CFG), ids=lambda s: s.name)
def test_numpy_simulator_matches_qiskit(spec):
    rng = np.random.default_rng(1)
    x, th = rng.uniform(0, np.pi, spec.n_qubits), rng.uniform(-3, 3, spec.n_params)
    for b in spec.bases:
        qc = append_model(QuantumCircuit(spec.n_qubits), spec, list(range(spec.n_qubits)), x, th, b)
        ref = Statevector(qc).probabilities()
        got = probabilities(spec, x, th if spec.n_params else None, b)[0]
        assert np.allclose(ref, got, atol=1e-10)


def test_parameter_shift_matches_finite_difference():
    spec = model_specs(CFG)[0]
    rng = np.random.default_rng(2)
    X, th = rng.uniform(0, np.pi, (3, spec.n_qubits)), rng.uniform(-3, 3, spec.n_params)
    _, dz = feature_grad(spec, X, th)
    for k in (0, 7, spec.n_params - 1):
        e = np.zeros_like(th)
        e[k] = 1e-6
        fd = (exact_features(spec, X, th + e) - exact_features(spec, X, th - e)) / 2e-6
        assert np.allclose(fd, dz[k], atol=1e-6)


def test_splits_are_reproducible_disjoint_and_balanced():
    a, b = build_dataset("two_curves", CFG), build_dataset("two_curves", CFG)
    assert a["hash"] == b["hash"]
    sp = a["splits"]
    allidx = np.concatenate([sp[s] for s in sp])
    assert len(allidx) == len(set(allidx))
    cal_y = a["y"][sp["cal"]]
    for k in (8, 16, 32):
        assert cal_y[:k].sum() == k // 2


def test_test_split_does_not_move_when_train_size_changes():
    y = np.r_[np.zeros(300, int), np.ones(300, int)]
    sizes = {"train": 100, "val": 50, "cal": 32, "test": 150}
    s1 = stratified_splits(y, sizes, np.random.default_rng(0))
    s2 = stratified_splits(y, {**sizes, "train": 200}, np.random.default_rng(0))
    assert np.array_equal(s1["test"], s2["test"]) and np.array_equal(s1["cal"], s2["cal"])


def test_readout_mitigation_recovers_known_distribution():
    n, rng = 3, np.random.default_rng(3)
    true = rng.dirichlet(np.ones(2**n))
    e0, e1 = np.array([0.02, 0.05, 0.03]), np.array([0.04, 0.06, 0.08])
    t = true.reshape((2,) * n)            # axis n-1-q holds qubit q (little-endian index)
    for q in range(n):                    # forward model: independent per-qubit confusion
        A = np.array([[1 - e0[q], e1[q]], [e0[q], 1 - e1[q]]])
        t = np.moveaxis(np.moveaxis(t, n - 1 - q, -1) @ A.T, -1, n - 1 - q)
    noisy = t.reshape(-1)
    assert not np.allclose(noisy, true)
    assert np.allclose(mitigate(noisy, e0, e1)[0], true, atol=1e-10)


def test_readout_from_calibration_counts():
    n = 2
    zeros = np.array([90, 6, 4, 0])     # index bit q = qubit q
    ones = np.array([0, 3, 7, 90])
    e0, e1 = readout_from_cal(zeros, ones, n)
    assert np.allclose(e0, [0.06, 0.04]) and np.allclose(e1, [0.07, 0.03])


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(4)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    scores = np.where(y[:5000] == 1, 1 - p[:5000], p[:5000])
    q = conformal_quantile(scores, 0.1)
    cov, _ = set_stats(prediction_sets(p[5000:], q), y[5000:])
    assert abs(cov - 0.9) < 0.02
    covered, _ = aci_stream(p[5000:8000], y[5000:8000], scores[:200], 0.1, 0.03, update=True)
    assert abs(covered.mean() - 0.9) < 0.03


def test_ece_reference_values():
    assert ece([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == 0
    assert abs(ece([1, 1, 0, 0], [0.75] * 4) - 0.25) < 1e-12


def test_head_batch_mode_removes_scale_and_shift():
    rng = np.random.default_rng(5)
    F = rng.normal(size=(300, 4))
    y = (F @ np.array([1.0, -1.0, 0.5, 0.0]) > 0).astype(int)
    h = fit_head(F, y, mode="batch")
    assert np.allclose(h.proba(F), h.proba(0.6 * F + 0.2), atol=1e-9)


def _meter(tmp_path, accounts=None):
    budget = {**CFG["budget"], "phase_caps": dict(CFG["budget"]["phase_caps"])}
    accounts = accounts or {"main": {"saved_account": "x", "limit_s": 600, "phases": ["pilot", "campaign", "rerun"]}}
    return BudgetMeter(tmp_path / "ledger.jsonl", budget, accounts)


def test_meter_blocks_over_budget_and_unknown_accounts(tmp_path):
    m = _meter(tmp_path)
    rid = m.reserve("main", "pilot", 80, "ibm_fez")
    with pytest.raises(BudgetExceeded, match="phase 'pilot'"):
        m.reserve("main", "pilot", 20, "ibm_fez")
    m.settle(rid, 30)
    m.reserve("main", "pilot", 50, "ibm_fez")           # actual usage freed the estimate
    with pytest.raises(BudgetExceeded, match="not approved"):
        m.reserve("second", "campaign", 1, "ibm_fez")
    with pytest.raises(BudgetExceeded, match="phase 'addon'"):
        m.reserve("main", "addon", 1, "ibm_fez")


def test_meter_account_window_and_global_cap(tmp_path):
    acc = {"main": {"saved_account": "x", "limit_s": 600, "phases": ["campaign", "rerun", "addon"]},
           "classroom": {"saved_account": "y", "limit_s": 600, "phases": ["addon"]}}
    m = _meter(tmp_path, acc)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    m.settle(m.reserve("main", "campaign", 390, "ibm_fez", now=old), 390)
    m.settle(m.reserve("main", "rerun", 100, "ibm_fez", now=old), 100)
    st = m.status()
    assert st["accounts"]["main"]["used_28d_s"] == 0        # the 28-day window rolled over
    m.settle(m.reserve("main", "addon", 500, "ibm_fez"), 500)
    with pytest.raises(BudgetExceeded, match="global cap"):
        m.reserve("classroom", "addon", 300, "ibm_fez")     # 990 + 300 > 1200


def test_meter_persists_and_dry_run_costs_nothing(tmp_path):
    m = _meter(tmp_path)
    rid = m.reserve("main", "campaign", 30, "ibm_fez")
    m.attach(rid, ["job1"])
    m2 = _meter(tmp_path)
    assert m2.status()["pending"] == [rid] and m2.entries[rid]["job_ids"] == ["job1"]
    d = BudgetMeter(tmp_path / "ledger.jsonl", CFG["budget"], {}, dry_run=True)
    d.settle(d.reserve("anything", "campaign", 999, "fake_fez"), 0)
    assert d.status()["global"]["used_s"] == 30


def test_estimate_formula():
    assert estimate_seconds(10000, 1, 0.00035, 2.0, 1.0) == pytest.approx(5.5)


def test_no_ibm_key_in_repository():
    import re
    pattern = re.compile(r"(token|api_?key|apikey)\s*[=:]\s*['\"][A-Za-z0-9_\-]{40,}['\"]", re.I)
    skip = {".venv", ".git", "data", "results", "node_modules"}
    for p in ROOT.rglob("*"):
        if p.is_file() and not skip & set(p.parts) and p.suffix in {".py", ".toml", ".md", ".json", ".txt", ".tex"}:
            assert not pattern.search(p.read_text(errors="ignore")), f"possible credential in {p}"
    ignored = (ROOT / ".gitignore").read_text()
    assert "config/accounts.toml" in ignored and "qiskit-ibm.json" in ignored
