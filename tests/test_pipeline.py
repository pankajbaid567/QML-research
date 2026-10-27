"""Integration tests on fake backends: packed PUBs through SamplerV2, and a tiny dry-run campaign."""

import json

import numpy as np
import pytest

from qdrift.config import load_config

SMALL = {
    "experiment": {"bootstrap": 50},
    "data": {"n_train": 40, "n_val": 20, "n_cal_pool": 16, "n_test": 28},
    "models": {"m1_steps": 10, "m1_restarts": 1, "noise_aware_shots": 200, "refit_k": [8, 16]},
    "hardware": {"shots": 100, "readout_cal_shots": 200, "emulator_shots": 200, "max_patches": 4},
    "pilot": {"rows": 1},
    "dryrun": {"devices": ["fake_fez", "fake_marrakesh"], "sessions_per_device": 2},
}


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    from qdrift.train import load_processed, prepare
    root = tmp_path_factory.mktemp("qdrift")
    cfg = load_config(root=root, overrides=SMALL, dry_run=True)
    prepare(cfg, log=lambda *_: None)
    return cfg, load_processed(cfg)


def test_packed_pubs_run_on_sampler_and_unpack(small):
    from qiskit_ibm_runtime import SamplerV2

    from qdrift.device import fake_backend
    from qdrift.session import build_packed, make_layout, unpack
    cfg, proc = small
    backend = fake_backend("fake_fez")
    patches = make_layout(cfg, backend, "fake_fez")["patches"][:2]
    pubs, index = build_packed(cfg, proc, backend, patches, ["two_curves"], ["M1"])
    result = SamplerV2(mode=backend).run(pubs).result()
    counts = unpack(result, index, cfg["models"]["n_qubits"], len(patches))
    n_samples = cfg["data"]["n_cal_pool"] + cfg["data"]["n_test"]
    assert counts["two_curves__M1__z"].shape == (n_samples, 16)
    assert np.all(counts["two_curves__M1__z"].sum(1) == cfg["hardware"]["shots"])
    assert counts["readout__0"].shape == (2, 16) and counts["readout__0"][:, 0].mean() > 150


def test_dry_run_campaign_end_to_end(small):
    from qdrift.analysis import analyze, analyze_pilot
    from qdrift.budget import BudgetMeter
    from qdrift.session import noise_aware_counts, run_pilot, run_session, session_dirs, validate_session
    cfg, proc = small
    quiet = lambda *_: None
    pilot = analyze_pilot(cfg, proc, run_pilot(cfg, proc, "fake_fez", log=quiet))
    assert "two_curves__M1__z" in pilot and 0 <= pilot["two_curves__M1__z"]["acc_packed"] <= 1
    for device in cfg["dryrun"]["devices"]:
        noise_aware_counts(cfg, proc, device, log=quiet)
        for _ in range(cfg["dryrun"]["sessions_per_device"]):
            run_session(cfg, proc, device, log=quiet)
        dirs = session_dirs(cfg, device)
        assert len(dirs) == 2
        for d in dirs:
            validate_session(cfg, proc, d)
        meta = json.loads((dirs[1] / "meta.json").read_text())
        assert meta["dry_run"] and meta["session"] == 2 and "dry_run_drift" in json.loads(
            (dirs[1] / "snapshot.json").read_text())
    out = analyze(cfg, proc, log=quiet)
    met = out["tables"]["metrics"]
    for meth in ("SIM", "EMU", "R0", "R1", "R2_8", "R2_16", "R2_16_s1", "R3m", "R3r", "R4"):
        assert meth in set(met["method"]), meth
    cp = out["tables"]["conformal"]
    assert set(cp["method"]) == {"CP_sim", "CP_s1", "CP_same", "ACI"}
    assert cp["coverage"].between(0, 1).all()
    assert len(out["figures"]) == 6 and all(p.exists() for p in out["figures"])
    assert {"H1_emulator_overestimates", "H3_refit16_beats_normalisation"} <= set(out["hypotheses"])
    status = BudgetMeter(cfg.ledger, cfg["budget"], {}, dry_run=True).status()
    assert status["global"]["used_s"] == 0 and status["dry_run_jobs"] >= 5


class FakeJob:
    """Stand-in for a Runtime job: queued while interrupted, later DONE with a real result."""

    def __init__(self):
        self.state, self.payload, self.seconds = "QUEUED", None, 0.0

    def job_id(self):
        return "job-test-1"

    def status(self):
        return self.state

    def usage(self):
        return self.seconds if self.state in ("DONE", "ERROR", "CANCELLED") else 0

    def result(self):
        if self.state != "DONE":
            raise KeyboardInterrupt
        return self.payload


def test_interrupted_wait_keeps_reservation_and_fetch_saves(small, monkeypatch):
    from qiskit_ibm_runtime import SamplerV2

    import qdrift.session as S
    from qdrift.budget import BudgetMeter
    from qdrift.device import fake_backend, snapshot
    dry_cfg, proc = small
    cfg = load_config(root=dry_cfg.root, overrides=SMALL, dry_run=False)
    accounts = {"main": {"saved_account": "x", "limit_s": 600, "phases": ["pilot", "campaign", "rerun"]}}
    monkeypatch.setattr(S, "load_accounts", lambda *a: accounts)
    job = FakeJob()
    monkeypatch.setattr(S, "submit", lambda *a, **k: job)
    backend = fake_backend("fake_fez")
    patches = S.make_layout(dry_cfg, backend, "fake_fez")["patches"][:2]
    pubs, index = S.build_packed(cfg, proc, backend, patches, ["two_curves"], ["M1"])
    meta = S._meta(cfg, proc, "fake_fez", backend, "pilot", "main", patches, {"pilot": cfg["pilot"]})
    record = {"kind": "pilot", "device": "fake_fez", "phase": "pilot", "account": "main", "note": "t", "meta": meta}
    assert S.submit_and_wait(cfg, proc, backend, pubs, index, record, snapshot(backend), {}, lambda s: True,
                             lambda *_: None) is None
    meter = BudgetMeter(cfg.ledger, cfg["budget"], accounts)
    (rid,) = meter.status()["pending"]
    assert meter.entries[rid]["job_ids"] == ["job-test-1"] and len(S.pending_records(cfg)) == 1
    with pytest.raises(SystemExit, match="pending job"):
        S.ensure_no_pending(cfg, "fake_fez")
    job.state, job.seconds = "DONE", 7.0
    job.payload = SamplerV2(mode=backend).run(pubs).result()
    path = S.finalize(cfg, proc, S.pending_records(cfg)[0], job, log=lambda *_: None)
    assert path is not None and (path / "result.json").exists() and not S.pending_records(cfg)
    c = np.load(path / "counts.npz")["two_curves__M1__z"]
    assert np.all(c.sum(1) == cfg["hardware"]["shots"])
    meter = BudgetMeter(cfg.ledger, cfg["budget"], accounts)
    assert meter.status()["pending"] == [] and meter.status()["global"]["used_s"] == 7.0
