"""Sessions: packed sampler PUBs, emulation, hardware submission and raw-data storage.

Packing rule: hardware sample i of a dataset runs on row i // P, slot (patch) i % P, so a
sample sits on the same physical qubits in every session of a device.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .budget import BudgetMeter, estimate_seconds
from .circuits import ModelSpec
from .config import ROOT, load_accounts
from .device import (calibration_emulator, counts_vector, drifted_emulator, fake_backend, model_template,
                     plan_patches, readout_template, run_bound, slot_values, snapshot, snapshot_from_properties,
                     transpile_fixed, verify_isa, bind_array)
from .train import theta_for

SAMPLER_OPTIONS = {"dynamical_decoupling": False, "twirling_gates": False, "twirling_measure": False}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def key(dataset: str, model: str, basis: str) -> str:
    return f"{dataset}__{model}__{basis}"


def circuit_types(specs, models=None) -> list[tuple[ModelSpec, str]]:
    return [(s, b) for s in specs for b in s.bases if models is None or s.name in models]


def sample_order(ds) -> np.ndarray:
    """Hardware samples of a dataset: the labelled calibration pool, then the fixed test set."""
    return np.r_[ds["splits"]["cal"], ds["splits"]["test"]].astype(int)


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def config_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg.data, sort_keys=True).encode()).hexdigest()


def get_backend(device: str, dry_run: bool, account: str | None = None):
    if dry_run:
        return fake_backend(device)
    accounts = load_accounts()
    if account not in accounts:
        raise SystemExit(f"account '{account}' is not listed in config/accounts.toml")
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(name=accounts[account]["saved_account"]).backend(device)


# ---------------------------------------------------------------- layout

def layout_path(cfg, device: str) -> Path:
    return cfg.layouts / f"{device}.json"


def load_layout(cfg, device: str) -> dict:
    path = layout_path(cfg, device)
    if not path.exists():
        raise SystemExit(f"no frozen layout for {device}; run `qdrift layout --device {device}` first")
    return json.loads(path.read_text())


def make_layout(cfg, backend, device: str, force: bool = False) -> dict:
    """Freeze the patch layout of a device from its current calibration (no QPU time)."""
    path = layout_path(cfg, device)
    if path.exists() and not force:
        return json.loads(path.read_text())
    hw, n = cfg["hardware"], cfg["models"]["n_qubits"]
    snap = snapshot(backend)
    patches = plan_patches(snap, n, hw["max_patches"], hw["patch_buffer"], hw["max_readout_error"],
                           hw["max_twoq_error"])
    if len(patches) < 4:
        raise RuntimeError(f"{device}: only {len(patches)} usable patches")
    layout = {"device": device, "backend": backend.name, "patches": patches, "created_utc": now_utc(),
              "snapshot": snap}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout, indent=1))
    return layout


# ---------------------------------------------------------------- circuits

def build_packed(cfg, proc, backend, patches, datasets=None, models=None, max_rows=None):
    """Packed PUBs (one per dataset x model x basis, plus two readout-calibration PUBs)."""
    hw, n = cfg["hardware"], cfg["models"]["n_qubits"]
    P, slots = len(patches), list(range(len(patches)))
    types = circuit_types(proc["specs"], models)
    templates = [model_template(s, b, patches, slots, backend.num_qubits) for s, b in types]
    ro = [readout_template(patches, slots, n, backend.num_qubits, ones) for ones in (False, True)]
    isa = transpile_fixed(templates + ro, backend, hw["optimization_level"], hw["seed_transpiler"])
    for circ in isa:
        verify_isa(circ, patches)
    pubs, index = [], []
    for name in datasets or cfg["data"]["datasets"]:
        ds = proc["datasets"][name]
        order = sample_order(ds)[: max_rows * P] if max_rows else sample_order(ds)
        rows = math.ceil(len(order) / P)
        for (spec, b), circ in zip(types, isa):
            theta = theta_for(proc, name, spec)
            values = []
            for r in range(rows):
                row = {}
                for k in slots:
                    i = r * P + k
                    s = order[i] if i < len(order) else order[0]  # padding, dropped on unpack
                    row.update(slot_values(spec, k, ds["X"][s], theta))
                values.append(row)
            pubs.append((circ, bind_array(circ, values), hw["shots"]))
            index.append({"kind": "model", "key": key(name, spec.name, b), "n": len(order), "rows": rows})
    for prep, circ in zip((0, 1), isa[len(types):]):
        pubs.append((circ, None, hw["readout_cal_shots"]))
        index.append({"kind": "readout", "key": f"readout__{prep}"})
    return pubs, index


def build_isolated(cfg, proc, backend, patches, dataset: str, models, n_samples: int):
    """Pilot PUBs: each sample alone on its own patch (no neighbouring patches active)."""
    hw = cfg["hardware"]
    P = len(patches)
    types = circuit_types(proc["specs"], models)
    singles = [model_template(s, b, [p], [k], backend.num_qubits) for s, b in types for k, p in enumerate(patches)]
    isa = transpile_fixed(singles, backend, hw["optimization_level"], hw["seed_transpiler"])
    ds = proc["datasets"][dataset]
    order = sample_order(ds)[:n_samples]
    pubs, index = [], []
    for t, (spec, b) in enumerate(types):
        theta = theta_for(proc, dataset, spec)
        for k in range(P):
            members = [i for i in range(len(order)) if i % P == k]
            if not members:
                continue
            circ = isa[t * P + k]
            verify_isa(circ, [patches[k]])
            values = [slot_values(spec, k, ds["X"][order[i]], theta) for i in members]
            pubs.append((circ, bind_array(circ, values), hw["shots"]))
            index.append({"kind": "isolated", "key": key(dataset, spec.name, b), "slot": k, "members": members,
                          "n": len(order)})
    return pubs, index


def executions(pubs) -> int:
    total = 0
    for circ, values, shots in pubs:
        rows = 1 if values is None else int(np.asarray(values).reshape(-1, max(1, len(circ.parameters))).shape[0])
        total += rows * shots
    return total


def unpack(result, index, n: int, P: int) -> dict[str, np.ndarray]:
    """Counts per sample (little-endian outcome index) from a SamplerV2 result."""
    out, iso = {}, {}
    for j, meta in enumerate(index):
        data = result[j].data
        if meta["kind"] == "readout":
            out[meta["key"]] = np.array([counts_vector(getattr(data, f"s{k}").get_counts(), n) for k in range(P)])
        elif meta["kind"] == "model":
            rows = [[counts_vector(getattr(data, f"s{k}").get_counts(loc=r), n) for k in range(P)]
                    for r in range(meta["rows"])]
            out[meta["key"]] = np.array(rows).reshape(-1, 2**n)[: meta["n"]]
        else:
            reg = getattr(data, f"s{meta['slot']}")
            arr = iso.setdefault(meta["key"], np.zeros((meta["n"], 2**n), int))
            for r, i in enumerate(meta["members"]):
                arr[i] = counts_vector(reg.get_counts(loc=r), n)
    out.update({f"isolated__{k}": v for k, v in iso.items()})
    return out


def emulate(cfg, proc, backend, patches, sim, shots: int, seed: int, orders: dict, readout_shots=None,
            models=None) -> dict[str, np.ndarray]:
    """Counts in the packed format, emulated one patch at a time (no crosstalk is modelled)."""
    hw, n = cfg["hardware"], cfg["models"]["n_qubits"]
    P = len(patches)
    types = circuit_types(proc["specs"], models)
    singles = [model_template(s, b, [p], [k], backend.num_qubits) for s, b in types for k, p in enumerate(patches)]
    isa = transpile_fixed(singles, backend, hw["optimization_level"], hw["seed_transpiler"])
    circuits, keys = [], []
    for name, order in orders.items():
        ds = proc["datasets"][name]
        for t, (spec, b) in enumerate(types):
            theta = theta_for(proc, name, spec)
            for i, s in enumerate(order):
                k = i % P
                circuits.append(isa[t * P + k].assign_parameters(slot_values(spec, k, ds["X"][s], theta)))
            keys.append((key(name, spec.name, b), len(order)))
    flat = run_bound(sim, circuits, shots, seed)
    out, pos = {}, 0
    for k_, m in keys:
        out[k_] = np.array(flat[pos:pos + m], int)
        pos += m
    if readout_shots:
        ro = [readout_template([p], [k], n, backend.num_qubits, ones) for ones in (False, True)
              for k, p in enumerate(patches)]
        ro_isa = transpile_fixed(ro, backend, hw["optimization_level"], hw["seed_transpiler"])
        c = np.array(run_bound(sim, ro_isa, readout_shots, seed + 1), int)
        out["readout__0"], out["readout__1"] = c[:P], c[P:]
    return out


# ---------------------------------------------------------------- hardware submission

def submit(backend, pubs, cfg, phase: str):
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    o = sampler.options
    o.max_execution_time = cfg["hardware"]["max_execution_time_s"]
    o.dynamical_decoupling.enable = SAMPLER_OPTIONS["dynamical_decoupling"]
    o.twirling.enable_gates = SAMPLER_OPTIONS["twirling_gates"]
    o.twirling.enable_measure = SAMPLER_OPTIONS["twirling_measure"]
    o.environment.job_tags = ["qdrift", phase]
    return sampler.run(pubs)


def pending_records(cfg) -> list[Path]:
    """Submitted jobs whose data are not saved yet (data/raw/pending/<job id>/)."""
    d = cfg.raw / "pending"
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.exists() else []


def _record(rec: Path) -> dict:
    return json.loads((rec / "record.json").read_text())


def ensure_no_pending(cfg, device: str) -> None:
    for rec in pending_records(cfg):
        if _record(rec)["device"] == device:
            raise SystemExit(f"{device} already has a pending job ({rec.name}); run `uv run qdrift fetch` first")


def submit_and_wait(cfg, proc, backend, pubs, index, record: dict, snap: dict, predicted: dict, confirm, log):
    """Reserve, confirm, submit one job, keep a pending record, then wait and save.

    Ctrl+C only stops the waiting: the job keeps its place on IBM, the reservation stays
    pending, and `qdrift fetch` saves the data and settles the usage later.
    """
    hw = cfg["hardware"]
    ensure_no_pending(cfg, record["device"])
    ex = executions(pubs)
    est = estimate_seconds(ex, 1, hw["per_execution_s"], hw["job_overhead_s"], cfg["budget"]["safety_factor"])
    meter = BudgetMeter(cfg.ledger, cfg["budget"], load_accounts())
    if not confirm(f"{backend.name}: {len(pubs)} PUBs, {ex} executions, estimate {est:.1f} s "
                   f"(phase {record['phase']}, account {record['account']})"):
        raise SystemExit("cancelled; no QPU time used")
    rid = meter.reserve(record["account"], record["phase"], est, backend.name, record["note"])
    try:
        job = submit(backend, pubs, cfg, record["phase"])
    except BaseException:
        meter.settle(rid, 0.0, "NOT_SUBMITTED")
        raise
    meter.attach(rid, [job.job_id()])
    rec = cfg.raw / "pending" / job.job_id()
    rec.mkdir(parents=True)
    (rec / "snapshot.json").write_text(json.dumps(snap, indent=1))
    if predicted:
        np.savez_compressed(rec / "emulator.npz", **predicted)
    record = {**record, "job_id": job.job_id(), "reservation": rid, "index": index,
              "meta": {**record["meta"], "executions": ex, "estimate_s": est}}
    (rec / "record.json").write_text(json.dumps(record, indent=1))
    log(f"submitted job {job.job_id()}; waiting. Ctrl+C is safe: the job stays queued and "
        f"`uv run qdrift fetch` saves it later.")
    try:
        job.result()
    except KeyboardInterrupt:
        log(f"\nstopped waiting; job {job.job_id()} is still on IBM. Run `uv run qdrift fetch` later.")
        return None
    except Exception as exc:
        log(f"job {job.job_id()} did not complete: {exc}")
    return finalize(cfg, proc, rec, job, log)


def finalize(cfg, proc, rec: Path, job, log=print) -> Path | None:
    """Save a finished job and settle its usage; move failed or cancelled jobs to data/raw/failed."""
    record = _record(rec)
    status = str(job.status())
    if status not in ("DONE", "ERROR", "CANCELLED"):
        log(f"job {record['job_id']} ({record['device']} {record['kind']}) is {status}; try again later")
        return None
    usage = float(job.usage() or 0.0)
    meter = BudgetMeter(cfg.ledger, cfg["budget"], load_accounts())
    if status != "DONE":
        meter.settle(record["reservation"], usage, "FAILED")
        failed = cfg.raw / "failed"
        failed.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rec), str(failed / rec.name))
        log(f"job {record['job_id']} {status}; settled at {usage:.1f} s")
        return None
    result = job.result()
    meter.settle(record["reservation"], usage, "DONE")
    from qiskit_ibm_runtime import RuntimeEncoder
    meta = record["meta"]
    counts = unpack(result, record["index"], meta["n_qubits"], len(meta["patches"]))
    predicted = dict(np.load(rec / "emulator.npz")) if (rec / "emulator.npz").exists() else {}
    snap = json.loads((rec / "snapshot.json").read_text())
    meta = {**meta, "job_ids": [record["job_id"]], "usage_s": usage, "completed_utc": now_utc()}
    # Open Plan jobs can queue for hours, so also keep the calibration in force when the job ran.
    run_snap = None
    try:
        ts = job.metrics().get("timestamps", {})
        meta["job_timestamps"] = ts
        if ts.get("running") and ts.get("created"):
            meta["queue_wait_s"] = (datetime.fromisoformat(ts["running"]) -
                                    datetime.fromisoformat(ts["created"])).total_seconds()
            b = job.backend()
            run_snap = snapshot_from_properties(b, b.properties(datetime=datetime.fromisoformat(ts["running"])),
                                                ts["running"])
    except Exception as exc:
        log(f"note: calibration at run time not saved ({exc})")
    session = record["kind"] == "session"
    path = _next_dir(cfg.raw / record["device"] if session else cfg.raw / "pilot" / record["device"],
                     "s" if session else "p")
    _save(path, counts, predicted, snap, meta, json.dumps(result, cls=RuntimeEncoder))
    if run_snap:
        (path / "snapshot_at_run.json").write_text(json.dumps(run_snap, indent=1))
    if session:
        validate_session(cfg, proc, path)
    shutil.rmtree(rec)
    log(f"saved {path.relative_to(cfg.root)} (usage {usage:.1f} s)")
    return path


def fetch_pending(cfg, proc, log=print) -> list[tuple[str, Path]]:
    """Save every pending job that has finished; leave the rest pending."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    accounts, saved = load_accounts(), []
    for rec in pending_records(cfg):
        record = _record(rec)
        job = QiskitRuntimeService(name=accounts[record["account"]]["saved_account"]).job(record["job_id"])
        path = finalize(cfg, proc, rec, job, log)
        if path:
            saved.append((record["kind"], path))
    return saved


# ---------------------------------------------------------------- sessions

def _next_dir(base: Path, prefix: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    k = len([p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]) + 1
    return base / f"{prefix}{k:02d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def dry_drift(cfg, device: str, session: int) -> dict:
    d = cfg["dryrun"]
    rng = np.random.default_rng([cfg["experiment"]["seed"], zlib.crc32(device.encode()), session])
    ro = max(0.0, rng.normal(d["readout_drift_mean"], d["readout_drift_sd"]))
    tq = max(0.0, rng.normal(d["twoq_drift_mean"], d["twoq_drift_sd"]))
    f = d["reported_fraction"]
    return {"ro_total": ro, "twoq_total": tq, "ro_reported": f * ro, "twoq_reported": f * tq}


def _report_drift(snap: dict, patches, drift: dict) -> dict:
    """Dry run: the snapshot shows only the reported share of the injected drift."""
    on = {q for p in patches for q in p}
    edges = {frozenset(e) for p in patches for e in zip(p, p[1:])}
    for q in snap["qubits"]:
        if q["q"] in on:
            for f in ("readout_error", "prob_meas1_prep0", "prob_meas0_prep1"):
                if q[f] is not None:
                    q[f] += drift["ro_reported"]
    for e in snap["edges"]:
        if frozenset((e["a"], e["b"])) in edges and e["error"] is not None:
            e["error"] += drift["twoq_reported"]
    snap["dry_run_drift"] = drift
    return snap


def _save(path: Path, counts: dict, predicted: dict, snap: dict, meta: dict, raw_json: str | None = None):
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / "counts.npz", **counts)
    if predicted:
        np.savez_compressed(path / "emulator.npz", **predicted)
    (path / "snapshot.json").write_text(json.dumps(snap, indent=1))
    (path / "meta.json").write_text(json.dumps(meta, indent=1))
    if raw_json:
        (path / "result.json").write_text(raw_json)


def _meta(cfg, proc, device, backend, phase, account, patches, extra: dict) -> dict:
    hw = cfg["hardware"]
    return {"device": device, "backend": backend.name, "phase": phase,
            "account": "dry-run" if cfg.dry_run else account,
            "dry_run": cfg.dry_run, "created_utc": now_utc(), "shots": hw["shots"],
            "readout_shots": hw["readout_cal_shots"], "emulator_shots": hw["emulator_shots"],
            "n_qubits": cfg["models"]["n_qubits"], "patches": patches, "sampler_options": SAMPLER_OPTIONS,
            "datasets": {n: {"hash": proc["datasets"][n]["hash"],
                             "order": sample_order(proc["datasets"][n]).tolist()} for n in cfg["data"]["datasets"]},
            "config_sha256": config_hash(cfg), "git_commit": _git_commit(), **extra}


def run_session(cfg, proc, device: str, phase: str = "campaign", account: str | None = None,
                confirm=None, log=print) -> Path:
    """One session: emulator prediction, then hardware (or the drifted emulator in a dry run).

    Returns the saved session folder, or None if the wait for hardware was interrupted.
    """
    hw = cfg["hardware"]
    if not cfg.dry_run:
        ensure_no_pending(cfg, device)
    backend = get_backend(device, cfg.dry_run, account)
    patches = (make_layout(cfg, backend, device) if cfg.dry_run else load_layout(cfg, device))["patches"]
    P, n = len(patches), cfg["models"]["n_qubits"]
    base = cfg.raw / device
    index_no = len(session_dirs(cfg, device)) + 1
    seed = cfg["experiment"]["seed"] + 1000 * index_no + zlib.crc32(device.encode()) % 997
    snap = snapshot(backend)
    orders = {name: sample_order(proc["datasets"][name]) for name in cfg["data"]["datasets"]}
    pubs, index = build_packed(cfg, proc, backend, patches)  # validates the ISA circuits in every mode
    extra = {"session": index_no, "pubs": len(pubs), "executions": executions(pubs)}
    if cfg.dry_run:
        drift = dry_drift(cfg, device, index_no)
        snap = _report_drift(snap, patches, drift)
        pred_sim = drifted_emulator(backend, patches, drift["ro_reported"], drift["twoq_reported"])
        hw_sim = drifted_emulator(backend, patches, drift["ro_total"], drift["twoq_total"])
    else:
        pred_sim = calibration_emulator(backend)
    log(f"{device} session {index_no}: emulator prediction")
    predicted = emulate(cfg, proc, backend, patches, pred_sim, hw["emulator_shots"], seed, orders,
                        hw["readout_cal_shots"])
    if not cfg.dry_run:
        record = {"kind": "session", "device": device, "phase": phase, "account": account,
                  "note": f"{device} session {index_no}",
                  "meta": _meta(cfg, proc, device, backend, phase, account, patches, extra)}
        return submit_and_wait(cfg, proc, backend, pubs, index, record, snap, predicted, confirm, log)
    meter = BudgetMeter(cfg.ledger, cfg["budget"], {}, dry_run=True)
    rid = meter.reserve("dry-run", phase, 0.0, backend.name, f"session {index_no}")
    counts = emulate(cfg, proc, backend, patches, hw_sim, hw["shots"], seed + 7, orders, hw["readout_cal_shots"])
    meter.settle(rid, 0.0)
    extra.update(job_ids=[], usage_s=0.0, estimate_s=0.0)
    path = _next_dir(base, "s")
    _save(path, counts, predicted, snap, _meta(cfg, proc, device, backend, phase, account, patches, extra))
    validate_session(cfg, proc, path)
    log(f"saved {path.relative_to(cfg.root)}")
    return path


def run_pilot(cfg, proc, device: str, account: str | None = None, confirm=None, log=print) -> Path:
    """Timing and crosstalk check: the same samples packed with all patches vs alone on their patch.

    Returns the saved pilot folder, or None if the wait for hardware was interrupted.
    """
    pc, hw = cfg["pilot"], cfg["hardware"]
    if not cfg.dry_run:
        ensure_no_pending(cfg, device)
    backend = get_backend(device, cfg.dry_run, account)
    patches = (make_layout(cfg, backend, device) if cfg.dry_run else load_layout(cfg, device))["patches"]
    P, n = len(patches), cfg["models"]["n_qubits"]
    pubs, index = build_packed(cfg, proc, backend, patches, [pc["dataset"]], pc["models"], pc["rows"])
    ipubs, iindex = build_isolated(cfg, proc, backend, patches, pc["dataset"], pc["models"], pc["rows"] * P)
    pubs, index = pubs + ipubs, index + iindex
    snap = snapshot(backend)
    extra = {"pilot": pc, "pubs": len(pubs), "executions": executions(pubs)}
    if not cfg.dry_run:
        record = {"kind": "pilot", "device": device, "phase": "pilot", "account": account,
                  "note": f"{device} pilot", "meta": _meta(cfg, proc, device, backend, "pilot", account, patches, extra)}
        return submit_and_wait(cfg, proc, backend, pubs, index, record, snap, {}, confirm, log)
    order = {pc["dataset"]: sample_order(proc["datasets"][pc["dataset"]])[: pc["rows"] * P]}
    sim = drifted_emulator(backend, patches, 0.0, 0.0)
    packed = emulate(cfg, proc, backend, patches, sim, hw["shots"], 1, order, hw["readout_cal_shots"], pc["models"])
    iso = emulate(cfg, proc, backend, patches, sim, hw["shots"], 2, order, None, pc["models"])
    counts = {**packed, **{f"isolated__{k}": v for k, v in iso.items()}}
    meter = BudgetMeter(cfg.ledger, cfg["budget"], {}, dry_run=True)
    meter.settle(meter.reserve("dry-run", "pilot", 0.0, backend.name, "pilot"), 0.0)
    extra.update(job_ids=[], usage_s=0.0, estimate_s=0.0)
    path = _next_dir(cfg.raw / "pilot" / device, "p")
    _save(path, counts, {}, snap, _meta(cfg, proc, device, backend, "pilot", account, patches, extra))
    log(f"saved {path.relative_to(cfg.root)}")
    return path


def noise_aware_counts(cfg, proc, device: str, account: str | None = None, log=print) -> Path:
    """R4 training data: the training split emulated with the device calibration (no QPU time)."""
    backend = get_backend(device, cfg.dry_run, account)
    patches = (make_layout(cfg, backend, device) if cfg.dry_run else load_layout(cfg, device))["patches"]
    sim = drifted_emulator(backend, patches, 0.0, 0.0) if cfg.dry_run else calibration_emulator(backend)
    orders = {name: proc["datasets"][name]["splits"]["train"] for name in cfg["data"]["datasets"]}
    counts = emulate(cfg, proc, backend, patches, sim, cfg["models"]["noise_aware_shots"],
                     cfg["experiment"]["seed"], orders)
    cfg.noise_aware.mkdir(parents=True, exist_ok=True)
    path = cfg.noise_aware / f"{device}.npz"
    np.savez_compressed(path, **counts)
    (cfg.noise_aware / f"{device}.json").write_text(json.dumps(
        {"device": device, "created_utc": now_utc(), "snapshot_captured": snapshot(backend)["captured_utc"]}))
    log(f"saved {path.relative_to(cfg.root)}")
    return path


# ---------------------------------------------------------------- validation

def validate_session(cfg, proc, path: Path) -> None:
    meta = json.loads((path / "meta.json").read_text())
    snap = json.loads((path / "snapshot.json").read_text())
    counts = np.load(path / "counts.npz")
    n, P = meta["n_qubits"], len(meta["patches"])
    problems = []
    for f in ("device", "backend", "phase", "shots", "patches", "datasets", "config_sha256", "job_ids"):
        if f not in meta:
            problems.append(f"meta missing {f}")
    if not snap.get("qubits"):
        problems.append("snapshot has no qubits")
    for name, d in meta["datasets"].items():
        if d["hash"] != proc["datasets"][name]["hash"]:
            problems.append(f"{name}: dataset hash differs from data/processed")
        for spec, b in circuit_types(proc["specs"]):
            k = key(name, spec.name, b)
            if k not in counts.files:
                problems.append(f"missing counts {k}")
                continue
            c = counts[k]
            if c.shape != (len(d["order"]), 2**n):
                problems.append(f"{k}: shape {c.shape}")
            elif not np.all(c.sum(1) == meta["shots"]):
                problems.append(f"{k}: rows do not sum to {meta['shots']} shots")
    for prep in (0, 1):
        c = counts.get(f"readout__{prep}")
        if c is None or c.shape != (P, 2**n) or not np.all(c.sum(1) == meta["readout_shots"]):
            problems.append(f"readout__{prep} invalid")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))


def session_dirs(cfg, device: str) -> list[Path]:
    base = cfg.raw / device
    return sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("s")) if base.exists() else []
