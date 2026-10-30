"""Command line: `uv run qdrift <command>`. Only `pilot` and `session` without --dry-run use QPU time."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_accounts, load_config


def _confirm_factory(cfg, assume_yes: bool):
    from .budget import BudgetMeter, format_status

    def confirm(summary: str) -> bool:
        print("\n=== REAL QPU JOB ===")
        print(summary)
        print(format_status(BudgetMeter(cfg.ledger, cfg["budget"], load_accounts()).status()))
        if assume_yes:
            return True
        return input("Type RUN to submit, anything else cancels: ").strip() == "RUN"

    return confirm


def _proc(cfg):
    from .train import load_processed
    try:
        return load_processed(cfg)
    except FileNotFoundError:
        raise SystemExit("run `uv run qdrift prepare` first")


def cmd_prepare(cfg, a):
    from .train import prepare
    out = prepare(cfg)
    for r in out["baselines"]:
        print(f"{r['dataset']:14s} {r['model']:8s} acc {r['acc']:.3f} [{r['acc_lo']:.3f}, {r['acc_hi']:.3f}]")


def cmd_budget(cfg, a):
    from .budget import BudgetMeter, format_status
    meter = BudgetMeter(cfg.ledger, cfg["budget"], load_accounts(), dry_run=cfg.dry_run)
    if a.action == "audit":
        # Re-check every settled job against the usage IBM reports now and correct the ledger.
        from qiskit_ibm_runtime import QiskitRuntimeService
        accounts = load_accounts()
        for rid, e in list(meter.entries.items()):
            if e.get("dry_run") or not e["job_ids"] or e["status"] == "PENDING":
                continue
            service = QiskitRuntimeService(name=accounts[e["account"]]["saved_account"])
            jobs = [service.job(j) for j in e["job_ids"]]
            states = [str(j.status()) for j in jobs]
            total = sum(float(j.usage() or 0.0) for j in jobs)
            if not all(s in ("DONE", "ERROR", "CANCELLED") for s in states):
                meter._write({"type": "reserve", **{k: e[k] for k in ("id", "ts", "account", "phase", "backend",
                                                                      "estimate_s", "note")}})
                meter.attach(rid, e["job_ids"])
                print(f"{rid}: job still {', '.join(states)} on IBM; reservation reopened as pending")
            elif abs(total - (e["actual_s"] or 0.0)) > 1e-6:
                meter.settle(rid, total, "DONE" if all(s == "DONE" for s in states) else "FAILED")
                print(f"{rid}: corrected {e['actual_s']} s -> {total:.1f} s")
    if a.action == "reconcile":
        from qiskit_ibm_runtime import QiskitRuntimeService
        accounts = load_accounts()
        for rid in meter.status()["pending"]:
            e = meter.entries[rid]
            if not e["job_ids"]:
                print(f"{rid}: no job id attached; settling at 0 s (never submitted)")
                meter.settle(rid, 0.0, "NOT_SUBMITTED")
                continue
            service = QiskitRuntimeService(name=accounts[e["account"]]["saved_account"])
            total, states = 0.0, []
            for jid in e["job_ids"]:
                job = service.job(jid)
                states.append(str(job.status()))
                total += float(job.usage() or 0.0)
            if all(s in ("DONE", "ERROR", "CANCELLED") for s in states):
                meter.settle(rid, total, "DONE" if all(s == "DONE" for s in states) else "FAILED")
                print(f"{rid}: settled at {total:.1f} s ({', '.join(states)})")
            else:
                print(f"{rid}: jobs still running ({', '.join(states)}); try again later")
    print(format_status(meter.status()))


def cmd_backends(cfg, a):
    from qiskit_ibm_runtime import QiskitRuntimeService
    accounts = load_accounts()
    if a.account not in accounts:
        raise SystemExit(f"account '{a.account}' is not listed in config/accounts.toml")
    service = QiskitRuntimeService(name=accounts[a.account]["saved_account"])
    for b in service.backends(operational=True):
        st, proc = b.status(), getattr(b, "processor_type", None) or {}
        print(f"{b.name:16s} {b.num_qubits:4d} qubits  {proc.get('family', '?')} r{proc.get('revision', '?')}  "
              f"status {st.status_msg:12s} pending jobs {st.pending_jobs}")


def cmd_layout(cfg, a):
    from .session import get_backend, make_layout
    lay = make_layout(cfg, get_backend(a.device, cfg.dry_run, a.account), a.device, force=a.force)
    q = {r["q"]: r for r in lay["snapshot"]["qubits"]}
    print(f"{a.device}: {len(lay['patches'])} patches (frozen {lay['created_utc']})")
    for k, p in enumerate(lay["patches"]):
        ro = sum(q[i]["readout_error"] for i in p) / len(p)
        print(f"  slot {k:2d}: qubits {p}  mean readout error {ro:.4f}")


def cmd_noise_aware(cfg, a):
    from .session import noise_aware_counts
    noise_aware_counts(cfg, _proc(cfg), a.device, a.account)


def cmd_estimate(cfg, a):
    from .budget import BudgetMeter, estimate_seconds, format_status
    from .session import build_isolated, build_packed, executions, get_backend, load_layout, make_layout
    proc, hw = _proc(cfg), cfg["hardware"]
    backend = get_backend(a.device, cfg.dry_run, a.account)
    patches = (make_layout(cfg, backend, a.device) if cfg.dry_run else load_layout(cfg, a.device))["patches"]
    pc = cfg["pilot"]
    plans = {"session": build_packed(cfg, proc, backend, patches)[0]}
    pilot = build_packed(cfg, proc, backend, patches, [pc["dataset"]], pc["models"], pc["rows"])[0]
    pilot += build_isolated(cfg, proc, backend, patches, pc["dataset"], pc["models"], pc["rows"] * len(patches))[0]
    plans["pilot"] = pilot
    for name, pubs in plans.items():
        ex = executions(pubs)
        est = estimate_seconds(ex, 1, hw["per_execution_s"], hw["job_overhead_s"], cfg["budget"]["safety_factor"])
        print(f"{name:8s} {len(pubs):3d} PUBs  {ex:7d} executions  estimate {est:6.1f} s (with safety factor)")
    print(format_status(BudgetMeter(cfg.ledger, cfg["budget"], load_accounts(), cfg.dry_run).status()))


def cmd_pilot(cfg, a):
    from .analysis import analyze_pilot
    from .session import run_pilot
    proc = _proc(cfg)
    path = run_pilot(cfg, proc, a.device, a.account, _confirm_factory(cfg, a.yes))
    if path:
        print(json.dumps(analyze_pilot(cfg, proc, path), indent=1))


def cmd_fetch(cfg, a):
    from .analysis import analyze_pilot
    from .session import fetch_pending, pending_records
    proc = _proc(cfg)
    for kind, path in fetch_pending(cfg, proc):
        if kind == "pilot":
            print(json.dumps(analyze_pilot(cfg, proc, path), indent=1))
    left = pending_records(cfg)
    print(f"{len(left)} job(s) still pending" + (": " + ", ".join(p.name for p in left) if left else ""))


def cmd_session(cfg, a):
    from .session import run_session
    run_session(cfg, _proc(cfg), a.device, a.phase, a.account, _confirm_factory(cfg, a.yes))


def cmd_dryrun_campaign(cfg, a):
    from .analysis import analyze, analyze_pilot
    from .session import noise_aware_counts, run_pilot, run_session
    from .train import prepare
    if not (cfg.processed / "models.json").exists():
        prepare(cfg)
    proc = _proc(cfg)
    d = cfg["dryrun"]
    for device in d["devices"]:
        noise_aware_counts(cfg, proc, device)
        if device == d["devices"][0]:
            print(json.dumps(analyze_pilot(cfg, proc, run_pilot(cfg, proc, device)), indent=1))
        for _ in range(d["sessions_per_device"]):
            run_session(cfg, proc, device, "campaign")
    analyze(cfg, proc)


def cmd_analyze(cfg, a):
    from .analysis import analyze
    analyze(cfg, _proc(cfg))


def cmd_paper(cfg, a):
    from .paper import build
    build(cfg, _proc(cfg), compile_pdf=a.build)


def cmd_validate(cfg, a):
    from .analysis import devices_for
    from .session import session_dirs, validate_session
    proc, n = _proc(cfg), 0
    for device in devices_for(cfg):
        for p in session_dirs(cfg, device):
            validate_session(cfg, proc, p)
            n += 1
    print(f"{n} session(s) valid")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qdrift", description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="fake backends and data/dry-run; never uses QPU time")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="datasets, baselines and simulator training")
    b = sub.add_parser("budget", help="QPU budget meter")
    b.add_argument("action", choices=["status", "reconcile", "audit"])
    sub.add_parser("fetch", help="save finished jobs that were submitted earlier (no new QPU time)")
    sub.add_parser("backends", help="list operational backends (no QPU time)").add_argument("--account", default="main")
    for name, fn in (("layout", "freeze the patch layout of a device"),
                     ("noise-aware", "emulate the training split for R4"),
                     ("estimate", "QPU-time estimate for a pilot and a session")):
        p = sub.add_parser(name, help=fn)
        p.add_argument("--device", required=True)
        p.add_argument("--account", default="main")
        if name == "layout":
            p.add_argument("--force", action="store_true", help="replace an existing layout")
    for name in ("pilot", "session"):
        p = sub.add_parser(name, help=f"run a {name} (real QPU unless --dry-run)")
        p.add_argument("--device", required=True)
        p.add_argument("--account", default="main")
        p.add_argument("--yes", action="store_true", help="skip the typed confirmation")
        if name == "session":
            p.add_argument("--phase", default="campaign", choices=["campaign", "rerun", "addon"])
    sub.add_parser("dryrun-campaign", help="full rehearsal on fake backends")
    sub.add_parser("analyze", help="tables, hypotheses and figures")
    sub.add_parser("paper", help="LaTeX numbers, tables, figures and bibliography for paper/").add_argument(
        "--build", action="store_true", help="also compile paper/build/main.pdf with tectonic")
    sub.add_parser("validate", help="schema checks for every saved session")
    a = ap.parse_args(argv)
    dry = a.dry_run or a.cmd == "dryrun-campaign"
    if getattr(a, "device", "").startswith("fake_") and not dry:
        raise SystemExit("fake devices need --dry-run")
    if dry and getattr(a, "device", None) and not a.device.startswith("fake_"):
        raise SystemExit("--dry-run needs a fake_* device, e.g. fake_fez")
    cfg = load_config(dry_run=dry)
    handlers = {"prepare": cmd_prepare, "budget": cmd_budget, "backends": cmd_backends, "layout": cmd_layout,
                "noise-aware": cmd_noise_aware, "estimate": cmd_estimate, "pilot": cmd_pilot, "fetch": cmd_fetch,
                "session": cmd_session, "dryrun-campaign": cmd_dryrun_campaign, "analyze": cmd_analyze,
                "paper": cmd_paper, "validate": cmd_validate}
    handlers[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
