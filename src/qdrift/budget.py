"""QPU budget meter.

Every hardware job is reserved here before submission and settled with the usage IBM
reports. A reservation is refused if it would break any cap:
  - the global cap across all accounts (1,200 s),
  - the phase cap (pilot, campaign, rerun, addon),
  - 90% of the account's limit over its rolling 28-day window,
  - or if the account is not on the approved list.
Pending reservations count at their estimate until settled. The ledger is append-only.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def estimate_seconds(executions: int, n_jobs: int, per_execution_s: float, job_overhead_s: float,
                     safety: float) -> float:
    """IBM's usage formula: per-job overhead + per-execution time, with a safety factor."""
    return safety * (n_jobs * job_overhead_s + executions * per_execution_s)


class BudgetMeter:
    def __init__(self, ledger: Path, budget_cfg: dict, accounts: dict[str, dict], dry_run: bool = False):
        self.ledger = Path(ledger)
        self.cfg = budget_cfg
        self.accounts = accounts
        self.dry_run = dry_run
        self._load()

    def _load(self) -> None:
        """Re-read the ledger, so reservations written by other processes are counted."""
        self.entries: dict[str, dict] = {}
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    def _apply(self, ev: dict) -> None:
        if ev["type"] == "reserve":
            self.entries[ev["id"]] = {**ev, "job_ids": [], "status": "PENDING", "actual_s": None}
        elif ev["type"] == "attach":
            self.entries[ev["id"]]["job_ids"] += ev["job_ids"]
        elif ev["type"] == "settle":
            self.entries[ev["id"]].update(status=ev["status"], actual_s=ev["actual_s"], settled=ev["ts"])

    def _write(self, ev: dict) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a") as f:
            f.write(json.dumps(ev) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._apply(ev)

    @staticmethod
    def charged(e: dict) -> float:
        if e.get("dry_run"):
            return 0.0
        return float(e["actual_s"]) if e["actual_s"] is not None else float(e["estimate_s"])

    def _sum(self, pred) -> float:
        return sum(self.charged(e) for e in self.entries.values() if pred(e))

    def status(self, now: datetime | None = None) -> dict:
        now = now or _now()
        start = now - timedelta(days=self.cfg["window_days"])
        accounts = {}
        for name, acc in self.accounts.items():
            used = self._sum(lambda e: e["account"] == name and datetime.fromisoformat(e["ts"]) >= start)
            accounts[name] = {"used_28d_s": used, "cap_s": acc["limit_s"] * self.cfg["account_fraction"]}
        phases = {p: {"used_s": self._sum(lambda e: e["phase"] == p), "cap_s": c}
                  for p, c in self.cfg["phase_caps"].items()}
        pending = [e["id"] for e in self.entries.values() if e["status"] == "PENDING" and not e.get("dry_run")]
        return {"global": {"used_s": self._sum(lambda e: True), "cap_s": self.cfg["global_cap_s"]},
                "accounts": accounts, "phases": phases, "pending": pending,
                "dry_run_jobs": sum(1 for e in self.entries.values() if e.get("dry_run"))}

    def reserve(self, account: str, phase: str, estimate_s: float, backend: str, note: str = "",
                now: datetime | None = None) -> str:
        if self.dry_run:
            ev = {"type": "reserve", "id": uuid.uuid4().hex[:12], "ts": (now or _now()).isoformat(),
                  "account": "dry-run", "phase": phase, "backend": backend, "estimate_s": 0.0,
                  "note": note, "dry_run": True}
            self._write(ev)
            return ev["id"]
        if account not in self.accounts:
            raise BudgetExceeded(f"account '{account}' is not approved in config/accounts.toml")
        if phase not in self.cfg["phase_caps"]:
            raise BudgetExceeded(f"unknown phase '{phase}'")
        if phase not in self.accounts[account].get("phases", []):
            raise BudgetExceeded(f"account '{account}' is not approved for phase '{phase}'")
        self._load()
        st = self.status(now)
        checks = [("global cap", st["global"]), (f"phase '{phase}'", st["phases"][phase]),
                  (f"account '{account}' 28-day window",
                   {"used_s": st["accounts"][account]["used_28d_s"], "cap_s": st["accounts"][account]["cap_s"]})]
        for label, c in checks:
            if c["used_s"] + estimate_s > c["cap_s"] + 1e-9:
                raise BudgetExceeded(f"{label}: {c['used_s']:.1f} s used + {estimate_s:.1f} s estimated "
                                     f"> {c['cap_s']:.1f} s cap")
        ev = {"type": "reserve", "id": uuid.uuid4().hex[:12], "ts": (now or _now()).isoformat(),
              "account": account, "phase": phase, "backend": backend, "estimate_s": float(estimate_s),
              "note": note}
        self._write(ev)
        return ev["id"]

    def attach(self, rid: str, job_ids: list[str]) -> None:
        self._write({"type": "attach", "id": rid, "ts": _now().isoformat(), "job_ids": list(job_ids)})

    def settle(self, rid: str, actual_s: float, status: str = "DONE") -> None:
        self._write({"type": "settle", "id": rid, "ts": _now().isoformat(), "actual_s": float(actual_s),
                     "status": status})


def format_status(st: dict) -> str:
    g = st["global"]
    lines = [f"global   {g['used_s']:7.1f} / {g['cap_s']:.0f} s"]
    lines += [f"account  {k:<10} {v['used_28d_s']:7.1f} / {v['cap_s']:.0f} s (28 days)" for k, v in st["accounts"].items()]
    lines += [f"phase    {k:<10} {v['used_s']:7.1f} / {v['cap_s']:.0f} s" for k, v in st["phases"].items()]
    lines.append(f"pending  {len(st['pending'])} reservation(s) {' '.join(st['pending'])}".rstrip())
    lines.append(f"dry-run  {st['dry_run_jobs']} job(s) logged at 0 QPU seconds")
    return "\n".join(lines)
