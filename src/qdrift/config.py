"""Experiment configuration, approved accounts and project paths."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Config:
    data: dict
    root: Path
    dry_run: bool = False

    def __getitem__(self, key: str):
        return self.data[key]

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def work(self) -> Path:
        return self.root / "data" / "dry-run" if self.dry_run else self.root / "data"

    @property
    def raw(self) -> Path:
        return self.work / "raw"

    @property
    def layouts(self) -> Path:
        return self.work / "layouts"

    @property
    def noise_aware(self) -> Path:
        return self.work / "noise_aware"

    @property
    def ledger(self) -> Path:
        return self.work / "budget" / "ledger.jsonl"

    @property
    def results(self) -> Path:
        return self.root / "results" / "dry-run" if self.dry_run else self.root / "results"


def load_config(path=None, root=None, overrides: dict | None = None, dry_run: bool = False) -> Config:
    path = Path(path) if path else ROOT / "config" / "experiment.toml"
    data = tomllib.loads(path.read_text())
    if overrides:
        data = _merge(data, overrides)
    return Config(data, Path(root) if root else ROOT, dry_run)


def load_accounts(path=None) -> dict[str, dict]:
    """Approved accounts from config/accounts.toml; an empty dict if the file is missing."""
    path = Path(path) if path else ROOT / "config" / "accounts.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text()).get("accounts", {})
