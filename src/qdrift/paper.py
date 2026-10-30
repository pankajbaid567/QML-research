"""Paper assets (Task 11): LaTeX numbers, tables, figures and bibliography for paper/.

Everything in paper/generated/ is rebuilt by `uv run qdrift paper` from config, the budget
ledger, data/raw and results/. Prose in paper/sections/ refers to numbers with \\val{key},
so no number is copied by hand.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyze_pilot, devices_for, load_session, patch_covariates
from .budget import BudgetMeter
from .config import load_accounts
from .session import session_dirs

DS = {"two_curves": "Two curves", "breast_cancer": "Breast cancer"}
MODEL = {"M1": "M1 re-uploading", "M2": "M2 projected", "M3": "M3 QELM",
         "logreg": "Logistic regression", "rbf_svm": "RBF-SVM", "mlp": "MLP"}
METHOD = {"SIM": "SIM", "EMU": "EMU", "R0": "R0", "R1": "R1", "R2_8": r"R2$_{8}$", "R2_16": r"R2$_{16}$",
          "R2_32": r"R2$_{32}$", "R2_16_s1": r"R2$_{16}^{\mathrm{s1}}$", "R3m": r"R3$_{\mathrm{m}}$",
          "R3r": r"R3$_{\mathrm{r}}$", "R4": "R4", "CP_sim": r"CP$_{\mathrm{sim}}$",
          "CP_s1": r"CP$_{\mathrm{s1}}$", "CP_same": r"CP$_{\mathrm{same}}$", "ACI": "ACI"}
HYP = {"H1_emulator_overestimates": ("H1", "Emulator overestimates hardware accuracy"),
       "H2_drift_beyond_shot_noise": ("H2", "Session-to-session variation exceeds shot noise"),
       "H3_refit16_beats_normalisation": ("H3", r"Refit on 16 labels (R2$_{16}$) beats normalisation (R1)"),
       "H4_readout_strongest": ("H4", "Readout error is the strongest calibration predictor"),
       "H5_split_undercovers_aci_holds": ("H5", "Split CP under-covers later sessions; ACI holds")}


def esc(s) -> str:
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def tt(device: str) -> str:
    return r"\texttt{" + esc(device) + "}"


def short(device: str) -> str:
    """Device name without the ibm_ prefix, for narrow table columns."""
    return device.removeprefix("ibm_")


def key(*parts) -> str:
    return "-".join(re.sub(r"[^A-Za-z0-9]", "", str(p)) for p in parts)


def f3(x) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def ci(m, lo, hi, d=3) -> str:
    if m is None or not np.isfinite(m):
        return "--"
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if np.isfinite(lo) and np.isfinite(hi) else f"{m:.{d}f}"


def table(env: str, caption: str, label: str, colspec: str, header: list[str], rows: list[list[str]],
          note: str = "") -> str:
    lines = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\\[2pt]\parbox{{\linewidth}}{{\scriptsize {note}}}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- inputs

def sessions(cfg) -> list[dict]:
    """Hardware sessions and pilots with timing, usage and calibration age at run time."""
    out = []
    for device in devices_for(cfg):
        runs = [("session", p) for p in session_dirs(cfg, device)]
        pdir = cfg.raw / "pilot" / device
        runs += [("pilot", p) for p in sorted(pdir.iterdir())] if pdir.exists() else []
        for kind, p in runs:
            S = load_session(p)
            m = S["meta"]
            ts = m.get("job_timestamps") or {}
            age = patch_covariates(S["snap"], m["patches"])[0]["age_h"]
            out.append({"device": device, "kind": kind, "session": m.get("session"),
                        "start": ts.get("running") or m.get("created_utc"), "queue_h": (m.get("queue_wait_s") or np.nan) / 3600,
                        "usage_s": m.get("usage_s"), "age_h": age, "job": (m.get("job_ids") or [""])[0], "path": p})
    return sorted(out, key=lambda r: (r["device"], r["kind"] != "pilot", r["start"] or ""))


def _csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------- numbers

def numbers(cfg, proc, runs) -> dict[str, str]:
    m, hw, d, ex, b = cfg["models"], cfg["hardware"], cfg["data"], cfg["experiment"], cfg["budget"]
    specs = {s.name: s for s in proc["specs"]}
    st = BudgetMeter(cfg.ledger, b, load_accounts()).status()
    camp = [r for r in runs if r["kind"] == "session"]
    v = {"n-qubits": m["n_qubits"], "shots": hw["shots"], "readout-shots": hw["readout_cal_shots"],
         "emulator-shots": hw["emulator_shots"], "patches": hw["max_patches"], "patch-buffer": hw["patch_buffer"],
         "n-train": d["n_train"], "n-val": d["n_val"], "n-cal": d["n_cal_pool"], "n-test": d["n_test"],
         "n-hw-samples": d["n_cal_pool"] + d["n_test"], "alpha": ex["alpha"],
         "coverage-target": f"{100 * (1 - ex['alpha']):.0f}\\%", "aci-gamma": ex["aci_gamma"],
         "bootstrap": f"{ex['bootstrap']:,}", "seed": ex["seed"],
         "m1-layers": m["m1_layers"], "m1-params": specs["M1"].n_params, "m2-layers": m["m2_layers"],
         "m3-layers": m["m3_layers"], "m1-features": specs["M1"].n_features, "m2-features": specs["M2"].n_features,
         "m3-features": specs["M3"].n_features, "m1-steps": m["m1_steps"], "m1-restarts": m["m1_restarts"],
         "refit-lambda": m["refit_lambda_per_sample"], "noise-aware-shots": f"{m['noise_aware_shots']:,}",
         "sessions-per-device": hw["sessions_per_device"], "devices": " and ".join(tt(x) for x in hw["devices"]),
         "fallback": ", ".join(tt(x) for x in hw.get("fallback_devices", [])),
         "per-execution-us": f"{hw['per_execution_s'] * 1e6:.0f}", "budget-cap": b["global_cap_s"],
         "campaign-cap": b["phase_caps"]["campaign"], "pilot-cap": b["phase_caps"]["pilot"],
         "qpu-used": f"{st['global']['used_s']:.0f}", "qpu-pilot": f"{st['phases']['pilot']['used_s']:.0f}",
         "qpu-campaign": f"{st['phases']['campaign']['used_s']:.0f}", "sessions-done": len(camp),
         "sessions-planned": hw["sessions_per_device"] * len(hw["devices"])}
    waits = [r["queue_h"] for r in camp if np.isfinite(r["queue_h"])]
    if waits:
        v["queue-median-h"], v["queue-max-h"] = f"{np.median(waits):.1f}", f"{np.max(waits):.1f}"
    if camp:
        v["session-usage-mean"] = f"{np.mean([r['usage_s'] for r in camp]):.1f}"
        v["campaign-first"], v["campaign-last"] = min(r["start"] for r in camp)[:10], max(r["start"] for r in camp)[:10]
    for name in ("sim_metrics", "baselines"):
        t = _csv(cfg.results / f"{name}.csv")
        if t is not None:
            for _, r in t.iterrows():
                v[key("sim" if name == "sim_metrics" else "base", "acc", r["dataset"], r["model"])] = f3(r["acc"])
    met = _csv(cfg.results / "metrics.csv")
    if met is not None:
        g = met.groupby(["method", "dataset", "model"])["acc"].mean()
        for (meth, dn, mo), a in g.items():
            v[key("acc", meth, dn, mo)] = f3(a)
        w = met.pivot_table(index=["device", "session", "dataset", "model"], columns="method", values="acc")
        gap = (w["SIM"] - w["R0"]) * 100
        v["gap-mean-pp"], v["gap-min-pp"], v["gap-max-pp"] = f"{gap.mean():.1f}", f"{gap.min():.1f}", f"{gap.max():.1f}"
    hyp = cfg.results / "hypotheses.json"
    if hyp.exists():
        for k, h in json.loads(hyp.read_text()).items():
            tag = HYP.get(k, (k,))[0].lower()
            v[f"{tag}-decision"] = {True: "supported", False: "not supported", None: "undecided"}[h.get("supported")]
            stat = next((h[s] for s in ("mean_emu_minus_hw", "mean_ratio", "mean_diff") if s in h), None)
            if stat is not None:
                v[f"{tag}-mean"], v[f"{tag}-lo"], v[f"{tag}-hi"] = f3(stat), f3(h["ci"][0]), f3(h["ci"][1])
    return {k: str(x) for k, x in v.items()}


# ---------------------------------------------------------------- tables

def tab_sim(cfg) -> str | None:
    sim, base = _csv(cfg.results / "sim_metrics.csv"), _csv(cfg.results / "baselines.csv")
    if sim is None or base is None:
        return None
    t = pd.concat([sim, base])
    rows = []
    for mo in ["M1", "M2", "M3", "logreg", "rbf_svm", "mlp"]:
        cells = []
        for dn in DS:
            r = t[(t["model"] == mo) & (t["dataset"] == dn)]
            cells.append(ci(r["acc"].iloc[0], r["acc_lo"].iloc[0], r["acc_hi"].iloc[0]) if len(r) else "--")
        rows.append([MODEL[mo]] + cells)
    rows.insert(3, [r"\midrule\multicolumn{3}{l}{\emph{Tuned classical baselines, same inputs}}"])
    return table("table", r"Noiseless simulator test accuracy of the quantum models and tuned classical "
                 rf"baselines ({cfg['data']['n_test']} test samples; 95\% bootstrap interval).",
                 "tab:sim", "lcc", ["Model"] + list(DS.values()), rows)


def tab_protocol(runs) -> str | None:
    if not runs:
        return None
    year = (runs[0]["start"] or "")[:4]
    rows = [[short(r["device"]), "pilot" if r["kind"] == "pilot" else str(r["session"]),
             esc((r["start"] or "")[5:16].replace("T", " ")), f"{r['queue_h']:.1f}" if np.isfinite(r["queue_h"]) else "--",
             f"{r['usage_s']:.0f}" if r["usage_s"] is not None else "--", f"{r['age_h']:.1f}" if np.isfinite(r["age_h"]) else "--"]
            for r in runs]
    return table("table", f"Hardware runs on \\texttt{{ibm\\_}}* devices: job start ({year}, UTC), queue wait, billed QPU "
                 "time and age of the reported calibration at the job start.",
                 "tab:protocol", "llrrrr", ["Device", "Run", "Start", "Queue (h)", "QPU (s)", "Cal.\\ age (h)"], rows)


def tab_accuracy(cfg, n_sessions: int) -> str | None:
    met = _csv(cfg.results / "metrics.csv")
    if met is None:
        return None
    cols = [c for c in ["SIM", "EMU", "R0", "R1", "R2_16", "R2_16_s1", "R3m", "R3r", "R4"] if c in set(met["method"])]
    g = met.groupby(["dataset", "model", "method"])["acc"].mean().unstack()
    repairs = [c for c in cols if c not in ("SIM", "EMU", "R0")]
    rows = []
    for (dn, mo), r in g.iterrows():
        best = r[repairs].max() if repairs else np.nan
        cells = [(r"\textbf{" + f3(r[c]) + "}") if c in repairs and np.isclose(r[c], best) else f3(r[c]) for c in cols]
        rows.append([DS[dn], mo] + cells)
    pre = f" Preliminary: {n_sessions} of {cfg['hardware']['sessions_per_device'] * len(cfg['hardware']['devices'])} sessions." \
        if n_sessions < cfg["hardware"]["sessions_per_device"] * len(cfg["hardware"]["devices"]) else ""
    return table("table*", "Test accuracy by method, averaged over sessions and devices. SIM: noiseless simulator; "
                 "EMU: calibration emulator; R0: hardware without repair; bold: best repair per row." + pre,
                 "tab:accuracy", "ll" + "c" * len(cols), ["Dataset", "Model"] + [METHOD[c] for c in cols], rows)


def tab_repairs(cfg) -> str | None:
    rec = _csv(cfg.results / "recovered.csv")
    if rec is None:
        return None
    rows = [[METHOD.get(r["method"], esc(r["method"])), f"{100 * r['acc_gain']:+.1f}",
             f"{100 * r['recovered_frac']:.0f}" if np.isfinite(r["recovered_frac"]) else "--",
             f"{r['extra_qpu_s']:.2f}", str(int(r["n"]))] for _, r in rec.iterrows()]
    return table("table", "Repairs ranked by mean accuracy gain over R0, share of the simulator-to-hardware gap "
                 "they recover (cases with a gap above 2 points), and extra QPU seconds per session.",
                 "tab:repairs", "lrrrr", ["Repair", "Gain (pp)", "Gap (\\%)", "Extra QPU (s)", "$n$"], rows)


def tab_coverage(cfg) -> str | None:
    cp = _csv(cfg.results / "conformal.csv")
    if cp is None:
        return None
    rows = []
    for meth in ["CP_sim", "CP_s1", "CP_same", "ACI"]:
        g = cp[cp["method"] == meth]
        later = g[g["session"] > 1]["coverage"]
        rows.append([METHOD[meth], f3(g["coverage"].mean()), f3(later.mean()) if len(later) else "--",
                     f3(g["coverage"].min()), f"{g['set_size'].mean():.2f}", f"{g['extra_qpu_s'].mean():.2f}"])
    a = cfg["experiment"]["alpha"]
    return table("table", rf"Conformal prediction on hardware (target coverage {1 - a:.2f}): mean coverage over all "
                 "sessions and over sessions 2 onward, worst case, mean set size and extra QPU seconds.",
                 "tab:coverage", "lrrrrr", ["Method", "Cov.", "Cov.\\ $s\\geq2$", "Min", "Size", "QPU (s)"], rows)


def tab_hypotheses(cfg, n_sessions: int) -> str | None:
    path = cfg.results / "hypotheses.json"
    hyp = json.loads(path.read_text()) if path.exists() else {}
    rows = []
    for k, (tag, text) in HYP.items():
        h = hyp.get(k)
        if h is None:
            rows.append([tag, text, "--", "pending"])
            continue
        if "mean_emu_minus_hw" in h:
            stat = "EMU$-$R0 " + ci(h["mean_emu_minus_hw"], *h["ci"])
        elif "mean_ratio" in h:
            stat = "ratio " + ci(h["mean_ratio"], *h["ci"], d=2)
        elif "mean_diff" in h:
            stat = r"R2$_{16}-$R1 " + ci(h["mean_diff"], *h["ci"])
        elif "largest_term" in h:
            stat = "largest: " + esc(h["largest_term"])
        else:
            stat = r"CP$_{\mathrm{s1}}$ " + ci(*h["cp_s1"]) + ", ACI " + ci(*h["aci"])
        rows.append([tag, text, stat, {True: "supported", False: "not supported", None: "--"}[h.get("supported")]])
    planned = cfg["hardware"]["sessions_per_device"] * len(cfg["hardware"]["devices"])
    note = f" Preliminary: {n_sessions} of {planned} sessions." if n_sessions < planned else ""
    return table("table*", "Pre-registered hypotheses, test statistics (mean and 95\\% bootstrap interval) and "
                 "decisions." + note, "tab:hypotheses", "llll", ["", "Hypothesis", "Statistic", "Decision"], rows)


def tab_pilot(cfg, proc) -> str | None:
    rows = []
    for device in devices_for(cfg):
        pdir = cfg.raw / "pilot" / device
        for p in (sorted(pdir.iterdir()) if pdir.exists() else []):
            res = analyze_pilot(cfg, proc, p)
            for k, r in res.items():
                if isinstance(r, dict):
                    model = k.split("__")[1]
                    lo, hi = r["tvd_excess_ci"]
                    dlo, dhi = r["acc_diff_ci"]
                    rows.append([short(device), model, f"[{lo:+.3f}, {hi:+.3f}]", f"{r['acc_packed']:.2f}",
                                 f"{r['acc_isolated']:.2f}", f"[{dlo:+.2f}, {dhi:+.2f}]"])
    if not rows:
        return None
    return table("table", "Pilot: the same samples run packed with all patches active or alone on their patch. "
                 "Excess total-variation distance over a shot-noise null, and accuracy difference (packed minus "
                 "isolated), with 95\\% bootstrap intervals.", "tab:pilot", "llcccc",
                 ["Device", "Model", "Excess TVD", "Packed", "Isolated", r"$\Delta$acc"], rows)


# ---------------------------------------------------------------- bibliography and build

def paper_bib(text: str) -> str:
    """refs.bib adapted to IEEEtran.bst: keep title case, show arXiv ids and DOIs."""
    def fix(m):
        kind, k, body = m.group(1), m.group(2), m.group(3)
        f = dict(re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body))
        body = re.sub(r"title = \{(?!\{)(.*?)\},", lambda t: "title = {{" + t.group(1) + "}},", body, count=1, flags=re.S)
        add = []
        if "eprint" in f and kind.lower() == "misc" and "howpublished" not in f:
            add.append(f"howpublished = {{arXiv:{f['eprint']}}}")
        if "doi" in f and "note" not in f:
            add.append(f"note = {{doi: \\href{{https://doi.org/{f['doi']}}}{{{f['doi']}}}}}")
        elif "eprint" in f and kind.lower() != "misc" and "note" not in f:
            add.append(f"note = {{arXiv:{f['eprint']}}}")
        return f"@{kind}{{{k},{body.rstrip()}" + ("".join(",\n  " + a for a in add)) + "\n}"
    return re.sub(r"@(\w+)\{([^,]+),(.*?)\n\}", fix, text, flags=re.S)


def build(cfg, proc, log=print, compile_pdf: bool = False) -> Path:
    paper = cfg.root / "paper"
    gen = paper / "generated"
    (gen / "figures").mkdir(parents=True, exist_ok=True)
    runs = sessions(cfg)
    n_sessions = sum(r["kind"] == "session" for r in runs)
    newest = max((r["path"].stat().st_mtime for r in runs), default=0)
    met = cfg.results / "metrics.csv"
    if met.exists() and met.stat().st_mtime < newest:
        log("note: results/ is older than the newest session; run `uv run qdrift analyze` first")
    vals = numbers(cfg, proc, runs)
    lines = ["% Generated by `uv run qdrift paper`; do not edit.",
             r"\providecommand{\qdset}[2]{\expandafter\def\csname qd@#1\endcsname{#2}}"]
    lines += [rf"\qdset{{{k}}}{{{v}}}" for k, v in sorted(vals.items())]
    (gen / "numbers.tex").write_text("\n".join(lines) + "\n")
    tables = {"tab-sim": tab_sim(cfg), "tab-protocol": tab_protocol(runs), "tab-accuracy": tab_accuracy(cfg, n_sessions),
              "tab-repairs": tab_repairs(cfg), "tab-coverage": tab_coverage(cfg),
              "tab-hypotheses": tab_hypotheses(cfg, n_sessions), "tab-pilot": tab_pilot(cfg, proc)}
    for name, tex in tables.items():
        (gen / f"{name}.tex").write_text(tex or f"% {name}: no data yet\n")
    figs = sorted((cfg.results / "figures").glob("*.pdf")) if (cfg.results / "figures").exists() else []
    for f in figs:
        shutil.copy2(f, gen / "figures" / f.name)
    (gen / "refs.bib").write_text(paper_bib((cfg.root / "literature" / "refs.bib").read_text()))
    log(f"wrote {len(vals)} numbers, {sum(t is not None for t in tables.values())} tables and {len(figs)} figures "
        f"to {gen.relative_to(cfg.root)} ({n_sessions} hardware session(s))")
    if compile_pdf:
        if not shutil.which("tectonic"):
            raise SystemExit("tectonic not found; install it with `brew install tectonic` or use Overleaf")
        (paper / "build").mkdir(exist_ok=True)
        res = subprocess.run(["tectonic", "--keep-logs", "--outdir", "build", "main.tex"], cwd=paper,
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(res.stderr[-3000:])
        warn = list(dict.fromkeys(ln for ln in res.stderr.splitlines()
                                  if ("warning: " in ln.lower() and "underfull" not in ln.lower())
                                  or "undefined" in ln.lower()))
        log("\n".join(warn[:20]) if warn else "no LaTeX warnings (underfull boxes ignored)")
        log(f"built {(paper / 'build' / 'main.pdf').relative_to(cfg.root)}")
    return gen
