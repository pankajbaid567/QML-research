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
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyze_pilot, devices_for, load_session, patch_covariates
from .budget import BudgetMeter
from .config import load_accounts
from .metrics import bootstrap_ci
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
         "tc-degree": d["two_curves"]["degree"], "tc-offset": d["two_curves"]["offset"],
         "tc-noise": d["two_curves"]["noise"], "pilot-samples": cfg["pilot"]["rows"] * hw["max_patches"],
         "session-pubs": sum(len(s.bases) for s in proc["specs"]) * len(d["datasets"]) + 2,
         "max-readout-error": hw["max_readout_error"], "max-twoq-error": hw["max_twoq_error"],
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
    all_waits = [r["queue_h"] for r in runs if np.isfinite(r["queue_h"])]
    if all_waits:
        v["queue-max-all-h"] = f"{np.max(all_waits):.1f}"
    ages = [r["age_h"] for r in runs if np.isfinite(r["age_h"])]
    if ages:
        v["cal-age-min-h"], v["cal-age-max-h"] = f"{min(ages):.1f}", f"{max(ages):.1f}"
    v["max-cz"] = max(sum(op.gate == "cz" for op in s.ops) for s in proc["specs"])
    try:   # registration tag: date and commit
        tag = subprocess.run(["git", "for-each-ref", "refs/tags/prereg-v1",
                              "--format=%(taggerdate:short) %(*objectname:short)"],
                             cwd=cfg.root, capture_output=True, text=True, timeout=10).stdout.split()
        if len(tag) == 2:
            v["prereg-date"], v["prereg-commit"] = tag
    except Exception:
        pass
    v["coverage-target-frac"] = f"{1 - ex['alpha']:.2f}"
    if camp:
        v["session-usage-mean"] = f"{np.mean([r['usage_s'] for r in camp]):.1f}"
        v["campaign-first"], v["campaign-last"] = min(r["start"] for r in camp)[:10], max(r["start"] for r in camp)[:10]
        per_dev = Counter(r["device"] for r in camp)
        lo_n, hi_n, planned = min(per_dev.values()), max(per_dev.values()), hw["sessions_per_device"]
        v["sessions-done-per-device"] = str(lo_n) if lo_n == hi_n else f"{lo_n}--{hi_n}"
        starts = [datetime.fromisoformat(r["start"]) for r in camp if r["start"]]
        ist = [t + timedelta(hours=5, minutes=30) for t in starts]
        span = (max(starts) - min(starts)).total_seconds() / 86400   # first to last session start, in days
        v["campaign-days"], v["campaign-span-days"] = f"{span:.0f}", f"{span:.1f}"
        v["sessions-day"] = sum(6 <= t.hour < 18 for t in ist)
        v["sessions-night"] = len(ist) - v["sessions-day"]
        v["window-sentence"] = (
            f"All sessions started within the registered {WINDOW_DAYS}-day window." if span <= WINDOW_DAYS else
            f"The campaign spanned {span:.1f} days from the first to the last session, {span - WINDOW_DAYS:.1f} "
            f"days longer than the registered maximum of {WINDOW_DAYS}.")
        short_devs = [f"{tt(d)} completed {n} of the {planned} planned sessions" for d, n in sorted(per_dev.items())
                      if n < planned]
        v["sessions-sentence"] = "; ".join(short_devs) + "." if short_devs else ""
        sens_path = cfg.results / "sensitivity.json"
        sens = json.loads(sens_path.read_text()) if sens_path.exists() else {}
        rep = sens.get("excluded") or []
        if rep:
            pairs = "; ".join(f"on {tt(r['device'])}, session {r['session']} started {r['gap_h']:.1f}~h after "
                              f"session {r['previous']}" for r in rep)
            h2s = sens.get("h2", {})
            hp = cfg.results / "hypotheses.json"
            h2r = (json.loads(hp.read_text()) if hp.exists() else {}).get("H2_drift_beyond_shot_noise", {})
            v["spacing-sentence"] = (
                f"Some sessions followed the previous one within hours ({pairs}), so such a pair measures "
                f"variation within one night rather than between days.")
            if "mean_ratio" in h2s and "mean_ratio" in h2r:
                v["spacing-sentence"] += (
                    f" Excluding these sessions, an exploratory check that was not registered, gives a feature-variance "
                    f"ratio of {h2s['mean_ratio']:.2f} (interval {h2s['ci'][0]:.2f} to {h2s['ci'][1]:.2f}), against "
                    f"{h2r['mean_ratio']:.2f} in the registered analysis (Section~\\ref{{sec:rq2}}).")
            v["sessions-closepair"] = len(rep)
        else:
            v["spacing-sentence"] = ""
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
                if tag in ("h1", "h3"):   # accuracy differences, also in percentage points
                    v[f"{tag}-mean-pp"], v[f"{tag}-lo-pp"], v[f"{tag}-hi-pp"] = (
                        pp(x, signed=tag == "h3") for x in (stat, *h["ci"]))
                if tag == "h2":   # a variance ratio: two decimals, as in the tables
                    v["h2-fold"] = f"{stat:.1f}"
                    v["h2-mean"], v["h2-lo"], v["h2-hi"] = (f"{x:.2f}" for x in (stat, *h["ci"]))
            if "slope_hw_on_emu" in h:
                v["h1-slope"] = f"{h['slope_hw_on_emu']:.2f}"
            if "largest_term" in h:
                v["h4-largest"] = REG_TERM.get(h["largest_term"], esc(h["largest_term"]))
            for name in ("cp_s1", "aci"):
                if name in h:
                    k = key("h5", name)
                    v[k + "-mean"], v[k + "-lo"], v[k + "-hi"] = (f3(x) for x in h[name])
    v.update(result_numbers(cfg))
    return {k: str(x) for k, x in v.items()}


REG_TERM = {"readout": "readout error", "twoq": "CZ error", "t1_us": r"$T_1$", "t2_us": r"$T_2$",
            "age_h": "calibration age"}
WINDOW_DAYS = 14   # registered campaign window (preregistration.md: 5 sessions per device over 10-14 days)
LABEL = {"device": tt, "dataset": lambda n: DS[n].lower(), "model": esc}


def pp(x, signed: bool = True) -> str:
    """A fraction as percentage points."""
    if x is None or not np.isfinite(x):
        return "--"
    return f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"


def result_numbers(cfg) -> dict:
    """Macros for the Results text, recomputed from results/ on every build."""
    v = {}
    met = _csv(cfg.results / "metrics.csv")
    if met is not None and len(met):
        w = met.pivot_table(index=["device", "session", "dataset", "model"], columns="method", values="acc")
        v["n-cases"] = len(w)
        gap = w["SIM"] - w["R0"]
        v["gap-pos-count"] = int((gap > 0).sum())
        for level in ("device", "dataset", "model"):
            g = gap.groupby(level=level).mean()
            for name, x in g.items():
                v[key("gap", short(name), "pp")] = pp(x, signed=False)
            v[f"gap-max-{level}"], v[f"gap-min-{level}"] = LABEL[level](g.idxmax()), LABEL[level](g.idxmin())
            v[f"gap-max-{level}-pp"], v[f"gap-min-{level}-pp"] = pp(g.max(), False), pp(g.min(), False)
        r0rows = met[met["method"] == "R0"]
        v["r0-ci-half-pp"] = f"{100 * ((r0rows['acc_hi'] - r0rows['acc_lo']) / 2).mean():.0f}"
        if "EMU" in w:
            d = (w["EMU"] - w["R0"]).dropna()
            v["emu-over-count"], v["emu-cases"] = int((d > 0).sum()), len(d)
            pair = w[["EMU", "R0"]].dropna()
            if len(pair) > 2:
                v["emu-hw-r"] = f"{np.corrcoef(pair['EMU'], pair['R0'])[0, 1]:.2f}"
        r0 = met[met["method"] == "R0"].groupby(["device", "dataset", "model"])["acc"]
        spread = r0.max() - r0.min()
        v["r0-spread-mean-pp"], v["r0-spread-max-pp"] = pp(spread.mean(), False), pp(spread.max(), False)
        for meth, g in met.groupby("method"):
            v[key("ece", meth)], v[key("brier", meth)] = f3(g["ece"].mean()), f3(g["brier"].mean())
            v[key("auc", meth)] = f3(g["auc"].mean())
    rec = _csv(cfg.results / "recovered.csv")
    if rec is not None and len(rec):
        for _, r in rec.iterrows():
            k = key("gain", r["method"])
            v[k + "-pp"] = pp(r["acc_gain"])
            if "gain_lo" in rec:
                v[k + "-lo"], v[k + "-hi"] = pp(r["gain_lo"]), pp(r["gain_hi"])
            v[key("cost", r["method"])] = f"{r['extra_qpu_s']:.2f}"
            v[key("gapshare", r["method"])] = f"{100 * r['recovered_frac']:.0f}" if np.isfinite(r["recovered_frac"]) else "--"
        best = rec.sort_values("acc_gain", ascending=False).iloc[0]
        v["rep-best"], v["rep-best-gain-pp"] = METHOD.get(best["method"], esc(best["method"])), pp(best["acc_gain"])
        v["gapshare-max"] = f"{100 * rec['recovered_frac'].max():.0f}"
        if "n_gap" in rec:
            v["n-gap-cases"] = int(rec["n_gap"].max())
    fits = _csv(cfg.results / "noise_fits.csv")
    if fits is not None and len(fits):
        for col, name in (("gamma", "gamma"), ("r2", "r2")):
            q1, q2, q3 = fits[col].quantile([0.25, 0.5, 0.75])
            v[f"nm-{name}-q1"], v[f"nm-{name}-median"], v[f"nm-{name}-q3"] = f3(q1), f3(q2), f3(q3)
        v["nm-absbeta-median"], v["nm-fits"] = f3(fits["beta"].abs().median()), len(fits)
        for mo, g in fits.groupby("model"):
            v[key("nm", "gamma", mo)], v[key("nm", "r2", mo)] = f3(g["gamma"].median()), f3(g["r2"].median())
        sd = fits.groupby(["device", "dataset", "model", "feature"])["gamma"].std(ddof=1).dropna()
        if len(sd):
            v["nm-gamma-sd-median"] = f3(sd.median())
    od = _csv(cfg.results / "overdispersion.csv")
    if od is not None and len(od):
        for level in ("device", "model", "dataset"):
            for name, g in od.groupby(level):
                v[key("od", short(name))] = f"{g['ratio'].mean():.2f}"
        v["od-frac-above-1"] = f"{100 * (od['ratio'] > 1).mean():.0f}"
        dev = od.groupby("device")["ratio"].mean()
        v["od-max-device"], v["od-min-device"] = tt(dev.idxmax()), tt(dev.idxmin())
        v["od-max"], v["od-min"] = f"{dev.max():.2f}", f"{dev.min():.2f}"
        ex = cfg["experiment"]
        for name, g in od.groupby("device"):
            x = g["ratio"].to_numpy()
            lo, hi = bootstrap_ci(lambda i: x[i].mean(), len(x), ex["bootstrap"], ex["seed"])
            v[key("od", short(name), "lo")], v[key("od", short(name), "hi")] = f"{lo:.2f}", f"{hi:.2f}"
    smp = _csv(cfg.results / "samples.csv")
    if smp is not None and len(smp):
        fe = smp.groupby(["device", "session"])["feat_err"].mean()
        for name, g in fe.groupby(level="device"):
            g = g.droplevel("device").sort_index()
            v[key("fe", short(name), "first")], v[key("fe", short(name), "last")] = f3(g.iloc[0]), f3(g.iloc[-1])
    sens_path = cfg.results / "sensitivity.json"
    sens = json.loads(sens_path.read_text()) if sens_path.exists() else {}
    flagged = sens.get("flagged_couplers") or []
    v["flag-sentence"], v["flag-short"] = "", ""
    if flagged:
        by = Counter((f["device"], f["session"]) for f in flagged)
        where = " and ".join(f"{n} coupler{'s' if n > 1 else ''} of the frozen chains in session {s} on {tt(d)}"
                             for (d, s), n in sorted(by.items()))
        h4 = sens.get("h4_without_flagged") or {}
        terms = {t["term"]: t for t in h4.get("terms", [])}
        v["flag-short"] = f"the calibration at job start marked {where} as non-operational"
        sent = (f"This result rests on a few inputs: the calibration at job start marked {where} as "
                f"non-operational (CZ error 1.0), yet the job ran on them.")
        if terms:
            top = terms[h4["largest_term"]]
            sent += (f" Refitting without the {h4['n_dropped']} observations from these chains, an exploratory "
                     f"check, gives the largest coefficient to the {REG_TERM.get(top['term'], esc(top['term']))} "
                     f"({top['coef']:.4f}, interval {top['lo']:.4f} to {top['hi']:.4f})")
            if "twoq" in terms and h4["largest_term"] != "twoq":
                tq = terms["twoq"]
                sent += f" and the CZ error falls to {tq['coef']:.4f} ({tq['lo']:.4f} to {tq['hi']:.4f})"
            sent += "."
        v["flag-sentence"] = sent
    reg = _csv(cfg.results / "regression.csv")
    if reg is not None and len(reg):
        for _, r in reg.iterrows():
            k = key("reg", r["term"])
            v[k + "-coef"], v[k + "-lo"], v[k + "-hi"] = (f"{x:.4f}" for x in (r["coef"], r["lo"], r["hi"]))
            v[k + "-p"] = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        top = reg.iloc[reg["coef"].abs().argmax()]
        v["reg-largest"] = REG_TERM.get(top["term"], "--")
        v["reg-largest-coef"], v["reg-largest-lo"], v["reg-largest-hi"] = (
            f"{x:.4f}" for x in (top["coef"], top["lo"], top["hi"]))
    cp = _csv(cfg.results / "conformal.csv")
    if cp is not None and len(cp):
        for meth, g in cp.groupby("method"):
            k = key("cov", meth)
            later = g[g["session"] > 1]["coverage"]
            v[k], v[k + "-min"] = f3(g["coverage"].mean()), f3(g["coverage"].min())
            v[k + "-later"] = f3(later.mean()) if len(later) else "--"
            v[key("size", meth)] = f"{g['set_size'].mean():.2f}"
    return v


def claims(cfg) -> list[tuple[str, bool, str]]:
    """Directional statements of the text, re-checked against results/ on every build.

    Each id is tagged `% claim:<id>` next to its sentence in paper/. Numbers come from \\val{},
    but words such as "optimistic" or "mostly on one device" do not, so new data could silently
    contradict them; a failing claim means the tagged sentence must be rewritten."""
    met, rec, cp, fits, od, reg, smp = (_csv(cfg.results / f"{n}.csv") for n in
                                        ("metrics", "recovered", "conformal", "noise_fits", "overdispersion",
                                         "regression", "samples"))
    sp = cfg.results / "sensitivity.json"
    sens = json.loads(sp.read_text()) if sp.exists() else {}
    path = cfg.results / "hypotheses.json"
    hyp = json.loads(path.read_text()) if path.exists() else {}
    if met is None or rec is None or cp is None:
        return []
    target = 1 - cfg["experiment"]["alpha"]
    w = met.pivot_table(index=["device", "session", "dataset", "model"], columns="method", values="acc")
    gap = w["SIM"] - w["R0"]
    r = rec.set_index("method")
    cov, size = cp.groupby("method")["coverage"].mean(), cp.groupby("method")["set_size"].mean()
    later = cp[cp["session"] > 1].groupby("method")["coverage"].mean()
    ece = met.groupby("method")["ece"].mean()
    h1 = hyp.get("H1_emulator_overestimates", {})
    lo1 = lambda m: float(r.loc[m, "gain_lo"])
    spread = lambda level: np.ptp(gap.groupby(level=level).mean().to_numpy())
    checks = {
        "gap-task": lambda: (gap.groupby(level="dataset").mean().get("two_curves") -
                             gap.groupby(level="dataset").mean().get("breast_cancer") >
                             max(spread("model"), spread("device")), "task difference exceeds model and device spread"),
        "emu-optimistic": lambda: (h1["ci"][0] > 0, f"H1 interval {h1['ci']}"),
        "gap-positive": lambda: (gap.mean() > 0, f"mean gap {gap.mean():.3f}"),
        "h2-supported": lambda: (hyp["H2_drift_beyond_shot_noise"]["supported"] is True, "H2 supported"),
        "emu-ranks-well": lambda: (np.corrcoef(w["EMU"], w["R0"])[0, 1] >= 0.6, "r >= 0.6"),
        "emu-slope-above-1": lambda: (h1["slope_hw_on_emu"] > 1, f"slope {h1['slope_hw_on_emu']:.2f}"),
        "drift-both-devices": lambda: ((d := od.groupby("device")["ratio"].mean()).min() > 2
                                       and min((sens.get("h2") or {}).get("by_device", {"x": 9}).values()) > 2,
                                       f"registered {d.round(2).to_dict()}, sensitivity {(sens.get('h2') or {}).get('by_device')}"),
        "fe-decreases": lambda: (all((g := s.droplevel("device").sort_index()).iloc[-1] < g.iloc[0]
                                     for _, s in smp.groupby(["device", "session"])["feat_err"].mean()
                                     .groupby(level="device")), "last session below first on every device"),
        "h4-hinges-on-flagged": lambda: (bool(sens.get("flagged_couplers")) and
                                         (sens.get("h4_without_flagged") or {}).get("largest_term") == "readout",
                                         f"without flagged chains: {(sens.get('h4_without_flagged') or {}).get('largest_term')}"),
        "readout-positive": lambda: (reg.set_index("term").loc["readout", "lo"] > 0, "readout interval above zero"),
        "h4-not-readout": lambda: (hyp["H4_readout_strongest"]["largest_term"] != "readout",
                                   hyp["H4_readout_strongest"]["largest_term"]),
        "t2-negative": lambda: (reg.set_index("term").loc["t2_us", "hi"] < 0, "T2 interval below zero"),
        "gamma-below-1": lambda: (fits["gamma"].median() < 1, f"median {fits['gamma'].median():.3f}"),
        "gamma-stable": lambda: ((s := fits.groupby(["device", "dataset", "model", "feature"])["gamma"].std(ddof=1)
                                  .median()) < (fits["gamma"].quantile(0.75) - fits["gamma"].quantile(0.25)) / 2,
                                 f"session SD {s:.3f} vs half IQR"),
        "gamma-models-differ": lambda: (np.ptp(fits.groupby("model")["gamma"].median().to_numpy()) >
                                        2 * fits.groupby(["device", "dataset", "model", "feature"])["gamma"]
                                        .std(ddof=1).median(), "model medians differ by > 2 session SDs"),
        "small-recovery": lambda: (rec["recovered_frac"].max() <= 0.2, f"max {rec['recovered_frac'].max():.2f}"),
        "r4-r3m-positive": lambda: (lo1("R4") > 0 and lo1("R3m") > 0, f"R4 {lo1('R4'):.4f}, R3m {lo1('R3m'):.4f}"),
        "only-two-positive": lambda: (sum(lo1(m) > 0 for m in r.index) == 2, "exactly two lower bounds > 0"),
        "top2-r3m-r4": lambda: (set(rec.sort_values("acc_gain").iloc[-2:]["method"]) == {"R3m", "R4"},
                                "R3m and R4 have the two largest mean gains"),
        "r2-no-gain": lambda: (all(r.loc[m, "gain_lo"] <= 0 <= r.loc[m, "gain_hi"] for m in r.index if m.startswith("R2")),
                               "every R2 interval contains zero"),
        "r1-no-gain": lambda: (r.loc["R1", "gain_lo"] <= 0 and r.loc["R1", "acc_gain"] <= 0.005, "R1 gain <= 0.5 pp"),
        "r4-best-ece": lambda: (ece.drop(["SIM", "EMU", "R0"], errors="ignore").idxmin() == "R4", "lowest ECE"),
        "cpsim-under": lambda: (cov["CP_sim"] < target, f"{cov['CP_sim']:.3f}"),
        "hwcal-over": lambda: (later["CP_s1"] > target and cov["CP_same"] > target, "both above target"),
        "aci-near": lambda: (abs(cov["ACI"] - target) <= 0.02, f"{cov['ACI']:.3f}"),
        "aci-smaller-sets": lambda: (size["ACI"] < min(size["CP_s1"], size["CP_same"]), "ACI sets smallest"),
    }
    out = []
    for cid, fn in checks.items():
        try:
            ok, detail = fn()
        except Exception as exc:   # missing table or column: report it rather than pass silently
            ok, detail = False, f"could not evaluate ({type(exc).__name__}: {exc})"
        out.append((cid, bool(ok), str(detail)))
    return out


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
    planned = cfg["hardware"]["sessions_per_device"] * len(cfg["hardware"]["devices"])
    pre = f" Based on {n_sessions} of the {planned} planned sessions." if n_sessions < planned else ""
    return table("table*", "Test accuracy by method, averaged over sessions and devices. SIM: noiseless simulator; "
                 "EMU: calibration emulator; R0: hardware without repair; bold: best repair per row." + pre,
                 "tab:accuracy", "ll" + "c" * len(cols), ["Dataset", "Model"] + [METHOD[c] for c in cols], rows)


def tab_repairs(cfg) -> str | None:
    rec = _csv(cfg.results / "recovered.csv")
    if rec is None or not len(rec):
        return None
    rows = []
    for _, r in rec.iterrows():
        gain = f"{100 * r['acc_gain']:+.1f}"
        if "gain_lo" in rec and np.isfinite(r["gain_lo"]):
            gain += f" [{100 * r['gain_lo']:+.1f}, {100 * r['gain_hi']:+.1f}]"
        share = f"{100 * r['recovered_frac']:.0f}" if np.isfinite(r["recovered_frac"]) else "--"
        rows.append([METHOD.get(r["method"], esc(r["method"])), gain, share, f"{r['extra_qpu_s']:.2f}"])
    n_cases = int(rec["n_cases"].max()) if "n_cases" in rec else int(rec["n"].max())
    n_gap = int(rec["n_gap"].max()) if "n_gap" in rec else n_cases
    return table("table", "Repairs ranked by mean accuracy gain over R0 in percentage points, with a 95\\% bootstrap "
                 f"interval over {n_cases} cases (device, session, dataset, model); mean share of the "
                 f"simulator-to-hardware gap recovered in the {n_gap} cases with a gap above 2 points; and extra QPU "
                 "seconds per session.", "tab:repairs", "lccr", ["Repair", "Gain (pp)", "Gap (\\%)", "QPU (s)"], rows)


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
            stat = "largest term: " + REG_TERM.get(h["largest_term"], esc(h["largest_term"]))
        else:
            stat = r"CP$_{\mathrm{s1}}$ " + ci(*h["cp_s1"]) + ", ACI " + ci(*h["aci"])
        rows.append([tag, text, stat, {True: "supported", False: "not supported", None: "--"}[h.get("supported")]])
    planned = cfg["hardware"]["sessions_per_device"] * len(cfg["hardware"]["devices"])
    note = f" Based on {n_sessions} of the {planned} planned sessions." if n_sessions < planned else ""
    return table("table*", "Pre-registered hypotheses, test statistics (mean and 95\\% bootstrap interval) and "
                 "decisions." + note, "tab:hypotheses", "llll", ["", "Hypothesis", "Statistic", "Decision"], rows)


def pilot_results(cfg, proc) -> list[tuple[str, dict]]:
    """analyze_pilot for every saved pilot, as (device, result) pairs."""
    out = []
    for device in devices_for(cfg):
        pdir = cfg.raw / "pilot" / device
        for p in (sorted(q for q in pdir.iterdir() if q.is_dir()) if pdir.exists() else []):
            out.append((device, analyze_pilot(cfg, proc, p)))
    return out


def pilot_numbers(pilots) -> dict:
    tv = [r["tvd_excess_ci"] for _, res in pilots for r in res.values() if isinstance(r, dict)]
    ac = [r["acc_diff_ci"] for _, res in pilots for r in res.values() if isinstance(r, dict)]
    if not tv:
        return {}
    zero = all(lo <= 0 <= h for lo, h in tv + ac)
    return {"pilot-checks": len(tv), "pilot-tvd-hi-max": f"{max(h for _, h in tv):+.3f}",
            "pilot-tvd-lo-min": f"{min(lo for lo, _ in tv):+.3f}",
            "pilot-acc-lo-min": f"{min(lo for lo, _ in ac):+.2f}", "pilot-acc-hi-max": f"{max(h for _, h in ac):+.2f}",
            # The Methods text says that all intervals contain zero; flag it loudly if that is false.
            "pilot-zero": "All" if zero else r"\textbf{CHECK: not all}"}


def tab_pilot(pilots) -> str | None:
    rows = []
    for device, res in pilots:
        for k, r in res.items():
            if isinstance(r, dict):
                lo, hi = r["tvd_excess_ci"]
                dlo, dhi = r["acc_diff_ci"]
                rows.append([short(device), k.split("__")[1], f"[{lo:+.3f}, {hi:+.3f}]", f"{r['acc_packed']:.2f}",
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
    pilots = pilot_results(cfg, proc)
    vals = {**numbers(cfg, proc, runs), **{k: str(x) for k, x in pilot_numbers(pilots).items()}}
    lines = ["% Generated by `uv run qdrift paper`; do not edit.",
             r"\providecommand{\qdset}[2]{\expandafter\def\csname qd@#1\endcsname{#2}}"]
    # Negative numbers get a typographic minus sign ($-$) instead of a hyphen.
    lines += [rf"\qdset{{{k}}}{{{re.sub(r'^-(?=[0-9])', '$-$', v)}}}" for k, v in sorted(vals.items())]
    (gen / "numbers.tex").write_text("\n".join(lines) + "\n")
    tables = {"tab-sim": tab_sim(cfg), "tab-protocol": tab_protocol(runs), "tab-accuracy": tab_accuracy(cfg, n_sessions),
              "tab-repairs": tab_repairs(cfg), "tab-coverage": tab_coverage(cfg),
              "tab-hypotheses": tab_hypotheses(cfg, n_sessions), "tab-pilot": tab_pilot(pilots)}
    for name, tex in tables.items():
        (gen / f"{name}.tex").write_text(tex or f"% {name}: no data yet\n")
    figs = sorted((cfg.results / "figures").glob("*.pdf")) if (cfg.results / "figures").exists() else []
    for f in figs:
        shutil.copy2(f, gen / "figures" / f.name)
    (gen / "refs.bib").write_text(paper_bib((cfg.root / "literature" / "refs.bib").read_text()))
    log(f"wrote {len(vals)} numbers, {sum(t is not None for t in tables.values())} tables and {len(figs)} figures "
        f"to {gen.relative_to(cfg.root)} ({n_sessions} hardware session(s))")
    checks = claims(cfg)
    (gen / "claims.txt").write_text("".join(f"{'ok  ' if ok else 'FAIL'} {cid}: {d}\n" for cid, ok, d in checks))
    failed = [(cid, d) for cid, ok, d in checks if not ok]
    log(f"claims: {len(checks) - len(failed)} of {len(checks)} hold" +
        "".join(f"\n  FAIL {cid}: {d} -> rewrite the sentence tagged % claim:{cid}" for cid, d in failed))
    if compile_pdf:
        if not shutil.which("tectonic"):
            raise SystemExit("tectonic not found; install it with `brew install tectonic` or use Overleaf")
        (paper / "build").mkdir(exist_ok=True)
        for doc in ("main", "interim"):     # full paper and the three-section interim report
            res = subprocess.run(["tectonic", "--keep-logs", "--outdir", "build", f"{doc}.tex"], cwd=paper,
                                 capture_output=True, text=True)
            if res.returncode != 0:
                raise SystemExit(f"{doc}.tex failed:\n" + res.stderr[-3000:])
            warn = list(dict.fromkeys(ln for ln in res.stderr.splitlines()
                                      if ("warning: " in ln.lower() and "underfull" not in ln.lower())
                                      or "undefined" in ln.lower()))
            log(f"{doc}: " + ("\n".join(warn[:20]) if warn else "no LaTeX warnings (underfull boxes ignored)"))
            log(f"built {(paper / 'build' / f'{doc}.pdf').relative_to(cfg.root)}")
    return gen
