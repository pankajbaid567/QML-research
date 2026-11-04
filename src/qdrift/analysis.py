"""Analysis (Task 10): repairs per session, RQ1-RQ4 tables, hypothesis checks and figures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .circuits import features_from_probs
from .metrics import accuracy_ci, bootstrap_ci, classification_metrics
from .repairs import (aci_stream, anchored_refit, conformal_quantile, fit_head, mitigate, nonconformity,
                      prediction_sets, readout_from_cal, set_stats)
from .session import key, session_dirs
from .train import head

CALIBRATION_COVARIATES = ["readout", "twoq", "t1_us", "t2_us", "age_h"]


def load_session(path: Path) -> dict:
    """Session data. `snap` is the calibration in force when the job ran if it was saved
    (used for R3r and the calibration regression); `snap_submit` is the one the emulator used."""
    submit = json.loads((path / "snapshot.json").read_text())
    at_run = path / "snapshot_at_run.json"
    out = {"path": path, "meta": json.loads((path / "meta.json").read_text()), "snap_submit": submit,
           "snap": json.loads(at_run.read_text()) if at_run.exists() else submit, "pred": None}
    with np.load(path / "counts.npz") as z:
        out["counts"] = {k: z[k] for k in z.files}
    if (path / "emulator.npz").exists():
        with np.load(path / "emulator.npz") as z:
            out["pred"] = {k: z[k] for k in z.files}
    return out


def _probs(c) -> np.ndarray:
    c = np.asarray(c, float)
    return c / c.sum(1, keepdims=True)


def measured_readout(counts, n: int, P: int):
    return [readout_from_cal(counts["readout__0"][k], counts["readout__1"][k], n) for k in range(P)]


def reported_readout(snap, patches):
    q = {r["q"]: r for r in snap["qubits"]}

    def get(i, f):
        return q[i][f] if q[i][f] is not None else q[i]["readout_error"]

    return [(np.array([get(i, "prob_meas1_prep0") for i in p], float),
             np.array([get(i, "prob_meas0_prep1") for i in p], float)) for p in patches]


def features(spec, counts, dataset: str, P: int, mitig=None) -> np.ndarray:
    """Features from stored counts; sample i was measured on patch i % P."""
    pb = {}
    for b in spec.bases:
        pr = _probs(counts[key(dataset, spec.name, b)])
        if mitig is not None:
            slot = np.arange(len(pr)) % P
            for k in range(P):
                if (slot == k).any():
                    pr[slot == k] = mitigate(pr[slot == k], *mitig[k])
        pb[b] = pr
    return features_from_probs(spec, pb)


def patch_covariates(snap, patches) -> list[dict]:
    q = {r["q"]: r for r in snap["qubits"]}
    err = {frozenset((e["a"], e["b"])): e["error"] for e in snap["edges"]}
    try:
        # Age of the calibration when the job ran. Run-time snapshots carry `properties_at`
        # (the job start); their `captured_utc` is only when the snapshot was fetched.
        ref = snap.get("properties_at") or snap["captured_utc"]
        age = (datetime.fromisoformat(ref) - datetime.fromisoformat(snap["last_update"])).total_seconds() / 3600
    except Exception:
        age = np.nan

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else np.nan

    return [{"readout": mean(q[i]["readout_error"] for i in p), "twoq": mean(err.get(frozenset(e)) for e in zip(p, p[1:])),
             "t1_us": mean((q[i]["t1"] or 0) * 1e6 for i in p), "t2_us": mean((q[i]["t2"] or 0) * 1e6 for i in p),
             "age_h": age} for p in patches]


def devices_for(cfg) -> list[str]:
    hw = cfg["hardware"]
    return cfg["dryrun"]["devices"] if cfg.dry_run else hw["devices"] + hw.get("fallback_devices", [])


def evaluate(cfg, proc) -> dict[str, pd.DataFrame]:
    """Metrics for every session, method, dataset and model, plus per-sample and fit tables."""
    m, hw, ex = cfg["models"], cfg["hardware"], cfg["experiment"]
    alpha, nb, seed = ex["alpha"], ex["bootstrap"], ex["seed"]
    rows, cp_rows, fit_rows, sample_rows, streams = [], [], [], [], {}
    for device in devices_for(cfg):
        sessions = [load_session(p) for p in session_dirs(cfg, device)]
        if not sessions:
            continue
        na_path = cfg.noise_aware / f"{device}.npz"
        na = dict(np.load(na_path)) if na_path.exists() else None
        r4_heads, first = {}, sessions[0]
        P1 = len(first["meta"]["patches"])
        for si, S in enumerate(sessions):
            meta = S["meta"]
            P, n, shots = len(meta["patches"]), meta["n_qubits"], meta["shots"]
            mit_m, mit_r = measured_readout(S["counts"], n, P), reported_readout(S["snap"], meta["patches"])
            cov = patch_covariates(S["snap"], meta["patches"])
            for dname in cfg["data"]["datasets"]:
                ds = proc["datasets"][dname]
                sp, y = ds["splits"], ds["y"]
                ncal, ycal, ytest = len(sp["cal"]), y[sp["cal"]], y[sp["test"]]
                for spec in proc["specs"]:
                    per_sample = len(spec.bases) * shots * hw["per_execution_s"] / P
                    sim = proc["sim_features"][f"{dname}__{spec.name}"]
                    R0, R1 = head(proc, dname, spec.name, "R0"), head(proc, dname, spec.name, "R1")
                    F = features(spec, S["counts"], dname, P)
                    Fc, Ft = F[:ncal], F[ncal:]
                    F1c = features(spec, first["counts"], dname, P1)[:ncal]
                    base = {"device": device, "session": si + 1, "created_utc": meta["created_utc"],
                            "dataset": dname, "model": spec.name}
                    probs = {"SIM": R0.proba(sim[sp["test"]]), "R0": R0.proba(Ft), "R1": R1.proba(Ft)}
                    cost = dict.fromkeys(probs, 0.0)
                    if S["pred"] is not None:
                        probs["EMU"], cost["EMU"] = R0.proba(features(spec, S["pred"], dname, P)[ncal:]), 0.0
                    for k in m["refit_k"]:
                        probs[f"R2_{k}"] = anchored_refit(R0, Fc[:k], ycal[:k], m["refit_lambda_per_sample"] * k).proba(Ft)
                        cost[f"R2_{k}"] = k * per_sample
                    probs["R2_16_s1"] = anchored_refit(R0, F1c[:16], ycal[:16], m["refit_lambda_per_sample"] * 16).proba(Ft)
                    cost["R2_16_s1"] = 0.0 if si else 16 * per_sample
                    probs["R3m"] = R0.proba(features(spec, S["counts"], dname, P, mit_m)[ncal:])
                    cost["R3m"] = 2 * meta["readout_shots"] * hw["per_execution_s"]
                    probs["R3r"], cost["R3r"] = R0.proba(features(spec, S["counts"], dname, P, mit_r)[ncal:]), 0.0
                    if na is not None:
                        hk = (dname, spec.name)
                        if hk not in r4_heads:
                            r4_heads[hk] = fit_head(features(spec, na, dname, P), y[sp["train"]], "fixed",
                                                    m["head_Cs"], seed)
                        probs["R4"], cost["R4"] = r4_heads[hk].proba(Ft), 0.0
                    for meth, p in probs.items():
                        lo, hi = accuracy_ci(ytest, p, nb, seed)
                        rows.append({**base, "method": meth, **classification_metrics(ytest, p),
                                     "acc_lo": lo, "acc_hi": hi, "extra_qpu_s": cost[meth]})
                    # conformal prediction (R5)
                    sim_scores = nonconformity(R0.proba(sim[sp["val"]]), y[sp["val"]])
                    quantiles = {"CP_sim": (conformal_quantile(sim_scores, alpha), 0.0),
                                 "CP_s1": (conformal_quantile(nonconformity(R0.proba(F1c), ycal), alpha),
                                           0.0 if si else ncal * per_sample),
                                 "CP_same": (conformal_quantile(nonconformity(R0.proba(Fc), ycal), alpha),
                                             ncal * per_sample)}
                    for meth, (q, c) in quantiles.items():
                        cv, size = set_stats(prediction_sets(probs["R0"], q), ytest)
                        cp_rows.append({**base, "method": meth, "coverage": cv, "set_size": size, "extra_qpu_s": c})
                    streams.setdefault((device, dname, spec.name), {"scores": sim_scores, "parts": []})["parts"].append(
                        (si + 1, base, probs["R0"], ytest))
                    # noise model F_hw = gamma * F_sim + beta (RQ3)
                    St = sim[sp["test"]]
                    for j, fname in enumerate(spec.feature_names()):
                        x, z = St[:, j], Ft[:, j]
                        if np.var(x) < 1e-9:
                            continue
                        g, b0 = np.polyfit(x, z, 1)
                        r2 = 1 - np.sum((z - (g * x + b0)) ** 2) / max(np.sum((z - z.mean()) ** 2), 1e-12)
                        fit_rows.append({**base, "feature": fname, "gamma": g, "beta": b0, "r2": r2})
                    # per-sample table (RQ2 regression and drift beyond shot noise)
                    correct = (probs["R0"] >= 0.5).astype(int) == ytest
                    err = np.abs(Ft - St).mean(1)
                    for i, s in enumerate(sp["test"]):
                        slot = (ncal + i) % P
                        sample_rows.append({**base, "sample": int(s), "slot": slot, "correct": int(correct[i]),
                                            "feat_err": float(err[i]), **cov[slot],
                                            "feats": Ft[i].tolist(), "shots": shots})
    for parts in streams.values():   # adaptive conformal inference over the session stream
        ps = sorted(parts["parts"], key=lambda t: t[0])
        p = np.concatenate([t[2] for t in ps])
        y = np.concatenate([t[3] for t in ps])
        covered, sizes = aci_stream(p, y, parts["scores"], alpha, ex["aci_gamma"], update=True)
        pos = 0
        for _, base, pp, _ in ps:
            sl = slice(pos, pos + len(pp))
            cp_rows.append({**base, "method": "ACI", "coverage": float(covered[sl].mean()),
                            "set_size": float(sizes[sl].mean()), "extra_qpu_s": 0.0})
            pos += len(pp)
    return {"metrics": pd.DataFrame(rows), "conformal": pd.DataFrame(cp_rows),
            "noise_fits": pd.DataFrame(fit_rows), "samples": pd.DataFrame(sample_rows)}


# ---------------------------------------------------------------- research questions

def _paired(df, a: str, b: str, col: str = "acc") -> pd.DataFrame:
    idx = ["device", "session", "dataset", "model"]
    w = df.pivot_table(index=idx, columns="method", values=col)
    return w[[a, b]].dropna()


def _mean_ci(x, nb: int, seed: int):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan
    lo, hi = bootstrap_ci(lambda i: x[i].mean(), len(x), nb, seed)
    return float(x.mean()), lo, hi


def overdispersion(samples: pd.DataFrame) -> pd.DataFrame:
    """Across-session variance of each hardware feature divided by its expected shot variance."""
    out = []
    for (dev, dn, mo), g in samples.groupby(["device", "dataset", "model"]):
        if g["session"].nunique() < 2:
            continue
        for s, gs in g.groupby("sample"):
            F = np.array(gs["feats"].tolist())
            shot_var = ((1 - F**2) / gs["shots"].to_numpy()[:, None]).mean(0)
            ratio = F.var(0, ddof=1) / np.maximum(shot_var, 1e-6)
            # Under pure shot noise E[ratio] = 1. The median is biased low (scaled chi-square with
            # sessions - 1 degrees of freedom), so the mean is the test statistic.
            out.append({"device": dev, "dataset": dn, "model": mo, "sample": s, "ratio": float(np.mean(ratio)),
                        "sessions": int(len(gs))})
    return pd.DataFrame(out)


def calibration_regression(samples: pd.DataFrame) -> pd.DataFrame:
    """OLS of per-sample feature error on standardised patch calibration data (HC3 errors)."""
    import statsmodels.formula.api as smf
    d = samples.copy()
    used = []
    for c in CALIBRATION_COVARIATES:
        v = d[c].astype(float)
        if v.notna().all() and v.std() > 1e-12:
            d[c] = (v - v.mean()) / v.std()
            used.append(c)
    if not used:
        return pd.DataFrame()
    terms = used + (["C(device)"] if d["device"].nunique() > 1 else [])
    fit = smf.ols("feat_err ~ " + " + ".join(terms) + " + C(dataset):C(model)", data=d).fit(cov_type="HC3")
    ci = fit.conf_int()
    return pd.DataFrame({"term": used, "coef": [fit.params[c] for c in used], "lo": [ci.loc[c, 0] for c in used],
                         "hi": [ci.loc[c, 1] for c in used], "p": [fit.pvalues[c] for c in used]})


def hypotheses(cfg, t: dict) -> dict:
    nb, seed, alpha = cfg["experiment"]["bootstrap"], cfg["experiment"]["seed"], cfg["experiment"]["alpha"]
    met, cp, out = t["metrics"], t["conformal"], {}
    if "EMU" in set(met["method"]):
        w = _paired(met, "EMU", "R0")
        d = w["EMU"] - w["R0"]
        slope = np.polyfit(w["EMU"], w["R0"], 1)[0] if len(w) > 2 and w["EMU"].std() > 0 else np.nan
        mean, lo, hi = _mean_ci(d, nb, seed)
        out["H1_emulator_overestimates"] = {"mean_emu_minus_hw": mean, "ci": [lo, hi], "slope_hw_on_emu": slope,
                                           "supported": bool(lo > 0) if np.isfinite(lo) else None}
    od = overdispersion(t["samples"])
    if len(od):
        mean, lo, hi = _mean_ci(od["ratio"], nb, seed)
        out["H2_drift_beyond_shot_noise"] = {"mean_ratio": mean, "ci": [lo, hi], "supported": bool(lo > 1)}
    w = _paired(met, "R2_16", "R1")
    mean, lo, hi = _mean_ci(w["R2_16"] - w["R1"], nb, seed)
    out["H3_refit16_beats_normalisation"] = {"mean_diff": mean, "ci": [lo, hi],
                                             "supported": bool(lo > 0) if np.isfinite(lo) else None}
    reg = t.get("regression")
    if reg is not None and len(reg):
        top = reg.iloc[reg["coef"].abs().argmax()]["term"]
        out["H4_readout_strongest"] = {"largest_term": top, "supported": top == "readout"}
    later = cp[(cp["session"] > 1)]
    if len(later):
        s1 = later[later["method"] == "CP_s1"]["coverage"]
        aci = later[later["method"] == "ACI"]["coverage"]
        m1, lo1, hi1 = _mean_ci(s1, nb, seed)
        m2, lo2, hi2 = _mean_ci(aci, nb, seed)
        out["H5_split_undercovers_aci_holds"] = {"target": 1 - alpha, "cp_s1": [m1, lo1, hi1], "aci": [m2, lo2, hi2],
                                                 "supported": bool(hi1 < 1 - alpha and abs(m2 - (1 - alpha)) <= 0.03)}
    return out


def recovered(met: pd.DataFrame, nb: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Accuracy gain of each method over R0 (mean and bootstrap interval over cases), the share of
    the simulator-to-hardware gap it recovers (cases with a gap above 2 points), and its QPU cost.
    A case is one device, session, dataset and model."""
    idx = ["device", "session", "dataset", "model"]
    w = met.pivot_table(index=idx, columns="method", values="acc")
    c = met.pivot_table(index=idx, columns="method", values="extra_qpu_s")
    gap = (w["SIM"] - w["R0"]).where(lambda g: g > 0.02)
    rows = []
    for meth in w.columns:
        if meth in ("SIM", "R0", "EMU"):
            continue
        gain = (w[meth] - w["R0"]).dropna()
        mean, lo, hi = _mean_ci(gain, nb, seed)
        frac = ((w[meth] - w["R0"]) / gap).dropna()
        rows.append({"method": meth, "acc_gain": mean, "gain_lo": lo, "gain_hi": hi, "n_cases": int(len(gain)),
                     "recovered_frac": float(frac.mean()) if len(frac) else np.nan, "n_gap": int(len(frac)),
                     "extra_qpu_s": float(c[meth].mean())})
    return pd.DataFrame(rows).sort_values("acc_gain", ascending=False)


# ---------------------------------------------------------------- figures

DS_NAME = {"two_curves": "Two curves", "breast_cancer": "Breast cancer"}
METHOD_NAME = {"SIM": "simulator", "EMU": "emulator", "R0": "R0", "R1": "R1", "R2_8": r"R2$_{8}$",
               "R2_16": r"R2$_{16}$", "R2_32": r"R2$_{32}$", "R2_16_s1": r"R2$_{16}^{\mathrm{s1}}$",
               "R3m": r"R3$_{\mathrm{m}}$", "R3r": r"R3$_{\mathrm{r}}$", "R4": "R4",
               "CP_sim": r"CP$_{\mathrm{sim}}$", "CP_s1": r"CP$_{\mathrm{s1}}$", "CP_same": r"CP$_{\mathrm{same}}$",
               "ACI": "ACI"}
COL_IN, TEXT_IN = 3.5, 7.16   # IEEE two-column: column width and text width in inches


def _pyplot():
    """Matplotlib with an IEEE-like style: 8 pt serif text, embedded TrueType fonts in PDFs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "legend.frameon": False, "legend.handlelength": 1.6,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
    return plt


def figures(t: dict, out: Path, alpha: float) -> list[Path]:
    """Six figures sized for an IEEE two-column paper (PNG for viewing, PDF for LaTeX)."""
    plt = _pyplot()
    from matplotlib.lines import Line2D
    out.mkdir(parents=True, exist_ok=True)
    met, cp, fits, rec = t["metrics"], t["conformal"], t["noise_fits"], t["recovered"]
    datasets, models = sorted(met["dataset"].unique()), sorted(met["model"].unique())
    devices, sessions = sorted(met["device"].unique()), sorted(int(s) for s in met["session"].unique())
    dstyle = dict(zip(devices, ["-", "--", ":"]))
    mcolor = {m: f"C{i}" for i, m in enumerate(models)}
    paths = []

    def short(d):
        return d.removeprefix("ibm_")

    def save(fig, name, rect=None):
        p = out / name
        fig.tight_layout(rect=rect) if rect else fig.tight_layout()
        fig.savefig(p, dpi=300)
        fig.savefig(p.with_suffix(".pdf"))   # vector copy for the paper
        plt.close(fig)
        paths.append(p)

    def device_handles(color="k"):
        return [Line2D([], [], color=color, ls=dstyle[d], label=short(d)) for d in devices]

    # 1. simulator vs emulator vs hardware (mean and spread over sessions and devices)
    fig, axes = plt.subplots(1, len(datasets), figsize=(TEXT_IN, 2.0), squeeze=False)
    for ax, dn in zip(axes[0], datasets):
        g = met[met["dataset"] == dn]
        for j, meth in enumerate(["SIM", "EMU", "R0"]):
            s = g[g["method"] == meth].groupby("model")["acc"].agg(["mean", "std"]).reindex(models)
            ax.errorbar(np.arange(len(models)) + (j - 1) * 0.2, s["mean"], yerr=s["std"].fillna(0), fmt="o",
                        capsize=2, label={"SIM": "simulator", "EMU": "calibration emulator", "R0": "hardware"}[meth])
        ax.set_xticks(range(len(models)), models)
        ax.set_title(DS_NAME.get(dn, dn))
        ax.set_ylabel("test accuracy")
    axes[0][0].legend()
    save(fig, "fig1_sim_emulator_hardware.png")

    # 2. drift: accuracy per session, R0 with 95% intervals, selected repairs, simulator as reference
    shown = [x for x in ["R0", "R1", "R2_16", "R3m", "R4"] if x in set(met["method"])]
    rcolor = {m: f"C{i}" for i, m in enumerate(shown)}
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(TEXT_IN, 1.75 * len(datasets) + 0.35),
                             squeeze=False, sharex=True, sharey="row")
    for i, dn in enumerate(datasets):
        for j, mo in enumerate(models):
            ax = axes[i][j]
            g = met[(met["dataset"] == dn) & (met["model"] == mo)]
            ax.axhline(g[g["method"] == "SIM"]["acc"].mean(), color="0.5", ls=":", lw=0.9)
            for meth in shown:
                for dev in devices:
                    s = g[(g["method"] == meth) & (g["device"] == dev)].sort_values("session")
                    if s.empty:
                        continue
                    if meth == "R0":
                        ax.errorbar(s["session"], s["acc"], yerr=[s["acc"] - s["acc_lo"], s["acc_hi"] - s["acc"]],
                                    color=rcolor[meth], ls=dstyle[dev], marker="o", ms=2.8, capsize=1.5,
                                    elinewidth=0.5)
                    else:
                        ax.plot(s["session"], s["acc"], color=rcolor[meth], ls=dstyle[dev], marker="o", ms=2.2,
                                lw=0.8, alpha=0.9)
            ax.set_title(f"{DS_NAME.get(dn, dn)}, {mo}")
            if i == len(datasets) - 1:
                ax.set_xlabel("session")
            if j == 0:
                ax.set_ylabel("test accuracy")
    axes[0][0].set_xticks(sessions)
    handles = [Line2D([], [], color=rcolor[m], marker="o", ms=3, label=METHOD_NAME.get(m, m)) for m in shown]
    handles += [Line2D([], [], color="0.5", ls=":", label="simulator")] + device_handles()
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig2_drift.png", rect=(0, 0, 1, 0.92))

    # 3. emulator prediction vs hardware: colour = model, marker = dataset, filled = first device
    if "EMU" in set(met["method"]):
        w = met.pivot_table(index=["device", "session", "dataset", "model"], columns="method",
                            values="acc")[["EMU", "R0"]].dropna().reset_index()
        marker = dict(zip(datasets, ["o", "s", "^"]))
        fig, ax = plt.subplots(figsize=(COL_IN, 2.9))
        for _, r in w.iterrows():
            c = mcolor[r["model"]]
            ax.scatter(r["EMU"], r["R0"], marker=marker[r["dataset"]], s=18, linewidths=0.8, edgecolors=c,
                       facecolors=c if r["device"] == devices[0] else "none")
        lo = float(min(w["EMU"].min(), w["R0"].min())) - 0.03
        ax.plot([lo, 1], [lo, 1], "k--", lw=0.7)
        ax.set_xlim(lo, 1)
        ax.set_ylim(lo, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("emulator-predicted accuracy")
        ax.set_ylabel("hardware accuracy (R0)")
        handles = [Line2D([], [], ls="", marker="o", color=mcolor[m], label=m) for m in models]
        handles += [Line2D([], [], ls="", marker=marker[d], color="0.35", label=DS_NAME.get(d, d)) for d in datasets]
        handles += [Line2D([], [], ls="", marker="o", color="0.35", label=f"filled: {short(devices[0])}")]
        handles += [Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor="0.35",
                           label=f"open: {short(d)}") for d in devices[1:]]
        ax.legend(handles=handles, loc="upper left", handletextpad=0.3)
        save(fig, "fig3_emulator_vs_hardware.png")

    # 4. noise model F_hw = gamma F_sim + beta: median gamma and mean |beta| per session
    if len(fits):
        g = fits.assign(abs_beta=fits["beta"].abs()).groupby(["device", "model", "session"]).agg(
            gamma=("gamma", "median"), abs_beta=("abs_beta", "mean")).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 1.95), sharex=True)
        for ax, col, lab in zip(axes, ["gamma", "abs_beta"], [r"median scale $\gamma$", r"mean shift $|\beta|$"]):
            for (dev, mo), s in g.groupby(["device", "model"]):
                ax.plot(s["session"], s[col], color=mcolor[mo], ls=dstyle[dev], marker="o", ms=2.5)
            ax.set_xlabel("session")
            ax.set_ylabel(lab)
            ax.set_xticks(sessions)
        handles = [Line2D([], [], color=mcolor[m], label=m) for m in models] + device_handles()
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
        save(fig, "fig4_noise_model.png", rect=(0, 0, 1, 0.86))

    # 5. accuracy gain of each repair (mean and 95% interval) against its extra QPU time.
    # Repairs with the same cost are spread horizontally (by 0.03 s) so their intervals stay readable.
    fig, ax = plt.subplots(figsize=(COL_IN, 2.3))
    cost = rec["extra_qpu_s"].round(3)
    rank = rec.groupby(cost).cumcount().to_numpy()
    size = cost.map(cost.value_counts()).to_numpy()
    for (_, r), k, m in zip(rec.iterrows(), rank, size):
        x, y = r["extra_qpu_s"] + (k - (m - 1) / 2) * 0.03, 100 * r["acc_gain"]
        err = None
        if "gain_lo" in r and np.isfinite(r["gain_lo"]):
            err = [[100 * (r["acc_gain"] - r["gain_lo"])], [100 * (r["gain_hi"] - r["acc_gain"])]]
        ax.errorbar(x, y, yerr=err, fmt="o", color="C0", ms=3, capsize=2, elinewidth=0.6)
        ax.annotate(METHOD_NAME.get(r["method"], r["method"]), (x, y), xytext=(3, 2), textcoords="offset points",
                    fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("extra QPU seconds per session")
    ax.set_ylabel("accuracy gain over R0 (points)")
    save(fig, "fig5_repair_cost.png")

    # 6. conformal coverage per session (mean over devices, datasets and models)
    fig, ax = plt.subplots(figsize=(COL_IN, 2.2))
    g = cp.groupby(["method", "session"])["coverage"].mean().reset_index()
    for meth in ["CP_sim", "CP_s1", "CP_same", "ACI"]:
        s = g[g["method"] == meth]
        if len(s):
            ax.plot(s["session"], s["coverage"], marker="o", ms=3, label=METHOD_NAME[meth])
    ax.axhline(1 - alpha, color="k", ls="--", lw=0.7, label="target")
    ax.set_xticks(sessions)
    ax.set_xlabel("session")
    ax.set_ylabel("coverage")
    ax.legend(ncol=3, loc="best")
    save(fig, "fig6_coverage.png")
    return paths


def analyze(cfg, proc, log=print) -> dict:
    t = evaluate(cfg, proc)
    if t["metrics"].empty:
        raise SystemExit("no sessions found")
    ex = cfg["experiment"]
    t["regression"] = calibration_regression(t["samples"])
    t["recovered"] = recovered(t["metrics"], ex["bootstrap"], ex["seed"])
    t["overdispersion"] = overdispersion(t["samples"])
    hyp = hypotheses(cfg, t)
    res = cfg.results
    res.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "conformal", "noise_fits", "regression", "recovered", "overdispersion"):
        t[name].to_csv(res / f"{name}.csv", index=False)
    t["samples"].drop(columns=["feats"]).to_csv(res / "samples.csv", index=False)
    (res / "hypotheses.json").write_text(json.dumps(hyp, indent=1, default=float))
    figs = figures(t, res / "figures", cfg["experiment"]["alpha"])
    log(t["recovered"].to_string(index=False))
    log(json.dumps(hyp, indent=1, default=float))
    log(f"wrote {len(figs)} figures to {(res / 'figures').relative_to(cfg.root)}")
    return {"tables": t, "hypotheses": hyp, "figures": figs}


def analyze_pilot(cfg, proc, path: Path) -> dict:
    """Packed vs isolated: distribution distance against a shot-noise null, and accuracy difference."""
    S = load_session(path)
    meta, counts = S["meta"], S["counts"]
    shots, P = meta["shots"], len(meta["patches"])
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    out = {"usage_s": meta.get("usage_s"), "executions": meta.get("executions")}
    if meta.get("usage_s"):
        out["per_execution_s"] = (meta["usage_s"] - cfg["hardware"]["job_overhead_s"]) / meta["executions"]
    ds = proc["datasets"][meta["pilot"]["dataset"]]
    order = np.r_[ds["splits"]["cal"], ds["splits"]["test"]]
    for k, packed in counts.items():
        if not k.startswith(meta["pilot"]["dataset"]) or f"isolated__{k}" not in counts:
            continue
        iso = counts[f"isolated__{k}"]
        a, b = _probs(packed), _probs(iso)
        tvd = 0.5 * np.abs(a - b).sum(1)
        pooled = (a + b) / 2
        null = np.array([0.5 * np.abs(rng.multinomial(shots, p) - rng.multinomial(shots, p)).sum() / shots
                         for p in pooled])
        name, model, basis = k.split("__")
        spec = next(s for s in proc["specs"] if s.name == model)
        R0 = head(proc, name, model, "R0")
        y = ds["y"][order[: len(a)]]
        ca = (R0.proba(features_from_probs(spec, {basis: a})) >= 0.5).astype(int) == y
        cb = (R0.proba(features_from_probs(spec, {basis: b})) >= 0.5).astype(int) == y
        d = (ca.astype(float) - cb.astype(float))
        lo, hi = bootstrap_ci(lambda i: d[i].mean(), len(d), cfg["experiment"]["bootstrap"], 1)
        out[k] = {"tvd_mean": float(tvd.mean()), "null_tvd_mean": float(null.mean()),
                  "tvd_excess_ci": list(bootstrap_ci(lambda i: (tvd - null)[i].mean(), len(tvd), 2000, 2)),
                  "acc_packed": float(ca.mean()), "acc_isolated": float(cb.mean()), "acc_diff_ci": [lo, hi]}
    return out
