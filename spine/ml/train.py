"""Training: features -> cross-validated material + zone models, reports, ablation.

    python run.py train --source synthetic|real|mixed [--model rf|svm]

Reports always state which data source was used. Synthetic accuracy is NOT a real result.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .. import dataset as D
from ..config import CFG, MODELS, REAL_DIR
from ..functions.library import Env, TapEvent, build_pipeline, run_pipeline
from ..sim.generator import EventSynth
from .model import ModelBundle, feature_subset

SOURCES = ("synthetic", "real", "mixed")
BANNER = {"synthetic": "SYNTHETIC DATA ONLY - not a real-world result",
          "real": "REAL RECORDED DATA", "mixed": "SYNTHETIC + REAL DATA (synthetic used for pre-training)"}


def nominal_baseline_db(rate: int | None = None) -> np.ndarray:
    from ..features import probe_bands_db
    s = EventSynth(seed=99, rate=rate)
    return np.mean([probe_bands_db(s.baseline_probe(), s.fs) for _ in range(6)], axis=0)


def featurize(samples: list[dict], ref_baseline_db=None):
    """-> X (n, f), names. Uses the same F_PREPROCESS/F_FEATURES the live pipeline uses."""
    pipe = build_pipeline([{"code": "F_PREPROCESS"}, {"code": "F_FEATURES"}], ref_baseline_db=ref_baseline_db)
    rows, names = [], None
    for s in samples:
        prof = {}
        if s.get("baseline_db") is not None:
            prof["baseline_db"] = s["baseline_db"]
        if s.get("gain_corr") is not None:
            prof["gain_corr"] = s["gain_corr"]
        ev = TapEvent(raw=s, env=Env(profile=prof, models=None))
        ev = run_pipeline(pipe, ev)
        rows.append(ev.feats)
        names = ev.names
    return np.vstack(rows), names


def make_clf(kind: str):
    if kind == "svm":
        c = SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=0)
    else:
        c = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, n_jobs=-1, random_state=0)
    return Pipeline([("imp", SimpleImputer(strategy="mean", keep_empty_features=True)),
                     ("scale", StandardScaler()), ("clf", c)])


def _cv_predict(model, X, y, groups, folds, weights=None):
    uniq_groups = len(set(groups)) if groups is not None else 0
    if uniq_groups >= 3:
        cv, cv_name = GroupKFold(n_splits=min(folds, uniq_groups)), f"GroupKFold({min(folds, uniq_groups)}) by user"
        kw = dict(groups=groups)
    else:
        n_min = min(Counter(y).values())
        cv = StratifiedKFold(n_splits=max(2, min(folds, n_min)), shuffle=True, random_state=0)
        cv_name, kw = f"StratifiedKFold({cv.n_splits})", {}
    fit = {"clf__sample_weight": weights} if weights is not None else None
    pred = cross_val_predict(model, X, y, cv=cv, params=fit, **kw)
    return pred, cv_name


def _plot_cm(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=110)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _plot_ablation(abl: dict, title: str, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=110)
    names = list(abl)
    vals = [abl[n] * 100 for n in names]
    ax.bar(names, vals, color=["#4aa3ff", "#ffb14a", "#4ade80"][: len(names)])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    ax.set_ylim(0, 105); ax.set_ylabel("CV accuracy (%)"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def gather(source: str, user_samples: list[dict] | None = None):
    samples: list[dict] = []
    if source in ("synthetic", "mixed"):
        samples += D.load_synthetic()
    if source in ("real", "mixed"):
        samples += D.load_real(REAL_DIR)
    if source == "real" and not samples:
        raise SystemExit("No real data found in data/real/. Record some first (python run.py record).")
    for s in samples:
        s.setdefault("weight", 1.0)
    for s in user_samples or []:
        s = dict(s); s["weight"] = CFG["ml"]["user_sample_weight"]; s["source"] = "USER"
        samples.append(s)
    return samples


def train(source: str = "synthetic", model_type: str | None = None, user_samples: list[dict] | None = None,
          tag: str | None = None, set_latest: bool = True, verbose: bool = True,
          evaluate: bool = True) -> str:
    model_type = model_type or CFG["ml"]["model"]
    samples = gather(source, user_samples)
    if verbose:
        print(f"[train] source={source} model={model_type} samples={len(samples)}")
    if source == "real":
        bl = [s["baseline_db"] for s in samples if s.get("baseline_db") is not None]
        ref = np.mean(bl, axis=0) if bl else None
    else:
        ref = nominal_baseline_db()
    X, names = featurize(samples, ref)
    y_m = np.array([s["material"] for s in samples])
    y_z = np.array([s["zone"] for s in samples])
    groups = np.array([s.get("user", "?") for s in samples])
    w = np.array([s.get("weight", 1.0) for s in samples])
    w_arg = w if np.any(w != 1.0) else None
    has_probe = not np.isnan(X[:, names.index("probe_c0_b0")]).all()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = tag or f"v{stamp}-{source}-{model_type}"
    out = MODELS / tag
    out.mkdir(parents=True, exist_ok=True)
    folds = CFG["ml"]["cv_folds"]
    meta = dict(tag=tag, source=source, banner=BANNER[source], model_type=model_type, created=stamp,
                n_samples=len(samples), counts_material={str(k): int(v) for k, v in Counter(y_m).items()}, counts_zone=dict(Counter(y_z)),
                features={}, ref_baseline_db=None if ref is None else np.asarray(ref).tolist(),
                has_probe=bool(has_probe), user_samples=len(user_samples or []))
    report_md = [f"# Model report `{tag}`", "", f"**Data source: {BANNER[source]}**", "",
                 f"- samples: {len(samples)} (material counts {meta['counts_material']})",
                 f"- classifier: {model_type}"]
    results = {}
    for kind, y in (("material", y_m), ("zone", y_z)):
        if len(set(y)) < 2:
            report_md.append(f"- {kind}: only one class in data, skipped")
            continue
        fnames = feature_subset(names, kind)
        if kind == "material" and not has_probe:
            fnames = [n for n in fnames if n.startswith("tap_")]
        cols = [names.index(n) for n in fnames]
        Xk = X[:, cols]
        model = make_clf(model_type)
        fit = {"clf__sample_weight": w_arg} if w_arg is not None else {}
        model.fit(Xk, y, **fit)
        joblib.dump(model, out / f"{kind}.joblib")
        meta["features"][kind] = fnames
        if not evaluate:
            continue
        pred, cv_name = _cv_predict(make_clf(model_type), Xk, y, groups, folds, w_arg)
        labels = sorted(set(y))
        acc = float(accuracy_score(y, pred))
        rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, pred, labels=labels)
        _plot_cm(cm, labels, f"{kind} - {source} - CV acc {acc * 100:.1f}%\n{BANNER[source]}",
                 out / f"confusion_{kind}.png")
        results[kind] = dict(accuracy=acc, cv=cv_name, report=rep, labels=labels, confusion=cm.tolist())
        report_md += ["", f"## {kind}", f"- CV: {cv_name}", f"- cross-validated accuracy: **{acc * 100:.1f}%**", "",
                      "| class | precision | recall | support |", "|---|---|---|---|"]
        for c in labels:
            r = rep[c]
            report_md.append(f"| {c} | {r['precision']:.2f} | {r['recall']:.2f} | {int(r['support'])} |")
    # ablation (material): tap-only vs probe-only vs combined
    if "material" in results and evaluate:
        abl = {}
        variants = ["tap", "probe", "combined"] if has_probe else ["tap"]
        for v in variants:
            fn = feature_subset(names, "material", v)
            Xv = X[:, [names.index(n) for n in fn]]
            pred, _ = _cv_predict(make_clf(model_type), Xv, y_m, groups, folds, w_arg)
            abl[{"tap": "tap-only (passive)", "probe": "probe-only (active)",
                 "combined": "combined (hybrid)"}[v]] = float(accuracy_score(y_m, pred))
        results["ablation"] = abl
        _plot_ablation(abl, f"Material ablation - {BANNER[source]}", out / "ablation.png")
        report_md += ["", "## Ablation (material)", "", "| features | CV accuracy |", "|---|---|"]
        report_md += [f"| {k} | {v * 100:.1f}% |" for k, v in abl.items()]
        if not has_probe:
            report_md.append("\n_No probe data in this dataset: probe-only / combined not evaluated._")
    meta["results"] = {k: ({kk: vv for kk, vv in v.items() if kk != "report"} if k != "ablation" else v)
                       for k, v in results.items()}
    meta["accuracy"] = {k: results[k]["accuracy"] for k in ("material", "zone") if k in results}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "report.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")
    if set_latest:
        (MODELS / "latest.json").write_text(json.dumps({"tag": tag}))
    if verbose:
        print("\n".join(report_md))
        print(f"[train] saved models/{tag}")
    return tag


def ensure_model(verbose: bool = True) -> ModelBundle:
    """Used by the demo: guarantee a model exists (trains on synthetic if nothing is there)."""
    mb = ModelBundle.latest()
    if mb is None:
        if verbose:
            print("[model] none found -> training on SYNTHETIC data (takes ~1 min)...")
        train("synthetic", verbose=False)
        mb = ModelBundle.latest()
    return mb
