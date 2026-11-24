"""Clean the data, explore it, compare models with cross-validation, test once on held-out data.

    python scripts/run_pipeline.py --data data/diabetic_data.csv --out results
    python scripts/run_pipeline.py ... --min-auc 0.60      # exit 1 if the best model scores lower
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from readmit import data as D  # noqa: E402
from readmit import evaluate as E  # noqa: E402
from readmit import export, models, plots  # noqa: E402

SEED = 42


def eda(raw: pd.DataFrame, c: D.Cleaned, out: Path) -> dict:
    df, y = c.df, c.df["target"]
    base = float(y.mean())
    miss = (raw.replace("", np.nan).isna().mean().sort_values(ascending=False))
    missing = {k: float(v) for k, v in miss[miss > 0].items()}

    inp = df["number_inpatient"].clip(upper=4).astype(int)
    lab = inp.map(lambda v: "4+" if v >= 4 else str(v))
    order = ["0", "1", "2", "3", "4+"]
    g = y.groupby(lab).agg(["mean", "size"]).reindex(order)
    plots.rate_bars(order, g["mean"].tolist(), g["size"].astype(int).tolist(),
                    "Readmission rate by inpatient visits in the prior year", "Prior inpatient visits",
                    base, out / "eda_inpatient.png")

    age = df["age_years"].map(lambda a: "<40" if a < 40 else ("40-59" if a < 60 else ("60-79" if a < 80 else "80+")))
    aorder = ["<40", "40-59", "60-79", "80+"]
    ga = y.groupby(age).agg(["mean", "size"]).reindex(aorder)

    a1c = df["A1Cresult"].map(lambda v: "Not measured" if v == "Not measured" else "Measured")
    gt = y.groupby(a1c).agg(["mean", "size"])

    disc = y.groupby(df["discharge_group"]).agg(["mean", "size"]).sort_values("size", ascending=False)
    plots.rate_barh(list(disc.index), disc["mean"].tolist(), disc["size"].astype(int).tolist(),
                    "Readmission rate by discharge destination", base, out / "eda_discharge.png")

    as_dict = lambda t: {str(k): {"rate": float(r["mean"]), "n": int(r["size"])} for k, r in t.iterrows()}  # noqa: E731
    return {"base_rate": base, "rows": int(len(df)), "missing_raw": missing,
            "by_prior_inpatient": as_dict(g), "by_age": as_dict(ga), "by_a1c": as_dict(gt),
            "by_discharge": as_dict(disc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--site", default="site")
    ap.add_argument("--min-auc", type=float, default=None)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    out, site = Path(a.out), Path(a.site)
    out.mkdir(parents=True, exist_ok=True)
    site.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    raw = D.load_raw(a.data)
    c = D.clean(raw)
    df, y = c.df, c.df["target"].to_numpy()
    num, cat = c.numeric, c.categorical
    print(f"cleaned: {len(df):,} patients, readmit rate {y.mean():.3%}, {len(num)} numeric + {len(cat)} categorical")

    info = eda(raw, c, out)

    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = df.iloc[tr], df.iloc[te], y[tr], y[te]
    cv = StratifiedKFold(a.folds, shuffle=True, random_state=SEED)

    # Pick the regularisation strength of the logistic regression by cross-validation.
    c_sweep = {}
    for C in (0.01, 0.1, 1.0):
        s = cross_val_score(models.logistic(num, cat, C), Xtr, ytr, cv=cv, scoring="roc_auc")
        c_sweep[str(C)] = float(s.mean())
    best_C = float(max(c_sweep, key=c_sweep.get))
    print("LR C sweep (CV ROC-AUC):", {k: round(v, 4) for k, v in c_sweep.items()}, "-> C =", best_C)

    results, curves, preds = {}, {}, {}
    for name, pipe in models.all_models(num, cat, SEED, best_C).items():
        s = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        preds[name], curves[name] = p, (yte, p)
        r = {"cv_roc_auc_mean": float(s.mean()), "cv_roc_auc_std": float(s.std()), "test": E.scores(yte, p),
             "top10": E.top_fraction(yte, p, 0.10)}
        if "Baseline" not in name:
            r["test_roc_auc_95ci"] = E.bootstrap_auc(yte, p)
            r["calibration"] = E.calibration(yte, p)
        results[name] = r
        print(f"{name:38s} CV AUC {s.mean():.4f}±{s.std():.4f}  test AUC {r['test']['roc_auc']:.4f}  "
              f"PR-AUC {r['test']['pr_auc']:.4f}  Brier {r['test']['brier']:.4f}")

    real = [k for k in results if "Baseline" not in k]
    best = max(real, key=lambda k: results[k]["cv_roc_auc_mean"])      # chosen on CV, not on test
    pb = preds[best]
    age_band = Xte["age_years"].map(lambda v: "<40" if v < 40 else ("40-59" if v < 60 else ("60-79" if v < 80 else "80+")))
    groups = {"age": E.by_group(pd.DataFrame({"g": age_band.values}), yte, pb, "g"),
              "gender": E.by_group(Xte.reset_index(drop=True), yte, pb, "gender"),
              "race": E.by_group(Xte.reset_index(drop=True), yte, pb, "race")}

    # Compact model for the browser calculator.
    cn, cc = D.COMPACT_NUMERIC, D.COMPACT_CATEGORICAL
    compact = models.logistic(cn, cc, best_C).fit(Xtr, ytr)
    pc = compact.predict_proba(Xte)[:, 1]
    compact_scores = E.scores(yte, pc)
    meta = {"test_roc_auc": compact_scores["roc_auc"], "base_rate": float(ytr.mean()), "C": best_C,
            "trained_on": int(len(Xtr))}
    mj = export.to_json(compact, cn, cc, meta)
    (site / "model.json").write_text(json.dumps(mj, indent=1))
    rows = Xte.iloc[:60][cn + cc]
    check = {"records": json.loads(rows.to_json(orient="records")), "sklearn": pc[:60].tolist()}
    (out / "calculator_check.json").write_text(json.dumps(check))

    full_lr = models.all_models(num, cat, SEED, best_C)["Logistic regression"].fit(Xtr, ytr)
    effects = export.top_effects(export.to_json(full_lr, num, cat), 12)
    id_file = D.find_id_mapping(Path(a.data).parent)
    if id_file:
        maps = D.load_id_maps(id_file)
        (out / "ids_mapping.json").write_text(json.dumps(maps, indent=1))
        for e in effects:
            if e["feature"] in maps and e["level"] in maps[e["feature"]]:
                e["level"] = f"{e['level']}: {maps[e['feature']][e['level']]}"

    plots.roc(curves, {k: results[k]["test"]["roc_auc"] for k in results}, out / "roc.png")
    plots.calibration({k: results[k]["calibration"] for k in real}, out / "calibration.png")

    metrics = {
        "dataset": "Diabetes 130-US Hospitals 1999-2008 (UCI id 296)",
        "cleaning_steps": c.steps, "dropped_columns": D.DROPPED, "dropped_constant": c.dropped_constant,
        "n_features": {"numeric": len(num), "categorical": len(cat)},
        "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "folds": a.folds, "seed": SEED,
        "lr_c_sweep_cv_auc": c_sweep, "lr_C": best_C, "models": results, "best_model": best,
        "subgroups_best_model": groups, "compact_model": {"features": cn + cc, "test": compact_scores},
        "lr_top_effects": effects, "eda": info, "seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"best (by CV): {best}; compact calculator AUC {compact_scores['roc_auc']:.4f}; {metrics['seconds']} s")

    if a.min_auc is not None:
        auc = results[best]["test"]["roc_auc"]
        base_auc = results["Baseline (always the average rate)"]["test"]["roc_auc"]
        if auc < a.min_auc or results["Logistic regression"]["test"]["roc_auc"] <= base_auc + 0.05:
            print(f"accuracy gate FAILED: best test AUC {auc:.4f} < {a.min_auc}")
            return 1
        print(f"accuracy gate passed: best test AUC {auc:.4f} >= {a.min_auc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
