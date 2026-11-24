"""Write results/FINDINGS.md from results/metrics.json.   python scripts/write_findings.py"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def main() -> None:
    m = json.loads((RES / "metrics.json").read_text())
    e = m["eda"]
    best = m["best_model"]
    b = m["models"][best]
    lr = m["models"]["Logistic regression"]
    gb = m["models"]["Gradient boosting"]
    base = e["base_rate"]
    lo, hi = b["test_roc_auc_95ci"]
    inp = e["by_prior_inpatient"]
    a1c = e["by_a1c"]
    disc = e["by_discharge"]
    top_disc = max(disc.items(), key=lambda kv: kv[1]["rate"] if kv[1]["n"] >= 300 else -1)
    lines = [
        "# Findings: 30-day readmission of diabetic patients",
        "",
        "Data: Diabetes 130-US Hospitals 1999-2008 (UCI repository, CC BY 4.0; Strack et al., 2014).",
        "All numbers below are generated from `results/metrics.json` by `scripts/write_findings.py`.",
        "",
        "## Summary",
        f"- After cleaning, **{e['rows']:,} patients** remain; **{pct(base)}** were readmitted within 30 days.",
        f"- The best model by cross-validation is **{best.lower()}**: test ROC-AUC **{b['test']['roc_auc']:.3f}** "
        f"(95% bootstrap CI {lo:.3f}-{hi:.3f}), PR-AUC {b['test']['pr_auc']:.3f} against a {base:.3f} baseline.",
        (f"- Logistic regression is close behind (test ROC-AUC {lr['test']['roc_auc']:.3f}), so the simpler, explainable model "
         f"gives up only {gb['test']['roc_auc'] - lr['test']['roc_auc']:.3f} AUC."
         if gb['test']['roc_auc'] - lr['test']['roc_auc'] < 0.01 else
         f"- Logistic regression trails (test ROC-AUC {lr['test']['roc_auc']:.3f}).") if best != "Logistic regression" else
        f"- Gradient boosting did not beat it (test ROC-AUC {gb['test']['roc_auc']:.3f}).",
        f"- Flagging the riskiest 10% of patients catches **{pct(b['top10']['recall'], 0)}** of readmissions, "
        f"with {pct(b['top10']['precision'])} of flagged patients readmitted ({b['top10']['lift']:.1f}x the average).",
        "- Readmission is only modestly predictable from these administrative records: useful for ranking who gets a "
        "follow-up call, not for deciding individual care.",
        "",
        "## What drives the risk",
        f"- **Prior inpatient visits:** {pct(inp['0']['rate'])} readmitted with none in the past year "
        f"vs {pct(inp['4+']['rate'])} with 4 or more.",
        f"- **Discharge destination:** the highest rate among groups of 300+ patients is '{top_disc[0]}' "
        f"({pct(top_disc[1]['rate'])}, n={top_disc[1]['n']:,}).",
        f"- **HbA1c** was measured for only {pct(a1c.get('Measured', {'n': 0})['n'] / e['rows'])} of patients "
        f"(readmitted {pct(a1c.get('Measured', {'rate': 0})['rate'])} vs {pct(a1c.get('Not measured', {'rate': 0})['rate'])} when not measured).",
        "",
        "Largest logistic-regression effects (odds ratio; per SD for numbers, per category vs the average category):",
        "",
        "| Feature | Level | Odds ratio |",
        "|---|---|---|",
    ]
    lines += [f"| {x['feature']} | {x['level']} | {x['odds_ratio']:.2f} |" for x in m["lr_top_effects"][:10]]
    lines += ["", "## Model comparison (test set used once)", "",
              "| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | Brier | Caught in top 10% |", "|---|---|---|---|---|---|"]
    for n, r in m["models"].items():
        lines.append(f"| {n} | {r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_std']:.3f} | {r['test']['roc_auc']:.3f} | "
                     f"{r['test']['pr_auc']:.3f} | {r['test']['brier']:.4f} | {pct(r['top10']['recall'], 0)} |")
    lines += ["", f"Logistic regression C chosen by {m['folds']}-fold CV: " +
              ", ".join(f"C={k}: {v:.4f}" for k, v in m["lr_c_sweep_cv_auc"].items()) + f" -> C={m['lr_C']}.",
              "", "## Fairness check (best model, test set)", "",
              "| Attribute | Group | n | Observed | Predicted | ROC-AUC |", "|---|---|---|---|---|---|"]
    for col, items in m["subgroups_best_model"].items():
        for it in items:
            lines.append(f"| {col} | {it['group']} | {it['n']:,} | {pct(it['readmit_rate'])} | {pct(it['mean_predicted'])} | {it['roc_auc']:.3f} |")
    aucs = [it["roc_auc"] for items in m["subgroups_best_model"].values() for it in items]
    lines += ["", f"Subgroup ROC-AUC ranges from {min(aucs):.3f} to {max(aucs):.3f}; predicted and observed rates are "
              "listed so miscalibration for any group is visible.", "",
              "## Cleaning", "", "| Step | Rows left |", "|---|---|"]
    lines += [f"| {s} | {n:,} |" for s, n in m["cleaning_steps"]]
    lines += ["", "Dropped columns: " + "; ".join(f"{k} ({v})" for k, v in m["dropped_columns"].items()) +
              (". Constant columns: " + ", ".join(m["dropped_constant"]) if m["dropped_constant"] else "") + ".",
              "", "Most-missing raw columns: " +
              ", ".join(f"{k} {pct(v, 0)}" for k, v in list(e["missing_raw"].items())[:5]) + ".",
              "", "## Limits", "",
              "- Data from 1999-2008 US hospitals; practice and coding have changed since.",
              "- Administrative fields only: no social factors, follow-up care, or most lab values.",
              "- One encounter per patient keeps the evaluation honest but discards repeat-visit history beyond the counts.",
              "- The calculator is a demonstration, not a clinical tool."]
    (RES / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    print(RES / "FINDINGS.md")


if __name__ == "__main__":
    main()
