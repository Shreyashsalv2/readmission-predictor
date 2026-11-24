"""Build the static results page (site/) from results/metrics.json.   python scripts/build_site.py"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SITE = ROOT / "site"
REPO = "https://github.com/Shreyashsalv2/readmission-predictor"

NICE = {"age_years": "Age", "time_in_hospital": "Days in hospital", "num_lab_procedures": "Lab procedures",
        "num_procedures": "Procedures", "num_medications": "Medications given", "number_outpatient": "Outpatient visits (past year)",
        "number_emergency": "Emergency visits (past year)", "number_inpatient": "Inpatient visits (past year)",
        "number_diagnoses": "Number of diagnoses", "discharge_disposition_id": "Discharge disposition id",
        "admission_source_id": "Admission source id", "admission_type_id": "Admission type id",
        "medical_specialty": "Admitting specialty", "diag_1_group": "Main diagnosis", "diag_2_group": "Second diagnosis",
        "diag_3_group": "Third diagnosis", "A1Cresult": "HbA1c result", "max_glu_serum": "Glucose serum test",
        "diabetesMed": "On diabetes medication", "change": "Medication changed"}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def main() -> None:
    m = json.loads((RES / "metrics.json").read_text())
    SITE.mkdir(exist_ok=True)
    for f in ("roc.png", "calibration.png", "eda_inpatient.png", "eda_discharge.png"):
        shutil.copy2(RES / f, SITE / f)
    best = m["best_model"]
    b = m["models"][best]
    lr = m["models"]["Logistic regression"]
    base = m["eda"]["base_rate"]
    lo, hi = b["test_roc_auc_95ci"]
    t10 = b["top10"]
    cards = [
        (f"{b['test']['roc_auc']:.3f}", "Test ROC-AUC", f"{best.lower()}, 95% CI {lo:.3f}–{hi:.3f}"),
        (pct(t10["recall"], 0), "Readmissions in the top 10%", f"flagging the riskiest 10% catches this share; {t10['lift']:.1f}× the average rate"),
        (pct(base), "Readmitted within 30 days", f"of {m['eda']['rows']:,} patients after cleaning"),
        (f"{b['test']['brier']:.4f}", "Brier score", f"calibration + sharpness; always-average baseline {m['models']['Baseline (always the average rate)']['test']['brier']:.4f}"),
    ]
    card_html = "".join(f"<div class='stat'><b>{a}</b><span>{h}</span><small>{c}</small></div>" for a, h, c in cards)

    rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_std']:.3f}</td>"
        f"<td>{r['test']['roc_auc']:.3f}</td><td>{r['test']['pr_auc']:.3f}</td><td>{r['test']['brier']:.4f}</td>"
        f"<td>{pct(r['top10']['recall'], 0)}</td></tr>" for n, r in m["models"].items())
    steps = "".join(f"<tr><td>{html.escape(s)}</td><td>{n:,}</td></tr>" for s, n in m["cleaning_steps"])
    eff = "".join(f"<tr><td>{html.escape(NICE.get(e['feature'], e['feature']))}</td><td>{html.escape(e['level'])}</td>"
                  f"<td>{e['odds_ratio']:.2f}</td></tr>" for e in m["lr_top_effects"][:10])
    sub_rows = ""
    for col, items in m["subgroups_best_model"].items():
        for it in items:
            sub_rows += (f"<tr><td>{col}</td><td>{html.escape(it['group'])}</td><td>{it['n']:,}</td>"
                         f"<td>{pct(it['readmit_rate'])}</td><td>{pct(it['mean_predicted'])}</td><td>{it['roc_auc']:.3f}</td></tr>")
    cm = m["compact_model"]
    inp = m["eda"]["by_prior_inpatient"]
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>readmission-predictor · 30-day hospital readmission · Shreyash Singh Salv</title>
<meta name="description" content="Predicting 30-day readmission for {m['eda']['rows']:,} diabetic patients with pandas and scikit-learn: test ROC-AUC {b['test']['roc_auc']:.3f}, honest evaluation and an in-browser risk calculator.">
<style>
:root{{--b:#0B3C5D;--b2:#1D6FA3;--l:#EEF4F9;--t:#1B1B1B;--m:#5c6670}}*{{box-sizing:border-box}}
body{{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--t);line-height:1.55}}
header{{background:var(--b);color:#fff;padding:52px 20px 40px}}.w{{max-width:1000px;margin:0 auto;padding:0 20px}}
h1{{font-size:clamp(28px,4.2vw,42px);line-height:1.15;margin:0 0 12px}}h2{{color:var(--b);font-size:24px;margin:0 0 14px}}
header p{{color:#CFE3F3;font-size:18px;max-width:800px;margin:0 0 20px}}.btns a{{display:inline-block;margin:6px 10px 0 0;padding:10px 16px;border-radius:8px;background:#fff;color:var(--b);font-weight:700;text-decoration:none}}
section{{padding:36px 0;border-bottom:1px solid #e3eaf0}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}
.stat{{background:var(--l);border-radius:12px;padding:16px}}.stat b{{display:block;font-size:30px;color:var(--b)}}.stat span{{display:block;font-weight:700}}.stat small{{color:var(--m)}}
img{{max-width:100%;height:auto;border-radius:8px;border:1px solid #e1e7ec}}.muted{{color:var(--m);font-size:14px}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:22px;align-items:start}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid #e3eaf0}}th{{background:var(--l)}}
.scroll{{overflow-x:auto}}code{{background:var(--l);padding:1px 5px;border-radius:4px}}a{{color:var(--b2)}}
.calc{{display:grid;grid-template-columns:minmax(150px,1fr) minmax(140px,1fr);gap:8px 12px;align-items:center}}
.calc input,.calc select{{padding:7px;border:1px solid #c9d6e2;border-radius:6px;font:inherit;width:100%}}
.calc-out{{background:var(--l);border-radius:12px;padding:18px;margin-top:4px}}.risk{{font-size:44px;font-weight:800;color:var(--b)}}.risk-sub{{color:var(--m)}}
.note{{border-left:4px solid #d9822b;background:#fff7ee;padding:10px 14px;border-radius:6px;font-size:14px}}
footer{{padding:28px 0 46px;color:var(--m);font-size:14px}}
</style></head><body>
<header><div class="w"><h1>Which diabetic patients come back within 30 days?</h1>
<p>Cleaning {m['cleaning_steps'][0][1]:,} hospital encounters with pandas, comparing scikit-learn models with cross-validation, and checking the winner on held-out patients: ranking, calibration and fairness across groups. Plus a risk calculator that runs in your browser.</p>
<div class="btns"><a href="#calculator">Try the calculator</a><a href="{REPO}" target="_blank" rel="noopener">Code on GitHub ↗</a><a href="{REPO}/blob/main/results/FINDINGS.md" target="_blank" rel="noopener">Findings report ↗</a></div></div></header>
<main class="w">
<section><h2>Results on {m['test_rows']:,} held-out patients</h2><div class="stats">{card_html}</div>
<p class="muted">Data: Diabetes 130-US Hospitals 1999–2008 (UCI, CC BY 4.0). The model was chosen on {m['folds']}-fold cross-validation over {m['train_rows']:,} training patients; the test set was used once.</p></section>
<section class="two"><div><h2>What the data says</h2><img src="eda_inpatient.png" alt="Readmission rate rises with the number of inpatient visits in the prior year">
<p class="muted">Patients with no inpatient stay in the prior year were readmitted {pct(inp['0']['rate'])} of the time; with 4 or more, {pct(inp['4+']['rate'])}.</p></div>
<div><h2>&nbsp;</h2><img src="eda_discharge.png" alt="Readmission rate by discharge destination"></div></section>
<section><h2>Model comparison</h2><div class="scroll"><table><tr><th>Model</th><th>CV ROC-AUC (mean ± sd)</th><th>Test ROC-AUC</th><th>Test PR-AUC</th><th>Brier</th><th>Caught in top 10%</th></tr>{rows}</table></div>
<p class="muted">PR-AUC matters here because only {pct(base)} of patients are readmitted: an uninformed model scores {base:.3f}. Logistic regression's strength was picked by cross-validation (C = {m['lr_C']}).</p>
<div class="two" style="margin-top:16px"><img src="roc.png" alt="ROC curves for the baseline, logistic regression and gradient boosting"><img src="calibration.png" alt="Calibration: predicted risk against observed readmission rate by decile"></div></section>
<section class="two"><div><h2>Strongest signals</h2><p class="muted">Logistic regression odds ratios: per standard deviation for numbers, per category relative to the average category.</p>
<div class="scroll"><table><tr><th>Feature</th><th>Level</th><th>Odds ratio</th></tr>{eff}</table></div></div>
<div><h2>Does it work equally well for everyone?</h2><p class="muted">{html.escape(best)} on the test set, by group (groups under 300 patients skipped).</p>
<div class="scroll"><table><tr><th></th><th>Group</th><th>n</th><th>Observed</th><th>Predicted</th><th>ROC-AUC</th></tr>{sub_rows}</table></div></div></section>
<section id="calculator"><h2>Risk calculator</h2>
<p class="muted">A compact logistic regression on {len(cm['features'])} inputs (test ROC-AUC {cm['test']['roc_auc']:.3f}), exported from scikit-learn to JSON and scored in JavaScript; tests check it matches scikit-learn to 12 decimal places.</p>
<div class="two"><div id="calc-root"></div><div class="note"><b>Not for clinical use.</b> A demonstration on 1999–2008 US hospital records. It ranks risk modestly well and is not a diagnosis; real deployment would need recent local data, clinical review and prospective validation.</div></div></section>
<section><h2>How the data was cleaned</h2><div class="two"><div class="scroll"><table><tr><th>Step</th><th>Rows left</th></tr>{steps}</table></div>
<ul><li>'?' read as missing; lab results of 'None' kept as <i>not measured</i>, since whether a test was ordered is itself informative.</li>
<li>Expired and hospice discharges removed: those patients cannot be readmitted.</li>
<li>One encounter per patient (the first), so no patient sits in both training and test data.</li>
<li>ICD-9 diagnosis codes grouped into 9 disease groups; specialties capped at the top 10.</li>
<li>Dropped: weight (about 97% missing), payer code, identifiers{', and constant columns (' + ', '.join(m['dropped_constant']) + ')' if m['dropped_constant'] else ''}.</li></ul></div></section>
<section><h2>Limits</h2><p>Readmission is hard to predict from administrative records alone: the model ranks patients better than chance but misses many readmissions, and the data is from 1999–2008. Social factors, follow-up care and lab values beyond HbA1c and glucose aren't in the dataset.</p></section>
</main>
<footer class="w">Built by Shreyash Singh Salv · <a href="https://github.com/Shreyashsalv2" target="_blank" rel="noopener">github.com/Shreyashsalv2</a> · Data: Strack et al. (2014), UCI Machine Learning Repository, CC BY 4.0</footer>
<script src="calculator.js"></script>
<script>fetch("model.json").then(r => r.json()).then(m => buildCalculator(m, document.getElementById("calc-root")));</script>
</body></html>
"""
    (SITE / "index.html").write_text(page)
    print(SITE / "index.html")


if __name__ == "__main__":
    main()
