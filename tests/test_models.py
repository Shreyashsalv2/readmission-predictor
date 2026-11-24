import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from readmit import data as D
from readmit import evaluate as E
from readmit import export, models

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cleaned(raw_csv):
    return D.clean(D.load_raw(raw_csv))


@pytest.fixture(scope="module")
def compact(cleaned):
    df = cleaned.df
    pipe = models.logistic(D.COMPACT_NUMERIC, D.COMPACT_CATEGORICAL, C=0.5).fit(df, df["target"])
    return pipe, export.to_json(pipe, D.COMPACT_NUMERIC, D.COMPACT_CATEGORICAL, {"base_rate": float(df["target"].mean())})


def test_models_beat_baseline(cleaned):
    df = cleaned.df
    n = len(df) * 3 // 4
    tr, te = df.iloc[:n], df.iloc[n:]
    aucs = {}
    for name, pipe in models.all_models(cleaned.numeric, cleaned.categorical).items():
        p = pipe.fit(tr, tr["target"]).predict_proba(te)[:, 1]
        aucs[name] = roc_auc_score(te["target"], p)
    assert aucs["Baseline (always the average rate)"] == pytest.approx(0.5)
    assert aucs["Logistic regression"] > 0.6        # synthetic target depends on prior inpatient visits
    assert aucs["Gradient boosting"] > 0.6


def test_exported_json_matches_sklearn(cleaned, compact):
    pipe, mj = compact
    rows = cleaned.df.iloc[:200]
    want = pipe.predict_proba(rows)[:, 1]
    got = [export.score(mj, r) for r in rows[D.COMPACT_NUMERIC + D.COMPACT_CATEGORICAL].to_dict("records")]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_unseen_category_is_ignored(cleaned, compact):
    pipe, mj = compact
    row = cleaned.df.iloc[[0]].copy()
    row["diag_1_group"] = "Never seen"
    want = pipe.predict_proba(row)[0, 1]
    got = export.score(mj, row[D.COMPACT_NUMERIC + D.COMPACT_CATEGORICAL].iloc[0].to_dict())
    assert got == pytest.approx(want, abs=1e-12)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_calculator_matches_sklearn(cleaned, compact, tmp_path):
    pipe, mj = compact
    rows = cleaned.df.iloc[:100]
    recs = json.loads(rows[D.COMPACT_NUMERIC + D.COMPACT_CATEGORICAL].to_json(orient="records"))
    (tmp_path / "m.json").write_text(json.dumps(mj))
    (tmp_path / "r.json").write_text(json.dumps(recs))
    js = (f"const {{score}}=require({json.dumps(str(ROOT / 'site' / 'calculator.js'))});"
          f"const m=require({json.dumps(str(tmp_path / 'm.json'))});const r=require({json.dumps(str(tmp_path / 'r.json'))});"
          "console.log(JSON.stringify(r.map(x=>score(m,x))));")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    np.testing.assert_allclose(json.loads(out), pipe.predict_proba(rows)[:, 1], rtol=0, atol=1e-12)


def test_top_fraction():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .05])
    r = E.top_fraction(y, p, 0.3)
    assert r["flagged"] == 3 and r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["lift"] == pytest.approx((2 / 3) / 0.3)


def test_scores_and_bootstrap():
    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.1
    p = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.1, 2000), 0.01, 0.99)
    s = E.scores(y, p)
    assert 0.5 < s["roc_auc"] <= 1 and 0 < s["brier"] < 0.25 and s["base_rate"] == pytest.approx(y.mean())
    lo, hi = E.bootstrap_auc(y, p, n=200)
    assert lo < s["roc_auc"] < hi


def test_top_effects_are_sorted(compact):
    _, mj = compact
    eff = export.top_effects(mj, 5)
    mags = [abs(np.log(e["odds_ratio"])) for e in eff]
    assert mags == sorted(mags, reverse=True)
