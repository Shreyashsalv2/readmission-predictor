"""Export a fitted logistic-regression pipeline to JSON so a browser can score it.

The JSON holds the scaler's means and scales, the one-hot categories and every coefficient.
`score()` is the reference scorer; site/calculator.js implements the same arithmetic and
tests check that both match scikit-learn's predict_proba.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.pipeline import Pipeline


def to_json(pipe: Pipeline, numeric: list[str], categorical: list[str], meta: dict | None = None) -> dict:
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    scaler = pre.named_transformers_["num"]
    onehot = pre.named_transformers_["cat"]
    coef = clf.coef_.ravel()
    n_num = len(numeric)
    model = {
        "intercept": float(clf.intercept_[0]),
        "numeric": [{"name": c, "mean": float(m), "scale": float(s), "coef": float(w)}
                    for c, m, s, w in zip(numeric, scaler.mean_, scaler.scale_, coef[:n_num])],
        "categorical": [],
        "meta": meta or {},
    }
    pos = n_num
    for c, cats in zip(categorical, onehot.categories_):
        k = len(cats)
        model["categorical"].append({"name": c, "categories": [str(x) for x in cats],
                                     "coefs": [float(w) for w in coef[pos:pos + k]]})
        pos += k
    assert pos == len(coef), "coefficient count mismatch"
    return model


def score(model: dict, record: dict) -> float:
    """Probability of 30-day readmission for one patient record (dict of raw feature values)."""
    z = model["intercept"]
    for f in model["numeric"]:
        z += f["coef"] * (float(record[f["name"]]) - f["mean"]) / f["scale"]
    for f in model["categorical"]:
        v = str(record[f["name"]])
        if v in f["categories"]:                     # unseen value -> all zeros, like handle_unknown="ignore"
            z += f["coefs"][f["categories"].index(v)]
    return 1.0 / (1.0 + math.exp(-z))


def top_effects(model: dict, k: int = 12) -> list[dict]:
    """Largest effects as odds ratios: per standard deviation for numbers, per category vs average."""
    rows = []
    for f in model["numeric"]:
        rows.append({"feature": f["name"], "level": "+1 SD (" + f"{f['scale']:.1f}" + ")",
                     "odds_ratio": math.exp(f["coef"])})
    for f in model["categorical"]:
        c = np.array(f["coefs"])
        for cat, w in zip(f["categories"], c - c.mean()):
            rows.append({"feature": f["name"], "level": cat, "odds_ratio": math.exp(float(w))})
    rows.sort(key=lambda r: -abs(math.log(r["odds_ratio"])))
    return rows[:k]
