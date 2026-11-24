"""Metrics for a rare binary outcome: ranking, calibration, and what a ward would actually see."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "base_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
    }


def top_fraction(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> dict:
    """Flag the riskiest `frac` of patients: how many readmissions does that catch?"""
    y = np.asarray(y)
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-np.asarray(p), kind="stable")[:k]
    caught = y[idx].sum()
    precision = caught / k
    return {"fraction_flagged": frac, "flagged": int(k), "precision": float(precision),
            "recall": float(caught / max(1, y.sum())), "lift": float(precision / y.mean())}


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for ROC-AUC."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def by_group(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, col: str, min_n: int = 300) -> list[dict]:
    """ROC-AUC and calibration per subgroup (groups smaller than min_n are skipped)."""
    out = []
    for g, idx in df.groupby(col).indices.items():
        yy, pp = np.asarray(y)[idx], np.asarray(p)[idx]
        if len(idx) < min_n or yy.min() == yy.max():
            continue
        out.append({"group": str(g), "n": int(len(idx)), "readmit_rate": float(yy.mean()),
                    "mean_predicted": float(pp.mean()), "roc_auc": float(roc_auc_score(yy, pp))})
    return sorted(out, key=lambda r: -r["n"])


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return {"mean_predicted": mean_pred.tolist(), "observed": frac_pos.tolist()}
