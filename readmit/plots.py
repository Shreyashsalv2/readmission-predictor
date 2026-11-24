"""Result plots (matplotlib, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

COLORS = ["#9aa3ad", "#2a6fdb", "#d9822b"]


def roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], aucs: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    for (name, (y, p)), col in zip(curves.items(), COLORS):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#bbb", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (readmissions caught)")
    ax.set_title("ROC curves on the held-out test set", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration(cal: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    hi = 0.0
    for (name, c), col in zip(cal.items(), COLORS[1:]):
        ax.plot(c["mean_predicted"], c["observed"], "o-", color=col, lw=2, ms=4, label=name)
        hi = max(hi, max(c["mean_predicted"]), max(c["observed"]))
    hi = min(1.0, hi * 1.1)
    ax.plot([0, hi], [0, hi], ls=":", color="#bbb", lw=1, label="Perfect calibration")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Predicted readmission risk (decile mean)")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Calibration on the held-out test set", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rate_bars(labels: list[str], rates: list[float], counts: list[int], title: str, xlabel: str,
              base_rate: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    bars = ax.bar(labels, [r * 100 for r in rates], color="#2a6fdb", width=0.65)
    for b, n in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"n={n:,}", ha="center", fontsize=7, color="#555")
    ax.axhline(base_rate * 100, ls=":", color="#d9822b", lw=1.2, label=f"overall {base_rate:.1%}")
    ax.set_ylabel("Readmitted within 30 days (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rate_barh(labels: list[str], rates: list[float], counts: list[int], title: str,
              base_rate: float, path: Path) -> None:
    """Horizontal version for long category names (sorted top to bottom as given)."""
    fig, ax = plt.subplots(figsize=(5.6, 0.5 * len(labels) + 1.3), dpi=150)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, [r * 100 for r in rates], color="#2a6fdb", height=0.6)
    for yy, r, n in zip(y, rates, counts):
        ax.text(r * 100 + 0.3, yy, f"{r:.1%}  (n={n:,})", va="center", fontsize=7.5, color="#444")
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(base_rate * 100, ls=":", color="#d9822b", lw=1.2, label=f"overall {base_rate:.1%}")
    ax.set_xlabel("Readmitted within 30 days (%)")
    ax.set_xlim(0, max(rates) * 100 * 1.45)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
