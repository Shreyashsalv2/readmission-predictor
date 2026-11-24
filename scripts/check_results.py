"""Fail if a fresh run's headline metrics differ from the committed results (reproducibility check)."""
import json
import sys
from pathlib import Path

committed, fresh = Path(sys.argv[1]), Path(sys.argv[2])
if not committed.exists():
    print("no committed results yet; skipping")
    sys.exit(0)
a, b = json.loads(committed.read_text()), json.loads(fresh.read_text())
bad = []
if a["train_rows"] != b["train_rows"] or a["test_rows"] != b["test_rows"]:
    bad.append(f"rows differ: {a['train_rows']}/{a['test_rows']} vs {b['train_rows']}/{b['test_rows']}")
for name in a["models"]:
    for k in ("roc_auc", "pr_auc", "brier"):
        x, y = a["models"][name]["test"][k], b["models"][name]["test"][k]
        if abs(x - y) > 0.003:
            bad.append(f"{name} {k}: committed {x:.4f}, fresh {y:.4f}")
print("\n".join(bad) if bad else "fresh run matches committed results")
sys.exit(1 if bad else 0)
