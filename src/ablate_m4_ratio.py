import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import abstention_eval as ae  # noqa: E402

GOLD = ROOT / "data" / "items" / "b_gold_test.jsonl"
OUT = ROOT / "results" / "m4_ratio_ablation.csv"
OUT_JSON = ROOT / "results" / "m4_ratio_ablation.json"
B = 4000
SEED = 42
RUNS = [("0.3", "slm_m4r30_test.jsonl"), ("0.5", "slm_m4_test.jsonl"),
        ("0.7", "slm_m4r70_test.jsonl")]

gold = ae.load_jsonl(GOLD)
gold_by_id = ae.index_by_id(gold)
rows = []
detail = []

# Shared resample index for all ratios.
idx = np.random.default_rng(SEED).integers(0, len(gold), size=(B, len(gold)))

for ratio, filename in RUNS:
    preds = ae.refusal_preds(gold, ae.index_by_id(
        ae.load_jsonl(ROOT / "data" / "outputs_b" / filename)))
    rows_sorted = sorted(preds, key=lambda p: p["id"])
    m = ae.evaluate(rows_sorted)
    correct = np.array([bool(p["correct"]) for p in rows_sorted])
    answered = np.array([not p["abstain"] for p in rows_sorted])
    c = correct[idx]
    a = answered[idx]
    n = a.sum(axis=1)
    hit = (c & a).sum(axis=1)
    sel = np.where(n > 0, hit / np.maximum(n, 1), np.nan)
    beh = c.mean(axis=1)
    sel_lo, sel_hi = np.nanpercentile(sel, [2.5, 97.5])
    beh_lo, beh_hi = np.percentile(beh, [2.5, 97.5])
    row = {
        "ratio": ratio,
        "adapter": "r" + str(int(round(float(ratio) * 100))),
        "coverage": round(m["coverage"], 4),
        "selective_point": round(m["selective_accuracy"], 4),
        "selective_boot_mean": round(float(np.nanmean(sel)), 4),
        "selective_ci_low": round(float(sel_lo), 4),
        "selective_ci_high": round(float(sel_hi), 4),
        "overall_point": round(m["behaviour_accuracy"], 4),
        "overall_boot_mean": round(float(np.mean(beh)), 4),
        "overall_ci_low": round(float(beh_lo), 4),
        "overall_ci_high": round(float(beh_hi), 4),
        "correct_refusal": round(m["correct_refusal_rate"], 4),
        "over_refusal": round(m["over_refusal_rate"], 4),
    }
    rows.append(row)
    detail.append({**row, "bootstrap": {"B": B, "seed": SEED}})
    print(f"rho={ratio}: cov={row['coverage']:.3f} sel={row['selective_point']:.4f} "
          f"(boot {row['selective_boot_mean']:.4f}) overall={row['overall_point']:.4f} "
          f"(boot {row['overall_boot_mean']:.4f})")

with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
OUT_JSON.write_text(json.dumps(detail, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
print("wrote", OUT_JSON)
