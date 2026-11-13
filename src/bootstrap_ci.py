"""Paired bootstrap CIs for the fixed-coverage table (no model, no B).

Reads the six scored JSONL files written by run_fixed_coverage.py, resamples the
1,000 questions with replacement (the same resample for every method, so the
comparison is paired), and reports

  * per-method 95% CI for selective accuracy and overall behaviour accuracy;
  * pairwise differences vs M1 (and M3) with 95% CI and a McNemar test on the
    overall decision correctness.

Usage (project root):  python src/bootstrap_ci.py
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
B = 4000
RNG = np.random.default_rng(42)

METHODS = {
    ("M1", 0.3): "m1_test_cov0.3_scored.jsonl",
    ("M1", 0.5): "m1_test_cov0.5_scored.jsonl",
    ("M3_k5", 0.3): "m3_k5_test_cov0.3_scored.jsonl",
    ("M3_k5", 0.5): "m3_k5_test_cov0.5_scored.jsonl",
    ("TwoStage", 0.3): "twostage_test_cov0.3_scored.jsonl",
    ("TwoStage", 0.5): "twostage_test_cov0.5_scored.jsonl",
}
LABEL = {"M1": "M1", "M3_k5": "M3(k=5)", "TwoStage": "two-stage"}


def load(name):
    rows = [json.loads(l) for l in (OUT / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])
    return {
        "id": [r["id"] for r in rows],
        "correct": np.array([bool(r["correct"]) for r in rows]),
        "answered": np.array([not r["abstain"] for r in rows]),
        "impossible": np.array([bool(r["is_impossible"]) for r in rows]),
    }


def sel_acc(correct, answered):
    n = answered.sum(axis=1)
    hit = (correct & answered).sum(axis=1)
    out = np.full(len(n), np.nan)
    ok = n > 0
    out[ok] = hit[ok] / n[ok]
    return out, n


data = {k: load(v) for k, v in METHODS.items()}
ids = data[("M1", 0.3)]["id"]
n = len(ids)
for k, d in data.items():
    assert d["id"] == ids, f"id order mismatch in {k}"

idx = RNG.integers(0, n, size=(B, n))
stats = {}
for key, d in data.items():
    c = d["correct"][idx]
    a = d["answered"][idx]
    s, cnt = sel_acc(c, a)
    stats[key] = {
        "sel": s,
        "beh": c.mean(axis=1),
        "n_answered": int(d["answered"].sum()),
        "coverage": float(d["answered"].mean()),
        "sel_point": float((d["correct"] & d["answered"]).sum() / max(1, d["answered"].sum())),
        "beh_point": float(d["correct"].mean()),
        "n": n,
    }


def ci(x):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return float(math.erfc(math.sqrt(chi2 / 2.0))), b, c


rows = []
for key, s in stats.items():
    lo, hi = ci(s["sel"])
    blo, bhi = ci(s["beh"])
    rows.append({
        "method": LABEL[key[0]], "target_coverage": key[1], "coverage": round(s["coverage"], 4),
        "n_answered": s["n_answered"],
        "selective_accuracy": round(s["sel_point"], 4),
        "sel_ci_low": round(lo, 4), "sel_ci_high": round(hi, 4),
        "behaviour_accuracy": round(s["beh_point"], 4),
        "beh_ci_low": round(blo, 4), "beh_ci_high": round(bhi, 4),
    })

pairs = [("M1", "M3_k5"), ("M1", "TwoStage"), ("M3_k5", "TwoStage")]
prows = []
for target in (0.3, 0.5):
    for a, b in pairs:
        da, db = data[(a, target)], data[(b, target)]
        if target == 0.3 and (a == "M3_k5" or b == "M3_k5"):
            prows.append({
                "pair": f"{LABEL[a]} vs {LABEL[b]}", "target_coverage": target,
                "cov_a": "", "cov_b": "", "d_selective": "", "d_sel_ci_low": "",
                "d_sel_ci_high": "", "d_behaviour": "", "d_beh_ci_low": "", "d_beh_ci_high": "",
                "mcnemar_p": "", "note": "M3 cannot reach 0.3 (min coverage 0.483) - not compared",
            })
            continue
        ds = stats[(a, target)]["sel"] - stats[(b, target)]["sel"]
        dbeh = stats[(a, target)]["beh"] - stats[(b, target)]["beh"]
        slo, shi = ci(ds)
        blo, bhi = ci(dbeh)
        p, nb, nc = mcnemar(da["correct"], db["correct"])
        prows.append({
            "pair": f"{LABEL[a]} vs {LABEL[b]}", "target_coverage": target,
            "cov_a": round(float(da["answered"].mean()), 4),
            "cov_b": round(float(db["answered"].mean()), 4),
            "d_selective": round(float(np.nanmean(ds)), 4),
            "d_sel_ci_low": round(slo, 4), "d_sel_ci_high": round(shi, 4),
            "d_behaviour": round(float(np.mean(dbeh)), 4),
            "d_beh_ci_low": round(blo, 4), "d_beh_ci_high": round(bhi, 4),
            "mcnemar_p": round(p, 4),
            "note": f"discordant {nb}/{nc}",
        })

with (OUT / "ci_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

with (OUT / "pairwise_ci.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prows[0]))
    w.writeheader()
    w.writerows(prows)

for r in rows:
    print(f"{r['method']:9s} cov={r['target_coverage']:.1f} n={r['n_answered']:4d} "
          f"sel={r['selective_accuracy']:.3f} [{r['sel_ci_low']:.3f},{r['sel_ci_high']:.3f}]  "
          f"beh={r['behaviour_accuracy']:.3f} [{r['beh_ci_low']:.3f},{r['beh_ci_high']:.3f}]")
print()
for r in prows:
    if r["d_selective"] == "":
        print(f"{r['pair']:22s} cov={r['target_coverage']:.1f}  {r['note']}")
        continue
    print(f"{r['pair']:22s} cov={r['target_coverage']:.1f}  "
          f"dSel={r['d_selective']:+.3f} [{r['d_sel_ci_low']:+.3f},{r['d_sel_ci_high']:+.3f}]  "
          f"dBeh={r['d_behaviour']:+.3f} [{r['d_beh_ci_low']:+.3f},{r['d_beh_ci_high']:+.3f}]  "
          f"p={r['mcnemar_p']:.3f}")
