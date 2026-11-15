"""Ablation for the two-stage abstainer (A delivers, B reports).

Varies the Stage-2 sample count k in {3, 5, 8} and compares, at fixed coverage
(0.3 / 0.5), three cost-quality points:

  M1           1 sequence / question
  M3 @ k       k sequences / question
  two-stage    1 + k * p sequences / question, p = fraction sent to Stage 2

Everything runs on the cached M1 / M3 JSONL (the M3 file stores 8 samples, so
k = 3 / 5 / 8 are scored from the same generations; no model calls).

Output: results/two_stage_ablation.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import abstention_eval as ae  # noqa: E402

DATA = ROOT / "data" / "outputs_b"
ITEMS = ROOT / "data" / "items"
OUT = ROOT / "results" / "two_stage_ablation.csv"
KS = (3, 5, 8)
TARGETS = (0.3, 0.5)


def load(p):
    return ae.load_jsonl(p)


gold = {"val": load(ITEMS / "b_gold_val.jsonl"), "test": load(ITEMS / "b_gold_test.jsonl")}
gold_by_id = {s: ae.index_by_id(gold[s]) for s in gold}
m1 = {s: ae.m1_preds(gold[s], ae.index_by_id(load(DATA / f"slm_m1_{s}.jsonl")))
      for s in gold}
m3_raw = {s: ae.index_by_id(load(DATA / f"slm_m3_{s}.jsonl")) for s in gold}

rows = []
for k in KS:
    m3 = {s: ae.m3_preds(gold[s], m3_raw[s], k=k) for s in gold}
    for target in TARGETS:
        # --- pure single pass -------------------------------------------
        tau1, _ = ae.tune_at_coverage(m1["val"], target)
        m1t = ae.evaluate(ae.apply_threshold(m1["test"], gold_by_id["test"], tau1))
        rows.append({"method": "M1", "k": "", "target": target,
                     "coverage": round(m1t["coverage"], 4),
                     "selective": round(m1t["selective_accuracy"], 4),
                     "behaviour": round(m1t["behaviour_accuracy"], 4),
                     "samples": 1.0, "stage2": 0.0,
                     "tau_low": "", "tau_high": "", "tau_c": ""})

        # --- pure multi-sample consistency ------------------------------
        tau_c, _ = ae.tune_at_coverage(m3["val"], target)
        m3t = ae.evaluate(ae.apply_threshold(m3["test"], gold_by_id["test"], tau_c))
        rows.append({"method": "M3", "k": k, "target": target,
                     "coverage": round(m3t["coverage"], 4),
                     "selective": round(m3t["selective_accuracy"], 4),
                     "behaviour": round(m3t["behaviour_accuracy"], 4),
                     "samples": float(k), "stage2": 0.0,
                     "tau_low": "", "tau_high": "", "tau_c": round(tau_c, 4)})

        # --- two-stage --------------------------------------------------
        cfg, _, _ = ae.tune_two_stage_at_coverage(
            m1["val"], m3["val"], gold_by_id["val"], target)
        ts = ae.simulate_two_stage(m1["test"], m3["test"], gold_by_id["test"], *cfg)
        tsm = ae.evaluate(ts)
        stage2 = float(np.mean([p["stage"] == "borderline" for p in ts]))
        rows.append({"method": "two-stage", "k": k, "target": target,
                     "coverage": round(tsm["coverage"], 4),
                     "selective": round(tsm["selective_accuracy"], 4),
                     "behaviour": round(tsm["behaviour_accuracy"], 4),
                     "samples": round(float(np.mean([p["n_samples"] for p in ts])), 2),
                     "stage2": round(stage2, 4),
                     "tau_low": round(cfg[0], 4),
                     "tau_high": round(cfg[1], 4),
                     "tau_c": cfg[2]})
        print(f"k={k} target={target}: two-stage cov={tsm['coverage']:.3f} "
              f"sel={tsm['selective_accuracy']:.3f} beh={tsm['behaviour_accuracy']:.3f} "
              f"samples={rows[-1]['samples']:.2f} stage2={stage2:.2f} "
              f"(tau_low={cfg[0]:.3f} tau_high={cfg[1]:.3f} tau_c={cfg[2]:.1f})")

with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "k", "target", "coverage",
                                      "selective", "behaviour", "samples", "stage2",
                                      "tau_low", "tau_high", "tau_c"])
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT)

# --- guard against the exact drift this script once had --------------------
# The k=5 two-stage row MUST equal the main-table row.  Before this check the
# main table used a 5% threshold grid while the ablation used deciles, which
# produced two different "k=5" results for the same method.
main_path = ROOT / "results" / "main_table_fixed_coverage.csv"
if main_path.exists():
    with main_path.open(encoding="utf-8-sig", newline="") as f:
        main_rows = list(csv.DictReader(f))
    for target in TARGETS:
        main = next(r for r in main_rows
                    if r["method"] == "TwoStage"
                    and abs(float(r["target_coverage"]) - target) < 1e-9)
        abl = next(r for r in rows
                   if r["method"] == "two-stage" and r["k"] == 5
                   and abs(float(r["target"]) - target) < 1e-9)
        for key_main, key_abl in (("test_coverage", "coverage"),
                                  ("selective_accuracy", "selective"),
                                  ("behaviour_accuracy", "behaviour"),
                                  ("avg_samples", "samples")):
            a, b = float(main[key_main]), float(abl[key_abl])
            if abs(a - b) > 5e-4:
                raise SystemExit(
                    f"main table vs ablation mismatch at target {target}: "
                    f"{key_main}={a} vs {key_abl}={b}")
    print("check: k=5 two-stage rows match main_table_fixed_coverage.csv")
