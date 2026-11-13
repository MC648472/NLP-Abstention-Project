"""Fixed-coverage comparison table for M1 / M3 / two-stage (student A).

Protocol (same one A asked B to follow):
  * the threshold is chosen on **val only**, as the one whose val coverage is
    closest to the target (0.3 / 0.5);
  * test is scored once with that threshold; the achieved test coverage is
    reported, and `sel_acc_interp_at_target` linearly interpolates the test
    risk-coverage curve to the exact target;
  * metrics/tags/ids are identical to `abstention_eval.py`, i.e. the scorer B
    has, so both sides produce the same numbers from the same JSONL.

Usage (from the project root):
    python src/run_fixed_coverage.py
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import abstention_eval as ae  # noqa: E402

GOLD_FILES = {"val": ROOT / "data" / "items" / "b_gold_val.jsonl",
              "test": ROOT / "data" / "items" / "b_gold_test.jsonl"}
PRED_FILES = {
    "M1": {"mode": "m1",
           "val": ROOT / "data" / "outputs_b" / "slm_m1_val.jsonl",
           "test": ROOT / "data" / "outputs_b" / "slm_m1_test.jsonl"},
    "M3_k5": {"mode": "m3", "k": 5,
              "val": ROOT / "data" / "outputs_b" / "slm_m3_val.jsonl",
              "test": ROOT / "data" / "outputs_b" / "slm_m3_test.jsonl"},
}
TARGETS = (0.3, 0.5)
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

_gold = {}
_preds = {}


def gold(split):
    if split not in _gold:
        _gold[split] = ae.load_jsonl(GOLD_FILES[split])
    return _gold[split]


def gold_by_id(split):
    return ae.index_by_id(gold(split))


def method_preds(name, split):
    key = (name, split)
    if key not in _preds:
        cfg = PRED_FILES[name]
        raw = ae.index_by_id(ae.load_jsonl(cfg[split]))
        if cfg["mode"] == "m1":
            _preds[key] = ae.m1_preds(gold(split), raw)
        else:
            _preds[key] = ae.m3_preds(gold(split), raw, k=cfg["k"])
    return _preds[key]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fnum(x, nd=4):
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    if math.isnan(x):
        return ""
    return round(x, nd)


rows = []
config = {}

for target in TARGETS:
    # --- threshold methods ---------------------------------------------------
    for name in ("M1", "M3_k5"):
        val_preds = method_preds(name, "val")
        tau, val_cov = ae.tune_at_coverage(val_preds, target)
        test_preds = method_preds(name, "test")
        scored = ae.apply_threshold(test_preds, gold_by_id("test"), tau)
        m = ae.evaluate(scored)
        unreachable = abs(val_cov - target) > 0.05
        note = ""
        if unreachable:
            note = f"target unreachable: min val coverage {val_cov:.3f} at tau={tau:.3f}"
        rows.append({
            "method": name, "target_coverage": target, "val_tau": fnum(tau),
            "val_coverage": fnum(val_cov), "test_coverage": fnum(m["coverage"]),
            "n_answered": int(round(m["coverage"] * m["n"])),
            "selective_accuracy": fnum(m["selective_accuracy"]),
            "sel_acc_interp_at_target": fnum(ae.interp_selective_accuracy(test_preds, target)),
            "behaviour_accuracy": fnum(m["behaviour_accuracy"]),
            "correct_refusal_rate": fnum(m["correct_refusal_rate"]),
            "over_refusal_rate": fnum(m["over_refusal_rate"]),
            "auroc": fnum(m["auroc"]), "auprc": fnum(m["auprc"]),
            "avg_samples": fnum(m["avg_samples"], 2),
            "avg_latency_s": fnum(m["avg_latency_s"], 4),
            "note": note,
        })
        config.setdefault(name, {})[str(target)] = {
            "tau": tau, "val_coverage": val_cov, "rule": "answered iff confidence >= tau",
            "unreachable": unreachable,
        }
        write_jsonl(OUT / f"{name.lower()}_test_cov{target}_scored.jsonl", scored)

    # --- two-stage -----------------------------------------------------------
    m1_val, m3_val = method_preds("M1", "val"), method_preds("M3_k5", "val")
    m1_test, m3_test = method_preds("M1", "test"), method_preds("M3_k5", "test")
    cfg, val_acc, val_cov = ae.tune_two_stage_at_coverage(
        m1_val, m3_val, gold_by_id("val"), target,
        quantiles=[i / 20 for i in range(1, 20)],
    )
    ts = ae.simulate_two_stage(m1_test, m3_test, gold_by_id("test"), *cfg)
    m = ae.evaluate(ts)
    stage2 = float(np.mean([p["stage"] == "borderline" for p in ts]))
    rows.append({
        "method": "TwoStage", "target_coverage": target, "val_tau": fnum(cfg[0]),
        "val_coverage": fnum(val_cov), "test_coverage": fnum(m["coverage"]),
        "n_answered": int(round(m["coverage"] * m["n"])),
        "selective_accuracy": fnum(m["selective_accuracy"]),
        "sel_acc_interp_at_target": fnum(ae.interp_selective_accuracy(ts, target)),
        "behaviour_accuracy": fnum(m["behaviour_accuracy"]),
        "correct_refusal_rate": fnum(m["correct_refusal_rate"]),
        "over_refusal_rate": fnum(m["over_refusal_rate"]),
        # confidence mixes logprob (stage 1) and consistency (stage 2) -> AUROC invalid
        "auroc": "", "auprc": "",
        "avg_samples": fnum(m["avg_samples"], 2),
        "avg_latency_s": fnum(m["avg_latency_s"], 4),
        "note": f"tau_low={cfg[0]:.3f} tau_high={cfg[1]:.3f} tau_c={cfg[2]:.1f}; "
                f"stage2={stage2:.2f}; AUROC n/a (mixed scores)",
    })
    config.setdefault("TwoStage", {})[str(target)] = {
        "tau_low": cfg[0], "tau_high": cfg[1], "tau_c": cfg[2],
        "val_accuracy": val_acc, "val_coverage": val_cov,
    }
    write_jsonl(OUT / f"twostage_test_cov{target}_scored.jsonl", ts)

OUT.mkdir(parents=True, exist_ok=True)
cols = ["method", "target_coverage", "val_tau", "val_coverage", "test_coverage",
        "n_answered", "selective_accuracy", "sel_acc_interp_at_target",
        "behaviour_accuracy", "correct_refusal_rate", "over_refusal_rate",
        "auroc", "auprc", "avg_samples", "avg_latency_s", "note"]
with (OUT / "main_table_fixed_coverage.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

# --- curves + baselines ------------------------------------------------------
curves = {}
for name in ("M1", "M3_k5"):
    curves[name] = ae.risk_coverage_curve(method_preds(name, "test"))
test_gold = gold("test")
curves["baselines"] = {
    "abstain_all": {"coverage": 0.0,
                    "behaviour_accuracy": float(np.mean([1 if r["is_impossible"] else 0 for r in test_gold])),
                    "selective_accuracy": None},
}
for name in ("M1", "M3_k5"):
    preds = method_preds(name, "test")
    curves["baselines"][f"{name}_answer_all"] = {
        "coverage": 1.0,
        "behaviour_accuracy": float(np.mean([1 if p["forced_correct"] else 0 for p in preds])),
        "selective_accuracy": float(np.mean([1 if p["forced_correct"] else 0 for p in preds])),
    }

with (OUT / "coverage_accuracy_curves.json").open("w", encoding="utf-8") as f:
    json.dump(curves, f, ensure_ascii=False, indent=2)

with (OUT / "fixed_coverage_config.json").open("w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"wrote {OUT}")
for r in rows:
    print(f"{r['method']:8s} target={r['target_coverage']:.1f} "
          f"test_cov={r['test_coverage']:.3f} sel_acc={r['selective_accuracy']:.3f} "
          f"behav={r['behaviour_accuracy']:.3f} samples={r['avg_samples']:.2f} {r['note']}")
