import csv
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


gold = {s: load(ITEMS / f"b_gold_{s}.jsonl") for s in ("val", "test")}
gold_by_id = {s: ae.index_by_id(gold[s]) for s in gold}
m1 = {s: ae.m1_preds(gold[s], ae.index_by_id(load(DATA / f"slm_m1_{s}.jsonl")))
      for s in gold}
m3_raw = {s: ae.index_by_id(load(DATA / f"slm_m3_{s}.jsonl")) for s in gold}
m3_k5 = {s: ae.m3_preds(gold[s], m3_raw[s], k=5) for s in gold}

# fixed: keep the k=5 configuration and vary k.
# retuned: select the configuration for each k.
fixed_cfg = {t: ae.tune_two_stage_at_coverage(
    m1["val"], m3_k5["val"], gold_by_id["val"], t)[0] for t in TARGETS}

rows = []


def add(method, k, target, protocol, metrics, samples, stage2=0.0, cfg=None, tau_c=None):
    rows.append({
        "method": method, "k": "" if k is None else k, "target": target,
        "protocol": protocol,
        "coverage": round(metrics["coverage"], 4),
        "selective": round(metrics["selective_accuracy"], 4),
        "behaviour": round(metrics["behaviour_accuracy"], 4),
        "samples": round(float(samples), 2),
        "stage2": round(float(stage2), 4),
        "tau_low": "" if cfg is None else round(cfg[0], 4),
        "tau_high": "" if cfg is None else round(cfg[1], 4),
        "tau_c": "" if tau_c is None else round(float(tau_c), 4),
    })


for target in TARGETS:
    tau1, _ = ae.tune_at_coverage(m1["val"], target)
    m1t = ae.evaluate(ae.apply_threshold(m1["test"], gold_by_id["test"], tau1))
    add("M1", None, target, "", m1t, 1.0)

    for k in KS:
        m3 = {s: ae.m3_preds(gold[s], m3_raw[s], k=k) for s in gold}
        tau_c, _ = ae.tune_at_coverage(m3["val"], target)
        m3t = ae.evaluate(ae.apply_threshold(m3["test"], gold_by_id["test"], tau_c))
        add("M3", k, target, "", m3t, k, tau_c=tau_c)

        # retuned: one config per k.
        cfg, _, _ = ae.tune_two_stage_at_coverage(
            m1["val"], m3["val"], gold_by_id["val"], target)
        ts = ae.simulate_two_stage(m1["test"], m3["test"], gold_by_id["test"], *cfg)
        tsm = ae.evaluate(ts)
        stage2 = float(np.mean([p["stage"] == "borderline" for p in ts]))
        add("two-stage", k, target, "retuned", tsm, tsm["avg_samples"], stage2, cfg,
            tau_c=cfg[2])
        print(f"k={k} target={target} retuned: cov={tsm['coverage']:.3f} "
              f"sel={tsm['selective_accuracy']:.3f} beh={tsm['behaviour_accuracy']:.3f} "
              f"samples={tsm['avg_samples']:.2f} stage2={stage2:.1%} "
              f"(tau_low={cfg[0]:.3f} tau_high={cfg[1]:.3f} tau_c={cfg[2]:.1f})")

        # fixed: k=5 configuration, only k changes.
        fcfg = fixed_cfg[target]
        tsf = ae.simulate_two_stage(m1["test"], m3["test"], gold_by_id["test"], *fcfg)
        tsfm = ae.evaluate(tsf)
        stage2_f = float(np.mean([p["stage"] == "borderline" for p in tsf]))
        add("two-stage", k, target, "fixed", tsfm, tsfm["avg_samples"], stage2_f, fcfg,
            tau_c=fcfg[2])
        print(f"k={k} target={target} fixed:   cov={tsfm['coverage']:.3f} "
              f"sel={tsfm['selective_accuracy']:.3f} beh={tsfm['behaviour_accuracy']:.3f} "
              f"samples={tsfm['avg_samples']:.2f} stage2={stage2_f:.1%}")

with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "k", "target", "protocol", "coverage",
                                      "selective", "behaviour", "samples", "stage2",
                                      "tau_low", "tau_high", "tau_c"])
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT)

main_path = ROOT / "results" / "main_table_fixed_coverage.csv"
if main_path.exists():
    with main_path.open(encoding="utf-8-sig", newline="") as f:
        main_rows = list(csv.DictReader(f))
    for target in TARGETS:
        main = next(r for r in main_rows
                    if r["method"] == "TwoStage"
                    and abs(float(r["target_coverage"]) - target) < 1e-9)
        for protocol in ("retuned", "fixed"):
            abl = next(r for r in rows
                       if r["method"] == "two-stage" and r["k"] == 5
                       and r["protocol"] == protocol
                       and abs(float(r["target"]) - target) < 1e-9)
            for key_main, key_abl in (("test_coverage", "coverage"),
                                      ("selective_accuracy", "selective"),
                                      ("behaviour_accuracy", "behaviour"),
                                      ("avg_samples", "samples")):
                a, b = float(main[key_main]), float(abl[key_abl])
                if abs(a - b) > 5e-4:
                    raise SystemExit(
                        f"main table vs ablation mismatch at target {target} "
                        f"({protocol}): {key_main}={a} vs {key_abl}={b}")
    print("check: k=5 two-stage rows (retuned and fixed) match "
          "main_table_fixed_coverage.csv")
