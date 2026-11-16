import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import abstention_eval as ae  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results"

GOLD = {"val": DATA / "items" / "b_gold_val.jsonl", "test": DATA / "items" / "b_gold_test.jsonl"}
PREDS = {
    "M2-conf": {"kind": "confidence",
                "val": DATA / "outputs_b" / "slm_m2conf_val.jsonl",
                "test": DATA / "outputs_b" / "slm_m2conf_test.jsonl"},
    "M2-idk": {"kind": "refusal",
               "val": DATA / "outputs_b" / "slm_m2idk_val.jsonl",
               "test": DATA / "outputs_b" / "slm_m2idk_test.jsonl"},
    "M4": {"kind": "refusal",
           "val": DATA / "outputs_b" / "slm_m4_val.jsonl",
           "test": DATA / "outputs_b" / "slm_m4_test.jsonl"},
}
TARGETS = (0.3, 0.5)


def gold(split):
    return ae.load_jsonl(GOLD[split])


def build_confidence(gold_rows, pred_rows):
    by_id = ae.index_by_id(pred_rows)
    out = []
    for rec in gold_rows:
        p = by_id[rec["id"]]
        ans = p.get("answer", "")
        verbal = ae.is_abstention(ans)
        conf = float(p.get("confidence", 0.0))
        correct, em, f1 = ae.judge(rec, ans, verbal)
        out.append({
            "id": rec["id"], "is_impossible": rec["is_impossible"], "answer": ans,
            "abstain": verbal, "verbal_abstain": verbal,
            "confidence": conf, "prob": min(max(conf, 0.0), 1.0),
            "correct": correct, "forced_correct": ae.forced_correct(rec, ans),
            "em": em, "f1": f1, "n_samples": 1,
            "latency_s": p.get("time_s"),
        })
    return out


rows = []
base_preds = {
    "val": ae.m1_preds(gold("val"),
                       ae.index_by_id(ae.load_jsonl(DATA / "outputs_b" / "slm_m1_val.jsonl"))),
    "test": ae.m1_preds(gold("test"),
                        ae.index_by_id(ae.load_jsonl(DATA / "outputs_b" / "slm_m1_test.jsonl"))),
}
base_by_id = {s: ae.index_by_id(base_preds[s]) for s in base_preds}
conditional_rows = []

for name, cfg in PREDS.items():
    gold_val, gold_test = gold("val"), gold("test")
    pv = ae.load_jsonl(cfg["val"])
    pt = ae.load_jsonl(cfg["test"])
    if cfg["kind"] == "confidence":
        val_preds = build_confidence(gold_val, pv)
        test_preds = build_confidence(gold_test, pt)
        for target in TARGETS:
            tau, val_cov = ae.tune_at_coverage(val_preds, target)
            scored = ae.apply_threshold(test_preds, ae.index_by_id(gold_test), tau)
            m = ae.evaluate(scored)
            with (OUT / f"{name.lower().replace('-', '')}_test_cov{target}_scored.jsonl").open(
                    "w", encoding="utf-8") as fh:
                for rec in scored:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append({"method": name, "target_cov": target, "tau": round(tau, 4),
                         "val_cov": round(val_cov, 4), "test_cov": round(m["coverage"], 4),
                         "selective_acc": round(m["selective_accuracy"], 4),
                         "behaviour_acc": round(m["behaviour_accuracy"], 4),
                         "correct_refusal": round(m["correct_refusal_rate"], 4),
                         "over_refusal": round(m["over_refusal_rate"], 4),
                         "auroc": round(m["auroc"], 4) if m["auroc"] == m["auroc"] else "",
                         "avg_samples": 1.0, "note": ""})
    else:
        val_preds = ae.refusal_preds(gold_val, ae.index_by_id(pv))
        test_preds = ae.refusal_preds(gold_test, ae.index_by_id(pt))
        m = ae.evaluate(test_preds, base_preds=base_preds["test"])
        mv = ae.evaluate(val_preds, base_preds=base_preds["val"])
        with (OUT / f"{name.lower()}_test_scored.jsonl").open("w", encoding="utf-8") as fh:
            for rec in test_preds:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rows.append({"method": name, "target_cov": "natural", "tau": "",
                     "val_cov": round(mv["coverage"], 4), "test_cov": round(m["coverage"], 4),
                     "selective_acc": round(m["selective_accuracy"], 4),
                     "behaviour_acc": round(m["behaviour_accuracy"], 4),
                     "correct_refusal": round(m["correct_refusal_rate"], 4),
                     "over_refusal": round(m["over_refusal_rate"], 4),
                     "auroc": "",
                     "avg_samples": 1.0,
                     "note": "refuses by itself (no threshold); AUROC not reported"})
        for split, preds, base in (("test", test_preds, base_by_id["test"]),):
            right = [0, 0]
            wrong = [0, 0]
            for p in preds:
                if p["is_impossible"]:
                    continue
                target = right if base[p["id"]].get("forced_correct") else wrong
                target[0] += int(p["abstain"])
                target[1] += 1
            conditional_rows.append({
                "method": name, "split": split,
                "base_right_refused": right[0], "base_right_total": right[1],
                "base_right_rate": round(right[0] / right[1], 4) if right[1] else "",
                "base_wrong_refused": wrong[0], "base_wrong_total": wrong[1],
                "base_wrong_rate": round(wrong[0] / wrong[1], 4) if wrong[1] else "",
            })

OUT.joinpath("conditional_refusal.csv").write_text(
    "method,split,base_right_refused,base_right_total,base_right_rate,"
    "base_wrong_refused,base_wrong_total,base_wrong_rate\n"
    + "\n".join(",".join(str(r[k]) for k in ["method", "split", "base_right_refused",
                                            "base_right_total", "base_right_rate",
                                            "base_wrong_refused", "base_wrong_total",
                                            "base_wrong_rate"]) for r in conditional_rows)
    + "\n", encoding="utf-8-sig")

OUT.joinpath("m2_m4_table.csv").write_text(
    "method,target_cov,tau,val_cov,test_cov,selective_acc,behaviour_acc,"
    "correct_refusal,over_refusal,auroc,avg_samples,note\n"
    + "\n".join(",".join(str(r[k]) for k in ["method", "target_cov", "tau", "val_cov",
                                            "test_cov", "selective_acc", "behaviour_acc",
                                            "correct_refusal", "over_refusal", "auroc",
                                            "avg_samples", "note"]) for r in rows)
    + "\n", encoding="utf-8-sig")

for r in rows:
    print(f"{r['method']:8s} target={str(r['target_cov']):7s} test_cov={r['test_cov']:.3f} "
          f"sel={r['selective_acc']:.3f} beh={r['behaviour_acc']:.3f} "
          f"refuse_ok={r['correct_refusal']:.3f} over={r['over_refusal']:.3f}")

# M2-conf distinct confidence values.
conf_vals = sorted({round(float(p.get("confidence", -1)), 2)
                    for p in ae.load_jsonl(PREDS["M2-conf"]["val"])})
print(f"\nM2-conf distinct confidence values on val: {conf_vals[:12]} ... ({len(conf_vals)} total)")
m4 = ae.load_jsonl(PREDS["M4"]["test"])
refused = [p for p in m4 if ae.is_abstention(p.get("answer", ""))]
print(f"M4 test: {len(refused)}/{len(m4)} outputs are refusals "
      f"({len(refused)/len(m4):.1%})")
