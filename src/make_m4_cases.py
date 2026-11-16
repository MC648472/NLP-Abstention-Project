import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import abstention_eval as ae  # noqa: E402

GOLD = ROOT / "data" / "items" / "b_gold_test.jsonl"
M4 = ROOT / "data" / "outputs_b" / "slm_m4_test.jsonl"
M1 = ROOT / "data" / "outputs_b" / "slm_m1_test.jsonl"
OUT = ROOT / "results" / "m4_cases.json"
LIMIT = 10

gold = ae.load_jsonl(GOLD)
gold_by_id = ae.index_by_id(gold)
base = ae.index_by_id(ae.load_jsonl(M1))
m4 = ae.refusal_preds(gold, ae.index_by_id(ae.load_jsonl(M4)))

over, missed = [], []
for p in m4:
    g = gold_by_id[p["id"]]
    if not g["is_impossible"] and p["abstain"]:
        if ae.forced_correct(g, base[p["id"]]["answer"]):
            over.append({"id": p["id"], "question": g["question"], "gold": g["answers"],
                         "m4_answer": p["answer"], "base_answer": base[p["id"]]["answer"]})
    elif g["is_impossible"] and not p["abstain"]:
        missed.append({"id": p["id"], "question": g["question"], "m4_answer": p["answer"]})

OUT.write_text(json.dumps({"over_refusal": over[:LIMIT], "missed_refusal": missed[:LIMIT]},
                          indent=1, ensure_ascii=False), encoding="utf-8")
print(f"over-refusal cases: {len(over)} total, {min(LIMIT, len(over))} saved")
print(f"missed-refusal cases: {len(missed)} total, {min(LIMIT, len(missed))} saved")
print("wrote", OUT)
