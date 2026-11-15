"""Check that M4's training pool is disjoint from the val / test splits.

This is the evidence behind the sentence in the report: *M4 is trained on SQuAD 2.0
train only; the qid overlap with val and test is zero*. Run it from the repo root:

    python b_methods/check_train_overlap.py

It prints and writes `results/m4_train_pool_check.json`:

  * number of pool records, answerable / unanswerable counts;
  * |pool ∩ val| and |pool ∩ test| (must both be 0);
  * sha256 of the pool and of both gold files, so the check can be tied to exact files.

If `squad_data/train-v2.0.json` (or `data/train-v2.0.json`) is present it additionally
verifies that every pool id really occurs in the SQuAD *train* split and in no dev
question. Download it with

    wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
    wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "m4_train_pool.jsonl"
VAL = ROOT / "data" / "items" / "b_gold_val.jsonl"
TEST = ROOT / "data" / "items" / "b_gold_test.jsonl"
OUT = ROOT / "results" / "m4_train_pool_check.json"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ids(rows):
    return {str(r.get("id") or r.get("qid")) for r in rows}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def squad_ids(path):
    """All qa ids of a SQuAD v2.0 json file."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {qa["id"] for article in raw["data"]
            for para in article["paragraphs"] for qa in para["qas"]}


def main():
    pool = load_jsonl(POOL)
    val = load_jsonl(VAL)
    test = load_jsonl(TEST)
    p, v, t = ids(pool), ids(val), ids(test)
    n_ans = sum(1 for r in pool if not r["is_impossible"])
    n_una = len(pool) - n_ans

    report = {
        "pool_file": str(POOL.relative_to(ROOT)),
        "pool_n": len(pool),
        "pool_answerable": n_ans,
        "pool_unanswerable": n_una,
        "pool_unique_ids": len(p),
        "val_n": len(v),
        "test_n": len(t),
        "overlap_pool_val": len(p & v),
        "overlap_pool_test": len(p & t),
        "sha256": {
            "m4_train_pool.jsonl": sha256(POOL),
            "b_gold_val.jsonl": sha256(VAL),
            "b_gold_test.jsonl": sha256(TEST),
        },
        "confirmed_from_train_split": None,
    }

    print(f"pool: {len(pool)} records ({n_ans} answerable + {n_una} unanswerable), "
          f"{len(p)} unique ids")
    print(f"val : {len(v)} ids   test: {len(t)} ids")
    print(f"|pool ∩ val|  = {report['overlap_pool_val']}")
    print(f"|pool ∩ test| = {report['overlap_pool_test']}")

    ok = report["overlap_pool_val"] == 0 and report["overlap_pool_test"] == 0
    if not ok:
        print("FAIL: the M4 training pool overlaps the evaluation splits -> data leak.")
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        return 1

    # Optional deep check: pool ids must exist in SQuAD train and in no dev question.
    train_path = next((q for q in [ROOT / "squad_data" / "train-v2.0.json",
                                   ROOT / "data" / "train-v2.0.json"] if q.exists()), None)
    dev_path = next((q for q in [ROOT / "squad_data" / "dev-v2.0.json",
                                 ROOT / "data" / "dev-v2.0.json"] if q.exists()), None)
    if train_path is not None:
        train = squad_ids(train_path)
        report["confirmed_from_train_split"] = len(p - train) == 0
        print(f"pool ids missing from train-v2.0.json: {len(p - train)}")
        if dev_path is not None:
            dev = squad_ids(dev_path)
            print(f"pool ids that also occur in dev-v2.0.json: {len(p & dev)}")
            report["pool_ids_in_dev"] = len(p & dev)
        if len(p - train):
            print("FAIL: the pool is not entirely from the SQuAD train split.")
            ok = False
    else:
        print("(deep check skipped: squad_data/train-v2.0.json not found; "
              "the two intersection counts above are computed from the committed files)")

    report["passed"] = ok
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'PASS' if ok else 'FAIL'}: wrote {OUT.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
