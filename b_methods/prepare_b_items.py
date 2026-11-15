import argparse
import json
import os
import random


def load_squad_records(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = [a["text"].strip() for a in qa.get("answers", [])
                           if a["text"].strip()]
                records.append({
                    "qid": qa["id"],
                    "context": context,
                    "question": qa["question"].strip(),
                    "answers": answers,
                    "is_impossible": bool(qa.get("is_impossible", False)),
                })
    return records


def stratified_split(records, val_frac, seed):
    rng = random.Random(seed)
    answerable = [r for r in records if not r["is_impossible"]]
    unanswerable = [r for r in records if r["is_impossible"]]
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)

    def cut(items):
        k = int(len(items) * val_frac)
        return items[:k], items[k:]

    a_val, a_test = cut(answerable)
    u_val, u_test = cut(unanswerable)
    return a_val + u_val, a_test + u_test


def cap_buckets(records, max_per_bucket, seed):
    rng = random.Random(seed)
    answerable = [r for r in records if not r["is_impossible"]]
    unanswerable = [r for r in records if r["is_impossible"]]
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    return answerable[:max_per_bucket] + unanswerable[:max_per_bucket]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="squad_data/dev-v2.0.json")
    parser.add_argument("--from-splits", default="",
                        help="directory containing squad2_val.json / squad2_test.json")
    parser.add_argument("--out-dir", default="b_handoff")
    parser.add_argument("--val-frac", type=float, default=0.5)
    parser.add_argument("--max-per-bucket", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.from_splits:
        with open(os.path.join(args.from_splits, "squad2_val.json"), encoding="utf-8") as f:
            val = json.load(f)
        with open(os.path.join(args.from_splits, "squad2_test.json"), encoding="utf-8") as f:
            test = json.load(f)
    else:
        dev = load_squad_records(args.dev)
        val, test = stratified_split(dev, args.val_frac, args.seed)
        val = cap_buckets(val, args.max_per_bucket, args.seed)
        test = cap_buckets(test, args.max_per_bucket, args.seed)

    for split_name, records in (("val", val), ("test", test)):
        items = [{
            "id": r["qid"],
            "question": r["question"],
            "context": r["context"],
            "gold": r["answers"],
        } for r in records]
        gold = [{
            "id": r["qid"],
            "question": r["question"],
            "answers": r["answers"],
            "is_impossible": r["is_impossible"],
        } for r in records]
        write_jsonl(os.path.join(args.out_dir, f"b_items_{split_name}.jsonl"), items)
        write_jsonl(os.path.join(args.out_dir, f"b_gold_{split_name}.jsonl"), gold)
        answerable = sum(not r["is_impossible"] for r in records)
        print(f"{split_name}: {len(records)} questions "
              f"({answerable} answerable, {len(records) - answerable} unanswerable)")

    print(f"written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
