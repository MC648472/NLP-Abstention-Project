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
                    "id": qa["id"],
                    "question": qa["question"].strip(),
                    "context": context,
                    "answers": answers,
                    "is_impossible": bool(qa.get("is_impossible", False)),
                })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="squad_data/train-v2.0.json")
    ap.add_argument("--out", default="data/m4_train_pool.jsonl")
    ap.add_argument("--per-class", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = load_squad_records(args.train)
    ans = [r for r in recs if not r["is_impossible"] and r["answers"]]
    una = [r for r in recs if r["is_impossible"]]
    rng = random.Random(args.seed)
    rng.shuffle(ans)
    rng.shuffle(una)
    pool = ans[:args.per_class] + una[:args.per_class]
    rng.shuffle(pool)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train 全集 可答 {len(ans)} / 不可答 {len(una)}; "
          f"抽取 {len(pool)} 条写入 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
