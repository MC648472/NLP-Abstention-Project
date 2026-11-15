"""
prepare_m4_train.py -- 从 SQuAD 2.0 的 *训练集* (train-v2.0.json) 里抽 M4 的训练题池。

注意: A 同学的 val/test 都来自 dev-v2.0.json, M4 的训练数据必须来自 train-v2.0.json,
      否则会和评测题重叠 (数据泄漏)。SQuAD 的 train 和 dev 文章不重叠, 互不影响。

下载 (在 Colab 里):
    !wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

用法:
    python prepare_m4_train.py --train squad_data/train-v2.0.json \
        --out data/m4_train_pool.jsonl --per-class 2500

输出每行: {"id","question","context","answers","is_impossible"}
可答和不可答各抽 per-class 条, 之后由 m4_lora.build_train_examples 按需要的比例再抽样。
"""
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
