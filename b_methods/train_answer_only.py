import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "b_methods")]

import abstention_eval as ev  # noqa: E402
import m4_lora as m4  # noqa: E402


def accuracy_on_answered_answerable(scored, gold_path):
    """Of the answerable questions the model answers, how many are right? (M4 = 232/290)"""
    gold = ev.index_by_id(ev.load_jsonl(gold_path))
    hit = tot = 0
    for p in scored:
        g = gold[p["id"]]
        if g["is_impossible"] or p["abstain"]:
            continue
        tot += 1
        hit += int(bool(p.get("correct")))
    return hit, tot, (hit / tot if tot else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=1500, help="total SFT examples (all answerable)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="build the training set and stop (no GPU)")
    args = ap.parse_args()
    tag = args.tag or f"ao{args.max_n}"

    pool = m4.load_jsonl(ROOT / "data" / "m4_train_pool.jsonl")
    examples = m4.build_train_examples(pool, refusal_ratio=0.0, max_n=args.max_n, seed=args.seed)
    n_ans = sum(1 for e in examples if e["answerable"])
    print(f"[{tag}] {n_ans} answerable examples, "
          f"{len(examples) - n_ans} refusal examples (target ratio 0.0)")
    if args.dry_run:
        print("dry run: no training, no inference")
        return

    adapter = ROOT / "models" / f"m4_lora_{tag}"
    log = m4.train_lora(examples, str(adapter), epochs=args.epochs, seed=args.seed)

    items = {s: m4.load_jsonl(ROOT / "data" / "items" / f"b_items_{s}.jsonl")
             for s in ("val", "test")}
    gold = {s: ROOT / "data" / "items" / f"b_gold_{s}.jsonl" for s in ("val", "test")}
    summary = {"tag": tag, "max_n": args.max_n, "n_answerable": n_ans,
               "n_refusal": len(examples) - n_ans, "train_log": log, "splits": {}}

    for split in ("val", "test"):
        out = ROOT / "data" / "outputs_b" / f"slm_{tag}_{split}.jsonl"
        m4.infer_m4(str(adapter), items[split], str(out))
        metrics, scored = m4.score_m4(str(out), str(gold[split]))
        hit, tot, acc = accuracy_on_answered_answerable(scored, str(gold[split]))
        summary["splits"][split] = {
            "coverage": metrics["coverage"],
            "selective_accuracy": metrics["selective_accuracy"],
            "behaviour_accuracy": metrics["behaviour_accuracy"],
            "correct_refusal_rate": metrics["correct_refusal_rate"],
            "over_refusal_rate": metrics["over_refusal_rate"],
            "avg_samples": metrics["avg_samples"],
            "avg_latency_s": metrics.get("avg_latency_s"),
            "answerable_answered": tot,
            "answerable_answered_correct": hit,
            "accuracy_on_answered_answerable": acc,
        }
        if split == "test":
            with open(ROOT / "results" / f"{tag}_test_scored.jsonl", "w", encoding="utf-8") as f:
                for p in scored:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

    out = ROOT / "results" / f"{tag}_summary.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    t = summary["splits"]["test"]
    print(f"\n[{tag}] test: coverage {t['coverage']:.3f} | selective {t['selective_accuracy']:.3f} "
          f"| overall {t['behaviour_accuracy']:.3f} | over-refusal {t['over_refusal_rate']:.3f} "
          f"| answered-answerable acc {t['accuracy_on_answered_answerable']:.1%} "
          f"({t['answerable_answered_correct']}/{t['answerable_answered']})")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
