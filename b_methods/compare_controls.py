import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "b_methods")]

import abstention_eval as ev  # noqa: E402

GOLD = ROOT / "data" / "items" / "b_gold_test.jsonl"
RUNS = {
    "M1": ROOT / "data" / "outputs_b" / "slm_m1_test.jsonl",
    "ao1500": ROOT / "data" / "outputs_b" / "slm_ao1500_test.jsonl",
    "ao750": ROOT / "data" / "outputs_b" / "slm_ao750_test.jsonl",
    "M4": ROOT / "data" / "outputs_b" / "slm_m4_test.jsonl",
}


def answered_ids(gold_rows, pred_by_id):
    """Answerable question ids that this run did NOT refuse."""
    out = set()
    for g in gold_rows:
        if g["is_impossible"] or g["id"] not in pred_by_id:
            continue
        if not ev.is_abstention(pred_by_id[g["id"]]["answer"]):
            out.add(g["id"])
    return out


def accuracy(gold_rows, pred_by_id, ids=None):
    hit = tot = 0
    for g in gold_rows:
        if g["is_impossible"] or g["id"] not in pred_by_id:
            continue
        if ids is not None and g["id"] not in ids:
            continue
        tot += 1
        hit += int(ev.forced_correct(g, pred_by_id[g["id"]]["answer"]))
    return hit, tot, (hit / tot if tot else float("nan"))


def main():
    gold = ev.load_jsonl(GOLD)
    gid = ev.index_by_id(gold)
    preds = {}
    for name, path in RUNS.items():
        if path.exists():
            preds[name] = ev.index_by_id(ev.load_jsonl(path))
        else:
            print(f"skip {name}: {path.relative_to(ROOT)} not found")
    if "M4" not in preds:
        raise SystemExit("M4 generations are required for the matched-subset comparison")

    m4_answered = answered_ids(gold, preds["M4"])
    report = {"n_answerable": sum(1 for g in gold if not g["is_impossible"]),
              "m4_answered_answerable": len(m4_answered), "runs": {}}

    print(f"answerable questions: 500 | M4 answered {len(m4_answered)} of them\n")
    print(f"{'run':8s} {'answers/500':>11s} {'acc(all 500)':>13s} {'acc(M4-answered)':>17s}")
    for name, pred in preds.items():
        h_all, n_all, a_all = accuracy(gold, pred)
        h_sub, n_sub, a_sub = accuracy(gold, pred, ids=m4_answered)
        report["runs"][name] = {
            "answered_answerable": len(answered_ids(gold, pred)),
            "accuracy_all_answerable": a_all,
            "accuracy_on_m4_answered": a_sub,
            "n_m4_answered": n_sub,
        }
        print(f"{name:8s} {len(answered_ids(gold, pred)) / n_all:7.1%} "
              f"{a_all:12.1%} ({h_all}/{n_all}) {a_sub:20.1%} ({h_sub}/{n_sub})")

    if "ao1500" not in report["runs"]:
        print("\n(put slm_ao1500_test.jsonl / slm_ao750_test.jsonl in data/outputs_b/ to get the"
              " domain-QA vs refusal decomposition)")
        out = ROOT / "results" / "m4_attribution.json"
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
        return
    base = report["runs"]["ao1500"]["accuracy_all_answerable"]
    m4_forced = report["runs"]["M4"]["accuracy_all_answerable"]
    report["decomposition"] = {
        "m1_forced_on_answerable": report["runs"]["M1"]["accuracy_all_answerable"],
        "domain_qa_gain_over_m1": base - report["runs"]["M1"]["accuracy_all_answerable"],
        "refusal_training_cost_on_all_answerable": m4_forced - base,
        "m4_answered_accuracy": report["runs"]["M4"]["accuracy_on_m4_answered"],
        "ao1500_on_same_subset": report["runs"]["ao1500"]["accuracy_on_m4_answered"],
        "selection_effect": (report["runs"]["M4"]["accuracy_on_m4_answered"]
                             - report["runs"]["M4"]["accuracy_all_answerable"]),
    }
    d = report["decomposition"]
    print("\n--- decomposition ---")
    print(f"M1 forced on answerable                          {d['m1_forced_on_answerable']:.1%}")
    print(f"answer-only (domain QA) on answerable            {base:.1%}  "
          f"-> domain QA gain {d['domain_qa_gain_over_m1']:+.1%}")
    print(f"M4 forced on all answerable                      {m4_forced:.1%}  "
          f"-> refusal training cost {d['refusal_training_cost_on_all_answerable']:+.1%}")
    print(f"M4 accuracy on the {len(m4_answered)} it answers                        "
          f"{d['m4_answered_accuracy']:.1%}")
    print(f"answer-only on the same {len(m4_answered)}                 "
          f"{d['ao1500_on_same_subset']:.1%}  <- fair comparison")
    print(f"selection effect (same model, different subset)   {d['selection_effect']:+.1%}")

    out = ROOT / "results" / "m4_attribution.json"

    # paired test on the matched subset: is M4 really better than the answer-only control?
    if "ao1500" in preds:
        import numpy as np
        from math import comb

        ids = sorted(m4_answered)
        a = np.array([int(ev.forced_correct(gid[i], preds["M4"][i]["answer"])) for i in ids])
        b = np.array([int(ev.forced_correct(gid[i], preds["ao1500"][i]["answer"])) for i in ids])
        b01 = int(((a == 0) & (b == 1)).sum())
        b10 = int(((a == 1) & (b == 0)).sum())
        n_disc = b01 + b10
        p_mcnemar = (min(1.0, 2 * sum(comb(n_disc, k) for k in
                                     range(0, min(b01, b10) + 1)) / 2 ** n_disc)
                     if n_disc else 1.0)
        d = a - b
        rng = np.random.default_rng(0)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        report["matched_pair"] = {
            "n": len(ids), "m4": float(a.mean()), "ao1500": float(b.mean()),
            "delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "m4_wrong_control_right": b01, "m4_right_control_wrong": b10,
            "mcnemar_p": p_mcnemar,
        }
        print(f"\nM4 vs answer-only on the same {len(ids)} questions: "
              f"{a.mean():.1%} vs {b.mean():.1%}, delta {d.mean() * 100:+.1f} points "
              f"(95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]), McNemar p = {p_mcnemar:.2f}")

    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
