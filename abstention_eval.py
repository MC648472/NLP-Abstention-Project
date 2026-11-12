"""Shared SQuAD 2.0 abstention evaluation utilities (student A's module).

Pure Python + numpy, no torch/transformers, so both A and B can run it.

Typical use from the command line:

    python abstention_eval.py --mode m1 \
        --val-pred slm_m1_val.jsonl --val-gold b_gold_val.jsonl \
        --test-pred slm_m1_test.jsonl --test-gold b_gold_test.jsonl

Prediction JSONL (greedy / M1): one line per question with at least
    {"id": ..., "answer": ..., "avg_logprob": ...}
Prediction JSONL (sample / M3): one line per question with at least
    {"id": ..., "answers": [a1, ..., a8]}

Gold JSONL: {"id": ..., "question": ..., "answers": [...], "is_impossible": bool}
"""
import argparse
import json
import math
import re
import string
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Normalisation and correctness (identical to the notebook)
# ---------------------------------------------------------------------------
REFUSAL_PATTERNS = [
    "i dont know", "i do not know", "dont know", "do not know",
    "cannot answer", "can not answer", "cant answer",
    "could not answer", "couldnt answer", "unable to answer",
    "not enough information", "insufficient information", "no information",
    "cannot be determined", "cant be determined", "cannot determine",
    "not specified", "not stated", "not mentioned", "unanswerable",
    "no answer", "no relevant information", "i cannot find",
    "do not have enough", "there is no information",
]


def normalize_answer(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def em_score(prediction, gold):
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_against_golds(prediction, golds):
    em = max((em_score(prediction, g) for g in golds), default=0)
    f1 = max((f1_score(prediction, g) for g in golds), default=0.0)
    return em, f1


def is_abstention(text):
    norm = normalize_answer(text)
    if not norm:
        return True
    return any(pattern in norm for pattern in REFUSAL_PATTERNS)


def judge(record, prediction, abstain):
    if record["is_impossible"]:
        return abstain, 0, 0.0
    if abstain:
        return False, 0, 0.0
    em, f1 = score_against_golds(prediction, record["answers"])
    return em == 1, em, f1


def forced_correct(record, prediction):
    if record["is_impossible"]:
        return False
    return score_against_golds(prediction, record["answers"])[0] == 1


# ---------------------------------------------------------------------------
# Clustering for M3
# ---------------------------------------------------------------------------
def cluster_answers(answers):
    clusters = []
    for raw in answers:
        norm = normalize_answer(raw)
        placed = False
        for representative, members in clusters:
            if norm == representative or (
                norm and representative and (norm in representative or representative in norm)
            ):
                members.append(raw)
                placed = True
                break
        if not placed:
            clusters.append((norm, [raw]))
    return [members for _, members in clusters]


def consistency_score(answers):
    if not answers:
        return 0.0, 0
    clusters = cluster_answers(answers)
    return max(len(c) for c in clusters) / len(answers), len(clusters)


def majority_answer(answers):
    clusters = cluster_answers(answers)
    return max(clusters, key=len)[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def coverage(preds):
    answered = [p for p in preds if not p["abstain"]]
    return len(answered) / len(preds) if preds else float("nan")


def selective_accuracy(preds):
    answered = [p for p in preds if not p["abstain"]]
    return sum(p["correct"] for p in answered) / len(answered) if answered else float("nan")


def risk(preds):
    acc = selective_accuracy(preds)
    return float("nan") if math.isnan(acc) else 1.0 - acc


def correct_refusal_rate(preds):
    unans = [p for p in preds if p["is_impossible"]]
    return sum(p["abstain"] for p in unans) / len(unans) if unans else float("nan")


def over_refusal_rate(preds):
    ans = [p for p in preds if not p["is_impossible"]]
    return sum(p["abstain"] for p in ans) / len(ans) if ans else float("nan")


def correct_answer_refusal_rate(preds):
    """Unnecessary refusals among questions the model would have answered correctly."""
    eligible = [p for p in preds if not p["is_impossible"] and p.get("forced_correct")]
    return sum(p["abstain"] for p in eligible) / len(eligible) if eligible else float("nan")


def wrong_answer_refusal_rate(preds):
    """Refusals among answerable questions the model would have answered wrongly."""
    eligible = [p for p in preds if not p["is_impossible"] and not p.get("forced_correct")]
    return sum(p["abstain"] for p in eligible) / len(eligible) if eligible else float("nan")


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc_auprc(preds):
    # Threshold-independent label: would the answer be correct if we answered?
    labels = np.array([1 if p.get("forced_correct", p["correct"]) else 0 for p in preds], dtype=int)
    scores = np.array([p["confidence"] for p in preds], dtype=float)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    ranks = _rankdata(scores)
    auroc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, len(sorted_labels) + 1)
    auprc = float((precision * sorted_labels).sum() / n_pos)
    return float(auroc), auprc


def ece_score(preds, n_bins=10):
    if not preds:
        return float("nan")
    conf = np.array([min(max(p.get("prob", p["confidence"]), 0.0), 1.0) for p in preds])
    correct = np.array([1 if p.get("forced_correct", p["correct"]) else 0 for p in preds])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        members = (conf >= edges[i]) & (conf < edges[i + 1])
        if not members.any():
            continue
        ece += members.mean() * abs(correct[members].mean() - conf[members].mean())
    return float(ece)


def brier(preds):
    if not preds:
        return float("nan")
    return float(np.mean([(p.get("prob", p["confidence"])
                           - (1 if p.get("forced_correct", p["correct"]) else 0)) ** 2
                          for p in preds]))


def evaluate(preds):
    auroc, auprc = auroc_auprc(preds)
    return {
        "n": len(preds),
        "coverage": coverage(preds),
        "selective_accuracy": selective_accuracy(preds),
        "risk": risk(preds),
        "correct_refusal_rate": correct_refusal_rate(preds),
        "over_refusal_rate": over_refusal_rate(preds),
        "correct_answer_refusal_rate": correct_answer_refusal_rate(preds),
        "wrong_answer_refusal_rate": wrong_answer_refusal_rate(preds),
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece_score(preds),
        "brier": brier(preds),
        # Overall decision accuracy: correct answers + correctly refused unanswerables.
        # This is the metric to maximise when choosing a threshold; refusal F1 alone
        # rewards refusing everything and collapses coverage to 11%.
        "behaviour_accuracy": float(np.mean([1 if p["correct"] else 0 for p in preds])) if preds else float("nan"),
        "avg_samples": float(np.mean([p["n_samples"] for p in preds])) if preds else float("nan"),
        "avg_latency_s": float(np.mean([p.get("latency_s") or 0.0 for p in preds])) if preds else float("nan"),
    }


# ---------------------------------------------------------------------------
# JSONL loading and prediction construction
# ---------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_id(rows):
    return {row["id"]: row for row in rows}


def m1_preds(gold_rows, pred_rows):
    preds = []
    for record in gold_rows:
        pred = pred_rows[record["id"]]
        answer = pred.get("answer", "")
        confidence = float(pred.get("avg_logprob", -99.0))
        verbal = is_abstention(answer)
        correct, em, f1 = judge(record, answer, verbal)
        preds.append({
            "id": record["id"], "is_impossible": record["is_impossible"],
            "answer": answer, "abstain": verbal, "verbal_abstain": verbal,
            "confidence": confidence, "prob": math.exp(max(-50.0, confidence)),
            "correct": correct, "forced_correct": forced_correct(record, answer),
            "em": em, "f1": f1,
            "n_samples": 1, "latency_s": pred.get("time_s"),
        })
    return preds


def m3_preds(gold_rows, pred_rows, k=5):
    preds = []
    for record in gold_rows:
        answers = pred_rows[record["id"]].get("answers", [])[:k]
        consistency, _ = consistency_score(answers)
        answer = majority_answer(answers) if answers else ""
        refusal = is_abstention(answer)
        correct, em, f1 = judge(record, answer, refusal)
        preds.append({
            "id": record["id"], "is_impossible": record["is_impossible"],
            "answer": answer, "answers_sampled": answers,
            "abstain": refusal, "verbal_abstain": refusal,
            "confidence": consistency, "prob": consistency,
            "correct": correct, "forced_correct": forced_correct(record, answer),
            "em": em, "f1": f1,
            "n_samples": len(answers), "latency_s": pred_rows[record["id"]].get("time_s"),
        })
    return preds


def refusal_preds(gold_rows, pred_rows):
    """M2 / M4: the decision is the model's own refusal, no threshold is applied."""
    preds = []
    for record in gold_rows:
        pred = pred_rows[record["id"]]
        answer = pred.get("answer", "")
        abstain = is_abstention(answer)
        # Priority: an explicit 0-1 confidence (M2 verbal confidence) > token logprob
        # > a degenerate 1/0 indicator. Keeping the source explicit means M2/M4 are
        # not silently scored on logprobs when B sends a `confidence` field.
        if pred.get("confidence") is not None:
            confidence = float(pred["confidence"])
            prob = min(max(confidence, 0.0), 1.0)
        elif "avg_logprob" in pred:
            confidence = float(pred["avg_logprob"])
            prob = math.exp(max(-50.0, confidence))
        else:
            confidence = 1.0 if not abstain else 0.0
            prob = confidence
        correct, em, f1 = judge(record, answer, abstain)
        preds.append({
            "id": record["id"], "is_impossible": record["is_impossible"],
            "answer": answer, "abstain": abstain, "verbal_abstain": abstain,
            "confidence": confidence,
            "prob": prob,
            "correct": correct, "forced_correct": forced_correct(record, answer),
            "em": em, "f1": f1,
            "n_samples": 1, "latency_s": pred.get("time_s"),
        })
    return preds


# ---------------------------------------------------------------------------
# Threshold tuning (val only)
# ---------------------------------------------------------------------------
def gold_should_abstain(record, pred):
    if record["is_impossible"]:
        return True
    return not forced_correct(record, pred["answer"])


def tune_threshold(preds, gold_by_id):
    candidates = sorted({p["confidence"] for p in preds})
    best_tau, best_f1 = candidates[0], -1.0
    for tau in candidates:
        tp = fp = fn = 0
        for p in preds:
            record = gold_by_id[p["id"]]
            should = gold_should_abstain(record, p)
            predicted = p.get("verbal_abstain", False) or p["confidence"] < tau
            if predicted and should:
                tp += 1
            elif predicted and not should:
                fp += 1
            elif not predicted and should:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return float(best_tau), float(best_f1)


def apply_threshold(preds, gold_by_id, tau):
    out = []
    for p in preds:
        p = dict(p)
        p["abstain"] = p.get("verbal_abstain", False) or p["confidence"] < tau
        p["correct"], p["em"], p["f1"] = judge(gold_by_id[p["id"]], p["answer"], p["abstain"])
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Fixed-coverage operating points (the comparison protocol agreed with B)
# ---------------------------------------------------------------------------
def _answers_at(p, tau):
    return (not p.get("verbal_abstain", False)) and p["confidence"] >= tau


def coverage_at(preds, tau):
    return sum(_answers_at(p, tau) for p in preds) / len(preds)


def tune_at_coverage(preds, target_coverage):
    """Val-only: tau whose coverage is closest to `target_coverage`.

    Abstain iff confidence < tau, i.e. answered iff confidence >= tau.
    Returns (tau, achieved_val_coverage). If the method cannot go that low
    (e.g. M3's consistency only has 5 levels -> minimum coverage 0.49), the
    closest reachable point is returned and the caller must flag it.
    """
    levels = sorted({p["confidence"] for p in preds}, reverse=True)
    candidates = [levels[0] + 1e-9] + levels          # first one -> coverage 0
    best_tau, best_cov = candidates[0], 0.0
    for tau in candidates:
        cov = coverage_at(preds, tau)
        if abs(cov - target_coverage) < abs(best_cov - target_coverage):
            best_tau, best_cov = tau, cov
    return float(best_tau), float(best_cov)


def behaviour_accuracy_at(preds, tau):
    right = 0
    for p in preds:
        if p.get("verbal_abstain", False) or p["confidence"] < tau:
            right += int(p["is_impossible"])
        else:
            right += int(p.get("forced_correct", p["correct"]))
    return right / len(preds)


def tune_by_accuracy(preds):
    """Val-only: tau maximising overall behaviour accuracy (the honest objective)."""
    best_tau, best_acc = None, -1.0
    for tau in sorted({p["confidence"] for p in preds}):
        acc = behaviour_accuracy_at(preds, tau)
        if acc > best_acc:
            best_tau, best_acc = tau, acc
    return float(best_tau), float(best_acc)


def risk_coverage_curve(preds):
    """Step curve over distinct confidence levels (ties are never split, so the
    curve is independent of record order). Only non-verbal answers are ranked;
    verbal refusals are always abstentions."""
    n = len(preds)
    ranked = sorted([p for p in preds if not p.get("verbal_abstain", False)],
                    key=lambda p: -p["confidence"])
    points = [{"coverage": 0.0, "selective_accuracy": float("nan"),
               "behaviour_accuracy": float(np.mean([1 if p["is_impossible"] else 0 for p in preds])),
               "risk": float("nan"), "tau": float("inf")}]
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1]["confidence"] == ranked[i]["confidence"]:
            j += 1
        block = ranked[: j + 1]
        answered = len(block)
        correct = sum(1 for p in block if p.get("forced_correct"))
        refused = n - answered
        refused_impossible = sum(1 for p in preds
                                 if p["is_impossible"] and not _answers_at(p, ranked[i]["confidence"]))
        points.append({
            "coverage": answered / n,
            "selective_accuracy": correct / answered,
            "behaviour_accuracy": (correct + refused_impossible) / n,
            "risk": 1.0 - correct / answered,
            "tau": float(ranked[i]["confidence"]),
        })
        i = j + 1
    # points are ordered high-coverage -> low-coverage; sort ascending for plotting
    pts = sorted({(round(p["coverage"], 6), p["selective_accuracy"], p["behaviour_accuracy"],
                   p["risk"], p["tau"]) for p in points}, key=lambda t: t[0])
    return [{"coverage": c, "selective_accuracy": a, "behaviour_accuracy": b,
             "risk": r, "tau": tau} for c, a, b, r, tau in pts]


def interp_selective_accuracy(preds, target_coverage):
    """Interpolated selective accuracy on the risk-coverage step curve.
    Returns NaN when the target is outside the reachable coverage range."""
    pts = [p for p in risk_coverage_curve(preds)
           if p["selective_accuracy"] == p["selective_accuracy"]]   # drop the NaN at coverage 0
    xs = [p["coverage"] for p in pts]
    ys = [p["selective_accuracy"] for p in pts]
    if not xs or target_coverage < min(xs) or target_coverage > max(xs):
        return float("nan")
    for i in range(len(xs) - 1):
        if xs[i] <= target_coverage <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return float(ys[i])
            w = (target_coverage - xs[i]) / (xs[i + 1] - xs[i])
            return float(ys[i] + w * (ys[i + 1] - ys[i]))
    return float("nan")


# ---------------------------------------------------------------------------
# Two-stage abstainer (A's method) on cached M1 / M3 predictions
# ---------------------------------------------------------------------------
_JUDGE_CACHE = {}


def _judge_cached(gold_by_id, qid, answer, abstain):
    """judge() re-runs normalisation on every grid point; the answers repeat."""
    key = (qid, answer, bool(abstain))
    hit = _JUDGE_CACHE.get(key)
    if hit is None:
        hit = judge(gold_by_id[qid], answer, abstain)
        _JUDGE_CACHE[key] = hit
    return hit


def simulate_two_stage(m1_preds, m3_preds, gold_by_id, tau_low, tau_high, tau_c):
    """Stage 1 = M1 screening; borderline questions use M3's majority answer."""
    m3_by_id = {p["id"]: p for p in m3_preds}
    out = []
    for p1 in m1_preds:
        s1 = p1["confidence"]
        p3 = m3_by_id.get(p1["id"])
        if s1 >= tau_high:
            answer, stage, consistency, n = p1["answer"], "high", None, 1
            abstain = p1.get("verbal_abstain", False)
            confidence = float(p1["prob"])
        elif s1 <= tau_low:
            answer, stage, consistency, n = p1["answer"], "low", None, 1
            abstain = True
            confidence = float(p1["prob"])
        else:
            answer, stage = p3["answer"], "borderline"
            consistency = float(p3["confidence"])
            abstain = p3.get("verbal_abstain", False) or consistency < tau_c
            n = 1 + int(p3["n_samples"])
            confidence = consistency
        correct, em, f1 = _judge_cached(gold_by_id, p1["id"], answer, abstain)
        out.append({
            "id": p1["id"], "is_impossible": p1["is_impossible"], "answer": answer,
            "abstain": abstain, "verbal_abstain": p1.get("verbal_abstain", False),
            "confidence": confidence, "prob": confidence,
            "correct": correct, "forced_correct": forced_correct(gold_by_id[p1["id"]], answer),
            "em": em, "f1": f1, "stage": stage, "consistency": consistency,
            "n_samples": n,
            "latency_s": (p1.get("latency_s") or 0.0)
                         + ((p3.get("latency_s") or 0.0) if stage == "borderline" else 0.0),
        })
    return out


def tune_two_stage_at_coverage(m1_preds, m3_preds, gold_by_id, target_coverage,
                               quantiles=None, coverage_tol=0.02):
    """Grid-search (tau_low, tau_high, tau_c) on val for a target coverage.

    Objective: among configs whose val coverage is within `coverage_tol` of the
    target, maximise behaviour accuracy; ties are broken by *fewer samples*
    (that is the compute-aware claim, so it must not be decided by grid order).
    Returns ((tau_low, tau_high, tau_c), val_accuracy, val_coverage).
    """
    quantiles = quantiles or [i / 20 for i in range(1, 20)]
    scores = np.sort([p["confidence"] for p in m1_preds])
    cuts = sorted({float(np.quantile(scores, q)) for q in quantiles})
    best = None
    for i, low in enumerate(cuts[:-1]):
        for high in cuts[i + 1:]:
            for tau_c in (0.2, 0.4, 0.6, 0.8, 1.0):
                preds = simulate_two_stage(m1_preds, m3_preds, gold_by_id, low, high, tau_c)
                cov = float(np.mean([0 if p["abstain"] else 1 for p in preds]))
                if abs(cov - target_coverage) > coverage_tol:
                    continue
                acc = float(np.mean([1 if p["correct"] else 0 for p in preds]))
                ns = float(np.mean([p["n_samples"] for p in preds]))
                if (best is None or acc > best[1] + 1e-9
                        or (abs(acc - best[1]) <= 1e-9 and ns < best[3])):
                    best = ((float(low), float(high), float(tau_c)), acc, cov, ns)
    if best is None:      # target out of reach: take the closest coverage
        for i, low in enumerate(cuts[:-1]):
            for high in cuts[i + 1:]:
                for tau_c in (0.4, 0.6, 0.8, 1.0):
                    preds = simulate_two_stage(m1_preds, m3_preds, gold_by_id, low, high, tau_c)
                    cov = float(np.mean([0 if p["abstain"] else 1 for p in preds]))
                    acc = float(np.mean([1 if p["correct"] else 0 for p in preds]))
                    ns = float(np.mean([p["n_samples"] for p in preds]))
                    if best is None or abs(cov - target_coverage) < abs(best[2] - target_coverage):
                        best = ((float(low), float(high), float(tau_c)), acc, cov, ns)
    return best[0], best[1], best[2]


def format_metrics(metrics):
    return " | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in metrics.items())


def main():
    parser = argparse.ArgumentParser(description="Score B's JSONL with A's abstention metrics")
    parser.add_argument("--mode", choices=["m1", "m3", "refusal"], required=True,
                        help="m1=logprob+threshold, m3=consistency+threshold, refusal=M2/M4 raw refusal")
    parser.add_argument("--val-pred", required=True)
    parser.add_argument("--val-gold", required=True)
    parser.add_argument("--test-pred", required=True)
    parser.add_argument("--test-gold", required=True)
    parser.add_argument("--k", type=int, default=5, help="samples used for M3")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    val_gold = load_jsonl(args.val_gold)
    test_gold = load_jsonl(args.test_gold)
    val_gold_by_id = index_by_id(val_gold)
    test_gold_by_id = index_by_id(test_gold)
    val_preds_raw = index_by_id(load_jsonl(args.val_pred))
    test_preds_raw = index_by_id(load_jsonl(args.test_pred))

    if args.mode == "refusal":
        val_scored = refusal_preds(val_gold, val_preds_raw)
        test_scored = refusal_preds(test_gold, test_preds_raw)
        print(f"{args.mode.upper()} (no threshold: abstain iff the model refuses)")
        print("val :", format_metrics(evaluate(val_scored)))
        print("test:", format_metrics(evaluate(test_scored)))
    else:
        build = m1_preds if args.mode == "m1" else (lambda g, p: m3_preds(g, p, k=args.k))
        val_preds = build(val_gold, val_preds_raw)
        test_preds = build(test_gold, test_preds_raw)
        tau, val_f1 = tune_threshold(val_preds, val_gold_by_id)
        val_scored = apply_threshold(val_preds, val_gold_by_id, tau)
        test_scored = apply_threshold(test_preds, test_gold_by_id, tau)
        print(f"{args.mode.upper()} tau={tau:.4f}  val refusal F1={val_f1:.4f}")
        print("val :", format_metrics(evaluate(val_scored)))
        print("test:", format_metrics(evaluate(test_scored)))

    with open(f"{args.out_dir.rstrip('/')}/{args.mode}_test_scored.jsonl", "w", encoding="utf-8") as f:
        for p in test_scored:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
