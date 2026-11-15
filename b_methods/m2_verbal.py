import json
import math
import os
import re
from collections import Counter
from contextlib import contextmanager

import abstention_eval as ev   # A 同学的评分模块, 放在同一个 code/ 文件夹

PROMPT_CONF = """You are a reading comprehension assistant.
Read the context, then answer the question with a short phrase from the context.
Then say how confident you are that your answer is correct, as a number from 0 to 100.
Be honest: if the context does not clearly contain the answer, give a low number.
Always reply in exactly this two-line format and nothing else:
Answer: <short answer>
Confidence: <number>"""

# conf 版的示例以真实对话轮次放进提示 (小模型对这种方式的格式遵守更好)。
FEWSHOT_CONF = [
    ("Paris is the capital and largest city of France.",
     "What is the capital of France?",
     "Answer: Paris\nConfidence: 98"),
    ("The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
     "Who was the first visitor to the Eiffel Tower?",
     "Answer: Gustave Eiffel\nConfidence: 5"),
    ("The Amazon River is about 6,400 kilometres long and flows through Brazil.",
     "How long is the Amazon River?",
     "Answer: about 6,400 kilometres\nConfidence: 90"),
    ("Mount Everest is the highest mountain above sea level, located in the Himalayas.",
     "Which year did Mount Everest first erupt?",
     "Answer: 1953\nConfidence: 10"),
]

PROMPT_IDK = """You are a reading comprehension assistant.
Read the context, then answer the question with a short phrase from the context. Do not explain.
If the answer is not in the context, reply exactly: I don't know

Example 1
Context: Paris is the capital and largest city of France.
Question: What is the capital of France?
Answer: Paris

Example 2
Context: The Eiffel Tower was completed in 1889 for the World's Fair in Paris.
Question: Who was the first visitor to the Eiffel Tower?
Answer: I don't know"""

PROMPTS = {"conf": PROMPT_CONF, "idk": PROMPT_IDK}
M2_MAX_NEW_TOKENS = 48  # conf 版要多输出 "Answer: ... Confidence: ..."


FEWSHOT = {"conf": FEWSHOT_CONF, "idk": []}


def _make_encode(slm, shots):
    """生成带示例对话轮次的 _encode, 临时替换 slm._encode 使用。"""
    def _encode(question, context=None):
        msgs = [{"role": "system", "content": slm.system_prompt}]
        for c, q, a in shots:
            msgs.append({"role": "user", "content": f"Context: {c}\nQuestion: {q}"})
            msgs.append({"role": "assistant", "content": a})
        user = question if not context else f"Context: {context}\nQuestion: {question}"
        msgs.append({"role": "user", "content": user})
        text = slm.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        return slm.tok(text, return_tensors="pt").to(slm.device)
    return _encode


@contextmanager
def use_prompt(slm, version):
    """临时切换提示词、示例和生成长度, 用完自动还原, 不影响 M1/M3。"""
    old_prompt, old_len = slm.system_prompt, slm.max_new_tokens
    slm.system_prompt = PROMPTS[version]
    slm.max_new_tokens = M2_MAX_NEW_TOKENS
    if FEWSHOT[version]:
        slm._encode = _make_encode(slm, FEWSHOT[version])
    try:
        yield
    finally:
        slm.system_prompt, slm.max_new_tokens = old_prompt, old_len
        if "_encode" in slm.__dict__:
            del slm._encode


def parse_conf(raw):
    """从 'Answer: xxx\nConfidence: 85' 取出答案和置信度 (0-100)。失败时 conf=None。"""
    ans_m = re.search(r"Answer:\s*(.*?)\s*(?:\n|Confidence:|$)", raw, re.S | re.I)
    conf_m = re.search(r"Confidence:\s*(\d{1,3}(?:\.\d+)?)", raw, re.I)
    answer = ans_m.group(1).strip() if ans_m else raw.strip().split("\n")[0].strip()
    conf = max(0.0, min(100.0, float(conf_m.group(1)))) if conf_m else None
    return answer, conf


def parse_idk(raw):
    """返回答案文本 (拒答与否交给 A 的 is_abstention 判断)。"""
    m = re.search(r"Answer:\s*(.*)", raw, re.I)
    return (m.group(1) if m else raw).strip().split("\n")[0].strip()


def load_jsonl(path):
    return ev.load_jsonl(path)


def run_m2(slm, items, out_path, version="conf"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        done = {r["id"] for r in load_jsonl(out_path)}
    print(f"已完成 {len(done)} 题, 剩余 {len(items) - len(done)} 题")

    try:
        from tqdm import tqdm
        it_iter = tqdm(items)
    except ImportError:
        it_iter = items

    with use_prompt(slm, version), open(out_path, "a", encoding="utf-8") as f:
        for it in it_iter:
            if it["id"] in done:
                continue
            g = slm.generate(it["question"], it.get("context"))
            raw = g["answer"]
            rec = {"id": it["id"], "question": it["question"], "raw": raw,
                   "avg_logprob": g["avg_logprob"],      # A 要求每条都带 avg_logprob
                   "time_s": g["time_s"], "version": version}
            if version == "conf":
                ans, conf = parse_conf(raw)
                # confidence: 0-1, 供 A 的评分脚本直接读取; 解析失败按 0 处理
                rec.update(answer=ans, conf=conf, parse_ok=conf is not None,
                           confidence=(conf if conf is not None else 0.0) / 100.0)
            else:
                rec.update(answer=parse_idk(raw), parse_ok=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def build_preds(gold_rows, pred_by_id, version):
    """把 M2 结果整理成 A 的评分模块需要的格式。"""
    preds = []
    for g in gold_rows:
        r = pred_by_id[g["id"]]
        answer = r["answer"]
        verbal = ev.is_abstention(answer)           # 模型直接说 I don't know 也算拒答
        if version == "conf":
            conf = (r["conf"] if r.get("conf") is not None else 0.0) / 100.0
        else:
            conf = 0.0 if verbal else 1.0           # idk 版没有连续置信度
        correct, em, f1 = ev.judge(g, answer, verbal)
        preds.append({"id": g["id"], "is_impossible": g["is_impossible"],
                      "answer": answer, "abstain": verbal, "verbal_abstain": verbal,
                      "confidence": conf, "prob": conf,
                      "correct": correct, "em": em, "f1": f1,
                      # 假装必答时答案对不对 (A 的 AUROC / 答对被拒率等指标用这个字段)
                      "forced_correct": ev.forced_correct(g, answer),
                      "n_samples": 1, "latency_s": r.get("time_s")})
    return preds


def score_m2(version, val_pred, val_gold, test_pred, test_gold, out_dir=None):
    vg, tg = ev.load_jsonl(val_gold), ev.load_jsonl(test_gold)
    vg_by, tg_by = ev.index_by_id(vg), ev.index_by_id(tg)
    vp = build_preds(vg, ev.index_by_id(ev.load_jsonl(val_pred)), version)
    tp = build_preds(tg, ev.index_by_id(ev.load_jsonl(test_pred)), version)

    if version == "conf":
        tau, f1 = ev.tune_threshold(vp, vg_by)
        print(f"M2-conf tau={tau:.3f} (置信度 0-1), val 拒答 F1={f1:.4f}")
    else:
        tau = float("-inf")
        print("M2-idk: 不使用阈值")
    vs = ev.apply_threshold(vp, vg_by, tau)
    ts = ev.apply_threshold(tp, tg_by, tau)
    vm, tm = ev.evaluate(vs), ev.evaluate(ts)
    print("val :", ev.format_metrics(vm))
    print("test:", ev.format_metrics(tm))
    if out_dir:
        with open(f"{out_dir.rstrip('/')}/m2_{version}_test_scored.jsonl", "w", encoding="utf-8") as f:
            for p in ts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return vm, tm, tau, ts


def diagnostics(pred_path, gold_path):
    """conf 版的诊断: 解析成功率, 以及自报置信度能不能区分可答 / 不可答 / 答错。"""
    recs = load_jsonl(pred_path)
    gold = ev.index_by_id(ev.load_jsonl(gold_path))
    ok = [r for r in recs if r["parse_ok"]]
    print(f"解析成功率: {len(ok)}/{len(recs)} = {len(ok) / len(recs):.1%}")
    groups = {"可答且答对": [], "可答但答错": [], "不可答": []}
    for r in ok:
        g = gold[r["id"]]
        if g["is_impossible"]:
            groups["不可答"].append(r["conf"])
        elif ev.forced_correct(g, r["answer"]):
            groups["可答且答对"].append(r["conf"])
        else:
            groups["可答但答错"].append(r["conf"])
    for name, v in groups.items():
        if v:
            print(f"{name}: {len(v)} 题, 平均自报置信度 {sum(v) / len(v):.1f}, "
                  f">=90 的占比 {sum(c >= 90 for c in v) / len(v):.1%}")
    allc = [r["conf"] for r in ok]
    print("置信度分布 (每10分一档):",
          dict(sorted(Counter(int(c // 10) * 10 for c in allc).items())))
    print("(三组平均值越接近, 说明自报置信度越没有区分能力)")


