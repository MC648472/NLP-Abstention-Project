import json
import os
import re
from collections import Counter
from contextlib import contextmanager

import abstention_eval as ev   # Shared scoring module.

PROMPT_CONF = """You are a reading comprehension assistant.
Read the context, then answer the question with a short phrase from the context.
Then say how confident you are that your answer is correct, as a number from 0 to 100.
Be honest: if the context does not clearly contain the answer, give a low number.
Always reply in exactly this two-line format and nothing else:
Answer: <short answer>
Confidence: <number>"""

# Few-shot turns for M2-conf.
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
M2_MAX_NEW_TOKENS = 48  # Space for two lines.


FEWSHOT = {"conf": FEWSHOT_CONF, "idk": []}


def _make_encode(slm, shots):
    """Return an encoder that prepends few-shot turns."""
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
    """Temporarily switch prompt, examples, and max_new_tokens."""
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
    """Parse 'Answer: ... / Confidence: 0-100'."""
    ans_m = re.search(r"Answer:\s*(.*?)\s*(?:\n|Confidence:|$)", raw, re.S | re.I)
    conf_m = re.search(r"Confidence:\s*(\d{1,3}(?:\.\d+)?)", raw, re.I)
    answer = ans_m.group(1).strip() if ans_m else raw.strip().split("\n")[0].strip()
    conf = max(0.0, min(100.0, float(conf_m.group(1)))) if conf_m else None
    return answer, conf


def parse_idk(raw):
    """Return the answer text."""
    m = re.search(r"Answer:\s*(.*)", raw, re.I)
    return (m.group(1) if m else raw).strip().split("\n")[0].strip()


def load_jsonl(path):
    return ev.load_jsonl(path)


def run_m2(slm, items, out_path, version="conf", overwrite=False):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if overwrite and os.path.exists(out_path):
        os.remove(out_path)
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
                   "avg_logprob": g["avg_logprob"],      # Keep avg_logprob for scoring.
                   "time_s": g["time_s"], "version": version}
            if version == "conf":
                ans, conf = parse_conf(raw)
                # Confidence in [0, 1].
                rec.update(answer=ans, conf=conf, parse_ok=conf is not None,
                           confidence=(conf if conf is not None else 0.0) / 100.0)
            else:
                rec.update(answer=parse_idk(raw), parse_ok=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


# Scoring is handled by src/score_m2_m4.py.


def diagnostics(pred_path, gold_path):
    """Print parse rate and confidence by answerability bucket."""
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


