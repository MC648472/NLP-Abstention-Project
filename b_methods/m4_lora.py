"""
m4_lora.py  --  M4: LoRA 拒答微调 (B同学), 适配 A 同学的 SQuAD 2.0 数据

流程:
  1. 从 SQuAD 2.0 train 题池构造训练集: 可答 -> 标准答案, 不可答 -> "I don't know"
  2. LoRA 训练 (fp32 底模 + fp16 混合精度, T4 可跑), 记录训练时长和峰值显存
  3. 在 A 的 val/test 上推理, 用 A 的 abstention_eval 打分
  4. 导出过度拒答 / 漏拒答案例

训练和推理都使用全组统一的 SYSTEM_PROMPT (没有拒答指令), 拒答能力完全来自微调。
"""
import json
import math
import os
import random
import time

import torch

import abstention_eval as ev
from slm_interface import MODEL_NAME, SYSTEM_PROMPT

REFUSAL_TEXT = "I don't know"   # 能被 A 的 is_abstention 识别


def load_jsonl(path):
    return ev.load_jsonl(path)


# ----------------------------------------------------------------------
# 1. 构造训练集
# ----------------------------------------------------------------------
def build_train_examples(pool, refusal_ratio=0.5, max_n=1500, seed=0):
    """
    pool: prepare_m4_train.py 生成的题池 (含 is_impossible)
    refusal_ratio: 训练集中拒答样本占比 (默认 0.5)
    """
    rng = random.Random(seed)
    can = [x for x in pool if not x["is_impossible"] and x.get("answers")]
    cannot = [x for x in pool if x["is_impossible"]]
    rng.shuffle(can)
    rng.shuffle(cannot)

    n_ref = min(int(max_n * refusal_ratio), len(cannot))
    n_ans = min(max_n - n_ref, len(can))
    ex = [{"question": x["question"], "context": x["context"],
           "target": x["answers"][0], "answerable": True} for x in can[:n_ans]]
    ex += [{"question": x["question"], "context": x["context"],
            "target": REFUSAL_TEXT, "answerable": False} for x in cannot[:n_ref]]
    rng.shuffle(ex)
    print(f"训练样本: 可答 {n_ans} + 拒答 {n_ref} = {len(ex)} "
          f"(拒答占比 {n_ref / max(len(ex), 1):.0%})")
    return ex


# ----------------------------------------------------------------------
# 2. 训练
# ----------------------------------------------------------------------
def _prompt_text(tok, question, context=None):
    """必须和 slm_interface._encode 完全一致, 否则训练和推理的提示词不匹配。"""
    user = question if not context else f"Context: {context}\nQuestion: {question}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _encode_example(tok, ex, max_len):
    p = tok(_prompt_text(tok, ex["question"], ex.get("context")),
            add_special_tokens=False).input_ids
    t = tok(ex["target"] + "<|im_end|>", add_special_tokens=False).input_ids
    ids = p + t
    if len(ids) > max_len:
        return None
    return ids, [-100] * len(p) + t       # 只对答案部分算损失


def _collate(batch, pad_id, device):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(i)] = 1
    return {"input_ids": ids.to(device), "labels": lab.to(device),
            "attention_mask": att.to(device)}


def train_lora(examples, out_dir, model_name=MODEL_NAME, epochs=2, lr=2e-4,
               bs=4, max_len=512, r=8, alpha=16, dropout=0.05, seed=0):
    """训练并把 LoRA 权重和训练日志存到 out_dir。返回日志字典。"""
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 训练用 fp32 底模 + fp16 自动混合精度, 比纯 fp16 稳定
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32).to(device)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    data = [e for e in (_encode_example(tok, x, max_len) for x in examples) if e]
    print(f"有效样本 {len(data)} / {len(examples)} (超过 {max_len} token 的被丢弃)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(data) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses, step = [], 0
    for ep in range(epochs):
        random.shuffle(data)
        for s in range(0, len(data), bs):
            batch = _collate(data[s:s + bs], tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % 25 == 0:
                print(f"epoch {ep + 1}/{epochs} step {step}/{steps} "
                      f"loss {sum(losses[-25:]) / 25:.4f}")
    train_time = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    log = {"n_examples": len(data), "epochs": epochs, "lr": lr, "batch_size": bs,
           "lora_r": r, "lora_alpha": alpha, "max_len": max_len,
           "train_time_s": train_time,
           "peak_mem_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
           "final_loss": sum(losses[-25:]) / max(len(losses[-25:]), 1),
           "loss_history": losses}
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    print(f"训练完成: {train_time / 60:.1f} 分钟, 峰值显存 {log['peak_mem_mb']:.0f} MB")
    del model, opt
    torch.cuda.empty_cache()
    return log


# ----------------------------------------------------------------------
# 3. 推理与评分 (评分全部用 A 的 abstention_eval)
# ----------------------------------------------------------------------
def infer_m4(lora_path, items, out_path):
    """加载微调后的模型, 对 A 的题目贪心推理, 结果格式和 M1 完全相同 (含 avg_logprob)。"""
    from slm_interface import SLM
    slm = SLM(lora_path=lora_path)
    slm.reset_peak_memory()
    slm.run_batch(items, out_path, mode="greedy")
    recs = load_jsonl(out_path)
    print(f"峰值显存 {slm.peak_memory_mb():.0f} MB, "
          f"平均耗时 {sum(r['time_s'] for r in recs) / len(recs):.2f} s/题")
    del slm
    torch.cuda.empty_cache()


def score_m4(pred_path, gold_path, out_path=None):
    """
    M4 的拒答完全来自模型自己输出的 'I don't know', 不再额外调阈值。
    返回指标字典和逐题结果。
    """
    gold = ev.load_jsonl(gold_path)
    gold_by = ev.index_by_id(gold)
    preds = ev.m1_preds(gold, ev.index_by_id(ev.load_jsonl(pred_path)))
    scored = ev.apply_threshold(preds, gold_by, float("-inf"))
    metrics = ev.evaluate(scored)
    print(ev.format_metrics(metrics))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for p in scored:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return metrics, scored


# ----------------------------------------------------------------------
# 4. 案例分析
# ----------------------------------------------------------------------
def case_list(scored, gold_path, kind="over_refusal", base_pred_path=None, k=10):
    """
    kind="over_refusal":   可答题却被 M4 拒答 (过度拒答)。
                           若给 base_pred_path (M1 结果), 只保留"底模其实答对"的题。
    kind="missed_refusal": 不可答题 M4 却给了答案 (漏拒答)。
    """
    gold_by = ev.index_by_id(ev.load_jsonl(gold_path))
    base = ev.index_by_id(ev.load_jsonl(base_pred_path)) if base_pred_path else None
    out = []
    for p in scored:
        g = gold_by[p["id"]]
        if kind == "over_refusal" and not g["is_impossible"] and p["abstain"]:
            if base is not None and not ev.forced_correct(g, base[p["id"]]["answer"]):
                continue
            out.append({"id": p["id"], "question": g["question"], "gold": g["answers"],
                        "m4_answer": p["answer"],
                        "base_answer": base[p["id"]]["answer"] if base else None})
        elif kind == "missed_refusal" and g["is_impossible"] and not p["abstain"]:
            out.append({"id": p["id"], "question": g["question"],
                        "m4_answer": p["answer"]})
    print(f"{kind}: 共 {len(out)} 题, 返回前 {min(k, len(out))} 题")
    return out[:k]
