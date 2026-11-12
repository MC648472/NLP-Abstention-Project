"""
slm_interface.py  --  统一推理接口 (B同学负责, A同学的 M1 / M3 直接调用)

功能:
  1. generate(question)          单次贪心生成, 返回答案 + 每个token的对数概率 (给 M1 用)
  2. sample(question, n)         多次随机采样, 返回 n 个答案 (给 M3 用)
  3. run_batch(items, out_path)  批量运行, 逐题写入 JSONL, 断线后可从上次位置继续
  4. 资源统计: 单题耗时 / 峰值显存

用法见文件底部的 __main__ 示例, 或 Colab 里:
    from slm_interface import SLM
    slm = SLM()
    print(slm.generate("Who wrote Hamlet?"))
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 全组统一的提示词, 不要随便改, 否则各方法的结果不可比
SYSTEM_PROMPT = "Answer the question with a short phrase only. Do not explain."
MAX_NEW_TOKENS = 32


class SLM:
    def __init__(self, model_name=MODEL_NAME, lora_path=None,
                 system_prompt=SYSTEM_PROMPT, max_new_tokens=MAX_NEW_TOKENS):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:      # 防止 pad_token 缺失 (Qwen2.5 通常已有, 不影响现有结果)
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(self.device)

        # M4 训练完后, 传入 lora_path 即可加载微调后的模型
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.special_ids = set(self.tok.all_special_ids)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _encode(self, question, context=None):
        """把问题 (可选: 加原文段落) 变成模型输入。"""
        user = question if not context else f"Context: {context}\nQuestion: {question}"
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").to(self.device)

    def _now(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    # ------------------------------------------------------------------
    # 1. 单次贪心生成 (M1 / M2 / M4 使用)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, question, context=None):
        inputs = self._encode(question, context)
        n_in = inputs["input_ids"].shape[1]

        t0 = self._now()
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None, top_p=None, top_k=None,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tok.pad_token_id,
        )
        dt = self._now() - t0

        gen_ids = out.sequences[0, n_in:].tolist()
        tokens, logprobs, kept_ids = [], [], []
        for i, tid in enumerate(gen_ids):
            if tid in self.special_ids:      # 跳过结束符等特殊token
                continue
            lp = torch.log_softmax(out.scores[i][0].float(), dim=-1)[tid].item()
            kept_ids.append(tid)
            tokens.append(self.tok.decode([tid]))
            logprobs.append(lp)

        answer = self.tok.decode(kept_ids, skip_special_tokens=True).strip()
        n_tok = len(logprobs)
        return {
            "answer": answer,
            "tokens": tokens,
            "token_logprobs": logprobs,
            "sum_logprob": sum(logprobs) if n_tok else -99.0,
            # M1 的置信度分数: 长度归一化对数概率 (越大越有把握)
            "avg_logprob": (sum(logprobs) / n_tok) if n_tok else -99.0,
            "n_tokens": n_tok,
            "time_s": dt,
        }

    # ------------------------------------------------------------------
    # 2. 多次随机采样 (M3 使用)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, question, n=5, temperature=0.7, top_p=0.95,
               context=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)

        inputs = self._encode(question, context)
        n_in = inputs["input_ids"].shape[1]

        t0 = self._now()
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature, top_p=top_p, top_k=50,
            num_return_sequences=n,
            pad_token_id=self.tok.pad_token_id,
        )
        dt = self._now() - t0

        answers = [
            self.tok.decode(seq[n_in:], skip_special_tokens=True).strip()
            for seq in out
        ]
        return {"answers": answers, "n": n, "time_s": dt}

    # ------------------------------------------------------------------
    # 3. 批量运行, 可断点续跑
    # ------------------------------------------------------------------
    def run_batch(self, items, out_path, mode="greedy", n=8,
                  temperature=0.7, seed=0):
        """
        items: [{"id": ..., "question": ..., "context": 可选, "gold": 可选}, ...]
        mode:  "greedy" (单次生成) 或 "sample" (多次采样, 建议 n=8 一次存好,
               之后用前3/5/8个答案即可, 不用重复跑模型)
        out_path: 结果文件 (.jsonl), 建议放在 Google Drive 里
        """
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        done = set()
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done.add(json.loads(line)["id"])
        print(f"已完成 {len(done)} 题, 剩余 {len(items) - len(done)} 题")

        try:
            from tqdm import tqdm
            it_iter = tqdm(list(enumerate(items)))
        except ImportError:
            it_iter = enumerate(items)

        with open(out_path, "a", encoding="utf-8") as f:
            for i, it in it_iter:
                if it["id"] in done:
                    continue
                ctx = it.get("context")
                if mode == "greedy":
                    r = self.generate(it["question"], ctx)
                else:
                    r = self.sample(it["question"], n=n, temperature=temperature,
                                    context=ctx, seed=seed + i)
                r["id"] = it["id"]
                r["question"] = it["question"]
                if "gold" in it:
                    r["gold"] = it["gold"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()

    # ------------------------------------------------------------------
    # 4. 显存统计
    # ------------------------------------------------------------------
    def reset_peak_memory(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self):
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 ** 2
        return 0.0


if __name__ == "__main__":
    slm = SLM()
    slm.reset_peak_memory()
    r = slm.generate("Who wrote the play Hamlet?")
    print("答案:", r["answer"])
    print("长度归一化logprob:", round(r["avg_logprob"], 3))
    print("耗时(s):", round(r["time_s"], 3))
    s = slm.sample("Who wrote the play Hamlet?", n=5, seed=0)
    print("5次采样:", s["answers"])
    print("峰值显存(MB):", round(slm.peak_memory_mb(), 1))
