import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Shared system prompt.
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
        if self.tok.pad_token is None:      # Fall back to EOS if pad_token is missing.
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(self.device)

        # Optional LoRA adapter.
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.special_ids = set(self.tok.all_special_ids)

    def _encode(self, question, context=None):
        """Build the model input for one question."""
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
            if tid in self.special_ids:      # Skip special tokens (EOS etc.).
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
            # Length-normalised log-probability (higher = more confident).
            "avg_logprob": (sum(logprobs) / n_tok) if n_tok else -99.0,
            "n_tokens": n_tok,
            "time_s": dt,
        }

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

    def run_batch(self, items, out_path, mode="greedy", n=8,
                  temperature=0.7, seed=0, overwrite=False):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if overwrite and os.path.exists(out_path):
            os.remove(out_path)
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
