# Zhang Lu's methods: M2 (verbal confidence / "I don't know") and M4 (LoRA SFT)

Everything here scores with `src/abstention_eval.py`, the same file A uses, so the
numbers are on one scale. All three modules were written by Zhang Lu; `check_train_overlap.py`
is Deng Yunfan's verification script for Zhang Lu's training pool.

| File | What it does |
|---|---|
| `m2_verbal.py` | M2-conf: model answers then self-reports a 0–100 confidence; threshold tuned on val. M2-idk: prompt says "if the answer is not in the context, reply exactly: I don't know", no threshold. Includes `parse_conf` / `parse_idk`, resumable batch inference, and diagnostics (parse rate, confidence per answerability bucket). |
| `m4_lora.py` | M4: build the 1,500-example SFT set from the train pool (750 answerable → gold answer, 750 unanswerable → `I don't know`), LoRA training (fp32 base + fp16 autocast, T4), inference on val/test, scoring, and over-refusal / missed-refusal case extraction. |
| `prepare_m4_train.py` | builds `data/m4_train_pool.jsonl` from **SQuAD 2.0 train** (2,500 answerable + 2,500 unanswerable, seed 42). Never touches dev. |
| `prepare_b_items.py` | builds the val/test question files A's notebook and B's runs share (dev, stratified, 500 + 500 per split, seed 42). |
| `check_train_overlap.py` | asserts `pool ∩ val = pool ∩ test = ∅`, prints sha256 of all three files, writes `results/m4_train_pool_check.json`. |

## Reproducing M2 and M4 on a T4 (Colab)

```python
import sys; sys.path[:0] = ["src", "b_methods"]     # run from the repo root
import abstention_eval as ev, m2_verbal as m2, m4_lora as m4
from slm_interface import SLM

val_items  = m2.load_jsonl("data/items/b_items_val.jsonl")
test_items = m2.load_jsonl("data/items/b_items_test.jsonl")
gold_val, gold_test = "data/items/b_gold_val.jsonl", "data/items/b_gold_test.jsonl"

slm = SLM()                                          # Qwen2.5-0.5B-Instruct, fp16
for version, name in [("conf", "m2conf"), ("idk", "m2idk")]:
    m2.run_m2(slm, val_items,  f"data/outputs_b/slm_{name}_val.jsonl",  version)
    m2.run_m2(slm, test_items, f"data/outputs_b/slm_{name}_test.jsonl", version)
    m2.score_m2(version, f"data/outputs_b/slm_{name}_val.jsonl", gold_val,
                f"data/outputs_b/slm_{name}_test.jsonl", gold_test, out_dir="results")
m2.diagnostics("data/outputs_b/slm_m2conf_val.jsonl", gold_val)

pool = m2.load_jsonl("data/m4_train_pool.jsonl")
ex   = m4.build_train_examples(pool, refusal_ratio=0.5, max_n=1500, seed=0)
m4.train_lora(ex, "models/m4_lora_r50", r=8, alpha=16, epochs=2, lr=2e-4, bs=4)
m4.infer_m4("models/m4_lora_r50", val_items,  "data/outputs_b/slm_m4_val.jsonl")
m4.infer_m4("models/m4_lora_r50", test_items, "data/outputs_b/slm_m4_test.jsonl")
metrics, scored = m4.score_m4("data/outputs_b/slm_m4_test.jsonl", gold_test,
                              "results/m4_test_scored.jsonl")
```

> Note: `m2.score_m2` uses the legacy refusal-F1 threshold rule. The M2-conf numbers in the report
> and in `results/m2_m4_table.csv` come from `python src/score_m2_m4.py`, which uses the fixed-coverage
> protocol.

Training the reported adapter took **3.6 min** (213 s) and peaked at **9.0 GB** on a T4;
1,490 of 1,500 examples fit in 512 tokens (the other 10 were dropped).

## Verified facts about the M4 training data

- `data/m4_train_pool.jsonl`: 5,000 records, 2,500 answerable + 2,500 unanswerable,
  all from the SQuAD 2.0 **train** split; `check_train_overlap.py` reports
  `|pool ∩ val| = |pool ∩ test| = 0` (sha256 of all three files in
  `results/m4_train_pool_check.json`).
- Training set: 1,500 examples, **750 answerable with the gold answer as target +
  750 unanswerable with `I don't know`** — it is in-domain SFT (task + abstention),
  not pure abstention tuning. The report says so, together with the 42.0% over-refusal
  rate and the 80.0% vs 58.2% answerable-question comparison against M1.
- Refusal-ratio ablation (r30 = 30% refusal, r50 = 50%, r70 = 70%): logs in
  `models/m4_lora_r*/train_log.json`, generations in `data/outputs_b/slm_m4r30_*.jsonl`
  and `slm_m4r70_test.jsonl`. The reported row is **r50** (`slm_m4_*.jsonl`).

## Answer-only control: what part of M4's gain is domain QA?

`train_answer_only.py` trains the identical LoRA recipe on **answerable examples only**
(no refusal targets), at two sizes, so M4 and the control differ in one thing: whether the
refusal examples are in the SFT set. Test set, 1,000 questions:

| run | SFT set | answers (of 500 answerable) | acc. on what it answers | acc. on all 500 answerable | overall |
|---|---|---|---|---|---|
| `ao1500` | 1,500 answerable | 500 (100%) | 73.2% | 73.2% | 36.6% |
| `ao750` | 750 answerable | 500 (100%) | 70.6% | 70.6% | 35.3% |
| **M4** | 750 answerable + 750 refusal | 290 (58%) | 80.0% | 46.4% | **65.2%** |
| M1 | — | 500 (100%) | — | 58.2% | — |

- Both controls **never refuse** (over-refusal 0.000), so their overall accuracy stays below
  the 50.0% always-abstain line. M4's +28.6 points over the control are dominated by the
  refusal decision (correct refusals minus over-refusals); they cannot be separated strictly
  from the control's own in-domain QA gain (report §6.2/7.2).
- Domain QA alone lifts accuracy on answerable questions from 58.2% (M1) to 73.2%, i.e. about
  15 of M4's points are "learned to answer SQuAD".
- M4's 80.0% is measured on the 290 questions it chose to answer. Restricted to those same
  ids: M1 66.2%, ao750 79.0%, ao1500 **80.7%**, M4 80.0% — M4 and the answer-only control are
  **not different** (Δ = −0.7 points, 95% CI [−4.1, +2.8], exact McNemar p = 0.85). On all 500
  answerable questions M4 is 26.8 points *below* the control (95% CI [−31.2, −22.4]) because it
  refuses 42% of them.
- Bottom line: the refusal examples buy the **decision**, not better answers. All of M4's
  answering quality comes from domain QA (58.2% → 73.2% on answerable), and its +28.6 points of
  overall accuracy over the control (36.6% → 65.2%) come from refusing unanswerable questions.
  `compare_controls.py` prints this decomposition and writes `results/m4_attribution.json`.

Colab note: if `get_peft_model` raises `ImportError: Found an incompatible version of
torchao`, run `!pip uninstall -y torchao` — Colab ships torchao 0.10, newer peft wants
> 0.16, and this project does not use quantized LoRA. On a T4 the fp32 base + fp16 autocast
sits near the memory limit (8.4–8.7 GB peak); transient allocator OOM warnings do not stop
training.
