# M4 LoRA adapters

Base model: `Qwen/Qwen2.5-0.5B-Instruct`. LoRA on `q_proj, k_proj, v_proj, o_proj`,
r = 8, alpha = 16, dropout 0.05, 2 epochs, lr 2e-4, batch 4, max_len 512, seed 0.
Training: full SQuAD 2.0 train pool → 1,500 examples (750 answerable with gold answer,
750 unanswerable → `I don't know`), fp32 base + fp16 autocast on a T4.

| Directory | Refusal share | Train time | Peak memory | Final loss | Notes |
|---|---|---|---|---|---|
| `m4_lora_r30/` | 30% | see `train_log.json` | | | refusal-ratio ablation |
| `m4_lora_r50/` | 50% | 213 s (3.6 min) | 9,033 MB | 0.165 | **the adapter used for the reported M4 row** |
| `m4_lora_r70/` | 70% | see `train_log.json` | | | refusal-ratio ablation |

Each directory holds `adapter_config.json` and `train_log.json` (hyper-parameters,
wall-clock time, peak memory, loss history). The weights themselves:

- `m4_lora_r50/adapter_model.safetensors` is committed (4.2 MB) so M4 inference can be
  re-run without retraining;
- the r30 / r70 adapters are not committed — retrain them with
  `m4_lora.train_lora(..., refusal_ratio=0.3 / 0.7)` or take them from the project Drive
  folder.

Generations produced by these adapters are in `data/outputs_b/`:
`slm_m4_val.jsonl` / `slm_m4_test.jsonl` (r50, the reported row),
`slm_m4r30_val.jsonl` / `slm_m4r30_test.jsonl`, `slm_m4r70_test.jsonl`.
Over-refusal / missed-refusal cases: `results/m4_cases.json`.
