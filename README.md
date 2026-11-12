# NLP-Abstention-Project

CISC7021 course project: **When should a small language model say "I don't know"?**
Qwen2.5-0.5B-Instruct on SQuAD 2.0.

This repository is the **A part** (data, evaluation, M1 / M3, two-stage abstainer).
Student B adds the M2 / M4 methods and their training/inference code on a separate
branch; both branches are merged before submission.

## What is here

| Path | Content |
|---|---|
| `Abstention_in_Small_Language_Models.ipynb` | runnable Colab notebook: data, metrics, M1, M3, two-stage, quick demo, full 1,000-question experiment |
| `abstention_eval.py` | the **only** scoring implementation (coverage / selective accuracy / correct refusal / over-refusal / AUROC / AUPRC / ECE / Brier, threshold helpers) |
| `slm_interface.py` | B's unified inference interface (`generate` / `sample`); optional, the notebook has a local fallback |
| `data/items/` | fixed val / test questions and gold (500 answerable + 500 unanswerable each, seed 42) |
| `data/outputs_b/` | B's cached M1 / M3 / M2 / M4 outputs (so the analysis runs without a GPU) |
| `results/` | main table, M2/M4 table, CIs, coverage–accuracy curves, per-question scored jsonl |
| `figures/` | fig1 pipeline, fig2 coverage–quality–cost |
| `slides/` | presentation deck |

## Run it in Colab

1. Open the notebook in Colab, `Runtime -> Change runtime type -> T4 GPU`.
2. If the repo is not cloned yet:
   ```python
   !git clone https://github.com/MC648472/NLP-Abstention-Project.git
   %cd NLP-Abstention-Project
   ```
3. Run the cells top-to-bottom. The notebook finds `abstention_eval.py` in the
   repo root or in `src/`; `slm_interface.py` is optional.
4. Default mode is a **20-question end-to-end demo** (`RUN_QUICK_DEMO=True`).
   For the full 1,000-question table set `RUN_FULL_EXPERIMENT=True` at the bottom
   of the last cell. It stores 8 samples per question, scores the first 5, tunes
   thresholds on val only, and writes `results/main_table_fixed_coverage.csv`.
5. To show the already-computed report table without a GPU, set both flags to
   `False` and run `show_cached_results()`.

## Protocol

- 1,000 questions per split: 500 answerable + 500 unanswerable, seed 42.
- Thresholds tuned on **val only**; test scored once.
- Coverage = share answered; selective accuracy = accuracy on answered.
- Overall / behaviour accuracy = correct answers + correct refusals.
- Always-abstain baseline = **50.0%** (half the test set is unanswerable).
- M1 = length-normalised log-prob; M3 = 5-sample lexical consistency;
  two-stage = M1 screen, borderline questions go to M3.
- M3 has only five consistency levels, so its minimum coverage is 48.3%; the
  0.3 target is reported at the closest reachable point.
