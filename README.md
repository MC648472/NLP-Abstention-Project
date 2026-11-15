# NLP-Abstention-Project

CISC7021 Applied Natural Language Processing — group project.

**When should a small language model say "I don't know"?**
A compute-aware two-stage abstainer for `Qwen2.5-0.5B-Instruct` on SQuAD 2.0.

| Member | Scope |
|---|---|
| **A** | data pipeline and fixed splits, evaluation protocol and metrics, M1 / M3 baselines, two-stage compute-aware abstainer, statistics and figures |
| **B** | unified inference interface, M2 (verbal confidence / "I don't know"), M4 (in-domain LoRA SFT), experiment runs, case analysis |

## What is here

| Path | Content |
|---|---|
| `Abstention_in_Small_Language_Models.ipynb` | the one runnable notebook: data, metrics, M1, M3, two-stage, M2, M4, the refusal-share ablation, a 5-question demo and the full 1,000-question run |
| `src/abstention_eval.py` | **the only** scoring implementation (coverage / selective accuracy / correct refusal / over-refusal / cost, plus threshold helpers); both members score with this file |
| `src/run_fixed_coverage.py` | val-only threshold selection, fixed-coverage test table for M1 / M3 / two-stage |
| `src/score_m2_m4.py` | scores B's M2-conf / M2-idk / M4 outputs into `results/m2_m4_table.csv` |
| `src/bootstrap_ci.py` | paired bootstrap 95% CIs and McNemar tests |
| `src/ablate_two_stage.py` | Stage-2 sample-count ablation (k = 3 / 5 / 8) |
| `src/make_figures.py` | Figure 1 (pipeline) and Figure 2 (coverage vs accuracy), PIL only |
| `run_all.bat` | runs the four table / figure scripts in order (Windows) |
| `slm_interface.py` | B's unified inference interface (`generate` / `sample`); the notebook has a local fallback |
| `b_methods/` | B's M2 (`m2_verbal.py`) and M4 (`m4_lora.py`) implementations, the M4 pool builder, and `check_train_overlap.py` — see `b_methods/README.md` |
| `models/` | M4 LoRA adapter (r50 weights + r30/r50/r70 configs and training logs) — see `models/README.md` |
| `data/items/` | fixed val / test questions and gold (500 answerable + 500 unanswerable each, seed 42) |
| `data/outputs_b/` | cached M1 / M3 / M2 / M4 generations, so every table can be rebuilt without a GPU |
| `data/m4_train_pool.jsonl` | M4's training pool: 5,000 SQuAD 2.0 *train* questions (2,500 + 2,500) |
| `results/` | main table, M2/M4 table, CIs, ablation, training-pool check, coverage–accuracy curves, per-question scored jsonl |
| `figures/` | fig1 pipeline, fig2 coverage–quality–cost |

## Reproduce the report tables (CPU, no GPU, no model)

```bash
python src/run_fixed_coverage.py     # results/main_table_fixed_coverage.csv + curves + config
python src/score_m2_m4.py            # results/m2_m4_table.csv + M2/M4 per-question scores
python src/bootstrap_ci.py           # results/ci_table.csv + results/pairwise_ci.csv
python src/make_figures.py           # figures/fig1, figures/fig2
python src/ablate_two_stage.py       # results/two_stage_ablation.csv   (a few minutes)
```

`run_all.bat` runs the first four. Everything above reads `data/outputs_b/*.jsonl`
and needs only `numpy` + `Pillow` (see `requirements.txt`). Re-running the five
scripts on this repository reproduces the committed `results/` and `figures/`
files **byte for byte** — no numbers in the report come from anywhere else.

## Reproduce the method runs (GPU, Colab)

1. Open the notebook in Colab, `Runtime -> Change runtime type -> T4 GPU`.
2. `!git clone https://github.com/MC648472/NLP-Abstention-Project.git` then `%cd NLP-Abstention-Project`.
3. Run the cells top to bottom. `src/abstention_eval.py`, `b_methods/` and `slm_interface.py` are found
   automatically.
4. Default mode is a **5-question end-to-end demo** (`RUN_QUICK_DEMO=True`), about 1–2 minutes on a T4.
   For the full 1,000-question table set `RUN_FULL_EXPERIMENT=True` (section 10); it stores 8 samples per
   question, scores the first 5, tunes thresholds on val only, and writes `results/notebook_main_table.csv`
   (the curated report table stays at `results/main_table_fixed_coverage.csv`).
5. B's methods are behind their own flags: `RUN_M2=True` (section 11, both M2 variants, ≈10 min),
   `RUN_M4_TRAIN=True` / `RUN_M4_INFER=True` (section 12, retrain 3.6 min / re-infer with the committed
   adapter). Section 13 scores the r30 / r50 / r70 ablation from the committed generations — CPU only.
   Section 14 is the threshold-free diagnostic: AUROC labelled by *would the forced answer be correct*,
   the mean `avg_logprob` of the three question buckets, and the conditional refusal rates — also CPU only.
6. With every flag `False` the notebook needs no GPU: it prints the cached tables and the M4 case list.
7. Section 16 is the one-question live demo used in the presentation video (answered / correctly refused /
   answered although it should have refused), at the val-tuned 0.3-coverage operating point.

## Protocol

- 1,000 questions per split: 500 answerable + 500 unanswerable, seed 42.
- Thresholds are tuned on **val only**; test is scored once. No test question is ever used for fitting.
- Coverage = share answered; selective accuracy = accuracy on answered questions;
  overall (behaviour) accuracy = correct answers + correct refusals.
- Always-abstain baseline = **50.0%** overall, because half of the test set is unanswerable.
  This, not 0%, is the line every abstention method has to beat.
- Cost is reported as **average sequences per question**, not wall clock: sampling is batched,
  so 5 samples cost ≈1.8× the wall clock of 1, not 5×.
- M3 has only five consistency levels, so its minimum reachable coverage is 48.3%; the 0.3 target is
  reported at the closest reachable point. M2-conf's verbal confidence has only 11 distinct values on val.
  M2-idk and M4 refuse by themselves, so each has one natural operating point and no threshold.
- **M4 is in-domain LoRA SFT, not pure abstention tuning.** It is trained on 1,500 SQuAD-2.0 *train*
  examples (750 answerable with gold answers + 750 unanswerable), with zero qid overlap with val / test.
  Its gain therefore includes in-domain QA fine-tuning, and it still over-refuses 42.0% of answerable
  questions; on the answerable questions it does answer it is right 80.0% of the time, against 58.2%
  forced accuracy for M1.
- Calibration metrics (AUROC / AUPRC / ECE / Brier) are **not** compared across methods: M4's
  `avg_logprob` is confidence in its own output, including the string "I don't know", so it is not
  P(answer correct) on the same scale as M1 / M3. Cross-method tables report behaviour and cost only;
  calibration numbers appear as within-method diagnostics for M1 / M3.

## Which file is authoritative

| Number in the report | File |
|---|---|
| main table (M1 / M3 / two-stage, 0.3 and 0.5 coverage) | `results/main_table_fixed_coverage.csv` |
| M2-conf / M2-idk / M4 rows | `results/m2_m4_table.csv` |
| 95% CIs, paired differences, McNemar p | `results/ci_table.csv`, `results/pairwise_ci.csv` |
| Stage-2 sample-count ablation | `results/two_stage_ablation.csv` |
| coverage–accuracy curves (Figure 2) | `results/coverage_accuracy_curves.json` |

## What is reproducible and what is not

- **Reproducible on CPU**: cached generations → every table and both figures, byte-identical.
- **Reproducible on GPU**: M1 / M3 / two-stage inference in the notebook. M3 samples are seeded per qid,
  so a subset re-run uses the same samples as the full run.
- **Reproducible on GPU (B's part)**: M2-conf / M2-idk prompts and parsing (`b_methods/m2_verbal.py`),
  M4's training-set construction and LoRA training (`b_methods/m4_lora.py`), and M4 inference with the
  committed r50 adapter. Training the reported adapter takes 3.6 min and peaks at 9.0 GB on a T4.
- **Evidence for "no leak"**: `python b_methods/check_train_overlap.py` reports
  `|pool ∩ val| = |pool ∩ test| = 0` for the 5,000-question pool and writes sha256 of the pool and both
  gold files to `results/m4_train_pool_check.json`. Add `squad_data/train-v2.0.json` (not committed,
  41 MB) and it also proves every pool id occurs in the SQuAD **train** split.
- **Not in this repository**: the r30 / r70 adapter weights (configs and logs are here; retrain or take
  them from the project Drive folder) and the raw SQuAD json files.

## Branch layout

| Branch | Owner | Content |
|---|---|---|
| `main` | — | initial commit |
| `feature/A-data-M1-M3` | A | data, scorer, M1 / M3 / two-stage, statistics, figures, notebook |
| `feature/B-m2-m4` | B | M2 / M4 inference and training code, branched **from `feature/A-data-M1-M3`** |
| `final` | both | merge of the two, then one "freeze numbers" commit; tagged `submission-v1` |

Rules that keep the merge clean:

- `results/` and `figures/` are **generated**: after the merge, A re-runs the five scripts and commits the
  regenerated files once. B does not commit them.
- `src/abstention_eval.py` has exactly one copy and one owner (A); B's changes to it come as a diff.
- The presentation deck and the report are binary files; they are submitted through Moodle and kept out of
  this repository on purpose.
