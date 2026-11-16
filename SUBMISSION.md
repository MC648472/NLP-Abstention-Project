# Submission

## Team

- **A — Deng Yunfan**: data pipeline, evaluation protocol, M1 / M3, two-stage abstainer, statistics, figures.
- **B — Zhang Lu**: unified inference interface, M2, M4, experiment runs, case analysis.

## Development workflow

We developed separately in our own Colab notebooks and pushed code to GitHub feature branches:

- A: `feature/A-data-M1-M3`
- B: `feature/B-m2-m4`

The two branches were then merged into `final` and `main`. The merged repository was re-run and updated before submission.

## Submission package

The local submission package contains:

- `Source Code/` and `Source Code.zip`: notebook, `src/`, `b_methods/`, `slm_interface.py`, data, M4 adapter, docs, requirements and `run_all.bat`.
- `Final Results/` and `Final Results.zip`: `results/`, `figures/`, and `README_Results.md`.
- The report PDF and the presentation files are submitted separately through UMMoodle.

## Reproduce

- GPU / Colab: open the notebook in `Source Code/` and run it on a T4.
- CPU: from `Source Code/`, run `run_all.bat`, or run the scripts listed in `README.md`.
