@echo off
setlocal
REM CPU-only reproduction of the report's tables and figures.
python src\run_fixed_coverage.py || exit /b 1
python src\score_m2_m4.py || exit /b 1
python src\bootstrap_ci.py || exit /b 1
python src\ablate_m4_ratio.py || exit /b 1
python src\make_m4_cases.py || exit /b 1
python b_methods\check_train_overlap.py || exit /b 1
python src\ablate_two_stage.py || exit /b 1
python src\make_figures.py || exit /b 1
echo done
