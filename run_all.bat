@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === 1/5 fixed-coverage main table ===
python src\run_fixed_coverage.py || goto err
echo.
echo === 2/5 M2 / M4 table ===
python src\score_m2_m4.py || goto err
echo.
echo === 3/5 bootstrap confidence intervals ===
python src\bootstrap_ci.py || goto err
echo.
echo === 4/5 figures ===
python src\make_figures.py || goto err
echo.
echo === 5/5 Stage-2 k ablation + main-table consistency check ===
python src\ablate_two_stage.py || goto err
echo.
echo Done. Results in results\ , figures in figures\
pause
exit /b 0

:err
echo.
echo [ERROR] python or a dependency is missing, or a script failed.
echo Install Python 3.10+ with numpy + Pillow (see requirements.txt), or run the same commands in Colab.
pause
exit /b 1
