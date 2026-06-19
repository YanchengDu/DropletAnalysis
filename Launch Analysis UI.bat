@echo off
:: ── Change to this .bat file's own directory (your data folder) ────────────
cd /d "%~dp0"

:: ── Central pipeline location — update this path if you move the pipeline ──
set CENTRAL=C:\Users\duyan\Documents\Postdoc\2026_Summer\16_NS_Experiment\Data_analysis

:: Make pipeline modules importable
set PYTHONPATH=%CENTRAL%\pipeline;%PYTHONPATH%

echo Activating environment...
call "%CENTRAL%\droplet_env\Scripts\activate.bat"

echo.
echo Starting Analysis UI — opening in browser...
echo Close this window to stop the app.
echo.
voila "%CENTRAL%\notebooks\run_analysis.ipynb" --port=8866

if errorlevel 1 (
    echo.
    echo ERROR: Voila failed to start. See message above.
    pause
)
