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
echo Starting Batch Analysis — opening in browser...
echo Close this window to stop the app.
echo.
voila "%CENTRAL%\notebooks\run_batch.ipynb" --port=8868

if errorlevel 1 (
    echo.
    echo ERROR: Voila failed to start. See message above.
    pause
)
