@echo off
:: ── Detect this .bat file's own directory (works from any location) ─────────
set CENTRAL=%~dp0
:: Remove trailing backslash
if "%CENTRAL:~-1%"=="\" set CENTRAL=%CENTRAL:~0,-1%

:: Make pipeline modules importable
set PYTHONPATH=%CENTRAL%\pipeline;%PYTHONPATH%

echo Activating environment...
call "%CENTRAL%\droplet_env\Scripts\activate.bat"

echo.
echo Starting Z-stack Analysis UI — opening in browser...
echo Close this window to stop the app.
echo.
voila "%CENTRAL%\notebooks\run_zstack.ipynb" --port=8867

if errorlevel 1 (
    echo.
    echo ERROR: Voila failed to start. See message above.
    pause
)
