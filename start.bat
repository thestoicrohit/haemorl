@echo off
title HaemoRL v5.1 - Launcher
color 0B
cls
echo.
echo  ===================================================
echo    HaemoRL v5.1 - Smart Organ Allocation (India)
echo    Starting backend and opening the app in browser
echo  ===================================================
echo.

cd /d "%~dp0"

REM --- Detect Python (prefer "python", fall back to the "py" launcher) ---
python --version >nul 2>&1
if not errorlevel 1 (set PY=python) else (set PY=py)

REM --- First-run dependency install (only if FastAPI is missing) ---
%PY% -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo  Installing dependencies ^(first run only^)...
    %PY% -m pip install -r requirements.txt
    echo.
)

REM --- Auto-open the browser a few seconds after the server boots ---
start "" /b cmd /c "timeout /t 3 >nul & start http://localhost:7860"

echo  App  : http://localhost:7860
echo  Docs : http://localhost:7860/docs
echo  Press Ctrl+C to stop the server.
echo.

%PY% -m uvicorn app:app --host 127.0.0.1 --port 7860 --log-level info
