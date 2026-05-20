@echo off
:: ============================================================
:: run_pipeline.bat
:: Runs the annotation pipeline for all dealerships.
:: Triggered every Friday at 5pm by Windows Task Scheduler.
:: ============================================================

set PROJECT_DIR=C:\ai-train\roboflow_pipeline
set LOG_DIR=%PROJECT_DIR%\logs\scheduler

:: Activate conda environment
call C:\ProgramData\miniconda3\Scripts\activate.bat scanner

cd /d %PROJECT_DIR%

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Get date and time using PowerShell for a unique log file per run
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUNTIMESTAMP=%%i
set LOGFILE=%LOG_DIR%\run_%RUNTIMESTAMP%.log

echo ============================================================ >> "%LOGFILE%"
echo Run started: %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

:: ── Add or remove dealerships below ──────────────────────────
echo. >> "%LOGFILE%"
echo [1/1] prod-castle-hill-toyota >> "%LOGFILE%"
python pipeline.py --bucket prod-castle-hill-toyota --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [2/2] prod-worthington-bmw >> "%LOGFILE%"
python pipeline.py --bucket prod-worthington-bmw --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

:: Add more dealerships here, e.g.:
:: echo [2/2] prod-chatswood-toyota >> "%LOGFILE%"
:: python pipeline.py --bucket prod-chatswood-toyota >> "%LOGFILE%" 2>&1
:: echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo Run finished: %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

echo Done. Log saved to: %LOGFILE%