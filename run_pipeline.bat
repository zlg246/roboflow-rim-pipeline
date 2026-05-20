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

echo [3/3] prod-audi-artarmon-live >> "%LOGFILE%"
python pipeline.py --bucket prod-audi-artarmon-live --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [4/4] prod-hornsby >> "%LOGFILE%"
python pipeline.py --bucket prod-hornsby --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [5/5] prod-mercedes-benz-stockport >> "%LOGFILE%"
python pipeline.py --bucket prod-mercedes-benz-stockport --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [6/6] prod-northshore-bmw >> "%LOGFILE%"
python pipeline.py --bucket prod-northshore-bmw --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [7/7] prod-sydney-bmw >> "%LOGFILE%"
python pipeline.py --bucket prod-sydney-bmw --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [8/8] prod-motorline-bmw >> "%LOGFILE%"
python pipeline.py --bucket prod-motorline-bmw --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [9/9] cmt-prod-ap-southeast-2-mercedes-benz-melbourne >> "%LOGFILE%"
python pipeline.py --bucket cmt-prod-ap-southeast-2-mercedes-benz-melbourne --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [10/10] cmt-prod-ap-southeast-2-melbourne-bmw >> "%LOGFILE%"
python pipeline.py --bucket cmt-prod-ap-southeast-2-melbourne-bmw --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [11/11] cmt-prod-ap-southeast-2-melton-toyota >> "%LOGFILE%"
python pipeline.py --bucket cmt-prod-ap-southeast-2-melton-toyota --limit 1000 >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"


echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo Run finished: %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

echo Done. Log saved to: %LOGFILE%