@echo off
:: ============================================================
:: run_rim_pipeline.bat
:: Runs the rim pipeline for all dealerships.
:: Triggered weekly by Windows Task Scheduler.
:: ============================================================

set PROJECT_DIR=C:\ai-train\roboflow_rim_pipeline
set LOG_DIR=%PROJECT_DIR%\logs\scheduler

:: Activate conda environment
call C:\ProgramData\miniconda3\Scripts\activate.bat scanner

cd /d %PROJECT_DIR%

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Common CLI flags applied to every dealership.
:: Append any combination of:
::   --dry-run            run YOLO but skip all uploads
::   --skip-rim-scratch   upload to rim_seg project only
::   --skip-rim-seg       upload to rim_scratch project only
::   --from YYYY-MM-DD    override start date
::   --to   YYYY-MM-DD    override end date
::   --limit N            process at most N images per dealership (for testing)
set FLAGS=

:: Get timestamp using PowerShell for a unique log file per run
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUNTIMESTAMP=%%i
set LOGFILE=%LOG_DIR%\run_%RUNTIMESTAMP%.log

echo ============================================================ >> "%LOGFILE%"
echo Run started: %DATE% %TIME% >> "%LOGFILE%"
echo Flags: %FLAGS% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

:: ── Add or remove dealerships below ──────────────────────────
echo. >> "%LOGFILE%"
echo [1/11] prod-castle-hill-toyota >> "%LOGFILE%"
python rim_pipeline.py --bucket prod-castle-hill-toyota %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [2/11] prod-worthington-bmw >> "%LOGFILE%"
python rim_pipeline.py --bucket prod-worthington-bmw %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [3/11] prod-audi-artarmon-live >> "%LOGFILE%"
python rim_pipeline.py --bucket prod-audi-artarmon-live %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [4/11] prod-hornsby >> "%LOGFILE%"
python rim_pipeline.py --bucket prod-hornsby %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [5/11] prod-mercedes-benz-stockport >> "%LOGFILE%"
python rim_pipeline.py --bucket prod-mercedes-benz-stockport %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [6/11] prod-northshore-bmw >> "%LOGFILE%"
python rim_pipeline.py --bucket prod-northshore-bmw %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [7/11] prod-sydney-bmw >> "%LOGFILE%"
python rim_pipeline.py --bucket prod-sydney-bmw %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [8/11] prod-motorline-bmw >> "%LOGFILE%"
python rim_pipeline.py --bucket prod-motorline-bmw %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [9/11] cmt-prod-ap-southeast-2-mercedes-benz-melbourne >> "%LOGFILE%"
python rim_pipeline.py --bucket cmt-prod-ap-southeast-2-mercedes-benz-melbourne %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [10/11] cmt-prod-ap-southeast-2-melbourne-bmw >> "%LOGFILE%"
python rim_pipeline.py --bucket cmt-prod-ap-southeast-2-melbourne-bmw %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

echo [11/11] cmt-prod-ap-southeast-2-melton-toyota >> "%LOGFILE%"
python rim_pipeline.py --bucket cmt-prod-ap-southeast-2-melton-toyota %FLAGS% >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"

:: ─────────────────────────────────────────────────────────────

echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo Run finished: %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

echo Done. Log saved to: %LOGFILE%
