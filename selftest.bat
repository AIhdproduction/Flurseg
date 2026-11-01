@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Python executable not found at venv\Scripts\python.exe
    echo SELFTEST FAIL: venv\Scripts\python.exe missing
    pause
    exit /b 1
)

echo [1/3] Running environment check...
venv\Scripts\python -m flurseg check-env
if errorlevel 1 (
    echo SELFTEST FAIL: check-env
    pause
    exit /b 1
)

echo [2/3] Running tests...
venv\Scripts\python -m pytest -q -p no:cacheprovider
if errorlevel 1 (
    echo SELFTEST FAIL: pytest
    pause
    exit /b 1
)

echo [3/3] Running ruff lint check...
venv\Scripts\python -m ruff check flurseg tests
if errorlevel 1 (
    echo SELFTEST FAIL: ruff
    pause
    exit /b 1
)

echo SELFTEST PASS
exit /b 0
