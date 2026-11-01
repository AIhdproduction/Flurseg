@echo off
setlocal
cd /d "%~dp0"

py -3.13 -c "import sys; assert sys.version_info[:2] == (3, 13)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.13 x64 is required. Install it and ensure the Python launcher can find it.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating Python 3.13 virtual environment...
    py -3.13 -m venv venv
    if errorlevel 1 goto :error
)

echo Upgrading pip...
venv\Scripts\python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Installing pinned dependencies...
venv\Scripts\python -m pip install --retries 10 --timeout 120 -r requirements.txt
if errorlevel 1 goto :error

echo Checking the environment...
venv\Scripts\python -m flurseg check-env
if errorlevel 1 goto :error

echo Setup completed. Add models\sam3.pt, then run the next phase commands when available.
exit /b 0

:error
echo ERROR: Setup failed. Review the messages above and correct the problem before trying again.
pause
exit /b 1
