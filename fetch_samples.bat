@echo off
cd /d "%~dp0"
venv\Scripts\python -m flurseg fetch-samples %*
if errorlevel 1 (
  echo Fetching samples failed. Review the message above.
  pause
  exit /b 1
)
