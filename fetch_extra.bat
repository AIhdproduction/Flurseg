@echo off
cd /d "%~dp0"
venv\Scripts\python -m flurseg fetch-extra %*
if errorlevel 1 (
  echo Fetching extra sources failed. Review the message above.
  pause
  exit /b 1
)
