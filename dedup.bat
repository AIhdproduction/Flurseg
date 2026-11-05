@echo off
cd /d "%~dp0"
venv\Scripts\python -m flurseg dedup %*
if errorlevel 1 (
  echo Dedup failed. Review the message above.
  pause
  exit /b 1
)
