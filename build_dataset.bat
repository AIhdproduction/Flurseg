@echo off
cd /d "%~dp0"
venv\Scripts\python -m flurseg build-dataset %*
if errorlevel 1 (
  echo Dataset build failed. Review the message above.
  pause
  exit /b 1
)
