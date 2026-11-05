@echo off
cd /d "%~dp0"
venv\Scripts\python -m flurseg extract-frames %*
if errorlevel 1 (
  echo Frame extraction failed. Review the message above.
  pause
  exit /b 1
)
