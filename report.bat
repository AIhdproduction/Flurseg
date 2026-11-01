@echo off
cd /d "%~dp0"
venv\Scripts\python -m flurseg report --open %*
if errorlevel 1 (
  echo Report generation failed. Review the message above.
  pause
  exit /b 1
)
