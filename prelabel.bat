@echo off
cd /d "%~dp0"
venv\Scripts\python -m flurseg prelabel %*
if errorlevel 1 (
  echo Prelabeling failed. Review the message above.
  pause
  exit /b 1
)
pause
