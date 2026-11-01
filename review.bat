@echo off
cd /d "%~dp0"
venv\Scripts\python.exe -m flurseg review %*
if errorlevel 1 (
  echo Review UI could not be started.
  pause
  exit /b 1
)
