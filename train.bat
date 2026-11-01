@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Python executable not found at venv\Scripts\python.exe
  pause
  exit /b 1
)
if not exist "train\dataset\dataset.json" (
  echo run build_dataset.bat first
  pause
  exit /b 1
)
venv\Scripts\python -m flurseg train %*
if errorlevel 1 (
  echo Training failed. Review the message above.
  pause
  exit /b 1
)
echo Training completed. The run folder and best mIoU are shown above.
for /f "delims=" %%R in ('dir /b /ad /o-n train\runs 2^>nul') do (
  start "" "train\runs\%%R\report.html"
  goto :done
)
:done
