@echo off
setlocal
chcp 65001 >nul
set "PROJECT_ROOT=%~dp0"
set "PYTHONUTF8=1"
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo PixelRAG Studio is not installed in this folder.
  echo Run setup.cmd first.
  pause
  exit /b 1
)

"%PYTHON_EXE%" "%PROJECT_ROOT%pixelrag_studio.py"
if errorlevel 1 (
  echo.
  echo PixelRAG Studio exited with an error.
  echo Run: "%PYTHON_EXE%" "%PROJECT_ROOT%verify_environment.py"
  pause
  exit /b 1
)
