@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo Setup failed. Review the error above, then run setup.cmd again.
  pause
  exit /b 1
)
echo.
echo Setup completed. You can now run start-studio.cmd.
pause
