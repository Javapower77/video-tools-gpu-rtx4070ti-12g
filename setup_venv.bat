@echo off
setlocal
where pwsh.exe >nul 2>nul
if errorlevel 1 (
    powershell.exe -NoProfile -File "%~dp0scripts\Setup.ps1" %*
) else (
    pwsh.exe -NoProfile -File "%~dp0scripts\Setup.ps1" %*
)
exit /b %errorlevel%