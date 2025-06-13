@echo off
setlocal

echo 🔧 Launching Orpheus Convo...

REM Change to base project directory
cd /d "%~dp0"

REM Use full PowerShell call with encoded command (to avoid quote-hell)
powershell.exe -ExecutionPolicy Bypass -NoProfile -Command ^
    "Set-Location '%~dp0tts_engine';" ^
    "Write-Host '🧠 Activating environment...' -ForegroundColor Cyan;" ^
    "& '%~dp0orpheus-venv\Scripts\Activate.ps1';" ^
    "Write-Host '🚀 Running convo.py...' -ForegroundColor Green;" ^
    "python.exe .\convo.py"

endlocal
pause
