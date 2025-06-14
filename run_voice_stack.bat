@echo off
:: ——— Activate Orpheus virtual-env ———
cd /d "%~dp0"
call orpheus-venv\Scripts\activate.bat

:: ——— Spin up Orpheus TTS API in its own window ———
start "Orpheus-TTS" cmd /k "call orpheus-venv\Scripts\activate.bat ^& start-orpheus.bat"

:: ——— OPTIONAL: open a quick ASR test window (comment out if not needed) ———
:: start "NeMo-ASR-Test" cmd /k "call orpheus-venv\Scripts\activate.bat ^& python test_nemo_asr.py"

:: ——— Give the TTS server a moment to finish booting ———
timeout /t 5 > nul

:: ——— Launch the main conversation loop ———
python convo.py

pause
