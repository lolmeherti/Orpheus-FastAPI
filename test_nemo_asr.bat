@echo off
cd /d "%~dp0"
call orpheus-venv\Scripts\activate.bat
python test_nemo_asr.py
pause