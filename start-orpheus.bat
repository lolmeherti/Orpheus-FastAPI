@echo off
setlocal

REM Set paths
set MODEL_PATH=C:\Users\Admin\Desktop\orpheus\models\Orpheus-3b-FT-Q8_0.gguf
set SERVER_EXE=C:\Users\Admin\Desktop\orpheus\llama.cpp\bin\Release\llama-server.exe
set PY_ENV=C:\Users\Admin\Desktop\orpheus\orpheus-venv\Scripts\activate.bat
set API_PORT=5006
set UVICORN_PORT=5005

REM Launch llama.cpp server in a new window
start "LLAMA SERVER" cmd /k "%SERVER_EXE% -m %MODEL_PATH% --port %API_PORT% --ctx-size 8192 --n-predict 8192 --threads 8 --gpu-layers 100 --rope-scaling linear"

REM Activate virtual environment and launch Orpheus-FastAPI
call "%PY_ENV%"
cd /d C:\Users\Admin\Desktop\orpheus
uvicorn app:app --host 127.0.0.1 --port %UVICORN_PORT% --reload

endlocal