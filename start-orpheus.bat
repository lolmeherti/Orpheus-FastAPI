@echo off
setlocal

set MODEL_PATH=C:\Users\Admin\Desktop\orpheus\models\Orpheus-3b-FT-Q8_0.gguf
set SERVER_EXE=C:\Users\Admin\Desktop\orpheus\llama.cpp\bin\Release\llama-server.exe
set PY_ENV=C:\Users\Admin\Desktop\orpheus\orpheus-venv\Scripts\activate.bat
set API_PORT=5006
set ASSISTANT_PORT=5005
set TOOL_SERVER_PORT=5007

start "LLAMA SERVER" cmd /k "%SERVER_EXE% -m %MODEL_PATH% --port %API_PORT% --ctx-size 8192 --n-predict 8192 --threads 8 --gpu-layers 100 --rope-scaling linear --flash-attn --batch-size 2048 --ubatch-size 512"

call "%PY_ENV%"
cd /d C:\Users\Admin\Desktop\orpheus

start "ASSISTANT SERVER" uvicorn app:app --host 127.0.0.1 --port %ASSISTANT_PORT% --workers 4

start "TOOL SERVER" uvicorn browsing:app --host 127.0.0.1 --port %TOOL_SERVER_PORT% --reload

endlocal