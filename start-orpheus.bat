@echo off
setlocal

set "MODEL_PATH=C:\Users\Admin\Desktop\orpheus\models\Orpheus-3b-FT-Q8_0.gguf"
set "SERVER_EXE=C:\Users\Admin\Desktop\orpheus\llama.cpp\bin\Release\llama-server.exe"
set "PY_ENV=C:\Users\Admin\Desktop\orpheus\orpheus-venv\Scripts\activate.bat"
set "PROJECT_DIR=C:\Users\Admin\Desktop\orpheus"
set "API_PORT=5006"
set "ASSISTANT_PORT=5005"
set "TOOL_SERVER_PORT=5007"

set "CMD_LLAMA=cd /d "%PROJECT_DIR%\llama.cpp\bin\Release" && llama-server.exe -m "%MODEL_PATH%" --port %API_PORT% --ctx-size 8192 --n-predict 8192 --threads 8 --gpu-layers 100 --rope-scaling linear --flash-attn --batch-size 2048 --ubatch-size 512"
set "CMD_ASSISTANT=cd /d "%PROJECT_DIR%" && call "%PY_ENV%" && uvicorn app:app --host 127.0.0.1 --port %ASSISTANT_PORT% --workers 4"

set "CMD_TOOLS=cd /d "%PROJECT_DIR%" && call "%PY_ENV%" && python run_browsing_server.py"

echo Launching all servers...

wt.exe ^
    new-tab --title "LLAMA SERVER" cmd /k "%CMD_LLAMA%" ; ^
    split-pane -H --title "ASSISTANT SERVER" cmd /k "%CMD_ASSISTANT%" ; ^
    split-pane -V --title "TOOL SERVER" cmd /k "%CMD_TOOLS%"

endlocal